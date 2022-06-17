"""
This script runs Google's Language interpretability toolkit. 
"""
# import os
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")

from random import shuffle

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import model as lit_model
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

import json
import jsonlines
import torch
from torch.autograd import grad
from tqdm import tqdm

from models import BERTModel

MAX_ITEMS_PERGPU = 4
LABELS = ["Truthful", "Deceptive"]

class DeceptionDataset(lit_dataset.Dataset):
    """
    A LIT wrapper for deception datasets.

    Parameters:
    -----------
    path:
        The path to the jsonlines dataset
    description:
        The description to show in LIT
    limit:
        The maximum number of data points to sample. -1 for infinite
    """
    def __init__(self, path, description, limit=-1):
        super().__init__(description=description)
        with jsonlines.open(path) as dataset:
            data = list(dataset)
        
        if limit > 0 and limit < len(data):
            shuffle(data)
            data = data[0:limit]
        
        self._examples = [{
            'text': x['text'],
            'label': LABELS[x['is_deceptive']]
        } for x in data]

    def spec(self):
        return {
            'text': lit_types.TextSegment(),
            'label': lit_types.CategoryLabel(vocab=LABELS)
        }


class DeceptionModel(lit_model.Model):
    """
    A wrapper for the models

    Parameters:
    -----------
    model_name:
        The model to load
    devices:
        The devices to use when generating predictions
    """
    def __init__(self, model_name, devices=None) -> None:
        print("Creating model")
        super().__init__()
        model_json = f"{model_name}.json"
        model_weights_filename = f"{model_name}.th"
        with open(model_json, 'r') as jf:
            self.model_config = json.load(jf)
            
        if not devices:
            devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        self.devices = devices
        
        self.state_dict = torch.load(model_weights_filename, map_location=torch.device('cpu'))
        self.models = []

    def _batched_predict(self, inputs, **kw):
        """Internal helper to predict using minibatches."""
        # Setup models on GPU
        for device in self.devices:
            model = BERTModel(
                self.model_config['output_layers'], 
                dropout_rate=self.model_config["dropout_rate"], 
                base=self.model_config['base_model']
            )
            model.load_state_dict(self.state_dict)
            model.eval()
            model.to(device)
            self.models.append(model)

        minibatch_size = self.max_minibatch_size(**kw)
        minibatch = []
        for ex in tqdm(inputs):
            if len(minibatch) < minibatch_size:
                minibatch.append(ex)
            if len(minibatch) >= minibatch_size:
                yield from self.predict_minibatch(minibatch, **kw)
                minibatch = []
        if len(minibatch) > 0:  # pylint: disable=g-explicit-length-test
            yield from self.predict_minibatch(minibatch, **kw)

        # Cleanup
        self.models.clear()
        torch.cuda.empty_cache()
    
    def predict_minibatch(self, inputs):
        """Predict on a stream of examples."""
        texts = [x['text'] for x in inputs]
        tokenizer_outputs = self.models[0].tokenize(texts)

        input_ids = torch.split(torch.tensor(tokenizer_outputs['input_ids']), MAX_ITEMS_PERGPU)
        attention_masks = torch.split(torch.tensor(tokenizer_outputs['attention_mask']), MAX_ITEMS_PERGPU)
        token_type_ids = torch.split(torch.tensor(tokenizer_outputs['token_type_ids']), MAX_ITEMS_PERGPU) 

        # Collect model outputs
        outputs = []
        for device, model, input_id, attention_mask, token_type_id in zip(self.devices, self.models, input_ids, attention_masks, token_type_ids):
            logits, hidden_states = model(
                            input_id.to(device), 
                            attention_mask.to(device), 
                            token_type_id.to(device), 
                            return_hidden_states=True
                        )
            probs = torch.softmax(logits, dim=1)
            batch_embeddings: torch.Tensor = hidden_states[0]
            batch_embeddings.retain_grad()
            probs[:,1].sum().backward()

            data = [input_id, probs, hidden_states[-1], batch_embeddings]
            outputs.append(data)
        
        # Convert the model outputs to a lit
        results = []
        for input_ids, probs, ll_embed, embed in outputs: 
            data = zip(
                input_ids,
                probs.detach().cpu().numpy(),
                embed.detach().cpu().numpy(),
                ll_embed.detach().cpu().numpy(),
                embed.grad.detach().cpu().numpy(),
            )
            for input_ids, prob, embed, ll_embed, embed_grad in data:
                tokens = model.convert_ids_to_tokens(input_ids)
                sep_index = tokens.index("[SEP]")
                results.append(
                    {
                        'tokens': tokens[1:sep_index],
                        'pred_probs': prob,
                        'cls_embedding': ll_embed[0],
                        'token_embeddings': embed[1:sep_index],
                        'embedding_grads': embed_grad[1:sep_index]
                    }
                )
        
        return results
            
    def input_spec(self):
        return {
            'text':  lit_types.TextSegment(),
        }

    def output_spec(self):
        return {
            'tokens': lit_types.Tokens(parent='text'),
            'pred_probs': lit_types.MulticlassPreds(vocab=LABELS, parent='label'),
            'cls_embedding': lit_types.Embeddings(),
            'token_embeddings': lit_types.TokenEmbeddings(align='tokens'),
            'embedding_grads': lit_types.TokenGradients(align='tokens'),

        }

    def max_minibatch_size(self) -> int:
        return MAX_ITEMS_PERGPU * len(self.devices)

    def load(self, path: str):
        return DeceptionModel(path)


def main():
    datasets = {
        'amazon': DeceptionDataset("../data/Processed/amazon.jsonl", "Amazon Reviews"),
        'welfake_flipped': DeceptionDataset("../data/Processed/welfake_flipped.jsonl", "Welfake Fake News"),
        'email_benchmarking': DeceptionDataset("../data/Processed/email_benchmarking.jsonl", "Email Benchmarking"),
        'liar': DeceptionDataset("../data/Processed/liar_plus.jsonl", "Liar"),
        'job_scams': DeceptionDataset("../data/Processed/job_scams.jsonl", "Job Scams"),
        
        'amazon_cleaned': DeceptionDataset("../data/Processed/amazon.jsonl", "Amazon Reviews"),
        'welfake_cleaned': DeceptionDataset("../data/Processed/welfake_flipped.jsonl", "Welfake Fake News"),
        'email_benchmarking_cleaned': DeceptionDataset("../data/Processed/email_benchmarking.jsonl", "Email Benchmarking"),
        'liar_cleaned': DeceptionDataset("../data/Processed/liar_plus.jsonl", "Liar"),
        'job_scams_cleaned': DeceptionDataset("../data/Processed/job_scams.jsonl", "Job Scams"),
    }

    # NLIModel implements the Model API
    models = {
        'amazon_1': DeceptionModel('Pilot Study/models/amazon_1/amazon_1'),
        'welfake_flipped': DeceptionModel('Pilot Study/models/welfake_flipped/welfake_flipped'),
        'email_benchmarking1': DeceptionModel('Pilot Study/models/email_benchmarking1/email_benchmarking1'),
        'liar1': DeceptionModel('Pilot Study/models/liar1/liar1'),
        'job_scams1': DeceptionModel('Pilot Study/models/job_scams1/job_scams1'),
        
        'amazon_cleaned': DeceptionModel('cleaned_models/amazon/amazon'),
        'email_benchmarking_cleaned': DeceptionModel('cleaned_models/email_benchmarking/email_benchmarking'),
        'liar_cleaned': DeceptionModel('cleaned_models/liar/liar'),
        'welfake_cleaned': DeceptionModel('cleaned_models/welfake/welfake'),
        'job_scams_cleaned': DeceptionModel('cleaned_models/job_scams/job_scams'),
    }
    flags = server_flags.get_flags()
    flags['data_dir'] = 'lit_cache'
    print(flags)

    lit_demo = dev_server.Server(models, datasets, **flags)
    lit_demo.serve()

if __name__ == "__main__":
    main()