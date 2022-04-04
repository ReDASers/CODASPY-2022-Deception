
from typing import List

import torch
from torch import nn
from torch.autograd import grad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import transformers
from transformers import BertModel, BertTokenizerFast
from transformers import RobertaModel, RobertaTokenizerFast

MODEL_CONFIGS = {
    "bert-base-uncased": {
        'lowercase': True,
    },
    "bert-large-uncased": {
        'lowercase': True,
    },
    "roberta-base": {
        'lowercase': False,
    },
    "roberta-large": {
        'lowercase': False,
    },
}

class BERTModel(nn.Module):
    """
    A BERT binary classification model. This model consists of a base bert/roberta model and a linear layer. 

    It first feeds a sample through the BERT model and performs classification on the [cls] token embedding.
    """

    def __init__(self, output_layers=3, dropout_rate=0.0, base: str='bert-base-uncased'):
        super().__init__()

        if base not in MODEL_CONFIGS:
            raise ValueError(f"Unknown base {base}")
        
        verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()

        if base.startswith('bert'):
            self.tokenizer = BertTokenizerFast.from_pretrained(base)
            self.bert = BertModel.from_pretrained(base)
        else:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(base)
            self.bert = RobertaModel.from_pretrained(base)

        self.base_config = MODEL_CONFIGS[base]

        transformers.logging.set_verbosity(verbosity)

        self.hidden_size = self.bert.config.hidden_size

        linear_layers = []
        if dropout_rate > 0:
            linear_layers.append(nn.Dropout(p=dropout_rate))
        
        if output_layers == 3:
            linear_layers.append(nn.Linear(self.hidden_size, 256))
            linear_layers.append(nn.ReLU())
            linear_layers.append(nn.Linear(256, 256))
            linear_layers.append(nn.ReLU())
            linear_layers.append(nn.Linear(256, 2))
        elif output_layers == 2:
            linear_layers.append(nn.Linear(self.hidden_size, 256))
            linear_layers.append(nn.ReLU())
            linear_layers.append(nn.Linear(256, 2))
        elif output_layers == 1:
            linear_layers.append(nn.Linear(self.hidden_size, 2))
        else:
            raise ValueError(f'Output layer must be either 1, 2, or 3 got {output_layers}')

        self.linear = nn.Sequential(*linear_layers)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, token_type_ids: torch.LongTensor, return_hidden_states=False):
        """
        Performs a forward pass. 

        Parameters
        ----------

        input_ids
            The input token ids.

        attention_mask
            The attention mask.

        token_type_ids:
            The token type ids

        Returns
        -------

        The class logits

        """
        bert_result = self.bert(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, output_hidden_states=True)
        pred = self.linear(bert_result.last_hidden_state[:, 0])
        hidden_state = bert_result.hidden_states

        if return_hidden_states:
            return pred, hidden_state
        return pred

    def tokenize(self, text: List[str]):
        """
        Tokenizes and encodes a batch of strings. 

        Parameters
        ----------

        text
            The samples to tokenize

        Returns 
        -------

        A dictionary containing the tokenized output. (input_ids, attention_mask, token_type_ids)
        """
        if self.base_config['lowercase']:
            text = [x.lower() for x in text]
        tokenizer_output = self.tokenizer(
                                        text,
                                        truncation=True, 
                                        padding=True, 
                                        add_special_tokens=True, 
                                        return_token_type_ids=True
                                        )
        return tokenizer_output

    def convert_ids_to_tokens(self, input_ids) -> List[str]:
        """
        Converts token ids to tokens

        Parameters
        ----------

        input_ids
            The input ids to convert

        Returns 
        -------

        The tokens
        """
        return self.tokenizer.convert_ids_to_tokens(ids=input_ids, skip_special_tokens=False)

    def predict(self, text: str):
        """
        Generates predictions for a single sample.

        Parameters 
        ----------
        text
            The sample to generate the prediction for.

        Returns 
        -------

        The predicted class of the sample (0 or 1) as an integer.
        """
        tokenized_output = self.tokenize([text])

        device = next(self.parameters()).device
        input_ids = torch.tensor(tokenized_output['input_ids'], device=device)
        attention_mask = torch.tensor(
            tokenized_output['attention_mask'], device=device)
        token_type_ids = torch.tensor(
            tokenized_output['token_type_ids'], device=device)

        with torch.no_grad():
            out_tensor = self.forward(
                input_ids, attention_mask, token_type_ids)
            result = torch.argmax(out_tensor, dim=1)
        return result.cpu().item()

    def predict_batch(self, text: List[str]):
        """
        Generates predictions for a batch of samples. 

        Note: A batch must fit within memory. 

        Parameters
        ----------

        text
            The samples to generate the prediction for

        Returns 
        -------

        A numpy array containing the class of each sample (0 or 1) as an integer.

        """
        tokenized_output = self.tokenize(text)

        device = next(self.parameters()).device
        input_ids = torch.tensor(tokenized_output['input_ids'], device=device)
        attention_mask = torch.tensor(
            tokenized_output['attention_mask'], device=device)
        token_type_ids = torch.tensor(
            tokenized_output['token_type_ids'], device=device)

        with torch.no_grad():
            out_tensor = self.forward(
                input_ids, attention_mask, token_type_ids)
            result = torch.argmax(out_tensor, dim=1)
        return result.cpu().numpy()

    def predict_proba(self, text: str):
        """
        Generates probabilistic predictions for a single sample or bath of samples.

        Parameters 
        ----------
        text
            The sample/samples to generate the prediction for.

        Returns 
        -------

        The probabilities that the samples' class is 1. If a single sample is provided, 
        this is returned as a float. Otherwise, it is returned as a numpy array.
        """
        if isinstance(text, str):
            tokenized_output = self.tokenize([text])
        else:
            tokenized_output = self.tokenize(text)

        device = next(self.parameters()).device
        input_ids = torch.tensor(tokenized_output['input_ids'], device=device)
        attention_mask = torch.tensor(
            tokenized_output['attention_mask'], device=device)
        token_type_ids = torch.tensor(
            tokenized_output['token_type_ids'], device=device)

        with torch.no_grad():
            out_tensor = self.forward(
                input_ids, attention_mask, token_type_ids)
            result = torch.softmax(out_tensor, dim=1)[:, 1].squeeze()

        if len(result.shape) == 0:
            return result.cpu().item()
        return result.cpu().numpy()

    def generate_embedding_grad(self, text: str):
        tokenized_output = self.tokenize([text])

        device = next(self.parameters()).device
        input_ids = torch.tensor(tokenized_output['input_ids'], device=device)
        attention_mask = torch.tensor(
            tokenized_output['attention_mask'], device=device)
        token_type_ids = torch.tensor(
            tokenized_output['token_type_ids'], device=device)
        out_tensor, hidden_states = self.forward(
            input_ids, attention_mask, token_type_ids, return_hidden_states=True)

        result = torch.softmax(out_tensor, dim=1)[:, 1].squeeze()

        # (n_words, BERT dimension)
        embedding_space: torch.Tensor = hidden_states[0]
        embedding_grad = grad([result], [embedding_space])[0].squeeze()[1:-1]
        input_ids = input_ids.squeeze()[1:-1].cpu().numpy()

        return result, input_ids, embedding_grad


def pad_collate(data):
    input_ids = pad_sequence([i[0] for i in data], batch_first=True)
    attention_mask = pad_sequence([i[1] for i in data], batch_first=True)
    token_type_ids = torch.zeros_like(input_ids)
    labels = torch.LongTensor([i[3] for i in data])
    return input_ids, attention_mask, token_type_ids, labels


class BertClassificationDataset(Dataset):

    def __init__(self, texts: List[str], labels: List[int], model_base='bert-base-uncased'):
        if model_base.startswith('bert'):
            tokenizer = BertTokenizerFast.from_pretrained(model_base)
        elif model_base.startswith('roberta'):
            tokenizer = RobertaTokenizerFast.from_pretrained(model_base)

        self.input_ids = list()
        self.attention_mask = list()
        self.token_type_ids = list()

        for text in texts:

            if MODEL_CONFIGS[model_base]['lowercase']:
                text = text.lower()

            tokens = tokenizer(
                text, 
                truncation=True, 
                padding=True, 
                add_special_tokens=True, 
                return_tensors="pt", 
                return_token_type_ids=True
            )

            self.input_ids.append(tokens['input_ids'].squeeze())
            self.attention_mask.append(tokens['attention_mask'].squeeze())
            self.token_type_ids.append(tokens['token_type_ids'].squeeze())
                
        self.labels = labels

    def __getitem__(self, index) -> T_co:
        """
        Gets an item from the dataset

        Returns
        --------

        input_ids: torch.tensor
            The input IDs

        attention_mask: torch.tensor
            The attention mask

        token_type_ids: torch.tensor
            The token type id

        label:
            The label
        """
        return self.input_ids[index], self.attention_mask[index], self.token_type_ids[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
