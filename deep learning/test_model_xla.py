"""
Tests model on Google Colab TPUs. 

This package should be run by importing in google colab and calling test_xla_mp
"""
import argparse
import json
import os

# imports pytorch
import torch
# imports the torch_xla package
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch.distributed as dist
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import transformers

from models import BertClassificationDataset, BERTModel, pad_collate
from test_model import DATASET_FOLDER, load_test_set, calc_metrics

assert os.environ['COLAB_TPU_ADDR'], 'This should only be used on Google Colab with a TPU instance'


def load_dataset(model_name, dataset_name, rank=0, world_size=1):
    """
    Load datasets from disk and splits across proccesses

    Parameters
    ----------

    model_name: 
        the name of the model to load data for
    dataset_name:
        the dataset to load
    rank:
        this proccess's rank
    world_size:
        the number of processes
    
    Returns
    -------

    A BertClassificationDataset containing this proccess's share of the data.
    
    """
    model_dataset_filename = f"{model_name}_datasets.json"
    with open(model_dataset_filename, 'r', encoding='utf-8') as f:
        model_training_info = json.load(f)

    # Loads the dataset
    dataset_path = os.path.join(DATASET_FOLDER, dataset_name)
    texts, labels = load_test_set(dataset_path, model_training_info, verbose=rank==0)

    if world_size > 1:
        # Partitions the dataset amongst the proccesses
        n_texts = len(texts)
        indexes = list(range(n_texts))
        indexes = indexes[rank:len(texts):world_size]
        texts = [texts[x] for x in indexes]
        labels = [labels[x] for x in indexes]
    return BertClassificationDataset(texts, labels)


def load_model(model_name):
    """
    Loads the model from a disk. 
    """
    model_json = f"{model_name}.json"
    model_weights_filename = f"{model_name}.th"

    with open(model_json, 'r') as jf:
        model_config = json.load(jf)
    model = BERTModel(
        model_config['output_layers'], 
        dropout_rate=model_config["dropout_rate"], 
        base=model_config['base_model']
    )
    state_dict = torch.load(model_weights_filename, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def test_xla_single_core(model_name, datasets, batch_size=464):
    """
    Test models on single TPU cores.
    
    Parameters
    ----------
    model_name:
        The name of the model to evaluate

    datasets:
        The datasets to evaluate. 

    batch_size:
        The batch size to use when evaluating the model 

    """
    print(f"Testing {model_name} on:")
    for dataset_name in datasets:
        print(f"\t{dataset_name}")
    print("\n")

    model = load_model(model_name)
    device = xm.xla_device()
    model = model.to(device)

    for dataset_name in datasets:
        print(f"\nEvaluating on {dataset_name}")

        dataset = load_dataset(model_name, dataset_name)

        # Generate predictions on the TPU
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=pad_collate
        )
        predictions = []
        for batch in tqdm(dataloader, desc="Training"):
            input_ids: torch.Tensor = batch[0].to(device)
            attention_mask: torch.Tensor = batch[1].to(device)
            token_type_ids: torch.Tensor = batch[2].to(device)

            with torch.no_grad():
                out_tensor = model.forward(input_ids, attention_mask, token_type_ids)
                result = torch.argmax(out_tensor, dim=1).cpu().numpy()
                predictions.append(result)
        
        predictions = np.hstack(predictions)

        cm = confusion_matrix(dataset.labels, predictions)
        calc_metrics(cm)

# A shared-memory copy of the model
WRAPPED_MODEL = None


def eval_map_fn(index, flags):
    model_name = flags['model_name']
    datasets = flags['datasets']

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(
                backend="gloo", rank=index, world_size=8)
    
    # Sets a common random seed - both for initialization and ensuring graph is the same
    torch.manual_seed(1234)

    # Acquires the (unique) Cloud TPU core corresponding to this process's index
    device = xm.xla_device()

    # Load the model
    model = WRAPPED_MODEL.to(device)
    model.to(device, non_blocking=True)

    for dataset_name in datasets:
        # Load the dataset
        dataloader = torch.utils.data.DataLoader(
            load_dataset(model_name, dataset_name, index, 8),
            batch_size=flags['batch_size'],
            collate_fn=pad_collate,
            drop_last=False
        )

        if index == 0:
            dataloader = tqdm(dataloader, desc="Evaluating")

        predictions = []
        labels = []
        for batch in dataloader:
            input_ids: torch.Tensor = batch[0].to(device)
            attention_mask: torch.Tensor = batch[1].to(device)
            token_type_ids: torch.Tensor = batch[2].to(device)
            labels.append(batch[3].numpy())

            with torch.no_grad():
                out_tensor = model.forward(input_ids, attention_mask, token_type_ids)
                result = torch.argmax(out_tensor, dim=1).cpu().numpy()
            predictions.append(result)
        
        predictions = np.hstack(predictions)
        labels = np.hstack(labels)
        # computes per-proccess confusion matrix
        cm = confusion_matrix(labels, predictions)
        cm = torch.tensor(cm)
        # Collect confusion matries to proccess 0
        dist.all_reduce(cm, op=dist.ReduceOp.SUM)

        if index == 0:
            cm = cm.numpy()
            scores = calc_metrics(cm)
            print(f"Final Test Confusion Matrix:\n{scores['cm']}")
            print(f"Final Test Accuracy: {scores['acc']}")
            print(f"Final Test Recall: {scores['recall']}")
            print(f"Final Test Precision: {scores['precision']}")
            print(f"Final Test F1 Score: {scores['f1']}")


def test_xla_mp(model_name, datasets, batch_size=384):
    """
    Tests a model using 8 TPU cores. 
    """
    global WRAPPED_MODEL 
    WRAPPED_MODEL = xmp.MpModelWrapper(load_model(model_name))
    flags = {
        'model_name': model_name,
        'datasets': datasets,
        'batch_size': batch_size
    }
    xmp.spawn(eval_map_fn, args=(flags,), nprocs=8, start_method='fork')
