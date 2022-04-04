"""
This script evaluates models on test sets using GPUs. 

"""

import argparse
import os

import torch
import jsonlines
import json
from sklearn.metrics import confusion_matrix
from models import BERTModel

from trainer import Trainer


DATASET_FOLDER = "../data/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+', help="The datasets to use.")
    parser.add_argument("--model", default="model",
                        help="The name of the model")
    parser.add_argument("--quiet", action='store_true',
                        help="Do not display progress bars")
    return parser.parse_args()


def path_eq(a, b):
    a = os.path.normpath(a)
    a = os.path.normcase(a)
    b = os.path.normpath(b)
    b = os.path.normcase(b)
    return a == b


def calc_metrics(cm):
    """
    Computes the evaluation metrics.

    Parameters
    ----------

    cm: 
        The confusion matrix as a numpy array

    Returns
    -------

    A dictionary containing the accuracy, recall precision, f1 score, and confusion matrix.
    """

    tn, fp, fn, tp = cm.ravel()
    acc = (tn + tp) / (tn + tp+ fn + fp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {
        'acc': acc,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'cm': cm.tolist()
    }


def load_test_set(dataset_path: str, training_info: dict, verbose=True):
    """
    Loads the test set for evaluation.
    
    Parameters
    ----------
    dataset_path:
        The dataset to test on. 
    training_info:
        The dataset indecies used during training. 

    Returns 
    ----------
    texts:
        The test set texts
    labels:
        The test set labels
    """
    if verbose:
        print(f"Loading {dataset_path}")
    
    dataset_path = dataset_path.replace("\\", "/")
    with jsonlines.open(dataset_path) as dataset:
        items = list(dataset)

    if dataset_path in training_info:
        # We only want to evaluate on the test set
        test_split = training_info[dataset_path]['test']
        items = [items[x] for x in test_split]
    
    if verbose:
        print(f"{len(items)} loaded")

    texts = [item['text'].lower() for item in items]
    labels = [1 if item['is_deceptive'] else 0 for item in items]
    return texts, labels



def main():
    args = parse_args()
    
    print(f"Testing {args.model} on:")
    for dataset_name in args.datasets:
        print(f"\t{dataset_name}")
    print("\n")
    
    # Load the model
    model_json = f"{args.model}.json"
    model_weights_filename = f"{args.model}.th"
    model_dataset_filename = f"{args.model}_datasets.json"
    model_results_filename = f"{args.model}_results.json"

    with open(model_json, 'r') as jf:
        model_config = json.load(jf)

    model = BERTModel(model_config['output_layers'], dropout_rate=model_config["dropout_rate"], base=model_config['base_model'])
    state_dict = torch.load(model_weights_filename)
    model.load_state_dict(state_dict)
    model.cuda()
    trainer = Trainer(model, None, quiet=args.quiet)
    
    with open(model_dataset_filename, 'r', encoding='utf-8') as f:
        model_training_info = json.load(f)

    if os.path.isfile(model_results_filename):
        with open(model_results_filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = dict()
    
    # Evaluate the model on each dataset
    for dataset_name in args.datasets:
        print(f"\nEvaluating on {dataset_name}")
        texts, labels = load_test_set(os.path.join(
            DATASET_FOLDER, dataset_name), model_training_info)
        _, predictions = trainer.validate_model(texts, labels)
        scores = calc_metrics(confusion_matrix(labels, predictions))
        results[dataset_name] = scores
        print(f"Final Test Confusion Matrix:\n{scores['cm']}")
        print(f"Final Test Accuracy: {scores['acc']}")
        print(f"Final Test Recall: {scores['recall']}")
        print(f"Final Test Precision: {scores['precision']}")
        print(f"Final Test F1 Score: {scores['f1']}")

    
    with open(model_results_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
