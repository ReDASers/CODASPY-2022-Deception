"""
Fine-tunes a BERT model on a dataset
"""

import argparse
import json
import sys
from datetime import datetime
from os import path
from random import shuffle
from socket import gethostname

import jsonlines
import numpy as np
import torch
import torch.nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from models import BertClassificationDataset, BERTModel
from trainer import Trainer


def parse_args():
    """
    Parses args. If the config argument is present, loads the config from file. Otherwise, dumps the arguments
    into a config file. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="The config file to load model parameters from")
    parser.add_argument("--tensorboard_dir", type=str, default="runs", help="The folder to put tensorboard runs in")
    parser.add_argument("--datasets", nargs='+', help="The datasets to use.")
    parser.add_argument("--model_name", default="model.th", help="The name of the model")
    parser.add_argument("--base_model", default="bert-base-uncased", help="The name of the model")
    parser.add_argument("--weighted_loss", action='store_true', help='Whether or not to use a weighted '
                                                                     'CrossEntropyLoss')
    parser.add_argument("--quiet", action='store_true', help="Whether or not to display progress bars")
    parser.add_argument("--lr", type=float, default=5e-5, help="The learning rate to use")
    parser.add_argument("--weight_decay", type=float, default=0, help="The weight decay parameter to use")
    parser.add_argument("--dropout_rate", type=float, default=0, help="The dropout weight to use")
    parser.add_argument("--max_grad_norm", type=float, default=10, help="The l2 gradient norm")
    parser.add_argument("--epochs", type=int, default=10, help="The maximum number of epoches")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size to use between optimizer updates")
    parser.add_argument("--output_layers", type=int, default=3, help="The number of layers in the output head. "
                                                                     "Must be 1-3")

    args = parser.parse_args()
    if args.config:
        with open(args.config, "r", encoding='utf-8') as f:
            config = json.load(f)
        args.model_name = path.splitext(args.config)[0] + ".th"
    else:
        config = {
            "datasets": args.datasets,
            "base_model": args.base_model,
            "weighted_loss": args.weighted_loss,
            "lr": args.lr,
            'weight_decay': args.weight_decay,
            'dropout_rate': args.dropout_rate,
            "max_grad_norm": args.max_grad_norm,
            "batch_size": args.batch_size,
            "output_layers": args.output_layers,
            "epochs": args.epochs
        }
        args.config = path.splitext(args.model_name)[0] + ".json"
        with open(args.config, "w", encoding='utf-8') as f:
            json.dump(config, f, indent=4)
    return config, args


class RawDataset:

    def __init__(self):
        self.texts = []
        self.labels = []

    def append(self, text, label, index):
        for i in index:
            self.texts.append(text[i])
            self.labels.append(label[i])


def split(length):
    """
    Randomly splits the set of integers [0,length) into 80/10/10 train val, test split
    """
    indexes = [x for x in range(length)]
    shuffle(indexes)
    train_split = int(length * 0.8)
    test_split = int(length * 0.9)
    train_idx = indexes[0:train_split]
    val_idx = indexes[train_split:test_split]
    test_idx = indexes[test_split:]
    return train_idx, val_idx, test_idx


def setup_dataset(dataset_paths, model_name):
    """
    Loads a set of datasets, splitting each into a 80/10/10 train-val-test split and 
    dumps the validation and test set indices to disk. Indicies will be saved to 
    {mmodel_name}_datasets.json.

    Parameters
    ----------
    dataset_paths:
        An array containing the paths to the dataset jsonlines files.
    model_name:
        The model name
    
    Returns
    -------
    train:
        A RawDataset containing the training set.
    val:
        A RawDataset continaing the validation set
    test:
        A RawDataset containing the test set. 
    """
    train = RawDataset()
    val = RawDataset()
    test = RawDataset()

    out_filename = path.splitext(path.basename(model_name))[0] + "_datasets.json"
    out_filename = path.join(path.dirname(model_name), out_filename)
    save_data = {}

    for dataset_path in dataset_paths:
        texts = []
        labels = []
        with jsonlines.open(dataset_path) as dataset:
            for item in dataset:
                texts.append(item['text'])
                labels.append(1 if item['is_deceptive'] else 0)

        train_idx, val_idx, test_idx = split(len(texts))

        train.append(texts, labels, train_idx)
        val.append(texts, labels, val_idx)
        test.append(texts, labels, test_idx)
        save_data[dataset_path] = {
            "val": val_idx,
            "test": test_idx
        }
    
    with open(out_filename, "w") as f:
        json.dump(save_data, f)

    return train, val, test


def writer_path(config_name, tensorboard_dir):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    base = path.splitext(path.basename(config_name))[0]
    return path.join(tensorboard_dir, f"{base}_{current_time}_{gethostname()}")


def build_model(config):
    """
    Builds a BERTModel object from a config dictionary
    """
    return BERTModel(
        output_layers=config["output_layers"], 
        dropout_rate=config["dropout_rate"], 
        base=config['base_model']
    )


def main():
    config, args = parse_args()
    print(config)
    print(args)
    print(f"Training on {torch.cuda.device_count()} {torch.cuda.get_device_name()} GPUs")

    # Set up the dataset
    train, val, test = setup_dataset(config["datasets"], args.model_name)

    print("Split Statistics:")
    print(f"\tTrain Deceptive Ratio: {np.mean(train.labels)}")
    print(f"\tValid Deceptive Ratio: {np.mean(val.labels)}")
    print(f"\tTest Deceptive Ratio: {np.mean(test.labels)}")
    print()
    print()

    # Initialize the Model
    model = build_model(config)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if config['weighted_loss']:
        phish_ratio = sum(train.labels) / len(train.labels)
        weights = [phish_ratio, 1 - phish_ratio]
        loss_fn = torch.nn.CrossEntropyLoss(torch.Tensor(weights), reduction='none')
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    trainer = Trainer(model,
                      optimizer, 
                      writer=SummaryWriter(log_dir=writer_path(args.config, args.tensorboard_dir), flush_secs=5), 
                      loss_fn=loss_fn,
                      batch_size=config['batch_size'],
                      max_grad_norm=config['max_grad_norm'],
                      quiet=args.quiet
                    )

    # Get initial validation performance
    best_loss, predictions = trainer.validate_model(val.texts, val.labels)
    trainer.writer.add_scalar("loss/valid", best_loss, global_step=-1)
    trainer.writer.add_scalar("decptive_ratio/valid", np.mean(predictions), global_step=-1)

    accuracy = accuracy_score(val.labels, predictions)
    recall = recall_score(val.labels, predictions)
    f1 = f1_score(val.labels, predictions)
    trainer.writer.add_scalar("acc/valid", accuracy, global_step=-1)
    trainer.writer.add_scalar("recall/valid", recall, global_step=-1)
    trainer.writer.add_scalar("f1/valid", f1, global_step=-1)
    print(f"Validation Accuracy: {accuracy}")
    print(f"Validation Recall: {recall}")
    sys.stdout.flush()

    # Construct the dataset
    training_set = BertClassificationDataset(train.texts, train.labels, model_base=config['base_model'])

    # Train the Model
    for epoch in range(config['epochs']):

        trainer.train_epoch(training_set)

        valid_loss, predictions = trainer.validate_model(val.texts, val.labels)
        deceptive_ratio = np.mean(predictions)
        trainer.writer.add_scalar("loss/valid", valid_loss, global_step=epoch)
        trainer.writer.add_scalar("decptive_ratio/valid", deceptive_ratio, global_step=epoch)
        if deceptive_ratio == 0 or deceptive_ratio == 1:
            print("Stopping early due to model divergence")
            break

        accuracy = accuracy_score(val.labels, predictions)
        recall = recall_score(val.labels, predictions)
        f1 = f1_score(val.labels, predictions)
        trainer.writer.add_scalar("acc/valid", accuracy, global_step=epoch)
        trainer.writer.add_scalar("recall/valid", recall, global_step=epoch)
        trainer.writer.add_scalar("f1/valid", f1, global_step=epoch)
        
        print(f"Epoch {epoch} Validation Accuracy: {accuracy}")
        print(f"Epoch {epoch} Validation Recall: {recall}")
        sys.stdout.flush()

        if valid_loss < best_loss:
            torch.save(trainer.get_model().state_dict(), args.model_name)
            best_loss = valid_loss

    if path.exists(args.model_name):
        # Test the model
        # Skips if no model saved due to Epoch 0 divergence. 
        model = build_model(config)
        model.load_state_dict(torch.load(args.model_name))
        trainer = Trainer(model, None, batch_size=trainer.batch_size, quiet=args.quiet)
        _, predictions = trainer.validate_model(test.texts, test.labels)
        print(f"Final Test Accuracy: {accuracy_score(test.labels, predictions)}")
        print(f"Final Test Recall: {recall_score(test.labels, predictions)}")
        print(f"Final Test F1 Score: {f1_score(test.labels, predictions)}")

        print("\n\n")

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} max memory: {torch.cuda.max_memory_allocated() / (1024 ** 3)}GB")


if __name__ == '__main__':
    main()
