import math
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import BertClassificationDataset, BERTModel, pad_collate


def get_subbatch_size(quiet=False) -> int:
    """ 
    Calculate the batch size based on the amount of GPU memory available
    """
    if torch.cuda.device_count() > 1:
        return 8 * torch.cuda.device_count()
    
    gb_memory_available = torch.cuda.get_device_properties(
        0).total_memory / (1024 ** 3)
    if not quiet:
        print(f"{gb_memory_available}GB GPU memory available")
    if gb_memory_available >= 14:
        return 16
    else:
        return 4


def model_grad_norm(model: nn.Module):
    """
    Computes the L2 norm of the model gradients

    Parameters
    ----------
    model:
        The model
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    norms = [torch.norm(p.grad.detach(), 2) for p in parameters]
    return torch.norm(torch.stack(norms), 2)


def model_param_norm(model: nn.Module):
    """
    Computes the L2 norm of the model parameters
    
    Parameters
    ----------
    model:
        The model
    """
    parameters = [p for p in model.parameters()]
    norms = [torch.norm(p.detach(), 2) for p in parameters]
    return torch.norm(torch.stack(norms), 2)

class Trainer:

    def __init__(self, model: BERTModel, optimizer, writer=None, batch_size=4, quiet=False, loss_fn=None, max_grad_norm=10.0) -> None:
        """
        Initializes a Trainer with the provided parameters. If there is more than one GPU, it will automatically wrap the
        model into a DataParallel object

        Parameters
        ----------
        model:
            The model to train
        optimizer:
            The optimizer to train with
        writer:
            The tensorboard writer to output to
        batch_size:
            The global batch size to use
        quiet:
            Whether or not to suppress console output
        loss_fn:
            The loss function to use. If none, will use CrossEntropyLoss
        max_grad_norm:
            The threshold to use when clipping gradients. 
        """
        self.tokenize_fn = model.tokenize
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss(reduction='none')

        self.loss_fn = loss_fn.cuda()

        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.subbatch_size = get_subbatch_size(quiet)
        if model.hidden_size > 1000:
            self.subbatch_size //= 4
        print(f"Using subatch size {self.subbatch_size}")
        self.scaler = GradScaler()
        self.writer = writer
        self.batch_num = 0
        self.epoch_num = 0
        self.quiet = quiet

    def train_epoch(self, dataset: BertClassificationDataset):
        """
        Trains for one epoch on the provided data with sub-batching.

        Returns
        -------

        losses: List[float]
            The losses for each batch during training.
        """
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate
        )

        self.model.train()
        losses = list()
        self.optimizer.zero_grad()
        for batch in tqdm(dataloader, disable=self.quiet, desc="Training"):
            input_ids: torch.Tensor = batch[0].cuda()
            attention_mask: torch.Tensor = batch[1].cuda()
            token_type_ids: torch.Tensor = batch[2].cuda()
            labels: torch.LongTensor = batch[3].cuda()

            param_norm = model_param_norm(self.model)
            self.writer.add_scalar('param_norm', param_norm, global_step=self.batch_num)

            batch_size = labels.size()[0]
            subbatch_start = 0

            while subbatch_start < batch_size:
                subbatch_stop = min(
                    subbatch_start + self.subbatch_size, batch_size
                )

                with autocast():
                    output = self.model.forward(
                        input_ids[subbatch_start:subbatch_stop],
                        attention_mask[subbatch_start:subbatch_stop],
                        token_type_ids[subbatch_start:subbatch_stop]
                    )
                    loss = self.loss_fn(output, labels[subbatch_start:subbatch_stop])

                losses.append(loss.detach().cpu().numpy())
                loss = loss.sum() / batch_size
                self.scaler.scale(loss).backward()
                subbatch_start += self.subbatch_size

            self.scaler.unscale_(self.optimizer)
            grad_norm = model_grad_norm(self.model)
            self.writer.add_scalar('gradient_norm', grad_norm, global_step=self.batch_num)

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_grad_norm
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.batch_num += 1

        losses = np.hstack(losses)
        if self.writer is not None:
            self.writer.add_scalar(
                "loss/train", np.mean(losses), global_step=self.epoch_num
            )
        self.epoch_num += 1
        return losses

    def validate_model(self, text, labels) -> Tuple[float, np.ndarray]:
        """
        Validates the data against hte provided labels. 

        Returns
        -------

        total_loss:
            The mean validation loss

        predictions:
            A Numpy array containing the prdictions generated (0 or 1).
        """
        batch_size = self.subbatch_size * 4
        n_batches = int(math.ceil(len(text) / batch_size))
        predictions = list()
        losses = list()

        self.model.eval()
        for batch_num in tqdm(range(n_batches), desc="Validating", disable=self.quiet):
            batch_start = batch_num * batch_size
            batch_stop = min(batch_start + batch_size, len(text))

            batch_emails = text[batch_start:batch_stop]
            batch_labels = labels[batch_start:batch_stop]
            batch_labels = torch.tensor(batch_labels).cuda()

            tokenized_output = self.tokenize_fn(batch_emails)

            input_ids = torch.tensor(tokenized_output['input_ids']).cuda()
            attention_mask = torch.tensor(tokenized_output['attention_mask']).cuda()
            token_type_ids = torch.tensor(tokenized_output['token_type_ids']).cuda()

            with autocast(), torch.no_grad():
                out_tensor = self.model.forward(
                    input_ids, attention_mask, token_type_ids)
                loss = self.loss_fn(out_tensor, batch_labels)
                losses.append(loss)

                result = torch.argmax(out_tensor, dim=1).cpu().numpy()
                predictions.append(result)

        predictions = np.hstack(predictions)
        losses = torch.hstack(losses)
        total_loss = torch.mean(losses).cpu().item()
        return total_loss, predictions

    def get_model(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model
