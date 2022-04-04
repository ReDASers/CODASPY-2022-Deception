# Deep Learning Experiments 

## Dependencies 

Requirements can be downloaded by running 

```
pip install -r requirements.txt
```
## Training Script

The training script is `train_models.py` which loads a jsonlines datset, splits it 80/10/10 into a train-validate-test split, and trains for 10 epoches with early stopping. It supports training on multiple GPUs using PyTorch's `DataParallel` wrapper and mini-batch training with gradient accumulation. It outputs a trained model and a json file containing the indexies of the validation set used and the held out test set. 

Users can either pass in training paraemters via command-line arguments or via a json file. If a user chooses to use command-line arguments, the script will automatically save training parameters in a json file. 

Current trainaing parameters include the following: 

* `datasets` - The list of datasets to train on. 
* `base_model` - The pre-trained model to use as the base. Our code currently supports `bert-base-uncased` and `roberta-base`
* `output_layers` - The number of layers to use in the output head. (Can be 1, 2 or 3). 
* `weighted_loss` - Whether to use a weighted loss function
* `lr` - The ADAM learning rate to use
* `max_grad_norm` - The threshold used for gradient clipping. 
* `weight_decay` - The weight decay value to use
* `dropout_rate` - The droupout value to use before the linear layer
* `batch_size` - The batch size to use between optimizer steps


To load training parameters from a json file, run

```
python train_model.py --config [JSON FILE PATH]
```

## Evaluation Script

We have two options for evaluating models - one on GPUs and one on Google Colab TPUs. Both methods can evaluate a model against multiple datasets. If a dataset was used to train the model, they evaluates the model against the test set held out during training. Otherwise, they evaluate the model against the entire dataset. 

To test models on GPUs, use the `test_model.py` script as follows: 

```
test_model.py --model model_name --datasets dataset1.jsonl dataset2.jsonl
```

To test models on TPUs, import the test_xla_mp function as follows:

```
import os

from glob import glob 
os.chdir("/content/deep learning")
print(os.getcwd())
from test_model_xla import test_xla_mp
```

We have a full evaluation notebook in [here](CrossDataset_TPU.ipynb)