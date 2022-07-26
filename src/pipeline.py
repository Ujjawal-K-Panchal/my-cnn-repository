# -*- coding: utf-8 -*-
"""
Title: CNN.

Description: A Convolutional Neural Network.

Created on Sun Jul 10 2022 19:19:01 2022.

@author: Ujjawal .K. Panchal
===

Copyright (c) 2022, Ujjawal Panchal.
All Rights Reserved.
"""
import argparse, time, os
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets

from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from pathlib import Path

from tqdm import tqdm
from model import CNN
from typing import Union

#static_vars.
SUPPORTED_DATASETS = ["MNIST", "Fashion"]

dataset_loaders = {
    "MNIST" :  datasets.MNIST,
    "Fashion": datasets.FashionMNIST,
}

#load model.
def make_model(
    model_class: torch.nn.Module = CNN,
    device: str = "cpu",
    *args, **kwargs
):
    """
    load_model() helps the user to load any model from any class.

    ---
    Arguments:
        1. model_class: torch.nn.Module (default CNN) = any modelclass which is `torch.nn.Module`.
        2. device: str (default cpu) = any device str available on system.
        3. *args: any extra positional arguments; passed to `model_class`.
        4. **args: any extra keyword arguments; passed to `model_class`.
    """
    model = model_class(*args, **kwargs).to(device)
    return model

#load dataset.
def load_dataset(
    dname: str = SUPPORTED_DATASETS[0],
    root: str = "./data",
    transform: transforms = transforms.ToTensor(),
    download: bool = True,
    **kwargs
) -> datasets:
    """
        load_dataset() helps the user to load any dataset from supported ones.
    """
    assert dname in SUPPORTED_DATASETS, f"<!>: DATASET: `{dname}` not supported." 
    train_set = dataset_loaders[dname](
                        root = root,
                        train = True,
                        transform = transform,
                        download = download,
                        **kwargs
                    
                )
    test_set = dataset_loaders[dname](
                        root = root,
                        train = False,
                        transform = transform,
                        download = download,
                        **kwargs
                )
    return  train_set, test_set

def get_dataLoader(
    dset,
    batch_size: int = 32,
    shuffle: bool = False
) -> torch.utils.data.DataLoader:
    """
    Get an iterable form for the dataset.
    ---
    Args:
        1. batch_size: int (default = 32) = size of each batch coming out of the set.
        2. shuffle: bool (default = True) = if to or not to shuffle the dataset when creating batches.
    """
    dLoader = torch.utils.data.DataLoader(
                        dset,
                        batch_size = batch_size,
                        shuffle = shuffle
                )
    return dLoader

def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int = 1,
    lr: float = 1E-3,
    device: Union[str, torch.device] = "cpu",
    optimizer: torch.optim = torch.optim.Adam,
    loss_fn: torch.nn = torch.nn.CrossEntropyLoss(),

) -> torch.nn.Module:
    """
    Train a network on a given dataset iterable.
    ---
    Args:
        1. model: torch.nn.Module (req) = model which to train.
        2. train_loader: torch.utils.data.DataLoader (req) = train set iterable.
        3. test_loader: torch.utils.data.DataLoader (req) = test set iterable.
    """
    model.to(device)

    model.train()
    optimizer = optimizer(model.parameters(), lr = lr)
    softmax = F.softmax
    correct = 0
    total = 0
    total_loss = 0 

    #0. run n epochs.
    for epoch in range(epochs):
        with tqdm(train_loader, unit = "batch") as train_epoch:
            #1. set some bat stats.
            train_epoch.set_description(f"E {epoch}, loss: {total_loss:.2f}, train acc: {correct * 100 /total if total else total:.2f}%")
            total_loss = 0
            correct = 0
            total = 0
            for data, target in train_epoch:
                #2. get data and predict.
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = softmax(model(data), dim = 1)
                preds = output.argmax(dim = 1)

                
                #3. calculate loss & backpropogate gradients.
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                #4. some intermediate stat collection.
                correct += (preds == target).sum().item()
                total += len(preds)
                total_loss += loss.item()
    return model

def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: Union[str, torch.device] = "cpu",
) -> tuple:
    """
    Test a given model on a test set sharded iterable.
    ---
    Args:
        1. model: torch.nn.Module = Model that is to be tested.
        2. test_loader: torch.utils.data.DataLoader = Iterable sharded/batched test set. 
    """
    #0. make settings.
    model.eval()
    y_pred, y_truth = [], []
    test_bar = tqdm(test_loader, unit = "batch")
    
    #1. iterate through the test set.
    for i, (x, y) in enumerate(test_bar):
        test_bar.set_description(f"Testing batch: {i:3d}")
        x, y = x.to(device), y.to(device)
        
        #2. make predictions.
        outputs = model(x)
        predictions = F.softmax(outputs, dim = 1).argmax(dim = 1)
        
        #3. store predictions and truth.
        y_truth.extend(y.detach().cpu().numpy())
        y_pred.extend(predictions.detach().cpu().numpy())
    test_bar.close()
    
    #3. calculate scores.
    accuracy = accuracy_score(y_truth, y_pred)
    f1 = f1_score(y_truth, y_pred, average = None)
    cm = confusion_matrix(y_truth, y_pred, labels = [x for x in range(1, 11)])
    return accuracy, list(f1), cm


def save_model(
    model: torch.nn.Module,
    modelname: str = "CNN",
    overwrite: bool = True

) -> Path:
    """
    Save the weights of a given model with a particular name inside snapshots/.
    ---
    Args:
        1. model: torch.nn.Module = Model that is to be tested.
        2. modelname: str (default = "CNN") = Name under which to save.
    """
    #0. do some checks.
    assert (" " not in modelname), f"<!>: Name cannot have spaces in it."


    savepath = Path("snapshots", f"{modelname}.model")

    if (os.path.exists(savepath) and not overwrite):
        raise Exception("<!>: a model is already saved under this name.")

    elif os.path.exists(savepath):
        os.remove(savepath)

    #1. Make modelpath.
    torch.save(model.cpu().state_dict(), savepath)

    #2. return savepath.
    return savepath

def load_model(
    model: torch.nn.Module,
    modelname: str = "CNN",
    device: str = "cpu"
) -> torch.nn.Module:
    """
    Load model from the snapshots/ directory.
    ---
    Args:
        1. model: torch.nn.Module = source model on which to override with saved weights.
        2. modelname: str (default = "CNN") = model name which was previously saved in snapshots/ folder.
    """
    loadpath = Path("snapshots", f"{modelname}.model")
    model.load_state_dict(torch.load(loadpath))
    return model.to(device)

#unit test.
if __name__ == "__main__":
    print(f"unit tests for the pipeline module.")
    
    #1. make_models.
    model = make_model(CNN, c = 1)
    print(f"make_model() works fine.")

    #2. load_datasets.
    train_set, test_set = load_dataset(
        SUPPORTED_DATASETS[0],

    )
    print(f"load_dataset() works fine.")

    #3. get_dataLoader.
    loader = get_dataLoader(test_set, batch_size=512)
    print(f"get_dataLoader() works fine.")

    #4. train.
    model = train(model, loader)
    print(f"train() works fine.")

    #5. test.
    acc, f1, cm = test(model, loader)
    print(f"{acc=}\n{f1=}\n{cm=}")
    print("test() works fine.")

    #6. save model.
    save_model(model)
    print("save_model() works fine.")

    #7. load model.
    load_model(model)
    print(f"load_model() works fine.")

