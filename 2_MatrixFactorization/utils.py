import pandas as pd
import torch
from scipy import sparse
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import time 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
import utils


def loadTestData(path, bs):
    data = pd.read_csv(path)[["QuestionId", "UserId", "IsCorrect"]]
    sampleSize = data['UserId'].nunique()
    x_train = torch.tensor(data[["QuestionId", "UserId"]].values)
    y_train = torch.tensor(data[["IsCorrect"]].values)
    ds = TensorDataset(x_train, y_train)
    dl = DataLoader(ds, batch_size=bs, shuffle=True)    
    return dl, sampleSize

def loadData(path, bs):
    data = pd.read_csv(path)[["QuestionId", "UserId", "IsCorrect"]]
    x = (data[["QuestionId", "UserId"]].values)
    y = (data[["IsCorrect"]].values)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=2000000, random_state=20)
    sampleSize = len(np.unique(X_train[:, 1:2]))
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    
    ds_train = TensorDataset(X_train, y_train)
    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True)    
    dl_test = TensorDataset(X_test, y_test)
    dl_test = DataLoader(dl_test, batch_size=bs, shuffle=True)    
    return dl_train, dl_test, sampleSize
    
    
