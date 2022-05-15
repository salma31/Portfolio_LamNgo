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
import BasicMF as MF




#loading data
bs = 100000
train_dl, test_dl, train_sampleSize = utils.loadData('data/train_data/train_task_1_2.csv', bs)
val_dl, val_sampleSize = utils.loadTestData('data/test_data/test_private_answers_task_1.csv', bs)

#choosing learning parameters
parameters = dict()
parameters["nfactors"] = 32
parameters["train_sampleSize"] = train_sampleSize
parameters["loss_func"] = torch.nn.MSELoss()
parameters["lr"] = 1e-2
parameters["lambda"] = 0.01
parameters["epochs"] = 15

#train Model
f = open("01.txt",'w')
matrixFact = MF.MatrixFactorization(parameters, f)
model = matrixFact.train(train_dl, test_dl, val_dl)