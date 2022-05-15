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
from datetime import datetime
import utils
import BasicMF as MF


class InterpretableLens(torch.nn.Module):
   def __init__(self, ntopics, nfactors, model):
       super().__init__()
       self.g = torch.nn.Linear(nfactors, ntopics)
       self.model = model
       
   def forward(self, user, topic, difficulty, user_to_ix):
       sigmoid = torch.nn.Sigmoid()
       u_idx = np.empty((0,1), dtype=np.int32)
       for u in user:
            u_idx = np.vstack([u_idx, user_to_ix[u.item()]])

       userfactors = torch.squeeze(self.model.user_factors(torch.tensor(u_idx))).detach()
       r = self.g(userfactors).mul((topic*difficulty.reshape(len(difficulty), 1))).sum(axis = 1).reshape((-1, 1))
       return sigmoid(r)


class LensModelTrainer():
    def __init__(self, parameters, f):
        params = ["nfactors", "loss_func", "lr", "epochs", "ntopics"]
        for p in params:
            if p not in parameters.keys():
                print(p + " not provided. Taking default value.")
        self.nfactors = parameters["nfactors"] if ("nfactors" in parameters.keys()) else 32
        self.loss_func = parameters["loss_func"] if ("loss_func" in parameters.keys()) else torch.nn.MSELoss()
        self.lr = parameters["lr"] if ("lr" in parameters.keys()) else  1e-4
        self.epochs = parameters["epochs"] if ("epochs" in parameters.keys()) else 20
        modelPath = parameters["MF"] if ("MF" in parameters.keys()) else "MF_6.pt"
        model = torch.load(modelPath)
        model.eval()
        ntopics = parameters["ntopics"] if ("ntopics" in parameters.keys()) else 388
        self.lensmodel = InterpretableLens(ntopics, self.nfactors, model)
        self.optimizer = torch.optim.Adam(self.lensmodel.parameters(), self.lr)
        self.user_to_ix = parameters["user_to_ix"] 
        self.f = f
        
    def train(self, train_df, val_df):
        f = self.f
        print("Starting Lensmodel training....", file = f)
        print("Epochs: " + str(self.epochs), file = f)

        trainLossList = []
        validationLossList = []

        for epoch in range(self.epochs):
            print(epoch)
            for chunk in train_df:
                user = torch.tensor(chunk["UserId"].values)
                topic = torch.tensor(np.vstack(chunk["topic"].values))
                diff = torch.tensor(chunk["difficulty"].values).float()
                prediction = self.lensmodel(user, topic, diff, self.user_to_ix).float().reshape(len(chunk), 1)
                y_correct = torch.tensor(chunk["model_predRating"].values).float().reshape(len(chunk), 1)
                loss = self.loss_func(prediction, y_correct)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            with torch.no_grad():
                train_loss = 0
                for chunk in train_df:
                    user = torch.tensor(chunk["UserId"].values)
                    topic = torch.tensor(np.vstack(chunk["topic"].values))
                    diff = torch.tensor(chunk["difficulty"].values).float()
                    prediction = self.lensmodel(user, topic, diff, self.user_to_ix).float().reshape(len(chunk), 1)
                    y_correct = torch.tensor(chunk["model_predRating"].values).float().reshape(len(chunk), 1)

                    loss = self.loss_func(prediction, y_correct)
                    train_loss += loss
                print(train_loss)

                val_loss = 0    
                for chunk in val_df:
                    user = torch.tensor(chunk["UserId"].values)
                    topic = torch.tensor(np.vstack(chunk["topic"].values))
                    diff = torch.tensor(chunk["difficulty"].values).float()
                    prediction = self.lensmodel(user, topic, diff, self.user_to_ix).float().reshape(len(chunk), 1)
                    y_correct = torch.tensor(chunk["model_predRating"].values).float().reshape(len(chunk), 1)

                    loss = self.loss_func(prediction, y_correct)
                    val_loss += loss
                print(val_loss)

                trainLossList.append(train_loss/len(train_df))
                validationLossList.append(val_loss/len(val_df))


        print(trainLossList, file = f)
        print(validationLossList, file = f)
        
        torch.save(self.lensmodel, "Lensmodel_nograd_high_lr.pt")


        return self.lensmodel
                    




















    
        
        