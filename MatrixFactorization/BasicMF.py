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

    
class BasicMatrixFactorization(torch.nn.Module):
    def __init__(self, nQuestions, nUser, nfactors):
        super().__init__()
        self.user_factors = torch.nn.Embedding(nUser, nfactors, sparse=False)
        self.question_factors = torch.nn.Embedding(nQuestions, nfactors, sparse=False)
        self.user_biases = torch.nn.Embedding(nUser, 1, sparse=False)
        self.question_biases = torch.nn.Embedding(nQuestions, 1, sparse=False)

    def forward(self, question, user):
        sigmoid = torch.nn.Sigmoid()
        result = sigmoid((self.user_factors(user)* self.question_factors(question)).sum(1).unsqueeze(1) + self.user_biases(user) + self.question_biases(question))
        return result, self.user_factors(user), self.question_factors(question), self.user_biases(user), self.question_biases(question)

class MatrixFactorization():
    def __init__(self, parameters, f):
        params = ["nfactors", "train_sampleSize", "loss_func", "lr", "lambda", "epochs"]
        for p in params:
            if p not in parameters.keys():
                print(p + " not provided. Taking default value.")
        self.nfactors = parameters["nfactors"] if ("nfactors" in parameters.keys()) else 32
        self.train_sampleSize = parameters["train_sampleSize"]
        self.loss_func = parameters["loss_func"] if ("loss_func" in parameters.keys()) else torch.nn.MSELoss()
        self.lr = parameters["lr"] if ("lr" in parameters.keys()) else  1e-2
        self.lambda_reg = parameters["lambda"] if ("lambda" in parameters.keys()) else 0
        self.epochs = parameters["epochs"] if ("epochs" in parameters.keys()) else 25
        self.model = BasicMatrixFactorization(self.train_sampleSize+1,self.train_sampleSize+1, nfactors=self.nfactors)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.f = f
        
    def losscalc(self, dl):  
        loss = 0
        for x, y in dl:
            prediction, ufactors, qfactors, ubias, qbias = self.model(x[:, 0], x[:, 1])
            loss += self.loss_func(prediction, y)
            loss += self.lambda_reg * (torch.sum(ufactors ** 2) + torch.sum(qfactors ** 2) + torch.sum(ubias **2) + torch.sum(qbias ** 2)) / len(prediction)
        return loss
        
    def train(self, train_dl, test_dl, val_dl): 
        f = self.f
        print("Starting training....", file = f)
        print("Epochs: " + str(self.epochs), file = f)
        print("size latent space: "  + str(self.nfactors), file = f)
        print("lambda: " + str(self.lambda_reg), file = f)

        trainLossList = []
        testLossList = []
        validationLossList = []

        for epoch in range(self.epochs):
            print(epoch)
            for  x, y in train_dl:
                question = x[:, 0]
                user = x[:, 1]
                y_correct = y.type(torch.FloatTensor)      
                prediction, userfactors, questionfactors, userbias, questionbias = self.model(question, user)
                loss = self.loss_func(prediction, y_correct)
                loss += self.lambda_reg * (torch.sum(userfactors ** 2) + torch.sum(questionfactors ** 2) + torch.sum(userbias **2) + torch.sum(questionbias ** 2)) / len(y_correct)    
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            with torch.no_grad():
                valid_loss = 0
                train_loss = 0
                test_loss = 0
                train_loss = self.losscalc(train_dl)
                test_loss = self.losscalc(test_dl)
                valid_loss = self.losscalc(val_dl)                  

                trainLossList.append(train_loss/len(train_dl))
                testLossList.append(test_loss/len(test_dl))
                validationLossList.append(valid_loss/ len(val_dl))
                
                print(epoch, train_loss / len(train_dl),file = f)
                print(epoch, test_loss / len(test_dl),file = f)
                print(epoch, valid_loss / len(val_dl),file = f)



        print("finished", file = f)
        print(trainLossList, file =  f)
        print(testLossList, file = f)
        print(validationLossList, file = f)
        torch.save(self.model, "01.pt")
        return self.model
    
