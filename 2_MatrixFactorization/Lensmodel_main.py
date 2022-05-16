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
import Lensmodel as LM



#-------------------------PREPROCESSING DATA -----------------------------
f = open("Output_lm_nograd_highlr.txt",'w')
model = torch.load("MF_model001.pt")
model.eval()
data_path = "data/train_data/train_task_1_2.csv"
#meta_task_path = "data/metadata/answer_metadata_task_1_2.csv"
meta_question_path = "data/metadata/question_metadata_task_1_2.csv"
#meta_student_path = "data/metadata/student_metadata_task_1_2.csv"
subject_metadata_path = "data/metadata/subject_metadata.csv"
val_path = "data/test_data/test_private_answers_task_1.csv"


data = pd.read_csv(data_path)
#meta_task = pd.read_csv(meta_task_path)
meta_question = pd.read_csv(meta_question_path)
#meta_student = pd.read_csv(meta_student_path)
subject_metadata = pd.read_csv(subject_metadata_path)
val_data = pd.read_csv(val_path)


uniqueUsers = pd.unique(data["UserId"])
uniqueQuestions= pd.unique(data["QuestionId"])
user_to_ix = {(word): torch.tensor(i) for i, word in enumerate(uniqueUsers)}
question_to_ix = {word: torch.tensor(i) for i, word in enumerate(uniqueQuestions)}
ix_to_user = {i:word for i, word in enumerate(uniqueUsers)}
ix_to_question = {i:word for i, word in enumerate(uniqueQuestions)}
subject_to_idx = {subject:i for i, subject in enumerate(subject_metadata["SubjectId"])}


full_data = data.append(val_data)
question_difficulty = full_data[["QuestionId", "IsCorrect"]].groupby(by=["QuestionId"]).mean()
question_difficulty.columns = ["difficulty"]
data = data.merge(question_difficulty, left_on='QuestionId', right_on='QuestionId')
val_data = val_data.merge(question_difficulty, left_on='QuestionId', right_on='QuestionId')

def enc(vec):
    a = np.zeros(len(subject_metadata))
    ind = vec[vec.find("[")+1:vec.find("]")].split(",")
    ind = list(map(int, ind))
    ind = [subject_to_idx[x] for x in ind if x in subject_to_idx]
    a[ind] = 1   
    return a
    
meta_question["topic"]  = meta_question["SubjectId"].apply(enc)
data = data.merge(meta_question, left_on='QuestionId', right_on='QuestionId')
val_data = val_data.merge(meta_question, left_on='QuestionId', right_on='QuestionId')


data["model_predRating"] = 0
questions = torch.tensor(data["QuestionId"])
users = torch.tensor(data["UserId"])
model_predRating, _, _, _, _  = (model(questions, users))
data["model_predRating"]  = model_predRating.detach().numpy()

val_questions = torch.tensor(val_data["QuestionId"])
val_users = torch.tensor(val_data["UserId"])
val_model_predRating, _, _, _, _   = (model(val_questions, val_users))
val_data["model_predRating"]  = val_model_predRating.detach().numpy()


bs = 30000
train_df = [data[i:i+bs] for i in range(0,data.shape[0],bs)]
val_df = [val_data[i:i+bs] for i in range(0,val_data.shape[0],bs)]


#-------------------------TRAINING -------------------------

parameters = dict()
parameters["nfactors"] = 32
parameters["loss_func"] = torch.nn.MSELoss()
parameters["lr"] = 1e-4
parameters["epochs"] = 15
parameters["ntopics"] = len(subject_metadata)
parameters["user_to_ix"] = user_to_ix
parameters["MF"] = "MF_model001.pt"

lensmodel = LM.LensModelTrainer(parameters, f)
lensmodel.train(train_df, val_df)





