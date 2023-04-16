
import pandas as pd
import numpy as np
import time
from ast import literal_eval

from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel

import torch
from torch.utils.data import Dataset, DataLoader

from model_utils import *

MAX_LEN = 16
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = "cuda:0" if torch.cuda.is_available() else "cpu"

root_path = 'D:/A Course/interview/archive/'
addr = root_path + "mtsamples_added_target.csv"
df = pd.read_csv(addr)
df = df[df['transcription'].notna()]
df = df[:500]



t2 = time.time()
df['target_list'] = df['target_list'].apply(lambda x: literal_eval(x))
print('time for load target_list:',time.time()-t2)
df.info()

print(type(df.at[0, 'target_list']))


# print(df.at[200, 'target_list'])

    
train_size = 0.9
input_dataset = df.sample(frac=train_size,random_state=200)
test_dataset = df.drop(input_dataset.index).reset_index(drop=True)
input_dataset = input_dataset.reset_index(drop=True)

train_size = 0.8
train_dataset = input_dataset.sample(frac=train_size,random_state=200)
valid_dataset = input_dataset.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(valid_dataset.shape))


training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
validation_set = CustomDataset(valid_dataset, tokenizer, MAX_LEN)

test = training_set.__getitem__(10)
print(test, test['targets'].reshape(-1, 1).shape)
 
model = BERTClass()
model.to(device)

 
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

checkpoint_path = root_path+'checkpoint/current_checkpoint.pt'
 
best_model = root_path+'best_model/best_model.pt'

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **test_params)

t3 = time.time()
trained_model = train_model(1, 3, np.Inf, training_loader, validation_loader, model, 
                       optimizer,checkpoint_path,best_model)
print('time for train:',time.time()-t3)

# predict 
trained_model