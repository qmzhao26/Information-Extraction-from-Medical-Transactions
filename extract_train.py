
import pandas as pd
import numpy as np
import time
from ast import literal_eval

from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel

import torch
from torch.utils.data import Dataset

from interview_extract_model_utils import *

MAX_LEN = 16
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = "cuda:0" if torch.cuda.is_available() else "cpu"


addr = "D:/A Course/interview/archive/mtsamples_added_target.csv"
df = pd.read_csv(addr)
df = df[df['transcription'].notna()]
df = df[:200]



t2 = time.time()
df['target_list'] = df['target_list'].apply(lambda x: literal_eval(x))
print('time for load target_list:',time.time()-t2)
df.info()

print(type(df.at[0, 'target_list']))


# print(df.at[200, 'target_list'])

    
train_size = 0.8
train_dataset = df.sample(frac=train_size,random_state=200)
valid_dataset = df.drop(train_dataset.index).reset_index(drop=True)
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

checkpoint_path = 'checkpoint/current_checkpoint.pt'
 
best_model = 'best_model/best_model.pt'
 
trained_model = train_model(1, 4, np.Inf, training_set, validation_set, model, 
                       optimizer,checkpoint_path,best_model)