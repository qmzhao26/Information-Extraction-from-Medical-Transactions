
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
df = df[:128]
df['target_list'] = df['target_list'].apply(lambda x: literal_eval(x))

trained_model = BERTClass()
optimizer = torch.optim.Adam(params=trained_model.parameters(), lr=LEARNING_RATE)
model, optimizer, checkpoint, valid_loss_min = load_ckp(root_path+'best_model/best_model.pt', trained_model, optimizer)
model.to(device)
model.eval()


input = CustomDataset(df, tokenizer, MAX_LEN)
input_params = {'batch_size': 32,
                'shuffle': False,
                'num_workers': 0
                }
input_loader = DataLoader(input, **input_params)


t4 = time.time()
res = predict(trained_model, input_loader)
print('time for predict:',time.time()-t4)
print(type(res), len(res))

target_dict = np.load(root_path+'target_dict.npy', allow_pickle=True).item()
target_list = np.load(root_path+'target_list.npy', allow_pickle=True)

# Find the index of all non-zero values in a list
def find_nonzero_index(l):
    return [i for i, e in enumerate(l) if e != 0]

all_word_list = []

for pred in res:
    word_list = []
    indexs = find_nonzero_index(pred)
    print(indexs)
    for idx in indexs:
        word_list.append(target_dict[idx])
    print(word_list)
    all_word_list.append(word_list)

df['classification'] = all_word_list
df.info()
df.to_csv(root_path+"mtsamples_classify.csv", index=False)
