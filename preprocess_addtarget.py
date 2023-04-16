
import pandas as pd
import numpy as np
import time

root_path = 'D:/A Course/interview/archive/'
addr = root_path+"mtsamples_added_medi.csv"
df = pd.read_csv(addr)
df = df[df['keywords'].notna()]
# build target_list
target_list = []
target_set = set()

for index, row in df.iterrows():
    keys = row['keywords'].split(',')
    for it in keys:
        # print(it)
        target_set.add(it)

target_list = list(target_set)
print('target list len: ', len(target_list))

ids = [id for id in range(len(target_list))]

target_dict = dict(zip(ids, target_list))

np.save(root_path+'target_dict.npy', target_dict, allow_pickle=True)

np.save(root_path+'target_list.npy', target_list, allow_pickle=True)

def add_target_list(keyword_str):
    target = [0 for i in range(len(target_list))]
    keyword_list = keyword_str.split(',')
    for it in keyword_list:
        idx = target_list.index(it)
        # print(it, idx)
        if idx:
            target[idx] = 1
    return target

df['target_list'] = df['keywords'].apply(lambda x : add_target_list(x))
df['target_list'].astype(object)
df.info()


df.to_csv(root_path+"mtsamples_added_target.csv", index=False)