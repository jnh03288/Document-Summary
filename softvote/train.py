import os
import json
from sklearn.model_selection import train_test_split
import pandas as pd

# Load / split data into train / validation set
# 최초 1회만 실행

# 제출시 아래의 경로에서 데이터 읽어오도록 주석 해제해야함
train = pd.read_json("/DATA/Final_DATA/task05_train/train.json", orient='records', encoding='utf-8-sig')
test = pd.read_json("/DATA/Final_DATA/task05_test/test.json", orient='records', encoding='utf-8-sig')
    
models_list = ['kobert', 'funnel', 'elec']

for model_type in models_list:
    if model_type == 'elec':       
        tr = train.sample(frac=0.80,random_state=17)
        val = train.drop(tr.index)

    else:
        tr, val = train_test_split(train,train_size=0.8, random_state=1345)
    tr.to_json(f"./data/{model_type}/train/train.json", orient='records')
    val.to_json(f"./data/{model_type}/val/val.json", orient='records')
    test.to_json(f"./data/{model_type}/test/test.json", orient='records')

print("TRAIN KOBERT MODEL")
os.system("python3 train_kobert.py")
print("TRAIN FUNNEL MODEL")
os.system("python3 train_funnel.py")
print("TRAIN ELCE MODEL")
os.system("python3 train_elec.py")
