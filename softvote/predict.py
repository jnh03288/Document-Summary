import os
import numpy as np
import pandas as pd
import pickle
import torch
import json
from modules.utils import make_directory

os.system("python3 pred_kobert.py")
os.system("python3 pred_funnel.py")
os.system("python3 pred_elec.py")

with open("kobert.pkl", 'rb') as f:
    ko_pred = pickle.load(f)
with open("funnel.pkl", 'rb') as f:
    funnel_pred = pickle.load(f)
with open("elec.pkl", 'rb') as f:
    elec_pred = pickle.load(f)
    
TEST_PATH = '/DATA/Final_DATA/task05_test/test.json'

with open(TEST_PATH, "r", encoding="utf-8-sig") as input_json:
    test_json = json.load(input_json)

preds = []
output_list = []

for i in range(len(test_json)):
    preds.append(torch.topk((ko_pred[i] + funnel_pred[i] + elec_pred[i])/3, 3, axis=0).indices.tolist())
               
for i, t in enumerate(test_json):
    temp = {}
    temp["ID"] = int(t["id"])
    temp["summary_index1"] = preds[i][0]
    temp["summary_index2"] = preds[i][1]
    temp["summary_index3"] = preds[i][2]
    
    output_list.append(temp)
    
make_directory('../output')               
df= pd.DataFrame(output_list, columns= ['ID', 'summary_index1', 'summary_index2', 'summary_index3'])
df.to_json("../output/result.json", orient='records')