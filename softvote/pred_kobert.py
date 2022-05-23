""" 추론 코드
"""
import os
import random
import torch
from torch.utils.data import DataLoader
import numpy as np
from model.model import ElectraSummarizer
from model.model import FunnelSummarizer
from model.model import BertSummarizer
from modules.dataset import ElectraCCustomDataset
from modules.dataset import FunnelCCustomDataset
from modules.dataset import BertCCustomDataset
from modules import trainer
from modules.metrics import Hitrate
import pandas as pd
import pickle

# Set random seed
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
random.seed(42)
DATA_DIR = './data/kobert'
TRAINED_MODEL_PATH = './results/train/kobert/best.pt'
BATCH_SIZE = 4
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset & dataloader
test_dataset = BertCCustomDataset(data_dir=DATA_DIR, mode='test')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Model
model = BertSummarizer().to(device)
model.load_state_dict(torch.load(TRAINED_MODEL_PATH)['model'])

# Set metrics & Loss function
metric_fn = Hitrate
loss_fn = torch.nn.BCELoss(reduction='none')

# Set trainer
trainer1 = trainer.Trainer(model, device, loss_fn, metric_fn)

# Predict
preds3 = trainer1.test_epoch(test_dataloader, epoch_index=0)

with open("kobert.pkl", 'wb') as f:
    pickle.dump(preds3, f)
