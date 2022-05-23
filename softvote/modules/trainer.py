# -*- coding: utf-8 -*-
"""Trainer 클래스 정의
"""

import torch
from tqdm import tqdm
from transformers import ElectraModel, ElectraTokenizer
import json

class Trainer():
    """ Trainer
        epoch에 대한 학습 및 검증 절차 정의
    
    Attributes:
        model (`model`)
        device (str)
        loss_fn (Callable)
        metric_fn (Callable)
        optimizer (`optimizer`)
        scheduler (`scheduler`)
    """

    def __init__(self, model, device, loss_fn, metric_fn, optimizer=None, scheduler=None, logger=None):
        """ 초기화
        """
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger

    def train_epoch(self, dataloader, epoch_index):
        """ 한 epoch에서 수행되는 학습 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.train()
        self.train_total_loss = 0
        pred_lst = []
        target_lst = []
        for batch_index, (data, target) in enumerate(tqdm(dataloader)):
            self.optimizer.zero_grad()
            src = data[0].to(self.device)
            clss = data[1].to(self.device)
            segs = data[2].to(self.device)
            mask = data[3].to(self.device)
            mask_clss = data[4].to(self.device)
            target = target.float().to(self.device)
            sent_score = self.model(src, segs, clss, mask, mask_clss)
            loss = self.loss_fn(sent_score, target)
            loss = (loss * mask_clss.float()).sum()
            self.train_total_loss += loss
            loss.backward()
            self.optimizer.step()
            pred_lst.extend(torch.topk(sent_score, 3, axis=1).indices.tolist())
            target_lst.extend(torch.where(target==1)[1].reshape(-1,3).tolist())
        self.scheduler.step()    
        self.train_mean_loss = self.train_total_loss / len(dataloader)
        self.train_score = self.metric_fn(y_true=target_lst, y_pred=pred_lst)
        msg = f'Epoch {epoch_index}, Train, loss: {self.train_mean_loss}, Score: {self.train_score}'
        print(msg)
        self.logger.info(msg) if self.logger else print(msg)

    def validate_epoch(self, dataloader, epoch_index):
        """ 한 epoch에서 수행되는 검증 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.eval()
        self.val_total_loss = 0
        pred_lst = []
        target_lst = []
        sent_scores = []

        with torch.no_grad():
            for batch_index, (data, target) in enumerate(dataloader):
                src = data[0].to(self.device)
                clss = data[1].to(self.device)
                segs = data[2].to(self.device)
                mask = data[3].to(self.device)
                mask_clss = data[4].to(self.device)
                target = target.float().to(self.device)
                sent_score = self.model(src, segs, clss, mask, mask_clss)
                sent_scores.extend(sent_score.cpu())
                loss = self.loss_fn(sent_score, target)
                loss = (loss * mask_clss.float()).sum()
                self.val_total_loss += loss
                pred_lst.extend(torch.topk(sent_score, 3, axis=1).indices.tolist())
                target_lst.extend(torch.where(target==1)[1].reshape(-1,3).tolist())
            self.val_mean_loss = self.val_total_loss / len(dataloader)
            self.validation_score = self.metric_fn(y_true=target_lst, y_pred=pred_lst)
            msg = f'Epoch {epoch_index}, Validation, loss: {self.val_mean_loss}, Score: {self.validation_score}'
            print(msg)
            self.logger.info(msg) if self.logger else print(msg)
    
        return sent_scores

           
    def test_epoch(self, dataloader, epoch_index):
        """ 한 epoch에서 수행되는 검증 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.eval()
        self.val_total_loss = 0
        pred_lst = []
        target_lst = []
        sent_score1 = []
        with torch.no_grad():
            for batch_index, data in enumerate(tqdm(dataloader)):
                src = data[0].to(self.device)
                clss = data[1].to(self.device)
                segs = data[2].to(self.device)
                mask = data[3].to(self.device)
                mask_clss = data[4].to(self.device)
                #target = target.float().to(self.device)
                sent_score = self.model(src, segs, clss, mask, mask_clss)
                #loss = self.loss_fn(sent_score, target)
                #loss = (loss * mask_clss.float()).sum()
                #self.val_total_loss += loss
                sent_score1.extend(sent_score.cpu())
                pred_lst.extend(torch.topk(sent_score, 5, axis=1).indices.tolist())
                #target_lst.extend(torch.where(target==1)[1].reshape(-1,3).tolist())
                
          
        
        return sent_score1