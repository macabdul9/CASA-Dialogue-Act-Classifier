# utils 
import torch
import os
import pandas as pd
import numpy as np
import random 
import gc

# data
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchtext.data import BucketIterator

from transformers import BertConfig, BertTokenizerFast, BertModel
from transformers import AutoConfig, AutoModel, AutoTokenizer

# models 
import torch.nn as nn
from transformers import AutoTokenizer
from models.ContextAwareDAC import ContextAwareDAC

# training and evaluation
import wandb
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from dataset.dataset import DADataset



class LightningModel(pl.LightningModule):
    
    def __init__(self, config):
        super(LightningModel, self).__init__()
        
        self.config = config
        
        self.model = ContextAwareDAC(
            model_name=self.config['model_name'],
            hidden_size=self.config['hidden_size'],
            num_classes=self.config['hidden_size'],
            device=self.config['device']
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
    def forward(self, batch):
        logits  = self.model(batch)
        return logits
    
    def configure_optimizers(self):
        return optim.Adam(params=self.parameters(), lr=self.config['lr'])
    
    def train_dataloader(self):
        train_data = load_dataset("csv", data_files=os.path.join(self.config['data_dir'], self.config['dataset'], self.config['dataset']+"_train.csv"))
        train_dataset = DADataset(tokenizer=self.tokenizer, data=train_data, max_len=self.config['max_len'], text_field=self.config['text_field'], label_field=self.config['label_field'])
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])
        return train_loader
    
    def training_step(self, batch, batch_idx):
        
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['label'].squeeze()
        logits = self(batch)
        loss = F.cross_entropy(logits, targets)
        
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        wandb.log({"loss":loss, "accuracy":acc, "f1_score":f1})
        return {"loss":loss, "accuracy":acc, "f1_score":f1}
    
    def val_dataloader(self):
        valid_data = load_dataset("csv", data_files=os.path.join(self.config['data_dir'], self.config['dataset'], self.config['dataset']+"_valid.csv"))
        valid_dataset = DADataset(tokenizer=self.tokenizer, data=valid_data, max_len=self.config['max_len'], text_field=self.config['text_field'], label_field=self.config['label_field'])
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])
        return valid_loader
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['label'].squeeze()
        logits = self(batch)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        precision = precision_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        recall = recall_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        return {"val_loss":loss, "val_accuracy":torch.tensor([acc]), "val_f1":torch.tensor([f1]), "val_precision":torch.tensor([precision]), "val_recall":torch.tensor([recall])}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        avg_precision = torch.stack([x['val_precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['val_recall'] for x in outputs]).mean()
        wandb.log({"val_loss":avg_loss, "val_accuracy":avg_acc, "val_f1":avg_f1, "val_precision":avg_precision, "val_recall":avg_recall})
        return {"val_loss":avg_loss, "val_accuracy":avg_acc, "val_f1":avg_f1, "val_precision":avg_precision, "val_recall":avg_recall}
    
    def test_dataloader(self):
        test_data = load_dataset("csv", data_files=os.path.join(self.config['data_dir'], self.config['dataset'], self.config['dataset']+"_test.csv"))
        test_dataset = DADataset(tokenizer=self.tokenizer, data=test_data, max_len=self.config['max_len'], text_field=self.config['text_field'], label_field=self.config['label_field'])
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])
        return test_loader
    
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['label'].squeeze()
        logits = self(batch)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        precision = precision_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        recall = recall_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['average'])
        return {"test_loss":loss, "test_precision":torch.tensor([precision]), "test_recall":torch.tensor([recall]), "test_accuracy":torch.tensor([acc]), "test_f1":torch.tensor([f1])}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
        avg_precision = torch.stack([x['test_precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['test_recall'] for x in outputs]).mean()
        return {"test_loss":avg_loss, "test_precision":avg_precision, "test_recall":avg_recall, "test_acc":avg_acc, "test_f1":avg_f1}