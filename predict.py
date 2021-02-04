from config import config
import torch

from Trainer import LightningModel

import pytorch_lightning as pl

checkpoint_path='checkpoints/epoch=28-val_accuracy=0.746056.ckpt'
my_device = torch.device('cuda')

model = LightningModel(config=config)
model = model.to(my_device)
model.load_state_dict(torch.load(checkpoint_path, map_location=my_device)['state_dict'])

test_dataloader = model.test_dataloader()
batch = next(iter(test_dataloader))
input_ids = batch['input_ids'].to(my_device)
attention_mask =  batch['attention_mask'].to(my_device)
seq_len= batch['seq_len'].to(my_device)
#one_batch.cuda()

with torch.no_grad():
    outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, seq_len=seq_len).argmax(dim=-1)
    # outputs = model(batch).argmax(dim=-1)
    print(batch['text'])
    print(batch['target'])
    print(outputs)

