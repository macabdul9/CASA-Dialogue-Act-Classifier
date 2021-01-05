from torch.utils.data import Dataset, DataLoader
import torch

class DADataset(Dataset):
    
    __label_dict = dict()
    
    def __init__(self, tokenizer, data, text_field = "clean_text", label_field="act_label_1", max_len=512):
        self.text = data['train'][text_field]
        self.label = data['train'][label_field]
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        
        # build the label dictionary 
        DADataset.__label_dict.update(dict(zip(list(set(self.label)), torch.arange(start=0, end=len(list(set(self.label))), dtype=int).tolist())))
    
    def __len__(self):
        return len(self.text)
    
    def label_dict(self):
        return DADataset.__label_dict
    
    def __getitem__(self, index):
        
        text = self.text[index]
        label = self.label[index]
        target = DADataset.__label_dict[label]
        
        input_encoding = self.tokenizer.encode_plus(
            text=text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
        )
        
        seq_len = len(self.tokenizer.tokenize(text))
        
        return {
            "text":text,
            "input_ids":input_encoding['input_ids'].squeeze(),
            "attention_mask":input_encoding['attention_mask'].squeeze(),
            "seq_len":seq_len,
            "target":label,
            "label":torch.tensor([target], dtype=torch.long),
        }