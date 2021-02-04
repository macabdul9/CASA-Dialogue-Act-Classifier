from torch.utils.data import Dataset, DataLoader
import torch

class DADataset(Dataset):
    
    __label_dict = dict()
    
    def __init__(self, tokenizer, data, text_field = "clean_text", label_field="act_label_1", max_len=512, label_dict=None):
        
        self.text = list(data[text_field]) #data['train'][text_field]
        self.acts = list(data[label_field]) #['train'][label_field]
        self.tokenizer = tokenizer
        self.max_len = max_len
        

        if label_dict is None:
            # build/update the label dictionary
            classes = sorted(set(self.acts))
        
            for cls in classes:
                if cls not in DADataset.__label_dict.keys():
                    DADataset.__label_dict[cls]=len(DADataset.__label_dict.keys())
        else:
            DADataset.__label_dict = label_dict
    
    def __len__(self):
        return len(self.text)
    
    def label_dict(self):
        return DADataset.__label_dict
    
    def __getitem__(self, index):
        
        text = self.text[index]
        act = self.acts[index]
        label = DADataset.__label_dict[act]
        
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
            "act":act,
            "label":torch.tensor([label], dtype=torch.long),
        }
