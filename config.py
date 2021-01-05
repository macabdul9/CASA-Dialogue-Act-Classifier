import torch

config = {
    
    # data 
    "data_dir":"/home/macab/research/CASA-Dialogue-Act-Classifier/data/",
    "dataset":"switchboard",
    "text_field":"clean_text",
    "label_field":"act_label_1",
    
    "max_len":256,
    "batch_size":8,
    "num_workers":4,
    
    # model
    "model_name":"roberta-base", #roberta-base
    "hidden_size":768,
    "num_classes":54, # there were 54 classes in switchboard corpus 
    
    # training
    "save_dir":"../working/",
    "project":"dialogue-act-classification",
    "run_name":"context-aware-attention-dac",
    "lr":1e-5,
    "monitor":"val_accuracy",
    "min_delta":0.001,
    "filepath":"../working/{epoch}-{val_accuracy:4f}",
    "precision":32,
    "average":"micro",
    "epochs":50,
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
}