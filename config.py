import torch

config = {
    # data 
    "data_dir":"../input/a2g-dialogue-act-dataset/A2G-Dialogue-Act-Dataset/",
    "data_files":{
        "train":"a2g_train.csv",
        "valid":"a2g_valid.csv",
        "test":"a2g_test.csv",
    },
    "max_len":256,
    "batch_size":8,
    "num_workers":4,
    
    # model
    "model_name":"roberta-base",
    "hidden_size":768,
    "num_classes":18,
    
    # training
    "save_dir":"../working/",
    "project":"dialogue-act-classification",
    "run_name":"roberta",
    "lr":1e-5,
    "monitor":"val_accuracy",
    "min_delta":0.001,
    "filepath":"../working/{epoch}-{val_accuracy:4f}",
    "precision":32,
    "epochs":20,
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
    
    
}