
from config import config
from transformers import AutoTokenizer
from models.ContextAwareDAC import ContextAwareDAC
from Trainer import LightningModel
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import os


if __name__=="__main__":

    # create the checkpoints dir
    path = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.isdir(path):
        os.mkdir(path)

        
    logger = WandbLogger(
        name="grammarly-context-aware-attention",
        save_dir=config["save_dir"],
        project=config["project"],
        log_model=True,
    )
    early_stopping = EarlyStopping(
        monitor=config["monitor"],
        min_delta=config["min_delta"],
        patience=5,
    )
    checkpoints = ModelCheckpoint(
        filepath=config["filepath"],
        monitor=config["monitor"],
        save_top_k=1
    )

    # base = ContextAwareDAC()
    # tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = LightningModel(config=config)
    
    trainer = pl.Trainer(
        logger=logger,
        gpus=[0],
        checkpoint_callback=checkpoints,
        callbacks=[early_stopping],
        default_root_dir="./models/",
        max_epochs=config["epochs"],
        precision=config["precision"],
        automatic_optimization=True
    )
    
        
    trainer.fit(model)
    
    trainer.test(model)
    
    
    