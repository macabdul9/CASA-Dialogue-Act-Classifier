
from config import config
from transformers import AutoTokenizer
from models.ContextAwareDAC import ContextAwareDAC
from Trainer import LightningModel
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl




if __name__=="__main__":

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
    trainer = pl.Trainer(
        logger=logger,
        gpus=[0],
        checkpoint_callback=checkpoints,
        callbacks=[early_stopping],
        default_root_dir="../working/",
        max_epochs=config["epochs"],
        precision=config["precision"],
        automatic_optimization=True
    )


    base = ContextAwareDAC()
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = LightningModel(model=base, tokenizer=tokenizer, config=config)
    
    trainer = pl.Trainer(
        logger=logger,
        gpus=[0],
        checkpoint_callback=checkpoints,
        callbacks=[early_stopping],
        default_root_dir="../working/",
        max_epochs=config["epochs"],
        precision=config["precision"],
        automatic_optimization=True
    )
    
    trainer.fit(model)
    
    trainer.test(model)
    
    
    