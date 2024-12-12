import torch
import pytorch_lightning as pl
from transformers import MT5TokenizerFast
from src.model import MT5FineTuner
from src.utils import load_data
from src.config import Config
import logging
#from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)

def train_model():
    logging.info("Loading tokenizer...")
    tokenizer = MT5TokenizerFast.from_pretrained('google/mt5-small')

    logging.info("Loading data...")
    train_dataloader, val_dataloader, test_dataloader = load_data(
        data_dir=Config.DATA_DIR, 
        tokenizer=tokenizer, 
        batch_size=Config.BATCH_SIZE
    )

    logging.info("Initializing the model...")
    model = MT5FineTuner(
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=Config.LEARNING_RATE,
        target_max_length=Config.TARGET_MAX_LENGTH,
        mode=Config.MODE
    )

    logging.info("Configuring the PyTorch Lightning trainer...")
    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed',  # Use mixed precision for faster training
        #gradient_clip_val=1.0, # helps prevent NaNs
        accumulate_grad_batches=Config.ACCUMULATE_GRAD_BATCHES,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="output/checkpoints",
                filename="t5_finetuner-{epoch:02d}-{val_loss:.2f}",
                save_top_k=2,
                monitor="val_loss",
                mode="min"
            ),
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=Config.PATIENCE, mode="min")
        ],
        logger=pl.loggers.TensorBoardLogger("output/logs", name="MT5_FineTuning"),
        enable_progress_bar=True,  
        log_every_n_steps=10
    )

    
    logging.info("Starting training...")
    trainer.fit(model)
    
    
    logging.info("Training completed.")
    return model


if __name__ == "__main__":
    train_model()
