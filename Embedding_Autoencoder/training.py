
from model import Autoencoder
from data_set import EmbeddDataset
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import callbacks


def train():
    
    # DataLoader 정의
    EMBD_PICK_DIR = './data/embedding.pkl'
    dataset = EmbeddDataset(EMBD_PICK_DIR)
    data_loader = DataLoader(dataset, batch_size=32, shuffle = False)

    # Model 정의
    model_dir = "./saved_model"
    checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss",
            mode="min",
            dirpath=model_dir,
            filename="embedding_Autoencoder",
        )
    
    autoencoder = Autoencoder()
    trainer = pl.Trainer(
        #gpus = 1,
        max_epochs=3,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(autoencoder, data_loader)
    
    return

