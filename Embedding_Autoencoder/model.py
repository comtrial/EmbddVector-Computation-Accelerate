import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
lr = 1e-2


class Autoencoder(pl.LightningModule):
    '''
    ebedding 된 768 차원의 벡터 -> 20 차원으로 
    차원 축소를 해주는 AutoEncoder
    '''
    def __init__(self, nfeature):
        super().__init__()
        self.nfeatures_rna = 768
        self.hidden_size = nfeature

        # Encoder 정의
        self.encoder = nn.Sequential(
            nn.Linear(self.nfeatures_rna, 200),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(200, self.hidden_size), 
            nn.Dropout(0.2)
        )

          # Dncoder 정의
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 200),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(200, self.nfeatures_rna), 
        )

    
    # Traning Step 설정
    def training_step(self, batch, batch_idx):
        x, y = batch

        #encode
        z = self.encoder(x)
        #decode
        recons = self.decoder(z)
        #reconstruction loss 계산
        reconstruction_loss = nn.functional.mse_loss(x, recons)
        
        return reconstruction_loss

    #optimizer 설정
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=lr)

    # Forward Step 설정
    def forward(self, x):
        z = self.encoder(x)
        recons = self.decoder(z)
        embedding = z

        return embedding