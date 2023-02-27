import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn
from einops import repeat

from vit_pytorch.vit import ViT, Transformer


class LidarEncoder(nn.Module):
    def __init__(self, vit, encoder_dim, decoder_dim) -> None:
        super().__init__()
        self.vit = vit
        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )

    def forward(self, lidar_view):
        lidar_x = self.vit.to_patch_embedding(lidar_view)
        lidar_b, lidar_n, _ = lidar_x.shape
        lidar_cls_tokens = repeat(self.vit.cls_token, "1 1 d -> b 1 d", b=lidar_b)
        lidar_x = torch.cat((lidar_cls_tokens, lidar_x), dim=1)
        lidar_x += self.vit.pos_embedding[:, : (lidar_n + 1)]
        lidar_x = self.vit.dropout(lidar_x)
        lidar_tokens = self.vit.transformer(lidar_x)
        lidar_tokens = self.enc_to_dec(lidar_tokens)

        return lidar_tokens