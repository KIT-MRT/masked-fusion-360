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


class SphericalCenterFormer(nn.Module):
    def __init__(
        self,
        lidar_encoder,
        fusion_encoder,
        embedding_dim=1024,
        decoder_channels=[256, 256, 256, 256],
        num_classes=1,
    ) -> None:
        super().__init__()
        self.lidar_encoder = lidar_encoder
        self.cam_encoder = fusion_encoder
        self.embedding_dim = embedding_dim

        decoder_modules = []
        in_channels = [embedding_dim] + decoder_channels[:-1]
        out_channels = decoder_channels

        for idx, (in_ch, out_ch) in enumerate(zip(in_channels, out_channels)):
            decoder_modules.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=1,
                    stride=1,
                    padding=(0, 0),
                )
            )
            if idx != len(decoder_channels):
                decoder_modules.append(nn.Upsample(scale_factor=2, mode="bilinear"))

        self.decoder = nn.Sequential(
            *decoder_modules
        )  # SETR-PUP-like 8x8 instead of 16x16 -> less upsampling necessary
        self.center_heatmap_head = nn.Sequential(
            nn.Conv2d(
                in_channels=decoder_channels[-1],
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.GeLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.center_pos_head = nn.Sequential(
            nn.Conv2d(
                in_channels=decoder_channels[-1],
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.GeLU(),
            nn.Conv2d(
                in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0
            ),
        )
        self.box_dim_head = nn.Sequential(
            nn.Conv2d(
                in_channels=decoder_channels[-1],
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.GeLU(),
            nn.Conv2d(
                in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0
            ),
        )
        self.yaw_head = nn.Sequential(
            nn.Conv2d(
                in_channels=decoder_channels[-1],
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.GeLU(),
            nn.Conv2d(
                in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
        )

    def reshape_encoder_tokens(self, tokens, img_h=64, img_w=1024, patch_dim=8):
        tokens = tokens.view(
            tokens.size(0),
            int(img_h / patch_dim),
            int(img_w / patch_dim),
            self.embedding_dim,
        )
        tokens = tokens.permute(0, 3, 1, 2).contiguous()
        return tokens

    def forward(self, img_stack):
        lidar_img, cam_img = img_stack[:, 0:3, :, :], img_stack[:, 3:, :, :]

        lidar_tokens = self.lidar_encoder(lidar_img)
        fused_tokens = self.fusion_encoder(cam_img, lidar_tokens)

        # Skip mask and cls tokens:
        fused_tokens_raw = torch.index_select(
            fused_tokens, dim=1, index=torch.tensor(range(1, self.embedding_dim + 1))
        )
        decoder_tensor = self.reshape_encoder_tokens(fused_tokens_raw)

        decoded_tensor = self.decoder(decoder_tensor)
        heatmaps = self.center_heatmap_head(decoded_tensor).sigmoid()
        center_xyz = self.center_pos_head(decoded_tensor).relu()
        box_lwh = self.box_dim_head(decoded_tensor).relu()
        yaw_angle = self.yaw_head(decoded_tensor)

        return heatmaps, center_xyz, box_lwh, yaw_angle
