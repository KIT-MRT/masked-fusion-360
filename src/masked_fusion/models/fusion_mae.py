import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn
from einops import repeat

from vit_pytorch.vit import Transformer
from vit_pytorch.cross_vit import CrossTransformer


class FusionMAE(pl.LightningModule):
    def __init__(
        self,
        *,
        encoder,
        encoder2,
        decoder_dim,
        masking_ratio=0.75,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64,
        lr=1e-4,
    ):
        super().__init__()
        assert (
            masking_ratio > 0 and masking_ratio < 1
        ), "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4,
        )
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

        self.lr = lr
        self.encoder2 = encoder2
        self.cross_attn = CrossTransformer(
            sm_dim=decoder_dim,
            lg_dim=decoder_dim,
            depth=2,
            heads=8,
            dim_head=64,
            dropout=0.0,
        )

    def _get_tokens_preds_loss(self, img_stack):
        img, cam_img = img_stack[:, 0:3, :, :], img_stack[:, 3:, :, :]
        device = img.device

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1 : (num_patches + 1)]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = (
            rand_indices[:, :num_masked],
            rand_indices[:, num_masked:],
        )

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(
            unmasked_indices
        )

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.zeros(
            batch, num_patches, self.decoder_dim, device=device
        )
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens

        # New cam feats without masking
        # cam_tokens = self.encoder2(cam_img) # we don't want the mlp head
        # https://github.com/lucidrains/vit-pytorch/blob/5699ed7d139062020d1394f0e85a07f706c87c09/vit_pytorch/vit.py#L115
        cam_x = self.encoder2.to_patch_embedding(cam_img)
        cam_b, cam_n, _ = cam_x.shape
        cam_cls_tokens = repeat(self.encoder2.cls_token, "1 1 d -> b 1 d", b=cam_b)
        cam_x = torch.cat((cam_cls_tokens, cam_x), dim=1)
        cam_x += self.encoder2.pos_embedding[:, : (cam_n + 1)]
        cam_x = self.encoder2.dropout(cam_x)
        cam_tokens = self.encoder2.transformer(cam_x)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        cam_tokens = self.enc_to_dec(cam_tokens)

        # Cross-attn to fuse tokens
        decoder_tokens, _ = self.cross_attn(
            sm_tokens=decoder_tokens, lg_tokens=cam_tokens
        )

        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        return tokens, decoder_tokens, pred_pixel_values, recon_loss

    def training_step(self, batch, batch_idx):
        _, _, _, loss = self._get_tokens_preds_loss(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, _, loss = self._get_tokens_preds_loss(batch)
        return loss

    def forward(self, batch):
        tokens, decoder_tokens, preds, _ = self._get_tokens_preds_loss(batch)
        return tokens, decoder_tokens, preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer