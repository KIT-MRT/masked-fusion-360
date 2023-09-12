import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn
from einops import repeat, rearrange

from vit_pytorch.vit import ViT, Transformer
from vit_pytorch.cross_vit import CrossTransformer
from torchmetrics.functional import total_variation
from torchmetrics.functional import multiscale_structural_similarity_index_measure


class FusionMAE(pl.LightningModule):
    def __init__(
        self,
        *,
        mae_encoder,
        fusion_encoder,
        decoder_dim,
        masking_ratio=0.75,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64,
        mae_patch_size=8,
        mae_img_size=(64, 1024), # h x w
        lr=1e-4,
        epochs=50,
    ):
        super().__init__()
        assert (
            masking_ratio > 0 and masking_ratio < 1
        ), "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from mae_encoder (vision transformer to be trained)
        self.mae_encoder = mae_encoder
        num_patches, encoder_dim = mae_encoder.pos_embedding.shape[-2:]

        self.to_patch = mae_encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*mae_encoder.to_patch_embedding[1:])

        pixel_values_per_patch = mae_encoder.to_patch_embedding[2].weight.shape[-1]

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
        self.fusion_encoder = fusion_encoder
        self.epochs = epochs
        self.mae_patch_size = mae_patch_size
        self.mae_img_size = mae_img_size
        # self.save_hyperparameters() # Throws SIGSEGV on Juwels?

    def _get_tokens_preds_loss(self, img_stack):
        img, cam_img = img_stack[:, 0:3, :, :], img_stack[:, 3:, :, :]
        device = img.device

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to mae_encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.mae_encoder.pos_embedding[:, 1 : (num_patches + 1)]

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
        encoded_tokens = self.mae_encoder.transformer(tokens)

        # project mae_encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
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

        # New cam feats without masking and fusion
        decoder_tokens = self.fusion_encoder(
            cam_img=cam_img, decoder_tokens_lidar=decoder_tokens
        )

        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        return patches, masked_patches, masked_indices, pred_pixel_values, batch_range, recon_loss

    def training_step(self, batch, batch_idx):
        _, _, _, _, _, loss = self._get_tokens_preds_loss(batch)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, masked_patches, _, pred_pixel_values, _, loss = self._get_tokens_preds_loss(
            batch
        )
        ssim_full_input = multiscale_structural_similarity_index_measure(
            masked_patches[None, :], pred_pixel_values[None, :]
        )
        batch_no_cam = torch.zeros(batch.shape, device=torch.device("cuda"))
        batch_no_cam[:, 0:3, :, :] = batch[:, 0:3, :, :]
        _, _, _, pred_pixel_values_no_cam, _, _ = self._get_tokens_preds_loss(batch_no_cam)
        ssim_no_cam = multiscale_structural_similarity_index_measure(
            masked_patches[None, :], pred_pixel_values_no_cam[None, :]
        )
        delta_total_variation = total_variation(
            pred_pixel_values_no_cam[None, :]
        ) - total_variation(pred_pixel_values[None, :])

        self.log("val_loss", loss, sync_dist=True)
        self.log("ssim_full_input", ssim_full_input, sync_dist=True)
        self.log("ssim_no_cam", ssim_no_cam, sync_dist=True)
        self.log("delta_total_variation", delta_total_variation, sync_dist=True)

        return loss

    def forward(self, batch):
        patches, masked_patches, masked_indices, preds, batch_range, _ = self._get_tokens_preds_loss(
            batch
        )
        # Rearrange on GPU
        patches[batch_range, masked_indices] = preds.type(torch.float32)
        recon_img = rearrange(
            patches, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
            p1=self.mae_patch_size, 
            p2=self.mae_patch_size, 
            h=self.mae_img_size[0]//self.mae_patch_size, 
            w=self.mae_img_size[1]//self.mae_patch_size,
        )
        
        return patches, masked_patches, masked_indices, preds, recon_img

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.epochs,
                    eta_min=1e-6,
                ),
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            },
        }


class FusionEncoder(nn.Module):
    def __init__(
        self,
        image_size=(64, 1024),
        patch_size=8,  # Standard 16x16, SegFormer 4x4
        vit_num_classes=1000,
        vit_dim=2048,
        vit_depth=6,
        vit_heads=8,
        vit_mlp_dim=2048,
        cross_vit_depth=2,
        cross_vit_heads=8,
        cross_vit_dim_head=128,
        cross_vit_dropout=0.0,
        decoder_dim=1024,
    ):
        super().__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=vit_num_classes,
            dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
        )
        self.enc_to_dec = (
            nn.Linear(vit_dim, decoder_dim) if vit_dim != decoder_dim else nn.Identity()
        )
        self.cross_fusion = CrossTransformer(
            sm_dim=decoder_dim,
            lg_dim=decoder_dim,
            depth=cross_vit_depth,
            heads=cross_vit_heads,
            dim_head=cross_vit_dim_head,
            dropout=cross_vit_dropout,
        )

    def forward(self, cam_img, decoder_tokens_lidar):
        cam_x = self.vit.to_patch_embedding(cam_img)
        cam_b, cam_n, _ = cam_x.shape
        cam_cls_tokens = repeat(self.vit.cls_token, "1 1 d -> b 1 d", b=cam_b)
        cam_x = torch.cat((cam_cls_tokens, cam_x), dim=1)
        cam_x += self.vit.pos_embedding[:, : (cam_n + 1)]
        cam_x = self.vit.dropout(cam_x)
        cam_tokens = self.vit.transformer(cam_x)

        cam_tokens = self.enc_to_dec(cam_tokens)

        # Cross-attn to fuse tokens
        decoder_tokens, _ = self.cross_fusion(
            sm_tokens=decoder_tokens_lidar, lg_tokens=cam_tokens
        )

        return decoder_tokens
