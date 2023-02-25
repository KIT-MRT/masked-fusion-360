# %cd /p/home/jusers/wagner20/juwels/masked-fusion/src/masked_fusion
import torch

from vit_pytorch import ViT
from pytorch_lightning import Trainer
from datetime import datetime

from models.fusion_mae import FusionMAE, FusionEncoder
from data_utils.dataset_modules import KITTI360DataModule


def main():
    mae_encoder = ViT(
        image_size=(64, 1024),
        patch_size=8,  # Try 8, standard 16x16, SegFormer 4x4
        num_classes=1000,
        dim=2048,  # 1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
    )

    fusion_encoder = FusionEncoder()

    mae = FusionMAE(
        mae_encoder=mae_encoder,
        fusion_encoder=fusion_encoder,
        masking_ratio=0.5,  # the paper recommended 75% masked patches
        decoder_dim=1024,  # paper showed good results with just 512
        decoder_depth=6,  # anywhere from 1 to 8
    )

    trainer = Trainer(
        precision=16,
        accelerator="gpu",
        devices=4,
        num_nodes=1,
        max_time={"days": 0, "hours": 9},
        default_root_dir="/p/project/hai_mrt_pc/",
    )

    dm = KITTI360DataModule(
        batch_size=24,
        num_dataloader_workers=10,
    )

    trainer.fit(mae, datamodule=dm)

    save_time = datetime.utcnow().replace(microsecond=0).isoformat()
    torch.save(
        mae_encoder.state_dict(),
        f"/p/project/hai_mrt_pc/models/pre-training/mae-encoder-{save_time}.pt",
    )
    torch.save(
        fusion_encoder.state_dict(),
        f"/p/project/hai_mrt_pc/models/pre-training/fusion-encoder-{save_time}.pt",
    )
    torch.save(
        mae.state_dict(),
        f"/p/project/hai_mrt_pc/models/pre-training/mae-{save_time}.pt",
    )


if __name__ == "__main__":
    main()
