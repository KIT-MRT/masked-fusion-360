import torch
import argparse

from vit_pytorch import ViT
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from datetime import datetime

from models.fusion_mae import FusionMAE, FusionEncoder
from data_utils.dataset_modules import KITTI360DataModule, MRTJoyDataModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidar-encoder-patch-size", type=int, required=False, default=8)
    parser.add_argument("--lidar-encoder-img-height", type=int, required=False, default=64)
    parser.add_argument("--lidar-encoder-img-width", type=int, required=False, default=1024)
    parser.add_argument("--lidar-encoder-dim", type=int, required=False, default=2048)
    parser.add_argument("--lidar-encoder-depth", type=int, required=False, default=6)
    parser.add_argument("--lidar-encoder-heads", type=int, required=False, default=8)
    parser.add_argument("--lidar-encoder-mlp-dim", type=int, required=False, default=2048)

    parser.add_argument("--camera-encoder-patch-size", type=int, required=False, default=8)
    parser.add_argument("--camera-encoder-img-height", type=int, required=False, default=64)
    parser.add_argument("--camera-encoder-img-width", type=int, required=False, default=1024)

    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--num-nodes", type=int, required=False, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, required=False, default=4)
    parser.add_argument("--train-hours", type=int, required=False, default=9)
    parser.add_argument("--batch-size", type=int, required=False, default=24)

    parser.add_argument("--dataset-name", type=str, required=False, default="kitti360")
    parser.add_argument("--lr-epochs", type=int, required=False, default=50)
    parser.add_argument("--checkpoint", type=str, required=False, default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # LiDAR encoder
    mae_encoder = ViT(
        image_size=(args.lidar_encoder_img_height, args.lidar_encoder_img_width),
        patch_size=args.lidar_encoder_patch_size,  # Standard 16x16, SegFormer 4x4
        num_classes=1000,
        dim=args.lidar_encoder_dim,
        depth=args.lidar_encoder_depth,
        heads=args.lidar_encoder_heads,
        mlp_dim=args.lidar_encoder_mlp_dim,
    )

    # Camera encoder + fusion block
    fusion_encoder = FusionEncoder(
        image_size=(args.camera_encoder_img_height, args.camera_encoder_img_width),
        patch_size=args.camera_encoder_patch_size,
        vit_dim=args.lidar_encoder_dim,
        vit_mlp_dim=args.lidar_encoder_mlp_dim,
    )

    mae = FusionMAE(
        mae_encoder=mae_encoder,
        fusion_encoder=fusion_encoder,
        masking_ratio=0.5,
        decoder_dim=1024,
        decoder_depth=6,
        epochs=args.lr_epochs,
    )

    if args.checkpoint:
        mae.load_state_dict(torch.load(args.checkpoint))

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        precision=16,
        accelerator="gpu",
        devices=args.num_gpus_per_node,
        num_nodes=args.num_nodes,
        strategy="ddp",
        max_time={"days": 0, "hours": args.train_hours},
        default_root_dir=args.save_dir,
        callbacks=[lr_monitor],
    )

    if args.dataset_name == "kitti360":
        dm = KITTI360DataModule(
            train_path=args.train_path,
            batch_size=args.batch_size,
            num_dataloader_workers=10,
        )
    elif args.dataset_name == "mrt-joy":
        dm = MRTJoyDataModule(
            train_path=args.train_path,
            batch_size=args.batch_size,
            num_dataloader_workers=10,
            img_size=(args.lidar_encoder_img_height, args.lidar_encoder_img_width)
        )

    trainer.fit(mae, datamodule=dm)

    if trainer.is_global_zero:
        save_time = datetime.utcnow().replace(microsecond=0).isoformat()
        torch.save(
            mae_encoder.state_dict(),
            f"{args.save_dir}/models/pre-training/mae-encoder-{args.lidar_encoder_img_height}x{args.lidar_encoder_patch_size}-{save_time}.pt",
        )
        torch.save(
            fusion_encoder.state_dict(),
            f"{args.save_dir}/models/pre-training/fusion-encoder-{args.lidar_encoder_img_height}x{args.lidar_encoder_patch_size}-{save_time}.pt",
        )
        torch.save(
            mae.state_dict(),
            f"{args.save_dir}/models/pre-training/mae-{args.lidar_encoder_img_height}x{args.lidar_encoder_patch_size}-{save_time}.pt",
        )


if __name__ == "__main__":
    main()
