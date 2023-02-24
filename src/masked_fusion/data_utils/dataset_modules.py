import cv2
import glob
import numpy as np
import pytorch_lightning as pl

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

from .process_point_clouds import read_kitti_point_cloud, spherical_projection, add_range


class KITTI360RangeFishEye(Dataset):
    def __init__(self):
        self.imgs_path = "/p/project/hai_mrt_pc/KITTI-360"
        img2_paths = sorted(
            glob.glob(self.imgs_path + "/data_2d_raw/*/image_02/data_rgb/*.png")
        )
        img3_paths = sorted(
            glob.glob(self.imgs_path + "/data_2d_raw/*/image_03/data_rgb/*.png")
        )
        lidar_paths = sorted(
            glob.glob(self.imgs_path + "/data_3d_raw/*/velodyne_points/data/*.bin")
        )

        self.data = []

        for img2_path, img3_path, lidar_path in zip(
            img2_paths, img3_paths, lidar_paths
        ):
            self.data.append([img2_path, img3_path, lidar_path])

        self.img_dim = (1024, 64)  # w x h since cv2
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img2_path, img3_path, lidar_path = self.data[idx]
        img2 = cv2.imread(img2_path)
        img3 = cv2.imread(img3_path)

        # left and right img
        img = np.hstack((img2[670:977], img3[670:977]))
        img = cv2.resize(img, self.img_dim)

        lidar_points = read_kitti_point_cloud(lidar_path)
        _, _, lidar_img = spherical_projection(add_range(lidar_points))
        lidar_img = lidar_img[..., 2:5].astype(np.float32)

        tensor_stack = np.dstack((lidar_img, img))
        tensor_stack = self.transform(tensor_stack)

        return tensor_stack


class KITTI360DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_dataloader_workers=8, pin_memory=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            kitti_full = KITTI360RangeFishEye()
            self.kitti_train, self.kitti_val = random_split(
                kitti_full, [70000, 6251]
            )  # check how many samples and split 90:10

        if stage == "predict":
            self.kitti_predict = KITTI360RangeFishEye()

    def train_dataloader(self):
        return DataLoader(
            self.kitti_train,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.kitti_val,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.kitti_predict,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
