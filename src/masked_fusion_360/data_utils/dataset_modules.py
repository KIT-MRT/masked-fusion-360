import cv2
import glob
import torch
import numpy as np
import pytorch_lightning as pl

from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

from .process_point_clouds import (
    read_kitti_point_cloud,
    spherical_projection,
    add_range,
)


class KITTI360RangeFishEye(Dataset):
    def __init__(self, imgs_path="/p/project/hai_mrt_pc/KITTI-360"):
        img2_paths = sorted(
            glob.glob(imgs_path + "/data_2d_raw/*/image_02/data_rgb/*.png")
        )
        img3_paths = sorted(
            glob.glob(imgs_path + "/data_2d_raw/*/image_03/data_rgb/*.png")
        )
        lidar_paths = sorted(
            glob.glob(imgs_path + "/data_3d_raw/*/velodyne_points/data/*.bin")
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

        # Concat left and right img
        img = np.hstack((img2[670:977], img3[670:977]))
        img = cv2.resize(img, self.img_dim)
        img_tensor = self.transform(img)

        lidar_points = read_kitti_point_cloud(lidar_path)
        _, _, lidar_img = spherical_projection(add_range(lidar_points))

        lidar_img = lidar_img.astype(np.float32)
        height_view, intensity_view, range_view = (
            lidar_img[..., 2],
            lidar_img[..., 3],
            lidar_img[..., 4],
        )

        # To remove points way below ground level (~false detections)
        height_view[height_view < -3.0] = 0.0

        # LiDAR sensor is mounted at 1.73m -> set areas where no points are to ground level
        height_view[height_view == 0.0] = -1.73

        height_view, intensity_view, range_view = (
            min_max_scaling(height_view),
            min_max_scaling(intensity_view),
            min_max_scaling(range_view),
        )

        lidar_img_tensor = np.dstack((height_view, intensity_view, range_view))
        lidar_img_tensor = self.transform(lidar_img_tensor)

        tensor_stack = torch.cat((lidar_img_tensor, img_tensor), 0)

        return tensor_stack


class KITTI360DataModule(pl.LightningDataModule):
    def __init__(
        self, train_path, batch_size=32, num_dataloader_workers=8, pin_memory=True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.pin_memory = pin_memory
        self.train_path = train_path

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            kitti_full = KITTI360RangeFishEye(self.train_path)
            self.kitti_train, self.kitti_val = random_split(
                kitti_full, [70000, 6251]
            )  # check how many samples and split 90:10

        if stage == "predict":
            self.kitti_predict = KITTI360RangeFishEye(self.train_path)

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


def min_max_scaling(x: np.ndarray) -> np.ndarray:
    x_min, x_max = x.min(), x.max()
    x -= x_min
    x /= (x_max - x_min) + np.finfo(np.float32).eps

    return x


class MRTJoyDataset(Dataset):
    def __init__(self, imgs_path, img_size=(128, 2048)):

        back_img_paths = sorted(glob.glob(imgs_path + "/camera_back/*.jpg"))
        back_left_img_paths = sorted(glob.glob(imgs_path + "/camera_back_left/*.jpg"))
        back_right_img_paths = sorted(glob.glob(imgs_path + "/camera_back_right/*.jpg"))

        front_img_paths = sorted(glob.glob(imgs_path + "/camera_front/*.jpg"))
        front_left_img_paths = sorted(glob.glob(imgs_path + "/camera_front_left/*.jpg"))
        front_right_img_paths = sorted(
            glob.glob(imgs_path + "/camera_front_right/*.jpg")
        )

        lidar_intensity_paths = sorted(
            # glob.glob(imgs_path + "/lidar_intensity_image/*.png")
            glob.glob(imgs_path + "/lidar_intensity_image/*.npz")
        )
        # lidar_range_paths = sorted(glob.glob(imgs_path + "/lidar_range_image/*.png"))
        lidar_range_paths = sorted(glob.glob(imgs_path + "/lidar_range_image/*.npz"))

        self.data = []

        for (
            back_img_path,
            back_left_img_path,
            back_right_img_path,
            front_img_path,
            front_left_img_path,
            front_right_img_path,
            lidar_intensity_path,
            lidar_range_path,
        ) in zip(
            back_img_paths,
            back_left_img_paths,
            back_right_img_paths,
            front_img_paths,
            front_left_img_paths,
            front_right_img_paths,
            lidar_intensity_paths,
            lidar_range_paths,
        ):
            self.data.append(
                [
                    back_img_path,
                    back_left_img_path,
                    back_right_img_path,
                    front_img_path,
                    front_left_img_path,
                    front_right_img_path,
                    lidar_intensity_path,
                    lidar_range_path,
                ]
            )

        self.img_dim = (img_size[1], img_size[0])  # w x h since cv2

        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ColorJitter(
                    brightness=(0.7, 1.3),
                    contrast=(0.8, 1.2),
                    sturation=(0.9, 1.1),
                    hue=(0.9, 1.1),
                ),
                # transforms.RandomHorizontalFlip(),
            ]
        )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (
            back_img_path,
            back_left_img_path,
            back_right_img_path,
            front_img_path,
            front_left_img_path,
            front_right_img_path,
            lidar_intensity_img_path,
            lidar_range_img_path,
        ) = self.data[idx]
        back_img = cv2.imread(back_img_path)
        back_left_img = cv2.imread(back_left_img_path)
        back_right_img = cv2.imread(back_right_img_path)
        front_img = cv2.imread(front_img_path)
        front_left_img = cv2.imread(front_left_img_path)
        front_right_img = cv2.imread(front_right_img_path)

        # Crop and concat images:
        crop_width = int(3072 * 0.18)
        img_stack = np.hstack(
            (
                back_img[800:1350, 1536:-crop_width],
                back_left_img[800:1350, crop_width:-crop_width],
                front_left_img[800:1350, crop_width:-crop_width],
                front_img[800:1350, crop_width:-crop_width],
                front_right_img[800:1350, crop_width:-crop_width],
                back_right_img[800:1350, crop_width:-crop_width],
                back_img[800:1350, crop_width:1536],
            )
        )
        img_stack = cv2.resize(img_stack, self.img_dim)

        # Min-max on image
        img_stack = min_max_scaling(img_stack.astype(np.float32))

        # Min-max on LiDAR projs
        lidar_intensity_img = min_max_scaling(
            (
                # cv2.resize(cv2.imread(lidar_intensity_img_path), self.img_dim)[..., 0]
                cv2.resize(np.load(lidar_intensity_img_path), self.img_dim)
            ).astype(np.float32)
        )
        lidar_range_img = min_max_scaling(
            # (cv2.resize(cv2.imread(lidar_range_img_path), self.img_dim)[..., 0]).astype(
            (cv2.resize(np.load(lidar_range_img_path), self.img_dim)).astype(
                np.float32
            )
        )

        lidar_img_stack = np.dstack(
            (lidar_intensity_img, lidar_range_img, lidar_intensity_img)
        )  # Opt. change

        img_tensor = self.img_transform(img_stack)
        lidar_tensor = self.transform(lidar_img_stack)

        tensor_stack = torch.cat((lidar_tensor, img_tensor), 0)

        return tensor_stack


class MRTJoyDataModule(pl.LightningDataModule):
    def __init__(
        self, train_path, batch_size=32, num_dataloader_workers=8, pin_memory=True,
        img_size=(128, 2048)
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.pin_memory = pin_memory
        self.train_path = train_path
        self.img_size = img_size

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        num_samples = len(glob.glob(self.train_path + "/camera_front/*.jpg"))
        num_val_samples = int(0.2 * num_samples)

        if stage == "fit":
            dataset_full = MRTJoyDataset(self.train_path, img_size=self.img_size)
            self.train_split = Subset(dataset_full, torch.arange(0, num_samples - num_val_samples, 1))
            self.val_split = Subset(dataset_full, torch.arange(num_samples - num_val_samples, num_samples, 1))

        if stage == "predict":
            self.predict_split = MRTJoyDataset(self.train_path, img_size=self.img_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_split,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_split,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
