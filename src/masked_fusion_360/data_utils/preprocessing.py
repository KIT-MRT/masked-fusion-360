import cv2
import torch
import numpy as np

from torchvision import transforms


def min_max_scaling(x: np.ndarray) -> np.ndarray:
    x_min, x_max = x.min(), x.max()
    x -= x_min
    x /= (x_max - x_min) + np.finfo(np.float32).eps

    return x


def preprocess_sample(stitched_cam_img, lidar_intensity_img, lidar_range_img, img_dim=(1024, 64)):
    # Min-max on image
    img_stack = min_max_scaling(stitched_cam_img.astype(np.float32))

    # Min-max on LiDAR projs
    lidar_intensity_img = min_max_scaling(
        (
            cv2.resize(lidar_intensity_img, img_dim)
        ).astype(np.float32)
    )
    lidar_range_img = min_max_scaling(
        (cv2.resize(lidar_range_img, img_dim)).astype(
            np.float32
        )
    )

    lidar_img_stack = np.dstack(
        (lidar_intensity_img, lidar_range_img, lidar_intensity_img)
    )  # Opt. change

    img_tensor = transforms.functional.to_tensor(img_stack)
    lidar_tensor = transforms.functional.to_tensor(lidar_img_stack)

    tensor_stack = torch.cat((lidar_tensor, img_tensor), 0)

    return tensor_stack[None, ...]