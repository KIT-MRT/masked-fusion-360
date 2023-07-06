import cv2
import torch
import numpy as np

from torchvision import transforms
from dataset_modules import min_max_scaling


def get_patch_pixelrange(indice, img_width=1024, patch_size=8):
    offset = img_width // patch_size
    row = int(indice / offset)
    col = indice - row * offset

    y0 = row * patch_size
    y1 = y0 + patch_size
    x0 = col * patch_size
    x1 = x0 + patch_size

    return y0, y1, x0, x1


def generate_reconstructed_img(
    base_img, patch_indices, reconstructed_patches, img_width, patch_size
):
    for patch_idx, patch in zip(patch_indices, reconstructed_patches):
        y0, y1, x0, x1 = get_patch_pixelrange(patch_idx, img_width, patch_size)
        patch = patch.reshape((patch_size, patch_size, 3))
        base_img[y0:y1, x0:x1, :] = patch

    return base_img


def preprocess_data(
    back_img_path,
    back_left_img_path,
    back_right_img_path,
    front_img_path,
    front_left_img_path,
    front_right_img_path,
    lidar_intensity_img_path,
    lidar_range_img_path,
    crop_width_portion=0.18,
    image_size=(128, 2048),
    lidar_image_size=(128, 2048),
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    back_img = cv2.imread(back_img_path)
    back_left_img = cv2.imread(back_left_img_path)
    back_right_img = cv2.imread(back_right_img_path)
    front_img = cv2.imread(front_img_path)
    front_left_img = cv2.imread(front_left_img_path)
    front_right_img = cv2.imread(front_right_img_path)

    # Crop and concat images:
    crop_width = int(3072 * crop_width_portion)
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
    img_stack = cv2.resize(img_stack, (image_size[1], image_size[0]))

    # Min-max on image
    img_stack = min_max_scaling(img_stack.astype(np.float32))

    # Min-max on LiDAR projs
    lidar_intensity_img = min_max_scaling(
        (
            cv2.resize(
                cv2.imread(lidar_intensity_img_path),
                (lidar_image_size[1], lidar_image_size[0]),
            )[..., 0]
        ).astype(np.float32)
    )
    lidar_range_img = min_max_scaling(
        (
            cv2.resize(
                cv2.imread(lidar_range_img_path),
                (lidar_image_size[1], lidar_image_size[0]),
            )[..., 0]
        ).astype(np.float32)
    )

    lidar_img_stack = np.dstack(
        (lidar_intensity_img, lidar_range_img, lidar_intensity_img)
    )  # Opt. change

    img_tensor = transform(img_stack)
    lidar_tensor = transform(lidar_img_stack)

    tensor_stack = torch.cat((lidar_tensor, img_tensor), 0)

    return tensor_stack
