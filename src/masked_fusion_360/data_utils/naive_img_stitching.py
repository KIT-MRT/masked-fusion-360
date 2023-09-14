import cv2
import numpy as np


def stitch_boxring_imgs(
    back_img,
    back_left_img,
    back_right_img,
    front_img,
    front_left_img,
    front_right_img,
    crop_width: int = int(3072 * 0.18),
    stitched_img_dim: tuple = (1024, 64), # w x h for cv2
):
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
    img_stack = cv2.resize(img_stack, stitched_img_dim)

    return img_stack