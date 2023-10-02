import os
import cv2
import typer
import rospy
import torch
import signal
import rosgraph
import numpy as np
import message_filters

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vit_pytorch import ViT

from data_utils.naive_img_stitching import stitch_boxring_imgs
from ros_utils.ros_opencv_utils import (
    cv2_to_imgmsg,
    f32c1_imgmsg_to_nparray,
    f32_opencv_img_to_uint8,
)
from data_utils.preprocessing import preprocess_sample
from data_utils.visualize import generate_reconstructed_img
from models.fusion_mae import FusionMAE, FusionEncoder


def main(log_level: int = rospy.ERROR):
    def img_callback(*data):
        rospy.logdebug("[MaskedFusion360] Images received")
        back_img = opencv_bridge.imgmsg_to_cv2(data[0])
        back_left_img = opencv_bridge.imgmsg_to_cv2(data[1])
        back_right_img = opencv_bridge.imgmsg_to_cv2(data[2])
        front_img = opencv_bridge.imgmsg_to_cv2(data[3])
        front_left_img = opencv_bridge.imgmsg_to_cv2(data[4])
        front_right_img = opencv_bridge.imgmsg_to_cv2(data[5])
        intensity_img = f32c1_imgmsg_to_nparray(img_msg=data[6])
        range_img = f32c1_imgmsg_to_nparray(img_msg=data[7])

        # Naive stitching and pre-processing
        try:
            stitched_img = stitch_boxring_imgs(
                back_img,
                back_left_img,
                back_right_img,
                front_img,
                front_left_img,
                front_right_img,
            )
        except:
            rospy.logerr_throttle(
                period=60,
                msg="[MaskedFusion360] Error during stitching",
            )

        stitched_img_ros = cv2_to_imgmsg(stitched_img)
        image_pub.publish(stitched_img_ros)

        fusion_mae_input = preprocess_sample(stitched_img, intensity_img, range_img)

        range_pub.publish(cv2_to_imgmsg(f32_opencv_img_to_uint8(range_img)))
        intensity_pub.publish(cv2_to_imgmsg(f32_opencv_img_to_uint8(intensity_img)))

        # MaskedFusion360 inference
        try:
            with torch.no_grad():
                *_, masked_indices, preds, recon_img = fusion_mae(
                    fusion_mae_input.to(fusion_mae.device)
                )
        except:
            rospy.logerr_throttle(
                period=60,
                msg="[MaskedFusion360] Error during inference",
            )

        recon_img = torch.clamp(recon_img, min=0.0, max=1.0)
        recon_img_np = recon_img[0].to("cpu").numpy()
        recon_img_np = np.moveaxis(recon_img_np, 0, -1)
        recon_img_np = (recon_img_np * 255).astype(np.uint8)

        # Create masked lidar img
        lidar_input = np.copy(recon_img_np)
        masked_lidar_stack = generate_reconstructed_img(
            base_img=lidar_input,
            patch_indices=masked_indices[0].to("cpu").numpy(),
            reconstructed_patches=np.ones((512, 192), dtype=np.uint8) + 254,
            img_width=1024,
            patch_size=8,
        )

        debug_img = np.vstack((stitched_img, masked_lidar_stack, recon_img_np))
        debug_img_ros = cv2_to_imgmsg(debug_img)

        recon_pub.publish(debug_img_ros)

    def shutdown_callback(event):
        if not rosgraph.is_master_online():
            rospy.logerr_once(
                "[MaskedFusion360] Shutdown, because the ROS master is not reachable"
            )
            os.kill(1, signal.SIGTERM)

    rospy.init_node("masked_fusion_360", log_level=log_level)

    back_img_sub = message_filters.Subscriber(
        "/sensor/camera/box_ring/back/atl071s_cc/raw/image", Image
    )
    back_left_img_sub = message_filters.Subscriber(
        "/sensor/camera/box_ring/back_left/atl071s_cc/raw/image", Image
    )
    back_right_img_sub = message_filters.Subscriber(
        "/sensor/camera/box_ring/back_right/atl071s_cc/raw/image", Image
    )
    front_img_sub = message_filters.Subscriber(
        "/sensor/camera/box_ring/front/atl071s_cc/raw/image", Image
    )
    front_left_img_sub = message_filters.Subscriber(
        "/sensor/camera/box_ring/front_left/atl071s_cc/raw/image", Image
    )
    front_right_img_sub = message_filters.Subscriber(
        "/sensor/camera/box_ring/front_right/atl071s_cc/raw/image", Image
    )
    intensity_img_sub = message_filters.Subscriber(
        "/sensor/lidar/box_top/center/vls128_ap/intensity_image", Image
    )
    range_img_sub = message_filters.Subscriber(
        "/sensor/lidar/box_top/center/vls128_ap/range_image", Image
    )

    opencv_bridge = CvBridge()
    image_pub = rospy.Publisher(
        name="/perception/masked_fusion_360/stitched_camera_image",
        data_class=Image,
        queue_size=10,
    )
    range_pub = rospy.Publisher(
        name="/perception/masked_fusion_360/lidar_range_decoded",
        data_class=Image,
        queue_size=10,
    )
    intensity_pub = rospy.Publisher(
        name="/perception/masked_fusion_360/lidar_intensity_decoded",
        data_class=Image,
        queue_size=10,
    )
    recon_pub = rospy.Publisher(
        name="/perception/masked_fusion_360/reconstructed_lidar_image",
        data_class=Image,
        queue_size=10,
    )

    # LiDAR encoder
    mae_encoder = ViT(
        image_size=(64, 1024),
        patch_size=8,
        num_classes=1000,
        dim=2048,
        depth=6,
        heads=8,
        mlp_dim=2048,
    )

    # Camera encoder + fusion block
    fusion_encoder = FusionEncoder(
        image_size=(64, 1024),
        patch_size=8,
        vit_dim=2048,
        vit_mlp_dim=2048,
    )

    fusion_mae = FusionMAE(
        mae_encoder=mae_encoder,
        fusion_encoder=fusion_encoder,
        masking_ratio=0.5,
        decoder_dim=1024,
        decoder_depth=6,
    )

    fusion_mae.load_state_dict(torch.load("weights/mae-64x8-2023-09-18T06:07:44.pt"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fusion_mae.to(device)

    # http://wiki.ros.org/message_filters#Example_.28Python.29-1
    ts = message_filters.ApproximateTimeSynchronizer(
        fs=[
            back_img_sub,
            back_left_img_sub,
            back_right_img_sub,
            front_img_sub,
            front_left_img_sub,
            front_right_img_sub,
            intensity_img_sub,
            range_img_sub,
        ],
        queue_size=10,
        slop=0.2,  # in secs
    )
    ts.registerCallback(img_callback)
    rospy.Timer(rospy.Duration(secs=10), callback=shutdown_callback)

    rospy.spin()


if __name__ == "__main__":
    try:
        typer.run(main)
    except rospy.ROSInterruptException:
        rospy.logerr_once("[MaskedFusion360] Exiting due to ROSInterruptException")
