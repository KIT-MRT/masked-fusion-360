import rospy
import message_filters

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from data_utils.naive_img_stitching import stitch_boxring_imgs
from ros_utils.ros_opencv_utils import imgmsg_to_cv2, cv2_to_imgmsg


def main():
    def callback(*data):
        print("imgs received")
        
        # TODO: cvbridge -- fast enough single threaded?
        back_img = opencv_bridge.imgmsg_to_cv2(data[0])
        back_left_img = opencv_bridge.imgmsg_to_cv2(data[1])
        back_right_img = opencv_bridge.imgmsg_to_cv2(data[2])
        front_img = opencv_bridge.imgmsg_to_cv2(data[3])
        front_left_img = opencv_bridge.imgmsg_to_cv2(data[4])
        front_right_img = opencv_bridge.imgmsg_to_cv2(data[5])

        # TODO: naive stitching and pre-processing
        stitched_img = stitch_boxring_imgs(back_img, back_left_img, back_right_img, front_img, front_left_img, front_right_img)
        print(stitched_img.shape)
        stitched_img_ros = cv2_to_imgmsg(stitched_img)
        image_pub.publish(stitched_img_ros)

        # TODO: MaskedFusion360 inference

        # TODO: publish debug img
    
    rospy.init_node("boxring_imgs_sub", anonymous=True)

    back_img_sub = message_filters.Subscriber(
        "/sensor/camera/box_ring/back/atl071s_cc/raw/image",
        Image
    )
    back_left_img_sub = message_filters.Subscriber(
        "/sensor/camera/box_ring/back_left/atl071s_cc/raw/image",
        Image
    )
    back_right_img_sub = message_filters.Subscriber(
        "/sensor/camera/box_ring/back_right/atl071s_cc/raw/image",
        Image
    )
    front_img_sub = message_filters.Subscriber(
        "/sensor/camera/box_ring/front/atl071s_cc/raw/image",
        Image
    )
    front_left_img_sub = message_filters.Subscriber(
        "/sensor/camera/box_ring/front_left/atl071s_cc/raw/image",
        Image
    )
    front_right_img_sub = message_filters.Subscriber(
        "/sensor/camera/box_ring/front_right/atl071s_cc/raw/image",
        Image
    )
    intensity_img_sub = message_filters.Subscriber(
        "/sensor/lidar/box_top/center/vls128_ap/intensity_image",
        Image
    )
    range_img_sub = message_filters.Subscriber(
        "/sensor/lidar/box_top/center/vls128_ap/range_image",
        Image
    )

    opencv_bridge = CvBridge()
    image_pub = rospy.Publisher("/output/stitched_image", Image)

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
        slop=0.2, # in secs
    )
    ts.registerCallback(callback)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass