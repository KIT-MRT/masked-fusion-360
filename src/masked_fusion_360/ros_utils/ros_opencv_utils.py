# Src: https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/
import sys
import rospy
import numpy as np

from sensor_msgs.msg import Image


def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv


def imgmsg_to_cv2_f32(img_msg):
    print("encoding: ", img_msg.encoding)
    print("height: ", img_msg.height)
    print("width: ", img_msg.width)
    dtype = np.dtype("float32")
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(
        shape=(img_msg.height, img_msg.width, 1),
        dtype=dtype,
        buffer=img_msg.data,
    )
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    
    return image_opencv

def f32c1_immsg_to_nparray(img_msg):
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height * img_msg.width * 4), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    
    image_opencv = image_opencv.view(np.float32)

    image_opencv = np.nan_to_num(image_opencv, nan=0.0)

    image_opencv = np.ndarray(
        shape=(img_msg.height, img_msg.width),
        dtype=np.float32,
        buffer=image_opencv,
    )

    return image_opencv



def f32c1_imgmsg_to_opencv(img_msg, bridge):
    cv_img = bridge.imgmsg_to_cv2(img_msg, "32FC1")
    cv_img_array = np.array(cv_img, dtype=np.dtype("f8"))

    return cv_img_array
    


def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg