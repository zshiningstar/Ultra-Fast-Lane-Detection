import rospy
import cv2
import argparse
import os
import glob
from sensor_msgs.msg import Image as SensorImage
from cv_bridge_cus import CvBridge


def read_imgs(imgs_folder):
    path_list = glob.glob(os.path.join(imgs_folder, '*.jpg'))
    path_list.sort()
    mat_list = []
    for item in path_list:
        cvmat = cv2.imread(item)
        mat_list.append(cvmat)
    return mat_list

def run_node(mat_list):
    img_pub = rospy.Publisher('/image_raw', SensorImage, queue_size=1)
    cv_bridge = CvBridge()

    itt = 0
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        msg = cv_bridge.cv2_to_imgmsg(mat_list[itt%len(mat_list)], 'bgr8')
        msg.header.stamp = rospy.rostime.Time.now()
        img_pub.publish(msg)
        itt = itt+1
        
        rate.sleep()

def main():
    rospy.init_node('convert_culane')
    mat_list = read_imgs('/home/rdcas/rosbag/lane-detect/05251207_0380')

    run_node(mat_list)

if __name__ == '__main__':
    main()
    