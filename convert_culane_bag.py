import rospy
import cv2
import argparse
import os
import glob
from sensor_msgs.msg import Image as SensorImage
from cv_bridge_cus import CvBridge

def read_crowed_image(dataset_folder):
    test_split_crowed = os.path.join(dataset_folder, 'list/test_split/test1_crowd.txt')
    test_split_file = open(test_split_crowed, 'r')
    line_list = test_split_file.readlines()
    new_list = []
    for item in line_list: 
        item = item.rstrip(' \n')
        item = os.path.join(dataset_folder, item)
        new_list.append(item)
    new_list.sort()
    print(new_list)
    print(len(new_list))
    return new_list
    


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

def run_node_on_path_list(path_list):
    img_pub = rospy.Publisher('/image_raw', SensorImage, queue_size=1)
    cv_bridge = CvBridge()

    itt = 0
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        cur_im_mat = cv2.imread(path_list[itt%len(path_list)])
        msg = cv_bridge.cv2_to_imgmsg(cur_im_mat, 'bgr8')
        msg.header.stamp = rospy.rostime.Time.now()
        img_pub.publish(msg)
        itt = itt+1
        rate.sleep()

def main():
    path_list = read_crowed_image('/media/data/lane_detect_dataset/CULane')
    rospy.init_node('convert_culane')
    # mat_list = read_imgs('/home/rdcas/rosbag/lane-detect/05251207_0380')
    run_node_on_path_list(path_list)

if __name__ == '__main__':
    main()
    