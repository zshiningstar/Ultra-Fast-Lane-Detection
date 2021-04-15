#!/usr/bin/env python3

import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch

import numpy as np
import torchvision.transforms as transforms

from data.constant import culane_row_anchor
import rospy
import threading
from sensor_msgs.msg import Image as SensorImage
import PIL
from cv_bridge_cus import CvBridge
from matplotlib import pyplot as plt
import scipy.special

class LaneDetectRosNode():
    def __init__(self):
        self.msg_lock = threading.Lock()
        self.init_model()
        self.in_img = None

        rospy.Subscriber('/image_raw', SensorImage, self.image_callback, queue_size=1)

        self.detect_img_pub = rospy.Publisher('/lane_detected', SensorImage, queue_size=1)
        self.cv_bridge = CvBridge()
        

    def init_model(self):
        torch.backends.cudnn.benchmark = True
        self.args, self.cfg = merge_config()

        assert self.cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

        if self.cfg.dataset == 'CULane':
            cls_num_per_lane = 18
        elif self.cfg.dataset == 'Tusimple':
            cls_num_per_lane = 56
        else:
            raise NotImplementedError

        self.net = parsingNet(pretrained = False, backbone=self.cfg.backbone,cls_dim = (self.cfg.griding_num+1,cls_num_per_lane,4),
                use_aux=False).cuda() # we dont need auxiliary segmentation in testing

        state_dict = torch.load(self.cfg.test_model, map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        self.net.load_state_dict(compatible_state_dict, strict=False)
        self.net.eval()

        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    def image_callback(self, msg):
        bgr_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        rgb_img_cv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        img_h = rgb_img_cv.shape[0]
        img_w = rgb_img_cv.shape[1]
        # print(rgb_img_cv.shape)
        
        rgb_img = self.img_transforms(PIL.Image.fromarray(rgb_img_cv))

        imgs = torch.Tensor(rgb_img)
        imgs = torch.reshape(imgs, (1,3,288,800))

        imgs = imgs.cuda()
        with torch.no_grad():
            out = self.net(imgs)
        
        col_sample = np.linspace(0, 800 - 1, self.cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        row_anchor = culane_row_anchor
        cls_num_per_lane = 18

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(self.cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == self.cfg.griding_num] = 0
        out_j = loc

        # print(out_j.shape)
        color = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        cv2.circle(rgb_img_cv,ppp,5,color[i],-1)

        label_color_msg = self.cv_bridge.cv2_to_imgmsg(rgb_img_cv, 'rgb8')
        label_color_msg.header = msg.header
        self.detect_img_pub.publish(label_color_msg)



if __name__ == "__main__":
    rospy.init_node('lane_detect_node')

    node = LaneDetectRosNode()
    rospy.spin()
