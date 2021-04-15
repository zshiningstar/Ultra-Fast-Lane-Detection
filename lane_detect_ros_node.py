#!/usr/bin/env python3

import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch

import numpy as np
import torchvision.transforms as transforms

from data.constant import culane_row_anchor, tusimple_row_anchor
import rospy
import threading
from sensor_msgs.msg import Image as SensorImage
import PIL
from cv_bridge import CvBridge
import pylab


class LaneDetectRosNode():
    def __init__(self):
        self.msg_lock = threading.Lock()

        rospy.Subscriber('', SensorImage, self.image_callback, queue_size=1)

        self.cv_bridge = CvBridge()
        self.init_model()

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
        if self.msg_lock.acquire(False):
            self.in_img = msg
            self.msg_lock.release()

    def run_detect(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.msg_lock.acquire(False):
                img_msg = self.in_img
                self.in_img = None
                self.msg_lock.release
            else:
                rate.sleep()
                continue
            
            rgb_image = self.cv_bridge.imgmsg_to_cv2(img_msg, "passthrough")
            
            pylab.imshow(rgb_image)
            rate.sleep()



if __name__ == "__main__":
    rospy.init_node('lane_detect_node')
    # rospy.loginfo('start lane detect rosnode...')

    node = LaneDetectRosNode()
    node.run_detect()




    # if self.cfg.dataset == 'CULane':
    #     splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
    #     datasets = [LaneTestDataset(self.cfg.data_root,os.path.join(self.cfg.data_root, 'list/test_split/'+split),img_transform = img_transforms) for split in splits]
    #     img_w, img_h = 1640, 590
    #     row_anchor = culane_row_anchor
    # elif self.cfg.dataset == 'Tusimple':
    #     splits = ['test.txt']
    #     datasets = [LaneTestDataset(self.cfg.data_root,os.path.join(self.cfg.data_root, split),img_transform = img_transforms) for split in splits]
    #     img_w, img_h = 1280, 720
    #     row_anchor = tusimple_row_anchor
    # else:
    #     raise NotImplementedError
    # for split, dataset in zip(splits, datasets):
    #     loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
    #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #     print(split[:-3]+'avi')
    #     vout = cv2.VideoWriter(split[:-3]+'avi', fourcc , 30.0, (img_w, img_h))
    #     for i, data in enumerate(tqdm.tqdm(loader)):
    #         imgs, names = data
    #         imgs = imgs.cuda()
    #         with torch.no_grad():
    #             out = net(imgs)

    #         col_sample = np.linspace(0, 800 - 1, self.cfg.griding_num)
    #         col_sample_w = col_sample[1] - col_sample[0]


    #         out_j = out[0].data.cpu().numpy()
    #         out_j = out_j[:, ::-1, :]
    #         prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    #         idx = np.arange(self.cfg.griding_num) + 1
    #         idx = idx.reshape(-1, 1, 1)
    #         loc = np.sum(prob * idx, axis=0)
    #         out_j = np.argmax(out_j, axis=0)
    #         loc[out_j == self.cfg.griding_num] = 0
    #         out_j = loc

    #         # import pdb; pdb.set_trace()
    #         vis = cv2.imread(os.path.join(self.cfg.data_root,names[0]))
    #         # print(out_j.shape)
    #         color = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    #         for i in range(out_j.shape[1]):
    #             if np.sum(out_j[:, i] != 0) > 2:
    #                 for k in range(out_j.shape[0]):
    #                     if out_j[k, i] > 0:
    #                         ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
    #                         cv2.circle(vis,ppp,5,color[i],-1)
    #         vout.write(vis)
        
    #     vout.release()