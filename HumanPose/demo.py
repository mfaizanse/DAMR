import argparse

import cv2
import numpy as np
import torch
import math
import time

from .models.with_mobilenet import PoseEstimationWithMobileNet
from .modules.keypoints import extract_keypoints, group_keypoints
from .modules.load_state import load_state
from .modules.pose import Pose, track_poses

def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

class PoseEstimator(object):
    def __init__(self, checkpoint_path = 'HumanPose/checkpoint_iter_370000.pth', cpu = False, track= True, smooth= True, height_size = 256, draw_image = False):
        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        load_state(net, checkpoint)
        if not torch.cuda.is_available():
            cpu = True
        self.net = net.eval()
        if not cpu:
            self.net = net.cuda()
        self.cpu = cpu
        self.track = track
        self.smooth = smooth
        self.height_size = height_size
        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = Pose.num_kpts
        self.previous_poses = []
        self.delay = 1
        self.draw_image = draw_image

    def process_frame(self, img):
        start_time = time.monotonic()
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(self.net, img, self.height_size, self.stride, self.upsample_ratio, self.cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(self.num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(self.num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if self.track:
            track_poses(self.previous_poses, current_poses, smooth=self.smooth)
            self.previous_poses = current_poses
        if self.draw_image:
            for pose in current_poses:
                pose.draw(img)
            img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
            for pose in current_poses:
                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                            (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                if self.track:
                    cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            cv2.putText(img, "FPS: {:.3f}".format(1/(time.monotonic() - start_time)),(0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
            cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
            key = cv2.waitKey(self.delay)
            if key == 27:  # esc
                quit()
            elif key == 112:  # 'p'
                if self.delay == 33:
                    self.delay = 0
                else:
                    self.delay = 33        
        return current_poses
def is_inside_sphere(point,center, radius):
    diff = point-center
    return diff.T@diff < radius*radius

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Human Pose Estimation based on Light-weight human pose estimation from Daniil-Osokin''')
    parser.add_argument('--checkpoint-path', type=str, default='HumanPose/checkpoint_iter_370000.pth', help='path to the checkpoint')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()
    #glast_hit = 0
    #points = 0
    image_provider = VideoReader(0)
    estimator = PoseEstimator(args.checkpoint_path, not torch.cuda.is_available(), args.track, args.smooth, draw_image=True)
    for img in image_provider:
        poses = estimator.process_frame(img)
        ##Example usage:
        ## (Assuming only one person for this)
        ## (Could be expanded because it has IOU tracking)
        ## poses[0].keypoints[4] -> [x y] tuple for the right wrist
        #wrist_coord = poses[0].keypoints[4]
        ##Insert depth map instead of img here
        #hand_patch = img[wrist_coord[0]-1:wrist_coord[0]+2, wrist_coord[1]-1:wrist_coord[1]+2]
        #mean_depth = np.mean(hand_patch)
        #wrist_coord = np.array([wrist_coord[0], wrist_coord[1], mean_depth])
        #if is_inside_sphere(wrist_coord, np.array([0,0,0]), 1.0):
        #    if time.monotonic()- last_hit > 1:
        #        last_hit = time.monotonic()
        #        points += 1
        ## Point count
        #cv2.putText(img, "Points: {}".format(points),(0,50), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0))