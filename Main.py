import pyrender
import numpy as np
import trimesh
from numba import jit, cuda
import cv2
import time
import matplotlib.pyplot as plt

from PIL import Image
#from RealismEmbedding.Harmonization import Harmonization
from Communication.python.client import Client
from HumanPose.demo import PoseEstimator
import pickle


@jit(nopython=True)
def compute_mask(rows, cols, im_color, im_depth, obj_depth, mask_, result_):
    for ro in range(rows):
        for co in range(cols):
            if obj_depth[ro][co] != 0:
                if obj_depth[ro][co] > im_depth[ro][co]:
                    result_[ro][co] = im_color[ro][co]
                    mask_[ro][co] = 0.0

                else:
                    result_[ro][co] = color[ro][co]
                    mask_[ro][co] = 255.0
    return result_, mask_
def is_inside_sphere(point,center, radius):
    diff = point-center
    return diff.T@diff < radius*radius


if __name__ == "__main__":
    client = Client("10.152.42.93", 5002)

    client.send("WP3")

    fuze_trimesh = trimesh.load('data/plane/textured.obj')

    scale = trimesh.transformations.scale_matrix(0.5)
    window_size = (640, 480)
    harmonize = Harmonization()
    estimator = PoseEstimator()
    last_hit = 0
    points = 0
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 5, (640, 480))
    while 1:
        # Receive data and unserialize
        start_time = time.time()
        serializedData = client.receive(decode=False)
        data = pickle.loads(serializedData)

        img_depth = data["depthMap"]
        img_depth = img_depth.astype(np.float32)
        img_depth = img_depth / 1000
        # img_depth = ((img_depth * 3.96) / 4.0) + 0.04

        img_color = data["rgbFrame"]
        #Render the Points achieved, can be done later or some other way, but I think this works fine
        cv2.putText(img_color, "Points: {}".format(points),(0,50), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0))
        #Estimate Human Poses
        human_poses = estimator.process_frame(img_color)
        # Get coordinate of right_wrist  == Keypoint 4
        wrist_coord = human_poses[0].keypoints[4]
        # Take a 3 x 3 Patch around that coordinate in the depth map
        hand_patch = img_depth[wrist_coord[0]-1:wrist_coord[0]+2, wrist_coord[1]-1:wrist_coord[1]+2]
        mean_depth = np.mean(hand_patch)
        wrist_coord_3d = np.array([wrist_coord[0], wrist_coord[1], mean_depth])
        

        camera_pose = np.eye(4)
        intrinsics = data["depthIntrincics"]

        object_pose = data["objectPose"]
        ## "Game" Logic
        if is_inside_sphere(wrist_coord_3d, object_pose[:3, 3], 0.1):
            if time.monotonic()- last_hit > 1:
                last_hit = time.monotonic()
                points += 1
        print(object_pose)
        camera = pyrender.camera.IntrinsicsCamera(intrinsics[0][0], intrinsics[1][1], intrinsics[0][2],
                                                  intrinsics[1][2],
                                                  znear=0.04, zfar=4.0, name=None)

        r = pyrender.OffscreenRenderer(window_size[0], window_size[1])
        scene = pyrender.Scene(bg_color=[0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh, poses=object_pose)
        scene.add(mesh)

        scene.add(camera, pose=camera_pose)
        color, depth = r.render(scene)
        dim = depth.shape
        row = dim[0]
        col = dim[1]
        mask = np.zeros_like(depth)
        result = np.copy(img_color)

        indexes = depth > img_depth
        result[indexes] = img_color[indexes]
        mask[indexes] = 0.0
        result[~indexes] = color[~indexes]
        mask[~indexes] = 255.0
        z_indexes = depth == 0
        result[z_indexes] = img_color[z_indexes]

        # uncomment below line to enable harmonization
        # frame = harmonize.compute(Image.fromarray(mask), Image.fromarray(result))
        # comment below line to enable harmonization
        frame = result
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.namedWindow('RS_DepthMap', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RS_DepthMap', frame)
        out.write(frame)

        key = cv2.waitKey(1)
