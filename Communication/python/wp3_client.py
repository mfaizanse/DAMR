from client import Client
import numpy as np
import pickle
import cv2

if __name__ == "__main__":
    client = Client("0.0.0.0", 5002)

    client.send("WP3")

    while 1:
        ## Receive data and unserialize
        serializedData = client.receive(decode=False)
        data = pickle.loads(serializedData)
        # print(data)

        depthMap = data["depthMap"]
        bgrFrame = data["bgrFrame"]

        ## Visualize depth map
        rs_depth_sccaled  = cv2.convertScaleAbs(depthMap, alpha=0.03)  # (image converted to 8-bit per pixel)
        depth_colormap = cv2.applyColorMap(cv2.equalizeHist(rs_depth_sccaled), cv2.COLORMAP_JET)
        cv2.namedWindow('RS_DepthMap', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RS_DepthMap', depth_colormap)

        ## Visualize RGB frame
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', bgrFrame)

        key = cv2.waitKey(1)
