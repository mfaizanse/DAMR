import cv2
import numpy as np
import json
import pickle
from client import Client


def MatToString(mat):
    return ",".join(map(str,mat.flatten().tolist()))

if __name__ == "__main__":

    with open('MarkerTrackerPy/Config.txt', 'r') as infile:
        config = json.load(infile)


    try:
        client = Client(config['host'], config['port'])
        connected = True
        client.send("WP2")
        print("Connected")
    except:
        connected = False
        print("Could not connect to the specified host: {}:{}".format(config['host'], config['port']))
    

    

    cameraMat = np.array([[config['f_x'], 0, config['c_x']],
                        [0, config['f_y'], config['c_y']],
                        [0, 0, 1]])
    distCoeffs = np.array([config['k_1'],config['k_2'],config['p_1'],config['p_2'], config['k_3']])
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard_create(config['CharucoWidth'],config['CharucoHeight'],config['CheckerboardSize'],config['MarkerSize'], dictionary)

    parameters = cv2.aruco.DetectorParameters_create()

    imageFeed = cv2.VideoCapture(0)

    scale = 3
    while(imageFeed.grab()):
        _, frame = imageFeed.retrieve()
        corners, ids, rejects = cv2.aruco.detectMarkers(frame, board.dictionary, parameters=parameters, cameraMatrix=cameraMat, distCoeff=distCoeffs)
        
        if(ids is not None):
            _, charCorners, charIds =  cv2.aruco.interpolateCornersCharuco(corners, ids, frame, board,cameraMatrix=cameraMat, distCoeffs=distCoeffs)
            #cv2.aruco.drawDetectedCornersCharuco(frame, charCorners,charIds)
            rvec = np.array([0.0,0.0,0.0])
            tvec = np.array([0.0,0.0,0.0])
            _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charCorners, charIds, board, cameraMat, distCoeffs, rvec, tvec)
            if(np.any(tvec)):
                matR, _ = cv2.Rodrigues(rvec)
                centerOffset = matR@np.array([0.5*config['CharucoWidth']*config['CheckerboardSize'], 0.5*config['CharucoHeight'] * config['CheckerboardSize'], 0])
                tvec += centerOffset
                cv2.aruco.drawAxis(frame, cameraMat, distCoeffs, rvec, tvec, config['MarkerSize'])
                
                poseMatrix = np.zeros((4,4))
                poseMatrix[:3,:3], _ = cv2.Rodrigues(rvec)
                poseMatrix[:3, 3] = tvec * scale
                poseMatrix[2, 3] = poseMatrix[2, 3] 
                ##Convert to OpenGL Coordinate System (rotate 180 around x, i.e, negate y and z axis)
                poseMatrix[1:3, :4] = -poseMatrix[1:3, :4]
                ##Rotate Object 90° around x
                rotate = np.array([[1, 0, 0, 0], 
                                [0, 0, -1, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1]])
                poseMatrix = poseMatrix@rotate
                poseMatrix[3,3] = 1
                if connected:
                    serializedPose = pickle.dumps(poseMatrix)
                    client.send(serializedPose, encode=False)

                    reply = client.receive()
                
                    print("[Server]: {}".format(reply))
                    if("ACK" not in reply):
                        print("[Client]: No ACK received. Exiting...")
                        break

        #cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                

        cv2.imshow("Window", frame)
        key = cv2.waitKey(16)
        if(key == ord('-')):
            print("scaling down")
            scale = scale -1
            print(scale)
        if(key == ord('+')):
            print("scaling up")
            scale = scale +1
            print(scale)
        if(key == 27):
            break