from server import Server
import cv2
import numpy as np

if __name__ == "__main__":
    server = Server("0.0.0.0", 5002, False, False)

    # Check that connection works
    # message = server.receive()
    # print("[CLIENT]:" + message)

    server.send("START")

    count = 0
    while 1:
        poseStr = server.receive()
        # print(len(poseStr))

        pose1 = np.array(poseStr.split(","))
        # print(pose1)

        pose = np.asarray(pose1, dtype=np.float32, order='C')
        pose = np.reshape(pose, (4, 4))
        print(count, ", Pose: ", pose.shape)
        print(pose)
            
        server.send("ACK")
    
    

#   server.send("Shut up and send an image")

#   # Receive and show image
#   image = server.receive_image()
#   cv2.imshow('SERVER', image)
#   cv2.waitKey(100000)
#   server.send("Thanks!")
