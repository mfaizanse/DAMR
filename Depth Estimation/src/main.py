from server import Server
import cv2
import numpy as np
import pyrealsense2 as rs
import pickle
import enum
from kinfu_cv import KinfuPlarr

# Define enum for depth options
class DepthOption(enum.Enum):
   RAW = 1
   FILTERED = 2
   RAYCASTED = 3

def runDE(server, depthOption):
    width = 640
    height = 480

    fx = 643.338013
    fy = 643.096008
    px = 638.95697
    py = 402.330017

    depthIntrincics = np.array(
        [[fx, 0, px ],
        [  0, fy, py],
        [  0, 0, 1.0 ]])

    ##FocalLengthColor: 1393.239990 1394.219971
    ##PrincipalPointColor: 965.322998 558.158020
    colorIntrincics = np.array(
        [[1393.239990, 0, 965.322998 ],
        [  0, 1394.219971, 558.158020],
        [  0, 0, 1.0 ]])



    realSensePose = np.identity(4)
    objectPose = np.identity(4)
    depthMapToUse = np.zeros(shape=(height, width))

    # objectPose = np.array([
    #         [-0.01041361, -0.01674143, -0.99980562,  0.00567056],
    #         [-0.99791951,  0.06379406,  0.00932575,  0.00589692],
    #         [ 0.06362554,  0.99782265, -0.01737093, -0.30687113],
    #         [ 0,          0,          0,          1        ]
    #     ])



    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    res_x = width
    res_y = height

    config.enable_stream(rs.stream.depth, res_x, res_y, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, res_x, res_y, rs.format.bgr8, 30)

    # config.enable_stream(rs.stream.infrared, 1, res_x, res_y, rs.format.y8, 30)
    # config.enable_stream(rs.stream.infrared, 2, res_x, res_y, rs.format.y8, 30)

    # Start streaming
    pipeline_profile = pipeline.start(config)

    ## To set laser on/off or set laser power
    device = pipeline_profile.get_device()
    depth_sensor = device.query_sensors()[0]
    laser_pwr = depth_sensor.get_option(rs.option.laser_power)
    print("laser power = ", laser_pwr)
    laser_range = depth_sensor.get_option_range(rs.option.laser_power)
    print("laser power range = " , laser_range.min , "~", laser_range.max)
    depth_sensor.set_option(rs.option.laser_power, 200)

    spatial = rs.spatial_filter()
    # spatial.set_option(rs.option.filter_magnitude, 3)
    # spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    # spatial.set_option(rs.option.filter_smooth_delta, 20)
    # spatial.set_option(rs.option.holes_fill, 0)
    temporal = rs.temporal_filter()
    # decimation = rs.decimation_filter()
    ## decimation.set_option(rs.option.filter_magnitude, 2)
    hole_filling = rs.hole_filling_filter()
    # hole_filling.set_option(rs.option.holes_fill, 1)

    ## Get firs frame, just to warm up the camera
    f1 = pipeline.wait_for_frames()
    d1 = f1.get_depth_frame()

    ## Get Sensor and set a visual preset
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()

    preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
    print('Preset Range: ' + str(preset_range))

    for i in range(int(preset_range.max)):
        visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
        print(i, visulpreset)
        
        if visulpreset == "High Density":
            depth_sensor.set_option(rs.option.visual_preset, i)
            print('Preset set to ', visulpreset)

    ## Get Depth scale of camera (1 / depth_scale)
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    ## Create object of KinectFusion
    kfp = KinfuPlarr(width, height, depth_scale, fx, fy, px, py, True)

    count = 0

    ######## Start the frames
    while 1:
        ## Get frame from realsense
        frames = pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        if count < 50:
            count = count + 1
            continue

        depth_image = np.asanyarray(depth_frame.get_data()) # dtype of print(depth_image.dtype) is: uint16
        color_image = np.asanyarray(color_frame.get_data()) # FORMAT: BGR, dtype of print(color_image.dtype) is: uint8
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)


        ## Apply filters to real-sense depth
        filtered_rs_depth = spatial.process(depth_frame)
        filtered_rs_depth = temporal.process(filtered_rs_depth)
        filtered_rs_depth = hole_filling.process(filtered_rs_depth)
        filtered_rs_depth = np.asanyarray(filtered_rs_depth.get_data()) # dtype of print(filtered_rs_depth.dtype) is: uint16

        # ## Apply Kinect Fusion pipeline
        # depth_image_flatten = np.array(filtered_rs_depth).flatten()
        # depth_image_flatten = depth_image_flatten.tolist()

        # raycasted_depth = np.zeros(shape=(height, width))

        # isSuccess = kfp.integrateFrame(depth_image_flatten)
        # if isSuccess:
        #     # Get rs camera pose
        #     poseTmp = kfp.getPose()
        #     poseTmp = np.asarray(poseTmp, dtype=np.float32, order='C').reshape((4, -1))
        #     realSensePose = poseTmp

        #     # Get raycasted depth map
        #     d = kfp.getCurrentDepth()
        #     d = np.array(d)
        #     d = np.nan_to_num(d)
        #     d = d.reshape((height, -1))
        #     d = d * (1/depth_scale)
        #     d = d.astype(np.int16)
        #     raycasted_depth = d

        #     kfp.renderShow()

        #     # print(pose)
        #     #kfp.renderShow()

        # else:
        #     print("Kinect fusion frame integration failed !!!!")

        if depthOption == DepthOption.RAW:
            depthMapToUse = depth_image
        elif depthOption == DepthOption.FILTERED:
            depthMapToUse = filtered_rs_depth
        elif depthOption == DepthOption.RAYCASTED:
            depthMapToUse = raycasted_depth
        else:
            print("Error: Unknown depthOption !!!")
            break


        ## Recieve object pose from WP1
        if server.isWP2Connected():
            try:
                ## send confirmation to WP1 to send next pose
                server.send("ACK")
                ## receive & deserialize the pose
                # print("recv")

                serializedPose = server.receive(decode=False)
                objectPose = pickle.loads(serializedPose)
            except Exception as e:
                print("WP2, ", e)
        # print("objectPose: ", objectPose.shape)
        # print("objectPose.dtype: ", objectPose.dtype)
        # print(objectPose)

        data = {
            "objectPose": objectPose,
            "depthMap": depthMapToUse,
            "bgrFrame": color_image,
            "rgbFrame": color_image_rgb,
            "depthIntrincics": depthIntrincics,
            "colorIntrincics": colorIntrincics,
            "realSensePose": realSensePose
        }

        ## Send data to WP3
        serializedData = pickle.dumps(data)
        if server.isWP3Connected():
            try:
                server.sendToWp3(serializedData, encode=False)
            except Exception as e:
                print("WP3, ", e)
            
        ## process next frame

        ## Visualize depth map
        rs_depth_sccaled  = cv2.convertScaleAbs(data["depthMap"], alpha=0.03)  # (image converted to 8-bit per pixel)
        depth_colormap = cv2.applyColorMap(cv2.equalizeHist(rs_depth_sccaled), cv2.COLORMAP_JET)
        cv2.namedWindow('RS_DepthMap', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RS_DepthMap', depth_colormap)

        ## Visualize RGB frame
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', data["bgrFrame"])

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
    
    


if __name__ == "__main__":
    server = Server("0.0.0.0", 5002, automatic_port=False, wait_for_wp3=True)
    runDE(server, DepthOption.FILTERED)
    server.close()

    # while 1:
    #     server = None
    #     try:
    #         server = Server("0.0.0.0", 5002, automatic_port=False, wait_for_wp3=True)
    #         runDE(server)
    #         print("Exiting....")
    #         break
    #     except:
    #         try:
    #             print("Exception in main....")
    #             server.close()
    #         except:
    #             a = 1
    #         continue

