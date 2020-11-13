
// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp>
#include <librealsense2/hpp/rs_internal.hpp>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd/kinfu.hpp>
//#include <opencv2/viz.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/rgbd.hpp>

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

using namespace cv;
using namespace cv::kinfu;

static float max_dist = 2.5;
static float min_dist = 0;

int main()
{
    setUseOptimized(true);

    // Declare KinFu and params pointers
    Ptr<KinFu> kf;
    Ptr<Params> params = Params::defaultParams();

    // Create a pipeline and configure it
    rs2::pipeline p;
    rs2::config cfg;
    float depth_scale;
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16);
    auto profile = p.start(cfg);
    auto dev = profile.get_device();
    auto stream_depth = profile.get_stream(RS2_STREAM_DEPTH);

    // Get a new frame from the camera
    rs2::frameset data = p.wait_for_frames();
    auto d = data.get_depth_frame();


    for (rs2::sensor& sensor : dev.query_sensors())
    {
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            // Set some presets for better results
            dpt.set_option(RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_DENSITY);
            // Depth scale is needed for the kinfu_plarr set-up
            depth_scale = dpt.get_depth_scale();
            break;
        }
    }

    // Declare post-processing filters for better results
    auto decimation = rs2::decimation_filter();
    auto spatial = rs2::spatial_filter();
    auto temporal = rs2::temporal_filter();

    auto clipping_dist = max_dist / depth_scale; // convert clipping_dist to raw depth units

    // Use decimation once to get the final size of the frame
    d = decimation.process(d);
    auto w = d.get_width();
    auto h = d.get_height();
    Size size = Size(w, h);

    auto intrin = stream_depth.as<rs2::video_stream_profile>().get_intrinsics();

    // Configure kinfu_plarr's parameters
    params->frameSize = size;
    params->intr = Matx33f(intrin.fx, 0, intrin.ppx,
                           0, intrin.fy, intrin.ppy,
                           0, 0, 1);
    params->depthFactor = 1 / depth_scale;

    // Initialize KinFu object
    kf = KinFu::create(params);

    cv::namedWindow( "Kinect Fusion" );

    while (true)
    {
        // Block program until frames arrive
        rs2::frameset frames = p.wait_for_frames();

        // Try to get a frame of a depth image
        rs2::depth_frame depth = frames.get_depth_frame();

        // Use post processing to improve results
        depth = decimation.process(depth);
        depth = spatial.process(depth);
        depth = temporal.process(depth);

        // Get the depth frame's dimensions
        float width = depth.get_width();
        float height = depth.get_height();

        // Query the distance from the camera to the object in the center of the image
        float dist_to_center = depth.get_distance(width / 2, height / 2);

        // Print the distance
        //std::cout << "The camera is facing an object " << dist_to_center << " meters away \r";

        // Define matrices on the GPU for KinFu's use
//        UMat points;
//        UMat normals;
        // Copy frame from CPU to GPU
        Mat f(h, w, CV_16UC1, (void*)depth.get_data());
        UMat frame(h, w, CV_16UC1);
        f.copyTo(frame);
        f.release();

        // Run KinFu on the new frame(on GPU)
        if (!kf->update(frame))
        {
            kf->reset(); // If the algorithm failed, reset current state
            // Save the pointcloud obtained before failure
            //export_to_ply(_points, _normals);

            // To avoid calculating pointcloud before new frames were processed, set 'after_reset' to 'true'
            //after_reset = true;
            //points.release();
            //normals.release();
            std::cout << "reset" << std::endl;
        }
        else {
            std::cout << "OK" << std::endl;
            Affine3f pose = kf->getPose();
            std::cout << pose.matrix << std::endl;
            // Retrieve Rendering Image
            cv::Mat render;
            kf->render( render );

//            Mat render_cpu;
//            render.copyTo(render_cpu);
//
//            // Retrieve Point Cloud
//            cv::UMat points;
//            kf->getPoints( points );

            // Show Rendering Image and Point Cloud
            cv::namedWindow( "Kinect Fusion" );
            imshow( "Kinect Fusion", render );
//            viz::WCloud cloud( points, cv::viz::Color::white() );
//            viewer.showWidget( "cloud", cloud );
//            viewer.spinOnce();

            const int32_t key = cv::waitKey( 1 );
            if( key == 'r' ){
                kf->reset();
            }
            else if( key == 'q' ){
                break;
            }

        }
    }

    cv::destroyAllWindows();

    return 0;
}