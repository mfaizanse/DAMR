#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd/kinfu.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>
#include <tiff.h>
//#include "ndarray.h"

class KinfuPlarr {
public:
    KinfuPlarr(unsigned int width, unsigned int height, float depth_scale, float fx, float fy, float px, float py) {
        this->width = width;
        this->height = height;

        // Configure kinfu_plarr's parameters
        params = cv::kinfu::Params::defaultParams();
        params->frameSize = cv::Size(width, height);
        params->intr = cv::Matx33f(fx, 0,    px,
                                   0,     fy,    py,
                                   0,  0, 1);
        params->depthFactor = 1 / depth_scale;

        // Initialize KinFu object
        kf = cv::kinfu::KinFu::create(params);
    }

    ~KinfuPlarr() {}

    bool getTestValue(std::vector<int16> &d) {
        if(d.size() != width*height) {
            return -1;
        }

        int rows = height;
        int cols = width;

        cv::Mat m = cv::Mat(rows, cols, CV_16UC1);
        //copy vector to mat
        memcpy(m.data, d.data(), d.size()*sizeof(int16));

//        cv::Mat rs_depth_sccaled;
//        cv::convertScaleAbs(m, rs_depth_sccaled, 0.03);
//
//        cv::Mat tmp;
//        cv::equalizeHist(rs_depth_sccaled, tmp);
//
//        cv::Mat depth_colormap;
//        cv::applyColorMap(tmp, depth_colormap, cv::COLORMAP_JET);
//
//        cv::namedWindow( "Kinect Fusion" );
//        imshow( "Kinect Fusion", depth_colormap );
//        cv::waitKey( 1 );

        return update(m);
    }

    bool update(cv::Mat frameData) {
//        cv::UMat frame(height, width, CV_16UC1);
//        frameData.copyTo(frame);

        bool result = kf->update(frameData);
        //frame.release();

        return result;
    }

    void reset() {
        kf->reset();
    }

    void getCurrentPoints() {
        cv::Mat pts;
        kf->getCurrentFramePoints(pts);

        cv::Mat d;
        cv::extractChannel(pts, d, 2);



        //MatType(d);

//        std::cout << "dsa1 " << d.size << std::endl;
//        std::cout << "dsa2 " << d.channels() << std::endl;

//        cv::Mat rs_depth_sccaled;
//        cv::convertScaleAbs(d, rs_depth_sccaled, 0.03);
//
//        cv::Mat tmp;
//        cv::equalizeHist(rs_depth_sccaled, tmp);
//
//        cv::Mat depth_colormap;
//        cv::applyColorMap(tmp, depth_colormap, cv::COLORMAP_JET);

//        cv::namedWindow( "Kinect Fusion" );
//        imshow( "Kinect Fusion", d );
//        cv::waitKey( 1 );
    }

    void MatType( cv::Mat inputMat )
    {
        int inttype = inputMat.type();

        std::string r, a;
        uchar depth = inttype & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (inttype >> CV_CN_SHIFT);
        switch ( depth ) {
            case CV_8U:  r = "8U";   a = "Mat.at<uchar>(y,x)"; break;
            case CV_8S:  r = "8S";   a = "Mat.at<schar>(y,x)"; break;
            case CV_16U: r = "16U";  a = "Mat.at<ushort>(y,x)"; break;
            case CV_16S: r = "16S";  a = "Mat.at<short>(y,x)"; break;
            case CV_32S: r = "32S";  a = "Mat.at<int>(y,x)"; break;
            case CV_32F: r = "32F";  a = "Mat.at<float>(y,x)"; break;
            case CV_64F: r = "64F";  a = "Mat.at<double>(y,x)"; break;
            default:     r = "User"; a = "Mat.at<UKNOWN>(y,x)"; break;
        }
        r += "C";
        r += (chans+'0');
        std::cout << "Mat is of type " << r << " and should be accessed with " << a << std::endl;

    }

    cv::Affine3<float>::Mat4 getPose() {
        auto r = kf->getPose();
        return r.matrix;
//        float* p = r.matrix.val;
//
//        std::vector<float> dest(p, p+16);
//
//        return dest;
    }

    void renderShow() {
        cv::Mat render;
        kf->render( render );

        // Show Rendering Image and Point Cloud
        cv::namedWindow( "KinectFusion2" );
        imshow( "KinectFusion2", render );
        cv::waitKey( 1 );

        return;
    }


public:
    unsigned int width;
    unsigned int height;
    cv::Ptr<cv::kinfu::KinFu> kf;
    cv::Ptr<cv::kinfu::Params> params;
    // std::vector<Vector4uc> m_color; // color stored as 4 unsigned char
};