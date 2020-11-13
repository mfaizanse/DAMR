#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd/kinfu.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>
#include <pybind11/stl.h>
#include <tiff.h>

class KinfuPlarr {
public:
    KinfuPlarr(unsigned int width, unsigned int height, float depth_scale, float fx, float fy, float px, float py, bool useOptimized) {
        cv::setUseOptimized(useOptimized);

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

    int integrateFrame(std::vector<uint16> &d) {
//        if(d.size() != width*height) {
//            return -1;
//        }

        int rows = height;
        int cols = width;

        cv::Mat m = cv::Mat(rows, cols, CV_16UC1);
        //copy vector to mat
        memcpy(m.data, d.data(), d.size()*sizeof(uint16));

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

    int update(cv::Mat frameData) {
//        cv::UMat frame(height, width, CV_16UC1);
//        frameData.copyTo(frame);

        bool result = kf->update(frameData);
        //frame.release();

        return result;
    }

    void reset() {
        kf->reset();
    }

    std::vector<float> getPose() {
        auto r = kf->getPose();
        float* p = r.matrix.val;
        std::vector<float> dest(p, p+16);

//        std::vector<float> dest;
//        cv::Affine3f::Mat4 sm = r.matrix;
//
//        for (int i=0;i<sm.rows; i++) {
//            for (int j =0; j< sm.cols; j++) {
//                auto v = sm.row(i).col(j).val[0];
//                dest.push_back(v);
//            }
//        }

        return dest;
    }

    std::vector<float> getCurrentDepth() {
        cv::Mat pts;
        kf->getCurrentFramePoints(pts);

        cv::Mat d;
        cv::extractChannel(pts, d, 2);

//        cv::Mat rs_depth_sccaled;
//        cv::convertScaleAbs(d, rs_depth_sccaled, 0.03);
//
//        cv::Mat tmp;
//        cv::equalizeHist(rs_depth_sccaled, tmp);
//
//        cv::Mat depth_colormap;
//        cv::applyColorMap(tmp, depth_colormap, cv::COLORMAP_JET);
//
//        cv::namedWindow( "KinectFusion3" );
//        imshow( "KinectFusion3", d );
//        cv::waitKey( 1 );

//        std::vector<float> dest;
//        if (d.isContinuous()) {
//            dest.assign((float*)d.datastart, (float*)d.dataend);
//        }

        float* data = (float *) d.data;
        std::vector<float> dest(data, data + (d.rows * d.cols));

        return dest;
    }

    bool renderShow() {
        cv::Mat render;
        try {
            kf->render( render );
        } catch (const std::exception& e) {
            // will be executed if f() throws std::runtime_error
            return false;
        }

        // Show Rendering Image and Point Cloud
        cv::namedWindow( "KinectFusion2" );
        imshow( "KinectFusion2", render );
        cv::waitKey( 1 );
        return true;
    }


private:
    unsigned int width;
    unsigned int height;
    cv::Ptr<cv::kinfu::KinFu> kf;
    cv::Ptr<cv::kinfu::Params> params;
};