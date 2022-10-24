
#ifndef FRAME_H
#define FRAME_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "ORBextractor.h"

namespace Diasss
{

    class Frame
    {

    public:

        // Constructor 
        Frame(const int &id, const cv::Mat &mImg, const cv::Mat &mPose, const std::vector<double> &vAltt, 
              const std::vector<double> &vGrange, const cv::Mat &mAnno);

        // processing with raw image
        cv::Mat GetNormalizeSSS(const cv::Mat &sss_raw_img);
        cv::Mat GetFilteredMask(const cv::Mat &sss_raw_img);
        void DetectFeature(const cv::Mat &img, const cv::Mat &mask, std::vector<cv::KeyPoint> &kps, cv::Mat &dst);
        cv::Mat GetGeoImg(const int &row, const int &col, const cv::Mat &pose, const std::vector<double> &g_range);

        // Initialization items
        int img_id;
        cv::Mat anno_kps;
        cv::Mat raw_img;
        cv::Mat dr_poses;
        std::vector<double> altitudes;
        std::vector<double> ground_ranges;

        // Produced items
        cv::Mat norm_img; // normalized image;
        cv::Mat flt_mask; // binary, mask for filtering area that could be ignored;
        cv::Mat geo_img; // image geo-referenced location in x, y and z
        std::vector<cv::KeyPoint> kps; // detected keypoints
        cv::Mat dst; // descriptors of detected keypoints
        cv::Mat corres_kps; // correspondences of keypoints. row: frame_id, ref_frame_id, kp_x, kp_y, kp_ref_x, kp_ref_y
        cv::Mat est_poses; // estimated poses
        
        


    private:



    };

}

#endif