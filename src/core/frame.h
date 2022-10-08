
#ifndef FRAME_H
#define FRAME_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace Diasss
{

    class Frame
    {

    public:

        // Constructor 
        Frame(const int &id, const cv::Mat &mImg, const cv::Mat &mPose, const std::vector<double> &vAltt, const std::vector<double> &vGrange);

        // processing with raw image
        cv::Mat GetNormalizeSSS(const cv::Mat &sss_raw_img);
        cv::Mat GetFilteredMask(const cv::Mat &sss_raw_img);
        void DetectFeature(const cv::Mat &img, const cv::Mat &mask, std::vector<cv::KeyPoint> &kps, cv::Mat &dst);

        // Initialization items
        int img_id;
        cv::Mat raw_img;
        cv::Mat dr_poses;
        std::vector<double> altitudes;
        std::vector<double> ground_ranges;

        cv::Mat norm_img; // normalized image;
        cv::Mat flt_mask; // binary, mask for filtering area that could be ignored;
        std::vector<cv::KeyPoint> kps; // detected keypoints
        cv::Mat dst; // descriptors of detected keypoints

        // Image geo-referenced location in x, y and z
        cv::Mat geo_img;
        cv::Mat GetGeoImg(const int &row, const int &col, const cv::Mat &pose, const std::vector<double> &g_range);


    private:



    };

}

#endif