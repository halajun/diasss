
#ifndef UTIL_H
#define UTIL_H

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

// #include <pcl/visualization/cloud_viewer.h>

namespace Diasss
{

    class Util
    {
    public:

        static void LoadInputData(const std::string &strImageFolder, const std::string &strPoseFolder, const std::string &strAltitudeFolder, const std::string &strGroundRangeFolder, const std::string &strAnnotationFolder,
                         std::vector<cv::Mat> &vmImgs, std::vector<cv::Mat> &vmPoses, std::vector<std::vector<double>> &vvAltts, std::vector<std::vector<double>> &vvGranges, std::vector<cv::Mat> &vmAnnos);

        static cv::Mat GetFilterMask(cv::Mat &sss_raw_img);                  

        static cv::Mat NormalizeSSS(cv::Mat &sss_raw_img);
        static cv::Mat NormalizeConvertSSS(Eigen::MatrixXd &sss_wf_img);

        // static pcl::PointCloud<pcl::PointXYZI>::Ptr ImgMosaicOld(std::vector<cv::Mat> &coords, cv::Mat &img);
        // static pcl::PointCloud<pcl::PointXYZI>::Ptr ImgMosaic(cv::Mat &img, cv::Mat &pose, std::vector<double> &g_range);

    };

}

#endif