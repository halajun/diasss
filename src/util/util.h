
#ifndef UTIL_H
#define UTIL_H

#include<iostream>
#include <boost/filesystem.hpp>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

// #include <pcl/visualization/cloud_viewer.h>

namespace Diasss
{

    class Util
    {
    public:

        static float ComputeIntersection(const std::vector<cv::Mat> &geo_img_s, const std::vector<cv::Mat> &geo_img_t);

        static void LoadInputData(const std::string &strImageFolder, const std::string &strPoseFolder, const std::string &strAltitudeFolder, const std::string &strGroundRangeFolder, const std::string &strAnnotationFolder,
                         std::vector<cv::Mat> &vmImgs, std::vector<cv::Mat> &vmPoses, std::vector<std::vector<double>> &vvAltts, std::vector<std::vector<double>> &vvGranges, std::vector<cv::Mat> &vmAnnos);

        static cv::Mat GetFilterMask(cv::Mat &sss_raw_img);                  

        static cv::Mat NormalizeSSS(cv::Mat &sss_raw_img);
        static cv::Mat NormalizeConvertSSS(Eigen::MatrixXd &sss_wf_img);

        static void ShowAnnos(int &f1, int &f2, cv::Mat &img1, cv::Mat &img2, const cv::Mat &anno1, const cv::Mat &anno2);

        // static pcl::PointCloud<pcl::PointXYZI>::Ptr ImgMosaicOld(std::vector<cv::Mat> &coords, cv::Mat &img);
        // static pcl::PointCloud<pcl::PointXYZI>::Ptr ImgMosaic(cv::Mat &img, cv::Mat &pose, std::vector<double> &g_range);

    };

}

#endif