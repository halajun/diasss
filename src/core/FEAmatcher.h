
#ifndef FEAMATCHER_H
#define FEAMATCHER_H

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>

#include "frame.h"

namespace Diasss
{

    class FEAmatcher
    {
    public:

        static std::vector<std::pair<size_t, size_t>> RobustMatching(Frame &SourceFrame, Frame &TargetFrame);

        static std::vector<int> GeoNearNeighSearch(const int &img_id, const int &img_id_ref,
                                                   const cv::Mat &img, const cv::Mat &img_ref,
                                                   const std::vector<cv::KeyPoint> &kps, const cv::Mat &dst, const cv::Mat &geo_img,
                                                   const std::vector<cv::KeyPoint> &kps_ref, const cv::Mat &dst_ref, const cv::Mat &geo_img_ref,
                                                   std::vector<std::pair<int,double>> &scc);

        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);


    };

}

#endif