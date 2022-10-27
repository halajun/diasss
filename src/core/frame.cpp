#include<iostream>

#include "frame.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

namespace Diasss
{

using namespace std;
using namespace cv;

#define PI 3.14159265359

Frame::Frame(const int &id, const cv::Mat &mImg, const cv::Mat &mPose, const std::vector<double> &vAltt, 
             const std::vector<double> &vGrange, const cv::Mat &mAnno)
{

    // --- initialize --- //
    raw_img = mImg;
    dr_poses = mPose;
    altitudes = vAltt;
    ground_ranges = vGrange;
    img_id = id;
    anno_kps = mAnno;
    tf_stb = {-3.119, -0.405, -0.146}; // sensor offset (starboard) ENU
    tf_port = {-3.119, 0.405, -0.146}; // sensor offset (port) ENU


    // --- get normalized image --- //
    norm_img = GetNormalizeSSS(mImg);
    // --- get filtered mask --- //
    flt_mask = GetFilteredMask(mImg);
    // --- get geo-referenced img --- //
    geo_img = GetGeoImg(mImg.rows,mImg.cols,mPose,vGrange);
    // --- detect keypoints and extract descriptors -- //
    DetectFeature(norm_img,flt_mask,kps,dst);


}

cv::Mat Frame::GetNormalizeSSS(const cv::Mat &sss_raw_img)
{
    double factor = 2.5, min_val, max_val, max_used;
    cv::Mat output_img = cv::Mat::zeros(sss_raw_img.size(), CV_64FC1);

    cv::Scalar MeanofMat = cv::mean(sss_raw_img);
    max_used = MeanofMat[0]*factor;
    cv::minMaxLoc(sss_raw_img, &min_val, &max_val);
    // cout << "min max: " << min_val << " " << max_val << endl;

    for (int i = 0; i < sss_raw_img.rows; i++)
    {
        for (int j = 0; j < sss_raw_img.cols; j++)
        {
            output_img.at<double>(i,j) = (sss_raw_img.at<double>(i,j)-min_val)/(max_used-min_val)*255.0;
            if (output_img.at<double>(i,j)>255.0)
                output_img.at<double>(i,j)=255.0;
        }

    }

    output_img.convertTo(output_img, CV_8U);  
    
    return output_img;
}

cv::Mat Frame::GetFilteredMask(const cv::Mat &sss_raw_img)
{
    float factor = 2.5;
    int width = 10, r = 6, side = 150;

    cv::Mat output_mask(sss_raw_img.size(), CV_8UC1, Scalar(255));

    cv::Scalar MeanofMat = cv::mean(sss_raw_img);
    // cout << "mean: " << MeanofMat[0] << endl;

    for (int i = 0; i < output_mask.rows; i++)
    {
        for (int j = 0; j < output_mask.cols; j++)
        {
            // remove sensor buggy line
            if (sss_raw_img.at<double>(i,j)>MeanofMat[0]*factor)
            {
                for (size_t x = i-r; x < i+r; x++)
                    for (size_t y = j-r; y < j+r; y++)
                        output_mask.at<bool>(x,y) = 0;
            }
            // remove centre line
            if (j>output_mask.cols/2-width && j<output_mask.cols/2+width)
                output_mask.at<bool>(i,j) = 0;
            // remove the first and last turning pings
            if (i<side || i>output_mask.rows-side)
                output_mask.at<bool>(i,j) = 0;
            // remove the left and right side columns
            if (j<side*0.6 || j>output_mask.cols-side*0.6)
                output_mask.at<bool>(i,j) = 0;
        }

    }
    
    // cv::Mat out_demo;
    // sss_raw_img.copyTo(out_demo,output_mask);
    // cv::namedWindow("filtered mask", cv::WINDOW_AUTOSIZE);
    // cv::imshow("filtered mask", out_demo);
    // cv::waitKey(0);
    
    return output_mask;
}

std::vector<cv::Mat> Frame::GetGeoImg(const int &row, const int &col, const cv::Mat &pose, const std::vector<double> &g_range)
{
    // initialize x y and z
    cv::Mat bin_loc_x = cv::Mat::zeros(row, col, CV_64FC1);
    cv::Mat bin_loc_y = cv::Mat::zeros(row, col, CV_64FC1);
    // cv::Mat bin_loc_z = cv::Mat::zeros(row, col, CV_64FC1);

    for (int i = 0; i < row; i++)
    {
        int count  = 0; // for indexing ground range

        // first, fill the starboard side
        for (int j = col/2; j < col; j++)
        {
            bin_loc_x.at<double>(i,j) = pose.at<double>(i,3) + g_range[count]*cos(pose.at<double>(i,2)+PI/2);
            bin_loc_y.at<double>(i,j) = pose.at<double>(i,4) + g_range[count]*sin(pose.at<double>(i,2)+PI/2);
            count++;
        }
        // then the port side
        for (int j = 0; j < col/2; j++)
        {
            bin_loc_x.at<double>(i,j) = pose.at<double>(i,3) + g_range[count]*cos(pose.at<double>(i,2)-PI/2);
            bin_loc_y.at<double>(i,j) = pose.at<double>(i,4) + g_range[count]*sin(pose.at<double>(i,2)-PI/2);
            count--;
        }
    }

    // merge into one 3-channel matrix
    cv::Mat output;
    std::vector<cv::Mat> channels;

    channels.push_back(bin_loc_x);
    channels.push_back(bin_loc_y);
    // channels.push_back(bin_loc_z);

    // cv::merge(channels, output);

    return channels;
}

void Frame::DetectFeature(const cv::Mat &img, const cv::Mat &mask, std::vector<cv::KeyPoint> &kps, cv::Mat &dst)
{

    // cv::Ptr<SIFT> detector = SIFT::create(1000);
    // cv::Ptr<SiftDescriptorExtractor> descriptor = SiftDescriptorExtractor::create();
    // cv::Ptr<FeatureDetector> detector = cv::ORB::create(1000);
    // cv::Ptr<DescriptorExtractor> descriptor = cv::ORB::create();
    // detector->detect(img,kps,mask);
    // descriptor->compute(img, kps, dst);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    ORB_SLAM2::ORBextractor orb = ORB_SLAM2::ORBextractor(2000, 1.2, 6, 12, 7);
    orb(img, cv::Mat(), keypoints, descriptors);

    // mask out non-interested area
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        const int v = keypoints[i].pt.y;
        const int u = keypoints[i].pt.x;
        if (mask.at<bool>(v,u) == 1)
        {
            kps.push_back(keypoints[i]);
            dst.push_back(descriptors.row(i));
            // cout << keypoints[i].pt.y << " " << keypoints[i].pt.x << endl;
        }
        
    }
    
    // cv::Mat outimg;
    // cv::drawKeypoints(img, kps, outimg, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    // cv::imshow("Detected Features",outimg);
    // cv::waitKey(0); 

    return;
}


} // namespace Diasss
