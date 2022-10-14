#include <math.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>

#include "util.h"
#include "frame.h"
#include "FEAmatcher.h"
#include "cxxopts.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Diasss;

#define PI 3.14159265

int main(int argc, char** argv)
{
  std::string strImageFolder, strPoseFolder, strAltitudeFolder, strGroundRangeFolder, strAnnotationFolder;

  // --- read input data paths --- //
  {
    cxxopts::Options options("data_parsing", "Reads input files...");
    options.add_options()
        ("help", "Print help")
        ("image", "Input folder containing sss image files", cxxopts::value(strImageFolder))
        ("pose", "Input folder containing auv pose files", cxxopts::value(strPoseFolder))
        ("altitude", "Input folder containing auv altitude files", cxxopts::value(strAltitudeFolder))
        ("groundrange", "Input folder containing ground range files", cxxopts::value(strGroundRangeFolder))
        ("annotation", "Input folder containing annotation files", cxxopts::value(strAnnotationFolder));

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
      cout << options.help({"", "Group"}) << endl;
      exit(0);
    }
    if (result.count("image") == 0) {
      cout << "Please provide folder containing sss image files..." << endl;
      exit(0);
    }
    if (result.count("pose") == 0) {
      cout << "Please provide folder containing auv poses files..." << endl;
      exit(0);
    }
    if (result.count("altitude") == 0) {
      cout << "Please provide folder containing auv altitude files..." << endl;
      exit(0);
    }
    if (result.count("groundrange") == 0) {
      cout << "Please provide folder containing ground range files..." << endl;
      exit(0);
    }
    if (result.count("annotation") == 0) {
      cout << "Please provide folder containing annotation files..." << endl;
      exit(0);
    }
  }

  // --- parse input data --- //
  std::vector<cv::Mat> vmImgs;
  std::vector<cv::Mat> vmPoses;
  std::vector<std::vector<double>> vvAltts;
  std::vector<std::vector<double>> vvGranges;
  std::vector<cv::Mat> vmAnnos;
  Util::LoadInputData(strImageFolder,strPoseFolder,strAltitudeFolder,strGroundRangeFolder,strAnnotationFolder,
                      vmImgs,vmPoses,vvAltts,vvGranges,vmAnnos);

  int f1 = 0, f2 = 1;
  Frame frame0 = Frame(f1,vmImgs[f1],vmPoses[f1],vvAltts[f1],vvGranges[f1]);
  Frame frame1 = Frame(f2,vmImgs[f2],vmPoses[f2],vvAltts[f2],vvGranges[f2]);

  std::vector<pair<size_t, size_t> > vkpCorres = FEAmatcher::RobustMatching(frame0,frame1);

  // cv::Mat outimg_1, outimg_2;
  // cv::drawKeypoints(frame0.norm_img, frame0.kps, outimg_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  // cv::imshow("Detected Features 1",outimg_1);
  // cv::drawKeypoints(frame1.norm_img, frame1.kps, outimg_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  // cv::imshow("Detected Features 2",outimg_2);
  // cv::waitKey(0); 

  // cv::Mat flt_mask = Util::GetFilterMask(vmImgs[0]);
  // cv::Mat img_1 = Util::NormalizeSSS(vmImgs[0]);
  // cv::Mat img_2 = Util::NormalizeSSS(vmImgs[2]);

  // --- detect features --- //
  // Eigen::MatrixXd img_1_tmp;
  // cv::cv2eigen(vmImgs[0], img_1_tmp);
  // cv::Mat img_1 = Util::NormalizeConvertSSS(img_1_tmp);

  // std::vector<cv::KeyPoint> keypoints_1,keypoints_2;
  // cv::Mat descriptors_1,descriptors_2;
  // cv::Ptr<SIFT> detector = SIFT::create(200);
  // cv::Ptr<SiftDescriptorExtractor> descriptor = SiftDescriptorExtractor::create();
  // // cv::Ptr<FeatureDetector> detector = cv::ORB::create(100);
  // // cv::Ptr<DescriptorExtractor> descriptor = cv::ORB::create();
  // detector->detect(img_1,keypoints_1);
  // detector->detect(img_2,keypoints_2);
  // descriptor->compute(img_1, keypoints_1, descriptors_1);
  // descriptor->compute(img_2, keypoints_2, descriptors_2);

  // cv::Mat outimg_1, outimg_2;
  // cv::drawKeypoints(img_1, keypoints_1, outimg_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  // cv::imshow("Detected Features 1",outimg_1);
  // cv::drawKeypoints(img_2, keypoints_2, outimg_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  // cv::imshow("Detected Features 2",outimg_2);
  // cv::waitKey(0); 

	// cv::BFMatcher matcher(NORM_L2);
	// std::vector<cv::DMatch> matches;
	// matcher.match(descriptors_1, descriptors_2, matches);

	// // get max and min distance
	// double max_dist = 0;
	// double min_dist = 1000;
	// for (int i = 1; i < descriptors_1.rows; ++i)
	// {
	// 	double dist = matches[i].distance;
	// 	if (dist > max_dist)
	// 		max_dist = dist;
	// 	if (dist < min_dist)
	// 		min_dist = dist;
	// }
	// cout << "min_dist=" << min_dist << endl;
	// cout << "max_dist=" << max_dist << endl;

	// // get good matches    
	// vector<DMatch> goodMatches;
	// for (int i = 0; i < matches.size(); ++i)
	// {
	// 	double dist = matches[i].distance;
	// 	if (dist < 2 * min_dist)
	// 		goodMatches.push_back(matches[i]);
	// }
	// cout << "goodMatches:" << goodMatches.size() << endl;

  // // draw the matching result
	// cv::Mat result;
	// cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, goodMatches, result,
  //                 Scalar(255, 255, 0), Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	// cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
	// imshow("temperal matches", result);
	// waitKey(0);


  // // --- extract patch pairs for test --- //

  // // get patch locations (x_min,y_min,width,height)
  // std::vector<int> pat_loc_1{695,1410,60,200};
  // std::vector<int> pat_loc_2{400,1470,60,200}; 

  // // initialize the patch to frame class
  // cv::Mat img_pat1 = vmImgs[0](cv::Rect(pat_loc_1[0],pat_loc_1[1],pat_loc_1[2],pat_loc_1[3]));
  // cv::Mat pose_pat1 = vmPoses[0].rowRange(pat_loc_1[1],pat_loc_1[1]+pat_loc_1[3]);
  // std::vector<double> altt_pat1(vvAltts[0].begin() + pat_loc_1[1], vvAltts[0].begin() + pat_loc_1[1]+pat_loc_1[3]);
  // Frame frame1 = Frame(img_pat1,pose_pat1,altt_pat1,vvGranges[0]);
  // cv::Mat img_pat2 = vmImgs[2](cv::Rect(pat_loc_2[0],pat_loc_2[1],pat_loc_2[2],pat_loc_2[3]));
  // cv::Mat pose_pat2 = vmPoses[2].rowRange(pat_loc_2[1],pat_loc_2[1]+pat_loc_2[3]);
  // std::vector<double> altt_pat2(vvAltts[0].begin() + pat_loc_2[1], vvAltts[0].begin() + pat_loc_2[1]+pat_loc_2[3]);
  // Frame frame2 = Frame(img_pat2,pose_pat2,altt_pat2,vvGranges[2]);

  // // --- mosaicking process --- //

  // // cv::Mat aa = vmImgs[0].rowRange(170,1700), bb = vmPoses[0].rowRange(170,1700);
  // cv::Mat aa = vmImgs[0], bb = vmPoses[0];
  // pcl::PointCloud<pcl::PointXYZI>::Ptr mosa = Util::ImgMosaic(aa,bb,vvGranges[0]);
  // pcl::PointCloud<pcl::PointXYZI>::Ptr mosa_2 = Util::ImgMosaic(vmImgs[1],vmPoses[1],vvGranges[1]);
  // int num = mosa_2->points.size();
  // // for (size_t i = 0; i < num; i++)
  // //   mosa_2->points[i].y = mosa_2->points[i].y + 350;
  

  // // --- visualize point cloud --- //
  // pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("Side-scan Image Mosaicking Result"));
  // viewer->setBackgroundColor (0.2, 0.2, 0.2);
  // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> point_cloud_color_handler(mosa, "intensity");
  // viewer->addPointCloud<pcl::PointXYZI> (mosa, point_cloud_color_handler, "ssi cloud");
  // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "ssi cloud");
  // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> point_cloud_color_handler_2(mosa_2, "intensity");
  // viewer->addPointCloud<pcl::PointXYZI> (mosa_2, point_cloud_color_handler_2, "ssi-2 cloud");
  // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "ssi-2 cloud");
  // viewer->addCoordinateSystem (50.0);
  // viewer->initCameraParameters ();
  // while (!viewer->wasStopped ())
  // {
  //   viewer->spinOnce (100);
  //   boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  // }





  // --- detect high gredient area (line and curve) --- //

  // Eigen::MatrixXd pat2_tmp;
  // cv::cv2eigen(img_pat2, pat2_tmp);
  // cv::Mat wf_pat2 = Util::NormalizeConvertSSS(pat2_tmp);
  // cv::namedWindow("Waterfall image 2", cv::WINDOW_AUTOSIZE);
  // cv::imshow("Waterfall image 2", wf_pat2); 

  // cv::Mat wf_pat2_sm;
  // cv::bilateralFilter(wf_pat2, wf_pat2_sm, 7, 75, 75, BORDER_DEFAULT);
  // cv::namedWindow("smoothed image", cv::WINDOW_AUTOSIZE);
  // cv::imshow("smoothed image", wf_pat2_sm);

  // int lowThreshold = 25;
  // const int ratio = 3;
  // const int kernel_size_ca = 3;

  // cv::Mat edge_result, black_mask;
  // cv::Canny(wf_pat2_sm, edge_result, lowThreshold, lowThreshold*ratio, kernel_size_ca, true);
  // black_mask = Scalar::all(0);
  // edge_result.copyTo(edge_result, black_mask);
  // cv::namedWindow("Edge Map", cv::WINDOW_AUTOSIZE);
  // imshow("Edge Map", edge_result);
  // cv::waitKey(0);















  return 0;
}