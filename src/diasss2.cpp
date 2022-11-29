#include <math.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>

#include "util.h"
#include "frame.h"
#include "FEAmatcher.h"
#include "optimizer.h"
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

    // --- construct frame --- //
    int test_num = vmImgs.size();
    // int test_num = 3;
    std::vector<Frame> test_frames;
    for (size_t i = 0; i < test_num; i++)
        test_frames.push_back(Frame(i,vmImgs[i],vmPoses[i],vvAltts[i],vvGranges[i],vmAnnos[i]));

    // --- find correspondences between each pair of frames --- //
    for (size_t i = 0; i < test_frames.size(); i++)
        for (size_t j = i+1; j < test_frames.size(); j++)
            FEAmatcher::RobustMatching(test_frames[i],test_frames[j]);


    // --- optimize trajectory between images --- //
    Optimizer::TrajOptimizationPair(test_frames[0], test_frames[1]);
    // Optimizer::TrajOptimizationAll(test_frames);
    
    



    return 0;
}