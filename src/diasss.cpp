#include <math.h>

#include "cxxopts.hpp"
#include <boost/filesystem.hpp>

#include "util.h"

#include <data_tools/std_data.h>
#include <bathy_maps/sss_meas_data.h>

#include <opencv2/highgui/highgui.hpp>
#include <cereal/archives/json.hpp>
#include <opencv2/core/eigen.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ximgproc/edge_filter.hpp"
#include <opencv2/photo.hpp>

#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace std_data;

int main(int argc, char** argv)
{
  string folder_str;

	cxxopts::Options options("data_parsing", "Reads draping cereal files...");
	options.add_options()
      ("help", "Print help")
      ("folder", "Input folder containing cereal files", cxxopts::value(folder_str));

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
          cout << options.help({"", "Group"}) << endl;
          exit(0);
    }
    if (result.count("folder") == 0) {
		cout << "Please provide folder containing cereal files..." << endl;
		exit(0);
    }
	
	boost::filesystem::path folder(folder_str);

	cout << "Input cereal folder : " << folder << endl;

  // load draping data from files, as "images" (imgs)
  sss_meas_data::ImagesT imgs;
  for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(folder), {})) 
  {
    std::cout << entry << "\n";
    auto img = read_data<sss_meas_data>(entry);
    cv::Mat waterfall_img, waterfall_img_uc, waterfall_img_grey, waterfall_img_gf, wf_img_sm;
    waterfall_img_uc = diasss::NormalizeConvertSSS(img.sss_waterfall_image);
    // cv::eigen2cv(img.sss_waterfall_image, waterfall_img);
    // waterfall_img.convertTo(waterfall_img_uc, CV_8U);
    cv::namedWindow("Waterfall image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Waterfall image", waterfall_img_uc);
    // cv::imwrite("../waterfall_img.png", waterfall_img_grey);
    // cv::waitKey();

    // -------- smoothing ---------
    // cv::GaussianBlur(waterfall_img_uc, wf_img_sm, Size(5, 5), 1.4, 1.4, BORDER_DEFAULT);
    // cv::ximgproc::guidedFilter(waterfall_img_uc, waterfall_img_uc, wf_img_sm, 2, pow(0.5,2), -1);
    cv::bilateralFilter(waterfall_img_uc, wf_img_sm, 10, 75, 75, BORDER_DEFAULT);
    // cv::fastNlMeansDenoising(waterfall_img_uc, wf_img_sm);
    cv::namedWindow("smoothed image", cv::WINDOW_AUTOSIZE);
    cv::imshow("smoothed image", wf_img_sm);
    // cv::waitKey();

    // // ---------- Laplacian detector -------------
    // int kernel_size = 5;
    // int scale = 1;
    // int delta = 0;
    // int ddepth = CV_16S;

    // cv::Mat wf_img_log, wf_img_log_abs;
    // cv::Laplacian(wf_img_sm, wf_img_log, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
    // cv::convertScaleAbs(wf_img_log, wf_img_log_abs);
    // cv::namedWindow("LoG image", cv::WINDOW_AUTOSIZE);
    // cv::imshow("LoG image", wf_img_log_abs);
    // cv::waitKey();

    // --------- Canny detector ---------
    int lowThreshold = 28;
    const int ratio = 3;
    const int kernel_size_ca = 3;
    const char* window_name = "Edge Map";

    cv::Mat edge_result, black_mask;
    cv::Canny(wf_img_sm, edge_result, lowThreshold, lowThreshold*ratio, kernel_size_ca, true);
    black_mask = Scalar::all(0);
    edge_result.copyTo(edge_result, black_mask);
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    imshow(window_name, edge_result);
    cv::waitKey(0);
    


    imgs.push_back(img);    
  }

    return 0;
}