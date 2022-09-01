#include "cxxopts.hpp"
#include <boost/filesystem.hpp>

#include <data_tools/std_data.h>
#include <bathy_maps/sss_meas_data.h>

#include <opencv2/highgui/highgui.hpp>
#include <cereal/archives/json.hpp>
#include <opencv2/core/eigen.hpp>
#include "opencv2/imgproc/imgproc.hpp"

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

  // get draping data from files, as "images" (imgs)
  sss_meas_data::ImagesT imgs;
  for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(folder), {})) 
  {
    std::cout << entry << "\n";
    auto img = read_data<sss_meas_data>(entry);
    cv::Mat waterfall_img, img_small;
    eigen2cv(img.sss_waterfall_image, waterfall_img);
    // cv::resize(waterfall_img, img_small, cv::Size(waterfall_img.cols/1.8,waterfall_img.rows/1.8));
    // cv::namedWindow("Waterfall image", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Waterfall image", waterfall_img);
    // cv::namedWindow("Resize Waterfall image", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Resize Waterfall image", img_small);
    // cv::waitKey();

    imgs.push_back(img);    
  }









  

    return 0;
}

