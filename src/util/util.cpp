
#include "util.h"


namespace Diasss
{

using namespace std;
using namespace cv;

#define PI 3.14159265359

void Util::LoadInputData(const std::string &strImageFolder, const std::string &strPoseFolder, const std::string &strAltitudeFolder, const std::string &strGroundRangeFolder, const std::string &strAnnotationFolder,
                         std::vector<cv::Mat> &vmImgs, std::vector<cv::Mat> &vmPoses, std::vector<std::vector<double>> &vvAltts, std::vector<std::vector<double>> &vvGranges, std::vector<cv::Mat> &vmAnnos)
{
    // get data paths, ordered by name
    boost::filesystem::path path_img(strImageFolder);
    std::vector<boost::filesystem::path> path_img_ordered(
          boost::filesystem::directory_iterator(path_img)
          , boost::filesystem::directory_iterator{}
        );
    std::sort(path_img_ordered.begin(), path_img_ordered.end());

    boost::filesystem::path path_pose(strPoseFolder);
    std::vector<boost::filesystem::path> path_pose_ordered(
          boost::filesystem::directory_iterator(path_pose)
          , boost::filesystem::directory_iterator{}
        );
    std::sort(path_pose_ordered.begin(), path_pose_ordered.end());

    boost::filesystem::path path_altitude(strAltitudeFolder);
    std::vector<boost::filesystem::path> path_altitude_ordered(
          boost::filesystem::directory_iterator(path_altitude)
          , boost::filesystem::directory_iterator{}
        );
    std::sort(path_altitude_ordered.begin(), path_altitude_ordered.end());

    boost::filesystem::path path_groundrange(strGroundRangeFolder);
    std::vector<boost::filesystem::path> path_groundrange_ordered(
          boost::filesystem::directory_iterator(path_groundrange)
          , boost::filesystem::directory_iterator{}
        );
    std::sort(path_groundrange_ordered.begin(), path_groundrange_ordered.end());

    boost::filesystem::path path_annotation(strAnnotationFolder);
    std::vector<boost::filesystem::path> path_annotation_ordered(
          boost::filesystem::directory_iterator(path_annotation)
          , boost::filesystem::directory_iterator{}
        );
    std::sort(path_annotation_ordered.begin(), path_annotation_ordered.end());


    // get input sss images
    for(auto const& path : path_img_ordered)
    {
    //   std::cout << path << '\n';

      cv::Mat img_tmp;
      cv::FileStorage fs(path.string(),cv::FileStorage::READ);
      fs["ct_img"] >> img_tmp;
      fs.release();
      vmImgs.push_back(img_tmp);
      cout << "image size: "<< img_tmp.rows << " " << img_tmp.cols << endl;
      // cout.precision(10);
      // for (int i = 0; i < 100; i++)
      // {
      //     for (int j = 0; j < 100; j++)
      //     {
      //       cout << img_tmp.at<double>(i,j) << " ";
      //     }
      //     cout << endl;
      // }

    }

    // get input Dead-reckoning poses
    for(auto const& path : path_pose_ordered)
    {
    //   std::cout << path << '\n';

      cv::Mat pose_tmp;
      cv::FileStorage fs(path.string(),cv::FileStorage::READ);
      fs["auv_pose"] >> pose_tmp;
      fs.release();
      vmPoses.push_back(pose_tmp);
      cout << "pose size: " << pose_tmp.rows << " " << pose_tmp.cols << endl;
      // cout.precision(10);
      // for (int i = 0; i < 100; i++)
      // {
      //     for (int j = 0; j < 100; j++)
      //     {
      //       cout << pose_tmp.at<double>(i,j) << " ";
      //     }
      //     cout << endl;
      // }

    }

    // get auv altitude
    for(auto const& path : path_altitude_ordered)
    {
    //   std::cout << path << '\n';

      ifstream fAltt;
      fAltt.open(path.string().c_str());
      std::vector<double> vAltt;
      while(!fAltt.eof())
      {
          string s;
          getline(fAltt,s);
          if(!s.empty())
          {
              stringstream ss;
              ss << s;
              double altt_tmp;
              ss >> altt_tmp;
              vAltt.push_back(altt_tmp);
          }
      }
      fAltt.close();
      vvAltts.push_back(vAltt);
      cout << "alttitude size: " << vAltt.size() << endl;
      // for (double x : vAltt)
      //   cout << x << " ";
      // cout << endl;
    }

    // get ground range
    for(auto const& path : path_groundrange_ordered)
    {
    //   std::cout << path << '\n';

      ifstream fGRange;
      fGRange.open(path.string().c_str());
      std::vector<double> vGrange;
      while(!fGRange.eof())
      {
          string s;
          getline(fGRange,s);
          if(!s.empty())
          {
              stringstream ss;
              ss << s;
              double gr_tmp;
              ss >> gr_tmp;
              vGrange.push_back(gr_tmp);
          }
      }
      fGRange.close();
      vvGranges.push_back(vGrange);
      cout << "ground range size: " << vGrange.size() << endl;
      // for (double x : vGrange)
      //   cout << x << " ";
      // cout << endl;
    }

    // get input annotations of images
    for(auto const& path : path_annotation_ordered)
    {
    //   std::cout << path << '\n';

      cv::Mat anno_tmp;
      cv::FileStorage fs(path.string(),cv::FileStorage::READ);
      fs["anno_kps"] >> anno_tmp;
      fs.release();
      vmAnnos.push_back(anno_tmp);
      cout << "annotation size: "<< anno_tmp.rows << " " << anno_tmp.cols << endl;
      // cout.precision(10);
      // for (int i = 0; i < 100; i++)
      // {
      //     for (int j = 0; j < 100; j++)
      //     {
      //       cout << img_tmp.at<double>(i,j) << " ";
      //     }
      //     cout << endl;
      // }

    }

    return;                   
}

void Util::ShowAnnos(int &f1, int &f2, cv::Mat &img1, cv::Mat &img2, const cv::Mat &anno1, const cv::Mat &anno2)
{

    bool use_anno = 0;

    // --- load annotated keypoints --- //
    std::vector<cv::KeyPoint> PreKeys, CurKeys;
    std::vector<cv::DMatch> TemperalMatches;
    int count = 0;
    // --- from img1 to img2 ....... //
    for (size_t i = 0; i < anno1.rows; i++)
    {
        if (use_anno)
        {
            if (anno1.at<int>(i,1)==f2 && i%15==0)
            {
                PreKeys.push_back(cv::KeyPoint(anno1.at<int>(i,3),anno1.at<int>(i,2),0,0,0,-1));
                CurKeys.push_back(cv::KeyPoint(anno1.at<int>(i,5),anno1.at<int>(i,4),0,0,0,-1));
                TemperalMatches.push_back(cv::DMatch(count,count,0));
                count = count + 1;
                // cout << anno1.at<int>(i,2) << " " << anno1.at<int>(i,3) << " " << anno1.at<int>(i,4) << " " << anno1.at<int>(i,5) << endl;
            }  
        }
        else
        {
            if (anno1.at<double>(i,1)==f2)
            {
                PreKeys.push_back(cv::KeyPoint(anno1.at<double>(i,3),anno1.at<double>(i,2),0,0,0,-1));
                CurKeys.push_back(cv::KeyPoint(anno1.at<double>(i,5),anno1.at<double>(i,4),0,0,0,-1));
                TemperalMatches.push_back(cv::DMatch(count,count,0));
                count = count + 1;
                // cout << anno1.at<int>(i,2) << " " << anno1.at<int>(i,3) << " " << anno1.at<int>(i,4) << " " << anno1.at<int>(i,5) << endl;
            } 
        }
        
           
    }
    // // --- from img2 to img1 ....... //
    // for (size_t i = 0; i < anno2.rows; i++)
    // {
    //     if (anno2.at<int>(i,1)==f1)
    //     {
    //         PreKeys.push_back(cv::KeyPoint(anno2.at<int>(i,5),anno2.at<int>(i,4),0,0,0,-1));
    //         CurKeys.push_back(cv::KeyPoint(anno2.at<int>(i,3),anno2.at<int>(i,2),0,0,0,-1));
    //         TemperalMatches.push_back(cv::DMatch(count,count,0));
    //         count = count + 1;
    //     }     
    // }

    // cout << "number of matched keypoints: " << count << endl;

    // --- demonstrate --- //
    cv::Mat img_matches;
    cv::drawMatches(img1, PreKeys, img2, CurKeys, TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
    cv::imshow("temperal matches", img_matches);
    cv::waitKey(0);

    return;
}

cv::Mat Util::GetFilterMask(cv::Mat &sss_raw_img)
{
    float factor = 2.5;
    int width = 2, side = 100;

    cv::Mat output_mask = cv::Mat::ones(sss_raw_img.size(), CV_8U);

    cv::Scalar MeanofMat = cv::mean(sss_raw_img);
    // cout << "mean: " << MeanofMat[0] << endl;

    for (int i = 0; i < output_mask.rows; i++)
    {
        for (int j = 0; j < output_mask.cols; j++)
        {
            // remove sensor buggy line
            if (sss_raw_img.at<double>(i,j)>MeanofMat[0]*factor)
                output_mask.at<bool>(i,j) = 0;
            // remove centre line
            if (j>output_mask.cols/2-width && j<output_mask.cols/2+width)
                output_mask.at<bool>(i,j) = 0;
            // remove the first and last turning pings
            if (i<side || i>output_mask.rows-side)
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

cv::Mat Util::NormalizeSSS(cv::Mat &sss_raw_img)
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

cv::Mat Util::NormalizeConvertSSS(Eigen::MatrixXd &sss_wf_img)
{
    bool rs_by_column = 1, clip = 1;
    cv::Mat output_img, intermediate_img;

    std::cout << "image shape: " << sss_wf_img.cols() << " " << sss_wf_img.rows() << std::endl;

    // normalize by column
    for (int i = 0; i < sss_wf_img.cols(); i++)
    {
        // sss_wf_img.col(i).normalize();
        sss_wf_img.col(i) = sss_wf_img.col(i)/sss_wf_img.col(i).mean();
    }

    // rescale to 0-255
    Eigen::MatrixXd img_rescaled;
    if (clip)
    {
        // if rescale
        bool rs = 1;
        // clip to range ~(l,u)
        float l = 0, u = 3;
        img_rescaled = Eigen::MatrixXd::Zero(sss_wf_img.rows(),sss_wf_img.cols());

        for (size_t i = 0; i < sss_wf_img.rows(); i++)
        {
            for (size_t j = 0; j < sss_wf_img.cols(); j++)
            {
                if (sss_wf_img(i,j)<0)
                    img_rescaled(i,j) = 0;
                else if (sss_wf_img(i,j)>3)
                    img_rescaled(i,j) = 3;  
                else
                    img_rescaled(i,j) = sss_wf_img(i,j);
            }
            
        }
        if (rs)
        {
            float l = 0, u = 255;
            float min = img_rescaled.minCoeff();
            float max = img_rescaled.maxCoeff();
            img_rescaled = l + (img_rescaled.array() - min) * ((u - l) / (max - min));
        }
    }
    else if (rs_by_column)
    {
        // by column
        float l = 0, u = 255;

        Eigen::ArrayXd  min = sss_wf_img.colwise().minCoeff();
        Eigen::ArrayXd  max = sss_wf_img.colwise().maxCoeff();
        img_rescaled = Eigen::MatrixXd::Zero(sss_wf_img.rows(),sss_wf_img.cols());

        for (size_t i = 0; i < sss_wf_img.rows(); i++)
        {
            for (size_t j = 0; j < sss_wf_img.cols(); j++)
            {
                img_rescaled(i,j) = l + (sss_wf_img(i,j) - min(j)) * ((u - l) / (max(j) - min(j)));
            }
            
        }
    }
    else
    {
        // by whole matrix
        float l = 0, u = 255;
        float min = sss_wf_img.minCoeff();
        float max = sss_wf_img.maxCoeff();
        img_rescaled = l + (sss_wf_img.array() - min) * ((u - l) / (max - min));
    }
        
    // convert (8 bit uchar grey image, CV_8U)
    cv::eigen2cv(img_rescaled, intermediate_img);
    intermediate_img.convertTo(output_img, CV_8U); 

    // return intermediate_img;
    return output_img;
}

// pcl::PointCloud<pcl::PointXYZI>::Ptr Util::ImgMosaicOld(std::vector<cv::Mat> &coords, cv::Mat &img)
// {
    
//     // --- process the intensity image --- //
//     Eigen::MatrixXd img_norm, img_clip;
//     cv::cv2eigen(img, img_norm);
//     // normalize by column
//     for (int i = 0; i < img_norm.cols(); i++)
//         img_norm.col(i) = img_norm.col(i)/img_norm.col(i).mean();
//     // clip to (l,u)
//     float l = 0, u = 3;
//     img_clip = Eigen::MatrixXd::Zero(img_norm.rows(),img_norm.cols());
//     for (size_t i = 0; i < img_norm.rows(); i++)
//     {
//         for (size_t j = 0; j < img_norm.cols(); j++)
//         {
//             if (img_norm(i,j)<l)
//                 img_clip(i,j) = l;
//             else if (img_norm(i,j)>u)
//                 img_clip(i,j) = u;  
//             else
//                 img_clip(i,j) = img_norm(i,j);
//         }    
//     }

//     // --- fill cloud --- //
//     pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
//     int row = coords[0].rows, col = coords[0].cols;
//     for (int i=0;i<row;i++)
//     {
//         for (int j=0;j<col;j++)
//         {
//             pcl::PointXYZI point;
//             point.x = coords[0].at<double>(i,j);
//             point.y = coords[1].at<double>(i,j);
//             point.z = coords[2].at<double>(i,j);
//             point.intensity = img_clip(i,j);

//             cloud->points.push_back(point);
//         }
//     }   

//     return cloud;
// }

// pcl::PointCloud<pcl::PointXYZI>::Ptr Util::ImgMosaic(cv::Mat &img, cv::Mat &pose, std::vector<double> &g_range)
// {

//     // (1) --- process the intensity image --- //

//     Eigen::MatrixXd img_norm, img_clip;
//     cv::cv2eigen(img, img_norm);
//     // normalize by column
//     for (int i = 0; i < img_norm.cols(); i++)
//         img_norm.col(i) = img_norm.col(i)/img_norm.col(i).mean();
//     // clip to (l,u)
//     float l = 0, u = 3;
//     img_clip = Eigen::MatrixXd::Zero(img_norm.rows(),img_norm.cols());
//     for (size_t i = 0; i < img_norm.rows(); i++)
//     {
//         for (size_t j = 0; j < img_norm.cols(); j++)
//         {
//             if (img_norm(i,j)<l)
//                 img_clip(i,j) = l;
//             else if (img_norm(i,j)>u)
//                 img_clip(i,j) = u;  
//             else
//                 img_clip(i,j) = img_norm(i,j);
//         }    
//     }
//     // rescale to (rs_l,rs_u)
//     float rs_l = 0, rs_u = 255;
//     float min = img_clip.minCoeff();
//     float max = img_clip.maxCoeff();
//     img_clip = rs_l + (img_clip.array() - min) * ((rs_u - rs_l) / (max - min));


//     // (2) --- get geo-referenced location of the image --- //

//     // get bin locations of image
//     cv::Mat bin_loc_x = cv::Mat::zeros(img.size(), CV_64FC1);
//     cv::Mat bin_loc_y = cv::Mat::zeros(img.size(), CV_64FC1);
//     cv::Mat bin_loc_z = cv::Mat::zeros(img.size(), CV_64FC1);

//     for (int i = 0; i < img.rows; i++)
//     {
//         int count  = 0; // for indexing ground range

//         // first, fill the starboard side
//         for (int j = img.cols/2; j < img.cols; j++)
//         {
//             bin_loc_x.at<double>(i,j) = pose.at<double>(i,3) + g_range[count]*cos(pose.at<double>(i,2)+PI/2);
//             bin_loc_y.at<double>(i,j) = pose.at<double>(i,4) + g_range[count]*sin(pose.at<double>(i,2)+PI/2);
//             count++;
//         }
//         // then the port side
//         for (int j = 0; j < img.cols/2; j++)
//         {
//             bin_loc_x.at<double>(i,j) = pose.at<double>(i,3) + g_range[count]*cos(pose.at<double>(i,2)-PI/2);
//             bin_loc_y.at<double>(i,j) = pose.at<double>(i,4) + g_range[count]*sin(pose.at<double>(i,2)-PI/2);
//             count--;
//         }
//     }

//     // (3) --- fill cloud --- //
//     pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
//     int row = bin_loc_x.rows, col = bin_loc_x.cols;
//     for (int i=0;i<row;i++)
//     {
//         for (int j=0;j<col;j++)
//         {
//             pcl::PointXYZI point;

//             point.x = bin_loc_x.at<double>(i,j);
//             point.y = bin_loc_y.at<double>(i,j);
//             point.z = bin_loc_z.at<double>(i,j);
//             // point.z = img_clip(i,j);

//             point.intensity = img_clip(i,j);

//             cloud->points.push_back(point);
//         }
//     }

//     // // draw 3d point cloud (old method)
//     // viz::Viz3d window;
//     // window.showWidget("points", viz::WCloud(mMosa_1, viz::Color::white()));
//     // window.spin();

//     // pcl::PointCloud<pcl::PointXYZI>::Ptr mosa = Util::ImgMosaicOld(vm3ds, vmImgs[0]);
//     // pcl::visualization::CloudViewer viewer ("Side-scan Image Mosaicking Result");
//     // viewer.showCloud (mosa);
//     // viewer.addCoordinateSystem (1.0);
//     // while (!viewer.wasStopped ())
//     // {
//     // }   

//     return cloud;
// }

} // namespace Diasss
