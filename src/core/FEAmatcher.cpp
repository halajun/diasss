
#include "FEAmatcher.h"


namespace Diasss
{

using namespace std;
using namespace cv;

#define PI 3.14159265359

std::vector<std::pair<size_t, size_t>> FEAmatcher::RobustMatching(Frame &SourceFrame, Frame &TargetFrame)
{

    std::vector<std::pair<size_t, size_t> > CorresPairs;

    std::vector<cv::KeyPoint> kps_1 = SourceFrame.kps;
    std::vector<cv::KeyPoint> kps_2 = TargetFrame.kps;
    cv::Mat dst_1 = SourceFrame.dst;
    cv::Mat dst_2 = TargetFrame.dst;
    cv::Mat geo_img_1 = SourceFrame.geo_img;
    cv::Mat geo_img_2 = TargetFrame.geo_img;

    std::vector<int> CorresID_1 = FEAmatcher::GeoNearNeighSearch(SourceFrame.norm_img,TargetFrame.norm_img,kps_1,dst_1,geo_img_1,kps_2,dst_2,geo_img_2);
    std::vector<int> CorresID_2 = FEAmatcher::GeoNearNeighSearch(TargetFrame.norm_img,SourceFrame.norm_img,kps_2,dst_2,geo_img_2,kps_1,dst_1,geo_img_1);

    // --- cross-check --- //
    std::vector<cv::KeyPoint> PreKeys, CurKeys;
    std::vector<cv::DMatch> TemperalMatches;
    int count = 0;
    for (size_t i = 0; i < CorresID_1.size(); i++)
    {
        if (CorresID_1[i]==-1)
            continue;

        if (CorresID_2[CorresID_1[i]]==i)
        {
            PreKeys.push_back(kps_1[i]);
            CurKeys.push_back(kps_2[CorresID_1[i]]);
            TemperalMatches.push_back(cv::DMatch(count,count,0));
            count = count + 1;
        }     
    }

    // --- ratio test --- //


    // --- demonstrate --- //
    cv::Mat img_matches;
    cv::drawMatches(SourceFrame.norm_img, PreKeys, TargetFrame.norm_img, CurKeys,
                TemperalMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::namedWindow("temperal matches", cv::WINDOW_NORMAL);
    cv::imshow("temperal matches", img_matches);
    cv::waitKey(0);


    return CorresPairs;

}

std::vector<int> FEAmatcher::GeoNearNeighSearch(const cv::Mat &img, const cv::Mat &img_ref,
                                                const std::vector<cv::KeyPoint> &kps, const cv::Mat &dst, const cv::Mat &geo_img,
                                                const std::vector<cv::KeyPoint> &kps_ref, const cv::Mat &dst_ref, const cv::Mat &geo_img_ref)
{

    std::vector<int> CorresID = std::vector<int>(kps.size(),-1);

    // --- some parameters --- //
    int radius = 12; // seach circle size
    int x_min, y_min, x_max, y_max; // rectangle of seach window
    double bx_min, bx_max, by_min, by_max; // geometric border of reference image

    // get boundary of the reference geo-image
    cv::Mat channels_ref[3];
    cv::split(geo_img_ref, channels_ref);
    cv::minMaxLoc(channels_ref[0], &bx_min, &bx_max);
    cv::minMaxLoc(channels_ref[1], &by_min, &by_max);

    // cout << "boundary: " << bx_min << " " << bx_max << " " << by_min << " " << by_max << endl;

    // --- main loop --- //
    vector<int> candidate;
    std::vector<cv::KeyPoint> kps_show, kps_show_ref;
    cv::Mat channels[3];
    cv::split(geo_img, channels);
    for (size_t i = 0; i < kps.size(); i++)
    {
        double loc_x = channels[0].at<double>(kps[i].pt.y,kps[i].pt.x);
        double loc_y = channels[1].at<double>(kps[i].pt.y,kps[i].pt.x);

        if (loc_x<bx_min || loc_y<by_min || loc_x>bx_max || loc_y>by_max)
            continue;
        
        for (size_t j = 0; j < kps_ref.size(); j++)
        {
            double ref_loc_x = channels_ref[0].at<double>(kps_ref[j].pt.y,kps_ref[j].pt.x);
            double ref_loc_y = channels_ref[1].at<double>(kps_ref[j].pt.y,kps_ref[j].pt.x);

            double geo_dist = sqrt( (loc_x-ref_loc_x)*(loc_x-ref_loc_x) + (loc_y-ref_loc_y)*(loc_y-ref_loc_y) );
            if (geo_dist<radius)
            {
                candidate.push_back(j);
                kps_show_ref.push_back(kps_ref[j]); 
            }          
        }

        // cout << "candidate size: " << candidate.size() << endl;

        if(!candidate.size())
            continue;

        // --- using SIFT --- //
        {
            double best_dist = 1000, sec_best_dist = 1000, dist_bound = 100;
            int best_id = -1;
            double ratio_test = 0.3;
            for (size_t j = 0; j < candidate.size(); j++)
            {
                const double dst_dist = cv::norm(dst.row(i),dst_ref.row(candidate[j]),cv::NORM_L2);
                // cout << "dist: " << dst_dist << endl;
                if (dst_dist<best_dist)
                {
                    sec_best_dist = best_dist;
                    best_dist = dst_dist;
                    best_id = candidate[j];
                }
                else if (dst_dist<sec_best_dist)
                {
                  sec_best_dist = dst_dist;
                }
                
            }
            double fir_sec_ratio = best_dist/sec_best_dist;
            // cout << "best and second best ratio: " << fir_sec_ratio << endl;
            if (best_id!=-1 && best_dist<dist_bound && fir_sec_ratio<=ratio_test)
                CorresID[i] = best_id;
            else if (candidate.size()==1 && best_dist<dist_bound)
                CorresID[i] = best_id;
        }


        // // --- using ORB --- //
        // {
        //     int best_dist = 1000, sec_best_dist = 1000, dist_bound = 100;
        //     int best_id = -1;
        //     double ratio_test = 0.8;
        //     for (size_t j = 0; j < candidate.size(); j++)
        //     {
        //         const int dst_dist = FEAmatcher::DescriptorDistance(dst.row(i),dst_ref.row(candidate[j]));
        //         cout << "dist: " << dst_dist << endl;
        //         if (dst_dist<best_dist)
        //         {
        //             sec_best_dist = best_dist;
        //             best_dist = dst_dist;
        //             best_id = candidate[j];
        //         }
        //         else if (dst_dist<sec_best_dist)
        //         {
        //           sec_best_dist = dst_dist;
        //         }
                
        //     }
        //     double fir_sec_ratio = (double)best_dist/sec_best_dist;
        //     cout << "best and second best ratio: " << fir_sec_ratio << endl;
        //     if (best_id!=-1 && best_dist<dist_bound && fir_sec_ratio<=ratio_test)
        //         CorresID[i] = best_id;
        //     else if (candidate.size()==1 && best_dist<dist_bound)
        //         CorresID[i] = best_id;
        // }
            

        // if(candidate.size()>3)
        // {
        //     cout << "candidate size: " << candidate.size() << endl;
        //     cv::Mat outimg_1, outimg_2;
        //     kps_show.push_back(kps[i]);
        //     cv::drawKeypoints(img, kps_show, outimg_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        //     cv::imshow("Detected Features 1",outimg_1);
        //     cv::drawKeypoints(img_ref, kps_show_ref, outimg_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        //     cv::imshow("Detected Features 2",outimg_2);
        //     cv::waitKey(0);
        // }
      
        candidate.clear();
        kps_show.clear();
        kps_show_ref.clear();

    }
  
    return CorresID;
}

// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int FEAmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}


} // namespace Diasss
