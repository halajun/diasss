
#include "FEAmatcher.h"
#include <bits/stdc++.h>

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
    std::vector<std::pair<int,double>> scc_1, scc_2;

    std::vector<int> CorresID_1 = FEAmatcher::GeoNearNeighSearch(SourceFrame.img_id,TargetFrame.img_id,SourceFrame.norm_img,TargetFrame.norm_img,
                                                                 kps_1,dst_1,geo_img_1,kps_2,dst_2,geo_img_2, scc_1);
    std::vector<int> CorresID_2 = FEAmatcher::GeoNearNeighSearch(TargetFrame.img_id,SourceFrame.img_id,TargetFrame.norm_img,SourceFrame.norm_img,
                                                                 kps_2,dst_2,geo_img_2,kps_1,dst_1,geo_img_1, scc_2);

    std::sort(scc_1.rbegin(), scc_1.rend());
    std::sort(scc_2.rbegin(), scc_2.rend());
    for (size_t i = 0; i < 5; i++)
    {
        if (scc_1[i].first!=-1)       
            cout << "scc_1: " << scc_1[i].first << " " << scc_1[i].second+abs(SourceFrame.norm_img.rows-TargetFrame.norm_img.rows) << endl;
    }
    for (size_t i = 0; i < 5; i++)
    {
        if (scc_2[i].first!=-1)       
            cout << "scc_2: " << scc_2[i].first << " " << scc_2[i].second << endl;
    }
    
    
    std::vector<cv::KeyPoint> PreKeys, CurKeys;
    std::vector<cv::DMatch> TemperalMatches;
    int count = 0;
    // --- merge if the sliding compatibility check is consistent --- //
    if (abs(scc_1[0].second-scc_2[0].second)<1.0)
    {
        for (size_t i = 0; i < CorresID_1.size(); i++)
        {
            if (CorresID_1[i]==-1)
                continue;

            if (CorresID_2[CorresID_1[i]]==i)
                continue;

            PreKeys.push_back(kps_1[i]);
            CurKeys.push_back(kps_2[CorresID_1[i]]);
            TemperalMatches.push_back(cv::DMatch(count,count,0));
            count = count + 1;    
        }

        for (size_t i = 0; i < CorresID_2.size(); i++)
        {
            if (CorresID_2[i]==-1)
                continue;
                
            PreKeys.push_back(kps_2[i]);
            CurKeys.push_back(kps_1[CorresID_2[i]]);
            TemperalMatches.push_back(cv::DMatch(count,count,0));
            count = count + 1;    
        }       
    }
    // --- otherwise, keep the matched pairs from the direction with more pairs --- //
    else
    {
        const int inl_num_1 = CorresID_1.size()-std::count(CorresID_1.begin(), CorresID_1.end(), -1);
        const int inl_num_2 = CorresID_2.size()-std::count(CorresID_2.begin(), CorresID_2.end(), -1);
        if (inl_num_1>inl_num_2)
        {
            for (size_t i = 0; i < CorresID_1.size(); i++)
            {
                if (CorresID_1[i]==-1)
                    continue;
                    
                PreKeys.push_back(kps_1[i]);
                CurKeys.push_back(kps_2[CorresID_1[i]]);
                TemperalMatches.push_back(cv::DMatch(count,count,0));
                count = count + 1;    
            }
        }
        else
        {
            for (size_t i = 0; i < CorresID_2.size(); i++)
            {
                if (CorresID_2[i]==-1)
                    continue;
                    
                PreKeys.push_back(kps_2[i]);
                CurKeys.push_back(kps_1[CorresID_2[i]]);
                TemperalMatches.push_back(cv::DMatch(count,count,0));
                count = count + 1;    
            }
        }
               
    }
    
    // // --- cross-check --- //
    // std::vector<cv::KeyPoint> PreKeys, CurKeys;
    // std::vector<cv::DMatch> TemperalMatches;
    // int count = 0;
    // for (size_t i = 0; i < CorresID_1.size(); i++)
    // {
    //     if (CorresID_1[i]==-1)
    //         continue;

    //     if (CorresID_2[CorresID_1[i]]==i)
    //     {
    //         PreKeys.push_back(kps_1[i]);
    //         CurKeys.push_back(kps_2[CorresID_1[i]]);
    //         TemperalMatches.push_back(cv::DMatch(count,count,0));
    //         count = count + 1;
    //     }     
    // }

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

std::vector<int> FEAmatcher::GeoNearNeighSearch(const int &img_id, const int &img_id_ref,
                                                const cv::Mat &img, const cv::Mat &img_ref,
                                                const std::vector<cv::KeyPoint> &kps, const cv::Mat &dst, const cv::Mat &geo_img,
                                                const std::vector<cv::KeyPoint> &kps_ref, const cv::Mat &dst_ref, const cv::Mat &geo_img_ref,
                                                std::vector<std::pair<int,double>> &scc)
{
    cv::RNG rng((unsigned)time(NULL));
    std::vector<int> CorresID = std::vector<int>(kps.size(),-1);
    std::vector<int> ID_loc;

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
            double best_dist = 1000, sec_best_dist = 1000, dist_bound = 150;
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
            {
                CorresID[i] = best_id;
                ID_loc.push_back(i);
            }
            else if (candidate.size()==1 && best_dist<dist_bound)
            {
                CorresID[i] = best_id;
                ID_loc.push_back(i);
            }
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
            
             
        candidate.clear();
        kps_show.clear();
        kps_show_ref.clear();

    }

    // --- Sliding Compatibility Check on the X axis of keypoints --- //
    int final_inlier_num = 0, iter_num = 0, max_iter = 300, sam_num = 3;
    double PixError = 2.0;
    std::vector<int> CorresID_final = std::vector<int>(kps.size(),-1);
    while (iter_num<max_iter)
    {
        int cur_inlier_num = 0;
        std::vector<int> CorresID_iter = std::vector<int>(kps.size(),-1);

        // random sample matched ID from CorresID
        vector<int> sampled_ids(sam_num,0);
        for (size_t i = 0; i < sam_num; i++)
        {
            const int getID = rng.uniform(0,ID_loc.size());
            sampled_ids[i]=ID_loc[getID];
            // cout << sampled_ids[i] << " ";
        }
        // calculate model X
        double ModelX = 0;
        for (size_t i = 0; i < sampled_ids.size(); i++)
        {
            if (img_id%2!=img_id_ref%2)
                ModelX = ModelX + abs(kps[sampled_ids[i]].pt.y - (img_ref.rows-kps_ref[CorresID[sampled_ids[i]]].pt.y+1));
            else
                ModelX = ModelX + abs(kps[sampled_ids[i]].pt.y - kps_ref[CorresID[sampled_ids[i]]].pt.y);
        }
        ModelX = ModelX/sam_num;
        // fit all CorresID to Model, and find inliers
        for (size_t j = 0; j < CorresID.size(); j++)
        {
          if (CorresID[j]==-1)
              continue;
   
          double X_tmp;
          if (img_id%2!=img_id_ref%2)
          {
              X_tmp= abs(kps[j].pt.y - (img_ref.rows-kps_ref[CorresID[j]].pt.y+1));
          }
          else
              X_tmp= abs(kps[j].pt.y - kps_ref[CorresID[j]].pt.y);
              
          // cout << "distance: " << abs(ModelX-X_tmp) << endl;
          if (abs(ModelX-X_tmp)<=PixError)
          {
            CorresID_iter[j] = CorresID[j];
            cur_inlier_num = cur_inlier_num + 1;
          }          
        }
        // update the most inlier set
        if (final_inlier_num<cur_inlier_num)
        {
          CorresID_final = CorresID_iter;
          final_inlier_num = cur_inlier_num;
          scc.push_back(std::make_pair(cur_inlier_num,ModelX)); 
        }

        iter_num = iter_num + 1;
    }
    cout << "initial inlier number: " << CorresID.size()-std::count(CorresID.begin(), CorresID.end(), -1) << endl;
    CorresID = CorresID_final;
    cout << "final inlier number: " << CorresID.size()-std::count(CorresID.begin(), CorresID.end(), -1) << endl;
    
    // for (size_t i = 0; i < CorresID.size(); i++)
    // {
    //     if (CorresID[i]==-1)
    //         continue;

    //     int cur_inlier_num = 0;
    //     std::vector<int> CorresID_iter = std::vector<int>(kps.size(),-1);
    //     double ModelX;
    //     if (img_id%2!=img_id_ref%2)
    //         ModelX = abs(kps[i].pt.y - (img_ref.rows-kps_ref[CorresID[i]].pt.y+1));
    //     else
    //         ModelX = abs(kps[i].pt.y - kps_ref[CorresID[i]].pt.y);
    //     for (size_t j = 0; j < CorresID.size(); j++)
    //     {
    //       if (CorresID[j]==-1)
    //           continue;
   
    //       double X_tmp;
    //       if (img_id%2!=img_id_ref%2)
    //       {
    //           X_tmp= abs(kps[j].pt.y - (img_ref.rows-kps_ref[CorresID[j]].pt.y+1));
    //       }
    //       else
    //           X_tmp= abs(kps[j].pt.y - kps_ref[CorresID[j]].pt.y);
              
    //       // cout << "distance: " << abs(ModelX-X_tmp) << endl;
    //       if (abs(ModelX-X_tmp)<=PixError)
    //       {
    //         CorresID_iter[j] = CorresID[j];
    //         cur_inlier_num = cur_inlier_num + 1;
    //       }          
    //     }

    //     // cout << "current inlier number: " << cur_inlier_num << endl;

    //     if (final_inlier_num<cur_inlier_num)
    //     {
    //       CorresID_final = CorresID_iter;
    //       final_inlier_num = cur_inlier_num;
    //     }    
    //     scc[i] = std::make_pair(cur_inlier_num,ModelX);    
    // }
    // cout << "initial inlier number: " << CorresID.size()-std::count(CorresID.begin(), CorresID.end(), -1) << endl;
    // CorresID = CorresID_final;
    // cout << "final inlier number: " << CorresID.size()-std::count(CorresID.begin(), CorresID.end(), -1) << endl;
  
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
