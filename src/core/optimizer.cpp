
#include <bits/stdc++.h>
#include <iostream>

#include "optimizer.h"


namespace Diasss
{

using namespace std;
using namespace cv;
using namespace gtsam;

#define PI 3.14159265359

void Optimizer::TrajOptimizationPair(Frame &SourceFrame, Frame &TargetFrame)
{

    vector<Pose3> dr_poses;

    std::vector<Vector6> kps_pairs = Optimizer::GetKpsPairs(SourceFrame.anno_kps, SourceFrame.img_id, TargetFrame.img_id,
                                                     SourceFrame.altitudes,SourceFrame.ground_ranges,
                                                     TargetFrame.altitudes,TargetFrame.ground_ranges);

    vector<Point3> ini_points = Optimizer::TriangulateLandmarks(kps_pairs, SourceFrame.tf_stb, SourceFrame.tf_port,
                                                                SourceFrame.img_id, TargetFrame.img_id, 
                                                                SourceFrame.geo_img, TargetFrame.geo_img,
                                                                SourceFrame.altitudes, TargetFrame.altitudes,
                                                                SourceFrame.dr_poses, TargetFrame.dr_poses);


    // Create an iSAM2 object.
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    ISAM2 isam(parameters);

    // Create a Factor Graph and Values to hold the new data
    NonlinearFactorGraph graph;
    Values initialEstimate;




}

std::vector<Vector6> Optimizer::GetKpsPairs(const cv::Mat &kps, const int &id_s, const int &id_t,
                                     const std::vector<double> &alts_s, const std::vector<double> &gras_s,
                                     const std::vector<double> &alts_t, const std::vector<double> &gras_t)
{
    bool USE_ANNO = true;

    std::vector<Vector6> kps_pairs;

    for (size_t i = 0; i < kps.rows; i++)
    {
        // decide which frame id is the target (associated) frame
        int id_check;
        std::vector<int> kp_s, kp_t;
        if (USE_ANNO)
        {    
            id_check = kps.at<int>(i,1);
            kp_s = {kps.at<int>(i,2),kps.at<int>(i,3)};
            kp_t = {kps.at<int>(i,4),kps.at<int>(i,5)};
        }
        else
        {   
            id_check = (int)kps.at<double>(i,1);
            kp_s = {(int)kps.at<double>(i,2),(int)kps.at<double>(i,3)};
            kp_t = {(int)kps.at<double>(i,4),(int)kps.at<double>(i,5)};
        }

        // save keypoint pairs with slant ranges
        if (id_check==id_t)
        {
            // calculate slant ranges
            int gra_id_s = kp_s[1]- gras_s.size();
            double slant_range_s = sqrt(alts_s[kp_s[0]]*alts_s[kp_s[0]] + gras_s[abs(gra_id_s)]*gras_s[abs(gra_id_s)]);
            int gra_id_t = kp_t[1]- gras_t.size();
            double slant_range_t = sqrt(alts_t[kp_t[0]]*alts_t[kp_t[0]] + gras_t[abs(gra_id_t)]*gras_t[abs(gra_id_t)]);

            Vector6 kp_pair = (gtsam::Vector6() << kp_s[0], kp_s[1], slant_range_s, kp_t[0], kp_t[1], slant_range_t).finished();

            // for (size_t i = 0; i < kp_pair.size(); i++)
            //     cout << kp_pair(i) << " ";
            // cout << endl;
            
            kps_pairs.push_back(kp_pair);

        }
        
    }

    return kps_pairs;
                                            
}


vector<Point3> Optimizer::TriangulateLandmarks(const std::vector<Vector6> &kps_pairs, 
                                               const std::vector<double> &tf_stb, const std::vector<double> &tf_port,
                                               const int &img_id_s, const int &img_id_t,
                                               const std::vector<cv::Mat> &geo_s, const std::vector<cv::Mat> &geo_t,
                                               const std::vector<double> &alts_s, const std::vector<double> &alts_t, 
                                               const cv::Mat &dr_poses_s, const cv::Mat &dr_poses_t)
{
    vector<Point3> output_point;


    // Create a Factor Graph and Values to hold the new data
    NonlinearFactorGraph graph;
    Values initialEstimate;

    // Noise model parameters for keypoint
    double sigma_r = 0.1, alpha_bw =0.1*PI/180;

    // --- main loop --- //
    for (size_t i = 0; i < kps_pairs.size(); i++)
    {

        // sensor offset
        Pose3 Ts_s;
        if (kps_pairs[i](1)<geo_s[0].cols/2)
            Ts_s = Pose3(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(tf_stb[0], tf_stb[1], tf_stb[2]));
        else
            Ts_s = Pose3(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(tf_port[0], tf_port[1], tf_port[2]));
        Pose3 Ts_t;
        if (kps_pairs[i](4)<geo_t[0].cols/2)
            Ts_t = Pose3(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(tf_stb[0], tf_stb[1], tf_stb[2]));
        else
            Ts_t = Pose3(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(tf_port[0], tf_port[1], tf_port[2]));

        // ping pose
        int id_s = kps_pairs[i](0), id_t = kps_pairs[i](3);
        Pose3 Tp_s = Pose3(Rot3::Rodrigues(dr_poses_s.at<double>(id_s,0), dr_poses_s.at<double>(id_s,1), dr_poses_s.at<double>(id_s,2)), 
                           Point3(dr_poses_s.at<double>(id_s,3), dr_poses_s.at<double>(id_s,4), dr_poses_s.at<double>(id_s,5)));
        Pose3 Tp_t = Pose3(Rot3::Rodrigues(dr_poses_t.at<double>(id_t,0), dr_poses_t.at<double>(id_t,1), dr_poses_t.at<double>(id_t,2)), 
                           Point3(dr_poses_t.at<double>(id_t,3), dr_poses_t.at<double>(id_t,4), dr_poses_t.at<double>(id_t,5)));

        // noise model
        auto KP_NOISE_1 = noiseModel::Diagonal::Sigmas(Vector2(sigma_r,kps_pairs[i](2)*alpha_bw));
        auto KP_NOISE_2 = noiseModel::Diagonal::Sigmas(Vector2(sigma_r,kps_pairs[i](5)*alpha_bw));
        
        // add factor to graph
        graph.add(LMTriaFactor(1,Vector2(kps_pairs[i](2),0),Ts_s,Tp_s,KP_NOISE_1));
        graph.add(LMTriaFactor(1,Vector2(kps_pairs[i](5),0),Ts_t,Tp_t,KP_NOISE_2));

        // initialize point
        int id_ss = kps_pairs[i](1), id_tt = kps_pairs[i](4);
        double x_bar = (geo_s[0].at<double>(id_s,id_ss)+geo_t[0].at<double>(id_t,id_tt))/2;
        double y_bar = (geo_s[1].at<double>(id_s,id_ss)+geo_t[1].at<double>(id_t,id_tt))/2;
        double z_bar = ( (dr_poses_s.at<double>(id_s,5)-alts_s[id_s]) + (dr_poses_t.at<double>(id_t,5)-alts_t[id_t]) )/2;
        initialEstimate.insert(1, Point3(x_bar, y_bar, z_bar));
        // cout << "initial: " << x_bar << " " << y_bar << " " << z_bar << endl;

        // constrcut solver and optimize
        gtsam::LevenbergMarquardtParams params; 
        // params.setVerbosityLM("SUMMARY");
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
        Values result = optimizer.optimize();
        // cout << "final: " << result.at<Point3>(1)(0) << " " << result.at<Point3>(1)(1) << " " << result.at<Point3>(1)(2) << endl;
        output_point.push_back(result.at<Point3>(1));

        // Clear the factor graph and values for the next iteration
        graph.resize(0);
        initialEstimate.clear();
    }
    


    return output_point;

}


} // namespace Diasss
