
#include <bits/stdc++.h>
#include <iostream>
#include <random>
#include <math.h> 

#include "optimizer.h"
#include "util.h"

namespace Diasss
{

using namespace std;
using namespace cv;
using namespace gtsam;

#define PI 3.14159265359

void Optimizer::TrajOptimizationPair(Frame &SourceFrame, Frame &TargetFrame)
{
    bool USE_ANNO = 0, SHOW_IMG = 0; // use annotation or not (show image or not)
    if (SHOW_IMG)
        Util::ShowAnnos(SourceFrame.img_id,TargetFrame.img_id,SourceFrame.norm_img,TargetFrame.norm_img,
                        SourceFrame.anno_kps,TargetFrame.anno_kps);

    // Noise model paras for pose
    double ro1_ = 0.01*PI/180, pi1_ = 0.01*PI/180, ya1_ = 0.1*PI/180, x1_ = 0.05, y1_ = 0.05, z1_ = 0.01;
    double ro2_ = 0.01*PI/180, pi2_ = 0.01*PI/180, ya2_ = 0.01*PI/180, x2_ = 0.01, y2_ = 0.01, z2_ = 0.01;
    // Noise model paras for keypoint
    double sigma_r = 0.1, alpha_bw =0.1*PI/180;


    std::vector<Vector6> kps_pairs = Optimizer::GetKpsPairs(USE_ANNO,SourceFrame.corres_kps,SourceFrame.img_id,TargetFrame.img_id,
                                                            SourceFrame.altitudes,SourceFrame.ground_ranges,
                                                            TargetFrame.altitudes,TargetFrame.ground_ranges);

    vector<Point3> ini_points = Optimizer::TriangulateLandmarks(kps_pairs, SourceFrame.tf_stb, SourceFrame.tf_port,
                                                                SourceFrame.img_id, TargetFrame.img_id, 
                                                                SourceFrame.geo_img, TargetFrame.geo_img,
                                                                SourceFrame.altitudes, TargetFrame.altitudes,
                                                                SourceFrame.dr_poses, TargetFrame.dr_poses);

    vector<pair<Pose3,Vector6>> lc_tf_Conv = Optimizer::LoopClosingTFs(kps_pairs, SourceFrame.tf_stb, SourceFrame.tf_port,
                                                                       SourceFrame.img_id, TargetFrame.img_id, 
                                                                       SourceFrame.geo_img, TargetFrame.geo_img,
                                                                       SourceFrame.altitudes, TargetFrame.altitudes,
                                                                       SourceFrame.ground_ranges, TargetFrame.ground_ranges,
                                                                       SourceFrame.dr_poses, TargetFrame.dr_poses);

    // // --- Assign unique ID for each pose ---//
    // int id_tmp = 0;
    // std::vector<int> g_id_s(SourceFrame.dr_poses.rows), g_id_t(TargetFrame.dr_poses.rows);
    // for (size_t i = 0; i < SourceFrame.dr_poses.rows; i++)
    // {
    //     g_id_s[i] = id_tmp;
    //     id_tmp = id_tmp + 1;
    // }
    // for (size_t i = 0; i < TargetFrame.dr_poses.rows; i++)
    // {
    //     g_id_t[i] = id_tmp;
    //     id_tmp = id_tmp + 1;
    // }
    // cout << "Total number of pings and keypoint pairs: " << id_tmp << " " << kps_pairs.size() << endl;

    // // Create an iSAM2 object.
    // ISAM2Params parameters;
    // parameters.relinearizeThreshold = 0.1;
    // parameters.relinearizeSkip = 10;
    // parameters.factorization = ISAM2Params::QR;
    // // parameters.optimizationParams = ISAM2DoglegParams();
    // parameters.print();
    // ISAM2 isam(parameters);

    // // int relinearizeInterval = 3;
    // // NonlinearISAM isam(relinearizeInterval);

    // // Create a Factor Graph and Values to hold the new data
    // NonlinearFactorGraph graph;
    // // NonlinearFactorGraph graphSAVE;
    // Values initialEstimate;


    // // // --- loop on the poses of the SOURCE image --- // //
    // for (size_t i = 0; i < SourceFrame.dr_poses.rows; i++)
    // {
    //     Pose3 pose_dr = Pose3(
    //             Rot3::Rodrigues(SourceFrame.dr_poses.at<double>(i,0),SourceFrame.dr_poses.at<double>(i,1),SourceFrame.dr_poses.at<double>(i,2)), 
    //             Point3(SourceFrame.dr_poses.at<double>(i,3), SourceFrame.dr_poses.at<double>(i,4), SourceFrame.dr_poses.at<double>(i,5)));
        
    //     initialEstimate.insert(Symbol('X', g_id_s[i]), gtsam::Pose3::identity());

    //     // if it's the first pose, add fixed prior factor
    //     if (i==0)
    //     {
    //         auto PriorModel = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.000001), Vector3::Constant(0.000001))
    //                                                        .finished());
    //         graph.addPrior(Symbol('X', g_id_s[i]), pose_dr, PriorModel);
    //         // graphSAVE.addPrior(Symbol('X', g_id_s[i]), pose_dr, PriorModel);

    //     }
    //     // add odometry factor and update isam
    //     else
    //     {
    //         Pose3 pose_dr_pre = Pose3(
    //                 Rot3::Rodrigues(SourceFrame.dr_poses.at<double>(i-1,0),SourceFrame.dr_poses.at<double>(i-1,1),SourceFrame.dr_poses.at<double>(i-1,2)), 
    //                 Point3(SourceFrame.dr_poses.at<double>(i-1,3), SourceFrame.dr_poses.at<double>(i-1,4), SourceFrame.dr_poses.at<double>(i-1,5)));

    //         auto odo = pose_dr_pre.between(pose_dr);

    //         auto OdoModel = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3(ro1_, pi1_, ya1_), Vector3(x1_, y1_, z1_))
    //                                                        .finished());

    //         graph.add(BetweenFactor<Pose3>(Symbol('X',g_id_s[i-1]), Symbol('X',g_id_s[i]), odo, OdoModel));
    //         // graphSAVE.add(BetweenFactor<Pose3>(Symbol('X',g_id_s[i-1]), Symbol('X',g_id_s[i]), odo, OdoModel));

    //         // Update iSAM with the new factors
    //         isam.update(graph, initialEstimate);
    //         // One more time
    //         isam.update();
    //         Values currentEstimate = isam.calculateEstimate();
    //         // Values currentEstimate = isam.estimate();
    //         // cout << "Updating Current Ping #" << g_id_s[i] << ": " << endl;
    //         // cout << currentEstimate.at<Pose3>(Symbol('X',g_id_s[i])) << endl;

    //         // Clear the factor graph and values for the next iteration
    //         graph.resize(0);
    //         initialEstimate.clear();
    //     }
          
    // }


    // // // --- loop on the poses of the TARGET image --- // //
    // for (size_t i = 0; i < TargetFrame.dr_poses.rows; i++)
    // {
    //     Pose3 pose_dr = Pose3(
    //             Rot3::Rodrigues(TargetFrame.dr_poses.at<double>(i,0),TargetFrame.dr_poses.at<double>(i,1),TargetFrame.dr_poses.at<double>(i,2)), 
    //             Point3(TargetFrame.dr_poses.at<double>(i,3), TargetFrame.dr_poses.at<double>(i,4), TargetFrame.dr_poses.at<double>(i,5)));
        
    //     initialEstimate.insert(Symbol('X', g_id_t[i]), gtsam::Pose3::identity());


    //     // // get the last pose from end of last image 
    //     // // if it is the start pose in current image
    //     if (i==0)
    //     {
    //         int id = SourceFrame.dr_poses.rows - 1;
    //         Pose3 pose_dr_pre = Pose3(
    //                 Rot3::Rodrigues(SourceFrame.dr_poses.at<double>(id,0),SourceFrame.dr_poses.at<double>(id,1),SourceFrame.dr_poses.at<double>(id,2)), 
    //                 Point3(SourceFrame.dr_poses.at<double>(id,3), SourceFrame.dr_poses.at<double>(id,4), SourceFrame.dr_poses.at<double>(id,5)));

    //         auto odo = pose_dr_pre.between(pose_dr);

    //         auto OdoModel = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3(ro1_, pi1_, ya1_), Vector3(x1_, y1_, z1_))
    //                                                         .finished());

    //         graph.add(BetweenFactor<Pose3>(Symbol('X',g_id_s[id]), Symbol('X',g_id_t[i]), odo, OdoModel));
    //         // graphSAVE.add(BetweenFactor<Pose3>(Symbol('X',g_id_s[id]), Symbol('X',g_id_t[i]), odo, OdoModel));
    //     }
    //     else
    //     {
    //         Pose3 pose_dr_pre = Pose3(
    //                 Rot3::Rodrigues(TargetFrame.dr_poses.at<double>(i-1,0),TargetFrame.dr_poses.at<double>(i-1,1),TargetFrame.dr_poses.at<double>(i-1,2)), 
    //                 Point3(TargetFrame.dr_poses.at<double>(i-1,3), TargetFrame.dr_poses.at<double>(i-1,4), TargetFrame.dr_poses.at<double>(i-1,5)));

    //         auto odo = pose_dr_pre.between(pose_dr);

    //         auto OdoModel = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3(ro1_, pi1_, ya1_), Vector3(x1_, y1_, z1_))
    //                                                         .finished());

    //         graph.add(BetweenFactor<Pose3>(Symbol('X',g_id_t[i-1]), Symbol('X',g_id_t[i]), odo, OdoModel));
    //         // graphSAVE.add(BetweenFactor<Pose3>(Symbol('X',g_id_t[i-1]), Symbol('X',g_id_t[i]), odo, OdoModel));
    //     }       
        
    //     // Update iSAM with the new factors
    //     isam.update(graph, initialEstimate);
    //     // One more time
    //     isam.update();
    //     Values currentEstimate = isam.calculateEstimate();
    //     // Values currentEstimate = isam.estimate();

    //     // Clear the factor graph and values for the next iteration
    //     graph.resize(0);
    //     initialEstimate.clear();

    // }

    // Values DRonlyEstimate = isam.calculateEstimate();
    // // ofstream DRonly_optimized_dot_file("../DRonlyOptimizedGraph.dot");
    // // graphSAVE.saveGraph(DRonly_optimized_dot_file, DRonlyEstimate);


    // // // --- loop on the keypoint pairs measurements --- // //
    // int step_size = 1;
    // for (size_t i = 0; i < kps_pairs.size(); i=i+step_size)
    // {
    //     cout << "***********************************************************" << endl;
    //     cout << "Add New KP Measurement: L" << i << " ";
    //     cout << "between X" << g_id_s[(int)kps_pairs[i](0)] << " and X" << g_id_t[(int)kps_pairs[i](3)] << " ";
    //     cout << "(" << kps_pairs[i](0) << " and " << kps_pairs[i](3) << ")" << endl;

    //     // initialize keypoint
    //     // initialEstimate.insert(Symbol('L',i), Point3(0.0,0.0,0.0));
    //     initialEstimate.insert(Symbol('L',i), ini_points[i]);

    //     // sensor offset
    //     int id_ss = kps_pairs[i](1), id_tt = kps_pairs[i](4);
    //     if (id_ss>=SourceFrame.norm_img.cols || id_tt>=TargetFrame.norm_img.cols)
    //         cout << "column index out of range !!! (in factor construction)" << endl;  
    //     Pose3 Ts_s;
    //     if (id_ss<SourceFrame.norm_img.cols/2)
    //         Ts_s = Pose3(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(SourceFrame.tf_stb[0], SourceFrame.tf_stb[1], SourceFrame.tf_stb[2]));
    //     else
    //         Ts_s = Pose3(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(SourceFrame.tf_port[0], SourceFrame.tf_port[1], SourceFrame.tf_port[2]));
    //     Pose3 Ts_t;
    //     if (id_tt<TargetFrame.norm_img.cols/2)
    //         Ts_t = Pose3(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(TargetFrame.tf_stb[0], TargetFrame.tf_stb[1], TargetFrame.tf_stb[2]));
    //     else
    //         Ts_t = Pose3(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(TargetFrame.tf_port[0], TargetFrame.tf_port[1], TargetFrame.tf_port[2]));       

    //     // noise model
    //     auto KP_NOISE_1 = noiseModel::Diagonal::Sigmas(Vector2(sigma_r,kps_pairs[i](2)*alpha_bw));
    //     auto KP_NOISE_2 = noiseModel::Diagonal::Sigmas(Vector2(sigma_r,kps_pairs[i](5)*alpha_bw));

    //     // add factor to graph
    //     int id_s = kps_pairs[i](0), id_t = kps_pairs[i](3);
    //     if (id_s>=SourceFrame.dr_poses.rows || id_t>=TargetFrame.dr_poses.rows)
    //         cout << "row index out of range when adding to graph!!!" << endl;
    //     // // TODO: don't use kps_id for landmark unique ID, if it is not for pair image optimization but more;
    //     Pose3 Tp_s = DRonlyEstimate.at<Pose3>(Symbol('X',g_id_s[id_s]));
    //     // Pose3 Tp_s = Pose3(
    //     //         Rot3::Rodrigues(SourceFrame.dr_poses.at<double>(id_s,0),SourceFrame.dr_poses.at<double>(id_s,1),SourceFrame.dr_poses.at<double>(id_s,2)), 
    //     //         Point3(SourceFrame.dr_poses.at<double>(id_s,3), SourceFrame.dr_poses.at<double>(id_s,4), SourceFrame.dr_poses.at<double>(id_s,5)));
    //     graph.add(LMTriaFactor(Symbol('L',i),Vector2(kps_pairs[i](2),0.0),Ts_s,Tp_s,KP_NOISE_1));
    //     // graph.add(SssPointFactor(Symbol('L',i),Symbol('X',g_id_s[id_s]),Vector2(kps_pairs[i](2),0.0),Ts_s,KP_NOISE_1));
    //     graph.add(SssPointFactor(Symbol('L',i),Symbol('X',g_id_t[id_t]),Vector2(kps_pairs[i](5),0.0),Ts_t,KP_NOISE_2));

    //     // Update iSAM with the new factors
    //     isam.update(graph, initialEstimate);
    //     // One more time
    //     isam.update();
    //     Values currentEstimate = isam.calculateEstimate();
    //     // Values currentEstimate = isam.estimate();

    //     // Show results before and after optimization
    //     bool printinfo = 1;
    //     if (printinfo)
    //     {
    //         cout << "Updating Current Ping #" << g_id_t[id_t] << ": " << endl;

    //         cout << "NEW POSE 2: " << endl << currentEstimate.at<Pose3>(Symbol('X',g_id_t[id_t])) << endl;
    //         Pose3 pose_dr = Pose3(
    //             Rot3::Rodrigues(TargetFrame.dr_poses.at<double>(id_t,0),TargetFrame.dr_poses.at<double>(id_t,1),TargetFrame.dr_poses.at<double>(id_t,2)), 
    //             Point3(TargetFrame.dr_poses.at<double>(id_t,3), TargetFrame.dr_poses.at<double>(id_t,4), TargetFrame.dr_poses.at<double>(id_t,5)));
    //         cout << "OLD POSE 2: " << endl << pose_dr << endl;
    //         cout << "NEW POINT: " << endl << currentEstimate.at<Point3>(Symbol('L',i)) << endl;
    //         cout << "OLD POINT: " << endl << ini_points[i] << endl << endl;
    //     }

    //     // Clear the factor graph and values for the next iteration
    //     graph.resize(0);
    //     initialEstimate.clear();



    // }
    
    
    




}

std::vector<Vector6> Optimizer::GetKpsPairs(const bool &USE_ANNO, const cv::Mat &kps, const int &id_s, const int &id_t,
                                     const std::vector<double> &alts_s, const std::vector<double> &gras_s,
                                     const std::vector<double> &alts_t, const std::vector<double> &gras_t)
{

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
            // cout << gra_id_s << " " << gra_id_t << endl;
            
            kps_pairs.push_back(kp_pair);

        }
        
    }

    return kps_pairs;
                                            
}

std::vector<pair<Pose3,Vector6>>  Optimizer::LoopClosingTFs(const std::vector<Vector6> &kps_pairs, 
                                                const std::vector<double> &tf_stb, const std::vector<double> &tf_port,
                                                const int &img_id_s, const int &img_id_t,
                                                const std::vector<cv::Mat> &geo_s, const std::vector<cv::Mat> &geo_t,
                                                const std::vector<double> &alts_s, const std::vector<double> &alts_t,
                                                const std::vector<double> &gras_s, const std::vector<double> &gras_t,
                                                const cv::Mat &dr_poses_s, const cv::Mat &dr_poses_t)
{

    ofstream save_result_1;
    string path1 = "../ini_lm_error.txt";
    save_result_1.open(path1.c_str(),ios::trunc);
    ofstream save_result_2;
    string path2 = "../fnl_lm_errors.txt";
    save_result_2.open(path2.c_str(),ios::trunc);

    Pose3 cps_pose = gtsam::Pose3::identity();
    if (img_id_s%2!=img_id_t%2)
        cps_pose = Pose3(Rot3::Rodrigues(0.0, 0.0, PI), Point3(0.0,0.0,0.0));
    

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);
    
    std::vector<pair<Pose3,Vector6>> output_tfs;

    // Create a Factor Graph and Values to hold the new data
    NonlinearFactorGraph graph;
    Values initialEstimate;

    // Noise model parameters for keypoint
    double sigma_r = 0.1, alpha_bw =0.1*PI/180;

    // --- main loop --- //
    bool graph_option = 0;
    int step_size = 1;
    for (size_t i = 0; i < kps_pairs.size(); i=i+step_size)
    {   
        // noise model
        auto KP_NOISE_1 = noiseModel::Diagonal::Sigmas(Vector2(sigma_r,kps_pairs[i](2)*alpha_bw));
        auto KP_NOISE_2 = noiseModel::Diagonal::Sigmas(Vector2(sigma_r,kps_pairs[i](5)*alpha_bw));

        // sensor offset
        Pose3 Ts_s;
        if (kps_pairs[i](1)<geo_s[0].cols/2)
            Ts_s = Pose3(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(tf_stb[0], tf_stb[1], tf_stb[2]));
        else
            Ts_s = Pose3(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(tf_port[0], tf_port[1], tf_port[2]));
        Pose3 Ts_t;
        if (kps_pairs[i](4)<geo_t[0].cols/2)
            Ts_t = Pose3(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(tf_stb[0], tf_stb[1], tf_stb[2]))*cps_pose;
        else
            Ts_t = Pose3(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(tf_port[0], tf_port[1], tf_port[2]))*cps_pose;

        // ping pose
        int id_s = kps_pairs[i](0), id_t = kps_pairs[i](3);
        if (id_s>=dr_poses_s.rows || id_t>=dr_poses_t.rows)
            cout << "row index out of range !!!" << endl;    
        Pose3 Tp_s = Pose3(Rot3::Rodrigues(dr_poses_s.at<double>(id_s,0), dr_poses_s.at<double>(id_s,1), dr_poses_s.at<double>(id_s,2)), 
                           Point3(dr_poses_s.at<double>(id_s,3), dr_poses_s.at<double>(id_s,4), dr_poses_s.at<double>(id_s,5)));
        Pose3 Tp_t = Pose3(Rot3::Rodrigues(dr_poses_t.at<double>(id_t,0), dr_poses_t.at<double>(id_t,1), dr_poses_t.at<double>(id_t,2)), 
                           Point3(dr_poses_t.at<double>(id_t,3), dr_poses_t.at<double>(id_t,4), dr_poses_t.at<double>(id_t,5)));
        Pose3 Tp_st = Tp_s.between(Tp_t*cps_pose);

        if (graph_option)
        {
            // fix the relative transform with DR prior
            double ro_ = 0.1*PI/180, pi_ = 0.1*PI/180, ya_ = 4.0*PI/180, x_ = abs(Tp_st.x()*2), y_ = abs(Tp_st.y()/10), z_ = 0.5; // noise paras
            auto PosePriorModel = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3(ro_, pi_, ya_), Vector3(x_, y_, z_))
                                                            .finished());
            graph.addPrior(Symbol('X', 2), Tp_t, PosePriorModel);
            
            // add factor to graph
            graph.add(LMTriaFactor(Symbol('L', 1),Vector2(kps_pairs[i](2),0),Ts_s,Tp_s,KP_NOISE_1));
            graph.add(SssPointFactor(Symbol('L',1),Symbol('X',2),Vector2(kps_pairs[i](5),0.0),Ts_t,KP_NOISE_2));

            // initialize point
            int id_ss = kps_pairs[i](1), id_tt = kps_pairs[i](4);
            if (id_ss>=geo_s[0].cols || id_tt>=geo_t[0].cols)
                cout << "column index out of range !!!" << endl;  
            double x_bar = (geo_s[0].at<double>(id_s,id_ss)+geo_t[0].at<double>(id_t,id_tt))/2;
            double y_bar = (geo_s[1].at<double>(id_s,id_ss)+geo_t[1].at<double>(id_t,id_tt))/2;
            double z_bar = ( (dr_poses_s.at<double>(id_s,5)-alts_s[id_s]) + (dr_poses_t.at<double>(id_t,5)-alts_t[id_t]) )/2;
            initialEstimate.insert(Symbol('L',1), Point3(x_bar, y_bar, z_bar));

            // initialize pose
            std::vector<double> seeds;
            for (size_t j = 0; j < 6; j++)
                seeds.push_back(distribution(generator));        
            Pose3 add_noise(Rot3::Rodrigues(seeds[0]*2*PI/180, seeds[1]*2*PI/180, seeds[2]*5*PI/180), Point3(seeds[3]*5, seeds[4]*5, seeds[5]*2));
            initialEstimate.insert(Symbol('X',2), Tp_t.compose(add_noise));
            // initialEstimate.insert(Symbol('X',2), Pose3::identity());
        }
        else
        {
            // fix at the source pose with DR prior
            auto PosePriorModel = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.000001), Vector3::Constant(0.000001))
                                                                    .finished());
            graph.addPrior(Symbol('X', 1), Tp_s, PosePriorModel);    

            // add odometry factor to graph      
            double ro_ = 0.1*PI/180, pi_ = 0.1*PI/180, ya_ = 2.0*PI/180, x_ = abs(Tp_st.x()*2), y_ = abs(Tp_st.y()/10), z_ = 0.5;
            auto OdometryNoiseModel = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3(ro_, pi_, ya_), Vector3(x_, y_, z_))
                                                            .finished());
            graph.add(BetweenFactor<Pose3>(Symbol('X',1), Symbol('X',2), Tp_st, OdometryNoiseModel));
            // cout << "Odo noise on x, y and yaw: " << x_ << " " << y_ << " " << ya_ << endl;
      
            // add keypoint measurement factor to graph
            graph.add(SssPointFactor(Symbol('L',1),Symbol('X',1),Vector2(kps_pairs[i](2),0.0),Ts_s,KP_NOISE_1));
            graph.add(SssPointFactor(Symbol('L',1),Symbol('X',2),Vector2(kps_pairs[i](5),0.0),Ts_t,KP_NOISE_2));

            // initialize point
            int id_ss = kps_pairs[i](1), id_tt = kps_pairs[i](4);
            if (id_ss>=geo_s[0].cols || id_tt>=geo_t[0].cols)
                cout << "column index out of range !!!" << endl;  
            double x_bar = (geo_s[0].at<double>(id_s,id_ss)+geo_t[0].at<double>(id_t,id_tt))/2;
            double y_bar = (geo_s[1].at<double>(id_s,id_ss)+geo_t[1].at<double>(id_t,id_tt))/2;
            double z_bar = ( (dr_poses_s.at<double>(id_s,5)-alts_s[id_s]) + (dr_poses_t.at<double>(id_t,5)-alts_t[id_t]) )/2;
            initialEstimate.insert(Symbol('L',1), Point3(x_bar, y_bar, z_bar));

            // initialize pose
            initialEstimate.insert(Symbol('X',1), Tp_s);
            initialEstimate.insert(Symbol('X',2), Tp_t*cps_pose);
            // std::vector<double> seeds;
            // for (size_t j = 0; j < 6; j++)
            //     seeds.push_back(distribution(generator));        
            // Pose3 add_noise(Rot3::Rodrigues(seeds[0]*PI/180, seeds[1]*PI/180, seeds[2]*4*PI/180), Point3(seeds[3]*4, seeds[4]*4, seeds[5]));
            // // cout << "noise: " << seeds[0] << " " << seeds[1] << " " << seeds[2] << " " << seeds[3] << " " << seeds[4] << " " << seeds[5] << endl;
            // initialEstimate.insert(Symbol('X',2), Tp_t.compose(add_noise));
        }

        // constrcut solver and optimize
        gtsam::LevenbergMarquardtParams params; 
        params.setVerbosityLM("SUMMARY");
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
        // GaussNewtonParams parameters;
        // parameters.relativeErrorTol = 1e-5;
        // parameters.maxIterations = 100;
        // GaussNewtonOptimizer optimizer(graph, initialEstimate, parameters);
        Values result = optimizer.optimize();

        // Show results before and after optimization
        bool printinfo = 1;
        if (printinfo)
        {
            Pose3 new_pose = result.at<Pose3>(Symbol('X', 2))*cps_pose.inverse();
            cout << "***********************************************************" << endl;
            cout << "Add New KP Measurement: L" << i << " ";
            cout << "between ping " << kps_pairs[i](0) << " and " << kps_pairs[i](3) << " " << endl;
            cout << "NEW POSE: " << endl << new_pose.translation() << endl;
            cout << "OLD POSE: " << endl << Tp_t.translation() << endl;
            // cout << "INI POSE: " << endl << (Tp_t.compose(add_noise)).translation() << endl;
        }

        // evaluate if the estimation improves after optimization
        bool eval_1 = 1;
        if (eval_1)
        {
            double x_dist, y_dist;

            // initial landmark distance observed between two dr ping poses
            int id_ss = kps_pairs[i](1), id_tt = kps_pairs[i](4);
            if (id_ss>=geo_s[0].cols || id_tt>=geo_t[0].cols)
                cout << "column index out of range !!!" << endl;  
            x_dist = (geo_s[0].at<double>(id_s,id_ss)-geo_t[0].at<double>(id_t,id_tt));
            y_dist = (geo_s[1].at<double>(id_s,id_ss)-geo_t[1].at<double>(id_t,id_tt));
            double ini_point_dist = sqrt(x_dist*x_dist + y_dist*y_dist);

            // final landmark distance observed between two estimated ping poses
            double lm_geo_t_x, lm_geo_t_y;
            Pose3 new_pose = result.at<Pose3>(Symbol('X', 2))*cps_pose.inverse();
            // Pose3 new_pose = Tp_t;
            // cout << new_pose.x() << " " << new_pose.y() << " " << new_pose.rotation().yaw() << endl;
            // cout << dr_poses_t.at<double>(id_t,3) << " " << dr_poses_t.at<double>(id_t,4)  << " " << dr_poses_t.at<double>(id_t,2) << endl;
            if (kps_pairs[i](4)<geo_t[0].cols/2)
            {
                int gr_idx = geo_t[0].cols/2 - kps_pairs[i](4);
                lm_geo_t_x = new_pose.x() + gras_t[gr_idx]*cos(new_pose.rotation().yaw()+PI/2-PI);
                lm_geo_t_y = new_pose.y() + gras_t[gr_idx]*sin(new_pose.rotation().yaw()+PI/2-PI);
            }
            else
            {
                int gr_idx = kps_pairs[i](4) - geo_t[0].cols/2;
                lm_geo_t_x = new_pose.x() + gras_t[gr_idx]*cos(new_pose.rotation().yaw()-PI/2-PI);
                lm_geo_t_y = new_pose.y() + gras_t[gr_idx]*sin(new_pose.rotation().yaw()-PI/2-PI);
            }
            x_dist = (geo_s[0].at<double>(id_s,id_ss)-lm_geo_t_x);
            y_dist = (geo_s[1].at<double>(id_s,id_ss)-lm_geo_t_y);
            double final_point_dist = sqrt(x_dist*x_dist + y_dist*y_dist);   

            cout << "landmark distance (initial VS final): " << ini_point_dist << " " << final_point_dist << endl << endl;
            save_result_1 << ini_point_dist << endl;
            save_result_2 << final_point_dist << endl;

        }

        bool eval_2 = 0;
        if (eval_2)
        {
            int id_ss = kps_pairs[i](1), id_tt = kps_pairs[i](4);
            if (id_ss>=geo_s[0].cols || id_tt>=geo_t[0].cols)
                cout << "column index out of range !!!" << endl;  
            double x_bar = (geo_s[0].at<double>(id_s,id_ss)+geo_t[0].at<double>(id_t,id_tt))/2;
            double y_bar = (geo_s[1].at<double>(id_s,id_ss)+geo_t[1].at<double>(id_t,id_tt))/2;
            double z_bar = ( (dr_poses_s.at<double>(id_s,5)-alts_s[id_s]) + (dr_poses_t.at<double>(id_t,5)-alts_t[id_t]) )/2;

            Point3 point_dr = Optimizer::TriangulateOneLandmark(kps_pairs[i],Ts_s,Ts_t,Tp_s,Tp_t,Point3(x_bar, y_bar, z_bar));

            Point3 point_est = Optimizer::TriangulateOneLandmark(kps_pairs[i],Ts_s,Ts_t,Tp_s,
                                                                 result.at<Pose3>(Symbol('X', 2)),Point3(x_bar, y_bar, z_bar));

        }
        
        

        Marginals marginals(graph, result, Marginals::QR);

        output_tfs.push_back(std::make_pair(Tp_s.between(result.at<Pose3>(Symbol('X',2))*cps_pose.inverse()), marginals.marginalCovariance(Symbol('X',2)).diagonal()) );

        // Clear the factor graph and values for the next iteration
        graph.resize(0);
        initialEstimate.clear();
    }

    save_result_1.close();
    save_result_2.close();
    
    return output_tfs;

}

Point3 Optimizer::TriangulateOneLandmark(const Vector6 &kps_pair, 
                                         const Pose3 &Ts_s, const Pose3 &Ts_t,
                                         const Pose3 &Tp_s, const Pose3 &Tp_t,
                                         const Point3 &lm_ini)
{

    // Create a Factor Graph and Values to hold the new data
    NonlinearFactorGraph graph;
    Values initialEstimate;

    // Noise model parameters for keypoint
    double sigma_r = 0.1, alpha_bw =0.1*PI/180;

    // noise model
    auto KP_NOISE_1 = noiseModel::Diagonal::Sigmas(Vector2(sigma_r,kps_pair(2)*alpha_bw));
    auto KP_NOISE_2 = noiseModel::Diagonal::Sigmas(Vector2(sigma_r,kps_pair(5)*alpha_bw));
    
    // add factor to graph
    graph.add(LMTriaFactor(1,Vector2(kps_pair(2),0),Ts_s,Tp_s,KP_NOISE_1));
    graph.add(LMTriaFactor(1,Vector2(kps_pair(5),0),Ts_t,Tp_t,KP_NOISE_2));

    initialEstimate.insert(1, lm_ini);

    // constrcut solver and optimize
    gtsam::LevenbergMarquardtParams params; 
    // params.setVerbosityLM("SUMMARY");
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
    Values result = optimizer.optimize();
    // cout << "final: " << result.at<Point3>(1)(0) << " " << result.at<Point3>(1)(1) << " " << result.at<Point3>(1)(2) << endl;

    return result.at<Point3>(1);

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
        if (id_s>=dr_poses_s.rows || id_t>=dr_poses_t.rows)
            cout << "row index out of range !!!" << endl;    
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
        if (id_ss>=geo_s[0].cols || id_tt>=geo_t[0].cols)
            cout << "column index out of range !!!" << endl;  
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
