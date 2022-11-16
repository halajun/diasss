
#include <bits/stdc++.h>
#include <iostream>
#include <random>
#include <math.h> 
#include <tuple>
#include <string>

#include "optimizer.h"
#include "util.h"

namespace Diasss
{

using namespace std;
using namespace cv;
using namespace gtsam;

#define PI 3.14159265359

void Optimizer::TrajOptimizationAll(std::vector<Frame> &AllFrames)
{
    // use annotation or not, add loopclosure or not
    bool USE_ANNO = 0, ADD_LC = 1;
    // Noise model paras for pose
    double ro1_ = 0.01*PI/180, pi1_ = 0.01*PI/180, ya1_ = 0.05*PI/180, x1_ = 0.05, y1_ = 0.05, z1_ = 0.01;
    // random noise generator
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);
    double noise_xyz = 2, noise_rpy = 2*PI/180;

    // --- get all the keypoint pairs (all images) --- //
    std::vector<std::vector<Vector6>> kps_pairs_all;
    std::vector<pair<int,int>> img_pairs_ids;
    for (size_t i = 0; i < AllFrames.size(); i++)
    {
        for (size_t j = i+1; j < AllFrames.size(); j++)
        {
            std::vector<Vector6> kps_pairs;
            if (USE_ANNO)
            {
                kps_pairs = Optimizer::GetKpsPairs(USE_ANNO,AllFrames[i].anno_kps,AllFrames[i].img_id,AllFrames[j].img_id,
                                                                        AllFrames[i].altitudes,AllFrames[i].ground_ranges,
                                                                        AllFrames[j].altitudes,AllFrames[j].ground_ranges);
            }
            else
            {
                kps_pairs = Optimizer::GetKpsPairs(USE_ANNO,AllFrames[i].corres_kps,AllFrames[i].img_id,AllFrames[j].img_id,
                                                                        AllFrames[i].altitudes,AllFrames[i].ground_ranges,
                                                                        AllFrames[j].altitudes,AllFrames[j].ground_ranges);
            }

            kps_pairs_all.push_back(kps_pairs);
            img_pairs_ids.push_back(make_pair(AllFrames[i].img_id,AllFrames[j].img_id));
            
        }
        
    }

    // --- get all the loop closing measurements --- //
    std::vector<std::vector<tuple<Pose3,Vector6,double>>> lc_tf_all;
    int idx = 0;
    for (size_t i = 0; i < AllFrames.size(); i++)
    {
        for (size_t j = i+1; j < AllFrames.size(); j++)
        {
            cout << "***********************************************************************" << endl;
            cout << "Compute lc tfs between frame " << AllFrames[i].img_id << " and " << AllFrames[j].img_id << " ";
            cout << "( "  << kps_pairs_all[idx].size() << " in total...)" << endl;
            vector<tuple<Pose3,Vector6,double>> lc_tf_Conv = Optimizer::LoopClosingTFs(kps_pairs_all[idx], 
                                                                            AllFrames[i].tf_stb, AllFrames[i].tf_port,
                                                                            AllFrames[i].img_id, AllFrames[j].img_id, 
                                                                            AllFrames[i].geo_img, AllFrames[j].geo_img,
                                                                            AllFrames[i].altitudes, AllFrames[j].altitudes,
                                                                            AllFrames[i].ground_ranges, AllFrames[j].ground_ranges,
                                                                            AllFrames[i].dr_poses, AllFrames[j].dr_poses);
            lc_tf_all.push_back(lc_tf_Conv);
            idx = idx + 1;         
            
        }

    }

    // --- assign unique ID for each pose ---//  
    int id_sum = 0;  
    std::vector<std::vector<int>> unique_id;
    for (size_t i = 0; i < AllFrames.size(); i++)
    {
        std::vector<int> id_tmp(AllFrames[i].dr_poses.rows);
        for (size_t j = 0; j < AllFrames[i].dr_poses.rows; j++)
        {
            id_tmp[j] = id_sum;
            id_sum = id_sum + 1;
        }
        unique_id.push_back(id_tmp);

    }

    // --- record unique ID for each keypoint pair, --- //
    // --- only the ID of second keypoint is recorded. --- //
    std::vector<std::vector<int>> id_in_kps;
    for (size_t i = 0; i < kps_pairs_all.size(); i++)
    {
        std::vector<int> id_tmp(kps_pairs_all[i].size());
        for (size_t j = 0; j < kps_pairs_all[i].size(); j++)
        {
            int ping_num = kps_pairs_all[i][j](3);
            if (ping_num>=AllFrames[img_pairs_ids[i].second].dr_poses.rows)
                cout << "! index issue in TrajOptimizationAll(): " << ping_num << ">" << AllFrames[img_pairs_ids[i].second].dr_poses.rows << endl;      
            id_tmp[j] = unique_id[img_pairs_ids[i].second][ping_num];
        }
        id_in_kps.push_back(id_tmp);
        
    }
    
    // Create an iSAM2 object.
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    parameters.factorization = ISAM2Params::QR;
    parameters.print();
    ISAM2 isam(parameters);

    // Create a Factor Graph and Values to hold the new data
    NonlinearFactorGraph graph;
    Values initialEstimate;

    // Main loop for all images
    for (size_t i = 0; i < AllFrames.size(); i++)
    {
        for (size_t j = 0; j < AllFrames[i].dr_poses.rows; j++)
        {
            Pose3 pose_dr = Pose3(
                Rot3::Rodrigues(AllFrames[i].dr_poses.at<double>(j,0),AllFrames[i].dr_poses.at<double>(j,1),AllFrames[i].dr_poses.at<double>(j,2)), 
                Point3(AllFrames[i].dr_poses.at<double>(j,3), AllFrames[i].dr_poses.at<double>(j,4), AllFrames[i].dr_poses.at<double>(j,5)));

            std::vector<double> seeds;
            for (size_t k = 0; k < 6; k++)
                seeds.push_back(distribution(generator));        
            Pose3 add_noise(Rot3::Rodrigues(seeds[0]*noise_rpy, seeds[1]*noise_rpy, seeds[2]*noise_rpy),
                            Point3(seeds[3]*noise_xyz, seeds[4]*noise_xyz, seeds[5]*noise_xyz));
            
            initialEstimate.insert(Symbol('X', unique_id[i][j]), pose_dr.compose(add_noise));
            // initialEstimate.insert(Symbol('X', unique_id[i][j]), gtsam::Pose3::identity());

            // if it's the first pose of the first image, add fixed prior factor
            if (i==0 && j==0)
            {
                auto PriorModel = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.000001), Vector3::Constant(0.000001))
                                                            .finished());
                graph.addPrior(Symbol('X', unique_id[i][j]), pose_dr, PriorModel);
                continue;
            }

            // if it's the first pose BUT NOT the first image, get previous pose from last image
            if (i!=0 && j==0)
            {
                int id = AllFrames[i-1].dr_poses.rows - 1;
                Pose3 pose_dr_pre = Pose3(
                    Rot3::Rodrigues(AllFrames[i-1].dr_poses.at<double>(id,0),AllFrames[i-1].dr_poses.at<double>(id,1),AllFrames[i-1].dr_poses.at<double>(id,2)), 
                    Point3(AllFrames[i-1].dr_poses.at<double>(id,3), AllFrames[i-1].dr_poses.at<double>(id,4), AllFrames[i-1].dr_poses.at<double>(id,5)));

                auto odo = pose_dr_pre.between(pose_dr);

                auto OdoModel = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3(ro1_, pi1_, ya1_), Vector3(x1_, y1_, z1_))
                                                                .finished());

                graph.add(BetweenFactor<Pose3>(Symbol('X',unique_id[i-1][id]), Symbol('X',unique_id[i][j]), odo, OdoModel));
            }
            // otherwise, get previous pose from last ping
            else
            {
                Pose3 pose_dr_pre = Pose3(
                    Rot3::Rodrigues(AllFrames[i].dr_poses.at<double>(j-1,0),AllFrames[i].dr_poses.at<double>(j-1,1),AllFrames[i].dr_poses.at<double>(j-1,2)), 
                    Point3(AllFrames[i].dr_poses.at<double>(j-1,3), AllFrames[i].dr_poses.at<double>(j-1,4), AllFrames[i].dr_poses.at<double>(j-1,5)));

                auto odo = pose_dr_pre.between(pose_dr);

                auto OdoModel = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3(ro1_, pi1_, ya1_), Vector3(x1_, y1_, z1_))
                                                            .finished());

                graph.add(BetweenFactor<Pose3>(Symbol('X',unique_id[i][j-1]), Symbol('X',unique_id[i][j]), odo, OdoModel));
            }

            // check loop closing constraint
            if (i>0 && ADD_LC)
            {
                // find out which the image pair id it is now
                int img_pair_id = -1;
                for (size_t k = 0; k < img_pairs_ids.size(); k++)
                {
                    if (img_pairs_ids[k].first==i-1 && img_pairs_ids[k].second==i)
                    {
                        img_pair_id = k;
                        break;
                    }
                    
                }
                
                if (img_pair_id == -1)
                    cout << "image id not found..." << endl;
                else
                {
                    // check if current ping has loop closing measurement
                    int kps_id = -1;
                    for (size_t k = 0; k < id_in_kps[img_pair_id].size(); k++)
                    {
                        if (id_in_kps[img_pair_id][k]==unique_id[i][j])
                        {
                            kps_id = k;
                            break;
                        }
                    }
                    
                    // --- if loop closing measurement found, construct factor and add to graph --- //
                    if (kps_id!=-1)
                    {
                        int id_1 = kps_pairs_all[img_pair_id][kps_id](0), id_2 = kps_pairs_all[img_pair_id][kps_id](3);
                        if (id_1>=AllFrames[img_pairs_ids[img_pair_id].first].dr_poses.rows || id_2>=AllFrames[img_pairs_ids[img_pair_id].second].dr_poses.rows)
                            cout << "row index out of range !!! (when add lc constraint in all...)" << endl; 

                        cout << "***********************************************************" << endl;
                        cout << "Add New Loop Closure Constraint" << " ";
                        cout << "between X" << unique_id[img_pairs_ids[img_pair_id].first][id_1] << " and X" << unique_id[img_pairs_ids[img_pair_id].second][id_2] << " ";
                        cout << "(" << id_1 << " and " << id_2 << ")" << endl;

                        // loop closure uncertainty model
                        auto LoopClosureNoiseModel = gtsam::noiseModel::Diagonal::Variances(get<1>(lc_tf_all[img_pair_id][kps_id]));

                        // add loop closure measurement
                        Pose3 lc_tf = get<0>(lc_tf_all[img_pair_id][kps_id]);

                        // add factor to graph
                        graph.add(BetweenFactor<Pose3>(Symbol('X',unique_id[img_pairs_ids[img_pair_id].first][id_1]), Symbol('X',unique_id[img_pairs_ids[img_pair_id].second][id_2]), lc_tf, LoopClosureNoiseModel));

                    }
                }
            }
            
            

            // Update iSAM with the new factors
            isam.update(graph, initialEstimate);
            // One more time
            isam.update();
            Values currentEstimate = isam.calculateEstimate();             

            // Clear the factor graph and values for the next iteration
            graph.resize(0);
            initialEstimate.clear();

        }
        
    }

    // get latest estimated result
    Values FinalEstimate = isam.calculateEstimate();

    // --- Save trajectories (estimated, dead-reckoning, ...) --- //
    std::vector<cv::Mat> dr_poses_all;
    for (size_t i = 0; i < AllFrames.size(); i++)
        dr_poses_all.push_back(AllFrames[i].dr_poses);    
    Optimizer::SaveTrajactoryAll(FinalEstimate,unique_id,dr_poses_all);


    // --- Evaluated with annotated keypoints --- //
    std::vector<std::vector<cv::Mat>> geo_img_all;
    std::vector<std::vector<double>> gras_all;
    for (size_t i = 0; i < AllFrames.size(); i++)
    {
        geo_img_all.push_back(AllFrames[i].geo_img); 
        gras_all.push_back(AllFrames[i].ground_ranges);
    }
    std::vector<std::vector<Vector6>> anno_kps_pairs_all;
    for (size_t i = 0; i < AllFrames.size(); i++)
    {
        for (size_t j = i+1; j < AllFrames.size(); j++)
        {
            std::vector<Vector6> kps_pairs;
            kps_pairs = Optimizer::GetKpsPairs(true,AllFrames[i].anno_kps,AllFrames[i].img_id,AllFrames[j].img_id,
                                               AllFrames[i].altitudes,AllFrames[i].ground_ranges,
                                               AllFrames[j].altitudes,AllFrames[j].ground_ranges);
            anno_kps_pairs_all.push_back(kps_pairs);         
        }      
    }
    Optimizer::EvaluateByAnnosAll(FinalEstimate,unique_id,geo_img_all,gras_all,anno_kps_pairs_all,img_pairs_ids);



    return;
}

void Optimizer::TrajOptimizationPair(Frame &SourceFrame, Frame &TargetFrame)
{
    bool USE_ANNO = 0, SHOW_IMG = 0, ADD_LC = 1; // use annotation or not, show image or not, add loopclosure or not
    if (USE_ANNO && SHOW_IMG)
        Util::ShowAnnos(SourceFrame.img_id,TargetFrame.img_id,SourceFrame.norm_img,TargetFrame.norm_img,
                        SourceFrame.anno_kps,TargetFrame.anno_kps);

    // Noise model paras for pose
    double ro1_ = 0.01*PI/180, pi1_ = 0.01*PI/180, ya1_ = 0.05*PI/180, x1_ = 0.05, y1_ = 0.05, z1_ = 0.01;
    double ro2_ = 0.01*PI/180, pi2_ = 0.01*PI/180, ya2_ = 0.01*PI/180, x2_ = 0.01, y2_ = 0.01, z2_ = 0.01;
    // Noise model paras for keypoint
    double sigma_r = 0.1, alpha_bw =0.1*PI/180;
    // random noise generator
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);
    double noise_xyz = 5, noise_rpy = 5*PI/180;


    std::vector<Vector6> kps_pairs = Optimizer::GetKpsPairs(USE_ANNO,SourceFrame.corres_kps,SourceFrame.img_id,TargetFrame.img_id,
                                                            SourceFrame.altitudes,SourceFrame.ground_ranges,
                                                            TargetFrame.altitudes,TargetFrame.ground_ranges);

    vector<Point3> ini_points = Optimizer::TriangulateLandmarks(kps_pairs, SourceFrame.tf_stb, SourceFrame.tf_port,
                                                                SourceFrame.img_id, TargetFrame.img_id, 
                                                                SourceFrame.geo_img, TargetFrame.geo_img,
                                                                SourceFrame.altitudes, TargetFrame.altitudes,
                                                                SourceFrame.dr_poses, TargetFrame.dr_poses);

    vector<tuple<Pose3,Vector6,double>> lc_tf_Conv = Optimizer::LoopClosingTFs(kps_pairs, SourceFrame.tf_stb, SourceFrame.tf_port,
                                                                       SourceFrame.img_id, TargetFrame.img_id, 
                                                                       SourceFrame.geo_img, TargetFrame.geo_img,
                                                                       SourceFrame.altitudes, TargetFrame.altitudes,
                                                                       SourceFrame.ground_ranges, TargetFrame.ground_ranges,
                                                                       SourceFrame.dr_poses, TargetFrame.dr_poses);

    // --- Assign unique ID for each pose ---//
    int id_tmp = 0;
    std::vector<int> g_id_s(SourceFrame.dr_poses.rows), g_id_t(TargetFrame.dr_poses.rows), g_id_in_kps(kps_pairs.size());;
    for (size_t i = 0; i < SourceFrame.dr_poses.rows; i++)
    {
        g_id_s[i] = id_tmp;
        id_tmp = id_tmp + 1;
    }
    for (size_t i = 0; i < TargetFrame.dr_poses.rows; i++)
    {
        g_id_t[i] = id_tmp;
        id_tmp = id_tmp + 1;
    }
    // // record unique ID for each keypoint pair,
    // // only the ID of second keypoint is recorded;
    for (size_t i = 0; i < kps_pairs.size(); i++)
    {
        int ping_num = kps_pairs[i](3);
        if (ping_num>=TargetFrame.dr_poses.rows)
            cout << "!!! index out of range: " << ping_num << ">" << TargetFrame.dr_poses.rows << endl;      
        g_id_in_kps[i] = g_id_t[ping_num];
    }
    cout << "Total number of pings and keypoint pairs: " << id_tmp << " " << kps_pairs.size() << endl;

    // Create an iSAM2 object.
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    parameters.factorization = ISAM2Params::QR;
    // parameters.optimizationParams = ISAM2DoglegParams();
    parameters.print();
    ISAM2 isam(parameters);

    // Create a Factor Graph and Values to hold the new data
    NonlinearFactorGraph graph;
    // NonlinearFactorGraph graphSAVE;
    Values initialEstimate;

    // // --- loop on the poses of the SOURCE image --- // //
    for (size_t i = 0; i < SourceFrame.dr_poses.rows; i++)
    {
        Pose3 pose_dr = Pose3(
                Rot3::Rodrigues(SourceFrame.dr_poses.at<double>(i,0),SourceFrame.dr_poses.at<double>(i,1),SourceFrame.dr_poses.at<double>(i,2)), 
                Point3(SourceFrame.dr_poses.at<double>(i,3), SourceFrame.dr_poses.at<double>(i,4), SourceFrame.dr_poses.at<double>(i,5)));

        std::vector<double> seeds;
        for (size_t j = 0; j < 6; j++)
            seeds.push_back(distribution(generator));        
        Pose3 add_noise(Rot3::Rodrigues(seeds[0]*noise_rpy, seeds[1]*noise_rpy, seeds[2]*noise_rpy),
                        Point3(seeds[3]*noise_xyz, seeds[4]*noise_xyz, seeds[5]*noise_xyz));
        
        initialEstimate.insert(Symbol('X', g_id_s[i]), pose_dr.compose(add_noise));
        // initialEstimate.insert(Symbol('X', g_id_s[i]), gtsam::Pose3::identity());

        // if it's the first pose, add fixed prior factor
        if (i==0)
        {
            auto PriorModel = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.000001), Vector3::Constant(0.000001))
                                                           .finished());
            graph.addPrior(Symbol('X', g_id_s[i]), pose_dr, PriorModel);
            // graphSAVE.addPrior(Symbol('X', g_id_s[i]), pose_dr, PriorModel);

        }
        // add odometry factor and update isam
        else
        {
            Pose3 pose_dr_pre = Pose3(
                    Rot3::Rodrigues(SourceFrame.dr_poses.at<double>(i-1,0),SourceFrame.dr_poses.at<double>(i-1,1),SourceFrame.dr_poses.at<double>(i-1,2)), 
                    Point3(SourceFrame.dr_poses.at<double>(i-1,3), SourceFrame.dr_poses.at<double>(i-1,4), SourceFrame.dr_poses.at<double>(i-1,5)));

            auto odo = pose_dr_pre.between(pose_dr);

            auto OdoModel = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3(ro1_, pi1_, ya1_), Vector3(x1_, y1_, z1_))
                                                           .finished());

            graph.add(BetweenFactor<Pose3>(Symbol('X',g_id_s[i-1]), Symbol('X',g_id_s[i]), odo, OdoModel));
            // graphSAVE.add(BetweenFactor<Pose3>(Symbol('X',g_id_s[i-1]), Symbol('X',g_id_s[i]), odo, OdoModel));

            // Update iSAM with the new factors
            isam.update(graph, initialEstimate);
            // One more time
            isam.update();
            Values currentEstimate = isam.calculateEstimate();
            // cout << "Updating Current Ping #" << g_id_s[i] << ": " << endl;
            // cout << currentEstimate.at<Pose3>(Symbol('X',g_id_s[i])).translation() << endl;

            // Clear the factor graph and values for the next iteration
            graph.resize(0);
            initialEstimate.clear();
        }
          
    }


    // // --- loop on the poses of the TARGET image --- // //
    for (size_t i = 0; i < TargetFrame.dr_poses.rows; i++)
    {
        Pose3 pose_dr = Pose3(
                Rot3::Rodrigues(TargetFrame.dr_poses.at<double>(i,0),TargetFrame.dr_poses.at<double>(i,1),TargetFrame.dr_poses.at<double>(i,2)), 
                Point3(TargetFrame.dr_poses.at<double>(i,3), TargetFrame.dr_poses.at<double>(i,4), TargetFrame.dr_poses.at<double>(i,5)));

        std::vector<double> seeds;
        for (size_t j = 0; j < 6; j++)
            seeds.push_back(distribution(generator));        
        Pose3 add_noise(Rot3::Rodrigues(seeds[0]*noise_rpy, seeds[1]*noise_rpy, seeds[2]*noise_rpy),
                        Point3(seeds[3]*noise_xyz, seeds[4]*noise_xyz, seeds[5]*noise_xyz));
        
        initialEstimate.insert(Symbol('X', g_id_t[i]), pose_dr.compose(add_noise));
        // initialEstimate.insert(Symbol('X', g_id_t[i]), gtsam::Pose3::identity());


        // // get the last pose from end of last image 
        // // if it is the start pose in current image
        if (i==0)
        {
            int id = SourceFrame.dr_poses.rows - 1;
            Pose3 pose_dr_pre = Pose3(
                    Rot3::Rodrigues(SourceFrame.dr_poses.at<double>(id,0),SourceFrame.dr_poses.at<double>(id,1),SourceFrame.dr_poses.at<double>(id,2)), 
                    Point3(SourceFrame.dr_poses.at<double>(id,3), SourceFrame.dr_poses.at<double>(id,4), SourceFrame.dr_poses.at<double>(id,5)));

            auto odo = pose_dr_pre.between(pose_dr);

            auto OdoModel = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3(ro1_, pi1_, ya1_), Vector3(x1_, y1_, z1_))
                                                            .finished());

            graph.add(BetweenFactor<Pose3>(Symbol('X',g_id_s[id]), Symbol('X',g_id_t[i]), odo, OdoModel));
        }
        else
        {
            Pose3 pose_dr_pre = Pose3(
                    Rot3::Rodrigues(TargetFrame.dr_poses.at<double>(i-1,0),TargetFrame.dr_poses.at<double>(i-1,1),TargetFrame.dr_poses.at<double>(i-1,2)), 
                    Point3(TargetFrame.dr_poses.at<double>(i-1,3), TargetFrame.dr_poses.at<double>(i-1,4), TargetFrame.dr_poses.at<double>(i-1,5)));

            auto odo = pose_dr_pre.between(pose_dr);

            auto OdoModel = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3(ro1_, pi1_, ya1_), Vector3(x1_, y1_, z1_))
                                                            .finished());

            graph.add(BetweenFactor<Pose3>(Symbol('X',g_id_t[i-1]), Symbol('X',g_id_t[i]), odo, OdoModel));

        } 

        // check if current ping has loop closing measurement
        int kps_id = -1;
        for (size_t j = 0; j < g_id_in_kps.size(); j++)
        {
            if (g_id_in_kps[j]==g_id_t[i])
            {
                kps_id = j;
                break;
            }
        }

        // --- if loop closing measurement found, construct factor and add to graph --- //
        if (kps_id!=-1 && ADD_LC)
        {
            cout << "***********************************************************" << endl;
            cout << "Add New Loop Closure Constraint" << " ";
            cout << "between X" << g_id_s[(int)kps_pairs[kps_id](0)] << " and X" << g_id_t[(int)kps_pairs[kps_id](3)] << " ";
            cout << "(" << kps_pairs[kps_id](0) << " and " << kps_pairs[kps_id](3) << ")" << endl;

            // loop closure uncertainty model
            auto LoopClosureNoiseModel = gtsam::noiseModel::Diagonal::Variances(get<1>(lc_tf_Conv[kps_id]));

            // add loop closure measurement
            Pose3 lc_tf = get<0>(lc_tf_Conv[kps_id]);

            // add fator to graph
            int id_s = kps_pairs[kps_id](0), id_t = kps_pairs[kps_id](3);
            if (id_s>=SourceFrame.dr_poses.rows || id_t>=TargetFrame.dr_poses.rows)
                cout << "row index out of range !!! (when add lc constrain...)" << endl; 
            graph.add(BetweenFactor<Pose3>(Symbol('X',g_id_s[id_s]), Symbol('X',g_id_t[id_t]), lc_tf, LoopClosureNoiseModel));

        }      
        
        // Update iSAM with the new factors
        isam.update(graph, initialEstimate);
        // One more time
        isam.update();
        Values currentEstimate = isam.calculateEstimate();

        // Show results before and after optimization
        bool printinfo = 0;
        if (printinfo && kps_id!=-1 && ADD_LC)
        {
            int id_t = kps_pairs[kps_id](3);
            cout << "NEW POSE 2: " << endl << currentEstimate.at<Pose3>(Symbol('X',g_id_t[id_t])).translation() << endl;
            cout << "OLD POSE 2: " << endl << pose_dr.translation() << endl;
        }
        else if (0)
        {
            cout << "Updating Current Ping #" << g_id_t[i] << ": " << endl;
            cout << currentEstimate.at<Pose3>(Symbol('X',g_id_t[i])).translation() << endl;
        }

        // Clear the factor graph and values for the next iteration
        graph.resize(0);
        initialEstimate.clear();

    }
    cout << endl;

    Values FinalEstimate = isam.calculateEstimate();

    Optimizer::SaveTrajactoryPair(FinalEstimate,g_id_s,g_id_t,SourceFrame.dr_poses,TargetFrame.dr_poses);
    Optimizer::EvaluateByAnnos(FinalEstimate,SourceFrame.img_id,TargetFrame.img_id,
                               g_id_s,g_id_t,
                               SourceFrame.geo_img,TargetFrame.geo_img,
                               SourceFrame.ground_ranges,TargetFrame.ground_ranges,
                               SourceFrame.anno_kps,TargetFrame.anno_kps,
                               kps_pairs);


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

std::vector<tuple<Pose3,Vector6,double>>  Optimizer::LoopClosingTFs(const std::vector<Vector6> &kps_pairs, 
                                                const std::vector<double> &tf_stb, const std::vector<double> &tf_port,
                                                const int &img_id_s, const int &img_id_t,
                                                const std::vector<cv::Mat> &geo_s, const std::vector<cv::Mat> &geo_t,
                                                const std::vector<double> &alts_s, const std::vector<double> &alts_t,
                                                const std::vector<double> &gras_s, const std::vector<double> &gras_t,
                                                const cv::Mat &dr_poses_s, const cv::Mat &dr_poses_t)
{
    bool save_result = 0;

    ofstream save_result_1, save_result_2;
    if (save_result)
    {
        string path1 = "../ini_lm_errors.txt";
        save_result_1.open(path1.c_str(),ios::trunc);
        string path2 = "../fnl_lm_errors.txt";
        save_result_2.open(path2.c_str(),ios::trunc);
    }
    

    Pose3 cps_pose = gtsam::Pose3::identity();
    if (img_id_s%2!=img_id_t%2)
        cps_pose = Pose3(Rot3::Rodrigues(0.0, 0.0, PI), Point3(0.0,0.0,0.0));
    

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);
    
    std::vector<tuple<Pose3,Vector6,double>> output_tfs;

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
            double ro_ = 0.1*PI/180, pi_ = 0.1*PI/180, ya_ = 1.5*PI/180, x_ = abs(Tp_st.x()*2), y_ = abs(Tp_st.y()/10), z_ = 0.5;
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
        // params.setVerbosityLM("SUMMARY");
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
            cout << "Add New KP Measurement: L" << i << " ";
            cout << "between ping " << kps_pairs[i](0) << " and " << kps_pairs[i](3) << " " << endl;
            // cout << "NEW POSE: " << endl << new_pose.translation() << endl;
            // cout << "OLD POSE: " << endl << Tp_t.translation() << endl;
            // cout << "INI POSE: " << endl << (Tp_t.compose(add_noise)).translation() << endl;
        }

        // evaluate if the estimation improves after optimization
        bool eval_1 = 1;
        double lm_dist_compare = 0;
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
            lm_dist_compare = ini_point_dist-final_point_dist;

            if (save_result)
            {
                save_result_1 << ini_point_dist << endl;
                save_result_2 << final_point_dist << endl;
            }  

        }

        Marginals marginals(graph, result, Marginals::QR);

        output_tfs.push_back(std::make_tuple(Tp_s.between(result.at<Pose3>(Symbol('X',2))*cps_pose.inverse()), 
                                             marginals.marginalCovariance(Symbol('X',2)).diagonal(),
                                             lm_dist_compare));

        // Clear the factor graph and values for the next iteration
        graph.resize(0);
        initialEstimate.clear();
    }

    if (save_result)
    {
        save_result_1.close();
        save_result_2.close();
    }
    
    
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

void Optimizer::SaveTrajactoryPair(const Values &FinalEstimate, 
                                   const std::vector<int> &g_id_s, const std::vector<int> &g_id_t,
                                   const cv::Mat &dr_poses_s, const cv::Mat &dr_poses_t)
{

    // --- Save dead-reckoning results --- //

    ofstream save_result_1;
    string path1 = "../dr_poses.txt";
    save_result_1.open(path1.c_str(),ios::trunc);

    for (size_t i = 0; i < dr_poses_s.rows; i++)
    {
        Pose3 save_pose = Pose3(
                Rot3::Rodrigues(dr_poses_s.at<double>(i,0),dr_poses_s.at<double>(i,1),dr_poses_s.at<double>(i,2)), 
                Point3(dr_poses_s.at<double>(i,3), dr_poses_s.at<double>(i,4), dr_poses_s.at<double>(i,5)));
        save_result_1 << fixed << setprecision(9) << save_pose.rotation().quaternion()(1) << " " << save_pose.rotation().quaternion()(2) << " "
                      << save_pose.rotation().quaternion()(3) << " " << save_pose.rotation().quaternion()(0) << " " << save_pose.x() << " " 
                      << save_pose.y() << " " << save_pose.z() << endl;
    }
    for (size_t i = 0; i < dr_poses_t.rows; i++)
    {
        Pose3 save_pose = Pose3(
                Rot3::Rodrigues(dr_poses_t.at<double>(i,0),dr_poses_t.at<double>(i,1),dr_poses_t.at<double>(i,2)), 
                Point3(dr_poses_t.at<double>(i,3), dr_poses_t.at<double>(i,4), dr_poses_t.at<double>(i,5)));
        save_result_1 << fixed << setprecision(9) << save_pose.rotation().quaternion()(1) << " " << save_pose.rotation().quaternion()(2) << " "
                      << save_pose.rotation().quaternion()(3) << " " << save_pose.rotation().quaternion()(0) << " " << save_pose.x() << " " 
                      << save_pose.y() << " " << save_pose.z() << endl;
    }

    save_result_1.close();

    // --- Save optimized results --- //

    ofstream save_result_2;
    string path2 = "../est_poses.txt";
    save_result_2.open(path2.c_str(),ios::trunc);   

    for (size_t i = 0; i < g_id_s.size(); i++)
    {
        Pose3 save_pose = FinalEstimate.at<Pose3>(Symbol('X',g_id_s[i]));
        save_result_2 << fixed << setprecision(9) << save_pose.rotation().quaternion()(1) << " " << save_pose.rotation().quaternion()(2) << " "
                      << save_pose.rotation().quaternion()(3) << " " << save_pose.rotation().quaternion()(0) << " " << save_pose.x() << " " 
                      << save_pose.y() << " " << save_pose.z() << endl;
        
    }
    for (size_t i = 0; i < g_id_t.size(); i++)
    {
        Pose3 save_pose = FinalEstimate.at<Pose3>(Symbol('X',g_id_t[i]));
        save_result_2 << fixed << setprecision(9) << save_pose.rotation().quaternion()(1) << " " << save_pose.rotation().quaternion()(2) << " "
                      << save_pose.rotation().quaternion()(3) << " " << save_pose.rotation().quaternion()(0) << " " << save_pose.x() << " " 
                      << save_pose.y() << " " << save_pose.z() << endl;
        
    }
     
    save_result_2.close();


    return;
}

void Optimizer::SaveTrajactoryAll(const Values &FinalEstimate, const std::vector<std::vector<int>> &unique_id,
                                      const std::vector<cv::Mat> &dr_poses_all)
{

    // --- Save dead-reckoning results --- //

    ofstream save_result_1;
    string path1 = "../dr_poses_all.txt";
    save_result_1.open(path1.c_str(),ios::trunc);

    for (size_t i = 0; i < dr_poses_all.size(); i++)
    {
        for (size_t j = 0; j < dr_poses_all[i].rows; j++)
        {
            Pose3 save_pose = Pose3(
                    Rot3::Rodrigues(dr_poses_all[i].at<double>(j,0),dr_poses_all[i].at<double>(j,1),dr_poses_all[i].at<double>(j,2)), 
                    Point3(dr_poses_all[i].at<double>(j,3), dr_poses_all[i].at<double>(j,4), dr_poses_all[i].at<double>(j,5)));
            save_result_1 << fixed << setprecision(9) << save_pose.rotation().quaternion()(1) << " " << save_pose.rotation().quaternion()(2) << " "
                        << save_pose.rotation().quaternion()(3) << " " << save_pose.rotation().quaternion()(0) << " " << save_pose.x() << " " 
                        << save_pose.y() << " " << save_pose.z() << endl;
        }
    }

    save_result_1.close();

    // --- Save optimized results --- //

    ofstream save_result_2;
    string path2 = "../est_poses_all.txt";
    save_result_2.open(path2.c_str(),ios::trunc);

    for (size_t i = 0; i < unique_id.size(); i++)
    {
        for (size_t j = 0; j < unique_id[i].size(); j++)
        {
            Pose3 save_pose = FinalEstimate.at<Pose3>(Symbol('X',unique_id[i][j]));
            save_result_2 << fixed << setprecision(9) << save_pose.rotation().quaternion()(1) << " " << save_pose.rotation().quaternion()(2) << " "
                        << save_pose.rotation().quaternion()(3) << " " << save_pose.rotation().quaternion()(0) << " " << save_pose.x() << " " 
                        << save_pose.y() << " " << save_pose.z() << endl;
            
        }
    }
     
    save_result_2.close();

    return;
}

void Optimizer::EvaluateByAnnos(const Values &FinalEstimate, const int &img_id_s, const int &img_id_t,
                                const std::vector<int> &g_id_s, const std::vector<int> &g_id_t,
                                const std::vector<cv::Mat> &geo_s, const std::vector<cv::Mat> &geo_t,
                                const std::vector<double> &gras_s, const std::vector<double> &gras_t,
                                const cv::Mat &anno_kps_s, const cv::Mat &anno_kps_t,
                                const std::vector<Vector6> &kps_pairs_est)
{
    bool show_est = 0, save_result = 0;

    ofstream save_result_1, save_result_2, save_result_3;
    if (save_result)
    { 
        string path1 = "../dr_lm_dist.txt";
        save_result_1.open(path1.c_str(),ios::trunc);
        string path2 = "../est_lm_dist.txt";
        save_result_2.open(path2.c_str(),ios::trunc);
        string path3 = "../lm_dist_compare.txt";
        save_result_3.open(path3.c_str(),ios::trunc);
    }

    // -- get all the keypoint pairs --- //
    std::vector<Vector4> kps_pairs;
    std::vector<bool> close_to_est(anno_kps_s.rows,false);
    int close_thres = 15;
    for (size_t i = 0; i < anno_kps_s.rows; i++)
    {
        // decide which frame id is the target (associated) frame
        int id_check;
        std::vector<int> kp_s, kp_t;
        id_check = anno_kps_s.at<int>(i,1);
        kp_s = {anno_kps_s.at<int>(i,2),anno_kps_s.at<int>(i,3)};
        kp_t = {anno_kps_s.at<int>(i,4),anno_kps_s.at<int>(i,5)};

        // save keypoint pairs
        if (id_check==img_id_t)
        {
            Vector4 kp_pair = (gtsam::Vector4() << kp_s[0], kp_s[1], kp_t[0], kp_t[1]).finished();
            kps_pairs.push_back(kp_pair);

            // for (size_t i = 0; i < kp_pair.size(); i++)
            //     cout << kp_pair(i) << " ";
            // cout << endl;

        }

        // check whether this pair is close to any of the estimated correspondences
        for (size_t j = 0; j < kps_pairs_est.size(); j++)
        {
            int id_est = kps_pairs_est[j](0);
            if (abs(id_est-kp_s[0])<close_thres)
            {
                close_to_est[i]=true;
                break;
            }     
        }
        
        
    }

    for (size_t i = 0; i < kps_pairs.size(); i++)
    {
        double x_dist, y_dist;
        int id_s = kps_pairs[i](0), id_t = kps_pairs[i](2);
        if (id_s>=geo_s[0].rows || id_t>=geo_t[0].rows)
            cout << "row index out of range !!! (in evaluation)" << endl;  
        int id_ss = kps_pairs[i](1), id_tt = kps_pairs[i](3);
        if (id_ss>=geo_s[0].cols || id_tt>=geo_t[0].cols)
            cout << "column index out of range !!! (in evaluation)" << endl;

        // --- initial landmark distance observed between two dr ping poses --- //  
        x_dist = (geo_s[0].at<double>(id_s,id_ss)-geo_t[0].at<double>(id_t,id_tt));
        y_dist = (geo_s[1].at<double>(id_s,id_ss)-geo_t[1].at<double>(id_t,id_tt));
        double ini_point_dist = sqrt(x_dist*x_dist + y_dist*y_dist);

        // --- final landmark distance observed between two estimated ping poses --- //
        double lm_geo_s_x, lm_geo_s_y, lm_geo_t_x, lm_geo_t_y;

        Pose3 new_pose_s = FinalEstimate.at<Pose3>(Symbol('X', g_id_s[id_s]));
        if (kps_pairs[i](1)<geo_s[0].cols/2)
        {
            int gr_idx = geo_s[0].cols/2 - kps_pairs[i](1);
            lm_geo_s_x = new_pose_s.x() + gras_s[gr_idx]*cos(new_pose_s.rotation().yaw()+PI/2-PI);
            lm_geo_s_y = new_pose_s.y() + gras_s[gr_idx]*sin(new_pose_s.rotation().yaw()+PI/2-PI);
        }
        else
        {
            int gr_idx = kps_pairs[i](1) - geo_s[0].cols/2;
            lm_geo_s_x = new_pose_s.x() + gras_s[gr_idx]*cos(new_pose_s.rotation().yaw()-PI/2-PI);
            lm_geo_s_y = new_pose_s.y() + gras_s[gr_idx]*sin(new_pose_s.rotation().yaw()-PI/2-PI);
        }

        Pose3 new_pose_t = FinalEstimate.at<Pose3>(Symbol('X', g_id_t[id_t]));
        if (kps_pairs[i](3)<geo_t[0].cols/2)
        {
            int gr_idx = geo_t[0].cols/2 - kps_pairs[i](3);
            lm_geo_t_x = new_pose_t.x() + gras_t[gr_idx]*cos(new_pose_t.rotation().yaw()+PI/2-PI);
            lm_geo_t_y = new_pose_t.y() + gras_t[gr_idx]*sin(new_pose_t.rotation().yaw()+PI/2-PI);
        }
        else
        {
            int gr_idx = kps_pairs[i](3) - geo_t[0].cols/2;
            lm_geo_t_x = new_pose_t.x() + gras_t[gr_idx]*cos(new_pose_t.rotation().yaw()-PI/2-PI);
            lm_geo_t_y = new_pose_t.y() + gras_t[gr_idx]*sin(new_pose_t.rotation().yaw()-PI/2-PI);
        }
        x_dist = (lm_geo_s_x-lm_geo_t_x);
        y_dist = (lm_geo_s_y-lm_geo_t_y);
        double final_point_dist = sqrt(x_dist*x_dist + y_dist*y_dist);   

        if (1)
        {
            cout << "lm distance (ini VS fnl) at SourcePing #" << id_s << " :" << ini_point_dist << " " << final_point_dist << " "
                <<  ini_point_dist-final_point_dist << endl;
        }

        if (save_result)
        {
            save_result_1 << ini_point_dist << endl;
            save_result_2 << final_point_dist << endl;   
            save_result_3 << ini_point_dist-final_point_dist << endl;       
        }

    }

    if (save_result)
    {
        save_result_1.close();
        save_result_2.close();
        save_result_3.close();
    }
    
    
    // show distances of estimated keypoints
    if (show_est)
    {
        for (size_t i = 0; i < kps_pairs_est.size(); i++)
        {
            double x_dist, y_dist;
            int id_s = kps_pairs_est[i](0), id_t = kps_pairs_est[i](3);
            if (id_s>=geo_s[0].rows || id_t>=geo_t[0].rows)
                cout << "row index out of range !!! (in evaluation)" << endl;  
            int id_ss = kps_pairs_est[i](1), id_tt = kps_pairs_est[i](4);
            if (id_ss>=geo_s[0].cols || id_tt>=geo_t[0].cols)
                cout << "column index out of range !!! (in evaluation)" << endl;

            // --- initial landmark distance observed between two dr ping poses --- //  
            x_dist = (geo_s[0].at<double>(id_s,id_ss)-geo_t[0].at<double>(id_t,id_tt));
            y_dist = (geo_s[1].at<double>(id_s,id_ss)-geo_t[1].at<double>(id_t,id_tt));
            double ini_point_dist = sqrt(x_dist*x_dist + y_dist*y_dist);

            // --- final landmark distance observed between two estimated ping poses --- //
            double lm_geo_s_x, lm_geo_s_y, lm_geo_t_x, lm_geo_t_y;

            Pose3 new_pose_s = FinalEstimate.at<Pose3>(Symbol('X', g_id_s[id_s]));
            if (kps_pairs_est[i](1)<geo_s[0].cols/2)
            {
                int gr_idx = geo_s[0].cols/2 - kps_pairs_est[i](1);
                lm_geo_s_x = new_pose_s.x() + gras_s[gr_idx]*cos(new_pose_s.rotation().yaw()+PI/2-PI);
                lm_geo_s_y = new_pose_s.y() + gras_s[gr_idx]*sin(new_pose_s.rotation().yaw()+PI/2-PI);
            }
            else
            {
                int gr_idx = kps_pairs_est[i](1) - geo_s[0].cols/2;
                lm_geo_s_x = new_pose_s.x() + gras_s[gr_idx]*cos(new_pose_s.rotation().yaw()-PI/2-PI);
                lm_geo_s_y = new_pose_s.y() + gras_s[gr_idx]*sin(new_pose_s.rotation().yaw()-PI/2-PI);
            }

            Pose3 new_pose_t = FinalEstimate.at<Pose3>(Symbol('X', g_id_t[id_t]));
            if (kps_pairs_est[i](4)<geo_t[0].cols/2)
            {
                int gr_idx = geo_t[0].cols/2 - kps_pairs_est[i](4);
                lm_geo_t_x = new_pose_t.x() + gras_t[gr_idx]*cos(new_pose_t.rotation().yaw()+PI/2-PI);
                lm_geo_t_y = new_pose_t.y() + gras_t[gr_idx]*sin(new_pose_t.rotation().yaw()+PI/2-PI);
            }
            else
            {
                int gr_idx = kps_pairs_est[i](4) - geo_t[0].cols/2;
                lm_geo_t_x = new_pose_t.x() + gras_t[gr_idx]*cos(new_pose_t.rotation().yaw()-PI/2-PI);
                lm_geo_t_y = new_pose_t.y() + gras_t[gr_idx]*sin(new_pose_t.rotation().yaw()-PI/2-PI);
            }
            x_dist = (lm_geo_s_x-lm_geo_t_x);
            y_dist = (lm_geo_s_y-lm_geo_t_y);
            double final_point_dist = sqrt(x_dist*x_dist + y_dist*y_dist);   

            if (1)
            {
                cout << "lm distance (ini VS fnl) at SourcePing #" << id_s << " :" << ini_point_dist << " " << final_point_dist << " "
                    <<  final_point_dist-ini_point_dist << endl;
            }

        }
    }
    




    return;
}

void Optimizer::EvaluateByAnnosAll(const Values &FinalEstimate, const std::vector<std::vector<int>> &unique_id,
                                    const std::vector<std::vector<cv::Mat>> &geo_img_all,
                                    const std::vector<std::vector<double>> &gras_all,
                                    const std::vector<std::vector<Vector6>> &kps_pairs_all,
                                    const std::vector<pair<int,int>> &img_pairs_ids)
{

    bool save_result = 1, show_result = 0;

    // loop for keypoint pairs in each image pair
    for (size_t i = 0; i < kps_pairs_all.size(); i++)
    {
        ofstream save_result_1, save_result_2, save_result_3;
        if (save_result)
        { 
            string path1 = "../result/anno_errors/dr_lm_dist_" + std::to_string(i) + ".txt";
            save_result_1.open(path1.c_str(),ios::trunc);
            string path2 = "../result/anno_errors/est_lm_dist_" + std::to_string(i) + ".txt";
            save_result_2.open(path2.c_str(),ios::trunc);
            string path3 = "../result/anno_errors/lm_dist_compare_" + std::to_string(i) + ".txt";
            save_result_3.open(path3.c_str(),ios::trunc);
        }

        // loop for each keypoint pair in current image pair
        for (size_t j = 0; j < kps_pairs_all[i].size(); j++)
        {
            double x_dist, y_dist;

            int id_s = kps_pairs_all[i][j](0), id_t = kps_pairs_all[i][j](3);
            if (id_s>=geo_img_all[img_pairs_ids[i].first][0].rows || id_t>=geo_img_all[img_pairs_ids[i].second][0].rows)
                cout << "row index out of range !!! (in evaluation all)" << endl;  
            int id_ss = kps_pairs_all[i][j](1), id_tt = kps_pairs_all[i][j](4);
            if (id_ss>=geo_img_all[img_pairs_ids[i].first][0].cols || id_tt>=geo_img_all[img_pairs_ids[i].second][0].cols)
                cout << "column index out of range !!! (in evaluation all)" << endl;

            // --- initial landmark distance observed between two DR ping poses --- //  
            x_dist = (geo_img_all[img_pairs_ids[i].first][0].at<double>(id_s,id_ss)-geo_img_all[img_pairs_ids[i].second][0].at<double>(id_t,id_tt));
            y_dist = (geo_img_all[img_pairs_ids[i].first][1].at<double>(id_s,id_ss)-geo_img_all[img_pairs_ids[i].second][1].at<double>(id_t,id_tt));
            double ini_point_dist = sqrt(x_dist*x_dist + y_dist*y_dist);

            // --- final landmark distance observed between two estimated ping poses --- //
            double lm_geo_s_x, lm_geo_s_y, lm_geo_t_x, lm_geo_t_y;

            Pose3 new_pose_s = FinalEstimate.at<Pose3>(Symbol('X', unique_id[img_pairs_ids[i].first][id_s]));
            if (kps_pairs_all[i][j](1)<geo_img_all[img_pairs_ids[i].first][0].cols/2)
            {
                int gr_idx = geo_img_all[img_pairs_ids[i].first][0].cols/2 - kps_pairs_all[i][j](1);
                lm_geo_s_x = new_pose_s.x() + gras_all[img_pairs_ids[i].first][gr_idx]*cos(new_pose_s.rotation().yaw()+PI/2-PI);
                lm_geo_s_y = new_pose_s.y() + gras_all[img_pairs_ids[i].first][gr_idx]*sin(new_pose_s.rotation().yaw()+PI/2-PI);
            }
            else
            {
                int gr_idx = kps_pairs_all[i][j](1) - geo_img_all[img_pairs_ids[i].first][0].cols/2;
                lm_geo_s_x = new_pose_s.x() + gras_all[img_pairs_ids[i].first][gr_idx]*cos(new_pose_s.rotation().yaw()-PI/2-PI);
                lm_geo_s_y = new_pose_s.y() + gras_all[img_pairs_ids[i].first][gr_idx]*sin(new_pose_s.rotation().yaw()-PI/2-PI);
            }

            Pose3 new_pose_t = FinalEstimate.at<Pose3>(Symbol('X', unique_id[img_pairs_ids[i].second][id_t]));
            if (kps_pairs_all[i][j](4)<geo_img_all[img_pairs_ids[i].second][0].cols/2)
            {
                int gr_idx = geo_img_all[img_pairs_ids[i].second][0].cols/2 - kps_pairs_all[i][j](4);
                lm_geo_t_x = new_pose_t.x() + gras_all[img_pairs_ids[i].second][gr_idx]*cos(new_pose_t.rotation().yaw()+PI/2-PI);
                lm_geo_t_y = new_pose_t.y() + gras_all[img_pairs_ids[i].second][gr_idx]*sin(new_pose_t.rotation().yaw()+PI/2-PI);
            }
            else
            {
                int gr_idx = kps_pairs_all[i][j](4) - geo_img_all[img_pairs_ids[i].second][0].cols/2;
                lm_geo_t_x = new_pose_t.x() + gras_all[img_pairs_ids[i].second][gr_idx]*cos(new_pose_t.rotation().yaw()-PI/2-PI);
                lm_geo_t_y = new_pose_t.y() + gras_all[img_pairs_ids[i].second][gr_idx]*sin(new_pose_t.rotation().yaw()-PI/2-PI);
            }
            x_dist = (lm_geo_s_x-lm_geo_t_x);
            y_dist = (lm_geo_s_y-lm_geo_t_y);
            double final_point_dist = sqrt(x_dist*x_dist + y_dist*y_dist);

            if (save_result)
            {
                save_result_1 << ini_point_dist << endl;
                save_result_2 << final_point_dist << endl;   
                save_result_3 << ini_point_dist-final_point_dist << endl;       
            } 

            if (show_result)
            {
                cout << "lm distance (ini VS fnl) at SourcePing #" << id_s << " :" << ini_point_dist << " " << final_point_dist << " "
                    <<  ini_point_dist-final_point_dist << endl;
            } 

        }

    if (save_result)
    {
        save_result_1.close();
        save_result_2.close();
        save_result_3.close();
    }
        
    }

    return;
}


} // namespace Diasss
