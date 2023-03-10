
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <fstream>
#include <iomanip>

#include "frame.h"
#include "SSSpointfactor.h"
#include "LMtriangulatefactor.h"

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>

#include <gtsam/nonlinear/ISAM2.h>
// #include <gtsam/nonlinear/NonlinearISAM.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
// #include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>


namespace Diasss
{

using namespace std;
using namespace cv;
using namespace gtsam;

    class Optimizer
    {

    public:

        // Trajectory optimization for a pair of side-scan waterfall images
        void static TrajOptimizationPair(Frame &SourceFrame, Frame &TargetFrame);

        void static TrajOptimizationAll(std::vector<Frame> &AllFrames);

        std::vector<Vector7> static GetKpsPairs(const bool &USE_ANNO, const cv::Mat &kps, const int &id_s, const int &id_t,
                                         const std::vector<double> &alts_s, const std::vector<double> &gras_s,
                                         const std::vector<double> &alts_t, const std::vector<double> &gras_t);

        std::vector<Point3> static TriangulateLandmarks(const std::vector<Vector7> &kps_pairs, 
                                                        const std::vector<double> &tf_stb, const std::vector<double> &tf_port,
                                                        const int &img_id_s, const int &img_id_t,
                                                        const std::vector<cv::Mat> &geo_s, const std::vector<cv::Mat> &geo_t,
                                                        const std::vector<double> &alts_s, const std::vector<double> &alts_t,
                                                        const cv::Mat &dr_poses_s, const cv::Mat &dr_poses_t);

        Point3 static TriangulateOneLandmark(const Vector7 &kps_pair, 
                                             const Pose3 &Ts_s, const Pose3 &Ts_t,
                                             const Pose3 &Tp_s, const Pose3 &Tp_t,
                                             const Point3 &lm_ini);

        std::vector<tuple<Pose3,Vector6,double>> static LoopClosingTFs(const std::vector<Vector7> &kps_pairs, 
                                                        const std::vector<double> &tf_stb, const std::vector<double> &tf_port,
                                                        const int &img_id_s, const int &img_id_t,
                                                        const std::vector<cv::Mat> &geo_s, const std::vector<cv::Mat> &geo_t,
                                                        const std::vector<double> &alts_s, const std::vector<double> &alts_t,
                                                        const std::vector<double> &gras_s, const std::vector<double> &gras_t,
                                                        const cv::Mat &dr_poses_s, const cv::Mat &dr_poses_t);

        void static SaveTrajactoryPair(const Values &FinalEstimate, 
                                       const std::vector<int> &g_id_s, const std::vector<int> &g_id_t,
                                       const cv::Mat &dr_poses_s, const cv::Mat &dr_poses_t);

        void static SaveTrajactoryAll(const Values &FinalEstimate, const std::vector<std::vector<int>> &unique_id,
                                      const std::vector<cv::Mat> &dr_poses_all);

        void static EvaluateByAnnos(const Values &FinalEstimate, const int &img_id_s, const int &img_id_t,
                                    const std::vector<int> &g_id_s, const std::vector<int> &g_id_t,
                                    const std::vector<cv::Mat> &geo_s, const std::vector<cv::Mat> &geo_t,
                                    const std::vector<double> &gras_s, const std::vector<double> &gras_t,
                                    const cv::Mat &anno_kps_s, const cv::Mat &anno_kps_t,
                                    const std::vector<double> &tf_stb, const std::vector<double> &tf_port,
                                    const cv::Mat &dr_poses_s, const cv::Mat &dr_poses_t,
                                    const std::vector<double> &alts_s, const std::vector<double> &alts_t,
                                    const std::vector<Vector7> &kps_pairs_est);

        void static EvaluateByAnnosAll(const Values &FinalEstimate, const std::vector<std::vector<int>> &unique_id,
                                       const std::vector<std::vector<cv::Mat>> &geo_img_all,
                                       const std::vector<std::vector<double>> &gras_all,
                                       const std::vector<std::vector<Vector7>> &kps_pairs_all,
                                       const std::vector<pair<int,int>> &img_pairs_ids,
                                       const std::vector<cv::Mat> &dr_poses_all,
                                       const std::vector<double> &tf_stb, const std::vector<double> &tf_port,
                                       const std::vector<std::vector<double>> &alts_all);

        

    };

}

#endif // OPTIMIZER_H