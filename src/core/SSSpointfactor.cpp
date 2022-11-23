
#include "SSSpointfactor.h"

namespace Diasss
{

// error function
// @param p        the 3d point in Point3
// @param T        the 6d pose in Pose3
// @param H1,H2    the optional Jacobian matrixes, which use boost optional and has default null pointer
gtsam::Vector SssPointFactor::evaluateError(const gtsam::Point3& p, const gtsam::Pose3& T,
                                            boost::optional<gtsam::Matrix&> H1,
                                            boost::optional<gtsam::Matrix&> H2) const {

    gtsam::Point3 p_s = Ts_.transformTo( T.transformTo(p) );

    // note that use boost optional like a pointer
    // only calculate jacobian matrix when non-null pointer exists
    if (H1)
    {
        gtsam::Rot3 J_s_kp = Ts_.rotation().inverse()*T.rotation().inverse();
        gtsam::Vector3 row_1 = p_s.transpose()*J_s_kp.matrix()/gtsam::norm3(p_s);
        gtsam::Vector3 row_2 = ((gtsam::Vector3() << 1.0, 0.0, 0.0).finished()).transpose()*J_s_kp.matrix();
        *H1 = (gtsam::Matrix23() << row_1(0), row_1(1), row_1(2), 
                                    row_2(0), row_2(1), row_2(2)).finished();
    }

    if (H2)
    {
        bool plan_a = 1;

        gtsam::Matrix3 block_t;
        gtsam::Matrix36 J_s_pose;
        
        if (plan_a)
        {
            block_t = -(Ts_.rotation().inverse()*T.rotation().inverse()).matrix();
            gtsam::Point3 p_m = T.transformTo(p);
            gtsam::Matrix3 m_x = (gtsam::Matrix3() << 0.0, -p_m.z(), p_m.y(),
                                                       p_m.z(), 0.0, -p_m.x(),
                                                       -p_m.y(), p_m.x(), 0.0).finished();
            gtsam::Matrix3 block_r = Ts_.rotation().inverse().matrix()*m_x;
            J_s_pose =  (gtsam::Matrix36() << block_r(0,0),block_r(0,1),block_r(0,2),block_t(0,0),block_t(0,1),block_t(0,2),
                                              block_r(1,0),block_r(1,1),block_r(1,2),block_t(1,0),block_t(1,1),block_t(1,2),
                                              block_r(2,0),block_r(2,1),block_r(2,2),block_t(2,0),block_t(2,1),block_t(2,2)).finished();
        }
        else
        {

            gtsam::Vector3 ypr = T.rotation().ypr();
            double ya = ypr(0), pi = ypr(1), ro = ypr(2);

            gtsam::Rot3 rot_dev_r = gtsam::Rot3(0, 0, 0,
                                    cos(ya)*sin(pi)*cos(ro)+sin(ya)*sin(ro), sin(ya)*sin(pi)*cos(ro)-cos(ya)*sin(ro), cos(pi)*cos(ro),
                                    -cos(ya)*sin(pi)*sin(ro)+sin(ya)*cos(ro), -sin(ya)*sin(pi)*sin(ro)-cos(ya)*cos(ro), -cos(pi)*sin(ro));
            gtsam::Rot3 rot_dev_p = gtsam::Rot3(-cos(ya)*sin(pi), -sin(ya)*sin(pi), -cos(pi),
                                    cos(ya)*cos(pi)*sin(ro), sin(ya)*cos(pi)*sin(ro), -sin(pi)*sin(ro),
                                    cos(ya)*cos(pi)*cos(ro), sin(ya)*cos(pi)*cos(ro), -sin(pi)*cos(ro));
            gtsam::Rot3 rot_dev_y = gtsam::Rot3(-sin(ya)*cos(pi), cos(ya)*cos(pi), 0,
                                    -sin(ya)*sin(pi)*sin(ro)-cos(ya)*cos(ro), cos(ya)*sin(pi)*sin(ro)-sin(ya)*cos(ro), 0,
                                    -sin(ya)*sin(pi)*cos(ro)+cos(ya)*sin(ro), cos(ya)*sin(pi)*sin(ro)+sin(ya)*sin(ro), 0);

            block_t = -(Ts_.rotation().inverse()*T.rotation().inverse()).matrix();
            gtsam::Vector3 col_4 = (Ts_.rotation().inverse()*rot_dev_r).rotate(p-T.translation());
            gtsam::Vector3 col_5 = (Ts_.rotation().inverse()*rot_dev_p).rotate(p-T.translation());
            gtsam::Vector3 col_6 = (Ts_.rotation().inverse()*rot_dev_y).rotate(p-T.translation());
            J_s_pose =  (gtsam::Matrix36() << col_4(0),col_5(0),col_6(0),block_t(0,0),block_t(0,1),block_t(0,2),
                                              col_4(1),col_5(1),col_6(1),block_t(1,0),block_t(1,1),block_t(1,2),
                                              col_4(2),col_5(2),col_6(2),block_t(2,0),block_t(2,1),block_t(2,2)).finished();
        }
        
        gtsam::Vector6 row_1 = p_s.transpose()*J_s_pose/gtsam::norm3(p_s);
        gtsam::Vector6 row_2 = ((gtsam::Vector3() << 1.0, 0.0, 0.0).finished()).transpose()*J_s_pose;
        *H2 = (gtsam::Matrix26() << row_1(0), row_1(1), row_1(2), row_1(3), row_1(4), row_1(5), 
                                    row_2(0), row_2(1), row_2(2), row_2(3), row_2(4), row_2(5)).finished();
    }
    
    // return error vector
    return (gtsam::Vector2() << gtsam::norm3(p_s) - mx_, p_s.x() - my_).finished();
}

} /// namespace diasss
