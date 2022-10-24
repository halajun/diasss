
/**
 * A simple 2D side-scan sonar 2d point factor
 * The factor contains a X-Y position measurement (mx, my) for a 2d point observed by a side-scan sonar sensor;
 * The error vector will be [x-mx, y-my]'
 */

#ifndef SSSPOINTFACTOR_H
#define SSSPOINTFACTOR_H

#include "SSSpointfactor.h"

namespace Diasss
{

// error function
// @param p        the 3d point in Point3
// @param T        the 6d pose in Pose3
// @param H1,H2    the optional Jacobian matrixes, which use boost optional and has default null pointer
gtsam::Vector SssPointFactor::evaluateError(const gtsam::Point3& p, const gtsam::Pose3& T,
                            boost::optional<gtsam::Matrix&> H1 = boost::none,
                            boost::optional<gtsam::Matrix&> H2 = boost::none) const {

    gtsam::Point3 p_s = Ts_.transformTo( T.transformTo(p) );
    gtsam::Vector3 ypr = T.rotation().ypr();

    gtsam::Rot3 rot_dev_r = gtsam::Rot3(0, 0, 0,
                            cos(y)*sin(p)*cos(r)+sin(y)*sin(r), sin(y)*sin(p)*cos(r)-cos(y)*sin(r), cos(p)*cos(r),
                            -cos(y)*sin(p)*sin(r)+sin(y)*cos(r), -sin(y)*sin(p)*sin(r)-cos(y)*cos(r), -cos(p)*sin(r));
    gtsam::Rot3 rot_dev_p = gtsam::Rot3(-cos(y)*sin(p), -sin(y)*sin(p), -cos(p),
                            cos(y)*cos(p)*sin(r), sin(y)*cos(p)*sin(r), -sin(p)*sin(r),
                            cos(y)*cos(p)*cos(r), sin(y)*cos(p)*cos(r), -sin(p)*cos(r));
    gtsam::Rot3 rot_dev_y = gtsam::Rot3(-sin(y)*cos(p), cos(y)*cos(p), 0,
                            -sin(y)*sin(p)*sin(r)-cos(y)*cos(r), cos(y)*sin(p)*sin(r)-sin(y)*cos(r), 0,
                            -sin(y)*sin(p)*cos(r)+cos(y)*sin(r), cos(y)*sin(p)*sin(r)+sin(y)*sin(r), 0);

    // note that use boost optional like a pointer
    // only calculate jacobian matrix when non-null pointer exists
    if (H1)
    {
        gtsam::Rot3 J_s_kp = Ts_.rotation().inverse()*T.rotation().inverse();
        gtsam::Vector3 row_1 = p_s.transpose()*J_s_kp/p_s.norm3();
        gtsam::Vector3 row_2 = ((gtsam::Vector3() << 1.0, 0.0, 0.0).finished()).transpose()*J_s_kp;
        *H1 = (gtsam::Matrix23() << row_1(0), row_1(1), row_1(2), 
                                    row_2(0), row_2(1), row_2(2)).finished();
    }

    if (H2)
    {
        gtsam::Matrix3 block_t = -(Ts_.rotation().inverse()*T.rotation().inverse()).matrix();
        gtsam::Vector3 col_4 = Ts_.rotation().inverse()*rot_dev_r.rotate(p_s-T.translation());
        gtsam::Vector3 col_5 = Ts_.rotation().inverse()*rot_dev_p.rotate(p_s-T.translation());
        gtsam::Vector3 col_6 = Ts_.rotation().inverse()*rot_dev_y.rotate(p_s-T.translation());
        gtsam::Matrix36 J_s_pose =  (gtsam::Matrix36() << col_4(0),col_5(0),col_6(0),block_t(0,0),block_t(0,1),block_t(0,2),
                                                            col_4(1),col_5(1),col_6(1),block_t(1,0),block_t(1,1),block_t(1,2),
                                                            col_4(2),col_5(2),col_6(2),block_t(2,0),block_t(2,1),block_t(2,2)).finished();
        gtsam::Vector3 row_1 = p_s.transpose()*J_s_pose/p_s.norm3();
        gtsam::Vector3 row_2 = ((gtsam::Vector3() << 1.0, 0.0, 0.0).finished()).transpose()*J_s_pose;
        *H2 = (gtsam::Matrix26() << row_1(0), row_1(1), row_1(2), row_1(3), row_1(4), row_1(5), 
                                    row_2(0), row_2(1), row_2(2), row_2(3), row_2(4), row_2(5)).finished();
    }
    
    // return error vector
    return (gtsam::Vector2() << p_s.norm3() - mx_, p_s.x() - my_).finished();
}

} /// namespace diasss

#endif // SSSPOINTFACTOR_H