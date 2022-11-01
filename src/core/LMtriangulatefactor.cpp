
#include "LMtriangulatefactor.h"

namespace Diasss
{

// error function
// @param p        the 3d point in Point3
// @param H        the optional Jacobian matrixes, which use boost optional and has default null pointer
gtsam::Vector LMTriaFactor::evaluateError(const gtsam::Point3& p, boost::optional<gtsam::Matrix&> H) const {

    gtsam::Point3 p_s = Ts_.transformTo( Tp_.transformTo(p) );

    // note that use boost optional like a pointer
    // only calculate jacobian matrix when non-null pointer exists
    if (H)
    {
        gtsam::Rot3 J_s_kp = Ts_.rotation().inverse()*Tp_.rotation().inverse();
        gtsam::Vector3 row_1 = p_s.transpose()*J_s_kp.matrix()/gtsam::norm3(p_s);
        gtsam::Vector3 row_2 = ((gtsam::Vector3() << 1.0, 0.0, 0.0).finished()).transpose()*J_s_kp.matrix();
        *H = (gtsam::Matrix23() << row_1(0), row_1(1), row_1(2), 
                                   row_2(0), row_2(1), row_2(2)).finished();
    }
    
    // return error vector
    return (gtsam::Vector2() << gtsam::norm3(p_s) - mx_, p_s.x() - my_).finished();
}

} /// namespace diasss
