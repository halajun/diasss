
#ifndef LMTRIAFACTOR_H
#define LMTRIAFACTOR_H

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

namespace Diasss
{

using namespace std;

    class GTSAM_EXPORT LMTriaFactor: public gtsam::NoiseModelFactor1<gtsam::Point3>
    {

        private:
        
        // measurement information
        double mx_, my_;
        // sensor offset
        gtsam::Pose3 Ts_;
        // pose of ping
        gtsam::Pose3 Tp_;


        public:

        /**
         * Constructor
         * @param pointKey   point varible key
         * @param model      noise model for the keypoint
         * @param m          Vector2 measurement
         */
        LMTriaFactor(gtsam::Key pointKey, const gtsam::Vector2 &m, const gtsam::Pose3 &Ts, const gtsam::Pose3 &Tp, gtsam::SharedNoiseModel model) :
            gtsam::NoiseModelFactor1<gtsam::Point3>(model, pointKey), mx_(m.x()), my_(m.y()), Ts_(Ts), Tp_(Tp) {}

        // error function
        // @param p        the 3d point in Point3
        // @param H        the optional Jacobian matrixs, which use boost optional and has default null pointer
        gtsam::Vector evaluateError(const gtsam::Point3& p, boost::optional<gtsam::Matrix&> H = boost::none) const override;

    };

}

#endif // LMTRIAFACTOR_H