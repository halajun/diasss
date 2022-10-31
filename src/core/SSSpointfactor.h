
/**
 * A side-scan sonar 3d point factor
 * The factor contains a slant-range and plane constraint measurement (mx = sqrt(X), my = X(0));
 * The error vector will be [r-mx, 0-my]'
 */

#ifndef SSSPOINTFACTOR_H
#define SSSPOINTFACTOR_H

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

namespace Diasss
{

using namespace std;

    class SssPointFactor: public gtsam::NoiseModelFactor2<gtsam::Point3, gtsam::Pose3>
    {

        private:
        
        // measurement information
        double mx_, my_;
        // sensor offset
        gtsam::Pose3 Ts_;


        public:

        /**
         * Constructor
         * @param pointKey   point varible key
         * @param poseKey    pose variable key
         * @param model      noise model for SSS sensor
         * @param m          Point2 measurement
         */
        SssPointFactor(gtsam::Key pointKey, gtsam::Key poseKey, const gtsam::Point2 &m, const gtsam::Pose3 &Ts, gtsam::SharedNoiseModel model) :
            gtsam::NoiseModelFactor2<gtsam::Point3, gtsam::Pose3>(model, pointKey, poseKey), mx_(m.x()), my_(m.y()), Ts_(Ts) {}

        // error function
        // @param p        the 3d point in Point3
        // @param T        the 6d pose in Pose3
        // @param H1,H2    the optional Jacobian matrixs, which use boost optional and has default null pointer
        gtsam::Vector evaluateError(const gtsam::Point3& p, const gtsam::Pose3& T,
                                    boost::optional<gtsam::Matrix&> H1 = boost::none,
                                    boost::optional<gtsam::Matrix&> H2 = boost::none) const;

    };

}

#endif // SSSPOINTFACTOR_H