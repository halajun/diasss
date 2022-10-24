
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "frame.h"

namespace Diasss
{

using namespace std;

    class Optimizer
    {

    public:

        // Trajectory optimization for a pair of side-scan waterfall images
        void static TrajOptimizationPair(Frame &SourceFrame, Frame &TargetFrame);

        

    };

}

#endif // OPTIMIZER_H