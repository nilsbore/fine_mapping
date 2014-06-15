#ifndef FINE_VERTEX_H
#define FINE_VERTEX_H

#include "scan.h"

// oplus, oplus_impl
// constructor
// check vertex_SO3
#include "g2o/config.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/hyper_graph_action.h"

class fine_vertex : public scan, public g2o::BaseVertex<6, Eigen::Matrix<double, 6, 1> > {
public:

    static const int orthogonalizeAfter = 1000; //< orthogonalize the rotation matrix after N updates

    VertexSE3();

    void setToOriginImpl() {
        _estimate = Isometry3d::Identity();
    }

    bool read(std::istream& is);
    bool write(std::ostream& os) const;

    /*bool setEstimateDataImpl(const double* est){
        Map<const Vector7d> v(est);
        _estimate=internal::fromVectorQT(v);
        return true;
    }

    bool getEstimateData(double* est) const{
        Map<Vector7d> v(est);
        v=internal::toVectorQT(_estimate);
        return true;
    }

    int estimateDimension() const {
        return 7;
    }

    bool setMinimalEstimateDataImpl(const double* est){
        Map<const Vector6d> v(est);
        _estimate = internal::fromVectorMQT(v);
        return true;
    }

    bool getMinimalEstimateData(double* est) const{
        Map<Vector6d> v(est);
        v = internal::toVectorMQT(_estimate);
        return true;
    }*/

    int minimalEstimateDimension() const {
        return 6;
    }

    /**
       * update the position of this vertex. The update is in the form
       * (x,y,z,qx,qy,qz) whereas (x,y,z) represents the translational update
       * and (qx,qy,qz) corresponds to the respective elements. The missing
       * element qw of the quaternion is recovred by
       * || (qw,qx,qy,qz) || == 1 => qw = sqrt(1 - || (qx,qy,qz) ||
       */
    void oplusImpl(const double* update)
    {
        Map<const Vector6d> v(update);
        Eigen::Isometry3d increment = internal::fromVectorMQT(v);
        _estimate = _estimate * increment;
        if (++_numOplusCalls > orthogonalizeAfter) {
            g2o::internal::approximateNearestOrthogonalMatrix(_estimate.matrix().topLeftCorner<3,3>());
        }
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // FINE_VERTEX_H
