#ifndef FINE_VERTEX_H
#define FINE_VERTEX_H

#include "scan.h"

// oplus, oplus_impl
// constructor
// check vertex_SO3
#include "g2o/config.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/hyper_graph_action.h"
#include "g2o/types/slam3d/isometry3d_mappings.h"
#include "g2o/types/slam3d/g2o_types_slam3d_api.h"

class fine_vertex : public scan, public g2o::BaseVertex<6, Eigen::Isometry3d> {
public:

    static const int orthogonalizeAfter = 1000; //< orthogonalize the rotation matrix after N updates

    void setToOriginImpl() {
        _estimate = Eigen::Isometry3d::Identity();
    }

    bool read(std::istream& is);
    bool write(std::ostream& os) const;

    /*bool setEstimateDataImpl(const double* est){
        //Eigen::Map<const g2o::Vector6d> v(est);
        //_estimate=internal::fromVectorQT(v);
        return true;
    }

    bool getEstimateData(double* est) const{
        //Map<Vector7d> v(est);
        //v=internal::toVectorQT(_estimate);
        return true;
    }

    int estimateDimension() const {
        return 7;
    }

    bool setMinimalEstimateDataImpl(const double* est){
        //Map<const Vector6d> v(est);
        //_estimate = internal::fromVectorMQT(v);
        return true;
    }

    bool getMinimalEstimateData(double* est) const{
        //Map<Vector6d> v(est);
        //v = internal::toVectorMQT(_estimate);
        return true;
    }

    int minimalEstimateDimension() const {
        return 6;
    }*/

    /**
       * update the position of this vertex. The update is in the form
       * (x,y,z,qx,qy,qz) whereas (x,y,z) represents the translational update
       * and (qx,qy,qz) corresponds to the respective elements. The missing
       * element qw of the quaternion is recovred by
       * || (qw,qx,qy,qz) || == 1 => qw = sqrt(1 - || (qx,qy,qz) ||
       */
    void oplusImpl(const double* update)
    {
        Eigen::Map<const g2o::Vector6d> v(update);
        Eigen::Isometry3d increment = g2o::internal::fromVectorMQT(v);
        _estimate = _estimate * increment;
        Eigen::Matrix4f T = _estimate.matrix().cast<float>();
        basis = T.topLeftCorner<3, 3>();
        origin = T.block<3, 1>(0, 3);
        /*if (++_numOplusCalls > orthogonalizeAfter) {
            _numOplusCalls = 0;
            g2o::internal::approximateNearestOrthogonalMatrix(_estimate.matrix().topLeftCorner<3,3>());
        }*/
    }

    fine_vertex(const std::string& pcdname, const std::string& tname) : scan(pcdname, tname), g2o::BaseVertex<6, Eigen::Isometry3d >() {}
    fine_vertex() : scan(), g2o::BaseVertex<6, Eigen::Isometry3d>() {}
    fine_vertex(const fine_vertex& vertex) : scan(vertex), g2o::BaseVertex<6, Eigen::Isometry3d>(vertex) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // FINE_VERTEX_H
