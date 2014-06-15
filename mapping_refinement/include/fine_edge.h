#ifndef FINE_EDGE_H
#define FINE_EDGE_H

#include "fine_registration.h"
#include "fine_vertex.h"

#include "g2o/core/base_binary_edge.h"

// compute_error
// jacobian
// check edge_SO3
//#include <Eigen/Dense>
//#include "g2o/types/slam3d/isometry3d_mappings.h"

class fine_edge : public fine_registration, public g2o::BaseBinaryEdge<6, Eigen::Matrix<double, 6, 1>, fine_vertex, fine_vertex> {
public:
    typedef Eigen::Matrix<double, 6, 1> pos_type;

    /*void setMeasurement(const pos_type& m) {
        _measurement = m;
        //_inverseMeasurement = m.inverse();
    }

    bool setMeasurementData(const double* d) {
        Eigen::Map<const pos_type> v(d);
        setMeasurement(g2o::internal::fromVectorQT(v));
        return true;
    }

    bool getMeasurementData(double* d) const {
        Eigen::Map<pos_type> v(d);
        v = g2o::internal::toVectorQT(_measurement);
        return true;
    }*/

    void computeError();

    void linearizeOplus();

    int measurementDimension() const { return 6; }

    bool setMeasurementFromState();

    fine_edge();

//    virtual double initialEstimatePossible(const OptimizableGraph::VertexSet& /*from*/,
//                                           OptimizableGraph::Vertex* /*to*/) {
//        return 1.;
//    }

//    virtual void initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* to);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
protected:
    //Isometry3d _inverseMeasurement;
};

#endif // FINE_EDGE_H
