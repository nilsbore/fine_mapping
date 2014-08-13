#ifndef FINE_MAPPING_H
#define FINE_MAPPING_H

#include <Eigen/Dense>
#include "g2o/core/sparse_optimizer.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "scan.h"

class fine_mapping {
private:
    std::vector<scan*> scans;
    g2o::SparseOptimizer optimizer;
    std::vector<g2o::VertexSE3*> vertices;
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > estimates;
    void compute_initial_transformation(Eigen::Matrix3f& R, Eigen::Vector3f& t, scan* scan1, scan* scan2);
public:
    void build_graph();
    void optimize_graph();
    fine_mapping(std::vector<scan*> scans) : scans(scans) {}
};

#endif // FINE_MAPPING_H
