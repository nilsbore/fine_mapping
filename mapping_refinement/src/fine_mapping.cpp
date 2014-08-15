#include "fine_mapping.h"

#include "fine_registration.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "g2o/core/block_solver.h"
#include "g2o/core/linear_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/slam3d/edge_se3.h"

#include <string>

void fine_mapping::compute_initial_transformation(Eigen::Matrix3f& R, Eigen::Vector3f& t, scan* scan1, scan* scan2)
{
    return;
    Eigen::Matrix3f R1, R2, Rd;
    Eigen::Vector3f t1, t2, td;
    scan1->get_transform(R1, t1);
    scan2->get_transform(R2, t2);
    Rd = R1.transpose()*R2;
    td = R1.transpose()*(t2-t1);
    Eigen::AngleAxisf a(Rd.transpose()*R);
    float angle = fmod(fabs(a.angle()), 2*M_PI);
    if (angle < 0.06 && (t - td).norm() < 0.1) {
        std::cout << "Correct, angle: " << angle << ", translation: " << (t - td).norm() << std::endl;
        return;
    }
    std::cout << "Incorrect, angle: " << angle << ", translation: " << (t - td).norm() << std::endl;
}

void fine_mapping::build_graph()
{
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1> >  SlamBlockSolver;
    typedef g2o::LinearSolverPCG<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);
    solver->setUserLambdaInit(5); // 0
    optimizer.setAlgorithm(solver);

    size_t n = scans.size();

    // adding the odometry to the optimizer
    // first adding all the vertices
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    Eigen::Matrix4d T;
    T.setIdentity();
    std::cout << "Optimization: Adding robot poses ... ";
    for (size_t i = 0; i < n; ++i) {
        scans[i]->get_transform(R, t);
        T.topLeftCorner<3, 3>() = R.cast<double>();
        T.block<3, 1>(0, 3) = t.cast<double>();
        Eigen::Isometry3d transform(T);
        estimates.push_back(transform);
    }
    std::cout << "done." << std::endl;

    typedef std::pair<size_t, size_t> edge_pair;
    std::vector<edge_pair> pairs;
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > measurements;
    std::vector<bool> measurement_correct;

    bool correct;
    for (size_t i = 0; i < n; ++i) { // n
        for (size_t j = 0; j < i; ++j) {
            if (!scans[i]->overlaps_with(*scans[j])) {
                continue;
            }
            correct = fine_registration::register_scans(R, t, scans[i], scans[j]);
            compute_initial_transformation(R, t, scans[i], scans[j]);
            T.topLeftCorner<3, 3>() = R.cast<double>();
            T.block<3, 1>(0, 3) = t.cast<double>();
            Eigen::Isometry3d transform(T);
            pairs.push_back(edge_pair(i, j));
            measurements.push_back(transform);
            measurement_correct.push_back(correct);
        }
    }

    // add vertices to optimizer
    for (size_t i = 0; i < n; ++i) {
        g2o::VertexSE3* robot = new g2o::VertexSE3;
        robot->setId(i);
        robot->setEstimate(estimates[i]);
        optimizer.addVertex(robot);
        vertices.push_back(robot);
    }
    // fix the origin
    vertices[0]->setFixed(true);
    vertices[n-1]->setFixed(true);

    // fix the opposite side of the scan
    /*float maxprod = 0.0;
    size_t maxi;
    scans[0]->get_transform(R, t);
    Eigen::Vector3f p0 = R.col(2);
    for (size_t i = 0; i < n; ++i) {
        scans[i]->get_transform(R, t);
        float prod = p0.dot(R.col(2));
        if (prod < maxprod) {
            maxprod = prod;
            maxi = i;
        }
    }
    vertices[maxi]->setFixed(true);*/

    // good information
    Eigen::Matrix<double, 6, 6> good_info;
    good_info.setIdentity();
    good_info.bottomRightCorner<3, 3>() *= 100.0;

    // bad information
    Eigen::Matrix<double, 6, 6> bad_info;
    bad_info.setIdentity();
    bad_info.bottomRightCorner<3, 3>() *= 100.0;
    bad_info /= 10.0;

    // second add the odometry constraints
    std::cout << "Optimization: Adding odometry measurements ... ";
    for (size_t i = 0; i < measurements.size(); ++i) {
        g2o::EdgeSE3* odometry = new g2o::EdgeSE3;
        odometry->vertices()[0] = optimizer.vertex(pairs[i].first);
        odometry->vertices()[1] = optimizer.vertex(pairs[i].second);
        odometry->setMeasurement(measurements[i]);
        if (measurement_correct[i]) {
            odometry->setInformation(good_info);
        }
        else {
            odometry->setInformation(bad_info);//bad_info
        }
        optimizer.addEdge(odometry);
    }
    std::cout << "done." << std::endl;
}

void fine_mapping::optimize_graph()
{
    optimizer.initializeOptimization();
    std::cout << "Optimizing..." << std::endl;
    optimizer.setVerbose(true);
    optimizer.optimize(10);
    std::cout << "Done optimizing!" << std::endl;

    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    size_t n = scans.size();
    // set the scans to the transforms in g2o
    for (size_t i = 0; i < n; ++i) {
        Eigen::Isometry3d estimate = vertices[i]->estimate();
        Eigen::Matrix4f transform = estimate.matrix().cast<float>();
        R = transform.topLeftCorner<3, 3>();
        t = transform.block<3, 1>(0, 3);
        scans[i]->set_transform(R, t);
    }

    for (g2o::HyperGraph::Edge* e : optimizer.edges()) {
        g2o::EdgeSE3* es = (g2o::EdgeSE3*)e;
        std::cout << es->error().transpose() << std::endl;
    }
}
