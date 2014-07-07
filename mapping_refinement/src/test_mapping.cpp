#include "scan.h"
#include "fine_registration.h"
#include "asynch_visualizer.h"
#include "fine_edge.h"
#include "fine_vertex.h"

#include <Eigen/Dense>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include "g2o/core/sparse_optimizer.h"
//#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/linear_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
//#include "g2o/solvers/slam2d_linear/solver_slam2d_linear.h"
//#include "g2o/solvers/csparse/g2o_csparse_api.h"
//#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/slam3d/vertex_se3.h"

bool register_scans(Eigen::Matrix3f& R, Eigen::Vector3f& t, scan* scan1, scan* scan2)
{
    Eigen::Matrix3f R_orig;
    Eigen::Vector3f t_orig;
    scan1->get_transform(R_orig, t_orig);
    Eigen::Quaternionf q = Eigen::Quaternionf(R);
    q.normalize();
    Eigen::Vector3f q_orig = q.vec().segment<3>(0);
    fine_registration r(*scan1, *scan2);
    q.setIdentity();
    Eigen::Vector3f t_last;
    t_last.setZero();
    Eigen::Vector3f q_last;
    q_last.setZero();
    Eigen::Vector3f q_diff, t_diff;
    do {
        t_last.setZero(); // if only taking into account how big current is
        q_last.setZero();
        r.step(R, t);
        //scan2->transform(R.transpose(), -R.transpose()*t);
        scan1->transform(R, t);
        float error = r.error();
        q = Eigen::Quaternionf(R);
        q.normalize();
        q_diff = q.vec().segment<3>(0) - q_last; // might as well take all four?
        t_diff = t - t_last;
        t_last = t;
        q_last = q.vec().segment<3>(0);
        std::cout << "Error: " << error << std::endl;
        std::cout << "Translation diff: " << t_diff.norm() << std::endl;
        std::cout << "Rotation diff: " << q_diff.norm() << std::endl;
    }
    while (t_diff.norm() > 0.003 || q_diff.norm() > 0.0008);
    scan1->get_transform(R, t);
    scan1->set_transform(R_orig, t_orig);
    std::cout << "REGISTRATION FINISHED!" << std::endl;
    t_diff = t_last - t_orig;
    q_diff = q_last - q_orig;
    if (t_diff.norm() > 0.05 || q_diff.norm() > 0.005) {
        return false;
    }
    else {
        return true;
    }
}

int main2(int argc, char** argv)
{
    /*boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    // Starting visualizer
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();*/

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1> >  SlamBlockSolver;
    typedef g2o::LinearSolverPCG<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

    g2o::SparseOptimizer optimizer;
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);
    solver->setUserLambdaInit(0); // 0
    optimizer.setAlgorithm(solver);

    std::string folder = std::string(getenv("HOME")) + std::string("/.ros/mapping_refinement");
    std::vector<std::string> scans;
    std::vector<std::string> transforms;
    std::vector<scan*> scan_objects;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;
    std::vector<fine_vertex*> vertices;
    size_t n = 34;
    clouds.resize(n);
    vertices.resize(n);
    scan_objects.resize(n);
    for (size_t i = 0; i < n; ++i) {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << i;
        scans.push_back(folder + std::string("/shot") + ss.str() + std::string(".pcd"));
        transforms.push_back(folder + std::string("/transform") + ss.str() + std::string(".txt"));
        /*vertices[i] = new fine_vertex(scans.back(), transforms.back());
        vertices[i]->get_transform(R, t);
        vertices[i]->setId(i);
        optimizer.addVertex(vertices[i]);*/

        scan_objects[i] = new scan(scans.back(), transforms.back());
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        /*clouds[i] = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::io::loadPCDFile<pcl::PointXYZRGB>(scans.back(), *clouds[i]);
        for (size_t j = 0; j < clouds[i]->points.size(); ++j) {
            clouds[i]->points[j].getVector3fMap() = R*clouds[i]->points[j].getVector3fMap() + t;
        }
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(clouds[i]);
        viewer->addPointCloud<pcl::PointXYZRGB>(clouds[i], rgb, std::string("cloud") + ss.str());
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 1, std::string("cloud") + ss.str());*/
        std::cout << "Cloud: " << i << std::endl;
    }

    /*fine_edge* test = new fine_edge(*vertices[1], *vertices[3]);
    Eigen::Matrix<double, 1, 1, 0, 1, 1> info;// = 0.00001*Eigen::Matrix<double, 6, 6>::setIdentity();
    info(0, 0) = 1.0;
    test->setInformation(info);
    optimizer.addEdge(test);*/

    //optimizer.initializeOptimization();
    std::cout << "Optimizing..." << std::endl;
    optimizer.setVerbose(true);
    //optimizer.optimize(10);
    std::cout << "Done optimizing!" << std::endl;

    /*while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        //boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }*/
    return 0;
}

int main(int argc, char** argv)
{
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1> >  SlamBlockSolver;
    typedef g2o::LinearSolverPCG<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

    g2o::SparseOptimizer optimizer;
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);
    solver->setUserLambdaInit(0); // 0
    optimizer.setAlgorithm(solver);

    std::string folder = std::string(getenv("HOME")) + std::string("/.ros/mapping_refinement");
    std::vector<std::string> scan_files;
    std::vector<std::string> t_files;
    std::vector<scan*> scans;
    size_t n = 34;

    // set the filenames for the pointclouds and transforms, initialize scan objects
    for (size_t i = 0; i < n; ++i) {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << i;
        scan_files.push_back(folder + std::string("/shot") + ss.str() + std::string(".pcd"));
        t_files.push_back(folder + std::string("/transform") + ss.str() + std::string(".txt"));
        scans.push_back(new scan(scan_files.back(), t_files.back()));
    }

    // adding the odometry to the optimizer
    // first adding all the vertices
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    Eigen::Matrix4d T;
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > estimates;
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
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i+1)%n;
        size_t k = (i+2)%n;
        if (i % 2 == 0) {
            correct = register_scans(R, t, scans[i], scans[j]);
            if (correct) {
                std::cout << "CORRECT!" << std::endl;
            }
            else {
                std::cout << "INCORRECT!" << std::endl;
            }
            T.topLeftCorner<3, 3>() = R.cast<double>();
            T.block<3, 1>(0, 3) = t.cast<double>();
            Eigen::Isometry3d transform(T);
            pairs.push_back(edge_pair(i, j));
            measurements.push_back(transform);
            measurement_correct.push_back(correct);
        }
        correct = register_scans(R, t, scans[i], scans[k]);
        T.topLeftCorner<3, 3>() = R.cast<double>();
        T.block<3, 1>(0, 3) = t.cast<double>();
        Eigen::Isometry3d transform(T);
        pairs.push_back(edge_pair(i, k));
        measurements.push_back(transform);
        measurement_correct.push_back(correct);
    }

    // add vertices to optimizer
    for (size_t i = 0; i < n; ++i) {
        g2o::VertexSE3* robot = new g2o::VertexSE3;
        robot->setId(i);
        robot->setEstimate(estimates[i]);
        optimizer.addVertex(robot);
    }

    // good information
    Eigen::Matrix<double, 6, 6> good_info;
    good_info.setIdentity();

    // bad information
    Eigen::Matrix<double, 6, 6> bad_info;
    bad_info.setIdentity();

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
            odometry->setInformation(bad_info);
        }
        optimizer.addEdge(odometry);
    }
    std::cout << "done." << std::endl;
}
