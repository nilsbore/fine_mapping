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

int main(int argc, char** argv)
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
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;
    std::vector<fine_vertex*> vertices;
    size_t n = 3;
    clouds.resize(n);
    vertices.resize(n);
    for (size_t i = 0; i < n; ++i) {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << i;
        scans.push_back(folder + std::string("/shot") + ss.str() + std::string(".pcd"));
        transforms.push_back(folder + std::string("/transform") + ss.str() + std::string(".txt"));
        vertices[i] = new fine_vertex(scans.back(), transforms.back());
        vertices[i]->get_transform(R, t);
        vertices[i]->setId(i);
        optimizer.addVertex(vertices[i]);
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

    fine_edge* test = new fine_edge(*vertices[1], *vertices[2]);
    Eigen::Matrix<double, 1, 1, 0, 1, 1> info;// = 0.00001*Eigen::Matrix<double, 6, 6>::setIdentity();
    info(0, 0) = 1;
    test->setInformation(info);
    optimizer.addEdge(test);

    /*for (size_t i = 1; i < n-1; ++i) {
        size_t j = (i+1)%n;
        size_t k = (i+2)%n;
        if (i % 2 == 0) {
            fine_edge* up = new fine_edge(vertices[i], vertices[j]);
            optimizer.addEdge(up);
        }
        if (k < n || k == n) {
            fine_edge* side = new fine_edge(vertices[i], vertices[k]);
            optimizer.addEdge(side);
        }
    }*/

    optimizer.initializeOptimization();
    std::cout << "Optimizing..." << std::endl;
    optimizer.setVerbose(true);
    optimizer.optimize(10);
    std::cout << "Done optimizing!" << std::endl;

    /*while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        //boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }*/
    return 0;
}
