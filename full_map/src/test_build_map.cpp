#include "simpleXMLparser.h"

#include <semanticMapSummaryParser.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

#include "pointcloud_common.h"

#include "g2o/core/block_solver.h"
#include "g2o/core/linear_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/slam3d/edge_se3.h"

typedef pcl::PointXYZRGB PointType;

typedef typename SemanticMapSummaryParser<PointType>::EntityStruct Entities;

using namespace std;

Eigen::Matrix4f register_scans(double& score, pcl::PointCloud<PointType>::Ptr& input_cloud_trans, pcl::PointCloud<PointType>::Ptr& target_cloud_trans,
                               const Eigen::Vector3f& filtered_translation, const Eigen::Vector3f& target_translation)
{
    pcl::PointCloud<PointType>::Ptr input_cloud(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*input_cloud_trans, *input_cloud, -filtered_translation, Eigen::Quaternionf::Identity());

    pcl::PointCloud<PointType>::Ptr target_cloud(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*target_cloud_trans, *target_cloud, -target_translation, Eigen::Quaternionf::Identity());
    // Initializing Normal Distributions Transform (NDT).
    pcl::NormalDistributionsTransform<PointType, PointType> ndt;

    // Setting scale dependent NDT parameters
    // Setting minimum transformation difference for termination condition.
    ndt.setTransformationEpsilon (0.01);
    // Setting maximum step size for More-Thuente line search.
    ndt.setStepSize(0.1);
    //Setting Resolution of NDT grid structure (VoxelGridCovariance).
    ndt.setResolution(1.0);

    // Setting max number of registration iterations.
    ndt.setMaximumIterations(35);

    // Setting point cloud to be aligned.
    ndt.setInputSource(input_cloud);
    // Setting point cloud to be aligned to.
    ndt.setInputTarget(target_cloud);

    // Set initial alignment estimate found using robot odometry.
    Eigen::AngleAxisf init_rotation = Eigen::AngleAxisf::Identity();
    Eigen::Translation3f init_translation(filtered_translation - target_translation);
    Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();

    // Calculating required rigid transform to align the input cloud to the target cloud.
    pcl::PointCloud<PointType>::Ptr output_cloud(new pcl::PointCloud<PointType>);
    ndt.align(*output_cloud, init_guess);

    std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged()
              << " score: " << ndt.getFitnessScore() << std::endl;

    score = ndt.getFitnessScore();

    // Transforming unfiltered, input cloud using found transform.
    pcl::transformPointCloud(*input_cloud, *output_cloud, ndt.getFinalTransformation());

    // Initializing point cloud visualizer
    boost::shared_ptr<pcl::visualization::PCLVisualizer>
            viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer_final->setBackgroundColor (0, 0, 0);

    // Coloring and visualizing target cloud (red).
    pcl::visualization::PointCloudColorHandlerCustom<PointType>
            target_color (target_cloud, 255, 0, 0);
    viewer_final->addPointCloud<PointType> (target_cloud, target_color, "target cloud");
    viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    1, "target cloud");

    // Coloring and visualizing transformed input cloud (green).
    pcl::visualization::PointCloudColorHandlerCustom<PointType>
            output_color (output_cloud, 0, 255, 0);
    viewer_final->addPointCloud<PointType> (output_cloud, output_color, "output cloud");
    viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    1, "output cloud");

    // Starting visualizer
    viewer_final->addCoordinateSystem (1.0);
    viewer_final->initCameraParameters ();

    // Wait until visualizer window is closed.
    while (!viewer_final->wasStopped ())
    {
        viewer_final->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    viewer_final->close();
}

int main(int argc, char** argv)
{
    SemanticMapSummaryParser<PointType> summary_parser("/home/nbore/Data/Semantic map/index.xml");
    summary_parser.createSummaryXML("/home/nbore/Data/Semantic map/");

    SimpleXMLParser<PointType> simple_parser;
    SimpleXMLParser<PointType>::RoomData roomData;
    typedef SimpleXMLParser<PointType>::CloudPtr CloudPtr;
    
    typedef pair<size_t, size_t> pair_t;
    vector<pair_t> pairs = {pair_t(2, 3), pair_t(4, 5), pair_t(7, 8), pair_t(9, 10), pair_t(11, 12), pair_t(13, 14), pair_t(15, 16),
                           pair_t(17, 18), pair_t(20, 21), pair_t(22, 23), pair_t(25, 26),
                           pair_t(2, 27), pair_t(5, 30), pair_t(6, 31), pair_t(7, 32), pair_t(9, 33), pair_t(12, 35), pair_t(13, 36), pair_t(15, 37),
                           pair_t(18, 28), pair_t(19, 29), pair_t(21, 30), pair_t(23, 31), pair_t(24, 32), pair_t(26, 34),
                           pair_t(27, 28), pair_t(28, 29), pair_t(29, 30), pair_t(30, 31), pair_t(31, 32), pair_t(32, 33),
                           pair_t(33, 34), pair_t(34, 35), pair_t(35, 36), pair_t(36, 37)};

    vector<Entities> allSweeps = summary_parser.getRooms();
    vector<CloudPtr> clouds(38);
    vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > origins(38);

    for (size_t i = 0; i < allSweeps.size(); ++i)
    {
        cout << "Parsing " << allSweeps[i].roomXmlFile << endl;

        roomData = simple_parser.loadRoomFromXML(allSweeps[i].roomXmlFile);
        string wid = roomData.roomWaypointId.substr(8);
        cout << "Waypoint: " << wid << endl;
        int ind = stoi(wid);
        //clouds[ind] = roomData.completeRoomCloud;
        tf::Vector3 tforigin = roomData.vIntermediateRoomCloudTransforms[0].getOrigin();
        Eigen::Vector3f origin;
        for (size_t j = 0; j < 3; ++j) {
            origin(j) = tforigin.m_floats[j];
        }
        origins[ind] = origin;
        //clouds[ind] = CloudPtr(new pcl::PointCloud<PointType>);
        //pcl::transformPointCloud(*roomData.completeRoomCloud, *clouds[ind], -origin, Eigen::Quaternionf::Identity());
        clouds[ind] = roomData.completeRoomCloud;

        /*cout<<"Complete cloud size "<<roomData.completeRoomCloud->points.size()<<endl;
        for (size_t i=0; i<roomData.vIntermediateRoomClouds.size(); i++)
        {
            cout<<"Intermediate cloud size "<<roomData.vIntermediateRoomClouds[i]->points.size()<<endl;
            cout<<"Fx: "<<roomData.vIntermediateRoomCloudCamParams[i].fx()<<" Fy: "<<roomData.vIntermediateRoomCloudCamParams[i].fy()<<endl;
        }*/
    }

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1> >  SlamBlockSolver;
    typedef g2o::LinearSolverPCG<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

    g2o::SparseOptimizer optimizer;
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);
    solver->setUserLambdaInit(5); // 0
    optimizer.setAlgorithm(solver);

    std::vector<g2o::VertexSE3*> vertices(38);
    for (size_t i = 27; i < 38; ++i)
    {
        g2o::VertexSE3* robot = new g2o::VertexSE3;
        robot->setId(i);
        Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
        transform.matrix().block<3, 1>(0, 3) = origins[i].cast<double>();
        robot->setEstimate(transform);
        optimizer.addVertex(robot);
        vertices[i] = robot;
    }
    Eigen::Matrix<double, 6, 6> information;
    information.setIdentity();
    information.bottomRightCorner<3, 3>() *= 1000.0;
    for (size_t i = 27; i < 38; ++i)
    {
        for (size_t j = 27; j < i; ++j) {
            //register(iter->first, iter->second);
            cout << i << ", " << j << endl;
            pcl::PointCloud<PointType>::Ptr temp_first(new pcl::PointCloud<PointType>);
            pcl::PointCloud<PointType>::Ptr temp_second(new pcl::PointCloud<PointType>);
            pointcloud_common<PointType> common(0.3);
            common.set_input(clouds[i]);
            common.set_target(clouds[j]);
            bool overlap = common.segment(*temp_first, *temp_second);
            if (!overlap) {
                continue;
            }
            double score;
            Eigen::Matrix4f res = register_scans(score, temp_first, temp_second, origins[i], origins[j]);
            Eigen::Isometry3d transform(res.cast<double>());
            g2o::EdgeSE3* odometry = new g2o::EdgeSE3;
            odometry->vertices()[0] = optimizer.vertex(i);
            odometry->vertices()[1] = optimizer.vertex(j);
            odometry->setMeasurement(transform);
            odometry->setInformation(information);
            if (score > 0.01) {
                Eigen::Isometry3d first = vertices[i]->estimate();
                Eigen::Isometry3d second = vertices[j]->estimate();
                odometry->setMeasurement(first.inverse()*second);
            }
            optimizer.addEdge(odometry);
        }
        /*vector<pair_t>::iterator iter = pairs.begin();
        while ((iter = std::find_if(iter, pairs.end(), [=](const pair_t& p) { return p.first == i; })) != pairs.end()) {
            //register(iter->first, iter->second);
            cout << iter->first << ", " << iter->second << endl;
            pcl::PointCloud<PointType>::Ptr temp_first(new pcl::PointCloud<PointType>);
            pcl::PointCloud<PointType>::Ptr temp_second(new pcl::PointCloud<PointType>);
            pointcloud_common<PointType> common(0.3);
            common.set_input(clouds[i]);
            common.set_target(clouds[iter->second]);
            bool overlap = common.segment(*temp_first, *temp_second);
            double score;
            Eigen::Matrix4f res = register_scans(score, temp_first, temp_second, origins[i], origins[iter->second]);
            Eigen::Isometry3d transform(res.cast<double>());
            g2o::EdgeSE3* odometry = new g2o::EdgeSE3;
            odometry->vertices()[0] = optimizer.vertex(iter->first);
            odometry->vertices()[1] = optimizer.vertex(iter->second);
            odometry->setMeasurement(transform);
            odometry->setInformation(information);
            if (score > 0.005) {
                Eigen::Isometry3d first = vertices[iter->first]->estimate();
                Eigen::Isometry3d second = vertices[iter->second]->estimate();
                odometry->setMeasurement(first.inverse()*second);
            }
            optimizer.addEdge(odometry);

            ++iter;
        }*/
    }

    optimizer.initializeOptimization();
    std::cout << "Optimizing..." << std::endl;
    optimizer.setVerbose(true);
    optimizer.optimize(10);
    std::cout << "Done optimizing!" << std::endl;

    CloudPtr full_cloud(new pcl::PointCloud<PointType>);
    //full_cloud.reserve(); // compute sum of points
    for (size_t i = 27; i < 38; ++i)
    {
        pcl::PointCloud<PointType> temp;
        pcl::transformPointCloud(*clouds[i], temp, -origins[i], Eigen::Quaternionf::Identity());
        Eigen::Isometry3d estimate = vertices[i]->estimate();
        Eigen::Isometry3f transform = estimate.cast<float>();
        for (PointType& p : temp.points) {
            p.getVector3fMap() = transform*p.getVector3fMap();
        }
        full_cloud->points.insert(full_cloud->points.end(), temp.points.begin(), temp.points.end());
    }

    pcl::PointCloud<PointType>::Ptr cloud_subsampled(new pcl::PointCloud<PointType>);
    pcl::VoxelGrid<PointType> sor;
    sor.setInputCloud(full_cloud);
    sor.setLeafSize(0.05f, 0.05f, 0.05f);
    sor.filter(*cloud_subsampled);

    // Initializing point cloud visualizer
    boost::shared_ptr<pcl::visualization::PCLVisualizer>
            viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer_final->setBackgroundColor (0, 0, 0);

    // Coloring and visualizing target cloud (red).
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(cloud_subsampled);
    viewer_final->addPointCloud<PointType> (cloud_subsampled, rgb, "target cloud");
    viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    1, "target cloud");

    // Starting visualizer
    viewer_final->addCoordinateSystem (1.0);
    viewer_final->initCameraParameters ();

    // Wait until visualizer window is closed.
    while (!viewer_final->wasStopped ())
    {
        viewer_final->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    viewer_final->close();
}
