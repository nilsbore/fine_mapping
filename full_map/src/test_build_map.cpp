#include "simpleXMLparser.h"

#include <semanticMapSummaryParser.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <boost/filesystem.hpp>

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
    ndt.setTransformationEpsilon (0.001);
    // Setting maximum step size for More-Thuente line search.
    ndt.setStepSize(0.1);
    //Setting Resolution of NDT grid structure (VoxelGridCovariance).
    ndt.setResolution(1.0);

    // Setting max number of registration iterations.
    ndt.setMaximumIterations(20);

    ndt.setOulierRatio(0.3);

    // Setting point cloud to be aligned.
    ndt.setInputSource(input_cloud);
    // Setting point cloud to be aligned to.
    ndt.setInputTarget(target_cloud);

    // Set initial alignment estimate found using robot odometry.
    Eigen::AngleAxisf init_rotation = Eigen::AngleAxisf::Identity();
    Eigen::Translation3f init_translation(filtered_translation - target_translation);
    Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();

    std::cout << "Init guess: " << std::endl;
    std::cout << init_guess << std::endl;

    // Calculating required rigid transform to align the input cloud to the target cloud.
    pcl::PointCloud<PointType>::Ptr output_cloud(new pcl::PointCloud<PointType>);
    ndt.align(*output_cloud, init_guess);

    std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged()
              << " score: " << ndt.getFitnessScore() << std::endl;

    score = ndt.getFitnessScore();

    /*pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp; // don't really need rgb
    // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
    icp.setMaxCorrespondenceDistance(0.05);
    // Set the maximum number of iterations (criterion 1)
    icp.setMaximumIterations(50);
    // Set the transformation epsilon (criterion 2)
    icp.setTransformationEpsilon(1e-3);
    // Set the euclidean distance difference epsilon (criterion 3)
    icp.setEuclideanFitnessEpsilon(1);
    icp.setInputSource(input_cloud);
    icp.setInputTarget(target_cloud);

    // Set initial alignment estimate found using robot odometry.
    Eigen::AngleAxisf init_rotation = Eigen::AngleAxisf::Identity();
    Eigen::Translation3f init_translation(filtered_translation - target_translation);
    Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    icp.align(*output_cloud, init_guess);
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
                 icp.getFitnessScore() << std::endl;

    score = icp.getFitnessScore();

    // Transforming unfiltered, input cloud using found transform.*/
    //pcl::transformPointCloud(*input_cloud, *output_cloud, ndt.getFinalTransformation());
    pcl::PointCloud<PointType>::Ptr vis_cloud(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*input_cloud, *vis_cloud, init_guess);

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

    // Coloring and visualizing transformed input cloud (blue).
    pcl::visualization::PointCloudColorHandlerCustom<PointType>
            input_color (vis_cloud, 0, 0, 255);
    viewer_final->addPointCloud<PointType> (vis_cloud, input_color, "input_cloud");
    viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    1, "input_cloud");

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

    return ndt.getFinalTransformation();
    //return icp.getFinalTransformation();
}

bool create_folder(const std::string& folder)
{
    boost::filesystem::path dir(folder);
    if (!boost::filesystem::exists(dir) && !boost::filesystem::create_directory(dir))  {
            std::cout << "Failed to create directory " << folder << "..." << std::endl;
            return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    SemanticMapSummaryParser<PointType> summary_parser("/home/nbore/Data/Semantic map/index.xml");
    summary_parser.createSummaryXML("/home/nbore/Data/Semantic map/");

    SimpleXMLParser<PointType> simple_parser;
    SimpleXMLParser<PointType>::RoomData roomData;
    typedef SimpleXMLParser<PointType>::CloudPtr CloudPtr;
    
    typedef pair<size_t, size_t> pair_t;
    /*vector<pair_t> pairs = {pair_t(2, 3), pair_t(4, 5), pair_t(7, 8), pair_t(9, 10), pair_t(11, 12), pair_t(13, 14), pair_t(15, 16),
                           pair_t(17, 18), pair_t(20, 21), pair_t(22, 23), pair_t(25, 26),
                           pair_t(2, 27), pair_t(5, 30), pair_t(6, 31), pair_t(7, 32), pair_t(9, 33), pair_t(12, 35), pair_t(13, 36), pair_t(15, 37),
                           pair_t(18, 28), pair_t(19, 29), pair_t(21, 30), pair_t(23, 31), pair_t(24, 32), pair_t(26, 34),
                           pair_t(27, 28), pair_t(28, 29), pair_t(29, 30), pair_t(30, 31), pair_t(31, 32), pair_t(32, 33),
                           pair_t(33, 34), pair_t(34, 35), pair_t(35, 36), pair_t(36, 37)};*/
    vector<vector<size_t> > room_waypoints = {{6}, {19}, {24}, {2, 3}, {4, 5}, {7, 8}, {9, 10}, {11, 12}, {13, 14}, {15, 16},
                           {17, 18}, {20, 21}, {22, 23}, {25, 26}, {27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37}};

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

    vector<vector<CloudPtr> > room_clouds(room_waypoints.size());
    size_t counter = 0;
    for (vector<size_t>& room : room_waypoints) {
        /*if (counter != room_waypoints.size() - 1) {
            ++counter;
            continue;
        }*/
        //room_clouds[counter] = CloudPtr(new pcl::PointCloud<PointType>);
        room_clouds[counter].resize(room.size());
        if (room.size() == 1) {
            room_clouds[counter][0] = CloudPtr(new pcl::PointCloud<PointType>);
            pcl::copyPointCloud(*clouds[room[0]], *room_clouds[counter][0]);
            ++counter;
            continue;
        }
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1> >  SlamBlockSolver;
        typedef g2o::LinearSolverPCG<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

        g2o::SparseOptimizer optimizer;
        SlamLinearSolver* linearSolver = new SlamLinearSolver();
        SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);
        solver->setUserLambdaInit(1); // 0
        optimizer.setAlgorithm(solver);

        std::vector<g2o::VertexSE3*> vertices(room.size());
        for (size_t i = 0; i < room.size(); ++i)
        {
            g2o::VertexSE3* robot = new g2o::VertexSE3;
            robot->setId(i);
            Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
            transform.matrix().block<3, 1>(0, 3) = origins[room[i]].cast<double>();
            robot->setEstimate(transform);
            optimizer.addVertex(robot);
            vertices[i] = robot;
        }

        vertices[0]->setFixed(true);
        if (room.size() > 5) {
            vertices[vertices.size() - 1]->setFixed(true);
        }
        Eigen::Matrix<double, 6, 6> information;
        information.setIdentity();
        information.bottomRightCorner<3, 3>() *= 1.0; // 100.0
        for (size_t i = 0; i < room.size(); ++i)
        {
            for (size_t j = 0; j < i; ++j) {
                //register(iter->first, iter->second);
                cout << room[i] << ", " << room[j] << endl;
                CloudPtr temp_first(new pcl::PointCloud<PointType>);
                CloudPtr temp_second(new pcl::PointCloud<PointType>);
                pointcloud_common<PointType> common(0.3);
                common.set_input(clouds[room[i]]);
                common.set_target(clouds[room[j]]);
                bool overlap = common.segment(*temp_first, *temp_second);
                if (!overlap && j < i-1) {
                    continue;
                }
                double score;
                Eigen::Matrix4f res = register_scans(score, temp_first, temp_second, origins[room[i]], origins[room[j]]);
                Eigen::Isometry3d transform(res.cast<double>());
                g2o::EdgeSE3* odometry = new g2o::EdgeSE3;
                odometry->vertices()[0] = optimizer.vertex(i);
                odometry->vertices()[1] = optimizer.vertex(j);
                odometry->setMeasurement(transform.inverse());
                odometry->setInformation(information);
                if (false) {//score > 0.02) {
                    Eigen::Isometry3d first = vertices[i]->estimate();
                    Eigen::Isometry3d second = vertices[j]->estimate();
                    odometry->setMeasurement(second*first.inverse());
                }
                optimizer.addEdge(odometry);
            }
        }

        if (true) {//room.size() > 2) {
            optimizer.initializeOptimization();
            std::cout << "Optimizing..." << std::endl;
            optimizer.setVerbose(true);
            optimizer.optimize(30);
            std::cout << "Done optimizing!" << std::endl;
        }

        string folder = string(getenv("HOME")) + string("/.ros/full_map/");
        stringstream ss;
        ss << "room" << setfill('0') << setw(6) << counter;
        create_folder(folder + ss.str());

        ofstream tfile;
        string tname = folder + ss.str() + string("/origins.txt");
        tfile.open(tname.c_str());

        // Initializing point cloud visualizer
        boost::shared_ptr<pcl::visualization::PCLVisualizer>
                viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer_final->setBackgroundColor (0, 0, 0);

        for (g2o::HyperGraph::Edge* e : optimizer.edges()) {
            g2o::EdgeSE3* es = (g2o::EdgeSE3*)e;
            g2o::VertexSE3* vs0 = (g2o::VertexSE3*)es->vertices()[0];
            g2o::VertexSE3* vs1 = (g2o::VertexSE3*)es->vertices()[1];
            Eigen::Isometry3d first = vs0->estimate();
            Eigen::Isometry3d second = vs1->estimate();
            std::cout << "After optimization: " << std::endl;
            Eigen::Matrix4d mat = (second.inverse()*first).matrix();
            std::cout << mat << std::endl;
            //std::cout << es->error().transpose() << std::endl;
        }

        //CloudPtr full_cloud(new pcl::PointCloud<PointType>);
        //full_cloud.reserve(); // compute sum of points
        for (size_t i = 0; i < room.size(); ++i)
        {
            // write the camera centers to a file
            /*for (size_t j = 0; j < 3; ++j) {
                tfile << origins[counter](j) << " ";
            }*/
            tfile << origins[room[i]].transpose() << "\n";

            pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>);
            //pcl::PointCloud<PointType>::Ptr temp2(new pcl::PointCloud<PointType>);
            pcl::transformPointCloud(*clouds[room[i]], *temp, -origins[room[i]], Eigen::Quaternionf::Identity());
            Eigen::Isometry3d estimate = vertices[i]->estimate();
            Eigen::Isometry3f transform = estimate.cast<float>();
            //pcl::transformPointCloud(*temp, *temp2, transform);
            for (PointType& p : temp->points) {
                p.getVector3fMap() = transform*p.getVector3fMap();
            }

            stringstream rr;
            rr << "/cloud" << setfill('0') << setw(6) << i << ".pcd";
            pcl::io::savePCDFileBinary(folder + ss.str() + rr.str(), *temp);
            //full_cloud->points.insert(full_cloud->points.end(), temp.points.begin(), temp.points.end());

            room_clouds[counter][i] = CloudPtr(new pcl::PointCloud<PointType>);
            pcl::copyPointCloud(*temp, *room_clouds[counter][i]);

            // Coloring and visualizing target cloud (red).
            pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(temp);
            viewer_final->addPointCloud<PointType> (temp, rgb, string("cloud") + to_string(i));
            viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                            1, string("cloud") + to_string(i));
        }
        tfile.close();
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

        /*pcl::VoxelGrid<PointType> sor;
        sor.setInputCloud(full_cloud);
        sor.setLeafSize(0.05f, 0.05f, 0.05f);
        sor.filter(*room_clouds[counter]);*/

        ++counter;
    }

    /*counter = 0;
    for (CloudPtr& cloud : room_clouds) {
    //{
        //CloudPtr& cloud = room_clouds[room_clouds.size() - 1];
        if (counter < 2) {
            ++counter;
            continue;
        }

        // Initializing point cloud visualizer
        boost::shared_ptr<pcl::visualization::PCLVisualizer>
                viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer_final->setBackgroundColor (0, 0, 0);

        // Coloring and visualizing target cloud (red).
        pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(cloud);
        viewer_final->addPointCloud<PointType> (cloud, rgb, "target cloud");
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

        std::string folder = std::string(getenv("HOME")) + std::string("/.ros/full_map/");
        stringstream ss;
        ss << "room" << std::setfill('0') << std::setw(6) << counter << ".pcd";
        pcl::io::savePCDFileBinary(folder + ss.str(), *cloud);
        ++counter;
    }*/
}
