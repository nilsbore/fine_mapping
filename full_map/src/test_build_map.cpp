#include "simpleXMLparser.h"

#include <semanticMapSummaryParser.h>
#include <pcl/registration/ndt.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

typedef pcl::PointXYZRGB PointType;

typedef typename SemanticMapSummaryParser<PointType>::EntityStruct Entities;

using namespace std;

void register_scans(pcl::PointCloud<PointType>::Ptr& input_cloud_trans, pcl::PointCloud<PointType>::Ptr& target_cloud_trans,
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
        clouds[ind] = roomData.completeRoomCloud;
        tf::Vector3 tforigin = roomData.vIntermediateRoomCloudTransforms[0].getOrigin();
        Eigen::Vector3f origin;
        for (size_t j = 0; j < 3; ++j) {
            origin(j) = tforigin.m_floats[j];
        }
        origins[ind] = origin;


        /*cout<<"Complete cloud size "<<roomData.completeRoomCloud->points.size()<<endl;
        for (size_t i=0; i<roomData.vIntermediateRoomClouds.size(); i++)
        {
            cout<<"Intermediate cloud size "<<roomData.vIntermediateRoomClouds[i]->points.size()<<endl;
            cout<<"Fx: "<<roomData.vIntermediateRoomCloudCamParams[i].fx()<<" Fy: "<<roomData.vIntermediateRoomCloudCamParams[i].fy()<<endl;
        }*/
    }

    for (size_t i = 2; i < 38; ++i)
    {
        //auto iterator = pairs.end();
        vector<pair_t>::iterator iter = pairs.begin();
        while ((iter = std::find_if(iter, pairs.end(), [=](const pair_t& p) { return p.first == i; })) != pairs.end()) {
            //register(iter->first, iter->second);
            cout << iter->first << ", " << iter->second << endl;
            register_scans(clouds[i], clouds[iter->second], origins[i], origins[iter->second]);
            ++iter;
        }
    }
}
