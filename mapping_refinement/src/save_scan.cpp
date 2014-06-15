#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/CameraInfo.h>
#include <tf/transform_listener.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include "mapping_refinement/SaveScan.h"

tf::TransformListener* listener;
std::string folder;
size_t num;
bool save;
ros::ServiceClient client;

void callback(const sensor_msgs::PointCloud2::ConstPtr& pcd_msg, const sensor_msgs::CameraInfo::ConstPtr& info_msg)
{
    if (!save) {
        return;
    }
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    pcl::fromROSMsg(*pcd_msg, cloud);
    
    // save transform from map to scan
    tf::StampedTransform transform;
    try {
        listener->lookupTransform("/map", pcd_msg->header.frame_id, pcd_msg->header.stamp, transform);
    }
    catch (tf::TransformException ex) {
        ROS_INFO("%s",ex.what());
        return;
    }
    
    // save point cloud
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << num;
    std::string pcdname = folder + std::string("/shot") + ss.str() + std::string(".pcd");
    pcl::io::savePCDFileBinary(pcdname, cloud);
    
    tf::Matrix3x3 basis = transform.getBasis();
    tf::Vector3 origin = transform.getOrigin();
    std::ofstream tfile;
    std::string tname = folder + std::string("/transform") + ss.str() + std::string(".txt");
    tfile.open(tname.c_str());
    
    Eigen::Matrix3f ebasis;
    Eigen::Vector3f eorigin;
    for (size_t i = 0; i < 3; ++i) {
        eorigin(i) = origin.m_floats[i];
        tfile << origin.m_floats[i] << " ";
    }
    tfile << "\n";
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            ebasis(i, j) = basis.getRow(i).m_floats[j];
            tfile << basis.getRow(i).m_floats[j] << " ";
        }
    }
    tfile << "\n";
    for (size_t i = 0; i < 9; ++i) {
        tfile << info_msg->K[i] << " ";
    }
    tfile.close();
    save = false;
    ++num;
}

bool srv_callback(mapping_refinement::SaveScan::Request& req,
                  mapping_refinement::SaveScan::Response& res)
{
    save = true;
    return true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "save_scan");
	ros::NodeHandle n;
	
	ros::NodeHandle pn("~");
    pn.param<std::string>("folder", folder, std::string(getenv("HOME")) + std::string("/.ros/mapping_refinement"));
    boost::filesystem::path dir(folder);
    if (!boost::filesystem::exists(dir) && !boost::filesystem::create_directory(dir))  {
            ROS_ERROR("Failed to create directory %s", folder.c_str());
            return 0;
    }
    ROS_INFO("Saving scans in %s", folder.c_str());
    
	std::string camera;
    pn.param<std::string>("camera", camera, std::string("head_xtion"));
    std::string pcd_input = std::string("/") + camera + std::string("/depth_registered/points");
    std::string info_input = std::string("/") + camera + std::string("/rgb/camera_info");
    
    listener = new tf::TransformListener();
    
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::CameraInfo> MySyncPolicy;
    message_filters::Subscriber<sensor_msgs::PointCloud2> pcd_sub(n, pcd_input, 1);
    message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub(n, info_input, 1);
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), pcd_sub, info_sub);
    sync.registerCallback(&callback);
    
    // register a service for saving a scan
    save = false;
    num = 0;
    ros::ServiceServer service = n.advertiseService("save_scan_service", &srv_callback);
    
    ros::spin();
	
	return 0;
}
