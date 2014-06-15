#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <boost/lexical_cast.hpp>

#include "scan.h"

int main(int argc, char** argv)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    // Starting visualizer
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    std::string folder = std::string(getenv("HOME")) + std::string("/.ros/mapping_refinement");
    std::vector<std::string> scans;
    std::vector<std::string> transforms;
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;
    clouds.resize(22);
    for (size_t i = 0; i < 22; ++i) {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << i;
        scans.push_back(folder + std::string("/shot") + ss.str() + std::string(".pcd"));
        transforms.push_back(folder + std::string("/transform") + ss.str() + std::string(".txt"));
        scan s(scans.back(), transforms.back());
        s.get_transform(R, t);
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        clouds[i] = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::io::loadPCDFile<pcl::PointXYZRGB>(scans.back(), *clouds[i]);
        for (size_t j = 0; j < clouds[i]->points.size(); ++j) {
            clouds[i]->points[j].getVector3fMap() = R*clouds[i]->points[j].getVector3fMap() + t;
        }
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(clouds[i]);
        viewer->addPointCloud<pcl::PointXYZRGB>(clouds[i], rgb, std::string("cloud") + ss.str());
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 1, std::string("cloud") + ss.str());
    }
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        //boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
    return 0;
}
