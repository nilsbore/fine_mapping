#include "scan.h"
#include "fine_registration.h"
#include "asynch_visualizer.h"

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

int main(int arg, char** argv)
{
    scan s1("/home/nbore/.ros/mapping_refinement/shot000001.pcd", "/home/nbore/.ros/mapping_refinement/transform000001.txt");
    scan s2("/home/nbore/.ros/mapping_refinement/shot000033.pcd", "/home/nbore/.ros/mapping_refinement/transform000033.txt");
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile<pcl::PointXYZRGB>("/home/nbore/.ros/mapping_refinement/shot000001.pcd", *cloud1);
    pcl::io::loadPCDFile<pcl::PointXYZRGB>("/home/nbore/.ros/mapping_refinement/shot000033.pcd", *cloud2);
    Eigen::Matrix3f R1, R2;
    Eigen::Vector3f t1, t2;
    s1.get_transform(R1, t1);
    s2.get_transform(R2, t2);
    Eigen::Matrix3f Ro = R1.transpose()*R2;
    Eigen::Vector3f to = R1.transpose()*(t2 - t1);
    for (size_t i = 0; i < cloud2->points.size(); ++i) {
        cloud2->points[i].getVector3fMap() = Ro*cloud2->points[i].getVector3fMap() + to;
    }
    
    fine_registration r(s1, s2);
    Eigen::Matrix3f R;
    Eigen::Vector3f t;

    asynch_visualizer viewer;
    viewer.cloud1 = cloud1;
    viewer.cloud2 = cloud2;
    viewer.cloud1_changed = true;
    viewer.cloud2_changed = true;
    viewer.create_thread();
    float error;
    Eigen::Matrix3f Rtot;
    Rtot.setIdentity();
    do {
        r.step(R, t);
        s1.transform(R, t);
        for (size_t i = 0; i < cloud1->points.size(); ++i) {
            cloud1->points[i].getVector3fMap() = R*cloud1->points[i].getVector3fMap() + Rtot*t;
        }
        Rtot = R*Rtot;
        viewer.lock();
        viewer.cloud1 = cloud1;
        viewer.cloud1_changed = true;
        viewer.unlock();
        error = r.error();
        std::cout << "Error: " << error << std::endl;
    } while (error > 10);
    viewer.join_thread();
    
    return 0;
}
