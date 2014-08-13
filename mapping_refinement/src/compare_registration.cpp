
#include "scan.h"
#include "fine_registration.h"
#include "asynch_visualizer.h"

#include <Eigen/Dense>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/linear_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/slam3d/vertex_se3.h"

#include <string>
#include <boost/algorithm/string.hpp>
#include <sstream>
#include <string>
#include <numeric>

bool register_clouds_icp(Eigen::Matrix3f& R, Eigen::Vector3f& t,
                         scan* scan1, scan* scan2,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud1, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud2)
{
    Eigen::Matrix3f R1, R2;
    Eigen::Vector3f t1, t2;
    scan1->get_transform(R1, t1);
    scan2->get_transform(R2, t2);
    Eigen::Matrix3f Rdelta = R1.transpose()*R2;
    Eigen::Vector3f tdelta = R1.transpose()*(t2 - t1);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloudt(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud2, *cloudt);
    for (pcl::PointXYZRGB& p : cloudt->points) {
        p.getVector3fMap() = Rdelta*p.getVector3fMap()+tdelta;
    }
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp; // don't really need rgb
    // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
    icp.setMaxCorrespondenceDistance(0.15);
    // Set the maximum number of iterations (criterion 1)
    icp.setMaximumIterations(100);
    // Set the transformation epsilon (criterion 2)
    icp.setTransformationEpsilon(1e-12);
    // Set the euclidean distance difference epsilon (criterion 3)
    icp.setEuclideanFitnessEpsilon(0.05);
    icp.setInputSource(cloud1);
    icp.setInputTarget(cloudt);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr final(new pcl::PointCloud<pcl::PointXYZRGB>);
    icp.align(*final);
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
                 icp.getFitnessScore() << std::endl;
    Eigen::Matrix4f T = icp.getFinalTransformation();
    T /= T(3, 3);
    Eigen::Matrix3f Rcomp = T.topLeftCorner<3, 3>();
    Eigen::Vector3f tcomp = T.block<3, 1>(0, 3);

    R = Rdelta*Rcomp.transpose();
    t = tdelta - Rdelta*Rcomp.transpose()*tcomp;

    return icp.hasConverged();
}

using namespace std;
using namespace Eigen;

int main(int argc, char** argv)
{
    string folder = "/home/nbore/Data/rgbd_dataset_freiburg1_room/";
    string pcd_folder = folder + "pointclouds1/";
    string clouds = folder + "pointclouds1.txt";

    ifstream pointclouds;
    pointclouds.open(clouds);

    Matrix3f R1, R2, identity;
    Vector3f t1, t2, zeros;
    identity.setIdentity();
    zeros.setZero();

    //float focalLength = 525.0;
    float focalLengthX = 517.3;//focalLength
    float focalLengthY = 516.5;//focalLength
    float centerX = 255.3;//318.6;//319.5;
    float centerY = 239.5;
    Matrix3f K;
    K << focalLengthX, 0.0, centerX, 0.0, focalLengthY, centerY, 0.0, 0.0, 1.0;

    typedef pair<size_t, size_t> scan_pair;
    vector<scan_pair> pairs;
    for (size_t i = 10; i < 1300; i += 10) { //1300
        pairs.push_back(scan_pair(i, i+7));
    }

    vector<float> rot_offsets = {0.035};//{0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055,0.06, 0.07, 0.075, 0.08};
    vector<float> t_offsets = {0.01};//{0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20};
    string line;
    pcl::PointCloud<pcl::PointXYZRGB>* cloud;
    Matrix3f* Rp;
    Vector3f* tp;

    vector<float> mean_errors;
    for (float offset : t_offsets) {

        typedef pair<float, float> error_pair;
        vector<error_pair> errors;
        for (scan_pair& p : pairs) {
            size_t n1 = p.first;
            size_t n2 = p.second;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1 = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2 = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
            size_t counter = 0;
            while (getline(pointclouds, line) && (counter <= n1 || counter <= n2)) {
                if (counter == n1) {
                    cloud = &(*cloud1); Rp = &R1; tp = &t1;
                }
                else if (counter == n2) {
                    cloud = &(*cloud2); Rp = &R2; tp = &t2;
                }
                else {
                    ++counter;
                    continue;
                }
                vector<string> strs;
                boost::split(strs, line, boost::is_any_of("\n "));
                if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcd_folder + strs[0], *cloud) == -1) //* load the file
                {
                    printf("Couldn't read file %s", strs[0].c_str());
                    return -1;
                }
                std::cout << "Read " << strs[0] << " with " << cloud->points.size() << " points" << std::endl;
                *tp << stof(strs[2]), stof(strs[3]), stof(strs[4]);
                Quaternionf q(stof(strs[8]), stof(strs[5]), stof(strs[6]), stof(strs[7]));
                *Rp = q.matrix();

                ++counter;
            }
            pointclouds.seekg(0, ios::beg);
            scan scan1(*cloud1, t1, R1, K);
            scan scan2(*cloud2, t2, R2, K);
            if (scan1.is_empty() || scan2.is_empty()) {
                continue;
            }
            //scan scan1(*cloud1, zeros, identity, K);
            //scan scan2(*cloud2, zeros, identity, K);
            Matrix3f R_correct = R1.transpose()*R2;
            Vector3f t_correct = R1.transpose()*(t2-t1);

            // set the scan to some offset
            //Matrix3f Rd = Matrix3f(AngleAxisf(-0.01*M_PI, Vector3f::UnitZ()));
            Matrix3f Rd = Matrix3f(AngleAxisf(-0.035*M_PI, Vector3f::UnitY()));
            Vector3f td;
            td.setOnes();
            td *= 1.0/sqrt(3.0)*offset;
            R1 = R1*Rd;
            t1 += td;
            scan1.set_transform(R1, t1);
            //scan1.set_transform(Rd, td);

            // register scans using optical flow or icp
            Matrix3f R;
            Vector3f t;
            register_clouds_icp(R, t, &scan1, &scan2, cloud1, cloud2);
            //fine_registration::register_scans(R, t, &scan1, &scan2);
            //scan2.transform(R, t);
            scan1.set_transform(R2*R.transpose(), t2 - R2*R.transpose()*t);
            scan1.get_transform(R1, t1);
            scan2.get_transform(R2, t2);

            // visualize registration result
            /*for (pcl::PointXYZRGB& p : cloud1->points) {
                p.getVector3fMap() = R1*p.getVector3fMap() + t1;
                //p.r = 255;
                //p.g = 0;
                //p.b = 0;
            }
            for (pcl::PointXYZRGB& p : cloud2->points) {
                p.getVector3fMap() = R2*p.getVector3fMap() + t2;
                //p.r = 0;
                //p.g = 0;
                //p.b = 255;
            }
            cloud1->points.insert(cloud1->points.end(), cloud2->points.begin(), cloud2->points.end());
            //cloudt->points.insert(cloudt->points.end(), cloud1->points.begin(), cloud1->points.end());
            boost::shared_ptr<pcl::visualization::PCLVisualizer>
                    viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
            viewer->setBackgroundColor(0, 0, 0);

            // Starting visualizer
            viewer->addCoordinateSystem(1.0);
            viewer->initCameraParameters();
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud1);
            viewer->addPointCloud<pcl::PointXYZRGB>(cloud1, rgb, "cloud");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                     1, "cloud");

            // Wait until visualizer window is closed.
            while (!viewer->wasStopped())
            {
                viewer->spinOnce(100);
                boost::this_thread::sleep(boost::posix_time::microseconds(100000));
            }
            viewer->close();*/

            // compute error
            Matrix3f R_error = R_correct.transpose()*R;
            AngleAxisf a_error(R_error);
            Vector3f t_error = t - t_correct;
            errors.push_back(error_pair(t_error.norm(), fabs(a_error.angle())));
            std::cout << "Translation error: " << errors.back().first << std::endl;
            std::cout << "Rotation error: " << errors.back().second << std::endl;
        }

        std::cout << "========================" << std::endl;
        for (error_pair& p : errors) {
            std::cout << p.first << " ";
        }
        std::cout << std::endl;
        std::cout << "========================" << std::endl;
        for (error_pair& p : errors) {
            std::cout << p.second << " ";
        }
        std::cout << std::endl;
        std::cout << "========================" << std::endl;

        float mean_t = 0.0;
        float mean_rot = 0.0;
        float rot_divide = 0.0;
        float t_divide = 0.0;
        for (error_pair& p : errors) {
            if (!isnan(p.first)) {
                mean_t += p.first;
                t_divide += 1.0;
            }
            if (!isnan(p.second)) {
                mean_rot += p.second;
                rot_divide += 1.0;
            }
        }
        mean_t /= t_divide;
        mean_rot /= rot_divide;
        mean_errors.push_back(mean_t);
    }

    std::cout << "========================" << std::endl;
    for (float& p : mean_errors) {
        std::cout << p << " ";
    }
    std::cout << std::endl;
    std::cout << "========================" << std::endl;

    return 0;
}
