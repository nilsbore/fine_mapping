#include "stitched_map.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

void stitched_map::visualize()
{
    size_t n = scans.size();
    std::vector<cv::Mat> counters;
    construct_counters(counters);

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;
    clouds.resize(n);
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    for (size_t i = 0; i < n; ++i) {
        clouds[i] = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
        scans[i]->reproject(*clouds[i], &counters[i]);
        scans[i]->get_transform(R, t);
        for (pcl::PointXYZRGB& point : clouds[i]->points) {
            point.getVector3fMap() = R*point.getVector3fMap() + t;
        }
    }

    boost::shared_ptr<pcl::visualization::PCLVisualizer>
            viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Starting visualizer
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    for (size_t i = 0; i < n; ++i) {
        std::string cloudname = std::string("cloud") + std::to_string(i);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(clouds[i]);
        viewer->addPointCloud<pcl::PointXYZRGB>(clouds[i], rgb, cloudname);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 1, cloudname);
    }

    // Wait until visualizer window is closed.
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
    viewer->close();
}

void stitched_map::construct_counters(std::vector<cv::Mat>& counters)
{
    size_t n = scans.size();

    counters.resize(n);
    for (size_t i = 0; i < n; ++i) {
        counters[i] = cv::Mat::ones(480, 640, CV_8UC1);
    }

    for (size_t i = 0; i < n; ++i) {
        size_t j = (i+1)%n;
        size_t k = (i+2)%n;
        if (i % 2 == 0) {
            if (scans[i]->is_behind(*scans[j])) {
                average_scans(*scans[i], *scans[j], counters[i], counters[j]);
            }
            else {
                average_scans(*scans[j], *scans[i], counters[j], counters[i]);
            }
        }
        if (scans[i]->is_behind(*scans[k])) {
            average_scans(*scans[i], *scans[k], counters[i], counters[k]);
        }
        else {
            average_scans(*scans[k], *scans[i], counters[k], counters[i]);
        }
    }
}

void stitched_map::average_scans(scan& scan1, scan& scan2, cv::Mat& counter1, cv::Mat& counter2)
{
    cv::Mat depth2, rgb2;
    cv::Mat depth1, rgb1;
    size_t ox, oy;
    cv::Mat ind;
    if (!scan1.project(depth2, rgb2, ox, oy, scan2, 1.0, false, &ind)) {
        return;
    }
    cv::Rect roi(ox, oy, depth2.cols, depth2.rows);
    depth1 = scan1.depth_img(roi);
    rgb1 = scan1.rgb_img(roi);

    float d2;
    cv::Vec3b r2;
    size_t ind2;
    size_t x2, y2;
    for (size_t y = 0; y < depth2.rows; ++y) {
        for (size_t x = 0; x < depth2.cols; ++x) {
            float& d1 = depth1.at<float>(y, x);
            d2 = depth2.at<float>(y, x);
            if (d1 == 0.0 || d2 == 0.0) {
                continue;
            }

            uchar& c1 = counter1.at<uchar>(oy + y, ox + x);
            ind2 = ind.at<int32_t>(y, x);
            x2 = ind2 % scan2.depth_img.cols;
            y2 = ind2 / scan2.depth_img.cols;
            uchar& c2 = counter2.at<uchar>(y2, x2);

            if (c1 == -1 || c2 == -1) {
                continue;
            }

            cv::Vec3b& r1 = rgb2.at<cv::Vec3b>(y, x);
            r2 = rgb2.at<cv::Vec3b>(y, x);

            d1 = (float(c1)*d1 + float(c2)*d2) / float(c1 + c2);
            r1[0] = uchar((float(c1)*r1[0] + float(c2)*r2[0]) / float(c1 + c2));
            r1[1] = uchar((float(c1)*r1[1] + float(c2)*r2[1]) / float(c1 + c2));
            r1[2] = uchar((float(c1)*r1[2] + float(c2)*r2[2]) / float(c1 + c2));

            c1 = c1 + c2;
            c2 = -1;
        }
    }
}