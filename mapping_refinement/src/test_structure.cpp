#include <iostream>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

int main()
{
    size_t i = 0;
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << i;
    std::string folder = std::string(getenv("HOME")) + std::string("/.ros/mapping_refinement");
    std::string cloud_file = folder + std::string("/shot") + ss.str() + std::string(".pcd");


    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(cloud_file, cloud) == -1) //* load the file
    {
        printf("Couldn't read file %s", cloud_file.c_str());
        return -1;
    }

    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);

    pcl::PointXYZRGB point;
    for (size_t y = 0; y < image.rows; ++y) {
        for (size_t x = 0; x < image.cols; ++x) {
            point = cloud.points[y*image.cols + x];
            image.at<cv::Vec3b>(y, x)[0] = point.b;
            image.at<cv::Vec3b>(y, x)[1] = point.g;
            image.at<cv::Vec3b>(y, x)[2] = point.r;
        }
    }

    cv::namedWindow("Rgb", CV_WINDOW_AUTOSIZE);
    cv::imshow("Rgb", image);
    cv::waitKey(0);

    return 0;
}
