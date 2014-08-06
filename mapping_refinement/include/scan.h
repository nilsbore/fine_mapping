#ifndef SCAN_H
#define SCAN_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>

class scan {
protected:
    Eigen::MatrixXf points;
    uint8_t* red;
    uint8_t* green;
    uint8_t* blue;
    float fx, fy, cx, cy;
    float minz, maxz;
    size_t height, width;
    Eigen::Vector3f origin;
    Eigen::Matrix3f basis;
    void initialize(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const Eigen::Vector3f& origin, const Eigen::Matrix3f& basis, const Eigen::Matrix3f& K);
    void camera_cone(Eigen::ArrayXXf& confining_points) const;
public:
    cv::Mat depth_img;
    cv::Mat rgb_img;
    void set_transform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t);
    void get_transform(Eigen::Matrix3f& R, Eigen::Vector3f& t) { R = basis; t = origin; }
    void transform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t);
    Eigen::Vector3f reproject_point(int x, int y, float depth, float scale = 1.0) const;
    void submatrices(cv::Mat& depth, cv::Mat& rgb, size_t ox, size_t oy, size_t w, size_t h);
    bool is_behind(const scan& other) const;
    bool project(cv::Mat& depth, cv::Mat& rgb, size_t& ox, size_t& oy, const scan& other, float scale = 1.0, bool init = false) const;
    bool is_empty() { return points.cols() == 0; }
    void initialize_from_files(const std::string& pcdname, const std::string& tname);
    scan(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const Eigen::Vector3f& origin, const Eigen::Matrix3f& basis, const Eigen::Matrix3f& K);
    scan(const std::string& pcdname, const std::string& tname);
    scan() { red = green = blue = NULL; }
    ~scan();
};
#endif // SCAN_H
