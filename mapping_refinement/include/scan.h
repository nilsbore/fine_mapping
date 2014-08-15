#ifndef SCAN_H
#define SCAN_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>

class scan {
protected:
    float fx, fy, cx, cy;
    float minz, maxz;
    size_t height, width;
    Eigen::Vector3f origin;
    Eigen::Matrix3f basis;
    void initialize(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const Eigen::Vector3f& origin, const Eigen::Matrix3f& basis, const Eigen::Matrix3f& K);
    void camera_cone(Eigen::ArrayXXf& confining_points) const;
    void convex_hull(std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> >& res, const Eigen::Vector2f& c,
                     const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> >& p) const;
    float compute_overlap_area(const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> >& p) const;
    int find_next_point(const Eigen::Vector2f& q, const Eigen::Vector2f& c, const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> >& p, std::vector<int>& used) const;
    void project_onto_self(cv::Mat& depth, cv::Mat& rgb) const;
public:
    pcl::PointCloud<pcl::PointXYZRGB> points; // maybe subclass instead
    cv::Mat depth_img;
    cv::Mat rgb_img;
    void set_transform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t);
    void get_transform(Eigen::Matrix3f& R, Eigen::Vector3f& t) { R = basis; t = origin; }
    void transform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t);
    Eigen::Vector3f reproject_point(int x, int y, float depth, float scale = 1.0) const;
    void submatrices(cv::Mat& depth, cv::Mat& rgb, size_t ox, size_t oy, size_t w, size_t h);
    bool is_behind(const scan& other) const;
    bool overlaps_with(const scan& other) const;
    bool project(cv::Mat& depth, cv::Mat& rgb, size_t& ox, size_t& oy, const scan& other, float scale = 1.0, bool init = false, cv::Mat* ind = NULL) const;
    void reproject(pcl::PointCloud<pcl::PointXYZRGB>& cloud, cv::Mat* counter = NULL) const;
    bool is_empty() { return points.points.empty(); }
    void initialize_from_files(const std::string& pcdname, const std::string& tname);
    scan(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const Eigen::Vector3f& origin, const Eigen::Matrix3f& basis, const Eigen::Matrix3f& K);
    scan(const std::string& pcdname, const std::string& tname);
    scan() {}
    ~scan();
};
#endif // SCAN_H
