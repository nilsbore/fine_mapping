#include "scan.h"

//#include <istream>
#include <fstream>
#include <sstream>
#include <string>
#include <pcl/io/pcd_io.h>
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

using namespace Eigen;

scan::scan(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const Vector3f& origin, const Matrix3f& basis, const Matrix3f& K)
{
    red = green = blue = NULL;
    initialize(cloud, origin, basis, K);
}

scan::scan(const std::string& pcdname, const std::string& tname)
{
    red = green = blue = NULL;
    initialize_from_files(pcdname, tname);
}

void scan::initialize_from_files(const std::string& pcdname, const std::string& tname)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;

    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcdname, cloud) == -1) //* load the file
    {
        ROS_ERROR("Couldn't read file %s", pcdname.c_str());
        return;
    }

    std::ifstream tfile;
    tfile.open(tname.c_str());
    if (!tfile.is_open()) {
        ROS_ERROR("Couldn't read file %s", tname.c_str());
        return;
    }

    Vector3f eorigin;
    Matrix3f ebasis;
    Matrix3f K;
    std::string line;
    if (!std::getline(tfile, line)) {
        ROS_ERROR("Couldn't read file %s", tname.c_str());
    }
    std::istringstream iss(line);
    for (size_t i = 0; i < 3; ++i) {
        if (!(iss >> eorigin(i))) {
            ROS_ERROR("Couldn't read file %s", tname.c_str());
        }
    }
    if (!std::getline(tfile, line)) {
        ROS_ERROR("Couldn't read file %s", tname.c_str());
    }
    iss.str(line);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (!(iss >> ebasis(i, j))) {
                ROS_ERROR("Couldn't read file %s", tname.c_str());
            }
        }
    }
    if (!std::getline(tfile, line)) {
        ROS_ERROR("Couldn't read file %s", tname.c_str());
    }
    iss.str(line);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (!(iss >> K(i, j))) {
                ROS_ERROR("Couldn't read file %s", tname.c_str());
            }
        }
    }
    tfile.close();
    initialize(cloud, eorigin, ebasis, K);
}

void scan::initialize(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const Vector3f& eorigin, const Matrix3f& ebasis, const Matrix3f& K)
{
    static int one = 0;
    origin = eorigin;
    Matrix3f R = Matrix3f(AngleAxisf(-0.02*M_PI, Vector3f::UnitZ()));
    //basis = R*ebasis; // DEBUG!! Remove later!!
    //origin += Vector3f(0.0, 0.0, 0.0);
    basis = ebasis;
    if (one == 0) {
        //origin += Vector3f(0.0, 0.1, 0.0);
        //basis = R*basis;
    }
    one += 1.0;
    size_t n = cloud.points.size(); 
    points.resize(3, n);
    red = new uint8_t[n];
    green = new uint8_t[n];
    blue = new uint8_t[n];
    size_t counter = 0;
    for (size_t i = 0; i < n; ++i) {
        if (isnan(cloud.points[i].z) || isinf(cloud.points[i].z)) {
            continue;
        }
        points.col(counter) = cloud.points[i].getVector3fMap();
        red[counter] = cloud.points[i].r;
        green[counter] = cloud.points[i].g;
        blue[counter] = cloud.points[i].b;
        ++counter;
    }
    points.conservativeResize(3, counter);
    if (counter == 0) {
        ROS_INFO("No points in cloud...");
        return;
    }
    //VectorXf meant = points.rowwise().mean();
    //origin += basis*meant;
    //points -= meant.replicate(1, counter);
    fx = K(0, 0);
    fy = K(1, 1);
    cx = K(0, 2);
    cy = K(1, 2);
    minz = points.row(2).minCoeff();
    maxz = points.row(2).maxCoeff();
    height = 480; // get from camerainfo
    width = 640;
    
    std::stringstream ss;
    ss << origin.transpose();
    ROS_INFO("Loaded origin %s", ss.str().c_str());
    ss.str("");
    ss << basis;
    ROS_INFO("Loaded basis\n %s", ss.str().c_str());
    ss.str("");
    ss << K;
    ROS_INFO("Loaded camera matrix\n %s", ss.str().c_str());
    ROS_INFO("Min depth %f", minz);
    ROS_INFO("Max depth %f", maxz);
    
    size_t ox, oy;
    project(depth_img, rgb_img, ox, oy, *this, true);
    ArrayXXf confining_points;
    camera_cone(confining_points);
}

void scan::set_transform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
{
    basis = R;
    origin = t;
}

void scan::transform(const Matrix3f& R, const Vector3f& t)
{
    basis = basis*R; // add to total rotation
    origin += basis*t; // add to total translation
}

void scan::camera_cone(ArrayXXf& confining_points) const
{
    confining_points.resize(3, 8);
    confining_points(0, 0) = 0.0;
    confining_points(1, 0) = 0.0;
    confining_points(0, 1) = float(width);
    confining_points(1, 1) = 0.0;
    confining_points(0, 2) = 0.0;
    confining_points(1, 2) = float(height);
    confining_points(0, 3) = float(width);
    confining_points(1, 3) = float(height);
    confining_points.block(0, 4, 2, 4) = confining_points.block(0, 0, 2, 4);
    confining_points.block(2, 0, 1, 4) = Array4f::Constant(minz).transpose();
    confining_points.block(2, 4, 1, 4) = Array4f::Constant(maxz).transpose();
    confining_points.row(0) = 1.0/fx*(confining_points.row(0) - cx)*confining_points.row(2);
    confining_points.row(1) = 1.0/fy*(confining_points.row(1) - cy)*confining_points.row(2);
}

Vector3f scan::reproject_point(int x, int y, float depth) const
{
    Vector3f rtn(float(x), float(y), depth);
    rtn(0) = 1.0/fx*(rtn(0) - cx)*rtn(2);
    rtn(1) = 1.0/fy*(rtn(1) - cy)*rtn(2);
    return rtn;
}

bool scan::is_behind(const scan& other) const {
    Vector3f t = basis.transpose()*(other.origin - origin);
    return t(2) > 0;
}

void scan::submatrices(cv::Mat& depth, cv::Mat& rgb, size_t ox, size_t oy, size_t w, size_t h)
{
    depth = depth_img.colRange(ox, ox + w).rowRange(oy, oy + h);
    rgb = rgb_img.colRange(ox, ox + w).rowRange(oy, oy + h);
}

bool scan::project(cv::Mat& depth, cv::Mat& rgb, size_t& ox, size_t& oy, const scan& other, bool init) const
{   
    Matrix3f R = basis.transpose()*other.basis;
    Vector3f t = basis.transpose()*(other.origin - origin);
    
    int pwidth, pheight;
    int minx, miny;
    if (init) {
        pwidth = width;
        pheight = height;
        minx = miny = 0;
    }
    else {
        ArrayXXf confining_points;
        other.camera_cone(confining_points);
        confining_points = R*confining_points.matrix();
        confining_points.matrix() += t.replicate(1, 8);

        confining_points.row(0) = fx*confining_points.row(0)/confining_points.row(2) + cx;
        confining_points.row(1) = fy*confining_points.row(1)/confining_points.row(2) + cy;
        minx = int(confining_points.row(0).minCoeff());
        int maxx = int(confining_points.row(0).maxCoeff());
        miny = int(confining_points.row(1).minCoeff());
        int maxy = int(confining_points.row(1).maxCoeff());
        minx = std::max(minx, 0);
        miny = std::max(miny, 0);
        maxx = std::min(maxx, int(width)); // + 0.5?
        maxy = std::min(maxy, int(height));

        pwidth = maxx - minx;
        pheight = maxy - miny;

        //std::cout << "Cropped width: " << pwidth << ", height: " << pheight << std::endl;
        if (maxx <= minx || maxy <= miny) {
            std::cout << "Scans are not overlapping!" << std::endl;
            return false;
        }
    }

    MatrixXf copy = other.points;
    size_t n = copy.cols();
    copy = R*copy;
    copy += t.replicate(1, n);
    
    copy.row(0) = fx*copy.row(0).array()/copy.row(2).array() + cx; // *=, +=
    copy.row(1) = fy*copy.row(1).array()/copy.row(2).array() + cy;
    
    depth = cv::Mat::zeros(pheight, pwidth, CV_32FC1);
    rgb = cv::Mat::zeros(pheight, pwidth, CV_8UC3);
    
    int x, y;
    float z, curr_z;
    for (size_t i = 0; i < n; ++i) {
        x = int(copy(0, i)) - minx;
        y = int(copy(1, i)) - miny;
        if (x >= 0 && x < pwidth && y >= 0 && y < pheight) {
            z = copy(2, i);
            curr_z = depth.at<float>(y, x);
            if (curr_z == 0 || curr_z > z) {
                depth.at<float>(y, x) = z;
                rgb.at<cv::Vec3b>(y, x)[0] = other.red[i];
                rgb.at<cv::Vec3b>(y, x)[1] = other.green[i];
                rgb.at<cv::Vec3b>(y, x)[2] = other.blue[i];
            }
        }
    }
    
    ox = minx;
    oy = miny;
    
    if (&other == this) {
        return true;
    }
    
    /*cv::namedWindow("Depth1", CV_WINDOW_AUTOSIZE);
    cv::imshow("Depth1", depth);
    
    cv::namedWindow("Depth2", CV_WINDOW_AUTOSIZE);
    cv::imshow("Depth2", depth_img);
    cv::waitKey(0);
        
    cv::namedWindow("Rgb1", CV_WINDOW_AUTOSIZE);
    cv::imshow("Rgb1", rgb);
    
    cv::namedWindow("Rgb2", CV_WINDOW_AUTOSIZE);
    cv::imshow("Rgb2", rgb_img);
    cv::waitKey(0);*/
    return true;
}

scan::~scan()
{
    delete[] red;
    delete[] green;
    delete[] blue;
}
