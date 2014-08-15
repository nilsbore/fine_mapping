#include "scan.h"

//#include <istream>
#include <fstream>
#include <sstream>
#include <string>
#include <pcl/io/pcd_io.h>
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include "octave_convenience.h"

using namespace Eigen;

scan::scan(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const Vector3f& origin, const Matrix3f& basis, const Matrix3f& K)
{
    initialize(cloud, origin, basis, K);
}

scan::scan(const std::string& pcdname, const std::string& tname)
{
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
    origin = eorigin;
    basis = ebasis;
    pcl::copyPointCloud(cloud, points);
    minz = 1000.0;
    maxz = 0.0;
    bool empty = true;
    for (const pcl::PointXYZRGB& point : points.points) {
        if (isnan(point.z) || isinf(point.z)) {
            continue;
        }
        empty = false;
        maxz = std::max(maxz, point.z);
        minz = std::min(minz, point.z);
    }
    if (empty) {
        ROS_INFO("No points in cloud...");
        return;
    }
    fx = K(0, 0);
    fy = K(1, 1);
    cx = K(0, 2);
    cy = K(1, 2);
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
    project(depth_img, rgb_img, ox, oy, *this, 1.0, true);
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

Vector3f scan::reproject_point(int x, int y, float depth, float scale) const
{
    Vector3f rtn(float(x), float(y), depth);
    rtn(0) = scale/fx*(rtn(0) - cx/scale)*rtn(2);
    rtn(1) = scale/fy*(rtn(1) - cy/scale)*rtn(2);
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

void scan::reproject(pcl::PointCloud<pcl::PointXYZRGB>& cloud, cv::Mat* counter) const
{
    cloud.reserve(height*width);
    float depth;
    pcl::PointXYZRGB point;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            if (counter != NULL && counter->at<uchar>(y, x) == 0) {
                continue;
            }
            depth = depth_img.at<float>(y, x);
            if (depth == 0.0) {
                continue;
            }
            point.getVector3fMap() = reproject_point(x, y, depth);
            point.r = rgb_img.at<cv::Vec3b>(y, x)[0];
            point.g = rgb_img.at<cv::Vec3b>(y, x)[1];
            point.b = rgb_img.at<cv::Vec3b>(y, x)[2];
            cloud.push_back(point);
        }
    }
}

void scan::project_onto_self(cv::Mat& depth, cv::Mat& rgb) const
{
    depth = cv::Mat::zeros(height, width, CV_32FC1);
    rgb = cv::Mat::zeros(height, width, CV_8UC3);

    pcl::PointXYZRGB point;
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            point = points.points[y*width + x];
            rgb.at<cv::Vec3b>(y, x)[0] = point.r;
            rgb.at<cv::Vec3b>(y, x)[1] = point.g;
            rgb.at<cv::Vec3b>(y, x)[2] = point.b;
            if (!isnan(point.z) && !isinf(point.z)) {
                depth.at<float>(y, x) = point.z;
            }
        }
    }
}

bool scan::project(cv::Mat& depth, cv::Mat& rgb, size_t& ox, size_t& oy, const scan& other, float scale, bool init, cv::Mat* ind) const
{   
    Matrix3f R = basis.transpose()*other.basis;
    Vector3f t = basis.transpose()*(other.origin - origin);

    float scaled_fx = fx / scale;
    float scaled_fy = fy / scale;
    float scaled_cx = cx / scale;
    float scaled_cy = cy / scale;
    
    int pwidth, pheight;
    int minx, miny, maxx, maxy;
    if (init || &other == this) {
        pwidth = width / int(scale);
        pheight = height / int(scale);
        minx = miny = 0;
        maxx = width;
        maxy = height;
        if (scale == 1.0 && ind == NULL) {
            project_onto_self(depth, rgb);
            return true;
        }
    }
    else {
        ArrayXXf confining_points;
        other.camera_cone(confining_points);
        confining_points = R*confining_points.matrix();
        confining_points.matrix() += t.replicate(1, 8);

        confining_points.row(0) = scaled_fx*confining_points.row(0)/confining_points.row(2) + scaled_cx;
        confining_points.row(1) = scaled_fy*confining_points.row(1)/confining_points.row(2) + scaled_cy;
        minx = int(confining_points.row(0).minCoeff());
        maxx = int(confining_points.row(0).maxCoeff());
        miny = int(confining_points.row(1).minCoeff());
        maxy = int(confining_points.row(1).maxCoeff());
        minx = std::max(minx, 0);
        miny = std::max(miny, 0);
        maxx = std::min(maxx, int(width) / int(scale)); // + 0.5?
        maxy = std::min(maxy, int(height) / int(scale));

        pwidth = maxx - minx;
        pheight = maxy - miny;

        //std::cout << "Cropped width: " << pwidth << ", height: " << pheight << std::endl;
        if (maxx <= minx || maxy <= miny) {
            std::cout << "minx: " << minx << ", " << "maxx: " << maxx << ", " << "miny: " << miny << ", " << "maxy: " << maxy << std::endl;
            std::cout << "Confining points: " << std::endl;
            std::cout << confining_points << std::endl;
            std::cout << "R: " << std::endl;
            std::cout << R << std::endl;
            std::cout << "t: " << std::endl;
            std::cout << t << std::endl;
            std::cout << "minz:" << minz << ", " << "maxz:" << maxz << std::endl;
            std::cout << "Scans are not overlapping!" << std::endl;
            return false;
        }
    }
    
    depth = cv::Mat::zeros(pheight, pwidth, CV_32FC1);
    rgb = 125*cv::Mat::ones(pheight, pwidth, CV_8UC3);
    bool compute_inds = ind != NULL;
    if (compute_inds) {
        *ind = cv::Mat::zeros(pheight, pwidth, CV_32SC1);
    }
    
    int x, y;
    float z, curr_z;
    Vector3f vec;
    const size_t n = other.points.points.size();
    for (size_t i = 0; i < n; ++i) {
        if (isnan(other.points.points[i].z) || isinf(other.points.points[i].z)) {
            continue;
        }
        vec = R*other.points.points[i].getVector3fMap() + t;
        x = int(scaled_fx*vec(0)/vec(2)+scaled_cx) - minx; // +0.5?
        y = int(scaled_fy*vec(1)/vec(2)+scaled_cy) - miny; // +0.5?
        if (x >= 0 && x < pwidth && y >= 0 && y < pheight) {
            z = vec(2);
            curr_z = depth.at<float>(y, x);
            if (curr_z == 0 || curr_z > z) {
                depth.at<float>(y, x) = z;
                rgb.at<cv::Vec3b>(y, x)[0] = other.points.points[i].r;
                rgb.at<cv::Vec3b>(y, x)[1] = other.points.points[i].g;
                rgb.at<cv::Vec3b>(y, x)[2] = other.points.points[i].b;
                if (compute_inds) {
                    ind->at<int32_t>(y, x) = i;
                }
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

int scan::find_next_point(const Vector2f& q, const Vector2f& c, const std::vector<Vector2f, aligned_allocator<Vector2f> >& p, std::vector<int>& used) const
{
    Vector2f p0;
    Vector2f p1;
    Vector2f v;
    Vector2f o;
    bool found;
    for (size_t j = 0; j < p.size(); ++j) {
        if (used[j]) {
            continue;
        }
        p0 = q;
        p1 = p[j];
        if (p0(0) == p1(0) && p0(1) == p1(1)) {
            continue;
        }
        v = p1 - p0;
        o = Vector2f(-v(1), v(0));
        if (o.dot(c - p0) < 0) {
            continue;
        }
        found = true;
        for (size_t k = 0; k < p.size(); ++k) {
            if (k == j || (p[k](0) == q(0) && p[k](1) == q(1))) {
                continue;
            }
            if (o.dot(p[k] - p0) < 0) {
                found = false;
                break;
            }
        }
        if (found) {
            used[j] = 1;
            return j;
        }
    }
    return -1;
}

void scan::convex_hull(std::vector<Vector2f, aligned_allocator<Vector2f> >& res, const Vector2f& c, const std::vector<Vector2f, aligned_allocator<Vector2f> >& p) const
{
    std::vector<int> used;
    used.resize(p.size(), 0);

    int first_ind;
    int previous_ind;
    for (size_t i = 0; i < p.size(); ++i) {
        int ind = find_next_point(p[i], c, p, used);
        if (ind != -1) {
            used[i] = 1;
            used[ind] = 1;
            first_ind = i;
            previous_ind = ind;
            break;
        }
        used[i] = 1;
    }
    res.push_back(p[first_ind]);
    while (previous_ind != first_ind) {
        res.push_back(p[previous_ind]);
        int ind = find_next_point(p[previous_ind], c, p, used);
        previous_ind = ind;
        used[first_ind] = 0;
    }
}

float scan::compute_overlap_area(const std::vector<Vector2f, aligned_allocator<Vector2f> >& p) const
{
    Vector2f p0 = p[0];
    Matrix2f D;
    float area_triangle;
    float area = 0.0;
    for (size_t i = 1; i < p.size()-1; ++i) {
        Vector2f p1 = p[i];
        Vector2f p2 = p[i+1];
        D.col(0) = p1 - p0;
        D.col(1) = p2 - p0;
        area_triangle = 0.5*D.determinant();
        area += fabs(area_triangle);
        /*float a = (p1-p0).norm();
        float b = (p2-p0).norm();
        float c = (p2-p1).norm();
        float s = 0.5*(a + b + c);
        area += sqrt(s*(s-a)*(s-b)*(s-c));*/
    }
    return area;
}

/*bool scan::overlaps_with(const scan& other) const
{
    Matrix3f R = basis.transpose()*other.basis;
    Vector3f t = basis.transpose()*(other.origin - origin);

    if (R(2, 2) < 0.0) {
        return false;
    }

    ArrayXXf confining_points;
    other.camera_cone(confining_points);
    confining_points = R*confining_points.matrix();
    confining_points.matrix() += t.replicate(1, 8);

    Matrix<float, 2, 8> m;
    m.row(0) = fx*confining_points.row(0)/confining_points.row(2) + cx;
    m.row(1) = fy*confining_points.row(1)/confining_points.row(2) + cy;

    std::vector<Vector2f, aligned_allocator<Vector2f> > p;
    p.resize(8);
    for (size_t i = 0; i < 8; ++i) {
        m(0, i) = std::max(0.0f, m(0, i));
        m(0, i) = std::min(float(width-1), m(0, i));
        m(1, i) = std::max(0.0f, m(1, i));
        m(1, i) = std::min(float(height-1), m(1, i));
        p[i] = m.col(i);
    }

    Vector2f c = m.rowwise().mean();
    std::vector<Vector2f, aligned_allocator<Vector2f> > hull;

    std::vector<double> xi;
    std::vector<double> yi;
    for (double i : {0.0, 480.0}) {
        for (double j : {0.0, 640.0}) {
            xi.push_back(j);
            yi.push_back(i);
        }
    }

    convex_hull(hull, c, p);
    float overlap_area = compute_overlap_area(hull);

    std::vector<double> x;
    std::vector<double> y;
    for (size_t i = 0; i < hull.size(); ++i) {
        x.push_back(hull[i](0));
        y.push_back(hull[i](1));
    }

    std::cout << "Overlap area: " << overlap_area << std::endl;
    std::cout << "Hull points: " << hull.size() << std::endl;
    for (size_t i = 0; i < hull.size(); ++i) {
        std::cout << hull[i].transpose() << ", ";
    }
    std::cout << std::endl;

    octave_convenience o;
    o << "plot(";
    o.append_vector(x);
    o << ", ";
    o.append_vector(y);
    o << ", 'b'); axis equal; pause";
    //o << " hold on; plot(";
    //o.append_vector(xi);
    //o << ", ";
    //o.append_vector(yi);
    //o << ", 'r'); axis equal; pause";
    o.eval();

    if (overlap_area > float(width*height)/2.0) {
        return true;
    }
    else {
        return false;
    }
}*/

bool scan::overlaps_with(const scan& other) const
{
    Matrix3f R = basis.transpose()*other.basis;
    Vector3f t = basis.transpose()*(other.origin - origin);

    if (R(2, 2) < 0.0) {
        return false;
    }

    int x, y;
    int ok_counter = 0;
    int counter = 0;
    Vector3f vec;
    for (size_t i = 0; i < other.points.size(); i += 30) {
        if (isnan(other.points.points[i].z) || isinf(other.points.points[i].z)) {
            continue;
        }
        vec = R*other.points.points[i].getVector3fMap() + t;
        x = int(fx*vec(0)/vec(2)+cx);
        y = int(fy*vec(1)/vec(2)+cy);
        if (x > 0 && x < width && y > 0 && y < height) {
            ++ok_counter;
        }
        ++counter;
    }
    return float(ok_counter)/float(counter) > 0.5;
}

scan::~scan()
{

}
