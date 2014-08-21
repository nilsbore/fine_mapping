#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <functional>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ros/ros.h"

#include "visualization_msgs/Marker.h"

#include "primitive_core.h"
#include "plane_primitive.h"
#include "sphere_primitive.h"
#include "cylinder_primitive.h"

using namespace Eigen;

ros::Publisher pub;

const double angle_threshold = 0.2;
const double distance_threshold = 0.3;

void hull_as_marker(std::vector<Vector3d, aligned_allocator<Vector3d> >& p, const Vector3f& c)
{
    static size_t counter = 0;
    visualization_msgs::Marker marker;
    marker.header.frame_id = "/map";
    marker.header.stamp = ros::Time();
    marker.ns = "my_namespace"; // what's this for?
    marker.id = counter;
    ++counter;
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.pose.position.x = 0.0;
    marker.pose.position.y = 0.0;
    marker.pose.position.z = 0.0;
    Eigen::Quaterniond quat;
    // these markers are in the camera's frame of reference
    quat.setIdentity();
    marker.pose.orientation.x = quat.x();
    marker.pose.orientation.y = quat.y();
    marker.pose.orientation.z = quat.z();
    marker.pose.orientation.w = quat.w();
    marker.scale.x = 0.02;
    marker.points.resize(2*p.size());
    for (size_t i = 0; i < p.size()-1 ; ++i) {
        marker.points[2*i+2].x = p[i](0);
        marker.points[2*i+2].y = p[i](1);
        marker.points[2*i+2].z = p[i](2);
        marker.points[2*i+3].x = p[i+1](0);
        marker.points[2*i+3].y = p[i+1](1);
        marker.points[2*i+3].z = p[i+1](2);
    }
    marker.points[0].x = p[p.size() - 1](0);
    marker.points[0].y = p[p.size() - 1](1);
    marker.points[0].z = p[p.size() - 1](2);
    marker.points[1].x = p[0](0);
    marker.points[1].y = p[0](1);
    marker.points[1].z = p[0](2);
    marker.color.a = 1.0;
    marker.color.r = c(0);
    marker.color.g = c(1);
    marker.color.b = c(2);
    usleep(100000);
    pub.publish(marker);
}

bool visualize_hulls(base_primitive* p, base_primitive* q)
{
    // get convex hull of p and q: P, Q
    std::vector<Vector3d, aligned_allocator<Vector3d> > hull_p;
    std::vector<Vector3d, aligned_allocator<Vector3d> > hull_q;
    std::vector<Vector3d, aligned_allocator<Vector3d> > hull;

    plane_primitive* pp = static_cast<plane_primitive*>(p);
    plane_primitive* pq = static_cast<plane_primitive*>(q);
    plane_primitive pr;
    pr.merge_planes(*pp, *pq);

    p->shape_points(hull_p);
    q->shape_points(hull_q);
    pr.shape_points(hull);

    hull_as_marker(hull, Vector3f(1, 0, 0));
    hull_as_marker(hull_p, Vector3f(0, 1, 0));
    hull_as_marker(hull_q, Vector3f(0, 0, 1));
}

bool are_coplanar(base_primitive* p, base_primitive* q)
{
    VectorXd par_p, par_q;
    Vector3d v_p, v_q;
    Vector3d c_p, c_q;
    if (p == q || p->get_shape() != base_primitive::PLANE ||
            q->get_shape() != base_primitive::PLANE) {
        return false;
    }
    p->direction_and_center(v_p, c_p);
    q->direction_and_center(v_q, c_q);
    p->shape_data(par_p);
    q->shape_data(par_q);
    Vector3d diff = c_p - c_q;
    diff.normalize();
    bool same_direction = acos(fabs(v_p.dot(v_q))) < angle_threshold;
    double dist_p = fabs(c_p.dot(par_q.segment<3>(0)) + par_q(3));
    double dist_q = fabs(c_q.dot(par_p.segment<3>(0)) + par_p(3));
    bool same_depth = dist_p < distance_threshold && dist_q < distance_threshold;
    return same_direction && same_depth;
}

bool contained_in_hull(const Vector3d& point, const std::vector<Vector3d, aligned_allocator<Vector3d> >& hull, const Vector3d& c)
{
    size_t n = hull.size();
    Vector2d d = (hull[1] - hull[0]).tail<2>();
    Vector2d v(-d(1), d(0));
    Vector2d point2 = point.tail<2>();
    double sign = v.dot((c  - hull[0]).tail<2>());
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        Vector2d p = hull[i].tail<2>();
        d = hull[j].tail<2>() - p;
        v = Vector2d(-d(1), d(0));
        if (sign*v.dot(point2  - p) < 0) {
            return false;
        }
    }
    return true;
}

plane_primitive* check_connectedness(base_primitive* p, base_primitive* q, primitive_extractor<pcl::PointXYZRGB>& extractor, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, float res)
{
    Vector3f center(-4.28197, 7.81703, 1.67754); // camera center, hardcoded for now

    MatrixXd points_p, points_q;
    extractor.primitive_inlier_points(points_p, p);
    extractor.primitive_inlier_points(points_q, q);
    visualize_hulls(p, q);

    // get convex hull of p and q: P, Q
    std::vector<Vector3d, aligned_allocator<Vector3d> > hull_p;
    std::vector<Vector3d, aligned_allocator<Vector3d> > hull_q;
    std::vector<Vector3d, aligned_allocator<Vector3d> > hull;

    plane_primitive* pp = static_cast<plane_primitive*>(p);
    plane_primitive* pq = static_cast<plane_primitive*>(q);
    plane_primitive* pr = new plane_primitive;
    pr->merge_planes(*pp, *pq);

    p->shape_points(hull_p);
    q->shape_points(hull_q);
    pr->shape_points(hull);

    // project the points into a common coordinate system, maybe the camera plane?
    Vector3d v, c;
    pr->direction_and_center(v, c); // should really be the rotation instead
    if (v.dot(c - center.cast<double>()) > 0) {
        v = -v;
    }
    float d = -v.dot(c);

    // just pick the basis of the first one
    VectorXd data;
    pr->shape_data(data);
    Quaterniond q_r(data(12), data(9), data(10), data(11));
    Matrix3d R(q_r);

    for (Vector3d& point : hull) {
        point = R.transpose()*point;
    }

    // lambdas don't need to be stored, they are just declarations, function pointer is enough
    auto first_comparator = [](const Vector3d& p1, const Vector3d& p2) { return p1(1) < p2(1); };
    auto second_comparator = [](const Vector3d& p1, const Vector3d& p2) { return p1(2) < p2(2); };
    Vector3d xmin = *std::min_element(hull.begin(), hull.end(), first_comparator);
    Vector3d xmax = *std::max_element(hull.begin(), hull.end(), first_comparator);
    Vector3d ymin = *std::min_element(hull.begin(), hull.end(), second_comparator);
    Vector3d ymax = *std::max_element(hull.begin(), hull.end(), second_comparator);

    double width = xmax(1) - xmin(1);
    double height = ymax(2) - ymin(2);

    std::cout << "xmin: " << xmin(1) << std::endl;
    std::cout << "xmax: " << xmax(1) << std::endl;
    std::cout << "ymin: " << ymin(2) << std::endl;
    std::cout << "ymax: " << ymax(2) << std::endl;

    int w = int(width/res);
    int h = int(height/res);

    cv::Mat im = cv::Mat::zeros(h, w, CV_32SC1);
    // project the entire pointcloud? sounds expensive...

    Vector3f point;
    Vector3f dir;
    Matrix3f Rf = R.cast<float>().transpose();
    Vector3f vf = v.cast<float>();
    Vector3f minf(0, xmin(1), ymin(2));
    for (const pcl::PointXYZRGB& pp : cloud->points) {
        if (isinf(pp.z) || isnan(pp.z)) {
            continue;
        }
        point = pp.getVector3fMap();
        dir = point - center;
        if (dir.dot(vf) > 0) { // correct
            continue;
        }
        dir.normalize();
        float prod = dir.dot(vf);
        if (fabs(prod) < 1e-5) {
            continue;
        }
        float a = -(d + vf.dot(point))/prod;
        point += a*dir;
        point = Rf*point; // check if inside convex hull
        if (!contained_in_hull(point.cast<double>(), hull, c)) {
            continue;
        }
        point -= minf;
        int y = int(point(2)/res);
        int x = int(point(1)/res);
        if (y < 0 || y >= h || x < 0 || x >= w) {
            continue;
        }
        if (a > 0) { // point is on the camera side of the plane
            im.at<int>(y, x) = 1;
        }
    }

    int x_p = -1;
    int y_p = -1;
    for (size_t i = 0; i < points_p.cols(); ++i) {
        point = Rf*points_p.col(i).cast<float>() - minf;
        int y = int(point(2)/res);
        int x = int(point(1)/res);
        if (y < 0 || y >= h || x < 0 || x >= w) {
            continue;
        }
        im.at<int>(y, x) = 1;
        if (x_p == -1) {
            x_p = x;
            y_p = y;
        }
    }

    int x_q = -1;
    int y_q = -1;
    for (size_t i = 0; i < points_q.cols(); ++i) {
        point = Rf*points_q.col(i).cast<float>() - minf;
        int y = int(point(2)/res);
        int x = int(point(1)/res);
        if (y < 0 || y >= h || x < 0 || x >= w) {
            continue;
        }
        im.at<int>(y, x) = 1;
        if (x_q == -1) {
            x_q = x;
            y_q = y;
        }
    }
    if (x_p == -1 || x_q == -1) {
        return false;
    }

    cv::Mat imcopy = im.clone();
    imcopy = 65535*imcopy;
    cv::imshow("im", imcopy);
    cv::waitKey(0);

    int largest = base_primitive::find_blobs(im, false, false);
    cv::Mat result = cv::Mat::zeros(h, w, CV_32SC1);
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            if (im.at<int>(i, j) == largest) {
                result.at<int>(i, j) = 65535;
            }
        }
    }

    cv::imshow("result", result);
    cv::waitKey(0);

    u_char c1 = im.at<int>(y_p, x_p);
    u_char c2 = im.at<int>(y_q, x_q);
    if (c1 == c2) {
        return pr;
    }
    else {
        delete pr;
        return NULL;
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_refinement");
    ros::NodeHandle n;

    ros::NodeHandle pn("~");
    std::string output;
    pn.param<std::string>("output", output, std::string("primitive_marker"));
    pub = n.advertise<visualization_msgs::Marker>(output, 1);

    if (argc < 2) {
        std::cout << "Please supply path to pointcloud to process..." << std::endl;
        return 0;
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    std::string filename(argv[1]);

    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (filename, *cloud) == -1)
    {
        std::cout << "Couldn't read file " << filename << std::endl;
        return 0;
    }

    // sphere_primitive and cylinder_primitive have not been ported to the new framework yet
    float subsampling_voxel_size = 0.04;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr subsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(subsampling_voxel_size, subsampling_voxel_size, subsampling_voxel_size);
    sor.filter(*subsampled_cloud);
    std::cout << "Downsampled to " << subsampled_cloud->points.size() << std::endl;

    std::vector<base_primitive*> primitives = { new plane_primitive() };
    // sphere_primitive and cylinder_primitive have not been ported to the new framework yet
    primitive_params params;
    params.number_disjoint_subsets = 10;
    params.octree_res = 0.3;
    params.normal_neigbourhood = 0.07;
    params.inlier_threshold = 0.06;
    params.angle_threshold = 0.4;
    params.add_threshold = 0.01;
    params.min_shape = 1000;
    params.inlier_min = params.min_shape;
    params.connectedness_res = 0.12;
    params.distance_threshold = 4.0;

    primitive_visualizer<pcl::PointXYZRGB> viewer;
    primitive_extractor<pcl::PointXYZRGB> extractor(subsampled_cloud, primitives, params, &viewer);
    viewer.cloud = extractor.get_cloud();
    viewer.cloud_changed = true;
    viewer.cloud_normals = extractor.get_normals();
    viewer.normals_changed = true;
    viewer.create_thread();
    std::vector<base_primitive*> extracted;
    extractor.extract(extracted);
    // in the primitives are the indices of all the inliers
    // use these to find out which planes might be connected through an occluded region

    // first question: where is the camera situated? lets assume (0, 0, 0) for now

    // find all co-planar planes
    typedef std::pair<size_t, size_t> plane_pair;
    std::vector<plane_pair> plane_pairs;

    std::cout << "Primitives: " << extracted.size() << std::endl;
    while (true) {
        bool do_break = false;
        for (size_t i = 0; i < extracted.size(); ++i) {
            base_primitive* p = extracted[i];
            for (size_t j = 0; j < i; ++j) {
                base_primitive* q = extracted[j];
                if (!are_coplanar(p, q)) {
                    continue;
                }
                std::cout << "The primitives are co-planar" << std::endl;
                plane_primitive* pp = check_connectedness(extracted[i], extracted[j], extractor, subsampled_cloud, 1.2*params.connectedness_res);
                if (pp != NULL) { // once found, redo the whole scheme for i
                    std::cout << "And they may also be connected" << std::endl;
                    plane_pairs.push_back(plane_pair(i, j));
                    extracted.erase(std::remove_if(extracted.begin(), extracted.end(), [=](const base_primitive* b) { return b == p || b == q; } ), extracted.end());
                    delete p;
                    delete q;
                    extracted.push_back(pp);
                    do_break = true;
                    break;
                }
            }
            if (do_break) {
                break;
            }
        }
        if (!do_break) {
            break;
        }
    }

    for (base_primitive* b : extracted) {
        if (b->get_shape() != base_primitive::PLANE) {
            continue;
        }
        std::vector<Vector3d, aligned_allocator<Vector3d> > hull;
        b->shape_points(hull);
        hull_as_marker(hull, Vector3f(0, 1, 0));
    }

    // find regions through which they would be connected

    // are these regions occluded? -> connect

    std::cout << "The algorithm has finished..." << std::endl;

    viewer.join_thread();

    return 0;
}
