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

#include "refinement_core.h"

using namespace Eigen;

ros::Publisher pub;

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

/*bool visualize_hulls(base_primitive* p, base_primitive* q)
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
}*/

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
    primitive_refiner<pcl::PointXYZRGB> extractor(subsampled_cloud, primitives, params, &viewer);
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
