#include "pointcloud_common.h"
#include <pcl/octree/octree.h>

template <typename Point>
bool pointcloud_common<Point>::segment(cloud_type& first_segmented, cloud_type& second_segmented) const
{
    pcl::octree::OctreePointCloud<point_type> first_octree(resolution);
    first_octree.setInputCloud(first);
    pcl::octree::OctreePointCloud<point_type> second_octree(resolution);
    second_octree.setInputCloud(second);
    first_octree.addPointsFromInputCloud();
    second_octree.addPointsFromInputCloud();
    for (const point_type& p : second->points) {
        if (first_octree.isVoxelOccupiedAtPoint(p)) {
            second_segmented.push_back(p);
        }
    }
    for (const point_type& p : first->points) {
        if (second_octree.isVoxelOccupiedAtPoint(p)) {
            first_segmented.push_back(p);
        }
    }
    /*first_octree::AlignedPointTVector first_centers;
    first_octree.getOccupiedVoxelCenters(first_centers);
    second_octree::AlignedPointTVector second_centers;
    second_octree.getOccupiedVoxelCenters(second_centers);*/
    return first_segmented.size() > threshold && second_segmented.size() > threshold;
}

template <typename Point>
double pointcloud_common<Point>::overlap_volume() const
{
    pcl::octree::OctreePointCloud<point_type> first_octree(resolution);
    first_octree.setInputCloud(first);
    pcl::octree::OctreePointCloud<point_type> second_octree(resolution);
    second_octree.setInputCloud(second);
    first_octree.addPointsFromInputCloud();
    second_octree.addPointsFromInputCloud();
    typename pcl::octree::OctreePointCloud<point_type>::AlignedPointTVector vec;
    first_octree.getOccupiedVoxelCenters(vec);
    size_t counter = 0;
    for (const point_type& p : vec) {
        if (second_octree.isVoxelOccupiedAtPoint(p)) {
            ++counter;
        }
    }
    return double(counter)*resolution*resolution*resolution;
}

template <typename Point>
void pointcloud_common<Point>::set_input(const_ptr_type firstin)
{
    first = firstin;
}

template <typename Point>
void pointcloud_common<Point>::set_target(const_ptr_type secondin)
{
    second = secondin;
}
