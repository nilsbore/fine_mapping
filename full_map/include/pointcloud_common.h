#ifndef POINTCLOUD_COMMON_H
#define POINTCLOUD_COMMON_H

#include <pcl/pcl_base.h>

template <typename Point>
class pointcloud_common {
public:
    typedef Point point_type;
    typedef pcl::PointCloud<Point> cloud_type;
    typedef typename cloud_type::Ptr ptr_type;
    typedef typename cloud_type::ConstPtr const_ptr_type;
private:
    float resolution;
    const_ptr_type first;
    const_ptr_type second;
    size_t threshold;
public:
    void set_input(const_ptr_type firstin);
    void set_target(const_ptr_type secondin);
    bool segment(cloud_type& first_segmented, cloud_type& second_segmented) const;
    pointcloud_common(float resolution, size_t threshold = 10000) : resolution(resolution), threshold(threshold) {}
};

#include "pointcloud_common.hpp"

#endif // POINTCLOUD_COMMON_H
