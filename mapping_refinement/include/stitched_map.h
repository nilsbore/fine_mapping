#ifndef STITCHED_MAP_H
#define STITCHED_MAP_H

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include "scan.h"

class stitched_map
{
private:
    std::vector<scan*>& scans;
    void average_scans(scan& scan1, scan& scan2, cv::Mat& counter1, cv::Mat& counter2);
    void construct_counters(std::vector<cv::Mat>& counters);
public:
    void visualize();
    stitched_map(std::vector<scan*>& scans) : scans(scans)
    {

    }
};

#endif // STITCHED_MAP_H
