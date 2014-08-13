#include "scan.h"
#include "fine_mapping.h"
#include "stitched_map.h"

#include <string>

int main(int argc, char** argv)
{
    std::string folder = std::string(getenv("HOME")) + std::string("/.ros/mapping_refinement");
    std::vector<std::string> scan_files;
    std::vector<std::string> t_files;
    std::vector<scan*> scans;

    size_t n = 50;
    // set the filenames for the pointclouds and transforms, initialize scan objects
    for (size_t i = 0; i < n; ++i) {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << i;
        scan_files.push_back(folder + std::string("/shot") + ss.str() + std::string(".pcd"));
        t_files.push_back(folder + std::string("/transform") + ss.str() + std::string(".txt"));
        scans.push_back(new scan(scan_files.back(), t_files.back()));
    }

    //view_registered_pointclouds(scan_files, scans);
    fine_mapping f(scans);
    f.build_graph();
    f.optimize_graph();
    stitched_map map(scans);
    map.visualize();

    return 0;
}
