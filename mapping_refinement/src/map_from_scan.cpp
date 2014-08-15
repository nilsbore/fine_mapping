#include "simpleXMLparser.h"
#include "scan.h"
#include "fine_mapping.h"
#include "stitched_map.h"

#include <tf/tf.h>

int main(int argc, char** argv)
{
    std::vector<scan*> scans;

    {
        SimpleXMLParser<pcl::PointXYZRGB> parser;
        SimpleXMLParser<pcl::PointXYZRGB>::RoomData room_data;

        room_data = parser.loadRoomFromXML("/home/nbore/Data/20140813/patrol_run_1/room_0/room.xml");

        std::cout << "Complete cloud size: " << room_data.completeRoomCloud->points.size() << std::endl;

        tf::Matrix3x3 tfbasis;
        tf::Vector3 tforigin;
        Eigen::Matrix3f basis;
        Eigen::Vector3f origin;
        Eigen::Matrix3f K;
        K << 525, 0, 319.5,
             0, 525, 239.5,
             0, 0, 1;
        for (size_t i = 0; i < room_data.vIntermediateRoomClouds.size(); ++i)
        {
            std::cout << "Intermediate cloud size: "<< room_data.vIntermediateRoomClouds[i]->points.size() << std::endl;
            tfbasis = room_data.vIntermediateRoomCloudTransforms[i].getBasis();
            tforigin = room_data.vIntermediateRoomCloudTransforms[i].getOrigin();

            for (size_t i = 0; i < 3; ++i) {
                origin(i) = tforigin.m_floats[i];
            }
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    basis(i, j) = tfbasis.getRow(i).m_floats[j];
                }
            }
            scans.push_back(new scan(*room_data.vIntermediateRoomClouds[i], origin, basis, K));
        }
    }

    fine_mapping f(scans);
    f.build_graph();
    f.optimize_graph();
    stitched_map map(scans);
    map.visualize();
    return 0;
}
