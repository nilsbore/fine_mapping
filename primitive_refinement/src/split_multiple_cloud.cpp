#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <functional>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "pcl_ros/point_cloud.h"

#include "visualization_msgs/Marker.h"

#include "primitive_core.h"
#include "plane_primitive.h"
#include "sphere_primitive.h"
#include "cylinder_primitive.h"

#include "refinement_core.h"
#include "graph_extractor.h"

using namespace Eigen;

ros::Publisher pub;
ros::Publisher cloud_pub;

bool create_folder(const std::string& folder)
{
    boost::filesystem::path dir(folder);
    if (!boost::filesystem::exists(dir) && !boost::filesystem::create_directory(dir))  {
            std::cout << "Failed to create directory " << folder << "..." << std::endl;
            return false;
    }
    return true;
}

void hull_as_marker(std::vector<Vector3d, aligned_allocator<Vector3d> >& p, const Vector3f& c, const Vector3d& v, const Vector3d& center)
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

    marker.id = counter;
    ++counter;
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::ARROW;
    //marker.pose.position.x = 0.1;
    //marker.pose.position.y = 0.05;
    //marker.pose.position.z = 0;
    marker.scale.x = 0.05;
    marker.scale.y = 0.1;
    marker.scale.z = 0;
    marker.points.resize(2);
    marker.points[0].x = center(0);
    marker.points[0].y = center(1);
    marker.points[0].z = center(2);
    marker.points[1].x = center(0) + v(0);
    marker.points[1].y = center(1) + v(1);
    marker.points[1].z = center(2) + v(2);
    usleep(100000);
    pub.publish(marker);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_refinement");
    ros::NodeHandle n;

    ros::NodeHandle pn("~");
    string output;
    pn.param<std::string>("output", output, std::string("/primitive_marker"));
    pub = n.advertise<visualization_msgs::Marker>(output, 1);

    string cloud_out;
    pn.param<std::string>("cloud", cloud_out, std::string("/cloud_pcd"));
    cloud_pub = n.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(output, 1);

    if (argc < 2) {
        std::cout << "Please supply folder to process..." << std::endl;
        return 0;
    }
    std::string folder(argv[1]);
    std::string graph_folder = folder + string("/graphs9");

    vector<vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > clouds;
    //clouds.push_back(subsampled_cloud);
    vector<vector<Vector3d, aligned_allocator<Vector3d> > > cameras;
    //cameras.push_back(Vector3d(-4.28197, 7.81703, 1.67754)); // camera center, hardcoded for now
    boost::filesystem::path p(folder);   // p reads clearer than argv[1] in the following code
    try {
        if (boost::filesystem::exists(p) && boost::filesystem::is_directory(p)) { // does p actually exist?
            cout << p << " is a directory containing:\n";

            typedef vector<boost::filesystem::path> vec;             // store paths,

            vec v;                                // so we can sort them later
            copy(boost::filesystem::directory_iterator(p), boost::filesystem::directory_iterator(), back_inserter(v));
            sort(v.begin(), v.end()); // sort, since directory iteration
            // is not ordered on some file systems, gets rooms in right order

            for (vec::const_iterator it (v.begin()); it != v.end(); ++it)
            {
                string roomname = it->stem().string();
                if (roomname.size() < 6) {
                    continue;
                }

                if (boost::filesystem::is_directory(*it)) { // is p a directory?
                    // if room, initialize containers
                    if (roomname.substr(0, 4) != "room") {
                        continue;
                    }
                    cout << *it << endl;
                    clouds.push_back(vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>());
                    cameras.push_back(vector<Vector3d, aligned_allocator<Vector3d> >());
                    vec u;                                // so we can sort them later
                    copy(boost::filesystem::directory_iterator(*it), boost::filesystem::directory_iterator(), back_inserter(u));
                    sort(u.begin(), u.end()); // sort, since directory iteration
                    // is not ordered on some file systems, gets clouds in right order
                    for (vec::const_iterator sit (u.begin()); sit != u.end(); ++sit)
                    {
                        string cloudname = sit->stem().string();
                        if (cloudname.size() < 7) {
                            continue;
                        }
                        if (boost::filesystem::is_regular_file(*sit)) { // is p a regular file?
                            cout << *sit << " size is " << boost::filesystem::file_size(*sit) << '\n';
                            if (cloudname.substr(0, 5) == "cloud") { // cloud
                                clouds.back().push_back(pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>()));
                                if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (sit->string(), *clouds.back().back()) == -1)
                                {
                                    cout << "Couldn't read file " << *sit << endl;
                                    return 0;
                                }
                                cout << "Read " << sit->string() << " successfully, " << clouds.back().back()->size() << " points" << endl;
                            }
                            else if (cloudname == "origins") { // origins
                                ifstream tfile(sit->string());
                                string line;
                                while (getline(tfile, line)) {
                                    if (line.size() < 4) {
                                        continue;
                                    }
                                    vector<string> strs;
                                    boost::split(strs, line, boost::is_any_of(" \n"));
                                    cameras.back().push_back(Eigen::Vector3d());
                                    size_t j = 0;
                                    for (const string& str : strs) {
                                        if (str.empty()) {
                                            continue;
                                        }
                                        if (j > 2) {
                                            cout << "Too many entries on line, quitting..." << endl;
                                            break;
                                        }
                                        cameras.back().back()(j) = stod(str);
                                        ++j;
                                    }
                                    cout << cameras.back().back().transpose() << endl;
                                }
                                tfile.close();
                                // readline, pushback
                            }
                        }
                    }
                }

            }
        }
        else {
            cout << p << " does not exist\n";
        }
    }
    catch (const boost::filesystem::filesystem_error& ex)
    {
        cout << ex.what() << '\n';
    }

    create_folder(graph_folder);
    //for (size_t i = 0; i < clouds.size(); ++i) {
    size_t i = 0;
    float patch_width = 3.0;
    // sphere_primitive and cylinder_primitive have not been ported to the new framework yet
    /*float subsampling_voxel_size = 0.04;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr subsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(subsampling_voxel_size, subsampling_voxel_size, subsampling_voxel_size);
        sor.filter(*subsampled_cloud);
        std::cout << "Downsampled to " << subsampled_cloud->points.size() << std::endl;*/

    auto x_comp = [](const pcl::PointXYZRGB& p1, const pcl::PointXYZRGB& p2) { return p1.x < p2.x; };
    auto y_comp = [](const pcl::PointXYZRGB& p1, const pcl::PointXYZRGB& p2) { return p1.y < p2.y; };
    auto maxx = std::max_element(clouds[i]->points.begin(), clouds[i]->points.end(), x_comp)->x;
    auto minx = std::min_element(clouds[i]->points.begin(), clouds[i]->points.end(), x_comp)->x;
    auto miny = std::min_element(clouds[i]->points.begin(), clouds[i]->points.end(), y_comp)->y;
    auto maxy = std::max_element(clouds[i]->points.begin(), clouds[i]->points.end(), y_comp)->y;

    vector<base_primitive*> primitives = { new plane_primitive() };
    // sphere_primitive and cylinder_primitive have not been ported to the new framework yet
    primitive_params params;
    // Not subsampled
    /*params.number_disjoint_subsets = 60;
        params.octree_res = 0.6;
        params.normal_neigbourhood = 0.07;
        params.inlier_threshold = 0.07;
        params.angle_threshold = 0.4;
        params.add_threshold = 0.01;
        params.min_shape = 6000;
        params.inlier_min = params.min_shape;
        params.connectedness_res = 0.06;
        params.distance_threshold = 0;*/
    // 0.05 subsampled
    params.number_disjoint_subsets = 80;
    params.octree_res = 0.6;
    params.normal_neigbourhood = 0.09;
    params.inlier_threshold = 0.12;
    params.angle_threshold = 0.4;
    params.add_threshold = 0.05;
    params.min_shape = 3000;
    params.inlier_min = params.min_shape;
    params.connectedness_res = 0.07;
    params.distance_threshold = 0;

    cout << "Starting algorithm..." << endl;
    cout << "Cameras size: " << cameras[i].size() << endl;
    cout << "Clouds size: " << clouds[i].size() << endl;

    primitive_refiner<pcl::PointXYZRGB> extractor(cameras[i], clouds[i], primitives, params, NULL);//&viewer);
    /*auto tcloud = extractor.get_cloud();
        auto tnormals = extractor.get_normals();

        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->setBackgroundColor(0, 0, 0);
        viewer->initCameraParameters();
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(tcloud, 0, 255, 0);
        viewer->addPointCloud<pcl::PointXYZRGB>(tcloud, single_color, "cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 1, "cloud");
        viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(tcloud, tnormals, 20, 1e-1f, "normals");
        while (!viewer->wasStopped())
        {
            viewer->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }*/
    vector<base_primitive*> extracted;
    std::vector<std::vector<int> > camera_ids;
    extractor.extract(extracted, camera_ids);

    cout << "Primitives: " << extracted.size() << endl;

    for (base_primitive* b : extracted) {
        if (b->get_shape() != base_primitive::PLANE) {
            continue;
        }
        vector<Vector3d, aligned_allocator<Vector3d> > hull;
        b->shape_points(hull);
        Eigen::Vector3d c, v;
        b->direction_and_center(v, c);
        hull_as_marker(hull, Vector3f(0, 1, 0), v, c);
    }

    vector<Eigen::MatrixXd> inliers;
    inliers.resize(extracted.size());
    for (int i = 0; i < extracted.size(); ++i) {
        extractor.primitive_inlier_points(inliers[i], extracted[i]);
    }

    graph_extractor ge(extracted, inliers, camera_ids, 1.0); // how close to be considered connected

    stringstream ss;
    ss << "/graph" << setfill('0') << setw(6) << i << ".dot";
    cout << ss.str() << endl;
    string graphfile = graph_folder + ss.str();
    string imagefile = graph_folder + "test.png";
    ge.generate_dot_file(graphfile);
    string command = "dot -Tpng " + graphfile + " > " + imagefile + " && gvfs-open " + imagefile;
    //system(command.c_str());

    ss.str("");
    ss << "/indices" << setfill('0') << setw(6) << i << ".txt";
    string indexfile = graph_folder + ss.str();
    ge.generate_index_file(indexfile);

    ss.str("");
    ss << "/cloud" << setfill('0') << setw(6) << i << ".pcd";
    string cloudfile = graph_folder + ss.str();
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud = extractor.get_cloud();
    pcl::io::savePCDFileBinary(cloudfile, *cloud);

    pcl::PointCloud<pcl::PointXYZRGB> pub_cloud;
    pcl::copyPointCloud(*cloud, pub_cloud);
    pub_cloud.header.frame_id = "/map";
    cloud_pub.publish(pub_cloud);

    cout << "The algorithm has finished..." << endl;

    return 0;
}
