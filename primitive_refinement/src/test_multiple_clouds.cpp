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

#include "visualization_msgs/Marker.h"

#include "primitive_core.h"
#include "plane_primitive.h"
#include "sphere_primitive.h"
#include "cylinder_primitive.h"

#include "refinement_core.h"
#include "graph_extractor.h"

using namespace Eigen;

ros::Publisher pub;

bool create_folder(const std::string& folder)
{
    boost::filesystem::path dir(folder);
    if (!boost::filesystem::exists(dir) && !boost::filesystem::create_directory(dir))  {
            std::cout << "Failed to create directory " << folder << "..." << std::endl;
            return false;
    }
    return true;
}

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
        std::cout << "Please supply folder to process..." << std::endl;
        return 0;
    }
    std::string folder(argv[1]);
    std::string graph_folder = folder + string("/graphs");

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
    for (size_t i = 0; i < clouds.size(); ++i) {
        // sphere_primitive and cylinder_primitive have not been ported to the new framework yet
        /*float subsampling_voxel_size = 0.04;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr subsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(subsampling_voxel_size, subsampling_voxel_size, subsampling_voxel_size);
        sor.filter(*subsampled_cloud);
        std::cout << "Downsampled to " << subsampled_cloud->points.size() << std::endl;*/

        vector<base_primitive*> primitives = { new plane_primitive() };
        // sphere_primitive and cylinder_primitive have not been ported to the new framework yet
        primitive_params params;
        params.number_disjoint_subsets = 30;
        params.octree_res = 0.3;
        params.normal_neigbourhood = 0.07;
        params.inlier_threshold = 0.06;
        params.angle_threshold = 0.4;
        params.add_threshold = 0.01;
        params.min_shape = 3000;
        params.inlier_min = params.min_shape;
        params.connectedness_res = 0.08;
        params.distance_threshold = 0;

        cout << "Starting algorithm..." << endl;
        cout << "Cameras size: " << cameras[i].size() << endl;
        cout << "Clouds size: " << clouds[i].size() << endl;

        primitive_visualizer<pcl::PointXYZRGB> viewer;
        primitive_refiner<pcl::PointXYZRGB> extractor(cameras[i], clouds[i], primitives, params, &viewer);
        viewer.cloud = extractor.get_cloud();
        viewer.cloud_changed = true;
        viewer.cloud_normals = extractor.get_normals();
        viewer.normals_changed = true;
        viewer.create_thread();
        vector<base_primitive*> extracted;
        extractor.extract(extracted);
        // in the primitives are the indices of all the inliers
        // use these to find out which planes might be connected through an occluded region

        // first question: where is the camera situated? lets assume (0, 0, 0) for now

        cout << "Primitives: " << extracted.size() << endl;

        for (base_primitive* b : extracted) {
            if (b->get_shape() != base_primitive::PLANE) {
                continue;
            }
            vector<Vector3d, aligned_allocator<Vector3d> > hull;
            b->shape_points(hull);
            hull_as_marker(hull, Vector3f(0, 1, 0));
        }

        vector<Eigen::MatrixXd> inliers;
        inliers.resize(extracted.size());
        for (int i = 0; i < extracted.size(); ++i) {
            extractor.primitive_inlier_points(inliers[i], extracted[i]);
        }

        graph_extractor ge(extracted, inliers, 0.1); // how close to be considered connected
        stringstream ss;
        ss << "/graph" << setfill('0') << setw(6) << i << ".dot";
        cout << ss.str() << endl;
        string graphfile = graph_folder + ss.str();
        string imagefile = graph_folder + "test.png";
        ge.generate_dot_file(graphfile);
        string command = "dot -Tpng " + graphfile + " > " + imagefile + " && gvfs-open " + imagefile;
        system(command.c_str());

        ss.str("");
        ss << "/indices" << setfill('0') << setw(6) << i << ".txt";
        string indexfile = graph_folder + ss.str();
        ge.generate_index_file(indexfile);

        // find regions through which they would be connected

        // are these regions occluded? -> connect

        cout << "The algorithm has finished..." << endl;

        viewer.join_thread();
    }

    return 0;
}
