#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
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

#define VISUALIZE true

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

void rectify_normals(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, pcl::PointCloud<pcl::Normal>::Ptr cloud_normals,
                     vector<Vector3d, aligned_allocator<Vector3d> >& cameras, std::vector<size_t>& dividers)
{
    size_t divider = 0;
    Eigen::Vector3f cam = cameras[0].cast<float>();
    float prod;
    //pcl::Normal normal;
    //Eigen::Vector3f n;
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        if (i >= dividers[divider]) {
            ++divider;
            cam = cameras[divider].cast<float>();
        }
        //normal = cloud_normals->points[i];
        //n = Vector3f(normal.normal_x, normal.normal_y, normal.)
        prod = cloud_normals->points[i].getNormalVector3fMap().dot(cam - cloud->points[i].getVector3fMap());
        if (prod < 0) {
            cloud_normals->points[i].normal_x *= -1.0;
            cloud_normals->points[i].normal_y *= -1.0;
            cloud_normals->points[i].normal_z *= -1.0;
        }
    }
}

void update_dividers(vector<size_t>& new_dividers, vector<int>& indices, vector<size_t>& dividers,
                     vector<Vector3d, aligned_allocator<Vector3d> >& new_cameras, vector<Vector3d, aligned_allocator<Vector3d> >& cameras,
                     vector<int>& camera_ids)
{
    size_t divider = 0;
    size_t counter = 0;
    while (counter < indices.size()) {
        while (dividers[divider] < indices[counter]) {
            ++divider;
        }
        while (indices[counter] < dividers[divider]) { // loop to next containing divider
            ++counter;
        }
        new_dividers.push_back(counter);
        new_cameras.push_back(cameras[divider]);
        camera_ids.push_back(divider);
        ++divider;
        ++counter;
    }
}

void estimate_normals(vector<pcl::PointCloud<pcl::Normal>::Ptr>& cloud_normals,
                      vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clouds,
                      vector<Vector3d, aligned_allocator<Vector3d> >& cameras,
                      float normal_neighbourhood)
{
    cloud_normals.resize(clouds.size());
    for (size_t i = 0; i < clouds.size(); ++i) {
        // Create the normal estimation class, and pass the input dataset to it
        pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
        ne.setInputCloud(clouds[i]);

        // Create an empty kdtree representation, and pass it to the normal estimation object.
        // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
        typedef pcl::search::KdTree<pcl::PointXYZRGB> kd_tree_type;
        typedef typename kd_tree_type::Ptr kd_tree_type_ptr;
        kd_tree_type_ptr tree(new kd_tree_type());
        //pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB>::Ptr ptr(octree);
        ne.setSearchMethod(tree);

        // Use all neighbors in a sphere of radius normal_radius m
        ne.setRadiusSearch(normal_neighbourhood);

        // Compute the features
        cloud_normals[i] = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
        ne.compute(*cloud_normals[i]);

        float prod;
        Vector3f cam = cameras[i].cast<float>();
        for (size_t j = 0; j < clouds[i]->size(); ++j) {
            prod = cloud_normals[i]->points[j].getNormalVector3fMap().dot(cam - clouds[i]->points[j].getVector3fMap());
            if (prod < 0) {
                cloud_normals[i]->points[j].getNormalVector3fMap() *= -1.0;
            }
        }
    }
}

pcl::PointCloud<pcl::Normal>::Ptr fuse_normals(std::vector<pcl::PointCloud<pcl::Normal>::Ptr>& cloud_normals_v)
{
    pcl::PointCloud<pcl::Normal>::Ptr rtn(new pcl::PointCloud<pcl::Normal>);
    for (const pcl::PointCloud<pcl::Normal>::Ptr& cloud : cloud_normals_v) {
        rtn->points.insert(rtn->points.end(), cloud->points.begin(), cloud->points.end());
    }
    return rtn;
}

void generate_index_file(const std::string& filename, std::vector<base_primitive*>& primitives, std::vector<int>& super_indices)
{
    std::ofstream file;
    file.open(filename);
    for (base_primitive* p : primitives) {
        for (const int& ind : p->supporting_inds) {
            file << super_indices[ind] << " ";
        }
        file << "\n";
    }
    file.close();
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
    string folder(argv[1]);
    string graph_folder = folder + string("/graphs34");
    string picked_folder = graph_folder + string("/picked");

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

    vector<string> picked_clouds;
    boost::filesystem::path pp(picked_folder);   // p reads clearer than argv[1] in the following code
    try {
        if (boost::filesystem::exists(pp) && boost::filesystem::is_directory(pp)) { // does p actually exist?
            cout << pp << " is a directory containing:\n";

            typedef vector<boost::filesystem::path> vec;             // store paths,

            vec v;                                // so we can sort them later
            copy(boost::filesystem::directory_iterator(pp), boost::filesystem::directory_iterator(), back_inserter(v));
            sort(v.begin(), v.end()); // sort, since directory iteration
            // is not ordered on some file systems, gets rooms in right order

            for (vec::const_iterator it (v.begin()); it != v.end(); ++it)
            {
                string cloudname = it->stem().string();
                if (cloudname.size() < 14) {
                    continue;
                }
                if (boost::filesystem::is_regular_file(*it)) {
                    if (cloudname.substr(0, 12) == "picked_cloud") {
                        picked_clouds.push_back(it->string());
                    }
                }
            }
        }
    }
    catch (const boost::filesystem::filesystem_error& ex)
    {
        cout << ex.what() << '\n';
    }

    cout << "Got all the clouds and camera centers..." << endl;

    primitive_params params;
    /*params.number_disjoint_subsets = 2;
    params.octree_res = 0.6;
    params.normal_neigbourhood = 0.07;
    params.inlier_threshold = 0.15;
    params.angle_threshold = 0.55;
    params.add_threshold = 0.05;
    params.min_shape = 1000;
    params.inlier_min = params.min_shape;
    params.connectedness_res = 0.05;
    params.distance_threshold = 0;*/
    params.number_disjoint_subsets = 2;
    params.octree_res = 0.3;
    params.normal_neigbourhood = 0.12;
    params.inlier_threshold = 0.15;//0.15; // 0.17 for experiments
    params.angle_threshold = 0.55; // 0.55
    params.add_threshold = 0.01;
    params.min_shape = int(200*3.0*3.0);//int(720*patch_width);//int(77.7*patch_width*patch_width);//700; // too small?
    params.inlier_min = params.min_shape;
    params.connectedness_res = 0.06; // 0.06
    params.distance_threshold = 0;

    size_t i = 0;

    vector<pcl::PointCloud<pcl::Normal>::Ptr> cloud_normals_v;
    estimate_normals(cloud_normals_v, clouds[i], cameras[i], params.normal_neigbourhood);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(primitive_refiner<pcl::PointXYZRGB>::fuse_clouds(clouds[i]));
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(fuse_normals(cloud_normals_v));
    std::vector<size_t> dividers;
    primitive_refiner<pcl::PointXYZRGB>::create_dividers(dividers, clouds[i]);

    cout << "Fused the input clouds and created original dividers..." << endl;
    cout << "Fused cloud size: " << cloud->size() << endl;

    cout << "Fixed the normals to be consistent with cameras..." << endl;

    auto x_comp = [](const pcl::PointXYZRGB& p1, const pcl::PointXYZRGB& p2) { return p1.x < p2.x; };
    auto y_comp = [](const pcl::PointXYZRGB& p1, const pcl::PointXYZRGB& p2) { return p1.y < p2.y; };

    vector<base_primitive*> primitives = { new plane_primitive() };
    size_t counter = 0;
    for (string picked_name : picked_clouds) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr picked_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (picked_name, *picked_cloud) == -1)
        {
            cout << "Couldn't read file " << picked_name << endl;
            return 0;
        }
        float maxx = std::max_element(picked_cloud->points.begin(), picked_cloud->points.end(), x_comp)->x;
        float minx = std::min_element(picked_cloud->points.begin(), picked_cloud->points.end(), x_comp)->x;
        float miny = std::min_element(picked_cloud->points.begin(), picked_cloud->points.end(), y_comp)->y;
        float maxy = std::max_element(picked_cloud->points.begin(), picked_cloud->points.end(), y_comp)->y;

        cout << "Found min x, y points, min: (" << minx << ", " << miny << "), max: (" << maxx << ", " << maxy << ")..." << endl;

        Eigen::Vector4f min_point;
        min_point[0] = minx;  // define minimum point x
        min_point[1] = miny;  // define minimum point y
        min_point[2] = -1.0;  // define minimum point z
        Eigen::Vector4f max_point;
        max_point[0]= maxx;  // define max point x
        max_point[1]= maxy;  // define max point y
        max_point[2] = 4.0;  // define max point z

        pcl::CropBox<pcl::PointXYZRGB> crop_filter;
        crop_filter.setInputCloud (cloud);
        crop_filter.setMin(min_point);
        crop_filter.setMax(max_point);
        boost::shared_ptr<std::vector<int> > indices(new std::vector<int>);
        crop_filter.filter(*indices);
        if (indices->size() < 5*params.inlier_min) {
            cout << "Too few points in box..." << endl;
            continue;
        }

        cout << "Got a box with enough points..." << endl;

        std::sort(indices->begin(), indices->end());
        std::vector<size_t> new_dividers;
        vector<Vector3d, aligned_allocator<Vector3d> > new_cameras;
        vector<int> mapped_camera_ids;
        update_dividers(new_dividers, *indices, dividers, new_cameras, cameras[i], mapped_camera_ids);

        cout << "Updated dividers..." << endl;

        pcl::ExtractIndices<pcl::Normal> normal_extract;
        normal_extract.setInputCloud(cloud_normals);
        normal_extract.setIndices(indices);
        pcl::PointCloud<pcl::Normal>::Ptr normals_out(new pcl::PointCloud<pcl::Normal>);
        normal_extract.filter(*normals_out);

        cout << "Extracted normals in box..." << endl;

        pcl::ExtractIndices<pcl::PointXYZRGB> cloud_extract;
        cloud_extract.setInputCloud(cloud);
        cloud_extract.setIndices(indices);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud_extract.filter(*cloud_out);

        cout << "Extracted points in box..." << endl;

        cout << "Starting algorithm..." << endl;
        cout << "Cameras size: " << new_cameras.size() << endl;
        cout << "Clouds size: " << cloud->size() << endl;

        primitive_refiner<pcl::PointXYZRGB> extractor(new_cameras, new_dividers, cloud_out, normals_out, primitives, params, NULL);//&viewer);
        vector<base_primitive*> extracted;
        vector<vector<int> > camera_ids;
        extractor.extract(extracted, camera_ids);

        vector<Eigen::MatrixXd> inliers;
        inliers.resize(extracted.size());
        for (int i = 0; i < extracted.size(); ++i) {
            for (int& id : camera_ids[i]) {
                id = mapped_camera_ids[id];
            }
            extractor.primitive_inlier_points(inliers[i], extracted[i]);
        }

        // send in the actual camera ids so we can match adjacent cameras
        graph_extractor ge(extracted, inliers, camera_ids, 0.25, params.inlier_threshold); // how close to be considered connected

        stringstream ss;
        ss << "/graph" << setfill('0') << setw(6) << counter << ".dot";
        cout << "Writing dot file" << ss.str() << "..." << endl;
        string graphfile = picked_folder + ss.str();
        string imagefile = picked_folder + "test.png";
        ge.generate_dot_file(graphfile);

        ss.str("");
        ss << "/indices" << setfill('0') << setw(6) << counter << ".txt";
        string indexfile = picked_folder + ss.str();
        ge.generate_index_file(indexfile);

        ss.str("");
        ss << "/super_indices" << setfill('0') << setw(6) << counter << ".txt";
        string super_indexfile = picked_folder + ss.str();
        generate_index_file(super_indexfile, extracted, *indices);

        ss.str("");
        ss << "/cloud" << setfill('0') << setw(6) << counter << ".pcd";
        string cloudfile = picked_folder + ss.str();
        pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr colored_cloud = extractor.get_cloud();
        pcl::io::savePCDFileBinary(cloudfile, *colored_cloud);

        pcl::PointCloud<pcl::PointXYZRGB> pub_cloud;
        pcl::copyPointCloud(*colored_cloud, pub_cloud);
        pub_cloud.header.frame_id = "/map";
        cloud_pub.publish(pub_cloud);

        if (VISUALIZE) {
            string command = "dot -Tpng " + graphfile + " > " + imagefile + " && gvfs-open " + imagefile;
            system(command.c_str());

            auto tcloud = extractor.get_cloud();
            auto tnormals = extractor.get_normals();

            boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
            viewer->setBackgroundColor(0, 0, 0);
            viewer->initCameraParameters();
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(tcloud, 0, 255, 0);
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(tcloud);
            viewer->addPointCloud<pcl::PointXYZRGB>(tcloud, rgb, "cloud");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                     3, "cloud");
            viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(tcloud, tnormals, 20, 1e-2f, "normals");
            while (!viewer->wasStopped())
            {
                viewer->spinOnce(100);
                boost::this_thread::sleep(boost::posix_time::microseconds(100000));
            }
        }

        cout << "The algorithm has finished..." << endl;
        ++counter;
    }

    return 0;
}
