#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/area_picking_event.h>
#include <pcl/filters/crop_box.h>
#include <boost/filesystem.hpp>

class AreaChooser {
public:
    typedef pcl::PointXYZRGB point_type;
    typedef pcl::PointCloud<point_type> cloud_type;
    typedef cloud_type::Ptr cloud_ptr;
    typedef std::pair<point_type, point_type> point_pair;

    AreaChooser() : vis_src_(new pcl::visualization::PCLVisualizer ("Class Viewer", true)) {
        //vis_src_.reset(new pcl::visualization::PCLVisualizer ("Class Viewer", true));
        first = true;
        vis_src_->registerPointPickingCallback(&AreaChooser::pp_callback, *this);
        vis_src_->registerKeyboardCallback(&AreaChooser::keyboard_callback, *this);
        //vis_src_->getInteractorStyle()->setKeyboardModifier(pcl::visualization::INTERACTOR_KB_MOD_CTRL);
    }

    void setInputCloud (cloud_ptr xyz)
    {
        xyz_ = xyz;
    }

    void simpleVis()
    {
        vis_src_->setBackgroundColor(0, 0, 0);
        vis_src_->initCameraParameters();
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(xyz_);
        vis_src_->addPointCloud<pcl::PointXYZRGB>(xyz_, rgb, "cloud");
        vis_src_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                 1, "cloud");
        // Visualizer routine
        /*pcl::visualization::PointCloudColorHandlerRGBField<point_type>::Ptr c;
        c.reset(new pcl::visualization::PointCloudColorHandlerRGBField<point_type> (xyz_));
        vis_src_->addPointCloud<point_type>(xyz_,*c, "Cloud");
        vis_src_->resetCameraViewpoint ("Cloud");*/
        /*while (!vis_src_->wasStopped())
        {
            vis_src_->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }*/
        vis_src_->spin();
    }

    std::vector<point_pair> get_min_maxes()
    {
        return min_maxes;
    }

protected:
    void pp_callback (const pcl::visualization::PointPickingEvent& event, void*)
    {
        if (event.getPointIndex() == -1) {
            return;
        }
        event.getPoint(picked_point.x, picked_point.y, picked_point.z);
        std::cout << "Temp point: " << picked_point.getVector3fMap().transpose() << std::endl;
    }

    void keyboard_callback(const pcl::visualization::KeyboardEvent &event, void*)
    {
        if (event.getKeySym () == "k" && event.keyDown ())
        {
            std::cout << "Picked point: " << picked_point.getVector3fMap().transpose() << std::endl;
            if (first) {
                min_maxes.push_back(point_pair());
                min_maxes.back().first = picked_point;
            }
            else {
                min_maxes.back().second = picked_point;
            }
            first = !first;
            //vis_src_->removeShape("PickedPoint"); // Delete previous picked point
            //vis_src_->addSphere(picked_point, 0.001, 1.0, 0.0, 1.0, "PickedPoint", 0); // Visualize the latest picked point
        }
    }

private:
    bool first;
    point_type picked_point;
    std::vector<point_pair> min_maxes;

    // Point cloud data
    cloud_ptr xyz_;

    // The visualizer
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_src_;
};

bool create_folder(const std::string& folder)
{
    boost::filesystem::path dir(folder);
    if (!boost::filesystem::exists(dir) && !boost::filesystem::create_directory(dir))  {
            std::cout << "Failed to create directory " << folder << "..." << std::endl;
            return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Please provide a pointcloud to save patches from..." << std::endl;
    }

    typedef pcl::PointXYZRGB point_type;
    typedef pcl::PointCloud<point_type> cloud_type;
    typedef cloud_type::Ptr cloud_ptr;
    typedef std::pair<point_type, point_type> point_pair;

    cloud_ptr cloud(new cloud_type);
    std::string filename(argv[1]);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (filename, *cloud) == -1)
    {
        cout << "Couldn't read file " << filename << endl;
        return 0;
    }

    AreaChooser myView;
    myView.setInputCloud(cloud); // A pointer to a cloud
    myView.simpleVis();

    std::vector<point_pair> min_maxes = myView.get_min_maxes();

    boost::filesystem::path p(filename);
    std::string folder = p.parent_path().string();
    std::cout << "Folder: " << folder << std::endl;
    std::string pickedfolder = folder + "/picked";
    create_folder(pickedfolder);
    size_t counter = 0;
    for (point_pair pair : min_maxes) {
        Eigen::Vector4f min_point = pair.first.getVector4fMap();
        Eigen::Vector4f max_point = pair.second.getVector4fMap();
        if (min_point(0) > max_point(0)) {
            float temp = min_point(0);
            min_point(0) = max_point(0);
            max_point(0) = temp;
        }
        if (min_point(1) > max_point(1)) {
            float temp = min_point(1);
            min_point(1) = max_point(1);
            max_point(1) = temp;
        }
        min_point(2) = -1.0;
        max_point(2) = 4.0;
        std::cout << "First: " << pair.first << ", Second: " << pair.second << std::endl;
        std::stringstream ss;
        ss << "/picked_cloud" << setfill('0') << setw(6) << counter << ".pcd";
        std::string pickedname = pickedfolder + ss.str();
        pcl::CropBox<pcl::PointXYZRGB> crop_filter;
        crop_filter.setInputCloud(cloud);
        crop_filter.setMin(min_point);
        crop_filter.setMax(max_point);
        cloud_ptr picked_cloud(new cloud_type);
        crop_filter.filter(*picked_cloud);
        pcl::io::savePCDFileBinary(pickedname, *picked_cloud);
        ++counter;
    }

    return 0;
}


