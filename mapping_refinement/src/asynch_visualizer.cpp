#include "asynch_visualizer.h"

#include <pcl/visualization/pcl_visualizer.h>

void asynch_visualizer::run_visualizer()
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer>
            viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Starting visualizer
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    // Wait until visualizer window is closed.
    while (!viewer->wasStopped())
    {
        lock();
        if (cloud1_changed) {
            viewer->removePointCloud("cloud1");
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud1);
            viewer->addPointCloud<pcl::PointXYZRGB>(cloud1, rgb, "cloud1");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                     1, "cloud1");
            cloud1_changed = false;
        }
        if (cloud2_changed) {
            viewer->removePointCloud("cloud2");
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud2);
            viewer->addPointCloud<pcl::PointXYZRGB>(cloud2, rgb, "cloud2");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                     1, "cloud2");
            cloud2_changed = false;
        }
        viewer->spinOnce(100);
        unlock();
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
    viewer->close();
}

void asynch_visualizer::lock()
{
    pthread_mutex_lock(&mutex);
}

void asynch_visualizer::unlock()
{
    pthread_mutex_unlock(&mutex);
}

void* viewer_thread(void* ptr)
{
    ((asynch_visualizer*)ptr)->run_visualizer();
    pthread_exit(NULL);
}

void asynch_visualizer::create_thread()
{
    pthread_create(&my_viewer_thread, NULL, viewer_thread, this);
}

void asynch_visualizer::join_thread()
{
    pthread_join(my_viewer_thread, NULL);
}
