#ifndef ASYNCH_VISUALIZER_H
#define ASYNCH_VISUALIZER_H

#include <boost/thread/thread.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

class asynch_visualizer
{
private:
    pthread_mutex_t mutex;
    pthread_t my_viewer_thread;
public:
    bool cloud1_changed;
    bool cloud2_changed;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2;
    void create_thread();
    void join_thread();
    void lock();
    void unlock();
    void run_visualizer();
    asynch_visualizer()
    {
        cloud1_changed = false;
        cloud2_changed = false;
        if (pthread_mutex_init(&mutex, NULL) != 0) {
            std::cout << "mutex init failed" << std::endl;
        }
        //pthread_mutex_lock(&mutex);
    }
};

void* viewer_thread(void* ptr);

#endif // ASYNCH_VISUALIZER_H
