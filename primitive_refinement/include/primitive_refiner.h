#ifndef PRIMITIVE_REFINER_H
#define PRIMITIVE_REFINER_H

#include "primitive_extractor.h"
#include "plane_primitive.h"

template <typename Point>
class primitive_refiner : public primitive_extractor<Point> {
public:
    typedef primitive_extractor<Point> super;
    typedef typename super::cloud_type cloud_type;
    typedef typename super::cloud_ptr cloud_ptr;
    typedef typename super::point_type point_type;
private:
    double angle_threshold;
    double distance_threshold;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > cameras;
    //Eigen::Vector3f camera;
    std::vector<size_t> dividers;

    bool are_coplanar(base_primitive* p, base_primitive* q);
    static bool contained_in_hull(const Vector3d& point, const std::vector<Vector3d, aligned_allocator<Vector3d> >& hull, const Vector3d& c);
    plane_primitive* try_merge(base_primitive* p, base_primitive* q,
                               cloud_ptr& cloud, float res,
                               std::vector<int>& p_camera_ids, std::vector<int>& q_camera_ids);
    void rectify_normals(std::vector<base_primitive*>& extracted);
    void merge_coplanar_planes(std::vector<base_primitive*>& extracted, std::vector<std::vector<int> >& camera_ids);
    void rectify_normals();
    void compute_camera_ids(std::vector<base_primitive*>& extracted, std::vector<std::vector<int> >& camera_ids);
    void remove_floor_planes(std::vector<base_primitive*>& extracted);
public:
    static cloud_ptr fuse_clouds(std::vector<cloud_ptr>& clouds);
    static void create_dividers(std::vector<size_t>& dividers, std::vector<cloud_ptr>& clouds);
    void extract(std::vector<base_primitive*>& extracted, std::vector<std::vector<int> >& camera_ids);
    primitive_refiner(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& cameras,
                      std::vector<cloud_ptr>& clouds, std::vector<base_primitive*>& primitives,
                      primitive_params params = primitive_params(),
                      primitive_visualizer<point_type>* vis = NULL) :
        super(fuse_clouds(clouds), primitives, params, vis), cameras(cameras)
    {
        create_dividers(dividers, clouds);
        std::cout << "Cloud size: " << super::cloud->size() << std::endl;
        angle_threshold = 0.6;
        distance_threshold = 0.25;
        //camera = cameras[0].cast<float>(); // LEGACY
        rectify_normals();
    }
    primitive_refiner(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& cameras,
                      std::vector<size_t>& dividers,
                      cloud_ptr& cloud, pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals,
                      std::vector<base_primitive*>& primitives,
                      primitive_params params = primitive_params(),
                      primitive_visualizer<point_type>* vis = NULL) :
        super(cloud, cloud_normals, primitives, params, vis), cameras(cameras), dividers(dividers)
    {
        //create_dividers(dividers, clouds);
        std::cout << "Cloud size: " << super::cloud->size() << std::endl;
        angle_threshold = 0.6;
        distance_threshold = 0.25;
        //camera = cameras[0].cast<float>(); // LEGACY
        //rectify_normals();
    }
};

#include "primitive_refiner.hpp"
#endif // PRIMITIVE_REFINER_H
