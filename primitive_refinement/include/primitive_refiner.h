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
    Eigen::Vector3f camera;
    std::vector<size_t> dividers;

    bool are_coplanar(base_primitive* p, base_primitive* q);
    static bool contained_in_hull(const Vector3d& point, const std::vector<Vector3d, aligned_allocator<Vector3d> >& hull, const Vector3d& c);
    plane_primitive* try_merge(base_primitive* p, base_primitive* q,
                               cloud_ptr& cloud, float res);
    void rectify_normals(std::vector<base_primitive*>& extracted);
    void merge_coplanar_planes(std::vector<base_primitive*>& extracted);
    static cloud_ptr fuse_clouds(std::vector<cloud_ptr>& clouds);
    static void create_dividers(std::vector<size_t>& dividers, std::vector<cloud_ptr>& clouds);
public:
    void extract(std::vector<base_primitive*>& extracted);
    primitive_refiner(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& cameras,
                      std::vector<cloud_ptr>& clouds, std::vector<base_primitive*>& primitives,
                      primitive_params params = primitive_params(),
                      primitive_visualizer<point_type>* vis = NULL) :
        super(fuse_clouds(clouds), primitives, params, vis), cameras(cameras)
    {
        create_dividers(dividers, clouds);
        std::cout << "Cloud size: " << super::cloud->size() << std::endl;
        angle_threshold = 0.2;
        distance_threshold = 0.3;
        camera = cameras[0].cast<float>(); // LEGACY
    }
};

#include "primitive_refiner.hpp"
#endif // PRIMITIVE_REFINER_H
