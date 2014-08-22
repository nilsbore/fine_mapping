#ifndef PRIMITIVE_REFINER_H
#define PRIMITIVE_REFINER_H

#include "primitive_extractor.h"
#include "plane_primitive.h"

template <typename Point>
class primitive_refiner : public primitive_extractor<Point> {
public:
    typedef primitive_extractor<Point> super;
    typedef typename super::cloud_ptr cloud_ptr;
    typedef typename super::point_type point_type;
private:
    double angle_threshold;
    double distance_threshold;

    bool are_coplanar(base_primitive* p, base_primitive* q);
    static bool contained_in_hull(const Vector3d& point, const std::vector<Vector3d, aligned_allocator<Vector3d> >& hull, const Vector3d& c);
    plane_primitive* try_merge(base_primitive* p, base_primitive* q,
                               cloud_ptr& cloud, float res);
public:
    void extract(std::vector<base_primitive*>& extracted);
    primitive_refiner(cloud_ptr cloud,
                      std::vector<base_primitive*>& primitives,
                      primitive_params params = primitive_params(),
                      primitive_visualizer<point_type>* vis = NULL) :
        super(cloud, primitives, params, vis)
    {
        angle_threshold = 0.2;
        distance_threshold = 0.3;
    }
};

#include "primitive_refiner.hpp"
#endif // PRIMITIVE_REFINER_H
