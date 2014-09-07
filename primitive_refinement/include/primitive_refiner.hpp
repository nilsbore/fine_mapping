#include "primitive_refiner.h"

#include <algorithm>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace Eigen;

template <typename Point>
typename primitive_refiner<Point>::cloud_ptr primitive_refiner<Point>::fuse_clouds(std::vector<cloud_ptr>& clouds)
{
    cloud_ptr rtn(new cloud_type);
    size_t divider = 0;
    for (const cloud_ptr& cloud : clouds) {
        rtn->points.insert(rtn->points.end(), cloud->points.begin(), cloud->points.end());
        divider += cloud->points.size();
    }
    // dividers hasn't been initialized!
    return rtn;
}

template <typename Point>
void primitive_refiner<Point>::create_dividers(std::vector<size_t>& dividers, std::vector<cloud_ptr>& clouds)
{
    size_t divider = 0;
    for (const cloud_ptr& cloud : clouds) {
        divider += cloud->points.size();
        dividers.push_back(divider);
    }
}



template <typename Point>
bool primitive_refiner<Point>::are_coplanar(base_primitive* p, base_primitive* q)
{
    VectorXd par_p, par_q;
    Vector3d v_p, v_q;
    Vector3d c_p, c_q;
    if (p == q || p->get_shape() != base_primitive::PLANE ||
            q->get_shape() != base_primitive::PLANE) {
        return false;
    }
    p->direction_and_center(v_p, c_p);
    q->direction_and_center(v_q, c_q);
    p->shape_data(par_p);
    q->shape_data(par_q);
    Vector3d diff = c_p - c_q;
    diff.normalize();
    bool same_direction = acos(fabs(v_p.dot(v_q))) < angle_threshold;
    double dist_p = fabs(c_p.dot(par_q.segment<3>(0)) + par_q(3));
    double dist_q = fabs(c_q.dot(par_p.segment<3>(0)) + par_p(3));
    bool same_depth = dist_p < distance_threshold && dist_q < distance_threshold;
    return same_direction && same_depth;
}

template <typename Point>
bool primitive_refiner<Point>::contained_in_hull(const Vector3d& point, const std::vector<Vector3d, aligned_allocator<Vector3d> >& hull, const Vector3d& c)
{
    size_t n = hull.size();
    Vector2d d = (hull[1] - hull[0]).tail<2>();
    Vector2d v(-d(1), d(0));
    Vector2d point2 = point.tail<2>();
    double sign = v.dot((c  - hull[0]).tail<2>());
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        Vector2d p = hull[i].tail<2>();
        d = hull[j].tail<2>() - p;
        v = Vector2d(-d(1), d(0));
        if (sign*v.dot(point2  - p) < 0) {
            return false;
        }
    }
    return true;
}

template <typename Point>
plane_primitive* primitive_refiner<Point>::try_merge(base_primitive* p, base_primitive* q,
                                                     cloud_ptr& cloud, float res)
{
    MatrixXd points_p, points_q;
    super::primitive_inlier_points(points_p, p);
    super::primitive_inlier_points(points_q, q);

    // get convex hull of p and q: P, Q
    std::vector<Vector3d, aligned_allocator<Vector3d> > hull_p;
    std::vector<Vector3d, aligned_allocator<Vector3d> > hull_q;
    std::vector<Vector3d, aligned_allocator<Vector3d> > hull;

    plane_primitive* pp = static_cast<plane_primitive*>(p);
    plane_primitive* pq = static_cast<plane_primitive*>(q);
    plane_primitive* pr = new plane_primitive;
    pr->merge_planes(*pp, *pq);

    p->shape_points(hull_p);
    q->shape_points(hull_q);
    pr->shape_points(hull);

    // project the points into a common coordinate system, maybe the camera plane?
    Vector3d v, c;
    pr->direction_and_center(v, c); // should really be the rotation instead
    if (v.dot(c - camera.cast<double>()) > 0) {
        v = -v;
    }
    float d = -v.dot(c);

    // just pick the basis of the first one
    VectorXd data;
    pr->shape_data(data);
    Quaterniond q_r(data(12), data(9), data(10), data(11));
    Matrix3d R(q_r);

    for (Vector3d& point : hull) {
        point = R.transpose()*point;
    }

    // lambdas don't need to be stored, they are just declarations, function pointer is enough
    auto first_comparator = [](const Vector3d& p1, const Vector3d& p2) { return p1(1) < p2(1); };
    auto second_comparator = [](const Vector3d& p1, const Vector3d& p2) { return p1(2) < p2(2); };
    Vector3d xmin = *std::min_element(hull.begin(), hull.end(), first_comparator);
    Vector3d xmax = *std::max_element(hull.begin(), hull.end(), first_comparator);
    Vector3d ymin = *std::min_element(hull.begin(), hull.end(), second_comparator);
    Vector3d ymax = *std::max_element(hull.begin(), hull.end(), second_comparator);

    double width = xmax(1) - xmin(1);
    double height = ymax(2) - ymin(2);

    int w = int(width/res);
    int h = int(height/res);

    cv::Mat im = cv::Mat::zeros(h, w, CV_32SC1);
    // project the entire pointcloud? sounds expensive...

    Vector3f point;
    Vector3f dir;
    Matrix3f Rf = R.cast<float>().transpose();
    Vector3f vf = v.cast<float>();
    Vector3f minf(0, xmin(1), ymin(2));
    for (const point_type& pp : cloud->points) {
        if (isinf(pp.z) || isnan(pp.z)) {
            continue;
        }
        point = pp.getVector3fMap();
        dir = point - camera;
        if (dir.dot(vf) > 0) { // correct
            continue;
        }
        dir.normalize();
        float prod = dir.dot(vf);
        if (fabs(prod) < 1e-5) {
            continue;
        }
        float a = -(d + vf.dot(point))/prod;
        point += a*dir;
        point = Rf*point; // check if inside convex hull
        if (!contained_in_hull(point.cast<double>(), hull, c)) {
            continue;
        }
        point -= minf;
        int y = int(point(2)/res);
        int x = int(point(1)/res);
        if (y < 0 || y >= h || x < 0 || x >= w) {
            continue;
        }
        if (a > 0) { // point is on the camera side of the plane
            im.at<int>(y, x) = 1;
        }
    }

    int x_p = -1;
    int y_p = -1;
    for (size_t i = 0; i < points_p.cols(); ++i) {
        point = Rf*points_p.col(i).cast<float>() - minf;
        int y = int(point(2)/res);
        int x = int(point(1)/res);
        if (y < 0 || y >= h || x < 0 || x >= w) {
            continue;
        }
        im.at<int>(y, x) = 1;
        if (x_p == -1) {
            x_p = x;
            y_p = y;
        }
    }

    int x_q = -1;
    int y_q = -1;
    for (size_t i = 0; i < points_q.cols(); ++i) {
        point = Rf*points_q.col(i).cast<float>() - minf;
        int y = int(point(2)/res);
        int x = int(point(1)/res);
        if (y < 0 || y >= h || x < 0 || x >= w) {
            continue;
        }
        im.at<int>(y, x) = 1;
        if (x_q == -1) {
            x_q = x;
            y_q = y;
        }
    }
    if (x_p == -1 || x_q == -1) {
        return false;
    }

    /*cv::Mat imcopy = im.clone();
    imcopy = 65535*imcopy;
    cv::imshow("im", imcopy);
    cv::waitKey(0);*/

    int largest = base_primitive::find_blobs(im, false, false);
    /*cv::Mat result = cv::Mat::zeros(h, w, CV_32SC1);
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            if (im.at<int>(i, j) == largest) {
                result.at<int>(i, j) = 65535;
            }
        }
    }

    cv::imshow("result", result);
    cv::waitKey(0);*/

    u_char c1 = im.at<int>(y_p, x_p);
    u_char c2 = im.at<int>(y_q, x_q);
    if (c1 == c2) {
        return pr;
    }
    else {
        delete pr;
        return NULL;
    }
}

template <typename Point>
void primitive_refiner<Point>::rectify_normals(std::vector<base_primitive*>& extracted)
{
    Vector3d v, c;
    for (base_primitive* p : extracted) {
        if (p->get_shape() != base_primitive::PLANE) {
            continue;
        }
        p->direction_and_center(v, c);
        // if all cameras are on the same side of the plane, continue
        double base_case = (cameras[0] - c).dot(v);
        bool same = true;
        for (const Vector3d& cc : cameras) {
            if (base_case*(cc - c).dot(v) < 0) { // bad
                same = false;
                break;
            }
        }
        if (same) {
            if (base_case < 0) {
                // switch direction
                plane_primitive* planep = static_cast<plane_primitive*>(p);
                planep->switch_direction();
                continue;
            }
            else {
                continue;
            }
        }
        //super.primitive_inlier_points();
        MatrixXd points;
        super::primitive_inlier_points(points, p);
        size_t place = 0;
        std::vector<size_t> counts(dividers.size());
        for (int inlier : p->supporting_inds) { // these should always be sorted after extraction
            if (inlier < dividers[place]) {
                counts[place]++;
            }
            else {
                ++place;
            }
        }
        std::vector<size_t>::iterator result = std::max_element(counts.begin(), counts.end());
        size_t which_camera = std::distance(counts.begin(), result);
        if ((cameras[which_camera] - c).dot(v) < 0) { // bad
            // switch direction
            plane_primitive* planep = static_cast<plane_primitive*>(p);
            planep->switch_direction();
        }
    }
}

template <typename Point>
void primitive_refiner<Point>::merge_coplanar_planes(std::vector<base_primitive*>& extracted)
{
    while (true) {
        bool do_break = false;
        for (size_t i = 0; i < extracted.size(); ++i) {
            base_primitive* p = extracted[i];
            for (size_t j = 0; j < i; ++j) {
                base_primitive* q = extracted[j];
                if (!are_coplanar(p, q)) {
                    continue;
                }
                std::cout << "The primitives are co-planar" << std::endl;
                plane_primitive* pp = try_merge(extracted[i], extracted[j], super::cloud, 1.2*super::params.connectedness_res);
                if (pp != NULL) { // once found, redo the whole scheme for now
                    std::cout << "And they may also be connected" << std::endl;
                    extracted.erase(std::remove_if(extracted.begin(), extracted.end(), [=](const base_primitive* b) { return b == p || b == q; } ), extracted.end());
                    delete p;
                    delete q;
                    extracted.push_back(pp);
                    do_break = true;
                    break;
                }
            }
            if (do_break) {
                break;
            }
        }
        if (!do_break) {
            break;
        }
    }
}

template <typename Point>
void primitive_refiner<Point>::extract(std::vector<base_primitive*>& extracted)
{
    super::extract(extracted);
    rectify_normals(extracted);
    merge_coplanar_planes(extracted);
}
