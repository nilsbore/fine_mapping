#include "graph_extractor.h"

#include <boost/graph/graphviz.hpp>
#include <fstream>

using namespace Eigen;

graph_extractor::graph_extractor(const std::vector<base_primitive*>& primitives, std::vector<MatrixXd>& inliers,
                                 std::vector<std::vector<int> >& cameras, double adjacency_dist, double inlier_distance) :
    primitives(primitives), cameras(cameras), adjacency_dist(adjacency_dist), inlier_distance(inlier_distance)
{
    connectedness_dist = 8.0; // assign very large if you want all-to-all connectedness
    std::vector<MatrixXd> sparse_inliers;
    sparse_inliers.resize(inliers.size());
    int skip = 5;//5; // for 3x3
    for (int i = 0; i < inliers.size(); ++i) {
        int newsize = int(inliers[i].cols()) / skip;
        sparse_inliers[i].resize(3, newsize);
        for (int j = 0; j < newsize; ++j) {
            sparse_inliers[i].col(j) = inliers[i].col(skip*j);
        }
    }
    construct_adjacency_graph(sparse_inliers);
}

bool graph_extractor::do_share_camera(int i, int j)
{
    int top_dist_threshold = 1;
    for (const int& iind : cameras[i]) {
        for (const int& jind : cameras[j]) {
            if (abs(iind - jind) <= top_dist_threshold) {
                return true;
            }
        }
    }
    return false;

    /*std::vector<int> intersection;

    // we could do this for all at the beginning instead
    std::sort(cameras[i].begin(), cameras[i].end());
    std::sort(cameras[j].begin(), cameras[j].end());

    std::set_intersection(cameras[i].begin(),cameras[i].end(),cameras[j].begin(),cameras[j].end(),std::back_inserter(intersection));

    return !intersection.empty();*/
}

void graph_extractor::construct_adjacency_graph(std::vector<MatrixXd>& inliers)
{
    std::vector<graph::vertex_descriptor> v;
    v.resize(primitives.size() + 1);
    for (int i = 0; i < primitives.size(); ++i) {
        v[i] = boost::add_vertex(g);
    }
    v[primitives.size()] = boost::add_vertex(g);
    double mindist;
    Vector3d d1;
    Vector3d c1;
    Vector3d d2;
    Vector3d c2;
    Vector3d dn(0.0, 0.0, 1.0);
    std::vector<Vector3d, aligned_allocator<Vector3d> > trans;
    for (int i = 0; i < inliers.size(); ++i) {
        primitives[i]->direction_and_center(d1, c1);
        mindist = floor_distance(inliers[i], primitives[i]->get_shape() == base_primitive::PLANE);
        primitive_edge p;
        p.dist = mindist;
        std::cout << "Mindist before comparison: " << mindist << std::endl;
        std::cout << "Adjacency dist: " << adjacency_dist << std::endl;
        if (mindist < adjacency_dist) {
            p.type = 0;
            if (primitives[i]->get_shape() == base_primitive::SPHERE) {
                p.angle = 0;
            }
            else if (primitives[i]->get_shape() == base_primitive::PLANE) {
                p.angle = acos(d1.dot(dn));
            }
            else {
                p.angle = acos(fabs(d1.dot(dn)));
            }
            edge_weight_property e = p;
            std::pair<boost::graph_traits<graph>::edge_descriptor, bool> result = boost::add_edge(v[i], v[primitives.size()], e, g);
        }
        else if (mindist < connectedness_dist) {
            p.type = 1;
            if (primitives[i]->get_shape() == base_primitive::SPHERE) {
                p.angle = 0;
            }
            //p.angle = acos(fabs(d1.dot(dn))); // no point in doing plane angle here because the one can can be in the middle of the other
            p.angle = acos(d1.dot(dn));
            edge_weight_property e = p;
            std::pair<boost::graph_traits<graph>::edge_descriptor, bool> result = boost::add_edge(v[i], v[primitives.size()], e, g);
        }

        for (int j = 0; j < i; ++j) {
            if (!do_share_camera(i, j)) { //TRY
                continue;
            }
            trans.clear();
            bool are_planes = primitives[i]->get_shape() == base_primitive::PLANE && primitives[j]->get_shape() == base_primitive::PLANE;
            bool is_sphere = primitives[i]->get_shape() == base_primitive::SPHERE || primitives[j]->get_shape() == base_primitive::SPHERE;
            primitives[j]->direction_and_center(d2, c2);
            double tangle = acos(d1.dot(d2));
            bool compute_trans = tangle < 3.0*M_PI/4.0 && fabs(tangle) > M_PI/4.0;
            mindist = primitive_distance(trans, inliers[i], inliers[j], are_planes && compute_trans);
            std::cout << "Min dist for " << i << " and " << j << " is " << mindist << std::endl;
            primitive_edge p;
            p.dist = mindist;
            if (mindist < adjacency_dist) {
                p.type = 0;
                if (is_sphere) {
                    p.angle = 0;
                }
                else if (are_planes) {
                    double angle = acos(d1.dot(d2));
                    if (angle > 3.0*M_PI/4.0) {
                        p.type = 1;
                        p.angle = angle; // TRY
                    }
                    else if (fabs(angle) < M_PI/4.0) {
                        p.angle = angle;
                    }
                    else {
                        p.angle = plane_angle(trans, d1, d2);
                    }
                }
                else {
                    p.angle = acos(fabs(d1.dot(d2)));
                }
            }
            else if (mindist < connectedness_dist) {
                p.type = 1;
                if (is_sphere) {
                    p.angle = 0;
                }
                else if (are_planes) {
                    p.angle = acos(d1.dot(d2));
                }
                else {
                    p.angle = acos(fabs(d1.dot(d2))); // no point in doing plane angle here because the one can can be in the middle of the other
                }
            }
            else {
                continue;
            }
            edge_weight_property e = p;
            std::pair<boost::graph_traits<graph>::edge_descriptor, bool> result = boost::add_edge(v[i], v[j], e, g);
        }
    }

}

double graph_extractor::plane_angle(const std::vector<Vector3d, aligned_allocator<Vector3d> >& trans, const Vector3d& d1, const Vector3d& d2)
{
    int s = 0;
    for (const Vector3d& t : trans) {
        if ((d1 - d2).dot(t) > 0) { // planes facing each other (think normal geometry)
            s += 1;
        }
        else { // outwarding facing edge, same reasoning
            s -= 1;
        }
    }
    std::cout << "Sum: " << s << std::endl;
    std::cout << "Total: " << trans.size() << std::endl;
    double angle = M_PI;
    //if (s > 0) {
    if (float(s)/float(trans.size()) > -0.15) { // 0.3
        angle -= acos(d1.dot(d2));
    }
    else {
        angle += acos(d1.dot(d2));
    }
    return angle;
}

double graph_extractor::primitive_distance(std::vector<Vector3d, aligned_allocator<Vector3d> >& trans,
                                           const MatrixXd& inliers1, const MatrixXd& inliers2, bool are_planes)
{
    double mindist = INFINITY;
    MatrixXd temp;
    double mincol;
    int index;
    for (int i = 0; i < inliers1.cols(); ++i) {
        temp = inliers1.col(i).replicate(1, inliers2.cols());
        temp -= inliers2;
        mincol = temp.colwise().squaredNorm().minCoeff(&index);
        if (sqrt(mincol) < adjacency_dist) {
            if (are_planes) {
                trans.push_back(inliers2.col(index) - inliers1.col(i));
            }
            else {
                return sqrt(mincol); // SPEEDUP
            }
        }
        if (sqrt(mincol) < mindist) {
            mindist = sqrt(mincol);
        }
    }
    return mindist;
}

double graph_extractor::floor_distance(const MatrixXd& inliers, bool is_plane)
{
    double mindist = INFINITY;
    Vector3d temp;
    double mincol;
    for (int i = 0; i < inliers.cols(); ++i) {
        temp = inliers.col(i);
        mincol = fabs(temp(2)) - inlier_distance;
        mincol = mincol < 0? 0.0 : mincol;
        if (mincol < adjacency_dist) {
            return mincol; // SPEEDUP
        }
        if (mincol < mindist) {
            mindist = mincol;
        }
    }
    std::cout << "Mindist: " << mindist << std::endl;
    return mindist;
}

void graph_extractor::generate_dot_file(const std::string& filename)
{
    std::ofstream file;
    file.open(filename);
    primitive_label_writer writer(primitives, cameras);
    primitive_edge_writer<graph> edge_writer(g);
    boost::write_graphviz(file, g, writer, edge_writer);//boost::make_label_writer(name)); // dot -Tpng test2.dot > test2.png
    file.close();
}

void graph_extractor::generate_index_file(const std::string& filename)
{
    std::ofstream file;
    file.open(filename);
    for (base_primitive* p : primitives) {
        p->write_indices_to_stream(file);
        file << "\n";
    }
    file.close();
}
