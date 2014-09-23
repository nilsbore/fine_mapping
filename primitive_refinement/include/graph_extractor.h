#ifndef GRAPH_EXTRACTOR_H
#define GRAPH_EXTRACTOR_H

#include <Eigen/Dense>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include "base_primitive.h"
#include "plane_primitive.h"
#include "conversions.h"

class primitive_label_writer {
public:
    primitive_label_writer(std::vector<base_primitive*>& primitives, std::vector<std::vector<int> >& camera_ids) :
        primitives(primitives), camera_ids(camera_ids) {}
    template <class VertexOrEdge>
    void operator()(std::ostream& out, const VertexOrEdge& v) const {
        std::string name;
        std::string shape;
        std::vector<int> cameras;// = camera_ids[v];
        base_primitive* p;// = primitives[v];
        plane_primitive pp;
        if (v == primitives.size()) {
            std::set<int> unique_cameras;
            for (std::vector<int>& ids : camera_ids) {
                unique_cameras.insert(ids.begin(), ids.end());
            }
            cameras.insert(cameras.end(), unique_cameras.begin(), unique_cameras.end());
            p = &pp;
        }
        else {
            cameras = camera_ids[v];
            p = primitives[v];
        }
        std::cout << "Shape: " << p->get_shape() << std::endl;
        switch (p->get_shape()) {
        case base_primitive::SPHERE:
            name = "Sphere";
            shape = "ellipse";
            break;
        case base_primitive::PLANE:
            name = "Plane";
            shape = "box";
            break;
        case base_primitive::CYLINDER:
            name = "Cylinder";
            shape = "ellipse";
            break;
        default:
            std::cout << "no match!!" << std::endl;
            break;
        }
        if (v == primitives.size()) {
            shape = "polygon";
        }
        out << "[label=\"" << name << "\"]";
        out << "[shape=\"" << shape << "\"]";
        out << "[style=\"filled\"]";
        if (v == primitives.size()) {
            out << "[fillcolor=\"" << convenience::rgb_to_hex_string(255, 255, 255) << "\"]";
        }
        else {
            out << "[fillcolor=\"" << convenience::rgb_to_hex_string(p->red, p->green, p->blue) << "\"]";
        }
        out << "[cameras=\"";
        for (size_t i = 0; i < cameras.size(); ++i) {
            out << cameras[i];
            if (i != cameras.size() - 1) {
                out << " "; // easier to parse list without comma for matlab
            }
        }
        out << "\"]";
        if (v == primitives.size()) {
            out << "[shapesize=\"" << 0.0 << "\"]";
            out << "[shapedata=\"floor\"]";
        }
        else {
            out << "[shapesize=\"" << p->shape_size() << "\"]";
            out << "[shapedata=\"";
            Eigen::VectorXd data;
            p->shape_data(data);
            for (int i = 0; i < data.rows(); ++i) {
                if (i != 0) {
                    out << " ";
                }
                out << data(i);
            }
            out << "\"]";
        }
    }
private:
    std::vector<base_primitive*>& primitives;
    std::vector<std::vector<int> >& camera_ids;
};

struct primitive_edge
{
    int type;
    double angle;
    double dist;
};

template <class Graph>
class primitive_edge_writer {
public:
    //typedef boost::property<boost::edge_weight_t, double> edge_weight_property;
    primitive_edge_writer(Graph& g) : g(g) {}
    //template <class VertexOrEdge>
    void operator()(std::ostream& out, const typename boost::graph_traits<Graph>::edge_descriptor& e) const {
        primitive_edge v = boost::get(boost::edge_weight_t(), g, e);
        out << "[label=\"" << v.angle << "\"]";
        out << "[shapedist=\"" << v.dist << "\"]";
        if (v.type == 1) {
            out << "[style=\"dashed\"]";
        }
    }
private:
    Graph& g;
};

class graph_extractor
{
public:
    typedef boost::property<boost::edge_weight_t, primitive_edge> edge_weight_property;
    typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS,
            boost::property<boost::vertex_color_t, boost::default_color_type>,
            edge_weight_property
            > graph;
protected:
    graph g;
    double adjacency_dist;
    double connectedness_dist;
    double inlier_distance;
    std::vector<base_primitive*> primitives;
    std::vector<std::vector<int> > cameras;
    void construct_adjacency_graph(std::vector<Eigen::MatrixXd>& inliers);
    double primitive_distance(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& trans,
                              const Eigen::MatrixXd& inliers1, const Eigen::MatrixXd& inliers2, bool are_planes);
    double floor_distance(const Eigen::MatrixXd& inliers, bool is_plane);
    double plane_angle(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& trans, const Eigen::Vector3d& d1, const Eigen::Vector3d& d2);
    bool do_share_camera(int i, int j);
public:
    void generate_dot_file(const std::string& filename);
    void generate_index_file(const std::string& filename);
    graph_extractor(const std::vector<base_primitive*>& primitives, std::vector<Eigen::MatrixXd>& inliers, std::vector<std::vector<int> >& cameras, double adjacency_dist, double inlier_distance = 0.15);
};

#endif // GRAPH_EXTRACTOR_H
