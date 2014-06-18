#include "fine_vertex.h"

//fine_vertex::fine_vertex(const std::string& pcdname, const std::string& tname) : scan(pcdname, tname), g2o::BaseVertex<6, Eigen::Isometry3d >() {}
//fine_vertex::fine_vertex() : scan(), g2o::BaseVertex<6, Eigen::Isometry3d>() {}
//fine_vertex::fine_vertex(const fine_vertex& vertex) : scan(vertex), g2o::BaseVertex<6, Eigen::Isometry3d>(vertex) {}

bool fine_vertex::read(std::istream& is)
{
    return true;
}

bool fine_vertex::write(std::ostream& os) const
{
    return true;
}
