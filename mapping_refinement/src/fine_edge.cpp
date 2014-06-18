#include "fine_edge.h"

fine_edge::fine_edge(fine_vertex& f1, fine_vertex& f2) : fine_registration(f1, f2)
{
    vertices()[0] = &f1;
    vertices()[1] = &f2;
}

void fine_edge::computeError()
{
    _error(0) = last_error;
}

bool fine_edge::setMeasurementFromState()
{
    return true;
}

void fine_edge::linearizeOplus()
{
    //fine_vertex* from = static_cast<fine_vertex*>(_vertices[0]);
    //fine_vertex* to   = static_cast<fine_vertex*>(_vertices[1]);
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    step(R, t);
    //Eigen::Matrix3d T;
    //T.setIdentity();
    //T.topLeftCorner<3, 3>() = R.cast<double>();
    //T.block<3, 1>(0, 3) = t.cast<double>();
    //Eigen::Isometry3d Ti(T);
    Eigen::Quaterniond q(R.cast<double>());
    _jacobianOplusXi.head<3>() = -t.cast<double>();
    _jacobianOplusXj.head<3>() = -(-R.transpose()*t).cast<double>();
    _jacobianOplusXi.tail<3>() = -q.vec();
    q = q.inverse();
    _jacobianOplusXj.tail<3>() = -q.vec();

}

bool fine_edge::read(std::istream& is) {
    return true;
}

bool fine_edge::write(std::ostream& os) const {
    return os.good();
}
