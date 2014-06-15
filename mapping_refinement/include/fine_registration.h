#ifndef FINE_REGISTRATION_H
#define FINE_REGISTRATION_H

#include <Eigen/Dense>
#include "scan.h"

class fine_registration {
protected:
    float last_error;
    scan& scan1;
    scan& scan2;
    // dense optical flow parameters
    double pyr_scale; // parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
    int levels; // number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
    int winsize; // averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
    int iterations; // number of iterations the algorithm does at each pyramid level.
    int poly_n; // size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
    double poly_sigma; // standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
    void compute_transform_jacobian(Eigen::MatrixXf &J, const Eigen::Vector3f& x) const;
    float compute_error(const cv::Mat& depth, const cv::Mat& wdepth, const cv::Mat& invalid) const;
    void compute_jacobian(Eigen::VectorXf& jacobian, const cv::Mat& depth, const cv::Mat& wdepth,
                          const cv::Mat& flow, const cv::Mat& invalid, Eigen::Vector3f& trans) const;
    void compute_transform(Eigen::Matrix3f &R, Eigen::Vector3f &t, const cv::Mat& depth, const cv::Mat& wdepth,
                           const cv::Mat& flow, const cv::Mat& invalid) const;
public:
    float error() const { return last_error; }
    void step(Eigen::Matrix3f& R, Eigen::Vector3f& t);
    fine_registration(scan& scan1, scan& scan2) : scan1(scan1), scan2(scan2)
    {
        pyr_scale = 0.5;
        levels = 3;
        winsize = 100; // 50
        iterations = 3;
        poly_n = 5;
        poly_sigma = 3.2;
    }
};
#endif // FINE_REGISTRATION_H
