#ifndef FINE_REGISTRATION_H
#define FINE_REGISTRATION_H

#include <Eigen/Dense>
#include <opencv2/gpu/gpu.hpp>
#include "scan.h"

//#define WITH_GPU

class fine_registration {
protected:
    float last_error;
    size_t iteration;
    scan& scan1;
    scan& scan2;
    std::vector<float> scales;
    std::vector<cv::Mat> rgbs1;
    std::vector<cv::Mat> rgbs2;
    std::vector<cv::Mat> depths1;
    std::vector<cv::Mat> depths2;
#ifdef WITH_GPU
    cv::gpu::GpuMat gpugray1;
    cv::gpu::GpuMat gpugray2;
#endif
    // dense optical flow parameters
    double pyr_scale; // parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
    int levels; // number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
    int winsize; // averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
    int iterations; // number of iterations the algorithm does at each pyramid level.
    int poly_n; // size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
    double poly_sigma; // standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
    float compute_error(const cv::Mat& depth, const cv::Mat& wdepth, const cv::Mat& invalid) const;
    void compute_transform(Eigen::Matrix3f &R, Eigen::Vector3f &t, const cv::Mat& depth, const cv::Mat& wdepth,
                           const cv::Mat& flow, const cv::Mat& invalid, float scale) const;
    static void get_transformation_from_correlation(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& cloud_src_demean,
                                                    const Eigen::Vector3f& centroid_src,
                                                    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& cloud_tgt_demean,
                                                    const Eigen::Vector3f& centroid_tgt,
                                                    Eigen::Matrix4f& transformation_matrix);
    void calculate_mean_flow(cv::Mat& flow, const cv::Mat& rgb1, const cv::Mat& rgb2);
    void calculate_gray_depth_flow(cv::Mat& flow, const cv::Mat& gray1, const cv::Mat& gray2, const cv::Mat& depth1, const cv::Mat& depth2);
    void calculate_dual_tvl1_flow(cv::Mat& flow, const cv::Mat& gray1, const cv::Mat& gray2);
    void calculate_flow_both_directions(cv::Mat& flow12, cv::Mat& binary, const cv::Mat& gray1, const cv::Mat& gray2);
public:
    static bool register_scans(Eigen::Matrix3f& R, Eigen::Vector3f& t, scan* scan1, scan* scan2);
    float error() const { return last_error; }
    void step(Eigen::Matrix3f& R, Eigen::Vector3f& t);
    fine_registration(scan& scan1, scan& scan2);
};
#endif // FINE_REGISTRATION_H
