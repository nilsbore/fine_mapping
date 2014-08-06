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
public:
    static bool register_scans(Eigen::Matrix3f& R, Eigen::Vector3f& t, scan* scan1, scan* scan2);
    float error() const { return last_error; }
    void step(Eigen::Matrix3f& R, Eigen::Vector3f& t);
    fine_registration(scan& scan1, scan& scan2) : scan1(scan1), scan2(scan2)
    {
        pyr_scale = 0.5;
        levels = 3;//5;
        winsize = 20;//200; // 100
        iterations = 2; // 3
        poly_n = 5;//9;//5;
        poly_sigma = 1.0;//1.3;//3.0;//3.2;
        /*levels = 5;
        pyr_scale = 0.5;
        winsize = 13;
        iterations = 10;
        poly_n = 5;
        poly_sigma = 1.1;*/
#ifdef WITH_GPU
        cv::Mat gray1, gray2;
        cvtColor(scan1.rgb_img, gray1, CV_RGB2GRAY);
        cvtColor(scan2.rgb_img, gray2, CV_RGB2GRAY);
        gpugray1.upload(gray1);
        gpugray2.upload(gray2);
#endif
        iteration = 0;
        size_t ox, oy;
        scales = {4.0};//{8.0, 4.0, 2.0, 1.0};
        rgbs1.resize(scales.size());
        rgbs2.resize(scales.size());
        depths1.resize(scales.size());
        depths2.resize(scales.size());
        for (size_t i = 0; i < scales.size(); ++i) {
            scan1.project(depths1[i], rgbs1[i], ox, oy, scan1, scales[i], true);
            scan2.project(depths2[i], rgbs2[i], ox, oy, scan2, scales[i], true);
        }
    }
};
#endif // FINE_REGISTRATION_H
