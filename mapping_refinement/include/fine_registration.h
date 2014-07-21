#ifndef FINE_REGISTRATION_H
#define FINE_REGISTRATION_H

#include <Eigen/Dense>
#include <opencv2/gpu/gpu.hpp>
#include "scan.h"

#define WITH_GPU

class fine_registration {
protected:
    float last_error;
    scan& scan1;
    scan& scan2;
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
                           const cv::Mat& flow, const cv::Mat& invalid) const;
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
        winsize = 100;//200; // 100
        iterations = 2; // 3
        poly_n = 7;//9;//5;
        poly_sigma = 3.0;//1.3;//3.0;//3.2;
        /*numLevels = 5;
        pyrScale = 0.5;
        fastPyramids = false;
        winSize = 13;
        numIters = 10;
        polyN = 5;
        polySigma = 1.1;
        flags = 0;*/
#ifdef WITH_GPU
        cv::Mat gray1, gray2;
        cvtColor(scan1.rgb_img, gray1, CV_RGB2GRAY);
        cvtColor(scan2.rgb_img, gray2, CV_RGB2GRAY);
        gpugray1.upload(gray1);
        gpugray2.upload(gray2);
#endif
    }
};
#endif // FINE_REGISTRATION_H
