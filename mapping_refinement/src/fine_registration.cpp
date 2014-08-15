#include "fine_registration.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctime>

using namespace Eigen;

fine_registration::fine_registration(scan& scan1, scan& scan2) : scan1(scan1), scan2(scan2)
{
    /*pyr_scale = 0.5;
    levels = 2;//5;
    winsize = 100;//200; // 100
    iterations = 2; // 3
    poly_n = 5;//9;//5;
    poly_sigma = 1.1;//1.3;//3.0;//3.2;*/
    pyr_scale = 0.5;
    levels = 2;//5;
    winsize = 40;//200; // 100
    iterations = 2; // 3
    poly_n = 7;//9;//5;
    poly_sigma = 1.1;//1.3;//3.0;//3.2;
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
    scales = {1.0};//{8.0, 4.0, 2.0, 1.0};
    rgbs1.resize(scales.size());
    rgbs2.resize(scales.size());
    depths1.resize(scales.size());
    depths2.resize(scales.size());
    for (size_t i = 0; i < scales.size(); ++i) {
        scan1.project(depths1[i], rgbs1[i], ox, oy, scan1, scales[i], true);
        scan2.project(depths2[i], rgbs2[i], ox, oy, scan2, scales[i], true);
    }
}

bool fine_registration::register_scans(Matrix3f& R, Vector3f& t, scan* scan1, scan* scan2)
{
    clock_t begin = std::clock();

    // DEBUG
    Matrix3f R_comp;
    R_comp.setIdentity();
    Vector3f t_comp;
    t_comp.setZero();
    // DEBUG

    Matrix3f R_orig;
    Vector3f t_orig;
    scan1->get_transform(R_orig, t_orig);
    fine_registration r(*scan1, *scan2);
    AngleAxisf a;
    size_t counter = 0;
    do {
        r.step(R, t);
        scan1->transform(R, t);
        a = AngleAxisf(R);
        ++counter;

        // DEBUG
        R_comp = R_comp*R; // add to total rotation
        t_comp += R_comp*t; // add to total translation
        // DEBUG
    }
    while (counter < 20 && (t.norm() > 0.003 || fabs(a.angle()) > 0.0005));
    Matrix3f R_final;
    Vector3f t_final;
    Matrix3f R2;
    Vector3f t2;
    scan2->get_transform(R2, t2);
    scan1->get_transform(R_final, t_final);
    scan1->set_transform(R_orig, t_orig); // set the original transform again

    clock_t end = std::clock();
    std::cout << "Took " << double(end - begin) / CLOCKS_PER_SEC << "s" << std::endl;

    //R = R_final.transpose()*R2;
    //t = R_final.transpose()*(t2-t_final);

    // DEBUG
    a = AngleAxisf(R_comp);
    float angle = fmod(fabs(a.angle()), 2*M_PI);
    if (t_comp.norm() > 0.2) {//0.1) {
        std::cout << "Incorrect because of translation: " << t_comp.norm() << std::endl;
        R = R_orig.transpose()*R2;
        t = R_orig.transpose()*(t2-t_orig);
        return false;
    }
    else if (angle > 0.04) {
        std::cout << "Incorrect because of rotation: " << angle << std::endl;
        R = R_orig.transpose()*R2;
        t = R_orig.transpose()*(t2-t_orig);
        return false;
    }
    else {
        std::cout << "Correct" << std::endl;
        std::cout << "Translation: " << t.norm() << std::endl;
        std::cout << "Rotation: " << angle << std::endl;
        R = R_final.transpose()*R2;
        t = R_final.transpose()*(t2-t_final);
        return counter < 20;
    }
}

static void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, cv::Mat& invalid, int step, double len, const cv::Scalar& color)
{
    for (int y = 0; y < cflowmap.rows; y += step) {
        for (int x = 0; x < cflowmap.cols; x += step) {
            if (invalid.at<bool>(y, x)) {
                continue;
            }
            const cv::Point2f& fxy = len*flow.at<cv::Point2f>(y, x);
            line(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, cv::Point(x,y), 2, color, -1);
        }
    }
}

// shamelessly stolen from pcl
void fine_registration::get_transformation_from_correlation(
        const Matrix<float, Dynamic, Dynamic>& cloud_src_demean,
        const Vector3f& centroid_src,
        const Matrix<float, Dynamic, Dynamic>& cloud_tgt_demean,
        const Vector3f& centroid_tgt,
        Matrix4f& transformation_matrix)
{
    transformation_matrix.setIdentity();

    // Assemble the correlation matrix H = source * target'
    Matrix3f H = (cloud_src_demean * cloud_tgt_demean.transpose()).topLeftCorner(3, 3);

    // Compute the Singular Value Decomposition
    JacobiSVD<Matrix3f > svd (H, ComputeFullU | ComputeFullV);
    Matrix3f u = svd.matrixU();
    Matrix3f v = svd.matrixV();

    // Compute R = V * U'
    if (u.determinant() * v.determinant() < 0) {
        for (int x = 0; x < 3; ++x) {
            v (x, 2) *= -1;
        }
    }

    Matrix3f R = v * u.transpose();

    // Return the correct transformation
    transformation_matrix.topLeftCorner(3, 3) = R;
    const Vector3f Rc(R * centroid_src.head(3));
    transformation_matrix.block(0, 3, 3, 1) = centroid_tgt.head(3) - Rc;
}

// calculate flow from 2 to 1 instead, have more information in 1
void fine_registration::compute_transform(Matrix3f& R, Vector3f& t, const cv::Mat& depth2, const cv::Mat& depth1,
                                          const cv::Mat& flow, const cv::Mat& invalid, float scale) const
{
    Vector3f p1, p2;
    size_t n = flow.rows*flow.cols;
    Matrix<float, Dynamic, Dynamic> x1(3, n);
    Matrix<float, Dynamic, Dynamic> x2(3, n);
    size_t counter = 0;
    float z1, z2;
    for (int y = 0; y < flow.rows; ++y) {
        for (int x = 0; x < flow.cols; ++x) {
            if (invalid.at<bool>(y, x)) {
                continue;
            }
            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
            z1 = depth1.at<float>(y, x);
            p1 = scan1.reproject_point(x, y, z1, scale);
            z2 = depth2.at<float>(y+fxy.y, x+fxy.x);
            if (z2 == 0 || fabs(z1-z2) > 0.4) {
                continue;
            }
            p2 = scan1.reproject_point(x+fxy.x, y+fxy.y, z2, scale);
            x1.col(counter) = p1;
            Vector3f diff = (p2 - p1);
            x2.col(counter) = p1 + diff;

            ++counter;
        }
    }
    x1.conservativeResize(3, counter);
    x2.conservativeResize(3, counter);
    Vector3f x1m = x1.rowwise().mean();
    Vector3f x2m = x2.rowwise().mean();
    x1 -= x1m.replicate(1, counter);
    x2 -= x2m.replicate(1, counter);
    Matrix4f transformation;
    get_transformation_from_correlation(x1, x1m, x2, x2m, transformation);
    R = transformation.topLeftCorner(3, 3);
    t = transformation.block(0, 3, 3, 1);
}

float fine_registration::compute_error(const cv::Mat& depth2, const cv::Mat& depth1, const cv::Mat& invalid) const
{
    float rtn = 0;
    float val = 0;
    float counter = 0;
    for (int y = 0; y < depth2.rows; ++y) {
        for (int x = 0; x < depth2.cols; ++x) {
            if (invalid.at<bool>(y, x)) {
                continue;
            }
            val = depth2.at<float>(y, x) - depth1.at<float>(y, x);
            rtn += fabs(val);
            counter += 1.0;
        }
    }
    return rtn/counter;
}

void fine_registration::calculate_mean_flow(cv::Mat& flow, const cv::Mat& rgb1, const cv::Mat& rgb2)
{
    std::vector<cv::Mat> channels1(3);
    // split img:
    cv::split(rgb1, channels1);

    std::vector<cv::Mat> channels2(3);
    // split img:
    cv::split(rgb2, channels2);

    std::vector<cv::Mat> flows(3);
    for (size_t i = 0; i < 3; ++i) {
        cv::calcOpticalFlowFarneback(channels1[i], channels2[i], flows[i], pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0);
    }

    flow = (flows[0] + flows[1] + flows[2]) / 3.0;
}

void fine_registration::calculate_gray_depth_flow(cv::Mat& flow, const cv::Mat& gray1, const cv::Mat& gray2, const cv::Mat& depth1, const cv::Mat& depth2)
{
    cv::Mat gray_flow, depth_flow;

    cv::calcOpticalFlowFarneback(gray1, gray2, gray_flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0);
    cv::calcOpticalFlowFarneback(depth1, depth2, depth_flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0);

    flow = (gray_flow + depth_flow) / 2.0;
}

void fine_registration::step(Matrix3f& R, Vector3f& t)
{
    R.setIdentity();
    t.setZero();
    cv::Mat depth2, rgb2;
    cv::Mat depth1, rgb1;
    size_t ox, oy;
    float scale = scales[iteration];
    bool behind = scan1.is_behind(scan2);
    if (behind) {
        if (!scan1.project(depth2, rgb2, ox, oy, scan2, scale)) {
            return;
        }
        cv::Rect roi(ox, oy, depth2.cols, depth2.rows);
        depth1 = depths1[iteration](roi);
        rgb1 = rgbs1[iteration](roi);
    }
    else {
        if (!scan2.project(depth2, rgb2, ox, oy, scan1, scale)) {
            return;
        }
        cv::Rect roi(ox, oy, depth2.cols, depth2.rows);
        depth1 = depths2[iteration](roi);
        rgb1 = rgbs2[iteration](roi);
    }
    
    //cv::Mat binary = depth2 == 0;
    cv::Mat binary;
    cv::bitwise_or(depth1 == 0, depth2 == 0, binary);
    const bool use_mask = true;
    if (use_mask) {
        int morph_size = 10;

        int morph_type = cv::MORPH_ELLIPSE;
        cv::Mat element = cv::getStructuringElement(morph_type,
                                           cv::Size(2*morph_size + 1, 2*morph_size+1),
                                           cv::Point(morph_size, morph_size));
        cv::Mat temp1, temp2;
        /// Apply the dilation operation

        cv::erode(binary, temp1, element);
        element = cv::getStructuringElement(morph_type,
                                           cv::Size(3*morph_size + 1, 3*morph_size+1),
                                           cv::Point(morph_size, morph_size));
        /// Apply the erosion operation
        cv::dilate(temp1, temp2, element);
        cv::bitwise_or(binary, temp2, temp1);
        binary = temp1;
    }
    
    cv::Mat gray2, gray1, flow, cflow;
    cvtColor(rgb2, gray2, CV_RGB2GRAY);
    cvtColor(rgb1, gray1, CV_RGB2GRAY);

    // if you want to use R, G or B channel instead for grey image
    /*std::vector<cv::Mat> channels1(3);
    // split img:
    cv::split(rgb1, channels1);

    std::vector<cv::Mat> channels2(3);
    // split img:
    cv::split(rgb2, channels2);*/

#ifdef WITH_GPU
    cv::gpu::GpuMat gpu1;
    if (behind) {
        gpu1 = gpugray1(cv::Rect(ox, oy, depth2.cols, depth2.rows));
    }
    else {
        gpu1 = gpugray2(cv::Rect(ox, oy, depth2.cols, depth2.rows));
    }
    cv::gpu::FarnebackOpticalFlow opflow;
    opflow.pyrScale = pyr_scale;
    opflow.numLevels = levels;
    opflow.winSize = winsize;
    opflow.numIters = iterations;
    opflow.polyN = poly_n;
    opflow.polySigma = poly_sigma;
    opflow.flags = 0;
    cv::gpu::GpuMat gpu2, gpuflowx, gpuflowy;
    gpu2.upload(gray2);
    opflow(gpu1, gpu2, gpuflowx, gpuflowy);
    cv::Mat flowx(gpuflowx);
    cv::Mat flowy(gpuflowy);
    std::vector<cv::Mat> flowar = {flowx, flowy};
    cv::merge(flowar, flow);
#else
    cv::calcOpticalFlowFarneback(gray1, gray2, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0);
    //cv::calcOpticalFlowFarneback(channels1[2], channels2[2], flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0);
    //calculate_mean_flow(flow, rgb1, rgb2);
    //calculate_gray_depth_flow(flow, rgb1, rgb2, depth1, depth2);
#endif

    compute_transform(R, t, depth2, depth1, flow, binary, scale);

    if (!behind) {
        Matrix3f Rt = R.transpose();
        t = -Rt*t;
        R = Rt;
    }

    const bool draw_flow = true;
    if (draw_flow) {
        cv::cvtColor(gray1, cflow, CV_GRAY2BGR);
        std::vector<cv::Mat> images(3);
        //images.at(0) = channels2[2]; //for blue channel
        images.at(0) = gray2; //for blue channel
        //images.at(1) = cv::Mat::zeros(channels2[2].rows, channels2[2].cols, channels2[2].type()); //for green channel
        images.at(1) = cv::Mat::zeros(gray2.rows, gray2.cols, gray2.type());   //for green channel
        //images.at(2) = channels1[2]; //for red channel
        images.at(2) = gray1;  //for red channel
        cv::Mat colorImage;
        cv::merge(images, colorImage);
        cv::namedWindow("Diff", CV_WINDOW_AUTOSIZE);
        cv::imshow("Diff", colorImage);
        cv::waitKey(10);

        drawOptFlowMap(flow, colorImage, binary, 20, 1.0, CV_RGB(0, 255, 0));

        int midx = colorImage.cols/2;
        int midy = colorImage.rows/2;
        line(colorImage, cv::Point(midx, midy), cv::Point(midx+cvRound(1000*t(0)), midy+cvRound(1000*t(1))), CV_RGB(255, 0, 0));
        circle(colorImage, cv::Point(midx,midy), 2, CV_RGB(255, 0, 0), -1);
        cv::namedWindow("Flow", CV_WINDOW_AUTOSIZE);
        cv::imshow("Flow", colorImage);
        cv::waitKey(10);
    }
}
