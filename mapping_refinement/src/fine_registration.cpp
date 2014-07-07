#include "fine_registration.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace Eigen;

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
    Eigen::JacobiSVD<Matrix3f > svd (H, Eigen::ComputeFullU | Eigen::ComputeFullV);
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

void fine_registration::compute_transform(Matrix3f& R, Vector3f& t, const cv::Mat& depth2, const cv::Mat& depth1,
                                          const cv::Mat& flow, const cv::Mat& invalid) const
{
    Vector3f p1, p2;
    //Vector3f delta;
    size_t n = flow.rows*flow.cols;
    Matrix<float, Dynamic, Dynamic> x1(3, n);
    Matrix<float, Dynamic, Dynamic> x2(3, n);
    size_t counter = 0;
    //float diff;
    float z1, z2;
    for (int y = 0; y < flow.rows; ++y) {
        for (int x = 0; x < flow.cols; ++x) {
            if (invalid.at<bool>(y, x)) {
                continue;
            }
            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
            //diff = (depth2.at<float>(y, x) - depth1.at<float>(y, x))/(depth2.at<float>(y, x) + depth1.at<float>(y, x));
            z1 = depth1.at<float>(y, x);
            p1 = scan1.reproject_point(x, y, z1);
            z2 = depth2.at<float>(y+fxy.y, x+fxy.x);
            if (z2 == 0 || fabs(z1-z2) > 0.1) {
                continue;
            }
            p2 = scan1.reproject_point(x+fxy.x, y+fxy.y, z2);
            //delta = 0.0005*Vector3f(fxy.x, fxy.y, 0.0*diff);
            x1.col(counter) = p1;
            x2.col(counter) = p1 + 0.80*(p2 - p1);//point + delta;

            ++counter;
        }
    }
    x1.conservativeResize(3, counter);
    x2.conservativeResize(3, counter);
    Vector3f x1m = x1.rowwise().mean();
    Vector3f x2m = x2.rowwise().mean();
    //x1m(3) = 0.0;
    //x2m(3) = 0.0;
    x1 -= x1m.replicate(1, counter);
    x2 -= x2m.replicate(1, counter);
    //x1.row(3).setZero();
    //x2.row(3).setZero();
    Matrix4f transformation;
    get_transformation_from_correlation(x1, x1m, x2, x2m, transformation);
    R = transformation.topLeftCorner(3, 3);
    t = transformation.block(0, 3, 3, 1);
    //std::cout << "Mean translation: " << (x2m-x1m).transpose() << std::endl;
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
            rtn += fabs(val);//val*val;
            counter += 1.0;
        }
    }
    return rtn/counter;
}

void fine_registration::step(Matrix3f& R, Vector3f& t)
{
    cv::Mat depth2, rgb2;
    cv::Mat depth1, rgb1;
    size_t ox, oy;
    bool behind = scan1.is_behind(scan2);
    if (behind) {
        scan1.project(depth2, rgb2, ox, oy, scan2);
        if (ox < 0 || ox >= depth2.size().width || oy < 0 || oy >= depth2.size().height) {
            std::cout << "Scans not overlapping!" << std::endl;
        }
        std::cout << "ox: " << ox << std::endl;
        std::cout << "oy: " << oy << std::endl;
        scan1.submatrices(depth1, rgb1, ox, oy, depth2.cols, depth2.rows);
    }
    else {
        scan2.project(depth2, rgb2, ox, oy, scan1);
        if (ox < 0 || ox >= depth2.size().width || oy < 0 || oy >= depth2.size().height) {
            std::cout << "Scans not overlapping!" << std::endl;
        }
        std::cout << "ox: " << ox << std::endl;
        std::cout << "oy: " << oy << std::endl;
        scan2.submatrices(depth1, rgb1, ox, oy, depth2.cols, depth2.rows);
    }
    //Matrix3f Rdiff = R1.transpose()*R2;
    //Vector3f tdiff = R1.transpose()*(t2 - t1);
    
    /*cv::namedWindow("Depth1", CV_WINDOW_AUTOSIZE);
    cv::imshow("Depth1", depth);
    
    cv::namedWindow("Depth2", CV_WINDOW_AUTOSIZE);
    cv::imshow("Depth2", depth1);*/
    
    /*cv::namedWindow("Rgb1", CV_WINDOW_AUTOSIZE);
    cv::imshow("Rgb1", rgb);
    
    cv::namedWindow("Rgb2", CV_WINDOW_AUTOSIZE);
    cv::imshow("Rgb2", rgb1);*/
    //cv::waitKey(0);
    
    cv::Mat binary;
    cv::bitwise_or(depth1 == 0, depth2 == 0, binary);
    
    if (true) {
        int morph_size = 4;
        int const max_elem = 2;
        int const max_kernel_size = 21;

        int morph_type = cv::MORPH_ELLIPSE;
        cv::Mat element = cv::getStructuringElement(morph_type,
                                           cv::Size(2*morph_size + 1, 2*morph_size+1),
                                           cv::Point(morph_size, morph_size));
        cv::Mat temp1, temp2;
        /// Apply the dilation operation

        cv::erode(binary, temp1, element);
        //cv::imshow("Erosion Demo", temp);
        element = cv::getStructuringElement(morph_type,
                                           cv::Size(3*morph_size + 1, 3*morph_size+1),
                                           cv::Point(morph_size, morph_size));
        /// Apply the erosion operation
        cv::dilate(temp1, temp2, element);
        cv::bitwise_or(binary, temp2, temp1);
        binary = temp1;
    }
    
    //cv::namedWindow("Binary", CV_WINDOW_AUTOSIZE);
    //cv::imshow("Binary", binary);
    //cv::waitKey(0);
    
    cv::Mat gray2, gray1, flow, cflow;
    cvtColor(rgb2, gray2, CV_RGB2GRAY);
    cvtColor(rgb1, gray1, CV_RGB2GRAY);
    //cv::calcOpticalFlowFarneback(gray1, gray2, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0); // cv::OPTFLOW_FARNEBACK_GAUSSIAN

    std::vector<cv::Mat> channels1(3);
    // split img:
    cv::split(rgb1, channels1);

    std::vector<cv::Mat> channels2(3);
    // split img:
    cv::split(rgb2, channels2);
    cv::calcOpticalFlowFarneback(channels1[2], channels2[2], flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0); // cv::OPTFLOW_FARNEBACK_GAUSSIAN

    cv::cvtColor(gray1, cflow, CV_GRAY2BGR);
    std::vector<cv::Mat> images(3);
    images.at(0) = gray2; //for blue channel
    images.at(1) = cv::Mat::zeros(gray2.rows, gray2.cols, gray2.type());   //for green channel
    images.at(2) = gray1;  //for red channel
    cv::Mat colorImage;
    cv::merge(images, colorImage);
    cv::namedWindow("Diff", CV_WINDOW_AUTOSIZE);
    cv::imshow("Diff", colorImage);
    cv::waitKey(10);

    drawOptFlowMap(flow, colorImage, binary, 20, 1.0, CV_RGB(0, 255, 0));
    
    //cv::Mat depth16, depth116;
    //depth.convertTo(depth16, CV_16U, 8.0*1000.0);
    //depth1.convertTo(depth116, CV_16U, 8.0*1000.0);
    //cv::calcOpticalFlowFarneback(depth16, depth116, flow, pyr_scale, 2, 40, iterations, poly_n, poly_sigma, 0); // cv::OPTFLOW_FARNEBACK_GAUSSIAN
    //cv::cvtColor(depth16, cflow, CV_GRAY2BGR);
    //drawOptFlowMap(flow, cflow, binary, 10, 0.05, CV_RGB(0, 255, 0));
    //cv::namedWindow("DepthFlow", CV_WINDOW_AUTOSIZE);
    //cv::imshow("DepthFlow", cflow);
    //cv::waitKey(10);

    compute_transform(R, t, depth2, depth1, flow, binary);

    int midx = colorImage.cols/2;
    int midy = colorImage.rows/2;
    line(colorImage, cv::Point(midx, midy), cv::Point(midx+cvRound(1000*t(0)), midy+cvRound(1000*t(1))), CV_RGB(255, 0, 0));
    circle(colorImage, cv::Point(midx,midy), 2, CV_RGB(255, 0, 0), -1);
    cv::namedWindow("Flow", CV_WINDOW_AUTOSIZE);
    cv::imshow("Flow", colorImage);
    cv::waitKey(10);

    Matrix3f Rt = R.transpose();
    if (!behind) {
        t = -Rt*t;
        R = Rt;
    }
    last_error = compute_error(depth2, depth1, binary);
}
