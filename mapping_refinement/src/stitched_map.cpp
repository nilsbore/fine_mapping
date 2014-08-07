#include "stitched_map.h"

void stitched_map::visualize()
{
    size_t n = scans.size();
    std::vector<cv::Mat> counters;
    counters.resize(n);
    for (size_t i = 0; i < n; ++i) {
        counters[i] = cv::Mat::ones(480, 640, CV_8UC1);
    }

    for (size_t i = 0; i < n; ++i) { // n
        size_t j = (i+1)%n;
        size_t k = (i+2)%n;
        if (i % 2 == 0) {
            if (scans[i]->is_behind(*scans[j])) {
                average_scans(*scans[i], *scans[j], counters[i], counters[j]);
            }
            else {
                average_scans(*scans[j], *scans[i], counters[j], counters[i]);
            }
        }
        if (scans[i]->is_behind(*scans[k])) {
            average_scans(*scans[i], *scans[k], counters[i], counters[k]);
        }
        else {
            average_scans(*scans[k], *scans[i], counters[k], counters[i]);
        }
    }
}

void stitched_map::average_scans(scan& scan1, scan& scan2, cv::Mat& counter1, cv::Mat& counter2)
{
    cv::Mat depth2, rgb2;
    cv::Mat depth1, rgb1;
    size_t ox, oy;
    if (!scan1.project(depth2, rgb2, ox, oy, scan2, 1.0)) {
        return;
    }
    cv::Rect roi(ox, oy, depth2.cols, depth2.rows);
    depth1 = scan1.depth_img(roi);
    rgb1 = scan1.rgb_img(roi);

    float d1, d2;
    cv::Vec3b r1, r2;
    uchar c1, c2;
    for (size_t y = 0; y < depth2.rows; ++y) {
        for (size_t x = 0; x < depth2.cols; ++x) {
            c1 = counter1.at<uchar>(oy + y, ox + x);
            c2 = counter2.at<uchar>(y, x);
            d1 = depth1.at<float>(y, x);
            d2 = depth2.at<float>(y, x);
            r1 = rgb2.at<cv::Vec3b>(y, x);
            r2 = rgb2.at<cv::Vec3b>(y, x);
        }
    }
}
