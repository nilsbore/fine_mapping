#include <iostream>
#include <opencv2/gpu/gpu.hpp>

int main()
{
    std::cout << cv::gpu::getCudaEnabledDeviceCount() << std::endl;
    return 0;
}
