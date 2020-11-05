#include <opencv2/opencv.hpp>
#include <iostream>

// Refer this: https://stackoverflow.com/questions/24337932/cannot-get-opencv-to-compile-because-of-undefined-references

using namespace cv;

int main() {
    Mat img = imread("bear.jpg", IMREAD_GRAYSCALE);
    // cv::imshow("s", img);
    std::cout << "Yes" << std::endl;
    return 0;
}