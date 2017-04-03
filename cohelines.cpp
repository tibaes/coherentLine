#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

int Ws(const cv::Point2d &a, const cv::Point2d &b, const int radius) {
  return ((cv::norm(cv::Mat(a), cv::Mat(b), cv::NORM_L2) < radius) ? 1 : 0);
}

double Wm(const cv::Point2d &a, const cv::Point2d &b, const cv::Mat &grad,
          const double eta = 1.0) {
  double ga = grad.at<>
}

cv::Mat coherentLines(const cv::Mat &img, const int kernel_radius = 5) {
  cv::Mat gray;
  cv::cvtColor(img, gray, CV_BGR2GRAY);

  cv::Mat gx, gy, gm, gt, gmnorm;
  cv::Sobel(gray, gx, CV_16S, 1, 0, 3);
  cv::Sobel(gray, gy, CV_16S, 0, 1, 3);
  cv::magnitude(gx, gy, gm);
  cv::normalize(gm, gt, 1.0, 0.0, cv::NORM_INF);
  gt.convertTo(gmnorm, CV_64FC1);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << argv[0] << " <path_to_image>" << std::endl;
    return 1;
  }

  auto img = cv::imread(argv[1]);
  cv::imshow("input", img);
  cv::waitKey(0);

  return 0;
}
