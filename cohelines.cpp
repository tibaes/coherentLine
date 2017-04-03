#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

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
