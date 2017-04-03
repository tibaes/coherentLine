#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

int Ws(const cv::Point2d &a, const cv::Point2d &b, const int radius) {
  return ((cv::norm(cv::Mat(a), cv::Mat(b), cv::NORM_L2) < radius) ? 1 : 0);
}

double Wm(const cv::Point2d &a, const cv::Point2d &b, const cv::Mat &grad,
          const double eta = 1.0) {
  double ga = grad.at<double>(a.y, a.x);
  double gb = grad.at<double>(b.y, b.x);
  return (0.5 * (1.0 + std::tanh(eta * (gb - ga))));
}

double Wd(const cv::Point2d &a, const cv::Point2d &b, const cv::Mat &tcurr) {
  double ta = tcurr.at<double>(a.y, a.x);
  double tb = tcurr.at<double>(b.y, b.x);
  double mult = ta * tb;
  return ((mult > 0) ? std::abs(mult) : -1.0 * std::abs(mult));
}

void ETFIteration(cv::Mat &tcurr, const cv::Mat &grad, const int kradius) {
  cv::Mat tnew = cv::Mat::zeros(tcurr.size(), CV_64FC1);
  for (auto y = 0; y < (int)tcurr.size().height; ++y) {
    for (auto x = 0; x < (int)tcurr.size().width; ++x) {
      const auto pa = cv::Point2d(x, y);
      double sigma = 0;
      double k = 0;
      for (auto oy = -kradius; oy <= kradius; ++oy) {
        int py = y + oy;
        if (py < 0 || py > tcurr.size().height)
          continue;
        for (auto ox = -kradius; ox <= kradius; ++ox) {
          int px = x + ox;
          if (px < 0 || px > tcurr.size().width)
            continue;
          k += 1.0;
          const auto pb = cv::Point2d(px, py);
          sigma += tcurr.at<double>(pb) * Ws(pa, pb, kradius) *
                   Wm(pa, pb, grad) * Wd(pa, pb, tcurr);
        }
      }
      tnew.at<double>(pa) = (k > 0.1) ? sigma / k : 0.0;
    }
  }
  tcurr = tnew;
}

cv::Mat coherentLines(const cv::Mat &img, const int kernel_radius = 5,
                      const int etf_iterations = 3) {
  cv::Mat gray;
  cv::cvtColor(img, gray, CV_BGR2GRAY);

  cv::Mat gx, gy, gm, gt, gmnorm;
  cv::Sobel(gray, gx, CV_64F, 1, 0, 3);
  cv::Sobel(gray, gy, CV_64F, 0, 1, 3);
  cv::magnitude(gx, gy, gm);
  cv::normalize(gm, gt, 1.0, 0.0, cv::NORM_INF);
  gt.convertTo(gmnorm, CV_64FC1);

  cv::imshow("Grad", gmnorm);
  cv::waitKey(100);

  cv::Mat etf;
  gmnorm.copyTo(etf);
  for (auto i = 0; i < etf_iterations; ++i)
    ETFIteration(etf, gmnorm, kernel_radius);

  cv::Mat etf_vis;
  cv::normalize(etf, etf_vis, 1.0, 0.0, cv::NORM_MINMAX);
  cv::imshow("ETF", etf_vis);

  return etf;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << argv[0] << " <path_to_image>" << std::endl;
    return 1;
  }

  auto img = cv::imread(argv[1]);
  cv::imshow("input", img);
  cv::waitKey(100);

  coherentLines(img);
  cv::waitKey(0);

  return 0;
}
