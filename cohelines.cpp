#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

// Edge Tangent Flow

double Ws(const cv::Point2d &a, const cv::Point2d &b, const int radius) {
  return ((cv::norm(cv::Mat(a), cv::Mat(b)) < radius) ? 1.0 : 0.0);
}

double Wm(const cv::Point2d &a, const cv::Point2d &b, const cv::Mat &mag,
          const double eta = 1.0) {
  const double ga = mag.at<double>(a);
  const double gb = mag.at<double>(b);
  return (0.5 * (1.0 + std::tanh(eta * (gb - ga))));
}

double Wd(const cv::Point2d &a, const cv::Point2d &b, const cv::Mat &tcurr) {
  const cv::Vec2d ta = tcurr.at<cv::Vec2d>(a);
  const cv::Vec2d tb = tcurr.at<cv::Vec2d>(b);
  const double mult = ta.dot(tb);
  return ((mult > 0) ? std::abs(mult) : -1.0 * std::abs(mult));
}

void ETFIteration(cv::Mat &tcurr, const cv::Mat &mag, const int kradius) {
  cv::Mat tnew = cv::Mat::zeros(tcurr.size(), CV_64FC2);
  for (auto y = 0; y < (int)tcurr.size().height; ++y) {
    for (auto x = 0; x < (int)tcurr.size().width; ++x) {
      const auto pa = cv::Point2d(x, y);
      auto sigma = cv::Vec2d(0.0, 0.0);
      for (auto oy = -kradius; oy <= kradius; ++oy) {
        int py = y + oy;
        if (py < 0 || py >= tcurr.size().height)
          continue;
        for (auto ox = -kradius; ox <= kradius; ++ox) {
          int px = x + ox;
          if ((px == 0 && py == 0) || px < 0 || px >= tcurr.size().width)
            continue;
          const auto pb = cv::Point2d(px, py);
          sigma += tcurr.at<cv::Vec2d>(pb) * Ws(pa, pb, kradius) *
                   Wm(pa, pb, mag) * Wd(pa, pb, tcurr);
        }
      }
      cv::normalize(sigma, tnew.at<cv::Vec2d>(pa));
    }
  }
  tcurr = tnew;
}

cv::Mat g_perpendicular(const cv::Mat &gx, const cv::Mat &gy) {
  cv::Mat ng;
  std::vector<cv::Mat> garray = {gy, -1 * gx};
  cv::merge(garray.data(), garray.size(), ng);
  ng.forEach<cv::Vec2d>([](cv::Vec2d &v, const int *pos) {
    auto cp = v;
    cv::normalize(cp, v);
  });
  return ng;
}

cv::Mat ETF(const cv::Mat &gray, const int kernel_radius,
            const int etf_iterations) {
  cv::Mat gx, gy, gm, mag;
  cv::Sobel(gray, gx, CV_64F, 1, 0, 3);
  cv::Sobel(gray, gy, CV_64F, 0, 1, 3);
  cv::magnitude(gx, gy, gm);
  cv::normalize(gm, mag, 1.0, 0.0, cv::NORM_MINMAX);

  cv::imshow("Grad", mag);
  cv::waitKey(100);

  cv::Mat etf = g_perpendicular(gx, gy);
  for (auto i = 0; i < etf_iterations; ++i) {
    cv::Mat etf_vis0, etf_vis1, s[2];
    cv::split(etf, s);
    etf_vis0 = 0.5 * s[0] + 0.5 * s[1];
    // cv::imshow("ETF0-0", s[0]);
    // cv::imshow("ETF0-1", s[1]);
    cv::imshow("ETF0-mean", etf_vis0);

    std::cout << "starting iteration " << i << std::endl;
    ETFIteration(etf, mag, kernel_radius);
    std::cout << "finished iteration " << i << std::endl;

    cv::split(etf, s);
    etf_vis1 = 0.5 * s[0] + 0.5 * s[1];
    cv::imshow("ETF1-0", s[0]);
    cv::imshow("ETF1-1", s[1]);
    cv::imshow("ETF1-mean", etf_vis1);
    cv::waitKey(100);
  }

  return etf;
}

// Flow-based Difference-of-Gaussian

double F(const cv::Point &s, const cv::Mat &gray, const cv::Mat &gauss_dog,
         const cv::Mat &etf, const int delta_q) {
  const cv::Vec2d direction = etf.at<cv::Vec2d>(s);
  const auto perpendicular = cv::Vec2d(direction[1], -direction[0]);

  double integral = 0.0;
  const int half_k = gauss_dog.size().area() / 2;
  for (int k = -half_k; k <= half_k; ++k) {
    const auto offset = k * delta_q * perpendicular;
    const auto target = s + cv::Point(offset);
    if (!target.inside(cv::Rect(cv::Point(0, 0), gray.size())))
      continue;
    integral +=
        gauss_dog.at<double>(k + half_k) * (double)gray.at<uchar>(target);
  }

  return integral;
}

cv::Mat FDOG(const cv::Mat &gray, const cv::Mat &etf, const double p_s,
             const double sigma_c, const double sigma_m, const double thrs,
             const double delta_p, const double delta_q) {
  int p = 2.0 * sigma_m;
  p += (p % 2) ? 0 : 1;
  const auto gauss_kernel_m = cv::getGaussianKernel(p, sigma_m, CV_64F);

  const double sigma_s = 1.6 * sigma_c;
  int q = 2.0 * sigma_c;
  q = (q % 2) ? 0 : 1;
  const auto gauss_kernel_c = cv::getGaussianKernel(q, sigma_c, CV_64F);
  const auto gauss_kernel_s = cv::getGaussianKernel(q, sigma_s, CV_64F);
  const auto gauss_dog_cs = gauss_kernel_c - p_s * gauss_kernel_s; // f(t)

  for (auto y = 0; y < gray.size().height; ++y) {
    for (auto x = 0; x < gray.size().width; ++x) {
      const auto pt_gray = cv::Point(x, y);
      // cx(0)
    }
  }
}

// Coherent Lines Filter

cv::Mat coherentLines(const cv::Mat &img, const int kernel_radius = 5,
                      const int etf_iterations = 5, const double p_s = 0.99,
                      const double sigma_c = 1.0, const double sigma_m = 3.0,
                      const double thrs = 0.2, const double delta_m = 1.0,
                      const double delta_n = 1.0) {
  cv::Mat gray;
  cv::cvtColor(img, gray, CV_BGR2GRAY);
  const auto etf = ETF(gray, kernel_radius, etf_iterations);
  const auto fdog =
      FDOG(gray, etf, p_s, sigma_c, sigma_m, thrs, delta_m, delta_n);
  return cv::Mat();
}

// Sample

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
