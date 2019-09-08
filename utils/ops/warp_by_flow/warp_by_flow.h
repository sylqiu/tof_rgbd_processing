#ifndef WARP_BY_FLOW_H_
#define WARP_BY_FLOW_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

void WarpByFlow(const GPUDevice& device,
  const float *dx,
  const float *dy,
  const float *input_img,
  float *output,
  float *conf,
  const int height,
  const int width,
  const int channels);
} // end namespace tensorflow

#endif // ifndef WARP_BY_FLOW_H_
