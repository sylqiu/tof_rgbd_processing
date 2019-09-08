#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <iostream>
#include "warp_by_flow.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

// The implementation assumes input_img is a depth image
namespace tensorflow{
typedef Eigen::GpuDevice GPUDevice;

__global__ void WarpByFlow_(
  const float *dx,
  const float *dy,
  const float *input_img,
  float *output,
  float *conf,
  const int height,
  const int width,
  const int channels) {

    for (int ind = 0; ind < height * width * channels; ind ++) {
    conf[ind] = 0.f;
    output[ind] = 0.f;
    }

  //CUDA_1D_KERNEL_LOOP(ind, nthreads) {
    for (int ind = 0; ind < height * width * channels; ind ++) {
    int x = ( ind / channels) % width;
    int y = (( ind / channels ) / width)  % height;
    int z = ind % channels;
    int x2 = x + round(dx[ind]);
    int y2 = y + round(dy[ind]);



    if (x2 >= 0 && x2 < width && y2 >= 0 && y2 < height) {
      int ind2 = ( y2 * width + x2 ) * channels + z;
      if (conf[ind2] == 1.f) {
        if (output[ind2] > input_img[ind]) {
          output[ind2] = input_img[ind];
        }
      }
      else {    
        conf[ind2] = 1.f;
        output[ind2] = input_img[ind];
      }
    }
    }
    //__syncthreads();
  //}
} 

void WarpByFlow(const GPUDevice& device,
  const float *dx,
  const float *dy,
  const float *in,
  float *out,
  float *conf,
  const int height,
  const int width,
  const int channels
  ) {
  //const int in_count_per_sample = 1; //height * width * channels;
  //CudaLaunchConfig config = GetCudaLaunchConfig(in_count_per_sample, device);
  WarpByFlow_ <<< 1, 1, 0, device.stream() >>> (
    dx, dy, in, out, conf, height, width, channels);
}
} // end namespace tensorflow

#endif // GOOGLE CUDA
