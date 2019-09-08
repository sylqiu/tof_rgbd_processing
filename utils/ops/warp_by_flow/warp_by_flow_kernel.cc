#define EIGEN_USE_THREADS

#include <utility>

#include "warp_by_flow.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
template<typename Device>
class WarpByFlowKernel : public OpKernel {
  public:
    explicit WarpByFlowKernel(OpKernelConstruction *ctx): OpKernel(ctx) {
    }
    
    void Compute(OpKernelContext *ctx) override {
      const Tensor& dx_t = ctx->input(0);
      const Tensor& dy_t = ctx->input(1);
      const Tensor& input_img_t = ctx->input(2);
      
      OP_REQUIRES(ctx, dx_t.dims() == 2, errors::InvalidArgument("dx must have dim 2"));
      OP_REQUIRES(ctx, dy_t.dims() == 2, errors::InvalidArgument("dy must have dim 2"));
      OP_REQUIRES(ctx, input_img_t.dims() == 3, errors::InvalidArgument("input_img must have dim 3"));
      
      int input_height = dx_t.dim_size(0);
      int input_width  = dx_t.dim_size(1);
      int img_channel = input_img_t.dim_size(2);

      Tensor *output_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(
                     0, TensorShape({ input_height, input_width, img_channel }), &output_t));
      Tensor *conf_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(
                     1, TensorShape({ input_height, input_width, img_channel }), &conf_t));

      auto dx = dx_t.tensor<float, 2>();
      auto dy = dy_t.tensor<float, 2>();
      auto input_img = input_img_t.tensor<float, 3>();
      auto output = output_t->tensor<float, 3>();
      auto conf = conf_t->tensor<float, 3>();

			// perform warping
      WarpByFlow(ctx->eigen_device<Device>(),
                 dx.data(),
                 dy.data(),
                 input_img.data(),
                 output.data(),
                 conf.data(),
                 input_height,
                 input_width,
                 img_channel);
      
    }
};

REGISTER_KERNEL_BUILDER(Name("WarpByFlow").Device(DEVICE_GPU), WarpByFlowKernel<GPUDevice>)

} // end namespace tensorflow
