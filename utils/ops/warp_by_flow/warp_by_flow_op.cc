#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

Status SetOutput(InferenceContext *c) {
  ShapeHandle dx, dy, input_img, input;
  
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &dx));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &dy));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &input_img));

  TF_RETURN_IF_ERROR(c->Merge(dx, dy, &input));
  
  int64 input_height = c->Value(c->Dim(input, 0));
  int64 input_width  = c->Value(c->Dim(input, 1));
  int64 input_img_channel = c->Value(c->Dim(input_img, 2));

  c->set_output(0, c->MakeShape({ input_height, input_width, input_img_channel}));
  c->set_output(1, c->MakeShape({ input_height, input_width, input_img_channel}));
  return Status::OK();
  
}

REGISTER_OP("WarpByFlow")
.Input("dx: float32")
.Input("dy: float32")
.Input("input_img: float32")
.Output("output: float32")
.Output("conf: float32")
.SetShapeFn(SetOutput);

} // namespace tensorflow
