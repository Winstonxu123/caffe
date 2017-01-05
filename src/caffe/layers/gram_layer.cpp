#include <vector>

#include "caffe/layers/gram_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GramLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.gram_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector
  K_ = bottom[0]->count(axis);
  // The first "axis-1" dimensions are independent Gram matrices; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis-1);
  // Gram matrices will be N_ by N_
  N_ = bottom[0]->shape(axis-1);
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void GramLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < M_; ++i) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, N_, K_,
      (Dtype)1., &bottom_data[i*K_*N_], &bottom_data[i*K_*N_],
      (Dtype)0., &top_data[i*N_*N_]);
  }
}

template <typename Dtype>
void GramLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < M_; ++i) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, K_, N_,
        (Dtype)1., &top_diff[i*N_*N_], &bottom_data[i*K_*N_],
        (Dtype)0., &bottom_diff[i*K_*N_]);
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, N_,
        (Dtype)1., &top_diff[i*N_*N_], &bottom_data[i*K_*N_],
        (Dtype)1., &bottom_diff[i*K_*N_]);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(GramLayer);
#endif

INSTANTIATE_CLASS(GramLayer);
REGISTER_LAYER_CLASS(Gram);

}  // namespace caffe
