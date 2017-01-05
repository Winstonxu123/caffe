#include <vector>

#include "caffe/layers/gram_layer.hpp"

namespace caffe {

template <typename Dtype>
void GramLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int i = 0; i < M_; ++i) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, N_, K_,
      (Dtype)1., &bottom_data[i*K_*N_], &bottom_data[i*K_*N_],
      (Dtype)0., &top_data[i*N_*N_]);
  }
}

template <typename Dtype>
void GramLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    for (int i = 0; i < M_; ++i) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, K_, N_,
        (Dtype)1., &top_diff[i*N_*N_], &bottom_data[i*K_*N_],
        (Dtype)0., &bottom_diff[i*K_*N_]);
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, N_,
        (Dtype)1., &top_diff[i*N_*N_], &bottom_data[i*K_*N_],
        (Dtype)1., &bottom_diff[i*K_*N_]);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GramLayer);

}  // namespace caffe
