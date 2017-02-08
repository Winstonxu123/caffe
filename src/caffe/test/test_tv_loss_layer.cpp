#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/layers/tv_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

const float bottomData[120] = {
    0.,   0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.,   1.1,
    1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.,   2.1,  2.2,  2.3,
    2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.,   3.1,  3.2,  3.3,  3.4,  3.5,
    3.6,  3.7,  3.8,  3.9,  4.,   4.1,  4.2,  4.3,  4.4,  4.5,  4.6,  4.7,
    4.8,  4.9,  5.,   5.1,  5.2,  5.3,  5.4,  5.5,  5.6,  5.7,  5.8,  5.9,
    6.,   6.1,  6.2,  6.3,  6.4,  6.5,  6.6,  6.7,  6.8,  6.9,  7.,   7.1,
    7.2,  7.3,  7.4,  7.5,  7.6,  7.7,  7.8,  7.9,  8.,   8.1,  8.2,  8.3,
    8.4,  8.5,  8.6,  8.7,  8.8,  8.9,  9.,   9.1,  9.2,  9.3,  9.4,  9.5,
    9.6,  9.7,  9.8,  9.9, 10.,  10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7,
    10.8, 10.9, 11.,  11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9
};

template <typename TypeParam>
class TVLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TVLossLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 3, 4, 10)),
        blob_top_loss(new Blob<Dtype>()) {
    Dtype* data = this->blob_bottom_->mutable_cpu_data();
    const int count = this->blob_bottom_->count();
    for (int i = 0; i < count; ++i) {
      data[i] = bottomData[i];
    }
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_loss);
  }
  virtual ~TVLossLayerTest() {
    delete blob_bottom_;
    delete blob_top_loss;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_loss;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TVLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(TVLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TVLossParameter* tv_loss_parameter = layer_param.mutable_tv_loss_param();
  tv_loss_parameter->set_beta(2.5);
  TVLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_loss->cpu_data();
  ASSERT_FLOAT_EQ(top_data[0], 82.0137624747);
}

TYPED_TEST(TVLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TVLossParameter* tv_loss_parameter = layer_param.mutable_tv_loss_param();
  tv_loss_parameter->set_beta(2.5);
  TVLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
