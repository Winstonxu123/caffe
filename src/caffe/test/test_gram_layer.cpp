#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/layers/gram_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

const float kGramBottomData[120] = {
    0.00,  0.01,  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08,
    0.09,  0.10,  0.11,  0.12,  0.13,  0.14,  0.15,  0.16,  0.17,
    0.18,  0.19,  0.20,  0.21,  0.22,  0.23,  0.24,  0.25,  0.26,
    0.27,  0.28,  0.29,  0.30,  0.31,  0.32,  0.33,  0.34,  0.35,
    0.36,  0.37,  0.38,  0.39,  0.40,  0.41,  0.42,  0.43,  0.44,
    0.45,  0.46,  0.47,  0.48,  0.49,  0.50,  0.51,  0.52,  0.53,
    0.54,  0.55,  0.56,  0.57,  0.58,  0.59,  0.60,  0.61,  0.62,
    0.63,  0.64,  0.65,  0.66,  0.67,  0.68,  0.69,  0.70,  0.71,
    0.72,  0.73,  0.74,  0.75,  0.76,  0.77,  0.78,  0.79,  0.80,
    0.81,  0.82,  0.83,  0.84,  0.85,  0.86,  0.87,  0.88,  0.89,
    0.90,  0.91,  0.92,  0.93,  0.94,  0.95,  0.96,  0.97,  0.98,
    0.99,  1.00,  1.01,  1.02,  1.03,  1.04,  1.05,  1.06,  1.07,
    1.08,  1.09,  1.10,  1.11,  1.12,  1.13,  1.14,  1.15,  1.16,
    1.17,  1.18,  1.19};

const float kGramTopData[18] = {
     0.247,   0.627,   1.007,   0.627,   1.807,   2.987,   1.007,
     2.987,   4.967,   9.727,  12.507,  15.287,  12.507,  16.087,
    19.667,  15.287,  19.667,  24.047};

template <typename TypeParam>
class GramLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  GramLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Dtype* data = this->blob_bottom_->mutable_cpu_data();
    const int count = this->blob_bottom_->count();
    for (int i = 0; i < count; ++i) {
      data[i] = kGramBottomData[i];
    }
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~GramLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(GramLayerTest, TestDtypesAndDevices);

TYPED_TEST(GramLayerTest, TestSetup) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    GramLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(this->blob_top_->shape().size(), 3);
    ASSERT_EQ(this->blob_top_->shape()[0], 2);
    ASSERT_EQ(this->blob_top_->shape()[1], 3);
    ASSERT_EQ(this->blob_top_->shape()[2], 3);
}

TYPED_TEST(GramLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    GramLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
        ASSERT_FLOAT_EQ(data[i], kGramTopData[i]);
    }
}

TYPED_TEST(GramLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    GramLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
}

}  // namespace caffe
