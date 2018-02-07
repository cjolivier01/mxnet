/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2017 by Contributors
 * \file batchnorm_test.cc
 * \brief batchnorm operator unit test utility functions
 * \author Chris Olivier
*/

#include <dmlc/logging.h>
#include <mxnet/tensor_blob.h>
#include "../../src/operator/nn/batch_norm-inl.h"
#include "../../src/operator/batch_norm_v1-inl.h"
#include "./test_legacy_op.h"
#include "./test_core_op.h"
#include "executor/exec_pass.h"

using namespace mxnet;

#define SIMPLE_DIMENSIONS  1
#define MXNET_DUMP_C  0
#define DISABLE_VALIDATION 0  // If performance profiling, may do things
// that cause validation to fail

#if !SIMPLE_DIMENSIONS
static constexpr int BATCH_SIZE = 5;
static constexpr int CHANNELS = 3;
static constexpr int DEPTH = 2;
static constexpr int DH = 2;
static constexpr int DW = 3;
#else
static constexpr int BATCH_SIZE = 1;
static constexpr int CHANNELS = 1;
static constexpr int DEPTH = 1;
static constexpr int DH = 2;
static constexpr int DW = 1;
#endif

static constexpr int TIMING_BATCH_SIZE = 128;
static constexpr int TIMING_CHANNELS = 3;
static constexpr int TIMING_DEPTH = 2;
static constexpr int TIMING_DH = 28;
static constexpr int TIMING_DW = 28;

/*! \brief BatchNorm-specific test data  */
template <typename DType, typename AccReal>
class BNOperatorExecutor : public test::op::CoreOpExecutor<DType, AccReal> {
  using Super = typename test::op::CoreOpExecutor<DType, AccReal>;
 public:
  BNOperatorExecutor(const bool isGPU, const TShape& inputShape,
                     const test::op::kwargs_t& kwargs,
                     const bool hasWeightAndBias = false)
    : test::op::CoreOpExecutor<DType, AccReal>(isGPU, { inputShape })
      , hasWeightAndBias_(hasWeightAndBias) {
    param_.Init(kwargs);
  }

  //using BlobVectorType = typename test::op::CoreOpExecutor<DType, AccReal>::BlobVectorType;
/*
  enum DataBlobs {
    kInputData,
    kOutputData,
    kOutputGrad,
    kInputGrad,
    kGamma,
    kBeta,
    kMovingMean,
    kMovingVar
  };
*/
  enum ForwardInputs { kForInData, kForGamma, kForBeta, kForInMovingMean, kForInMovingVar };
  enum ForwardOutputs { kForOutData, kForOutMean, kForOutVar };
  enum BackwardOutputs { kBackOutData, kBackOutGamma, kBackOutBeta, kBackOutMovingMean, kBackOutMovingVar };
  enum BackwardInputs { kBackOutGrad, kBackOutGradMean, kBackOutGradVar, kBackData,
    kBackGamma, kBackBeta, kBackInMovingMean, kBackInMovingVar, kBackInData, kBackInMean,
    kBackInVar };

//  enum WhichArray {
//    kForwardIn,
//    kForwardOut,
//    kBackwardIn,
//    kBackwardOut
//  };

  const NDArray *GetForwardInArray(const ForwardInputs idx) const {
    const std::vector<NDArray> &arrs = Super::inputs();
    CHECK_LT(idx, arrs.size());
    return &arrs[idx];
  }

  const NDArray *GetForwardOutArray(const ForwardOutputs idx) const {
    const std::vector<NDArray> &arrs = Super::outputs();
    CHECK_LT(idx, arrs.size());
    return &arrs[idx];
  }

  const NDArray *GetBackwardOutArray(const BackwardOutputs idx) const {
    const std::vector<NDArray> &arrs = Super::bwd_outputs();
    CHECK_LT(idx, arrs.size());
    return &arrs[idx];
  }

  const NDArray *GetBackwardInArray(const BackwardInputs idx) {
    const std::vector<NDArray> &arrs = Super::bwd_inputs();
    CHECK_LT(idx, arrs.size());
    return &arrs[idx];
  }

  NDArray *GetArray(const ForwardInputs idx) {
    return const_cast<NDArray *>(GetForwardInArray(idx));
  }

  const NDArray *GetArray(const ForwardOutputs idx) {
    return const_cast<NDArray *>(GetForwardOutArray(idx));
  }

  const NDArray *GetArray(const BackwardOutputs idx) {
    return const_cast<NDArray *>(GetBackwardOutArray(idx));
  }

  NDArray *GetArray(const BackwardInputs idx) {
    return const_cast<NDArray *>(GetBackwardInArray(idx));
  }

  const TBlob *GetBackwardInBlob(const BackwardInputs idx) {
    const NDArray * arr = GetBackwardInArray(idx);
    if(arr) {
      return &arr->data();
    }
    return nullptr;
  }

//  const NDArray *GetArray(const WhichArray wa, const int idx) {
//    switch(wa) {
//      case kForwardIn:
//        return GetForwardInArray(idx);
//      case kForwardOut:
//        return GetForwardOutArray(idx);
//      case kBackwardIn:
//        return GetBackwardOutArray(idx);
//      case kBackwardOut:
//      default:
//        CHECK(false);  // need to check params
//        return nullptr;
//    }
//  }

  inline const TBlob& Blob(const NDArray *arr) { return arr->data(); }

  template<typename EnumType>
  const TBlob& GetBlob(const EnumType idx) const {
    return const_cast<BNOperatorExecutor<DType, AccReal> *>(this)->GetArray(idx)->data();
  }

  void resetForward() override {
    Super::resetForward();
    // Start by filling all inputs and outputs with an arbitrary value
    for (size_t i = 0, n = Super::inputs().size(); i < n; ++i) {
      const TBlob& out = Blob(&Super::inputs()[i]);
      const int dtype = out.type_flag_;
      MSHADOW_TYPE_SWITCH(dtype, DTypeX, { test::fill(out, DTypeX(0.1234)); });
    }
    for (size_t i = 0, n = Super::outputs().size(); i < n; ++i) {
      const TBlob& out = Blob(&Super::outputs()[i]);
      const int dtype = out.type_flag_;
      MSHADOW_TYPE_SWITCH(dtype, DTypeX, { test::fill(out, DTypeX(0.1234)); });
    }
    // Init input data
    MSHADOW_TYPE_SWITCH(
      Blob(GetForwardInArray(kForInData)).type_flag_,
      DTypeX,
      {
        DTypeX val = 0;
        test::patternFill<DTypeX>(
          &Blob(GetForwardInArray(kForInData)),
          [&val]{ return val += 1; });
      });

    MSHADOW_TYPE_SWITCH(
      Blob(GetForwardInArray(kForGamma)).type_flag_,
      DTypeX, {
        const TBlob& blob = Blob(GetForwardInArray(kForGamma));
        test::fill(blob, DTypeX(1));
        if (hasWeightAndBias_) {
          if (blob.size(0) > 1) {
            blob.dptr<DTypeX>()[1] = DTypeX(3);
          }
        }
      });
    MSHADOW_TYPE_SWITCH(
      Blob(GetForwardInArray(kForBeta)).type_flag_,
      DTypeX, {
        const TBlob& blob = Blob(GetForwardInArray(kForBeta));
        if (!hasWeightAndBias_) {
          test::fill(blob, DTypeX(0));
        } else {  // This will cause forward pass check to fail when calculating sum == 0
          test::fill(blob, DTypeX(1));
          if (blob.size(0) > 0) {
            blob.dptr<DTypeX>()[0] = DTypeX(3);
          }
        }
      });

    // Init the moving data (all mean = 0, all var = 1)
    MSHADOW_TYPE_SWITCH(
      Blob(GetForwardInArray(kForInMovingMean)).type_flag_,
      DTypeX, {
        test::fill(Blob(GetForwardInArray(kForInMovingMean)), DTypeX(0));
      });
    MSHADOW_TYPE_SWITCH(
      Blob(GetForwardInArray(kForInMovingVar)).type_flag_,
      DTypeX, {
        test::fill(Blob(GetForwardInArray(kForInMovingVar)), DTypeX(1));
      });
  }

  void resetBackward() override {
    Super::resetBackward();

    // Join aux arrays
    *GetArray(kBackInMovingMean) = *GetArray(kForInMovingMean);
    *GetArray(kBackInMovingVar) = *GetArray(kForInMovingVar);
    *GetArray(kBackGamma) = *GetArray(kForGamma);
    *GetArray(kBackBeta) = *GetArray(kForBeta);
    *GetArray(kBackInMean) = *GetArray(kForOutMean);
    *GetArray(kBackInVar) = *GetArray(kForOutVar);
    *GetArray(kBackInData) = *GetArray(kForOutData);

    // Start by filling all backward inputs and outputs with an arbitrary value
//    for (size_t i = 0, n = Super::bwd_inputs().size(); i < n; ++i) {
//      const TBlob& out = Blob(&Super::bwd_inputs()[i]);
//      const int dtype = out.type_flag_;
//      MSHADOW_TYPE_SWITCH(dtype, DTypeX, { test::fill(out, DTypeX(0.5678)); });
//    }
//    for (size_t i = 0, n = Super::bwd_outputs().size(); i < n; ++i) {
//      const TBlob& out = Blob(&Super::bwd_outputs()[i]);
//      const int dtype = out.type_flag_;
//      MSHADOW_TYPE_SWITCH(dtype, DTypeX, { test::fill(out, DTypeX(0.5678)); });
//    }

    MSHADOW_TYPE_SWITCH(
      GetBlob(kBackOutGrad).type_flag_,
      DTypeX, {
        DType val = -.001;
        test::patternFill<DTypeX>( &GetBlob(kBackOutGrad), [&val]{ return val += 1; });
      });

    // out-grad weights
    MSHADOW_TYPE_SWITCH(
      GetBlob(kBackGamma).type_flag_,
      DTypeX,
      { test::try_fill(&GetBlob(kBackGamma), DTypeX(0.1)); });

    // out-grad biases
    MSHADOW_TYPE_SWITCH(
      GetBlob(kBackBeta).type_flag_,
      DTypeX,
      { test::try_fill(&GetBlob(kBackBeta), DTypeX(0.1)); });

    // in-grad
    MSHADOW_TYPE_SWITCH(
      GetBlob(kBackOutData).type_flag_,
      DTypeX,
      { test::try_fill(&GetBlob(kBackOutData), DTypeX(0)); });

//    MSHADOW_TYPE_SWITCH(
//      GetBlob(kBackInMovingMean).type_flag_,
//      DTypeX,
//      { test::try_fill(&GetBlob(kBackInMovingMean), DTypeX(0.5)); });
//
//    MSHADOW_TYPE_SWITCH(
//      GetBlob(kBackInMovingVar).type_flag_,
//      DTypeX,
//      { test::try_fill(&GetBlob(kBackInMovingVar), DTypeX(1.25)); });

    // in-grad weights
//    if (mxnet::op::batchnorm::kGamma < this->c_.blob_in_grad_.size()) {
//      MSHADOW_TYPE_SWITCH(
//        this->c_.blob_in_grad_[mxnet::op::batchnorm::kGamma].type_flag_,
//        DTypeX,
//        { test::try_fill(this->c_.blob_in_grad_, mxnet::op::batchnorm::kGamma, DTypeX(1)); });
//    }
//
//    // in-grad biases
//    if (mxnet::op::batchnorm::kBeta < this->c_.blob_in_grad_.size()) {
//      MSHADOW_TYPE_SWITCH(
//        this->c_.blob_in_grad_[mxnet::op::batchnorm::kBeta].type_flag_,
//        DTypeX,
//        { test::try_fill(this->c_.blob_in_grad_, mxnet::op::batchnorm::kBeta, DTypeX(0)); });
//    }
  }

  const bool hasWeightAndBias_;  // This will cause forward pass validation to fail
  op::BatchNormParam param_;
};

/*! \brief Validate batch norm test outputs */
template<typename DType, typename AccReal>
class BatchNormValidator : public test::op::Validator<DType, AccReal> {
  typedef test::op::Validator<DType, AccReal> Super;

  /*! \brief Only static functions in this class */
  BatchNormValidator() = delete;

  /*! \brief Check batch norm output - 1D */
  static void checkBatchNorm1D(const TBlob *blob) {
    const size_t dim = static_cast<size_t>(blob->ndim());
    CHECK_EQ(dim, 3U);

    const size_t num = blob->shape_[0];  // batch size
    const size_t channels = blob->shape_[1];
    const size_t length = blob->shape_[2];

    size_t itemCount = 0;

    for (size_t j = 0; j < channels; ++j) {
      AccReal sum = 0, var = 0;
      for (size_t i = 0; i < num; ++i) {
        for (size_t k = 0; k < length; ++k) {
          const AccReal data = test::data_at<DType>(blob, {i, j, k});
          sum += data;
          var += data * data;
          ++itemCount;
        }
      }

      const AccReal saveSum = sum, saveVar = var;

      // not channels
      sum /= length * num;
      var /= length * num;

      if (itemCount > 1) {
        // Due to eps, for a small number of entries, the error will be a bit higher for one pass
        const DType kErrorBound = Super::ErrorBound(blob);
        // expect zero mean
        EXPECT_NEAR(0, sum, kErrorBound);
        if (!Super::isNear(AccReal(0), sum, kErrorBound)) {
          LOG(WARNING) << "Sum is not close enough to zero "
                       << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if (!Super::isNear(AccReal(1), var, kErrorBound)) {
          LOG(WARNING) << "Variance is not close enough to 1 "
                       << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
      }
    }
  }

  /*! \brief Check batch norm output - 2D */
  static void checkBatchNorm2D(const TBlob *blob) {
    const size_t dim = static_cast<size_t>(blob->ndim());
    CHECK_EQ(dim, 4U);

    const size_t num = blob->shape_[0];  // batch size
    const size_t channels = blob->shape_[1];
    const size_t height = blob->shape_[2];
    const size_t width = blob->shape_[3];

    size_t itemCount = 0;

    for (size_t j = 0; j < channels; ++j) {
      AccReal sum = 0, var = 0;
      for (size_t i = 0; i < num; ++i) {
        for (size_t k = 0; k < height; ++k) {
          for (size_t l = 0; l < width; ++l) {
            const AccReal data = test::data_at<DType>(blob, {i, j, k, l});
            sum += data;
            var += data * data;
            ++itemCount;
          }
        }
      }

      const AccReal saveSum = sum, saveVar = var;

      // not channels
      sum /= height * width * num;
      var /= height * width * num;

      if (itemCount > 1) {
        const DType kErrorBound = Super::ErrorBound(blob);
        // expect zero mean
        EXPECT_NEAR(0, sum, kErrorBound);
        if (!Super::isNear(AccReal(0), sum, kErrorBound)) {
          LOG(WARNING) << "Sum is not close enough to zero "
                       << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if (!Super::isNear(AccReal(1), var, kErrorBound)) {
          LOG(WARNING) << "Variance is not close enough to 1"
                       << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
      }
    }
  }

  /*! \brief Check batch norm output - 3D */
  static void checkBatchNorm3D(const TBlob *blob) {
    const size_t dim = static_cast<size_t>(blob->ndim());
    CHECK_EQ(dim, 5U);
    const size_t num = blob->shape_[0];  // batch size
    const size_t channels = blob->shape_[1];
    const size_t depth = blob->shape_[2];
    const size_t height = blob->shape_[3];
    const size_t width = blob->shape_[4];

    size_t itemCount = 0;

    for (size_t j = 0; j < channels; ++j) {
      AccReal sum = 0, var = 0;
      for (size_t i = 0; i < num; ++i) {
        for (size_t d = 0; d < depth; ++d) {
          for (size_t k = 0; k < height; ++k) {
            for (size_t l = 0; l < width; ++l) {
              const AccReal data = test::data_at<DType>(blob, {i, j, d, k, l});
              sum = sum + data;
              var = var + (data * data);
              ++itemCount;
            }
          }
        }
      }

      const AccReal saveSum = sum, saveVar = var;

      // not channels
      sum /= depth * height * width * num;
      var /= depth * height * width * num;

      if (itemCount > 1) {
        const DType kErrorBound = Super::ErrorBound(blob);
        // expect zero mean
        EXPECT_NEAR(0, sum, kErrorBound);
        if (!Super::isNear(AccReal(0), sum, kErrorBound)) {
          LOG(WARNING) << "Sum is not close enough to zero "
                       << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
        if (!Super::isNear(AccReal(1), var, kErrorBound)) {
          LOG(WARNING) << "Variance is not close enough to 1 "
                       << saveSum << " (" << sum << "), "
                       << saveVar << " (" << var << ")";
        }
      }
    }
  }

 public:
  template <typename ExecutorType1, typename ExecutorType2, typename EnumType>
  static inline bool compare(const ExecutorType1& i1,
                             const ExecutorType2& i2,
                             const EnumType idx,
                             bool print = false) {
    const TBlob& b1 = i1.GetBlob(idx);
    const TBlob& b2 = i2.GetBlob(idx);
    if (print && test::debug_output) {
      test::print(RunContext(), &(std::cout << "Blob 1:"), b1, true, true);
      test::print(RunContext(), &(std::cout << "Blob 2:"), b2, true, true);
    }
    return test::op::Validator<DType, AccReal>::compare(b1, b2);
  }

  /*! \brief Check batch norm output */
  template<typename BNOperatorProp>
  static void validateForward(const BNOperatorProp& data) {
    //const TBlob& outputBlob = data.output_blobs()[mxnet::op::batchnorm::kData];
    const TBlob &outputBlob = data.GetBlob(BNOperatorProp::kForOutData);
    if (test::debug_output) {
      test::print(RunContext(), &(std::cout << "Fwd Output Blob:"), outputBlob, true, true);
    }
    switch (outputBlob.ndim()) {
      case 3:
        checkBatchNorm1D(&outputBlob);
        break;
      case 4:
        checkBatchNorm2D(&outputBlob);
        break;
      case 5:
        checkBatchNorm3D(&outputBlob);
        break;
      default:
        CHECK(false) << "Supplied shape is not supported for this test";
        break;
    }
  }

  /*! \brief Compare entire operator data between two test sets */
  template<typename PropType1, typename PropType2>
  static void compare(
    const test::op::OpInfo<PropType1, BNOperatorExecutor<DType, AccReal>>& info_1,
    const test::op::OpInfo<PropType2, BNOperatorExecutor<DType, AccReal>>& info_2) {
    // Input
    EXPECT_TRUE(compare(*info_1.executor_, *info_2.executor_,
                        BNOperatorExecutor<DType, AccReal>::kForInData));
    EXPECT_TRUE(compare(*info_1.executor_, *info_2.executor_,
                        BNOperatorExecutor<DType, AccReal>::kForGamma));
    EXPECT_TRUE(compare(*info_1.executor_, *info_2.executor_,
                        BNOperatorExecutor<DType, AccReal>::kForBeta));
    // Output
    EXPECT_TRUE(compare(*info_1.executor_, *info_2.executor_,
                        BNOperatorExecutor<DType, AccReal>::kForOutData));
    CHECK_EQ(info_2.prop_->getParam().use_global_stats,
             info_1.prop_->getParam().use_global_stats);

#if MXNET_USE_CUDNN != 1 /* CUDNN takes a different approach here on first pass */
    // Aux
    EXPECT_TRUE(compare(*info_1.executor_, *info_2.executor_,
                          BNOperatorExecutor<DType, AccReal>::kBackwardOut,
                          BNOperatorExecutor<DType, AccReal>::kForInMovingMean));
    EXPECT_TRUE(compare(*info_1.executor_, *info_2.executor_,
                          BNOperatorExecutor<DType, AccReal>::kBackwardOut,
                          BNOperatorExecutor<DType, AccReal>::kForInMovingVar));
#endif

    if (!info_2.prop_->getParam().use_global_stats) {
      EXPECT_TRUE(compare(*info_1.executor_, *info_2.executor_,
                          BNOperatorExecutor<DType, AccReal>::kBackInMean));
      EXPECT_TRUE(compare(*info_1.executor_, *info_2.executor_,
                          BNOperatorExecutor<DType, AccReal>::kBackInVar));
      // InGrad
      EXPECT_TRUE(compare(*info_1.executor_, *info_2.executor_,
                          BNOperatorExecutor<DType, AccReal>::kForInData));
      EXPECT_TRUE(compare(*info_1.executor_, *info_2.executor_,
                          BNOperatorExecutor<DType, AccReal>::kBackGamma));
      EXPECT_TRUE(compare(*info_1.executor_, *info_2.executor_,
                          BNOperatorExecutor<DType, AccReal>::kBackBeta));
      // OutGrad
      EXPECT_TRUE(compare(*info_1.executor_, *info_2.executor_,
                          BNOperatorExecutor<DType, AccReal>::kBackData));
    }
  }
};

static const test::op::kwargs_t blank_kwargs;
static const test::op::kwargs_t blank_kwargs_nocudnn = {
  {"cudnn_off", "True"} };
static const test::op::kwargs_t nonfixgamma_kwargs = {
  {"fix_gamma", "False"} };
static const test::op::kwargs_t nonfixgamma_kwargs_nocudnn = {
  {"fix_gamma", "False"}, {"cudnn_off", "True"} };
static const test::op::kwargs_t useglobalstats_kwargs = {
  {"use_global_stats", "True"} };
static const test::op::kwargs_t useglobalstats_kwargs_nocudnn = {
  {"use_global_stats", "True"}, {"cudnn_off", "True"} };
static const test::op::kwargs_t nfs_ugs_kwargs = {
  {"fix_gamma", "False"}, {"use_global_stats", "True"}};
static const test::op::kwargs_t nfs_ugs_kwargs_nocudnn = {
  {"fix_gamma", "False"}, {"use_global_stats", "True"}, {"cudnn_off", "True"}  };

#if !DISABLE_VALIDATION
static bool isUGS(const test::op::kwargs_t& kwargs) {
  for (test::op::kwargs_t::const_iterator i = kwargs.begin(),
         e = kwargs.end(); i != e; ++i) {
    if (!i->first.compare("use_global_stats")) {
      return i->second.compare("True") == 0;
    }
  }
  return false;
}
#endif  // DISABLE_VALIDATION

template<typename StreamType, typename OperatorExecutor>
static StreamType& PRT(StreamType *os, const OperatorExecutor& obj,
                       const typename OperatorExecutor::BlobVectorType bvt, const size_t idx) {
  *os << OperatorExecutor::bvt2String(bvt) << ": " << idx
      << ": ";
  const TBlob& blob = obj.getBlobVect(bvt)[idx];

  test::print(RunContext(), os, blob);
  return *os;
}

template<typename StreamType, typename Prop, typename OperatorExecutor>
static StreamType& dumpF(StreamType *os,
                         const test::op::OpInfo<Prop, OperatorExecutor>& prop,
                         const size_t x = 0) {
  if (test::debug_output) {
    *os << std::endl;
    if (x) {
      *os << "=============================" << std::endl;
      *os << "= " << x << std::endl;
      *os << "=============================" << std::endl;
    }
//    typedef typename OperatorExecutor::BlobVectorType BlobVectorType;
//    PRT(os, *prop.executor_, BlobVectorType::kInput, mxnet::op::batchnorm::kData);
//    PRT(os, *prop.executor_, BlobVectorType::kInput, mxnet::op::batchnorm::kGamma);
//    PRT(os, *prop.executor_, BlobVectorType::kInput, mxnet::op::batchnorm::kBeta);
//
//    PRT(os, *prop.executor_, BlobVectorType::kAux, mxnet::op::batchnorm::kMovingMean);
//    PRT(os, *prop.executor_, BlobVectorType::kAux, mxnet::op::batchnorm::kMovingVar);
//
//    PRT(os, *prop.executor_, BlobVectorType::kOutput, mxnet::op::batchnorm::kOut);
//    PRT(os, *prop.executor_, BlobVectorType::kOutput, mxnet::op::batchnorm::kMean);
//    PRT(os, *prop.executor_, BlobVectorType::kOutput, mxnet::op::batchnorm::kVar);
  }
  return *os;
}

template<typename StreamType, typename Prop, typename OperatorExecutor>
static StreamType& dumpB(StreamType *os,
                         const test::op::OpInfo<Prop, OperatorExecutor>& prop,
                         const size_t x = 0) {
  if (test::debug_output) {
    *os << std::endl;
    if (x) {
      *os << "=============================" << std::endl;
      *os << "= " << x << std::endl;
      *os << "=============================" << std::endl;
    }

//    typedef typename OperatorExecutor::BlobVectorType BlobVectorType;
//    PRT(os, *prop.executor_, BlobVectorType::kInGrad, mxnet::op::batchnorm::kData);
//    PRT(os, *prop.executor_, BlobVectorType::kInGrad, mxnet::op::batchnorm::kGamma);
//    PRT(os, *prop.executor_, BlobVectorType::kInGrad, mxnet::op::batchnorm::kBeta);
//
//    PRT(os, *prop.executor_, BlobVectorType::kAux, mxnet::op::batchnorm::kMovingMean);
//    PRT(os, *prop.executor_, BlobVectorType::kAux, mxnet::op::batchnorm::kMovingVar);
//
//    PRT(os, *prop.executor_, BlobVectorType::kOutGrad, mxnet::op::batchnorm::kOut);
  }
  return *os;
}

template<typename StreamType, typename Prop1, typename Prop2, typename OperatorExecutor>
static StreamType& dumpF(StreamType *os,
                         const test::op::OpInfoPair<Prop1, Prop2, OperatorExecutor>& bi) {
  return dumpF(&dumpF(os, bi.info_1_, 1), bi.info_2_, 2);
}

template<typename StreamType, typename Prop1, typename Prop2, typename OperatorExecutor>
static StreamType& dumpB(StreamType *os,
                         const test::op::OpInfoPair<Prop1, Prop2, OperatorExecutor>& bi) {
  return dumpB(&dumpB(os, bi.info_1_, 1), bi.info_2_, 2);
}

/*! \brief Test batch norm operator forward pass */
template<typename OperatorProp, typename OperatorExecutor>
static test::op::OpInfo<OperatorProp, OperatorExecutor> TestBatchNormOperatorForward(
  bool isGPU,
  const TShape& inputShape,
  const std::vector<std::pair<std::string, std::string> >& kwargs,
  const size_t count = 1) {
#if MXNET_USE_CUDA
  if (isGPU && !test::unitTestsWithCuda) {
    LOG(INFO) << "GPU not found, running test as non-GPU";
  }
#else
  isGPU = false;
#endif

  test::op::OpInfo<OperatorProp, OperatorExecutor> info = test::op::createOpAndInfoF<
    OperatorProp, OperatorExecutor>(
    OperatorExecutor::ArgsWithOpName(kwargs, "BatchNorm", "_backward_BatchNorm"),
    isGPU, inputShape, kwargs);

  info.executor_->initForward(*info.prop_, &info.in_type_);

  info.executor_->forward(count);

#if !DISABLE_VALIDATION
  if (!isUGS(kwargs)) {
    BatchNormValidator<typename OperatorExecutor::DataType,
      typename OperatorExecutor::AccRealType>::validateForward(*info.executor_);
  }
#endif

  return info;
}

/*! \brief Test batch norm operator backward pass */
template<typename OperatorProp, typename OperatorExecutor>
static test::op::OpInfo<OperatorProp, OperatorExecutor> runOperatorBackward(
  test::op::OpInfo<OperatorProp, OperatorExecutor> *info,
  const size_t count = 1) {
  info->executor_->initBackward(*info->prop_, &info->in_type_);

  info->executor_->backward(count);
  return *info;
}

static constexpr size_t CYCLE_COUNT = 3;

template<typename OperatorProp1, typename OperatorProp2, typename OperatorExecutor>
static test::op::OpInfoPair<OperatorProp1, OperatorProp2, OperatorExecutor> testForwardAndBackward(
  const bool isGPU1,
  const bool isGPU2,
  const TShape &inputShape,
  const test::op::kwargs_t& kwargs,
  const bool dumpC,
  const size_t count = 1,
  const size_t cycleCount = CYCLE_COUNT) {
  test::op::OpInfo<OperatorProp1, OperatorExecutor> info_1 =
    TestBatchNormOperatorForward<OperatorProp1, OperatorExecutor>(isGPU1, inputShape,
                                                                  kwargs, count);

  test::op::OpInfo<OperatorProp2, OperatorExecutor> info_2 =
    TestBatchNormOperatorForward<OperatorProp2, OperatorExecutor>(isGPU2, inputShape,
                                                                  kwargs, count);

  size_t thisCount = 0;

  typedef typename OperatorExecutor::DataType DType;
  typedef typename OperatorExecutor::AccRealType AccReal;

  do {
    const bool isLast = thisCount == cycleCount - 1;

    if (thisCount) {
      info_1.executor_->forward(count);
      info_2.executor_->forward(count);
    }

    if (isLast) {
      dumpF(&std::cout, info_1, 1);
      dumpF(&std::cout, info_2, 2);
    }

    // Check that everything is the same after the forward pass
    BatchNormValidator<DType, AccReal>::compare(info_1, info_2);

    BatchNormValidator<DType, AccReal>::compare(
      *info_1.executor_, *info_2.executor_,
      OperatorExecutor::kForInData,
      false);

    if (!thisCount) {
      // return backward
      runOperatorBackward(&info_1, count);
      runOperatorBackward(&info_2, count);
    } else {
      info_1.executor_->backward(count);
      info_2.executor_->backward(count);
    }

    if (isLast) {
      dumpB(&std::cout, info_1, 1);
      dumpB(&std::cout, info_2, 2);
    }

    // Check that everything is the same after the backward pass
    BatchNormValidator<DType, AccReal>::compare(info_1, info_2);
  } while (++thisCount < cycleCount);

//  if (dumpC) {
//    info_1.executor_->dumpC(&std::cerr, "BN_testForwardAndBackward");
//  }

  return  { info_1, info_2 };
}
template<typename OperatorProp1, typename OperatorProp2, typename OperatorExecutor>
static test::op::OpInfoPair<OperatorProp1, OperatorProp2, OperatorExecutor>
testForwardAndBackward(const bool isGPU,
                       const TShape &inputShape,
                       const test::op::kwargs_t kwargs,
                       const bool dumpC = false,
                       const size_t count = 1,
                       const size_t cycleCount = CYCLE_COUNT
) {
  return testForwardAndBackward<OperatorProp1, OperatorProp2, OperatorExecutor>(
    isGPU,
    isGPU,
    inputShape,
    kwargs,
    dumpC,
    count,
    cycleCount);
}

// NOTE: This should know which version to use (V1, mkl, etc)
struct BatchNormCoreOpProp : public mxnet::test::op::CoreOpProp {

  void Init(const mxnet::test::op::kwargs_t& kwargs) override {
    mxnet::test::op::CoreOpProp::Init(kwargs);
    params_.Init(kwargs, dmlc::parameter::kAllowUnknown);
  }

  const mxnet::op::BatchNormParam& getParam() const { return params_; }

  mxnet::op::BatchNormParam params_;
};

template<typename OperatorExecutor>
static test::op::OpInfoPair<BatchNormCoreOpProp, BatchNormCoreOpProp, OperatorExecutor>
testBNForwardAndBackward2D(const bool isGPU,
                           const TShape &inputShape,
                           const test::op::kwargs_t& kwargs,
                           const bool dumpC = false) {
  CHECK_EQ(inputShape.ndim(), 4);  // V1 can only handle 2D
  return testForwardAndBackward<BatchNormCoreOpProp,
    BatchNormCoreOpProp, OperatorExecutor>(
    isGPU,
    isGPU,
    inputShape,
    kwargs,
    dumpC);
}

/*
 * Forward tests
 */
TEST(BATCH_NORM, Test2DForwardV1V2) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32,
    DType,
    AccReal,
    {
      // Have to specify somehow v1 and v2
      auto infoA = testBNForwardAndBackward2D<BNOperatorExecutor<DType, AccReal>>(
        false, {BATCH_SIZE, CHANNELS, DH, DW}, blank_kwargs);
    });
}

#if 0

static const std::vector<int> v2_types = {mshadow::kFloat32,
                                          mshadow::kFloat64,
                                          mshadow::kFloat16};

TEST(BATCH_NORM, Test1DForward) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        TestBatchNormOperatorForward<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>>(
          false, {BATCH_SIZE, CHANNELS, DW}, blank_kwargs);
      });
  }
}

TEST(BATCH_NORM, Test2DForwardV1) {
  TestBatchNormOperatorForward<mxnet::op::BatchNormProp, BNOperatorExecutor<float, float>>(
    false,
    {BATCH_SIZE, CHANNELS, DH, DW},
    blank_kwargs);
}

TEST(BATCH_NORM, Test2DForward) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        auto opInfoFloatH = TestBatchNormOperatorForward<mxnet::op::BatchNormProp,
          BNOperatorExecutor<DType, AccReal>>(
          false, {BATCH_SIZE, CHANNELS, DH, DW}, blank_kwargs);
      });
  }
}

TEST(BATCH_NORM, Test3DForward) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        TestBatchNormOperatorForward<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>>(
          false, {BATCH_SIZE, CHANNELS, DEPTH, DH, DW}, blank_kwargs);
      });
  }
}

template<typename PropType, typename OperatorExecutor>
static void timingTest(const std::string& label,
                       const bool isGPU,
                       const bool stochastic,
                       const test::op::kwargs_t& kwargs,
                       const int dim = 0,
                       size_t count = 1) {
  std::cout << std::endl << std::flush;

#ifdef NDEBUG
  size_t COUNT = 50;
#else
  size_t COUNT = 5;
#endif
  if (mxnet::test::quick_test) {
    COUNT = 2;
    count = 1;
  }

  test::perf::TimingInstrument timing;

  std::stringstream ss;
  ss << "Timing: " << COUNT << " iterations";

  for (size_t i = 0; i < COUNT; ++i) {
    index_t batchSize;
    index_t channels;
    index_t depth;
    index_t height;
    index_t width;

    do {
      batchSize = stochastic ? test::rangedRand(1U, BATCH_SIZE * 2U) : TIMING_BATCH_SIZE;
      channels = stochastic ? test::rangedRand(1U, CHANNELS * 2U) : TIMING_CHANNELS;
      depth = stochastic ? test::rangedRand(1U, DEPTH * 2U) : TIMING_DEPTH;
      height = stochastic ? test::rangedRand(1U, DH * 2U) : TIMING_DH;
      width = stochastic ? test::rangedRand(1U, DW * 2U) : TIMING_DW;
    } while (stochastic && (height * width) == 1U);

    const size_t D = dim ? dim - 1U : test::rangedRand(0U, 2U);

    test::op::OpInfo<PropType, OperatorExecutor> info;
    switch (D) {
      case 0:
        info = TestBatchNormOperatorForward<PropType, OperatorExecutor>(
          isGPU,
          {batchSize, channels, width},
          kwargs, count);
        break;
      case 1:
        info = TestBatchNormOperatorForward<PropType, OperatorExecutor>(
          isGPU,
          {batchSize, channels, height, width},
          kwargs, count);
        break;
      case 2:
        info = TestBatchNormOperatorForward<PropType, OperatorExecutor>(
          isGPU,
          {batchSize, channels, depth, height, width},
          kwargs, count);
        break;
      default:
        CHECK(false) << "rangedRand() returned unexpected value";
    }
    if (info.executor_.get()) {
      runOperatorBackward<PropType, OperatorExecutor>(&info, count);
      timing += info.executor_->GetTiming();
    }
  } while (false);

  timing.print(&std::cout, label);
  std::cout << std::endl << std::flush;
}

#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
#define GPU_TEST_DIMENSIONS  2  /* Only support 2D */
#else
#define GPU_TEST_DIMENSIONS  0  /* Allow stochastic */
#endif  // MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5

/*! \brief Stress-test random batch size/channels/dimension(s) */
TEST(BATCH_NORM, TestStochasticTiming_2D) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      timingTest<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>>(
        "RANDOM: BatchNormProp<cpu>", false, true,
        blank_kwargs_nocudnn, GPU_TEST_DIMENSIONS); });
#if MXNET_USE_CUDA
  if (test::unitTestsWithCuda) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      mshadow::kFloat32, DType, AccReal,
      {
        timingTest<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>>(
          "RANDOM: BatchNormProp<gpu>", true, true,
          blank_kwargs_nocudnn, GPU_TEST_DIMENSIONS); });
  }
#endif
}

/*! \brief Performance tests */
#ifndef _WIN32
TEST(BATCH_NORM, TestTiming_2D) {
#ifdef NDEBUG
  size_t THISCOUNT = 10;
#else
  size_t THISCOUNT = 2;
#endif
  if (mxnet::test::quick_test) {
    THISCOUNT = 1;
  }
MSHADOW_REAL_TYPE_SWITCH_EX(
  mshadow::kFloat32, DType, AccReal, {
#if defined(MXNET_USE_MKL2017) && (MXNET_USE_MKL2017 == 1)
  timingTest<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>>(
    "MKL BatchNormProp<cpu> 2D",
    false, false,
    blank_kwargs_nocudnn,
    2, THISCOUNT);
#endif
  test::ScopeSet<volatile bool> disableMKL(&mxnet::op::batchnorm::disable_mkl, true);
  timingTest<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>>(
    "BatchNormProp<cpu> 2D",
    false, false,
    blank_kwargs_nocudnn,
    2, THISCOUNT);
#if MXNET_USE_CUDA
  if (test::unitTestsWithCuda) {
    timingTest<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>>(
      "BatchNormProp<gpu> 2D",
      true, false,
      blank_kwargs_nocudnn,
      2, THISCOUNT);
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
    timingTest<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>>(
      "CUDNN BatchNormProp<gpu> 2D",
      true, false,
      blank_kwargs,
      2, THISCOUNT);
#endif
  }
#endif
});
}
#endif  // _WIN32

/**
 * Backward tests (generally include forward tests as well)
 */

TEST(BATCH_NORM, TestIterAll) {
  TShape shapes[] = {
    TShape({BATCH_SIZE, CHANNELS, DH}),
    TShape({BATCH_SIZE, CHANNELS, DH, DW}),
    TShape({BATCH_SIZE, CHANNELS, DEPTH, DH, DW})
  };
  const char *tof[2] = { "False", "True" };
  test::op::kwargs_t kwargs;
  for (size_t x1 = 0; x1 < 2U; ++x1) {
    kwargs.push_back({ "fix_gamma", tof[x1] });
    for (size_t x2 = 0; x2 < 2U; ++x2) {
      kwargs.push_back({ "use_global_stats", tof[x2] });
      for (size_t x3 = 0; x3 < 2U; ++x3) {
        if (x3) {
          kwargs.push_back({ "cudnn_off", "True" });
        }
        for (TShape shape : shapes) {
          for (int g1 = 0; g1 < 2; ++g1) {
            for (int g2 = 0; g2 < 2; ++g2) {
              for (int type : v2_types) {
                MSHADOW_REAL_TYPE_SWITCH_EX(
                  type, DType, AccReal,
                  {
                    test::op::OpInfoPair<mxnet::op::BatchNormProp, mxnet::op::BatchNormProp,
                      BNOperatorExecutor<DType, AccReal>>
                      bi = testForwardAndBackward<mxnet::op::BatchNormProp,
                      mxnet::op::BatchNormProp,
                      BNOperatorExecutor<DType, AccReal>>(
                      g1 != 0, g2 != 0, shape, kwargs, false);  // Keep it simple
                  });
              }
            }
          }
        }
        if (x3) {
          kwargs.pop_back();
        }
      }
      kwargs.pop_back();
    }
    kwargs.pop_back();
  }
}

TEST(BATCH_NORM, Test2DBackward2DPlusLoadAndCompareLogic) {
  test::ScopeSet<volatile bool> disableMKL(&mxnet::op::batchnorm::disable_mkl, true);
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      Test2DBackward2DPlusLoadAndCompareLogicUtil::test<DType, AccReal>();
    });
}

template<typename PropType, typename OperatorExecutor>
void compare(const bool isGPU,
             const test::op::OpInfo<PropType, OperatorExecutor>& object,
             const std::vector<
               std::vector< std::vector<typename OperatorExecutor::DataType> > >& values) {
  test::op::OpInfo<PropType, OperatorExecutor> info_checkLoad =
    test::op::createOpAndInfoF<PropType, OperatorExecutor>(
      blank_kwargs, isGPU, object.executor_->inputs()[0].shape_);
  info_checkLoad.executor_->initForward(*info_checkLoad.prop_, &info_checkLoad.in_type_);
  info_checkLoad.executor_->initBackward(*info_checkLoad.prop_, &info_checkLoad.in_type_);
  info_checkLoad.executor_->load(values);
  BatchNormValidator<
    typename OperatorExecutor::DataType,
    typename OperatorExecutor::AccRealType>::compare(object, info_checkLoad);
}


#ifndef _WIN32
TEST(BATCH_NORM, TestBackward1D_Simple) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DTypeX, AccReal,
    {
      const TShape inputShape({1, 1, 2});
      test::op::OpInfo<mxnet::op::BatchNormProp, BNOperatorExecutor<DTypeX, AccReal>> info =
        TestBatchNormOperatorForward<mxnet::op::BatchNormProp, BNOperatorExecutor<DTypeX, AccReal>>(
          false, inputShape, blank_kwargs);
      info.executor_->initBackward(*info.prop_, &info.in_type_);
      runOperatorBackward(&info);

#if MXNET_DUMP_C
      info.executor_->dumpC(&std::cerr, "BN_TestBackward1D_Simple");
#endif

      // Expected data state when running forward+backward starting with default values
      // Note: This data structure generated by dumpC()
      static const std::vector< std::vector< std::vector<DTypeX> > >
        ___BN_TestBackward1D_Simple_data_shape_1_1_2___ = {
        { /* kInput */
          { 1.0f, 2.0f },
          { 1.0f },
          { 0.0f }
        },
        { /* kOutput */
          { -0.998006f, 0.998006f },
          { 1.5f },
          { 0.25f }
        },
        { /* kAux */
          { 0.15f },
          { 0.925f }
        },
        { /* kInGrad */
          { -0.00397621f, 0.00397609f },
          { 0.0f },
          { 2.998f }
        },
        { /* kOutGrad */
          { 0.999f, 1.999f }
        }
      };
      compare(false, info, ___BN_TestBackward1D_Simple_data_shape_1_1_2___);
    });
}
#endif  // _WIN32

#ifndef _WIN32
TEST(BATCH_NORM, TestBackward3D) {
  MSHADOW_REAL_TYPE_SWITCH_EX(
    mshadow::kFloat32, DType, AccReal,
    {
      const TShape inputShape({2, 3, 2, 3, 5});
      test::op::OpInfo<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>> info =
        TestBatchNormOperatorForward<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>>(
          false, inputShape, blank_kwargs);
      info.executor_->initBackward(*info.prop_, &info.in_type_);
      runOperatorBackward(&info);
#if MXNET_DUMP_C
      info.executor_->dumpC(&std::cerr, "TestBackward3D");
#endif
    });
}
#endif  // _WIN32

template<typename DType>
class ChannelAxisTestData {
 protected:
  enum Mode { LOAD, SAVE };

  void loadOrSave(const TBlob& blob, int channel_axis, const Mode mode) {
    mxnet::op::batchnorm::BNTensor3<DType> tensor3(blob, channel_axis);
    const TShape &shape = blob.shape_;
    CHECK_GT(shape.ndim(), 0);
    if (channel_axis < 0) {
      channel_axis = shape.ndim() + channel_axis;
    }
    CHECK_LT(channel_axis, shape.ndim());
    const size_t channel_count = shape[channel_axis];
    std::vector<size_t> indexes(channel_count, 0);
    for (size_t outer = 0, outerCount = tensor3.OuterSize(); outer < outerCount; ++outer) {
      for (size_t channel = 0, channelCount = tensor3.ChannelCount();
           channel < channelCount; ++channel) {
        CHECK_LT(channel, channel_data_.size());
        for (size_t inner = 0, innerCount = tensor3.InnerSize(); inner < innerCount; ++inner) {
          CHECK_LT(indexes[channel], channel_data_[channel].size());
          if (mode == SAVE) {
            tensor3.get_ref(outer, channel, inner) = channel_data_[channel][indexes[channel]++];
          } else {  // mode == LOAD
            channel_data_[channel][indexes[channel]++] = tensor3.get_ref(outer, channel, inner);
          }
        }
      }
    }
  }

 public:
  std::vector<std::vector<DType>>   channel_data_;

  static void print(const std::string& label, const std::vector<std::vector<DType>>& m) {
    if (test::debug_output) {
      if (!label.empty()) {
        std::cout << label << ": ";
      }
      for (size_t i = 0, n = m.size(); i < n; ++i) {
        const std::vector<DType> &vec = m[i];
        for (size_t j = 0, jn = vec.size(); j < jn; ++j) {
          if (j) {
            std::cout << ", ";
          }
          const DType val = vec[j];
          std::cout << std::fixed << std::setw(7)
                    << std::setprecision(mxnet::test::MPRINT_PRECISION)
                    << std::right << val;
        }
        std::cout << std::endl;
      }
      std::cout << "-----" << std::endl << std::flush;
    }
  }

  static void print(const std::string& label, const TBlob& blob) {
    if (test::debug_output) {
      if (!label.empty()) {
        std::cout << label << ": ";
      }
      const size_t totalSize = blob.Size();
      for (size_t i = 0; i < totalSize; ++i) {
        const float val = blob.dptr<DType>()[i];
        if (i) {
          std::cout << ", ";
        }
        std::cout << std::fixed << std::setw(7) << std::setprecision(mxnet::test::MPRINT_PRECISION)
                  << std::right << val;
      }
      std::cout << std::endl << std::flush;
    }
  }

  void save(const TBlob& blob, const int channel_axis) {
    loadOrSave(blob, channel_axis, SAVE);
  }

  void load(const TBlob& blob, const int channel_axis) {
    loadOrSave(blob, channel_axis, LOAD);
  }
};

template<typename DType, typename AccReal>
static void compare(const TBlob& blob, const std::vector<DType>& vals) {
  CHECK_EQ(blob.Size(), vals.size());
  const DType *v = blob.dptr<DType>();
  for (size_t i = 0, n = vals.size(); i < n; ++i) {
    const DType vBlob = v[i];
    const DType vVect = vals[i];
    const bool near = BatchNormValidator<DType, AccReal>::isNear(
      vBlob, vVect, BatchNormValidator<DType, AccReal>::ErrorBound(&blob));
    EXPECT_TRUE(near);
    if (!near) {
      LOG(WARNING) << vBlob << " is not near enough to " << vVect << std::endl;
    }
  }
}

#ifndef _WIN32
template<typename DType, typename AccReal>
static void compare(const std::vector<std::vector<float>>& d1,
                    const std::vector<std::vector<float>>& d2) {
  CHECK_EQ(d1.size(), d2.size());
  for (size_t x = 0, xn = d1.size(); x < xn; ++x) {
    const std::vector<float> &vec1 = d1[x];
    const std::vector<float> &vec2 = d2[x];
    CHECK_EQ(vec1.size(), vec2.size());
    for (size_t i = 0, n = vec1.size(); i < n; ++i) {
      const DType v1 = vec1[i];
      const DType v2 = vec2[i];
      const bool near = BatchNormValidator<DType, AccReal>::isNear(
        v1, v2, BatchNormValidator<DType, AccReal>::ERROR_BOUND());
      EXPECT_TRUE(near);
      if (!near) {
        LOG(WARNING) << v1 << " is not near enough to " << v2 << std::endl;
      }
    }
  }
}

template<typename DType, typename AccReal>
static void testSaveAndLoad(const std::vector<size_t>& dims,
                            const int channelAxis,
                            const std::vector<std::vector<DType>>& inputChannelData,
                            const std::vector<DType>& expectedBlobData) {
  ChannelAxisTestData<DType> data;
  data.channel_data_ = inputChannelData;

  TShape shape(dims.size());
  for (size_t i = 0, n = dims.size(); i < n; ++i) {
    shape[i] = index_t(dims[i]);
  }

  std::unique_ptr<test::StandaloneBlob> blob(new test::StandaloneBlob(
    shape, false, mshadow::DataType<DType>::kFlag));

  data.save(*blob, channelAxis);
  ChannelAxisTestData<DType>::print("saved to blob", *blob);
  compare<DType, AccReal>(*blob, expectedBlobData);
  data.load(*blob, channelAxis);
  compare<DType, AccReal>(data.channel_data_, inputChannelData);
}

/*! \brief Check normalization/denormalization of various channel positions */
TEST(BATCH_NORM, TestChannelAxisSaveAndLoad) {
  std::cout << std::endl << std::flush;

  typedef float DType;
  typedef float AccReal;

  const std::vector<std::vector<DType>> myData =
    { { 1.0f, 1.0f, 1.0f, 1.0f },
      { 2.0f, 2.0f, 2.0f, 2.0f },
      { 3.0f, 3.0f, 3.0f, 3.0f } };

  testSaveAndLoad<DType, AccReal>({ 1, 3, 2, 2 }, 1, myData,
                                  { 1.0f, 1.0f, 1.0f, 1.0f,
                                    2.0f, 2.0f, 2.0f, 2.0f,
                                    3.0f, 3.0f, 3.0f, 3.0f});

  testSaveAndLoad<DType, AccReal>({ 1, 2, 2, 3 }, 3, myData,
                                  { 1.0f, 2.0f, 3.0f,
                                    1.0f, 2.0f, 3.0f,
                                    1.0f, 2.0f, 3.0f,
                                    1.0f, 2.0f, 3.0f});

  testSaveAndLoad<DType, AccReal>({ 1, 2, 3, 2 }, 2, myData,
                                  { 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f,
                                    1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f});
}

/*! \brief Insert the channel field `channelCount` into the shape at `channelAxis` position */
static TShape MakeShape(const std::vector<index_t>& shape,
                        signed int channelAxis,
                        const size_t channelCount) {
  if (channelAxis < 0) {
    channelAxis += shape.size() + 1;
  }
  CHECK_LT(channelAxis, shape.size() + 1);
  const index_t dim = index_t(shape.size()) + 1;
  TShape newShape(dim);
  for (size_t x = 0; x < static_cast<size_t>(channelAxis); ++x) {
    newShape[x] = index_t(shape[x]);
  }
  newShape[channelAxis] = index_t(channelCount);
  for (index_t x = channelAxis + 1; x < dim; ++x) {
    newShape[x] = shape[x - 1];
  }
  return newShape;
}


/*! \brief Create and arrange equivalent data with different channel axes, then compare
 * normalized results */
static void runChannelAxisTest(
  const bool isGPU1,
  const bool isGPU2,
  const test::op::kwargs_t& base_kwargs,
  const std::vector<index_t> shape,
  const signed int channelAxis1,
  const signed int channelAxis2,
  const size_t channelCount,
  const bool simpleData,
  const size_t numberOfPasses = 5

) {
  typedef float DType;
  typedef float AccReal;

  size_t spatialSize = 1;
  for (size_t x = 1, n = shape.size(); x < n; ++x) {
    spatialSize *= shape[x];
  }

  const size_t batchSize = shape[0];

  // Create normalized input and output-grad data (inputs to forward and backward pass)
  std::vector<std::vector<DType>> myData, myGradOut;
  DType ival = 1.0f, gval = 0.1f;
  myData.resize(batchSize);
  myData.resize(channelCount);
  myGradOut.resize(channelCount);
  for (size_t c = 0; c < channelCount; ++c) {
    for (size_t i = 0; i < spatialSize; ++i) {
      if (!simpleData) {
        myData[c].push_back(ival += 1.0f);
        myGradOut[c].push_back(gval += 0.1f);
      } else {
        myData[c].push_back(c + 1);
        myGradOut[c].push_back(DType(c + 1) / 10.0f);
      }
    }
  }

  ChannelAxisTestData<DType>::print("myData", myData);
  ChannelAxisTestData<DType>::print("myGradOut", myGradOut);
  ChannelAxisTestData<DType> data_c1, data_c2, grad_c1, grad_c2;

  // For forward pass
  data_c1.channel_data_ = data_c2.channel_data_ = myData;

  // For backward pass
  grad_c1.channel_data_ = grad_c2.channel_data_ = myGradOut;

  test::op::kwargs_t kwargs = base_kwargs;

  // Insert the channel field into the shape at channelAxis position
  const TShape shape_c1 = MakeShape(shape, channelAxis1, channelCount);
  const TShape shape_c2 = MakeShape(shape, channelAxis2, channelCount);

  // Create operator 1 with ChannelAxis2 (normally the experimental one)
  kwargs.push_back({"axis", std::to_string(channelAxis1)});
  test::op::OpInfo<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>> info_c1 =
    test::op::createOpAndInfoF<
      mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>>(
      kwargs, isGPU1, shape_c1);

  // Create operator 2 with ChannelAxis2 (normally the control one)
  kwargs.pop_back();
  kwargs.push_back({"axis", std::to_string(channelAxis2)});
  test::op::OpInfo<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>> info_c2 =
    test::op::createOpAndInfoF<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>>(
      kwargs, isGPU2, shape_c2);
  kwargs.pop_back();

  // Init operators
  info_c1.executor_->initForward(*info_c1.prop_, &info_c1.in_type_);
  info_c1.executor_->initBackward(*info_c1.prop_, &info_c1.in_type_);
  info_c2.executor_->initForward(*info_c2.prop_, &info_c2.in_type_);
  info_c2.executor_->initBackward(*info_c2.prop_, &info_c2.in_type_);

  // Save input data to blob with new shape 1
  data_c1.save(info_c1.executor_->inputs()[0], channelAxis1);
  ChannelAxisTestData<DType>::print("blob 1 input", info_c1.executor_->inputs()[0]);

  // Save input data to blob with new shape 2
  data_c2.save(info_c2.executor_->inputs()[0], channelAxis2);
  ChannelAxisTestData<DType>::print("blob 2 input", info_c2.executor_->inputs()[0]);

  // Save output grad to blob with new shape 1
  grad_c1.save(info_c1.executor_->bwd_inputs()[0], channelAxis1);
  ChannelAxisTestData<DType>::print("blob 1 output grad", info_c1.executor_->bwd_inputs()[0]);

  // Save output grad to blob with new shape 2
  grad_c2.save(info_c2.executor_->bwd_inputs()[0], channelAxis2);
  ChannelAxisTestData<DType>::print("blob 2 output grad", info_c2.executor_->bwd_inputs()[0]);

  // Run both operators forward and backwards several times
  for (index_t x = 0; x < numberOfPasses; ++x) {
    info_c1.executor_->forward();
    info_c2.executor_->forward();

    info_c1.executor_->backward();
    info_c2.executor_->backward();
  }

  // Transform operator 1's blob output to a normalized shape
  data_c1.load(info_c1.executor_->outputs()[0], channelAxis1);
  ChannelAxisTestData<DType>::print("channel data 1", data_c1.channel_data_);

  // Transform operator 2's blob output to a normalized shape
  data_c2.load(info_c2.executor_->outputs()[0], channelAxis2);
  ChannelAxisTestData<DType>::print("channel data 2", data_c2.channel_data_);

  // Compare the operators' output data while they're in a normalized shape
  compare<DType, AccReal>(data_c1.channel_data_, data_c2.channel_data_);

  // Transform operator 1's input-grad blob to a normalized shape
  grad_c1.load(info_c1.executor_->bwd_outputs()[0], channelAxis1);
  ChannelAxisTestData<DType>::print("input grad 1", grad_c1.channel_data_);

  // Transform operator 2's input-grad blob to a normalized shape
  grad_c2.load(info_c2.executor_->bwd_outputs()[0], channelAxis2);
  ChannelAxisTestData<DType>::print("input grad 2", grad_c2.channel_data_);

  // Compare the operators' input grad data while they're in a normalized shape
  compare<DType, AccReal>(grad_c1.channel_data_, grad_c2.channel_data_);
}

TEST(BATCH_NORM, TestChannelAxisSimple) {
  std::cout << std::endl << std::flush;
  const size_t CHANNEL_COUNT = 4;
  const int DEFAULT_AXIS = 1;
  const int NEW_AXIS = -2;
  const bool useSimpleData = true;  // change to true sometimes for troubleshooting
  const std::vector<index_t> shape = {1, 2, 3};
  // Check against base-case of channel axis position 1
  runChannelAxisTest(false, false,
                     useglobalstats_kwargs_nocudnn,
                     shape,
                     DEFAULT_AXIS,
                     NEW_AXIS,
                     CHANNEL_COUNT,
                     useSimpleData);
}

/*! \brief Test varying channel axis shapes
 *  For several channel counts (1-3), test that result data (after reshape) is
 *  equivalent for the default (channel position 1) and all other channel positions
 *  in the shape vector
 *  Channel position 1 (default) is checked everywhere else, so for and
 *  backward result equivalence here implies correctness for other channel positions
 */
TEST(BATCH_NORM, TestChannelAxis) {
  test::ScopeSet<bool> noDebugOutput(&test::debug_output, false);

  test::op::kwargs_t kwargs;
  const std::vector<std::vector<index_t>> shapes =
    {{1, 2},
     {1, 2, 1},
     {1, 2, 3},
     {1, 2, 3, 4}};
  const char *tof[2] = {"False", "True"};

  for (size_t x1 = 0; x1 < 2U; ++x1) {
    kwargs.push_back({"fix_gamma", tof[x1]});
    for (size_t x2 = 0; x2 < 2U; ++x2) {
      kwargs.push_back({"use_global_stats", tof[x2]});
      for (size_t x3 = 0; x3 < 2U; ++x3) {
        kwargs.push_back({"cudnn_off", tof[x3]});
        for (index_t g1 = 0; g1 < 2U; ++g1) {
          for (index_t g2 = 0; g2 < 2U; ++g2) {
            for (const std::vector<index_t> &simpleShape : shapes) {
              const int dim = static_cast<int>(simpleShape.size());
              for (signed int channelAxis = -dim, shapeDim = dim;
                   channelAxis <= shapeDim;
                   ++channelAxis) {
                for (size_t channelCount = 1; channelCount <= 3; ++channelCount) {
                  // Check against base-case of channel axis position 1
                  runChannelAxisTest(g1 != 0, g2 != 0, kwargs, simpleShape,
                                     1, channelAxis, channelCount, false);
                }
              }
            }
          }
        }
        kwargs.pop_back();
      }
      kwargs.pop_back();
    }
    kwargs.pop_back();
  }
}
#endif

#if MXNET_USE_CUDA

TEST(BATCH_NORM, Test2DForward2D_gpu) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        TestBatchNormOperatorForward<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>>(
          true,
          {BATCH_SIZE, CHANNELS, DH, DW},
          blank_kwargs);
        TestBatchNormOperatorForward<mxnet::op::BatchNormProp, BNOperatorExecutor<DType, AccReal>>(
          true,
          {BATCH_SIZE, CHANNELS, DH, DW},
          blank_kwargs_nocudnn);
      });
  }
}

TEST(BATCH_NORM, Test2DBackwardMixed_gpu_cpu) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        const TShape inputShape({1, 1, 2, 1});
        testForwardAndBackward<mxnet::op::BatchNormProp, mxnet::op::BatchNormProp,
          BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, blank_kwargs, false);
        testForwardAndBackward<mxnet::op::BatchNormProp, mxnet::op::BatchNormProp,
          BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, blank_kwargs_nocudnn, false);
      });
  }
}

TEST(BATCH_NORM, Test2DBackwardMixedComplex_gpu_cpu) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
        testForwardAndBackward<mxnet::op::BatchNormProp, mxnet::op::BatchNormProp,
          BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, blank_kwargs, false);
        testForwardAndBackward<mxnet::op::BatchNormProp, mxnet::op::BatchNormProp,
          BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, blank_kwargs_nocudnn, false);
      });
  }
}

// nonfixgamma_kwargs

TEST(BATCH_NORM, Test2DBackwardMixed_gpu_cpu_nfg) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        const TShape inputShape({1, 1, 2, 1});
        testForwardAndBackward<mxnet::op::BatchNormProp, mxnet::op::BatchNormProp,
          BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, nonfixgamma_kwargs, false);
        testForwardAndBackward<mxnet::op::BatchNormProp, mxnet::op::BatchNormProp,
          BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, nonfixgamma_kwargs_nocudnn, false);
      });
  }
}

TEST(BATCH_NORM, Test2DBackwardMixedComplex_gpu_cpu_nfg) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
        testForwardAndBackward<mxnet::op::BatchNormProp, mxnet::op::BatchNormProp,
          BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, nonfixgamma_kwargs, false);
        testForwardAndBackward<mxnet::op::BatchNormProp, mxnet::op::BatchNormProp,
          BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, nonfixgamma_kwargs_nocudnn, false);
      });
  }
}

// useglobalstats_kwargs

TEST(BATCH_NORM, Test2DBackwardMixed_gpu_cpu_ugs) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        const TShape inputShape({2, 3, 2, 2});
        testForwardAndBackward<mxnet::op::BatchNormProp, mxnet::op::BatchNormProp,
          BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, useglobalstats_kwargs_nocudnn, false);
        testForwardAndBackward<mxnet::op::BatchNormProp, mxnet::op::BatchNormProp,
          BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, useglobalstats_kwargs, false);
      });
  }
}

TEST(BATCH_NORM, Test2DBackwardMixedComplex_gpu_cpu_ugs) {
  for (int type :  v2_types) {
    MSHADOW_REAL_TYPE_SWITCH_EX(
      type, DType, AccReal,
      {
        const TShape inputShape({BATCH_SIZE, CHANNELS, DH, DW});
        testForwardAndBackward<mxnet::op::BatchNormProp, mxnet::op::BatchNormProp,
          BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, useglobalstats_kwargs, false);
        testForwardAndBackward<mxnet::op::BatchNormProp, mxnet::op::BatchNormProp,
          BNOperatorExecutor<DType, AccReal>>(
          false, true, inputShape, useglobalstats_kwargs_nocudnn, false);
      });
  }
}

#endif  // MXNET_USE_CUDA

#endif

