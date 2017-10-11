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
 * \file fully_connect_op-inl.h
 * \brief fully connect operator and symbol
*/
#ifndef MXNET_OPERATOR_FULLY_CONNECTED_INL_H_
#define MXNET_OPERATOR_FULLY_CONNECTED_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./elemwise_op_common.h"
#include "linalg.h"

namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace fullc {
enum FullyConnectedOpInputs {kData, kWeight, kBias};
enum FullyConnectedOpOutputs {kOut};
}  // fullc

struct FullyConnectedParam : public dmlc::Parameter<FullyConnectedParam> {
  int num_hidden;
  bool no_bias;
  bool flatten;
  DMLC_DECLARE_PARAMETER(FullyConnectedParam) {
    // TODO(bing) add support for boolean
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output.");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(flatten).set_default(true)
    .describe("Whether to collapse all but the first axis of the input data tensor.");
  }
};

/**
 * \brief This is the implementation of fully connected operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename DType>
class FullyConnectedOp : public Operator {
 public:
  explicit FullyConnectedOp(FullyConnectedParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    if (req[fullc::kOut] == kNullOp) return;
    CHECK_EQ(req[fullc::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    // TODO(bing): check the BLAS Handle, be careful
    // maybe need blas handle from context
    // TODO(bing): judge shape to remove flatten op
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__
    const TShape& ishape = in_data[fullc::kData].shape_;
    const TShape& oshape = out_data[fullc::kOut].shape_;

    Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> data, out;
    if (!param_.flatten) {
      data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
          Shape2(ishape.ProdShape(0, ishape.ndim()-1), ishape[ishape.ndim()-1]), s);
      out = out_data[fullc::kOut].get_with_shape<xpu, 2, DType>(
          Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]), s);
    } else {
      data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
          Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
      out = out_data[fullc::kOut].get_with_shape<xpu, 2, DType>(
          Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);
    }

    // Legacy approach shown here for comparison:
    //   out = dot(data, wmat.T());
    linalg_gemm(data, wmat, out, false, true, s);
    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> bias = in_data[fullc::kBias].get<xpu, 1, DType>(s);
      // Legacy approach shown here for comparison:
      //   out += mshadow::expr::repmat(bias, data.size(0));
#if 1
      out += mshadow::expr::repmat(bias, data.size(0));
#else
      CHECK_EQ(data.size(0), out.size(0));  // check same row count in and out
      AddBias(s, out, data, bias);
#endif
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    // TODO(bing): check the BLAS Handle, be careful
    //  maybe need blas handle from context
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const TShape& ishape = in_data[fullc::kData].shape_;
    const TShape& oshape = out_grad[fullc::kOut].shape_;

    Tensor<xpu, 2, DType> wmat = in_data[fullc::kWeight].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> data, grad, gdata;
    if (!param_.flatten) {
      data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
          Shape2(ishape.ProdShape(0, ishape.ndim()-1), ishape[ishape.ndim()-1]), s);
      grad = out_grad[fullc::kOut].get_with_shape<xpu, 2, DType>(
          Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]), s);
      gdata = in_grad[fullc::kData].get_with_shape<xpu, 2, DType>(
          Shape2(ishape.ProdShape(0, ishape.ndim()-1), ishape[ishape.ndim()-1]), s);
    } else {
      data = in_data[fullc::kData].get_with_shape<xpu, 2, DType>(
          Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
      grad = out_grad[fullc::kOut].get_with_shape<xpu, 2, DType>(
          Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);
      gdata = in_grad[fullc::kData].get_with_shape<xpu, 2, DType>(
          Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
    }

#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    //  backprop
    CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
    // gradient of weight
    Tensor<xpu, 2, DType> gwmat = in_grad[fullc::kWeight].get<xpu, 2, DType>(s);
    // Legacy approach shown here for comparison:
    //   out = Assign(gwmat, req[fullc::kWeight], dot(grad.T(), data));
    linalg_gemm(grad, data, gwmat, true, false, s, req[fullc::kWeight]);
    // gradient of bias
    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> gbias = in_grad[fullc::kBias].get<xpu, 1, DType>(s);
      Assign(gbias, req[fullc::kBias], sum_rows(grad));
    }
    // gradient of data
    // Legacy approach shown here for comparison:
    //   Assign(gdata, req[fullc::kData], dot(grad, wmat));
    linalg_gemm(grad, wmat, gdata, false, false, s, req[fullc::kData]);
  }

 private:
#if defined(__CUDACC__)
  /*!
   * \brief Add bias on GPU using legacy mshadow
   * \param s - Stream (used to select which function)
   * \param out - Output data
   * \param input_data - Input data
   * \param bias - The bias to be broadcast
   */
  MSHADOW_CINLINE static void AddBias(Stream<gpu> *s,
                                      Tensor<gpu, 2, DType>& out,
                                      const Tensor<gpu, 2, DType>& input_data,
                                      const Tensor<gpu, 1, DType>& bias) {
    out += mshadow::expr::repmat(bias, input_data.size(0));
  }
#endif  // defined(__CUDACC__)
  /*!
   * \brief Add bias on CPU with cache-friendly memory access
   * \param s - Stream (used to select which function)
   * \param out - Output data
   * \param input_data - Input data
   * \param bias - The bias to be broadcast
   */
  MSHADOW_CINLINE static void AddBias(Stream<cpu> *s,
                                      Tensor<cpu, 2, DType>& out,
                                      const Tensor<cpu, 2, DType>& input_data,
                                      const Tensor<cpu, 1, DType>& bias) {
    const size_t out_size = out.shape_.Size();
    const size_t bias_size = bias.shape_.Size();
    if(out_size != bias_size) {
      const int row_count = out.shape_[0];
      const size_t gap = out_size / row_count;
      CHECK_EQ(gap, bias_size);
      DType *outp = out.dptr_;
      #pragma omp parallel for num_threads(Engine::Get()->num_omp_threads_per_worker())
      for (int r = 0; r < row_count; ++r) {
        const DType *in_ptr = bias.dptr_;
        DType *out_ptr = outp + r * gap;
        for (size_t i = 0; i < gap; ++i, ++out_ptr, ++in_ptr) {
          *out_ptr += *in_ptr;
        }
      }
    } else {
      mxnet_op::Kernel<mxnet_op::op_with_req<mshadow::op::plus, kWriteInplace>, cpu>::
      Launch(s, out.shape_.Size(), out.dptr_, out.dptr_, bias.dptr_);
    }
  }

  /*! \brief Parameters for this fully-connected layer */
  FullyConnectedParam param_;
};  // class FullyConnectedOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(FullyConnectedParam param, int dtype,
                   std::vector<TShape> *in_shape,
                   std::vector<TShape> *out_shape,
                   Context ctx);

#if DMLC_USE_CXX11
class FullyConnectedProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return {"data", "weight", "bias"};
    } else {
      return {"data", "weight"};
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
    }
    CHECK_EQ(out_shape->size(), 1U);
    TShape dshape = (*in_shape)[fullc::kData];
    TShape oshape = (*out_shape)[0];
    // require data to be known
    if (dshape.ndim() ==  0) return false;

    index_t num_input;
    if (!param_.flatten) {
      num_input = dshape[dshape.ndim()-1];
    } else {
      num_input = dshape.ProdShape(1, dshape.ndim());
    }
    SHAPE_ASSIGN_CHECK(*in_shape, fullc::kWeight, Shape2(param_.num_hidden, num_input));
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, fullc::kBias, Shape1(param_.num_hidden));
    }

    if (!param_.flatten) {
      TShape result_shape(dshape);
      result_shape[dshape.ndim()-1] = param_.num_hidden;
      SHAPE_ASSIGN_CHECK(*out_shape, 0, result_shape);
    } else {
      SHAPE_ASSIGN_CHECK(*out_shape, 0, Shape2(dshape[0], param_.num_hidden));
    }
    if (oshape.ndim() != 0) {
      dshape[0] = oshape[0];
      SHAPE_ASSIGN_CHECK(*in_shape, fullc::kData, dshape);
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    nnvm::NodeAttrs attrs;
    attrs.name = "FullyConnected";
    return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
      attrs, in_type, out_type, -1);
  }

  OperatorProperty* Copy() const override {
    auto* fc_sym = new FullyConnectedProp();
    fc_sym->param_ = this->param_;
    return fc_sym;
  }

  std::string TypeString() const override {
    return "FullyConnected";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[fullc::kOut], in_data[fullc::kData], in_data[fullc::kWeight]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{in_data[fullc::kData], in_grad[fullc::kData]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  FullyConnectedParam param_;
};  // class FullyConnectedSymbol
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_FULLY_CONNECTED_INL_H_
