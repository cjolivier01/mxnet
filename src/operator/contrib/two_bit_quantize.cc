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
 * \file two_bit_quantize.cc
 * \brief registers quantize_2bit, dequantize_2bit
 * and create_2bit operators with nnvm
 */
#include "./two_bit_quantize-inl.h"

namespace mxnet {
namespace op {

int quantize_2bit::sgn_[2] = { -1, 1 };
uint8_t quantize_2bit::bits_[8] = { 0x80, 0x10, 0x08, 0x01, 0xc0, 0x30, 0x0c, 0x03 };


DMLC_REGISTER_PARAMETER(TwoBitParam);

NNVM_REGISTER_OP(_contrib_quantize_2bit)
.describe(R"code(Quantize an input tensor into using 2bits for each value using
user-specified thresholds, while storing quantization error in residual array.

The quantize_2bit operator takes 5 arguments and is called as follows:
`quantize_2bit(array, residual, out, neg_threshold, pos_threshold)`.
The operator modifies `residual` and `out` arrays.
The `out`variable will be the quantized array. Note that, `out` array can be generated by
invoking `create_2bit(array)`, avoiding calculation of size of quantized array.
This `out` array has first three elements as negative threshold, positive threshold,
and size of the original uncompressed array. Any elements after these three elements
represent quantized data.
The operation sums up array and residual, and then
applies the thresholds to quantize the data into one of three states
represented by 2bits. 16 such quantized floats in the original array
are packed together into one float in the `out` array.
The quantization error is stored in residual array.

For example, assume the input array (gradient) is [5.0, -1.0, -5.0, -4.0], and the
residual is [0.0, -2.0, 0, 1.0]. Let the negative and positive thresholds be
-4.0 and +4.0, respectively. In this method, the elements whose
(gradient + residual) >= pos_threshold will be quantized into 2-bits '01',
and the elements whose (gradient + residual) <= neg_threshold will be
quantized into 2-bits '10'. The other elements will be quantized
as '00'. Every 16 floats in the original array will be packed
into one float variable in the output array.

In this example, 'out' has 4 elements. The first element stores the
neg_threshold (-4.0), the second element stores the pos_threshold (+4.0), the
third element stores the original size of the uncompressed array, and the
original array will be quantized into a single element in the last element.
The residual is also updated to [1.0, -3.0, -1.0, -3.0].
)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(0)
.set_attr_parser(ParamParser<TwoBitParam>)
.set_attr<nnvm::FInferShape>("FInferShape", Quantize2BitShape)
.set_attr<nnvm::FInferType>("FInferType", Quantize2BitType)
.set_attr<FCompute>("FCompute<cpu>", Quantize2BitCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_quantize_2bit"})
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
[](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{1, 2};
})
.add_argument("gradient_array", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
.add_argument("residual_array", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
.add_argument("quantized_array", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
.add_arguments(TwoBitParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_create_2bit)
  .describe(R"code(Generate an array with the right shape to store the input data after
two bit quantization. This array will be on the same context as input array.
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", Create2BitArrayShape)
.set_attr<nnvm::FInferType>("FInferType", Create2BitArrayType)
.set_attr<FCompute>("FCompute<cpu>", Create2BitArrayCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_create_2bit"})
.add_argument("input", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`");

NNVM_REGISTER_OP(_contrib_dequantize_2bit)
.describe(R"code(Dequantize an input tensor quantized by quantize_2bit.

The dequantize_2bit operator takes two input arguments. The first input is a NDArray,
which has been generated by quantize_2bit(). This operator expects the first
three elements to be the negative threshold, positive threshold, and the size
of the original uncompressed array. Starting from the fourth element are expected to
be quantized values of the original array.
The second input is a NDArray that has the same shape as the original
array before quantizing. The operator replaces the contents of this array
with dequantized data.

In the example was described for quantize_2bit,
invoking dequantize_2bit(out, array), the 'array' argument will become
[4.0, 0, -4.0, 0], where -4.0 and 4.0 are the negative and positive thresholds.
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(0)
.set_attr<nnvm::FInferShape>("FInferShape", Dequantize2BitShape)
.set_attr<nnvm::FInferType>("FInferType", Dequantize2BitType)
.set_attr<FCompute>("FCompute<cpu>", Dequantize2BitCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_dequantize_2bit"})
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
[](const nnvm::NodeAttrs& attrs) {
  return std::vector<uint32_t>{1};
})
.add_argument("quantized_data", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`")
.add_argument("dequantized_data", "NDArray-or-Symbol", "A ndarray/symbol of type `float32`");
}  // namespace op
}  // namespace mxnet
