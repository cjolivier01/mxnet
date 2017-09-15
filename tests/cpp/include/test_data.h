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
#ifndef MXNET_TESTS_CPP_TEST_MODELS_H_
#define MXNET_TESTS_CPP_TEST_MODELS_H_

#include <stdint.h>
#include <cstdlib>

namespace mxnet {
namespace test {

/*! \brief Inception v3 with batch normalization model in json format */
extern const char *model_inception_v3_bn;
extern const char *model_inception_v3_bn_params_url;

extern const char *model_resnet162;
extern const char *model_resnet162_params_url;

/*! \brief Image of a cat */
extern const uint8_t test_image_cat_jpg[];
extern const size_t  test_image_cat_length;

}  // namespace test
}  // namespace mxnet

#endif  // MXNET_TESTS_CPP_TEST_MODELS_H_
