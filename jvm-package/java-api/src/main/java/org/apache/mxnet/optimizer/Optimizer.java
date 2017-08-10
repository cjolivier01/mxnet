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

package org.apache.mxnet.optimizer;

import org.apache.mxnet.NDArray;

/**
 * The base interface inherited by all optimizer implementations.
 */
// TODO: Add implementation for more optimizers ...
public interface Optimizer {

  /**
   * The per weight state of the optimizer.
   */
  interface State {
    void setLearningRateMultiplier(float lr);
    float getLearningRateMultiplier();

    void setWeightDecayMultiplier(float wd);
    float getWeightDecayMultiplier();

    int getUpdateCount();
  }

  /**
   * Some optimizers require additional states, e.g as momentum, in addition to gradients in
   * order to update weights. This function creates state for a given weight which will be used
   * in `update`. This function is called only once for each weight.
   *
   * <p>
   *
   * Note: The state is bound to the optimizer which created it and the weight it was created for.
   *
   * @param weight The weight
   * @return
   */
  State createState(NDArray weight, NDArray grad);

  /**
   * Updates the given parameter using the corresponding gradient and state.
   *
   * @param state
   */
  void update(State state);
}
