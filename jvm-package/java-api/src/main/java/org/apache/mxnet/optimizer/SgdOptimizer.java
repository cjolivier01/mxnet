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

import java.util.HashMap;
import java.util.Map;

import org.apache.mxnet.NDArray;

/**
 * The SGD optimizer with momentum and weight decay.
 */
// TODO: Add support for multi precision
public class SgdOptimizer extends AbstractOptimizer<SgdOptimizer.SgdState> {

  static class SgdState extends AbstractState {
    private final NDArray momentum;

    SgdState(Optimizer optimizer, NDArray weights, NDArray grad, NDArray momentum) {
      super(optimizer, weights, grad);
      this.momentum = momentum;
    }
  }

  private final float momentum;

  /**
   *
   * @param learningRate
   * @param weightDecay
   * @param momentum
   * @param rescaleGrad Rescale gradient to grad = rescale_grad*grad
   * @param clipGradient Clip gradient to the range of [-clip_gradient, clip_gradient]
   *                     If clip_gradient <= 0, gradient clipping is turned off.
   *                     grad = max(min(grad, clip_gradient), -clip_gradient).
   */
  public SgdOptimizer(float learningRate, Float weightDecay, Float momentum, Float rescaleGrad,
                      Float clipGradient) {
    super(learningRate, weightDecay, rescaleGrad, clipGradient);
    this.momentum = momentum;
  }

  @Override
  public State createState(NDArray weight, NDArray grad) {
    if (momentum != 0) {
      return new SgdState(this, weight, grad,
          NDArray.zeros(weight.getContext(), weight.getDType(), weight.getShape()));
    }
    else {
      return new SgdState(this, weight, grad, null);
    }
  }

  @Override
  public void update(SgdState state) {

    Map<String, String> params = new HashMap<>();
    params.put("lr", Float.toString(state.getLearningRate(learningRate)));
    params.put("wd", Float.toString(state.getWeightDecay(weightDecay)));

    if (rescaleGrad != null) {
      params.put("rescale_grad", Float.toString(rescaleGrad));
    }

    if (clipGradient != null) {
      params.put("clip_gradient", Float.toString(clipGradient));
    }

    if (state.momentum != null) {
      NDArray.imperativeInvoke("sgd_mom_update", new NDArray[] {state.weight, state.grad, state.momentum},
          new NDArray[] {state.weight}, params);
    }
    else {
      NDArray.imperativeInvoke("sgd_update", new NDArray[] {state.weight, state.grad},
          new NDArray[] {state.weight}, params);
    }
  }
}
