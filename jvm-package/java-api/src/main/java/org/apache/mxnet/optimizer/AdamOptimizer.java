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
 * This class implements the optimizer described in *Adam: A Method for Stochastic Optimization*,
 * available at http://arxiv.org/abs/1412.6980.
 */
public class AdamOptimizer extends AbstractOptimizer<AdamOptimizer.AdamState> {

  static class AdamState extends AbstractState {
    private final NDArray mean;
    private final NDArray var;

    AdamState(Optimizer optimizer, NDArray weight, NDArray grad, NDArray mean, NDArray var) {
      super(optimizer, weight, grad);
      this.mean = mean;
      this.var = var;
    }
  }

  private final float beta1;
  private final float beta2;
  private final Float epsilon;

  /**
   * Create a new AdamOptimizer with the given parameters.
   *
   * @param learningRate The initial learning rate, default for each state
   * @param beta1 Exponential decay rate for the first moment estimates
   * @param beta2 Exponential decay rate for the second moment estimates
   * @param epsilon Small value to avoid division by 0
   * @param weightDecay
   */
  public AdamOptimizer(float learningRate, Float beta1, Float beta2, Float epsilon, Float weightDecay,
                       Float gradScale, Float clipGradient) {
    super(learningRate, weightDecay, gradScale, clipGradient);
    this.beta1 = beta1 != null ? beta1 : 0.9f;
    this.beta2 = beta2 != null ? beta2 : 0.999f;;
    this.epsilon = epsilon;

    // TODO: It should be possible to set the begin_update (initial update count value),
    //       states should be initialized to it
  }

  /**
   * Create a new AdamOptimizer with the given parameters.
   *
   * @param learningRate The initial learning rate, default for each state
   */
  public AdamOptimizer(float learningRate) {
    this(learningRate,null, null, null, null, null, null);
  }

  @Override
  public State createState(NDArray weight, NDArray grad) {
    NDArray mean = NDArray.zeros(weight.getContext(), weight.getDType(), weight.getShape());
    NDArray var = NDArray.zeros(weight.getContext(), weight.getDType(), weight.getShape());
    return new AdamState(this, weight, grad, mean, var);
  }

  @Override
  public void update(AdamState state) {
    float coef1 = 1 - (float) Math.pow(beta1, state.getUpdateCount());
    float coef2 = 1 - (float) Math.pow(beta2, state.getUpdateCount());

    float updateLearningRate = state.getLearningRate(learningRate)
        * ((float) Math.sqrt(coef2)) / coef1;

    Map<String, String> params = new HashMap<>();
    params.put("lr", Float.toString(updateLearningRate));
    params.put("wd", Float.toString(state.getWeightDecay(weightDecay)));
    params.put("beta1", Float.toString(beta1));
    params.put("beta2", Float.toString(beta2));

    if (epsilon != null) {
      params.put("epsilon", epsilon.toString());
    }

    if (rescaleGrad != null) {
      params.put("rescale_grad", rescaleGrad.toString());
    }

    if (clipGradient != null) {
      params.put("clip_gradient", clipGradient.toString());
    }

    NDArray.imperativeInvoke("adam_update",
        new NDArray[] {state.weight, state.grad, state.mean, state.var},
        new NDArray[] {state.weight}, params);
  }
}
