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
 * The AdaDelta optimizer.
 *
 * This class implements AdaDelta, an optimizer described in
 * *ADADELTA: An adaptive learning rate method*, available at https://arxiv.org/abs/1212.5701.
 */
public class AdaDeltaOptimizer extends AbstractOptimizer<AdaDeltaOptimizer.AdaState>  {

  static class AdaState extends AbstractState {
    private final NDArray g;
    private final NDArray delta;

    AdaState(Optimizer optimizer, NDArray weights, NDArray grad, NDArray g, NDArray delta) {
      super(optimizer, weights, grad);
      this.g = g;
      this.delta = delta;
    }
  }

  private static final float RHO_DEFAULT = 0.9f;
  private static final float EPSILON_DEFAULT = 1e-5f;

  private final float rho;
  private final float epsilion;

  public AdaDeltaOptimizer(float learningRate, float weightDecay,
                           Float rho, Float epsilion, Float rescaleGrad, Float clipGradient) {
    super(learningRate, weightDecay, rescaleGrad, clipGradient);
    this.rho = rho != null ? rho : RHO_DEFAULT;
    this.epsilion = epsilion != null ? epsilion : EPSILON_DEFAULT;
  }

  @Override
  public Optimizer.State createState(NDArray weight, NDArray grad) {
    NDArray g = NDArray.zeros(weight.getContext(), null, weight.getShape());
    NDArray delta = NDArray.zeros(weight.getContext(), null, weight.getShape());
    return new AdaState(this, weight, grad, g, delta);
  }

  // TODO: Implement this
  @Override
  public void update(AdaState state) {

    // grad *= self.rescaleGrad
    if (rescaleGrad != null) {
      state.grad.muli(rescaleGrad);
    }

    // if self.clip_gradient is not None:
    // grad = clip(grad, -self.clip_gradient, self.clip_gradient)

    if (clipGradient != null) {
      // TODO: All allocated arrrys, used not in-place need to be cleared up ...
      NDArray.clip(state.grad, -clipGradient, clipGradient); // TODO: Is this performed in place ?!?!?
    }

    // acc_g[:] = self.rho * acc_g + (1. - self.rho) * grad * grad

    // acc_g.muli(rho) + (1 - rho) * grad * grad

    // current_delta = sqrt(acc_delta + self.epsilon) / sqrt(acc_g + self.epsilon) * grad
    // acc_delta[:] = self.rho * acc_delta + (1. - self.rho) * current_delta * current_delta

    // update weight
    // weight[:] -= current_delta + wd * weight
  }
}
