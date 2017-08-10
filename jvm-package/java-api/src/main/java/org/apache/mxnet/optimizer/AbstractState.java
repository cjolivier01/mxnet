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

import java.util.Objects;

import org.apache.mxnet.NDArray;

abstract class AbstractState implements Optimizer.State {

  private final Optimizer boundOptimizer;
  protected final NDArray weight;
  protected final NDArray grad;

  private float learningRateMultiplier = 1;
  private float weightDecayMultiplier = 1;

  private int updateCount;

  AbstractState(Optimizer optimizer, NDArray weight, NDArray grad) {
    boundOptimizer = Objects.requireNonNull(optimizer);
    this.weight = Objects.requireNonNull(weight);
    this.grad = Objects.requireNonNull(grad);
  }

  void verifyState(Optimizer optimizer) {
    if (boundOptimizer != optimizer) {
      throw new IllegalArgumentException("state was not created by this optimizer");
    }
  }

  @Override
  public void setLearningRateMultiplier(float lr) {
    learningRateMultiplier = lr;
  }

  @Override
  public float getLearningRateMultiplier() {
    return learningRateMultiplier;
  }

  /**
   * Gets the weighted learning rate for this state.
   *
   * @param learningRate
   * @return
   */
  float getLearningRate(float learningRate) {
    return getLearningRateMultiplier() * learningRate;
  }

  @Override
  public void setWeightDecayMultiplier(float wd) {
    weightDecayMultiplier = wd;
  }

  @Override
  public float getWeightDecayMultiplier() {
    return weightDecayMultiplier;
  }

  /**
   * Gets the weighted weight decay for this state.
   *
   * @param weightDecay
   * @return
   */
  float getWeightDecay(float weightDecay) {
    return getWeightDecayMultiplier() * weightDecay;
  }

  @Override
  public int getUpdateCount() {
    return updateCount;
  }

  void incrementUpdateCount() {
    updateCount++;
  }
}
