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

// TODO: Casting of states should be done here!
abstract class AbstractOptimizer<T extends AbstractState> implements Optimizer  {

  protected final float learningRate;
  protected final float weightDecay;
  protected final Float rescaleGrad;
  protected final Float clipGradient;

  AbstractOptimizer(float learningRate, Float weightDecay, Float resacleGrad, Float clipGradient) {
    this.learningRate = learningRate;
    this.weightDecay = weightDecay != null ? weightDecay : 0;
    this.rescaleGrad = resacleGrad;
    this.clipGradient = clipGradient;
  }

  abstract void update(T state);

  @Override
  public void update(State state) {
    if (!(state instanceof AbstractState)) {
      throw new IllegalArgumentException("state was not created by this optimizer");
    }

    T tState = (T) state;
    tState.verifyState(this);
    tState.incrementUpdateCount();

    update(tState);
  }
}
