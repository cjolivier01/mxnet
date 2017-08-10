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

package org.apache.mxnet;

import org.apache.mxnet.javacpp.mxnet;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.PointerPointer;

/**
 * Executor is the object providing efficient symbolic graph execution and optimization.
 *
 * Note: Use Symbol.bind to instantiate a new Executor instance.
 */
@SuppressWarnings("unused")
public class Executor {

  private mxnet.ExecutorHandle handle;

  Executor(mxnet.ExecutorHandle handle) {
    this.handle = handle;
  }

  private void verify(int status) {
    if (handle == null) {
      throw new IllegalStateException("Executor was disposed!");
    }

    Util.check(status);
  }

  private static int toInt(boolean value) {
    return value ? 0 : 1; // TODO: this looks wrong !!!!
  }

  mxnet.ExecutorHandle getHandle() {
    return handle;
  }

  /**
   * Install callback for monitor.
   */
  public void setMonitorCallback() {
//    mxnet.ExecutorMonitorCallback

    // TODO: Implement

  }

  /**
   * Calculate the outputs specified by the bound symbol.
   *
   * @param isTrain whether this forward is for evaluation purpose.
   *                If True, a backward call is expected to follow.
   */
  public void forward(boolean isTrain) {
    // TODO: Create a Util.toInt() method to convert boolean to integer
    // TODO: toInt is inverted !!!!!!
    verify(mxnet.MXExecutorForward(handle, toInt(!isTrain)));
  }

  /**
   * Calculate the outputs specified by the bound symbol.
   */
  public void forward() {
    forward(false);
  }


  /**
   * Do backward pass to get the gradient of arguments.
   *
   * @param grads Gradient on the outputs to be propagated back. This parameter is only needed
   *              when bind is called on outputs that are not a loss function.
   * @param isTrain whether this backward is for training or inference. Note that in rare cases
   *                you want to call backward with is_train=False to get gradient during inference.
   */
  public void backward(NDArray[] grads,  boolean isTrain) {
    mxnet.NDArrayHandle[] gradsHandles = new mxnet.NDArrayHandle[grads != null ? grads.length : 0];
    for (int i = 0; i < gradsHandles.length; i++) {
      gradsHandles[i] = grads[i].getHandle();
    }

    // TODO: toInt is inverted !!!!!!
    verify(mxnet.MXExecutorBackwardEx(handle,  gradsHandles.length,
        new PointerPointer(gradsHandles), toInt(!isTrain)));
  }

  /**
   * Do backward pass to get the gradient of arguments.
   */
  public void backward() {
    backward(null, true);
  }

  /**
   * List all the output NDArray.
   *
   * @return a list of ndarray bound to the heads of executor.
   */
  public NDArray[] output() {
  // TODO: Should this fail if forward was not called ?!?! and others too
    IntPointer outSize = new IntPointer(1);

    PointerPointer outputs = new PointerPointer();

    verify(mxnet.MXExecutorOutputs(handle, outSize, outputs));

    NDArray[] outputArrays = new NDArray[outSize.get()];
    for (int i = 0; i < outputArrays.length; i++) {
      outputArrays[i] = new NDArray(new mxnet.NDArrayHandle(outputs.get(i)));
    }

    return outputArrays;
  }

  // TODO: Add missing methods ... compare to python executor.py

  public void dispose() {
    Util.check(mxnet.MXExecutorFree(handle));
    handle = null;
  }

  /**
   * Debug string of the executor.
   *
   * @return the content of the execution plan as debug string.
   */
  @Override
  public String toString() {
    if (handle != null) {
      PointerPointer debugString = new PointerPointer(1);
      Util.check(mxnet.MXExecutorPrint(handle, debugString));
      return debugString.getString(0);
    }
    else {
      return "Executor not bound to handle!";
    }
  }
}
