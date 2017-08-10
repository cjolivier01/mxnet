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

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;

public class SymbolTest {

  @Test
  public void listArguments() {
    Symbol a = Symbol.var("a");
    Assert.assertEquals("a", a.listArguments()[0]);
  }

  @Test
  public void listOutputs() {
    // TODO: Implement
  }

  @Test
  public void listAuxiliaryStates() {
    // TODO: Implement
  }

  @Test
  public void listAttr() {
    // TODO: Implement
  }

  public void attr() {
    // TODO: Implement
  }

  public void attrDict() {
    // TODO: Implement
  }

  @Test
  public void bind() {
    Symbol a = Symbol.var("a");

    Map<String, NDArray> grads = new HashMap<>();
    grads.put("a", NDArray.ones(null, null, 7));
    a.bind(Context.cpu(), grads);

    // TODO: Check if executor is ok ...
  }

  public void simpleBind() {
    // TODO: Implement
  }


  @Test
  public void var() {
    Symbol a = Symbol.var("a");
    Assert.assertEquals("a", a.name());
  }

  @Test
  public void zeros() {
    Symbol a = Symbol.zeros(null, 5);

    Executor executor = a.bind(Context.cpu(), Collections.<String, NDArray>emptyMap());

    NDArray[] outputs = executor.output();

    executor.forward();

    Assert.assertArrayEquals(new float[]{0, 0, 0, 0, 0}, outputs[0].toArray(),
        (float) NDArrayTest.DELTA);
  }

  @Test
  public void ones() {
    Symbol a = Symbol.ones(null, 5);

    Executor executor = a.bind(Context.cpu(), Collections.<String, NDArray>emptyMap());

    NDArray[] outputs = executor.output();

    executor.forward();

    Assert.assertArrayEquals(new float[]{1, 1, 1, 1, 1}, outputs[0].toArray(),
        (float) NDArrayTest.DELTA);
  }

  @Test
  public void arange() {
    Symbol range =
        Symbol.arange(null,  1, 2, 2, 7);

    Executor executor = range.bind(Context.cpu(), Collections.<String, NDArray>emptyMap());

    NDArray[] outputs = executor.output();

    executor.forward();

    Assert.assertArrayEquals(new float[]{1, 1, 3, 3, 5, 5}, outputs[0].toArray(),
        (float) NDArrayTest.DELTA);
  }

  @Test
  public void cast() {
    Symbol a = Symbol.var("a");

    Map<String, NDArray> values = new HashMap<>();

    NDArray aValue =  NDArray.ones(null, null, 5);
    aValue.addi(0.3f);
    values.put("a", aValue);

    Symbol b = Symbol.cast(a, DType.INT_32);
    Executor executor = b.bind(Context.cpu(), values);

    executor.forward();

    NDArray[] outputs = executor.output();

    Assert.assertArrayEquals(new float[]{1, 1, 1, 1, 1}, outputs[0].toArray(),
        (float) NDArrayTest.DELTA);
  }

  @Test
  public void reshape() {
    // TODO: Implement
  }

  @Test
  public void flatten() {
    // TODO: Implement
  }

  @Test
  public void expandDims() {
    // TODO: Implement
  }

  @Test
  public void broadcastTo() {
    // TODO: Implement
  }

  @Test
  public void broadcastAxes() {
    // TODO: Implement
  }

  @Test
  public void repeat() {
    // TODO: Implement
  }

  @Test
  public void tile() {
    // TODO: Implement
  }

  @Test
  public void pad() {
    // TODO: Implement
  }

  @Test
  public void transpose() {
    // TODO: Implement
  }

  @Test
  public void swapaxes() {
    // TODO: Implement
  }

  @Test
  public void flip() {
    // TODO: Implement
  }

  @Test
  public void split() {
    // TODO: Implement
  }

  @Test
  public void slice() {
    // TODO: Implement
  }

  @Test
  public void sliceAxis() {
    // TODO: Implement
  }

  @Test
  public void take() {
    // TODO: Implement
  }

  @Test
  public void batchTake() {
    // TODO: Implement
  }

  @Test
  public void oneHotEncode() {
    // TODO: Implement
  }

  @Test
  public void pick() {
    // TODO: Implement
  }

  @Test
  public void where() { // TODO: Missing in NDArray
    // TODO: Implement
  }




  @Test
  public void add() {
    Symbol a = Symbol.var("a");
    Symbol b = Symbol.var("b");

    Symbol c = a.add(b);

    Map<String, NDArray> values = new HashMap<>();
    values.put("a", NDArray.ones(null, null, 5));
    NDArray bValue =  NDArray.ones(null, null, 5);
    bValue.addi(2f);
    values.put("b", bValue);

    Executor executor = c.bind(Context.cpu(), values);

    executor.forward();

    NDArray[] outputs = executor.output();

    Assert.assertArrayEquals(new float[]{4, 4, 4, 4, 4}, outputs[0].toArray(),
        (float) NDArrayTest.DELTA);
  }

  @Test
  public void testToString() {
    Symbol.var("a").toString();
  }
}
