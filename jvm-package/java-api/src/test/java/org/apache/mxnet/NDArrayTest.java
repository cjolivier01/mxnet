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

import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;

public class NDArrayTest {

  static double DELTA = 0.0000001;

  @Test
  public void getShape() {
    NDArray emptyArray = NDArray.empty(null, null, 2, 3);

    emptyArray.getShape();
  }

  @Test
  public void getContext() {
    NDArray emptyArray = NDArray.empty(null, null, 2, 3);

    Context ctx = emptyArray.getContext();

    Assert.assertEquals(1, ctx.getDeviceType());
    Assert.assertEquals(0, ctx.getDeviceId());
  }

  @Test
  public void getDType() {
    NDArray emptyArray = NDArray.empty(null, null, 2, 3);

    emptyArray.getDType();
  }

  @Test
  public void copyTo() {
    NDArray a = NDArray.full(null, null, 3, 4);
    NDArray b = NDArray.zeros(null, null, 4);

    float[] expected = new float[] {3, 3, 3, 3};
    Assert.assertArrayEquals(expected, a.toArray(), (float) DELTA);

    Assert.assertArrayEquals(new float[] {0, 0, 0, 0}, b.toArray(), (float) DELTA);

    a.copyTo(b);

    Assert.assertArrayEquals(expected, a.toArray(), (float) DELTA);
    Assert.assertArrayEquals(expected, b.toArray(), (float) DELTA);
  }

  @Test
  public void copy() {
    NDArray a = NDArray.full(null, null, 3, 4);
    NDArray b = a.copy();

    float[] expected = new float[] {3, 3, 3, 3};
    Assert.assertArrayEquals(expected, a.toArray(), (float) DELTA);
    Assert.assertArrayEquals(expected, b.toArray(), (float) DELTA);

    b.addi(1);

    Assert.assertArrayEquals(expected, a.toArray(), (float) DELTA);
    Assert.assertArrayEquals(new float[] {4, 4, 4, 4}, b.toArray(), (float) DELTA);
  }

  @Test
  public void asInContext() {
    // TODO: Implement
  }

  @Test
  public void toArray() {
    NDArray sevenVector = NDArray.full(null, null, 7, 7);

    float[] sevens = sevenVector.toArray();

    Assert.assertEquals(7, sevens.length);

    for (float seven : sevens) {
      Assert.assertEquals(7, seven, DELTA);
    }
  }

  @Test
  public void asScalar() {
    NDArray sevenVector = NDArray.full(null, null, 7, 1);
    Assert.assertEquals(7, sevenVector.asScalar(), DELTA);
    sevenVector.dispose();
  }

  /**
   * Returns a copy of the array after casting to a specified type.
   */
  public void asType() {
    // TODO: Implement
  }

  // Array change shape

  public void  T() {
    // write test based on sample

    // x = mx.nd.arange(0,6).reshape((2,3))
    // x.asnumpy()
    // array([[ 0.,  1.,  2.],
    //   [ 3.,  4.,  5.]], dtype=float32)

  }

  // In-place arithmetic operations

  @Test
  public void addi() {
    NDArray sevenArray = NDArray.full(null, null, 7, 7);

    sevenArray.addi(7);

    for (float v : sevenArray.toArray()) {
      Assert.assertEquals(14, v, DELTA);
    }

    sevenArray.dispose();
  }

  @Test
  public void subi() {
    NDArray sevenArray = NDArray.full(null, null, 7, 7);

    sevenArray.subi(7);

    for (float v : sevenArray.toArray()) {
      Assert.assertEquals(0, v, DELTA);
    }

    sevenArray.dispose();
  }

  @Test
  public void muli() {
    NDArray sevenArray = NDArray.full(null, null, 7, 7);

    sevenArray.muli(7);

    for (float v : sevenArray.toArray()) {
      Assert.assertEquals(49, v, DELTA);
    }

    sevenArray.dispose();
  }

  @Test
  public void divi() {
    NDArray sevenArray = NDArray.full(null, null, 7, 7);

    sevenArray.divi(7);

    for (float v : sevenArray.toArray()) {
      Assert.assertEquals(1, v, DELTA);
    }

    sevenArray.dispose();
  }

  @Test
  public void modi() {
    NDArray sevenArray = NDArray.full(null, null, 7, 7);

    sevenArray.modi(7);

    for (float v : sevenArray.toArray()) {
      Assert.assertEquals(0, v, DELTA);
    }

    sevenArray.dispose();
  }

  @Test
  public void reshape() {
    // TODO: Implement this test ...
  }

  @Test
  public void empty() {
    NDArray emptyArray = NDArray.empty(null, null, 2, 3);

    Assert.assertEquals(6, emptyArray.size());

    emptyArray.dispose();
  }

  @Test
  public void array() {
    float[] source = new float[] {1, 2, 3, 4, 5, 6};

    NDArray array = NDArray.array(null, null, source, 2, 3);

    Assert.assertArrayEquals(new int[]{2,3}, array.getShape());
    Assert.assertArrayEquals(source, array.toArray(), (float) DELTA);
  }

  @Test
  public void zeros() {
    NDArray zerosArray = NDArray.zeros(null, null, 2, 3);

    float[] zeros  = zerosArray.toArray();

    Assert.assertEquals(6, zeros.length);

    for (float zero : zeros) {
      Assert.assertEquals(0, zero, DELTA);
    }

    zerosArray.dispose();
  }

  @Test
  public void ones() {
    NDArray onesArray = NDArray.ones(null, null, 2, 3);

    float[] ones = onesArray.toArray();

    for (float one : ones) {
      Assert.assertEquals(1, one, DELTA);
    }

    onesArray.dispose();
  }

  @Test
  public void arange() {
    NDArray rangeArray =
        NDArray.arange(null, null, 1, 2, 2, 7);

    Assert.assertArrayEquals(new float[] {1, 1, 3, 3, 5, 5},
        rangeArray.toArray(), (float) DELTA);
  }

  @Test
  public  void load() throws IOException  {
    // TODO: Implement
  }

  @Test
  public void save() throws IOException {
    // TODO: Implement
  }



  // Changing array shape and type

  @Test
  public  void cast() {
    NDArray array = NDArray.ones(null, DType.INT_32, 3);

    Assert.assertEquals(DType.INT_32,  array.getDType());
    NDArray.cast(array, DType.FLOAT_32);
    Assert.assertEquals(DType.FLOAT_32,  array.getDType());


    // TODO: Verify the content ...
  }
  
  @Test
  public void testFlatten() {
    // TODO: Implement
  }

  @Test
  public void expandDims() {
    // TODO: Implement
  }

  // Expanding array elements

  @Test
  public void broadcastTo() {
    NDArray array = NDArray.array(null, null, new float[]{1,2,3},1,3);
    NDArray result = NDArray.broadcastTo(array, 2,3);
    Assert.assertArrayEquals(new int[]{2,3}, result.getShape());
  }

  @Test
  public void broadcastAxis() {
    NDArray array = NDArray.array(null, null, new float[]{1,2},1,2,1);
    NDArray resultA = NDArray.broadcastAxis(array, new int[] {3},2);
    Assert.assertArrayEquals(new int[]{1,2,3}, resultA.getShape());

    NDArray resultB = NDArray.broadcastAxis(array, new int[] {2,3},0,2);
    Assert.assertArrayEquals(new int[]{2, 2,3}, resultB.getShape());
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

  // Rearranging elements

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



  // Joining and splitting arrays

  @Test
  public void concat() {
    // TODO: Implement
  }

  @Test
  public void split() {
    // TODO: Implement
  }



  // Indexing routines

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
  public void oneHot() {
    // TODO: Implement
  }

  @Test
  public void pick() {
    // TODO: Implement
  }



  // Arithmetic operations

  @Test
  public void add() {
    // TODO: Implement
  }

  @Test
  public void subtract() {
    // TODO: Implement
  }

  @Test
  public void negative() {
    // TODO: Implement
  }

  @Test
  public void multiply() {
    // TODO: Implement
  }

  @Test
  public void divide() {
    // TODO: Implement
  }

  @Test
  public void modulo() {
    // TODO: Implement
  }

  @Test
  public void dot() {
    // TODO: Implement
  }

  @Test
  public void batchDot() {
    // TODO: Implement
  }

  @Test
  public void addN() {
    // TODO: Implement
  }



  // Trigonometric functions

  @Test
  public void sin() {
    NDArray piArray = NDArray.full(null, null, (float) Math.PI, 7);

    NDArray zeroArray = NDArray.sin(piArray);

    for (float v : zeroArray.toArray()) {
      Assert.assertEquals(0, v, DELTA);
    }
  }

  @Test
  public void cos() {
    NDArray piArray = NDArray.full(null, null, (float) Math.PI, 7);

    NDArray zeroArray = NDArray.cos(piArray);

    for (float v : zeroArray.toArray()) {
      Assert.assertEquals(-1, v, DELTA);
    }
  }

  @Test
  public void tan() {
    // TODO: Implement
  }

  @Test
  public void arcsin() {
    // TODO: Implement
  }

  @Test
  public void arccos() {
    // TODO: Implement
  }

  @Test
  public void arctan() {
    // TODO: Implement
  }

  @Test
  public void degrees() {
    // TODO: Implement
  }

  @Test
  public void radians() {
    // TODO: Implement
  }



  // Hyperbolic functions

  @Test
  public void sinh() {
    // TODO: Implement
  }

  @Test
  public void cosh() {
    // TODO: Implement
  }

  @Test
  public void tanh() {
    // TODO: Implement
  }

  @Test
  public void arcsinh() {
    // TODO: Implement
  }

  @Test
  public void arccosh() {
    // TODO: Implement
  }

  @Test
  public void arctanh() {
    // TODO: Implement
  }



  // Reduce functions

  @Test
  public void sum() {
    // TODO: Implement
  }

  @Test
  public  void nansum() {
    // TODO: Implement
  }

  @Test
  public  void prod() {
    // TODO: Implement
  }

  @Test
  public  void nanprod() {
    // TODO: Implement
  }

  @Test
  public  void mean() {

    NDArray ones = NDArray.ones(null, null, 7,2);

    NDArray.mean(ones, true, false, 0);

    // TODO: Implement
  }

  @Test
  public  void max() {
    // TODO: Implement
  }

  @Test
  public  void min() {
    // TODO: Implement
  }

  @Test
  public  void norm() {
    // TODO: Implement
  }


  // Rounding

  @Test
  public  void round() {
    // TODO: Implement
  }

  @Test
  public  void rint() {
    // TODO: Implement
  }

  @Test
  public  void fix() {
    // TODO: Implement
  }

  @Test
  public  void floor() {
    // TODO: Implement
  }

  @Test
  public  void ceil() {
    // TODO: Implement
  }

  @Test
  public  void trunc() {
    // TODO: Implement
  }


  // Exponents and logarithms

  @Test
  public  void exp() {
    // TODO: Implement
  }

  @Test
  public  void expm1() {
    // TODO: Implement
  }

  @Test
  public  void log() {
    // TODO: Implement
  }

  @Test
  public  void log10() {
    // TODO: Implement
  }

  @Test
  public  void log2() {
    // TODO: Implement
  }

  @Test
  public  void log1p() {
    // TODO: Implement
  }

  // Powers

  @Test
  public  void power() {
    // TODO: Implement
  }

  @Test
  public void sqrt() {
    // TODO: Implement
  }

  @Test
  public void rsqrt() {
    // TODO: Implement
  }

  @Test
  public void square() {
    // TODO: Implement
  }

  // Logic functions

  @Test
  public void equal() {
    // TODO: Implement
  }

  @Test
  public void not_equal() {
    // TODO: Implement
  }

  @Test
  public void greater() {
    // TODO: Implement
  }

  @Test
  public void greater_equal() {
    // TODO: Implement
  }

  @Test
  public void lesser() {
    // TODO: Implement
  }

  @Test
  public void lesser_equal() {
    // TODO: Implement
  }



  // Random sampling

  @Test
  public  void random_uniform() {
    // TODO: Implement
  }

  @Test
  public  void random_normal() {
    // TODO: Implement
  }

  @Test
  public  void random_gamma() {
    // TODO: Implement
  }

  @Test
  public  void random_exponential() {
    // TODO: Implement
  }

  @Test
  public  void random_poisson() {
    // TODO: Implement
  }

  @Test
  public  void random_negative_binomial() {
    // TODO: Implement
  }

  @Test
  public  void random_generalized_negative_binomial() {
    // TODO: Implement
  }

  // TODO:
  // Seeds the random number generators in MXNet.
  // public static void mxnet.random.seed



  // Sorting and searching

  @Test
  public  void sort() {
    // TODO: Implement
  }

  @Test
  public  void testTopk() {
    // TODO: Implement
  }

  @Test
  public  void testArgsort() {
    // TODO: Implement
  }

  @Test
  public void testArgmax() {
    // TODO: Implement
  }

  @Test
  public void testArgmin() {
    // TODO: Implement
  }



  // Miscellaneous
  @Test
  public  void maximum() {
    // TODO: Implement
  }

  @Test
  public  void minimum() {
    // TODO: Implement
  }

  @Test
  public  void clip() {
    // TODO: Implement
  }

  @Test
  public  void abs() {
    // TODO: Implement
  }

  @Test
  public  void sign() {
    // TODO: Implement
  }

  @Test
  public  void gamma() {
    // TODO: Implement
  }

  @Test
  public  void gammaln() {
    // TODO: Implement
  }


  // Neural network - Basic

  @Test
  public  void fullyConnected() {
    // TODO: Implement
  }

  @Test
  public  void convolution() {
    // TODO: Implement
  }

  @Test
  public  void Activation() {
    // TODO: Implement
  }

  @Test
  public  void batchNorm() {
    // TODO: Implement
  }

  @Test
  public  void pooling() {
    // TODO: Implement
  }

  @Test
  public  void softmaxOutput() {
    // TODO: Implement
  }

  @Test
  public  void softmax() {
    // TODO: Implement
  }

  @Test
  public  void logSoftmax() {
    // TODO: Implement
  }


  // Neural network - More
  @Test
  public  void correlation() {
    // TODO: Implement
  }

  @Test
  public  void deconvolution() {
    // TODO: Implement
  }

  @Test
  public  void rnn() {
    // TODO: Implement
  }

  @Test
  public  void embedding() {
    // TODO: Implement
  }

  @Test
  public  void leakyReLU() {
    // TODO: Implement
  }

  @Test
  public  void testinstanceNorm() {
    // TODO: Implement
  }

  @Test
  public  void l2Normalization() {
    // TODO: Implement
  }

  @Test
  public  void lrn() {
    // TODO: Implement
  }

  @Test
  public  void roiPooling() {
    // TODO: Implement
  }

  @Test
  public  void softmaxActivation() {
    // TODO: Implement
  }

  @Test
  public  void dropout() {
    // TODO: Implement
  }

  @Test
  public  void bilinearSampler() {
    // TODO: Implement
  }

  @Test
  public  void gridGenerator() {
    // TODO: Implement
  }

  @Test
  public  void upSampling() {
    // TODO: Implement
  }

  @Test
  public  void spatialTransformer() {
    // TODO: Implement
  }

  @Test
  public  void linearRegressionOutput() {
    // TODO: Implement
  }

  @Test
  public  void logisticRegressionOutput() {
    // TODO: Implement
  }

  @Test
  public  void maeRegressionOutput() {
    // TODO: Implement
  }

  @Test
  public void svmOutput() {
    // TODO: Implement
  }

  @Test
  public void softmaxCrossEntropy() {
    // TODO: Implement
  }

  @Test
  public void smoothL1() {
    // TODO: Implement
  }

  @Test
  public void identityAttachKLSparseReg() {
    // TODO: Implement
  }

  @Test
  public void makeLoss() {
    // TODO: Implement
  }

  @Test
  public void blockGrad() {
    // TODO: Implement
  }

  @Test
  public void custom() {
    // TODO: Implement
  }

  // TODO: Remove this method, and listFunctions
  @Test
  public void listFunctions() {

    NDArray.listFunctions();
  }
}
