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
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import org.apache.mxnet.javacpp.mxnet;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;

/**
 * An NDArray represents a multi-dimensional, fixed-size homogenous array.
 */
// TODO: Many functions have option out parameter, how to handle that ?!?!?!
@SuppressWarnings("unused")
public class NDArray {

  private mxnet.NDArrayHandle handle;

  NDArray(mxnet.NDArrayHandle handle) {
    this.handle = Objects.requireNonNull(handle);
  }

  mxnet.NDArrayHandle getHandle() {
    if (handle == null) {
      throw new MXNetError("Symbol was disposed!");
    }

    return handle;
  }

  @Override
  protected void finalize() throws Throwable {
    super.finalize();
  }

  private void verify(int status) {
    if (getHandle() == null) {
      throw new IllegalStateException("NDArray was disposed!");
    }

    Util.check(status);
  }

  private static Context getOrDefault(Context ctx) {
    if (ctx == null) {
      return Context.cpu();
    }

    return ctx;
  }

  private static mxnet.FunctionHandle getFunction(String name) {
    mxnet.FunctionHandle setFunction = new mxnet.FunctionHandle();
    Util.check(mxnet.MXGetFunction(name, setFunction));

    if (setFunction.isNull()) {
      throw new IllegalArgumentException(String.format("Function [%s] does not exist!", name));
    }

    return setFunction;
  }

  // TODO: This should not be public, find some way to reduce visibility
  public static NDArray[] imperativeInvoke(String name, NDArray[] inputArrays,
                                                NDArray[] outputArrays, Map<String, String> params) {

    if (params == null) {
      params = Collections.emptyMap();
    }

    mxnet.AtomicSymbolCreator divFunc = Symbol.getSymbolCreator(name);

    mxnet.NDArrayHandle[] inputs = new mxnet.NDArrayHandle[inputArrays.length];

    for (int i = 0; i < inputArrays.length; i++) {
      inputs[i] = inputArrays[i].getHandle();
    }

    PointerPointer outputs = new PointerPointer(outputArrays.length);
    for (NDArray output : outputArrays) {
      outputs.put(output.getHandle());
    }

    String[] keys = new String[params.size()];
    String[] values = new String[params.size()];

    int i = 0;
    for (Map.Entry<String, String> entry : params.entrySet()) {
      keys[i] = entry.getKey();
      values[i] = entry.getValue();
      i++;
    }

    IntPointer numOutputs = new IntPointer(1);
    numOutputs.put(outputArrays.length);

    Util.check(mxnet.MXImperativeInvoke(divFunc, inputArrays.length, new PointerPointer(inputs), numOutputs,
        outputs, params.size(), new PointerPointer(keys), new PointerPointer(values)));

    // TODO: Is it necessary to reconstruct them from the outputs ?!?!
    return outputArrays;
  }

  private static NDArray imperativeInvoke(String name, NDArray input) {
    NDArray[] outputs = imperativeInvoke(name, new NDArray[]{input}, new NDArray[]{input},
        Collections.<String, String>emptyMap());
    return outputs[0];
  }

  private static void callScalarFunction(String name, NDArray input, float scalar) {
    Map<String, String> params = new HashMap<>();
    params.put("scalar", Float.toString(scalar));

    imperativeInvoke(name, new NDArray[]{input}, new NDArray[]{input}, params);
  }

  private void callScalarFunction(String name, float scalar) {
    callScalarFunction(name, this, scalar);
  }

  private static void addContextAndDType(Map<String, String> params, Context ctx, DType dtype) {
    if (ctx != null) {
      params.put("ctx", ctx.toContextString());
    }

    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }
  }

  /**
   * @return tuple of array dimensions
   */
  public int[] getShape() {
    IntPointer dimension = new IntPointer(1);
    IntPointer shapePointer = new IntPointer((IntPointer) null);

    verify(mxnet.MXNDArrayGetShape(getHandle(), dimension, shapePointer));

    int[] shape = new int[dimension.get()];
    shapePointer.get(shape);
    return shape;
  }

  public int getNDim() {
    return getShape().length;
  }

  /**
   * @return number of elements in the array, equivalent to the product of the array’s dimensions
   */
  public int size() {
    int size = 1;
    for (int dim : getShape()) {
      size *= dim;
    }
    return size;
  }

  /**
   * @return device context of the array
   */
  public Context getContext() {
    IntPointer deviceType = new IntPointer(1);
    IntPointer deviceId = new IntPointer(1);

    verify(mxnet.MXNDArrayGetContext(getHandle(), deviceType, deviceId));

    return new Context(deviceType.get(), deviceId.get());
  }

  /**
   * @return data-type of the array’s elements
   */
  public DType getDType() {
    IntPointer dtype = new IntPointer(1);
    verify(mxnet.MXNDArrayGetDType(getHandle(), dtype));
    return new DType(dtype.get());
  }



  // Array conversion

  /**
   * Copies the value of this array to another array.
   *
   * Note: other.getShape() and this.getShape() should be the same.
   *
   * @param other the destination array.
   * @return the copied array.
   */
  public NDArray copyTo(NDArray other) {
    // TODO: Assert identical shape, or leaeve this to imperative invoke ?!???!!!
    imperativeInvoke("_copyto", new NDArray[]{this}, new NDArray[]{other},
        Collections.<String, String>emptyMap());
    return other;
  }

  public NDArray copyTo(Context other) {
    return copyTo(empty(other, getDType(), getShape()));
  }

  /**
   * Makes a copy of this NDArray, keeping the same context.
   *
   * @return the copied array
   */
  public NDArray copy() {
    return copyTo(getContext());
  }

  /**
   * Returns an array on the target device with the same value as this array.
   *
   * If the target context is the same as this.getContext(), then this is returned.
   * Otherwise, a copy is made.
   *
   * @param context
   * @return
   */
  public NDArray asInContext(Context context) {
    if (getContext().equals(context)) {
      return this;
    }
    else {
      return copyTo(context);
    }
  }

  public void set(float[] source) {
    Util.check(mxnet.MXNDArraySyncCopyFromCPU(getHandle(), new FloatPointer(source), source.length));
  }

  public float[] toArray() {

    verify(mxnet.MXNDArrayWaitToRead(getHandle()));

    FloatPointer dataPointer = new FloatPointer((FloatPointer) null);
    verify(mxnet.MXNDArrayGetData(getHandle(), dataPointer));

    float[] shape = new float[size()];
    dataPointer.get(shape);
    return shape;
  }

  /**
   * Returns a scalar whose value is copied from this array.
   *
   * This function is equivalent to toArray()[0]. This NDArray must have shape (1,).
   */
  public float asScalar() {
    if (!Arrays.equals(new int[]{1}, getShape())) {
      throw new IllegalArgumentException("NDArray must have shape (1,)!");
    }

    return toArray()[0];
  }

  /**
   * Returns a copy of the array after casting to a specified type.
   *
   * @param type desired type of result array.
   */
  public NDArray asType(DType type) {
    NDArray array = NDArray.empty(getContext(), type, getShape());
    copyTo(array);
    return array;
  }



  // Array change shape

  /**
   * Returns a copy of the array with axes transposed.
   *
   * Equivalent to transpose() except that this is returned if this.ndim < 2.
   *
   * @return returns a copy rather than a view of the array unless this.ndim < 2
   */
  public NDArray T() {

    if (getNDim() < 2) {
      return this;
    }

    return NDArray.transpose(this);
  }

  // TODO: Arithmetic operations missing still ...

  // In-place arithmetic operations

  public void addi(float scalar) {
    callScalarFunction("_plus_scalar", scalar);
  }

  public void subi(float scalar) {
    callScalarFunction("_minus_scalar", scalar);
  }

  public void muli(float scalar) {
    callScalarFunction("_mul_scalar", scalar);
  }

  /**
   * Scalar division performed in-place.
   *
   * @param scalar
   */
  public void divi(float scalar) {
    callScalarFunction("_div_scalar", scalar);
  }

  public void modi(float scalar) {
    callScalarFunction("_mod_scalar", scalar);
  }

  // TODO: How to implement comparision operations ?!
  // This could implement Comparable

  /**
   * Release the native memory.
   */
  public void dispose() {
    verify(mxnet.MXNDArrayFree(getHandle()));
    handle = null;
  }



  // Array creation

  private static NDArray createNoneArray() {
    mxnet.NDArrayHandle handle = new mxnet.NDArrayHandle();
    Util.check(mxnet.MXNDArrayCreateNone(handle));
    return new NDArray(handle);
  }

  /**
   * Returns a new array of given shape and type, without initializing entries.
   *
   * @param ctx device context or null to use default
   * @param dtype dtype or null to use default
   * @param shape the shape of the array
   *
   * @return the created array
   */
  public static NDArray empty(Context ctx, DType dtype, int... shape) {
    ctx = getOrDefault(ctx);

    if (dtype == null) {
      dtype = DType.FLOAT_32;
    }

    mxnet.NDArrayHandle out = new mxnet.NDArrayHandle();

    int delay_alloc = 0;
    Util.check(mxnet.MXNDArrayCreateEx(shape, shape.length, ctx.getDeviceType(), ctx.getDeviceId(),
        delay_alloc, dtype.getDType(), out));

    return new NDArray(out);
  }

  // TODO: It would be good to create a series of methods to create the array
  //       from a Java array. In addition there should be ones with double and int

  /**
   * Create a new NDArray that copies content from the source array.
   *
   * @param ctx
   * @param dtype
   * @param source
   * @param shape the shape of the array
   *
   * @return the created array
   */
  public static NDArray array(Context ctx, DType dtype, float[] source, int... shape) {
    NDArray array = empty(ctx, dtype, shape);
    Util.check(mxnet.MXNDArraySyncCopyFromCPU(array.getHandle(), new FloatPointer(source), source.length));
    return array;
  }

  /**
   * Returns a new array of given shape and type, filled with the given value val.
   *
   * @param ctx
   * @param dtype
   * @param value
   * @param shape
   * @return
   */
  public static NDArray full(Context ctx, DType dtype, float value, int... shape) {
    NDArray array = empty(ctx, dtype, shape);

    mxnet.FunctionHandle setFunction = getFunction("_set_value");

    // TODO: Use imperative invoke here too?
    Util.check(mxnet.MXFuncInvoke(setFunction, null, new float[]{value},
        new PointerPointer(new Pointer[] {array.getHandle()})));

    return array;
  }

 /**
  * Returns a new array filled with all zeros, with the given shape and type.
  *
  * @param ctx
  * @param dtype
  * @param shape
  * @return the created array
  */
  public static NDArray zeros(Context ctx, DType dtype, int... shape) {
    return full(ctx, dtype, 0, shape);
  }

  /**
   * Returns a new array filled with all ones, with the given shape and type.
   *
   * @param ctx
   * @param dtype
   * @param shape
   *
   * @return the created array
   */
  public static NDArray ones(Context ctx, DType dtype, int... shape) {
    return full(ctx, dtype, 1, shape);
  }

  /**
   * Returns evenly spaced values within a given interval.
   *
   * Values are generated within the half-open interval [start, stop).
   * In other words, the interval includes start but excludes stop.
   *
   * @param ctx device context.
   * @param dtype the data type of the NDArray.
   * @param start start of interval.
   * @param step spacing between values.
   * @param repeat number of times to repeat each element.
   * @param stop end of interval.
   * @return
   */
  public static NDArray arange(Context ctx, DType dtype, float start, float step, int repeat,
                               float stop) {
    NDArray array = createNoneArray();

    Map<String, String> params = new HashMap<>();
    params.put("start", Float.toString(start));
    params.put("step", Float.toString(step));
    params.put("repeat", Integer.toString(repeat));
    params.put("stop", Float.toString(stop));

    addContextAndDType(params, ctx, dtype);

    imperativeInvoke("_arange", new NDArray[]{}, new NDArray[]{array}, params);

    return array;
  }

  // TODO: This might load multiple arrays from one file ...
  public static void load(Path file) throws IOException  {
    // TODO: Implement
  }

  public static void load(InputStream in) throws IOException  {
    // TODO: Implement
  }

  // to file, to stream
  public void save(Path file) throws IOException {
    // TODO: Implement
  }

  public void save(OutputStream out) throws IOException {
    // TODO: Implement
  }



  // Changing array shape and type

  /**
   * Casts all elements of the input to a new type.
   */
  public static NDArray cast(NDArray data, DType type) {

    Map<String, String> params = new HashMap<>();
    params.put("dtype", type.toString());

    imperativeInvoke("Cast", new NDArray[] {data}, new NDArray[] {data}, params);
    return data;
  }

  /**
   * @return a view of this array with a new shape without altering any data.
   */
  public NDArray reshape(int... shape) {
    // TODO: Use imperative invoke instead ?!?!?!!!!1
    mxnet.NDArrayHandle reshapedArrayView = new mxnet.NDArrayHandle();
    verify(mxnet.MXNDArrayReshape(getHandle(), shape.length, shape, reshapedArrayView));
    return new NDArray(reshapedArrayView);
  }

  /**
   * Flattens the input array into a 2-D array by collapsing the higher dimensions.
   */
  public static NDArray flatten(NDArray data) {
    return imperativeInvoke("flatten", data);
  }

  /**
   * Inserts a new axis of size 1 into the array shape.
   */
  public static NDArray expandDims(NDArray data, int axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Integer.toString(axis));

    imperativeInvoke("expand_dims", new NDArray[]{data}, new NDArray[]{data}, params);

    return data;
  }

  // Expanding array elements

  /**
   * Broadcasts the input array to a new shape.
   *
   * Broadcasting is only allowed on axes with size 1. The new shape cannot change the number of
   * dimensions. For example, you could broadcast from shape (2, 1) to (2, 3),
   * but not from shape (2, 3) to (2, 3, 3).
   */
  public static NDArray broadcastTo(NDArray data, int... shape) {
    Map<String, String> params = new HashMap<>();
    params.put("shape", Util.toTupleParam(shape));

    NDArray result = createNoneArray();

    imperativeInvoke("broadcast_to", new NDArray[]{data},
        new NDArray[]{result}, params);

    return result;
  }

  /**
   * Broadcasts the input array over particular axes.
   *
   * Broadcasting is allowed on axes with size 1, such as from (2,1,3,1) to (2,8,3,9).
   * Elements will be duplicated on the broadcasted axes.
   */
  public static NDArray broadcastAxis(NDArray data, int[] size, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("size", Util.toTupleParam(size));
    params.put("axis", Util.toTupleParam(axis));

    NDArray result = createNoneArray();
    imperativeInvoke("broadcast_axis", new NDArray[]{data}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Repeats elements of an array.
   */
  public static NDArray repeat(NDArray data, int repeat, int axis) {
    Map<String, String> params = new HashMap<>();
    params.put("repeats", Integer.toString(repeat));
    params.put("axis", Integer.toString(axis));

    NDArray result = createNoneArray();
    imperativeInvoke("repeat", new NDArray[]{data}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Repeats the whole array multiple times.
   */
  public static NDArray tile(NDArray data, int... reps) {
    Map<String, String> params = new HashMap<>();
    params.put("reps", Util.toTupleParam(reps));

    NDArray result = createNoneArray();
    imperativeInvoke("tile", new NDArray[]{data}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Pads an input array with a constant or edge values of the array.
   */
  public void pad() {
    // TODO: Implement
  }



  // Rearranging elements

  /**
   * Permutes the dimensions of an array.
   */
  public static NDArray transpose(NDArray data, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axes", Util.toTupleParam(axis));

    NDArray result = createNoneArray();
    imperativeInvoke("transpose", new NDArray[]{data}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Interchanges two axes of an array.
   */
  public static NDArray swapaxes(NDArray data, int dim1, int dim2) {
    Map<String, String> params = new HashMap<>();
    params.put("dim1", Integer.toString(dim1));
    params.put("dim2", Integer.toString(dim1));

    NDArray result = createNoneArray();
    imperativeInvoke("SwapAxis", new NDArray[]{data}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Reverses the order of elements along given axis while preserving array shape.
   */
  public static NDArray flip(NDArray data, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));

    NDArray result = createNoneArray();
    imperativeInvoke("reverse", new NDArray[]{data}, new NDArray[] {result}, params);
    return result;
  }



  // Joining and splitting arrays

  /**
   * Joins input arrays along a given axis.
   */
  public static NDArray concat(NDArray[] data, int dim) {
    Map<String, String> params = new HashMap<>();
    params.put("numArgs", Util.toTupleParam(data.length));
    params.put("dim", Util.toTupleParam(dim));

    NDArray result = createNoneArray();
    imperativeInvoke("concat", data, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Splits an array along a particular axis into multiple sub-arrays.
   */
  public static NDArray split(NDArray data) {
    // TODO: Finish

    Map<String, String> params = new HashMap<>();

    NDArray result = createNoneArray();
    imperativeInvoke("split", new NDArray[] {data}, new NDArray[] {result}, params);
    return result;
  }



  // Indexing routines

  /**
   * Slices a contiguous region of the array.
   */
  public static NDArray slice(NDArray data, int[] begin, int end[]) {
    Map<String, String> params = new HashMap<>();
    params.put("begin", Util.toTupleParam(begin));
    params.put("end", Util.toTupleParam(end));

    NDArray result = createNoneArray();
    imperativeInvoke("slice", new NDArray[] {data}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Slices along a given axis.
   */
  public static NDArray sliceAxis(NDArray data, int begin, int end, int axis) {
    Map<String, String> params = new HashMap<>();
    params.put("begin", Integer.toString(begin));
    params.put("end", Integer.toString(end));
    params.put("axis", Integer.toString(axis));

    NDArray result = createNoneArray();
    imperativeInvoke("slice_axis", new NDArray[] {data}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Takes elements from an input array along the given axis.
   */
  public static NDArray take(NDArray data, NDArray indices, int axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Integer.toString(axis));

    NDArray result = createNoneArray();
    imperativeInvoke("take", new NDArray[] {data, indices}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Takes elements from a data batch.
   */
  public NDArray batchTake(NDArray data, NDArray indices) {
    NDArray result = createNoneArray();
    imperativeInvoke("batch_take", new NDArray[] {data, indices}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Returns a one-hot array.
   */
  public static NDArray oneHotEncode(NDArray data) {
    NDArray result = createNoneArray();
    imperativeInvoke("_onehot_encode", new NDArray[] {data}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Picks elements from an input array according to the input indices along the given axis.
   */
  public void pick() {
    // TODO: Implement
  }



  // Arithmetic operations

  /**
   * Returns element-wise sum of the input arrays with broadcasting.
   */
  public static NDArray add(NDArray lhs, NDArray rhs) {
    NDArray result = createNoneArray();
    imperativeInvoke("broadcast_add", new NDArray[] {lhs, rhs}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Returns element-wise difference of the input arrays with broadcasting.
   */
  public static NDArray subtract(NDArray lhs, NDArray rhs) {
    NDArray result = createNoneArray();
    imperativeInvoke("broadcast_sub", new NDArray[] {lhs, rhs}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Numerical negative of the argument, element-wise.
   */
  public static NDArray negative(NDArray data) {
    return imperativeInvoke("negative", data);
  }

  /**
   * Returns element-wise product of the input arrays with broadcasting.
   */
  public static NDArray multiply(NDArray lhs, NDArray rhs) {
    NDArray result = createNoneArray();
    imperativeInvoke("broadcast_mul", new NDArray[] {lhs, rhs}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Returns element-wise division of the input arrays with broadcasting.
   */
  public static NDArray divide(NDArray lhs, NDArray rhs) {
    NDArray result = createNoneArray();
    imperativeInvoke("broadcast_div", new NDArray[] {lhs, rhs}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Returns element-wise modulo of the input arrays with broadcasting.
   */
  public static NDArray modulo(NDArray lhs, NDArray rhs) {
    NDArray result = createNoneArray();
    imperativeInvoke("broadcast_mod", new NDArray[] {lhs, rhs}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Dot product of two arrays.
   */
  public static NDArray dot(NDArray lhs, NDArray rhs, boolean transposeA, boolean transposeB) {
    Map<String, String> params = new HashMap<>();
    params.put("transpose_a", Boolean.toString(transposeA));
    params.put("transpose_b", Boolean.toString(transposeB));

    NDArray result = createNoneArray();
    imperativeInvoke("dot", new NDArray[] {}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Batchwise dot product.
   */
  public static NDArray batchDot(NDArray lhs, NDArray rhs, boolean transposeA, boolean transposeB) {
    Map<String, String> params = new HashMap<>();
    params.put("transpose_a", Boolean.toString(transposeA));
    params.put("transpose_b", Boolean.toString(transposeB));

    NDArray result = createNoneArray();
    imperativeInvoke("batch_dot", new NDArray[] {}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Adds all input arguments element-wise.
   */
  public static NDArray addN(NDArray... data) {
    NDArray result = createNoneArray();
    // TODO: Is this function correct? check with python ...
    imperativeInvoke("broadcast_add", data, new NDArray[] {result}, null);
    return result;
  }



  // Trigonometric functions

  /**
   * Computes the element-wise sine of the input array.
   */
  public static NDArray sin(NDArray data) {
    return imperativeInvoke("sin", data);
  }

  /**
   * Computes the element-wise cosine of the input array.
   */
  public static NDArray cos(NDArray data) {
    return imperativeInvoke("cos", data);
  }

  /**
   * Computes the element-wise tangent of the input array.
   */
  public static NDArray tan(NDArray data) {
    return imperativeInvoke("tan", data);
  }

  /**
   * Returns element-wise inverse sine of the input array.
   */
  public static NDArray arcsin(NDArray data) {
    return imperativeInvoke("arcsin", data);
  }

  /**
   * Returns element-wise inverse cosine of the input array.
   */
  public static NDArray arccos(NDArray data) {
    return imperativeInvoke("arccos", data);
  }

  /**
   * Returns element-wise inverse tangent of the input array.
   */
  public static NDArray arctan(NDArray data) {
    return imperativeInvoke("arctan", data);
  }

  /**
   * Converts each element of the input array from radians to degrees.
   */
  public static NDArray degrees(NDArray data) {
    return imperativeInvoke("degrees", data);
  }

  /**
   * Converts each element of the input array from degrees to radians.
   */
  public static NDArray radians(NDArray data) {
    return imperativeInvoke("radians", data);
  }



  // Hyperbolic functions

  /**
   * Returns the hyperbolic sine of the input array, computed element-wise.
   */
  public static NDArray sinh(NDArray data) {
    return imperativeInvoke("sinh", data);
  }

  /**
   * Returns the hyperbolic cosine of the input array, computed element-wise.
   */
  public static NDArray cosh(NDArray data) {
    return imperativeInvoke("cosh", data);
  }

  /**
   * Returns the hyperbolic tangent of the input array, computed element-wise.
   */
  public static NDArray tanh(NDArray data) {
    return imperativeInvoke("tanh", data);
  }

  /**
   * Returns the element-wise inverse hyperbolic sine of the input array, computed element-wise.
   */
  public static NDArray arcsinh(NDArray data) {
    return imperativeInvoke("arcsinh", data);
  }

  /**
   * Returns the element-wise inverse hyperbolic cosine of the input array, computed element-wise.
   */
  public static NDArray arccosh(NDArray data) {
    return imperativeInvoke("arccosh", data);
  }

  /**
   * Returns the element-wise inverse hyperbolic tangent of the input array, computed element-wise.
   */
  public static NDArray arctanh(NDArray data) {
    return imperativeInvoke("arctanh", data);
  }



  // Reduce functions

  /**
   * Computes the sum of array elements over given axes.
   */
  public static NDArray sum(NDArray data, boolean keepdims, boolean exclude, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));
    params.put("keepdims", Boolean.toString(keepdims));
    params.put("exclude", Boolean.toString(exclude));

    imperativeInvoke("sum", new NDArray[]{data}, new NDArray[]{data}, params);

    return data;
  }

  /**
   * Computes the sum of array elements over given axes treating Not a Numbers (NaN) as zero.
   */
  public static NDArray nansum(NDArray data, boolean keepdims, boolean exclude, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));
    params.put("keepdims", Boolean.toString(keepdims));
    params.put("exclude", Boolean.toString(exclude));

    imperativeInvoke("nansum", new NDArray[]{data}, new NDArray[]{data}, params);

    return data;
  }

  /**
   * Computes the product of array elements over given axes.
   */
  public static NDArray prod(NDArray data, boolean keepdims, boolean exclude, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));
    params.put("keepdims", Boolean.toString(keepdims));
    params.put("exclude", Boolean.toString(exclude));

    imperativeInvoke("prod", new NDArray[]{data}, new NDArray[]{data}, params);

    return data;
  }

  /**
   * Computes the product of array elements over given axes treating Not a Numbers (NaN) as one.
   */
  public static NDArray nanprod(NDArray data, boolean keepdims, boolean exclude, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));
    params.put("keepdims", Boolean.toString(keepdims));
    params.put("exclude", Boolean.toString(exclude));

    imperativeInvoke("nanprod", new NDArray[]{data}, new NDArray[]{data}, params);

    return data;
  }

  /**
   * Computes the mean of array elements over given axes.
   *
   * @param data the input.
   * @param keepdims if this is set to True, the reduced axes are left in the result as
   *                 dimension with size one.
   * @param exclude whether to perform reduction on axis that are NOT in axis instead.
   * @param axis the axis or axes along which to perform the reduction. The default, axis=(),
   *             will compute over all elements into a scalar array with shape (1,).
   *             If axis is int, a reduction is performed on a particular axis.
   *             If axis is a tuple of ints, a reduction is performed on all the axes specified in the tuple.
   *             If exclude is true, reduction will be performed on the axes that are NOT in axis instead.
   *             Negative values means indexing from right to left.
   *
   */
  public static NDArray mean(NDArray data, boolean keepdims, boolean exclude, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));
    params.put("keepdims", Boolean.toString(keepdims));
    params.put("exclude", Boolean.toString(exclude));

    imperativeInvoke("mean", new NDArray[]{data}, new NDArray[]{data}, params);

    return data;
  }

  /**
   * Computes the max of array elements over given axes.
   */
  public static NDArray max(NDArray data, boolean keepdims, boolean exclude, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));
    params.put("keepdims", Boolean.toString(keepdims));
    params.put("exclude", Boolean.toString(exclude));

    imperativeInvoke("max", new NDArray[]{data}, new NDArray[]{data}, params);

    return data;
  }

  /**
   * Computes the min of array elements over given axes.
   */
  public static NDArray min(NDArray data, boolean keepdims, boolean exclude, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));
    params.put("keepdims", Boolean.toString(keepdims));
    params.put("exclude", Boolean.toString(exclude));

    imperativeInvoke("min", new NDArray[]{data}, new NDArray[]{data}, params);

    return data;
  }

  /**
   * Flattens the input array and then computes the l2 norm.
   */
  public static NDArray norm(NDArray data) {
    return imperativeInvoke("norm", data);
  }



  // Rounding

  /**
   * Returns element-wise rounded value to the nearest integer of the input.
   */
  public static NDArray round(NDArray data) {
    return imperativeInvoke("round", data);
  }

  /**
   * Returns element-wise rounded value to the nearest integer of the input.
   */
  public static NDArray rint(NDArray data) {
    return imperativeInvoke("rint", data);
  }

  /**
   * Returns element-wise rounded value to the nearest integer towards zero of the input.
   */
  public static NDArray fix(NDArray data) {
    return imperativeInvoke("fix", data);
  }

  /**
   * Returns element-wise floor of the input.
   */
  public static NDArray floor(NDArray data) {
    return imperativeInvoke( "floor", data);
  }

  /**
   * Returns element-wise ceiling of the input.
   */
  public static NDArray ceil(NDArray data) {
    return imperativeInvoke( "ceil", data);
  }

  /**
   * Return the element-wise truncated value of the input.
   */
  public static NDArray trunc(NDArray data) {
    return imperativeInvoke( "trunc", data);
  }



  // Exponents and logarithms

  /**
   * Returns element-wise exponential value of the input.
   */
  public static NDArray exp(NDArray data) {
    return imperativeInvoke( "exp", data);
  }

  /**
   * Returns exp(x) - 1 computed element-wise on the input.
   */
  public static NDArray expm1(NDArray data) {
    return imperativeInvoke("expm1", data);
  }

  /**
   * Returns element-wise Natural logarithmic value of the input.
   */
  public static NDArray log(NDArray data) {
    return imperativeInvoke("log", data);
  }

  /**
   * Returns element-wise Base-10 logarithmic value of the input.
   */
  public static NDArray log10(NDArray data) {
    return imperativeInvoke("log10", data);
  }

  /**
   * Returns element-wise Base-2 logarithmic value of the input.
   */
  public static NDArray log2(NDArray data) {
    return imperativeInvoke("log2", data);
  }

  /**
   * Returns element-wise log(1 + x) value of the input.
   */
  public static NDArray log1p(NDArray data) {
    return imperativeInvoke( "log1p", data);
  }



  // Powers

  /**
   * Returns result of first array elements raised to powers from second array,
   * element-wise with broadcasting.
   */
  public static NDArray power(NDArray lhs, NDArray rhs) {
    NDArray result = createNoneArray();
    imperativeInvoke("_power", new NDArray[] {lhs, rhs}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Returns element-wise square-root value of the input.
   */
  public static NDArray sqrt(NDArray data) {
    return imperativeInvoke("sqrt", data);
  }

  /**
   * Returns element-wise inverse square-root value of the input.
   */
  public static NDArray rsqrt(NDArray data) {
    return imperativeInvoke("rsqrt", data);
  }

  /**
   * Returns element-wise squared value of the input.
   */
  public static NDArray square(NDArray data) {
    return imperativeInvoke("square", data);
  }



  // Logic functions

  /**
   * Returns the result of element-wise equal to (==) comparison operation with broadcasting.
   */
  public static NDArray equal(NDArray lhs, NDArray rhs) {
    NDArray result = createNoneArray();
    imperativeInvoke("broadcast_equal", new NDArray[] {lhs, rhs}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Returns the result of element-wise not equal to (!=) comparison operation with broadcasting.
   */
  public static NDArray notEqual(NDArray lhs, NDArray rhs) {
    NDArray result = createNoneArray();
    imperativeInvoke("broadcast_not_equal", new NDArray[] {lhs, rhs}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Returns the result of element-wise greater than (>) comparison operation with broadcasting.
   */
  public static NDArray greater(NDArray lhs, NDArray rhs) {
    NDArray result = createNoneArray();
    imperativeInvoke("broadcast_greater", new NDArray[] {lhs, rhs}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Returns the result of element-wise greater than or equal to (>=) comparison operation with broadcasting.
   */
  public static NDArray greaterEqual(NDArray lhs, NDArray rhs) {
    NDArray result = createNoneArray();
    imperativeInvoke("broadcast_greater_equal", new NDArray[] {lhs, rhs}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Returns the result of element-wise lesser than (<) comparison operation with broadcasting.
   */
  public static NDArray lesser(NDArray lhs, NDArray rhs) {
    NDArray result = createNoneArray();
    imperativeInvoke("broadcast_lesser", new NDArray[] {lhs, rhs}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Returns the result of element-wise lesser than or equal to (<=) comparison operation with broadcasting.
   */
  public static NDArray lesserEqual(NDArray lhs, NDArray rhs) {
    NDArray result = createNoneArray();
    imperativeInvoke("broadcast_lesser_equal", new NDArray[] {lhs, rhs}, new NDArray[] {result}, null);
    return result;
  }



  // Random sampling

  /**
   * Draw random samples from a uniform distribution.
   *
   * Samples are uniformly distributed over the half-open interval *[low, high)*
   * (includes *low*, but excludes *high*).
   */
  public static NDArray randomUniform(Context ctx, DType dtype, float low, float high, int... shape) {
    Map<String, String> params = new HashMap<>();
    params.put("low", Float.toString(low));
    params.put("high", Float.toString(high));
    params.put("shape", Util.toTupleParam(shape));
    addContextAndDType(params, ctx, dtype);

    NDArray result = createNoneArray();
    imperativeInvoke("_random_uniform", new NDArray[] {}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Draw random samples from a normal (Gaussian) distribution.
   */
  public static NDArray randomNormal(Context ctx, DType dtype, float loc, float scale, int... shape) {
    Map<String, String> params = new HashMap<>();
    params.put("loc", Float.toString(loc));
    params.put("scale", Float.toString(scale));
    params.put("shape", Util.toTupleParam(shape));
    addContextAndDType(params, ctx, dtype);

    NDArray result = createNoneArray();
    imperativeInvoke("_random_normal", new NDArray[] {}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Draw random samples from a gamma distribution.
   */
  public static NDArray randomGamma(Context ctx, DType dtype, float alpha, float beta, int... shape) {
    Map<String, String> params = new HashMap<>();
    params.put("alpha", Float.toString(alpha));
    params.put("beta", Float.toString(beta));
    params.put("shape", Util.toTupleParam(shape));
    addContextAndDType(params, ctx, dtype);

    NDArray result = createNoneArray();
    imperativeInvoke("_random_gamma", new NDArray[] {}, new NDArray[] {result}, params);
    return result;
  }

  // TODO: Add a seed to make it possible to generate the same distribution again for testing.
  //       The c api has mxnet.random.seed for that.

  /**
   * Draw random samples from an exponential distribution.
   */
  public static NDArray randomExponential(Context ctx, DType dtype, float lam, int... shape) {
    Map<String, String> params = new HashMap<>();
    params.put("lam", Float.toString(lam));
    params.put("shape", Util.toTupleParam(shape));
    addContextAndDType(params, ctx, dtype);

    NDArray result = createNoneArray();
    imperativeInvoke("_random_exponential", new NDArray[] {}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Draw random samples from a Poisson distribution.
   */
  public static NDArray randomPoisson(Context ctx, DType dtype, float lam, int... shape) {
    Map<String, String> params = new HashMap<>();
    params.put("lam", Float.toString(lam));
    params.put("shape", Util.toTupleParam(shape));
    addContextAndDType(params, ctx, dtype);

    NDArray result = createNoneArray();
    imperativeInvoke("_random_poisson", new NDArray[] {}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Draw random samples from a negative binomial distribution.
   */
  public static NDArray randomNegativeBinomial(Context ctx, DType dtype, int k, float p, int... shape) {
    Map<String, String> params = new HashMap<>();
    params.put("k", Integer.toString(k));
    params.put("p", Float.toString(p));
    params.put("shape", Util.toTupleParam(shape));
    addContextAndDType(params, ctx, dtype);

    NDArray result = createNoneArray();
    imperativeInvoke("_random_negative_binomial", new NDArray[] {},
        new NDArray[] {result}, params);
    return result;
  }

  /**
   * Draw random samples from a generalized negative binomial distribution.
   */
  public static NDArray randomGeneralizedNegativeBinomial(Context ctx, DType dtype, float mu,
                                                          float alpha, int... shape) {
    Map<String, String> params = new HashMap<>();
    params.put("mu", Float.toString(mu));
    params.put("alpha", Float.toString(alpha));
    params.put("shape", Util.toTupleParam(shape));
    addContextAndDType(params, ctx, dtype);

    NDArray result = createNoneArray();
    imperativeInvoke("_random_generalized_negative_binomial", new NDArray[] {},
        new NDArray[] {result}, params);
    return result;
  }



  // Sorting and searching

  /**
   * Returns a sorted copy of an input array along the given axis.
   */
  public static NDArray sort(NDArray data, int axis, boolean isAscend) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Integer.toString(axis));
    params.put("is_ascend", Boolean.toString(isAscend));

    NDArray result = createNoneArray();
    imperativeInvoke("sort", new NDArray[] {data}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Return type of topk.
   */
  public enum TopkReturnType {

    /**
     * Return a list of both values and indices of top k elements.
     */
    BOTH("both"),

    /**
     * Return the indices of the top k values.
     */
    INDICES("indices"),

    /**
     * Return a mask array containing 0 and 1. 1 means the top k values.
     */
    MASK("mask"),

    /**
     * Return the top k values.
     */
    VALUE("value");

    final String value;

    TopkReturnType(String value) {
      this.value = value;
    }
  }

  /**
   * Returns the top k elements in an input array along the given axis.
   */
  public static NDArray topk(NDArray data, int axis, int k, TopkReturnType returnType, boolean isAscend) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Integer.toString(axis));
    params.put("k", Integer.toString(axis));
    params.put("ret_typ", returnType.value);
    params.put("is_ascend", Boolean.toString(isAscend));

    NDArray result = createNoneArray();
    imperativeInvoke("topk", new NDArray[] {data}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Returns the indices that would sort an input array along the given axis.
   *
   * This function performs sorting along the given axis and returns an array of indices
   * having same shape as an input array that index data in sorted order.
   */
  public static NDArray argsort(NDArray data, int axis, boolean isAscend) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Integer.toString(axis));
    params.put("is_ascend", Boolean.toString(isAscend));

    NDArray result = createNoneArray();
    imperativeInvoke("argsort", new NDArray[] {data}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Returns indices of the maximum values along an axis.
   *
   * In the case of multiple occurrences of maximum values, the indices corresponding to
   * the first occurrence are returned.
   */
  public static NDArray argmax(NDArray data, int axis, boolean keepDims) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Integer.toString(axis));
    params.put("keepdims", Boolean.toString(keepDims));

    NDArray result = createNoneArray();
    imperativeInvoke("argmax", new NDArray[] {data}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Returns indices of the minimum values along an axis.
   *
   * In the case of multiple occurrences of minimum values, the indices corresponding to the
   * first occurrence are returned.
   */
  public static NDArray argmin(NDArray data, int axis, boolean keepDims) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Integer.toString(axis));
    params.put("keepdims", Boolean.toString(keepDims));

    NDArray result = createNoneArray();
    imperativeInvoke("argmin", new NDArray[] {data}, new NDArray[] {result}, params);
    return result;
  }



  // Miscellaneous

  /**
   * Returns element-wise maximum of the input arrays with broadcasting.
   */
  public static NDArray maximum(NDArray lhs, NDArray rhs) {
    NDArray result = createNoneArray();
    imperativeInvoke("broadcast_maximum", new NDArray[] {lhs, rhs}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Returns element-wise minimum of the input arrays with broadcasting.
   */
  public static NDArray minimum(NDArray lhs, NDArray rhs) {
    NDArray result = createNoneArray();
    imperativeInvoke("broadcast_minimum", new NDArray[] {lhs, rhs}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Clips (limits) the values in an array.
   */
  public static NDArray clip(NDArray data, float min, float max) {
    Map<String, String> params = new HashMap<>();
    params.put("a_min", Float.toString(min));
    params.put("a_max", Float.toString(max));

    imperativeInvoke("clip", new NDArray[] {data}, new NDArray[] {data}, params);
    return data;
  }

  /**
   * Returns element-wise absolute value of the input.
   */
  public static NDArray abs(NDArray data) {
    return imperativeInvoke("abs", data);
  }

  /**
   * Returns element-wise sign of the input.
   */
  public static NDArray sign(NDArray data) {
    return imperativeInvoke("sign", data);
  }

  /**
   * Returns the gamma function (extension of the factorial function to the reals),
   * computed element-wise on the input array.
   */
  public static NDArray gamma(NDArray data) {
    return imperativeInvoke("gamma", data);
  }

  /**
   * Returns element-wise log of the absolute value of the gamma function of the input.
   */
  public static NDArray gammaln(NDArray data) {
    return imperativeInvoke("gammaln", data);
  }



  // Neural network - Basic

  /**
   * Applies a linear transformation: Y=XWT+b
   */
  // TODO: Remove noBias flag, and allow null for bias instead
  public static NDArray fullyConnected(NDArray data, NDArray weight, NDArray bias, int numHidden,
                                    boolean noBias, boolean flatten) {
    Map<String, String> params = new HashMap<>();
    params. put("num_hidden", Integer.toString(numHidden));
    params. put("no_bias", Boolean.toString(noBias));
    params. put("flatten", Boolean.toString(flatten));

    NDArray result = createNoneArray();
    imperativeInvoke("FullyConnected", new NDArray[]{data, weight, bias},
        new NDArray[]{result}, params);
    return result;
  }

  // TODO: Move all optional parameters here ...
  public static class ConvolutionSettings {
    
  }

  /**
   * Compute N-D convolution on (N+2)-D input.
   */
  public static void convolution(NDArray data, NDArray weight, NDArray bias, int[] kernel,
                                 int[] stride, int[] dilate, int[] pad, int numFilter, int numGroup,
                                 long workspace, boolean noBias, Object cudnnTune, boolean cudnnOff,
                                 Object layout) {

    // maybe wrap the optional arguments in a parameter object ...

    // TODO: Implement

    // "Convolution"

  }

  public enum ActivationType {
    RELU("relu"),
    SIGMOID("sigmoid"),
    SOFTRELU("softrelu"),
    TANH("tanh");

    final String value;

    ActivationType(String value) {
      this.value = value;
    }
  }

  /**
   * Applies an activation function element-wise to the input.
   */
  public static NDArray activation(NDArray data, ActivationType actType) {
    Map<String, String> params = new HashMap<>();
    params.put("act_type", actType.value);

    NDArray result = createNoneArray();
    imperativeInvoke("Activation", new NDArray[] {data}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Batch normalization.
   */
  public static void batchNorm() {
    // TODO: Implement
  }

  /**
   * Performs pooling on the input.
   */
  public static void pooling() {
    // TODO: Implement
  }

  /**
   * Computes the gradient of cross entropy loss with respect to softmax output.
   */
  public static void softmaxOutput() {
    // TODO: Implement
  }

  /**
   * Applies the softmax function.
   */
  public static NDArray softmax(NDArray data, int axis) {
    Map<String, String> params = new HashMap<>();
    params. put("axis", Integer.toString(axis));

    NDArray result = createNoneArray();
    imperativeInvoke("softmax", new NDArray[]{data},
        new NDArray[]{result}, params);
    return result;
  }

  /**
   * Computes the log softmax of the input. This is equivalent to computing softmax followed by log.
   */
  public static NDArray logSoftmax(NDArray data, int axis) {
    Map<String, String> params = new HashMap<>();
    params. put("axis", Integer.toString(axis));

    NDArray result = createNoneArray();
    imperativeInvoke("log_softmax", new NDArray[]{data},
        new NDArray[]{result}, params);
    return result;
  }



  // Neural network - More

  /**
   * Applies correlation to inputs.
   */
  public static void correlation() {
    // TODO: Implement
  }

  /**
   * Computes 2D transposed convolution (aka fractionally strided convolution) of the input tensor.
   */
  public static void deconvolution() {
    // TODO: Implement
  }

  public enum RnnMode {
    GRU("gru"),
    LSTM("lstm"),
    RNN_RELU("rnn_relu"),
    RNN_TANH("rnn_tanh");

    final String value;

    RnnMode(String value) {
      this.value = value;
    }
  }

  /**
   * Applies a recurrent layer to input.
   */
  public static void rnn(NDArray data, NDArray parameters, NDArray state, NDArray stateCell,
                         int stateSize, int numLayers, boolean bidirectional, RnnMode mode,
                         float p, boolean stateOutpus) {
    // TODO: Implement
  }

  /**
   * Maps integer indices to vector representations (embeddings).
   */
  public static NDArray embedding(NDArray data, NDArray weight, int inputDim, int outputDim, DType type) {

    Map<String, String> params = new HashMap<>();
    params.put("input_dim", Integer.toString(inputDim));
    params.put("output_dim", Integer.toString(outputDim));

    if (type != null) {
      params.put("dtype", type.toString());
    }

    NDArray result = createNoneArray();
    imperativeInvoke("Embedding", new NDArray[]{data}, new NDArray[] {result}, params);
    return result;
  }

  public enum LeakyReLUActivationType {
    ELU("elu"),
    LEAKY("leaky"),
    PRELU("prelu"),
    RRELU("rrelu");

    final String value;

    LeakyReLUActivationType(String value) {
      this.value = value;
    }
  }

  /**
   * Applies Leaky rectified linear unit activation element-wise to the input.
   */
  public static NDArray leakyReLU(NDArray data, LeakyReLUActivationType actType, float slope,
                               float lowerBound, float upperBound) {
    Map<String, String> params = new HashMap<>();
    params.put("act_type", actType.value);
    params.put("slope", Float.toString(slope));
    params.put("lower_bound", Float.toString(lowerBound));
    params.put("upper_bound", Float.toString(upperBound));

    NDArray result = createNoneArray();
    imperativeInvoke("LeakyReLU", new NDArray[]{data}, new NDArray[] {result}, params);
    return result;
  }

  /**
   * Applies instance normalization to the n-dimensional input array.
   */
  public static void instanceNorm() {
    // TODO: Implement
  }

  /**
   * Normalize the input array using the L2 norm.
   */
  public static void l2Normalization() {
    // TODO: Implement
  }

  /**
   * Applies local response normalization to the input.
   */
  public static void lrn() {
    // TODO: Implement
  }

  /**
   * Performs region of interest(ROI) pooling on the input array.
   */
  public static void roiPooling() {
    // TODO: Implement
  }

  /**
   * Applies softmax activation to input.
   */
  public static void softmaxActivation() {
    // TODO: Implement
  }

  /**
   * Applies dropout operation to input array.
   */
  public static void dropout() {
    // TODO: Implement
  }

  /**
   * Applies bilinear sampling to input feature map.
   */
  public static void bilinearSampler() {
    // TODO: Implement
  }

  /**
   * Generates 2D sampling grid for bilinear sampling.
   */
  public static void gridGenerator() {
    // TODO: Implement
  }

  /**
   * Performs nearest neighbor/bilinear up sampling to inputs.
   */
  public static void upSampling() {
    // TODO: Implement
  }

  /**
   * Applies a spatial transformer to input feature map.
   */
  public static void spatialTransformer() {
    // TODO: Implement
  }

  /**
   * Computes and optimizes for squared loss during backward propagation.
   */
  public static void linearRegressionOutput() {
    // TODO: Implement
  }

  /**
   * Applies a logistic function to the input.
   */
  public static void logisticRegressionOutput() {
    // TODO: Implement
  }

  /**
   * Computes mean absolute error of the input.
   */
  public static void maeRegressionOutput() {
    // TODO: Implement
  }

  /**
   * Computes support vector machine based transformation of the input.
   */
  public static void svmOutput() {
    // TODO: Implement
  }

  /**
   * Calculate cross entropy of softmax output and one-hot label.
   */
  public static NDArray softmaxCrossEntropy(NDArray data, NDArray label) {
    NDArray result = createNoneArray();
    imperativeInvoke("softmax_cross_entropy", new NDArray[]{data}, new NDArray[] {result}, null);
    return result;
  }

  /**
   * Calculate Smooth L1 Loss(lhs, scalar) by summing
   */
  public static void smoothL1(NDArray data, float scalar) {
    callScalarFunction("smooth_l1", data, scalar);
  }

  /**
   * Apply a sparse regularization to the output a sigmoid activation function.
   */
  public static void identityAttachKLSparseReg() {
    // TODO: Implement
  }

  /**
   * Make your own loss function in network construction.
   */
  public static void makeLoss() {
    // TODO: Implement
  }

  /**
   * Stops gradient computation.
   */
  public static void blockGrad() {
    // TODO: Implement
  }

  /**
   * Apply a custom operator implemented in a frontend language (like Python).
   */
  public static void custom() {
    // TODO: Implement
  }

  @Override
  public String toString() {
    // TODO: Format is not identical with python -> shape should be printed as 128x64
    return "NDArray " + Arrays.toString(getShape()) + " @" + getContext();
  }

  // TODO: Remove this ...
  public static void listFunctions() {

    for (String symbolName : Symbol.getSymbolCreatorNames()) {

      mxnet.AtomicSymbolCreator symbolCreator = Symbol.getSymbolCreator(symbolName);

      PointerPointer name = new PointerPointer(1);
      PointerPointer description = new PointerPointer(1);
      IntPointer num_args = new IntPointer(1);
      PointerPointer arg_names = new PointerPointer(1);
      PointerPointer arg_type_infos = new PointerPointer(1);
      PointerPointer arg_descriptions = new PointerPointer(1);
      PointerPointer key_var_num_args = new PointerPointer(1);
      PointerPointer return_type = new PointerPointer(1);

      mxnet.MXSymbolGetAtomicSymbolInfo(symbolCreator, name, description, num_args,
          arg_names, arg_type_infos, arg_descriptions, key_var_num_args, return_type);

      String text = name.getString(0) + "\n";

      for (int i = 0; i < num_args.get(); i++) {
        text += " " + arg_names.getString(i) + " " + arg_type_infos.getString(i)
            + " " + arg_descriptions.getString(i) + "\n";
      }

      text +=  "\n" + description.getString(0);

      System.out.println("###################################################");
      System.out.println(text);
      System.out.println("###################################################");
    }
  }
}
