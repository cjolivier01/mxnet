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

import java.nio.IntBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import org.apache.mxnet.javacpp.mxnet;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;

/**
 * The Symbol API, defined in the symbol (or simply sym) package, provides neural network graphs
 * and auto-differentiation.
 *
 * A symbol represents a multi-output symbolic expression. They are composited by operators,
 * such as simple matrix operations (e.g. “+”), or a neural network layer (e.g. convolution layer).
 * An operator can take several input variables, produce more than one output variables,
 * and have internal state variables. A variable can be either free, which we can bind with value later,
 * or an output of another symbol.
 */
@SuppressWarnings("unused")
public class Symbol {

  private static Map<String, mxnet.AtomicSymbolCreator> atomicSymbolCreatorMap;

  private mxnet.SymbolHandle handle;

  private Symbol(mxnet.SymbolHandle handle) {
    this.handle = Objects.requireNonNull(handle);
  }

  private mxnet.SymbolHandle getHandle() {
    if (handle == null) {
      throw new MXNetError("Symbol was disposed!");
    }

    return handle;
  }

  private static void initSymbolCreatorMap() {
    // TODO: This is not tread safe ...
    if (atomicSymbolCreatorMap == null) {
      HashMap<String, mxnet.AtomicSymbolCreator> creatorMap = new HashMap<>();

      IntPointer symbolCreatorsSize = new IntPointer(1);
      PointerPointer symbolCreatorHandles = new PointerPointer();
      mxnet.MXSymbolListAtomicSymbolCreators(symbolCreatorsSize, symbolCreatorHandles);

      for (int i = 0; i < symbolCreatorsSize.get(); i++) {
        mxnet.AtomicSymbolCreator symbol = new mxnet.AtomicSymbolCreator(symbolCreatorHandles.get(i));
        PointerPointer symbolName = new PointerPointer(1);
        mxnet.MXSymbolGetAtomicSymbolName(symbol, symbolName);

        creatorMap.put(symbolName.getString(0), symbol);
      }

      atomicSymbolCreatorMap = Collections.unmodifiableMap(creatorMap);
    }
  }

  static Set<String> getSymbolCreatorNames() {
    initSymbolCreatorMap();
    return atomicSymbolCreatorMap.keySet();
  }

  static mxnet.AtomicSymbolCreator getSymbolCreator(String name) {
    initSymbolCreatorMap();
    mxnet.AtomicSymbolCreator creator = atomicSymbolCreatorMap.get(name);

    if (creator == null) {
      throw new IllegalArgumentException(String.format("SymbolCreator [%s] does not exist!", name));
    }

    return creator;
  }

  private static Symbol compose(String name, Symbol symbol, String[] keys, Symbol[] args) {

    if (keys == null) {
      keys = new String[0];
    }

    if (name == null)
      name = "";

    BytePointer namePointer = new BytePointer(name);

    mxnet.SymbolHandle[] argsHandles = new mxnet.SymbolHandle[args.length];

    int i = 0;
    for (Symbol arg : args) {
      argsHandles[i] = arg.getHandle();
      i++;
    }

    Util.check(mxnet.MXSymbolCompose(symbol.getHandle(), namePointer, argsHandles.length,
        new PointerPointer(keys), new PointerPointer(argsHandles)));

    return symbol;
  }

  private static Symbol compose(String name, Symbol symbol, Map<String, Symbol> args) {

    List<String> keys = new ArrayList<>();
    List<Symbol> symbols = new ArrayList<>();

    for (Map.Entry<String, Symbol> entry : args.entrySet()) {
      if (entry.getValue() != null) {
        keys.add(entry.getKey());
        symbols.add(entry.getValue());
      }
    }

    return compose(name, symbol, keys.toArray(new String[keys.size()]),
        symbols.toArray(new Symbol[symbols.size()]));
  }


  private static Symbol createAtomicSymbol(String name, Map<String, String> params) {

    if (params == null) {
      params = Collections.emptyMap();
    }

    mxnet.AtomicSymbolCreator function = Symbol.getSymbolCreator(name);

    String[] keys = new String[params.size()];
    String[] values = new String[params.size()];

    int i = 0;
    for (Map.Entry<String, String> entry : params.entrySet()) {
      keys[i] = entry.getKey();
      values[i] = entry.getValue();
      i++;
    }

    PointerPointer outSym = new PointerPointer(1);

    Util.check(mxnet.MXSymbolCreateAtomicSymbol(function, params.size(),
        new PointerPointer(keys), new PointerPointer(values), outSym));

    return new Symbol(new mxnet.SymbolHandle(outSym.get()));
  }

  private static Symbol createSymbol(String name, String function, Symbol data) {
    Symbol symbol = createAtomicSymbol(function, null);
    compose(name, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  private static Symbol createSymbol(String function, Symbol data) {
    return createSymbol(null, function, data);
  }

  private static Symbol createSymbol(String function, Symbol lhs, Symbol rhs) {
    Symbol symbol = createAtomicSymbol(function, null);
    compose(null, symbol, new String[]{"lhs", "rhs"}, new Symbol[] {lhs, rhs});
    return symbol;
  }

  private static Symbol createScalarSymbol(String name, Symbol input, float scalar) {
    Map<String, String> params = new HashMap<>();
    params.put("scalar", Float.toString(scalar));

    return compose(null, createAtomicSymbol(name, params),
        new String[] {"data"}, new Symbol[] {input});
  }

  // Arithmetic operations

  public Symbol add(Symbol other) {
    return createSymbol("elemwise_add", this, other);
  }

  public Symbol div(float scalar) {
    return createScalarSymbol("_div_scalar", this, scalar);
  }

  //Symbol.__sub__ 	x.__sub__(y) <=> x-y
  //Symbol.__rsub__ 	x.__rsub__(y) <=> y-x
  //Symbol.__neg__ 	x.__neg__() <=> -x
  //Symbol.__mul__ 	x.__mul__(y) <=> x*y
  //Symbol.__div__ 	x.__div__(y) <=> x/y
  //Symbol.__rdiv__ 	x.__rdiv__(y) <=> y/x
  //Symbol.__mod__ 	x.__mod__(y) <=> x%y
  //Symbol.__rmod__ 	x.__rmod__(y) <=> y%x
  //Symbol.__pow__ 	x.__pow__(y) <=> x**y

  // Comparison operators

  // Symbol creation

  // Symbol.zeros_like 	Convenience fluent method for zeros_like().
  // Symbol.ones_like 	Convenience fluent method for ones_like()



  // Changing shape and type
  // Symbol.astype 	Convenience fluent method for cast().
  // Symbol.reshape 	Convenience fluent method for reshape().
  // Symbol.flatten 	Convenience fluent method for flatten().
  // Symbol.expand_dims 	Convenience fluent method for expand_dims().



  // Expanding elements
  // Symbol.broadcast_to 	Convenience fluent method for broadcast_to().
  // Symbol.broadcast_axes 	Convenience fluent method for broadcast_axes().
  // Symbol.tile 	Convenience fluent method for tile().
  // Symbol.pad 	Convenience fluent method for pad().



  // Rearranging elements
  // Symbol.transpose 	Convenience fluent method for transpose().
  // Symbol.swapaxes 	Convenience fluent method for swapaxes().
  // Symbol.flip 	Convenience fluent method for flip().

  // Reduce functions
  // Symbol.sum 	Convenience fluent method for sum().
  // Symbol.nansum 	Convenience fluent method for nansum().
  // Symbol.prod 	Convenience fluent method for prod().
  // Symbol.nanprod 	Convenience fluent method for nanprod().
  // Symbol.mean 	Convenience fluent method for mean().
  // Symbol.max 	Convenience fluent method for max().
  // Symbol.min 	Convenience fluent method for min().
  // Symbol.norm 	Convenience fluent method for norm().

  // Rounding
  // Symbol.round 	Convenience fluent method for round().
  // Symbol.rint 	Convenience fluent method for rint().
  // Symbol.fix 	Convenience fluent method for fix().
  // Symbol.floor 	Convenience fluent method for floor().
  // Symbol.ceil 	Convenience fluent method for ceil().
  // Symbol.trunc 	Convenience fluent method for trunc().



  // Sorting and searching

  // Symbol.sort 	Convenience fluent method for sort().
  // Symbol.argsort 	Convenience fluent method for argsort().
  // Symbol.topk 	Convenience fluent method for topk().
  // Symbol.argmax 	Convenience fluent method for argmax().
  // Symbol.argmin 	Convenience fluent method for argmin().


  // Query information

  /**
   * Gets name string from the symbol, this function only works for non-grouped symbol.
   *
   * @return the symbol name or null when failure happens
   */
  public String name() {
    PointerPointer nameString = new PointerPointer(1);
    IntPointer success = new IntPointer(1);

    Util.check(mxnet.MXSymbolGetName(getHandle(), nameString, success));

    if (success.get() == 0) {
      return nameString.getString(0);
    }
    else {
      return null;
    }
  }

  /**
   * Lists all the arguments in the symbol.
   */
  public String[] listArguments() {
    IntPointer argumentNumber = new IntPointer(1);
    PointerPointer argumentsPointer = new PointerPointer();

    Util.check(mxnet.MXSymbolListArguments(getHandle(), argumentNumber, argumentsPointer));

    String[] arguments = new String[argumentNumber.get()];

    for (int i = 0; i < argumentNumber.get(); i++) {
      arguments[i] = argumentsPointer.getString(i);
    }

    return arguments;
  }

  /**
   * Lists all the outputs in the symbol.
   */
  public String[] listOutputs() {
    IntPointer outputsNumber = new IntPointer(1);
    PointerPointer outputsPointer = new PointerPointer();

    Util.check(mxnet.MXSymbolListOutputs(getHandle(), outputsNumber, outputsPointer));

    String[] outputs = new String[outputsNumber.get()];
    for (int i = 0; i < outputsNumber.get(); i++) {
      outputs[i] = outputsPointer.getString(i);
    }

    return outputs;
  }

  /**
   * Lists all the auxiliary states in the symbol.
   */
  public String[] listAuxiliaryStates() {
    IntPointer statesNumber = new IntPointer(1);
    PointerPointer statesPointer = new PointerPointer();

    Util.check(mxnet.MXSymbolListAuxiliaryStates(getHandle(), statesNumber, statesPointer));

    String[] states = new String[statesNumber.get()];
    for (int i = 0; i < states.length; i++) {
      states[i] = statesPointer.getString(i);
    }

    return states;
  }

  /**
   * Gets all attributes from the symbol.
   */
  public Map<String, String> listAttr() {
    IntPointer dictSize = new IntPointer(1);
    PointerPointer attrStringPointer = new PointerPointer();

    Util.check(mxnet.MXSymbolListAttr(getHandle(), dictSize, attrStringPointer));

    Map<String, String> dict = new HashMap<>();
    for (int i = 0; i + 1 < 2 * dictSize.get(); i+=2) {
      dict.put(attrStringPointer.getString(i), attrStringPointer.getString(i + 1));
    }

    return dict;
  }

  /**
   * Returns the attribute string for corresponding input key from the symbol.
   */
  public String attr(String key) {
    BytePointer keyPointer = new BytePointer();

    PointerPointer attrStringPointer = new PointerPointer();

    IntPointer successPointer = new IntPointer(1);

    mxnet.MXSymbolGetAttr(getHandle(), keyPointer, attrStringPointer, successPointer);

    if (successPointer.get() == 0) {
      return attrStringPointer.getString(0);
    }
    else {
      return null;
    }
  }

  /**
   * Recursively gets all attributes from the symbol and its children.
   */
  public void attrDict() {
    // TODO: Implement


  }



  // Indexing
  // Symbol.slice 	Convenience fluent method for slice().
  // Symbol.slice_axis 	Convenience fluent method for slice_axis().
  // Symbol.take 	Convenience fluent method for take().
  // Symbol.one_hot 	Convenience fluent method for one_hot().
  // Symbol.pick 	Convenience fluent method for pick().



  // Get internal and output symbol
  // Symbol.__getitem__ 	x.__getitem__(i) <=> x[i]
  // Symbol.__iter__ 	Returns a generator object of symbol.
  // Symbol.get_internals 	Gets a new grouped symbol sgroup.
  // Symbol.get_children 	Gets a new grouped symbol whose output contains inputs to output nodes of the original symbol.



  // Inference type and shape

  // Infers the type of all arguments and all outputs, given the known types for some arguments.
  // public void inferType

   //  @param args Provide shape of arguments in a positional way.
   //*             Unknown shape can be marked as None
   //* @return
   //    * argShapes
   // outShapes
   // auxShapes

  public static class InferredShapes {

    /**
     * List of shapes of arguments. The order is in the same order as list_arguments().
     */
    public List<int[]> argShapes;

    /**
     * List of shapes of outputs. The order is in the same order as list_outputs().
     */
    public List<int[]> outShapes;

    /**
     * List of shapes of outputs. The order is in the same order as list_auxiliary().
     */
    public List<int[]> auxShapes;
  }

  private List<int[]> toShapes(IntPointer size, PointerPointer ndimsPointerPointer,
                               PointerPointer dataPointerPointer) {


    Pointer ndimsPointer = ndimsPointerPointer.get();

    if (ndimsPointer != null) {
      ndimsPointer.capacity(4 * size.get());

      IntBuffer ndims = ndimsPointer.asByteBuffer().asIntBuffer();

      List<int[]> shapes = new ArrayList<>();
      int dataIndex = 0;
      for (int i = 0; i < ndims.capacity(); i++) {
        int ndim = ndims.get(i);

        Pointer dataPointer = dataPointerPointer.get();
        dataPointer.capacity(4 * (dataIndex + ndim));

        IntBuffer databuffer = dataPointer.asByteBuffer().asIntBuffer();

        int[] shape = new int[ndim];

        for (int j = 0; j < shape.length; j++) {
          shape[j] = databuffer.get(dataIndex + j);
        }
        shapes.add(shape);

        dataIndex += ndim;
      }

      return Collections.unmodifiableList(shapes);
    }
    else {
      return null;
    }
  }

  /**
   * Infers the shapes of all arguments and all outputs given the known shapes of some arguments.
   *
   * @param args Provide shape of arguments in a positional way. Unknown shape can be marked as null.
   */
  public InferredShapes inferShape(List<int[]> args) {

    // TODO: Deduplicate this ...

    int num_args = -1; // numbe of input arguments.
    PointerPointer keys = new PointerPointer(); // optional

    IntPointer arg_ind_ptr = new IntPointer(args.size());
    for (int i = 0; i < args.size(); i++) {
      arg_ind_ptr.put(args.get(i).length);
    }

    IntPointer arg_shape_data = new IntPointer(); // the content of the CSR
    for (int i = 0; i < args.size(); i++) {
      arg_shape_data.put(args.get(i), 0, args.get(i).length);
    }

    IntPointer in_shape_size = new IntPointer();
    PointerPointer in_shape_ndim = new PointerPointer();
    PointerPointer in_shape_data = new PointerPointer();

    IntPointer out_shape_size = new IntPointer();
    PointerPointer out_shape_ndim = new PointerPointer();
    PointerPointer out_shape_data = new PointerPointer();

    IntPointer  aux_shape_size = new IntPointer();
    PointerPointer aux_shape_ndim = new PointerPointer();
    PointerPointer aux_shape_data = new PointerPointer();
    IntPointer complete = new IntPointer(1);

    Util.check(mxnet.MXSymbolInferShape(getHandle(), num_args, keys, arg_ind_ptr, arg_shape_data,
        in_shape_size, in_shape_ndim, in_shape_data,
        out_shape_size, out_shape_ndim, out_shape_data,
        aux_shape_size, aux_shape_ndim, aux_shape_data, complete));


    InferredShapes inferredShapes = new InferredShapes();

    inferredShapes.argShapes = toShapes(in_shape_size, in_shape_ndim, in_shape_data);
    inferredShapes.outShapes = toShapes(out_shape_size, out_shape_ndim, out_shape_data);
    inferredShapes.auxShapes = toShapes(in_shape_size, aux_shape_ndim, aux_shape_data);

    return inferredShapes;
  }

  // TODO: Would be nice to improve this with better JavaCPP mappings ...
  public InferredShapes inferShape(Map<String, int[]> args) {

    int num_args = args.size();
    List<String> keys = new ArrayList<>();
    List<int[]> argShapes = new ArrayList<>();

    for (Map.Entry<String, int[]> arg : args.entrySet()) {
      keys.add(arg.getKey());
      argShapes.add(arg.getValue());
    }

    int[] argInd = new int[args.size() + 1];
    for (int i = 0; i < argShapes.size(); i++) {
      argInd[i + 1] = (argInd[i] + argShapes.get(i).length);
    }

    IntPointer arg_ind_ptr = new IntPointer(argInd);

    int[] argShapeData =
        argShapes.stream().flatMapToInt(shape -> Arrays.stream(shape)).toArray();

    IntPointer in_shape_size = new IntPointer(1);
    PointerPointer in_shape_ndim = new PointerPointer(1);
    PointerPointer in_shape_data = new PointerPointer();

    IntPointer out_shape_size = new IntPointer(1);
    PointerPointer out_shape_ndim = new PointerPointer(1);
    PointerPointer out_shape_data = new PointerPointer();

    IntPointer  aux_shape_size = new IntPointer(1);
    PointerPointer aux_shape_ndim = new PointerPointer(1);
    PointerPointer aux_shape_data = new PointerPointer();
    IntPointer complete = new IntPointer(1);

    Util.check(mxnet.MXSymbolInferShape(getHandle(), num_args,
        new PointerPointer(keys.toArray(new String[keys.size()])), arg_ind_ptr, new IntPointer(argShapeData),
        in_shape_size, in_shape_ndim, in_shape_data,
        out_shape_size, out_shape_ndim, out_shape_data,
        aux_shape_size, aux_shape_ndim, aux_shape_data, complete));


    InferredShapes inferredShapes = new InferredShapes();
    inferredShapes.argShapes = toShapes(in_shape_size, in_shape_ndim, in_shape_data);
    inferredShapes.outShapes = toShapes(out_shape_size, out_shape_ndim, out_shape_data);
    inferredShapes.auxShapes = toShapes(in_shape_size, aux_shape_ndim, aux_shape_data);

    return inferredShapes;
  }


  // Infers the shape partially.
  public void inferShapePartial() {
    // TODO: Implement
  }



  // Bind

  public enum BindGradReqType {
    WRITE(1),
    ADD(3),
    NULL(0);

    final int value;

    BindGradReqType(int value) {
      this.value = value;
    }
  }

  /**
   * Binds the current symbol to an executor and returns it.
   */
  public Executor bind(Context ctx, Map<String, NDArray> args, NDArray[] argsGrad,
                       BindGradReqType[] gradReqs, NDArray[] auxStates, Map<String, Context> group2Ctx,
                       Executor sharedExecutor) {

    // TODO: Remove the map for arguments, this makes it complicated with aligned args like grads and reqs
    String[] listArguments = listArguments();

    PointerPointer argsHandles = new PointerPointer(args.size());
    for (int i = 0; i < listArguments.length; i++) {
      if (args.get(listArguments[i]) != null) {
        argsHandles.put(i, args.get(listArguments[i]).getHandle());
      }
    }

    PointerPointer gradsHandles = new PointerPointer(argsGrad.length);
    for (int i = 0; i < argsGrad.length; i++) {
      if (argsGrad[i] != null) {
        gradsHandles.put(i, argsGrad[i].getHandle());
      }
      else {
        gradsHandles.put(i, null);
      }
    }

    // TODO: What should be the default here ?!?!?! Not initializing this is bad ...
    IntPointer gradReqTypes = new IntPointer(gradReqs != null ? gradReqs.length : 0);
    if (gradReqs != null) {
      for (int i = 0; i < gradReqs.length; i++) {
        gradReqTypes.put(i, gradReqs[i].value);
      }
    }

    if (auxStates == null) {
      auxStates = new NDArray[0];
    }

    PointerPointer auxStatesHandles = new PointerPointer();

    if (group2Ctx == null) {
      group2Ctx = Collections.emptyMap();
    }

    String[] mapKeys = new String[group2Ctx.size()];
    IntPointer map_dev_types = new IntPointer(group2Ctx.size());
    IntPointer map_dev_ids = new IntPointer(group2Ctx.size());

    int entryIndex = 0;
    for (Map.Entry<String, Context> entry : group2Ctx.entrySet()) {
      mapKeys[entryIndex] = entry.getKey();
      map_dev_types.put(entryIndex, entry.getValue().getDeviceType());
      map_dev_ids.put(entryIndex, entry.getValue().getDeviceId());
      entryIndex++;
    }

    PointerPointer executorHandle = new PointerPointer(1);

    mxnet.MXExecutorBindEX(getHandle(), ctx.getDeviceType(), ctx.getDeviceId(), group2Ctx.size(),
        new PointerPointer(mapKeys), map_dev_types, map_dev_ids, args.size(), argsHandles,
        gradsHandles, gradReqTypes, auxStates.length, auxStatesHandles,
        sharedExecutor != null ? sharedExecutor.getHandle() : new mxnet.ExecutorHandle(),
        executorHandle);

    return new Executor(new mxnet.ExecutorHandle(executorHandle.get()));
  }

  /**
   * Binds the current symbol to an executor and returns it.
   */
  public Executor bind(Context ctx, Map<String, NDArray> args, NDArray[] argsGrad, BindGradReqType[] gradReqs) {
    return bind(ctx, args, argsGrad, gradReqs, null, null, null);
  }

  /**
   * Binds the current symbol to an executor and returns it.
   */
  public Executor bind(Context ctx, Map<String, NDArray> args) {
    return bind(ctx, args, null, null);
  }

  /**
   * Bind current symbol to get an executor, allocate all the arguments needed.
   */
  public Executor simpleBind() {
    // TODO: Implement
    return null;
  }



  // Save

  /**
   * Saves symbol to a file.
   */
  public void save(Path file) {
    Util.check(mxnet.MXSymbolSaveToFile(getHandle(), file.toAbsolutePath().toString()));
  }

  /**
   * Saves symbol to a JSON string.
   */
  public String toJson() {
    PointerPointer jsonPointer = new PointerPointer(1);
    Util.check(mxnet.MXSymbolSaveToJSON(getHandle(), jsonPointer));

    return jsonPointer.getString(0);
  }


  // Miscellaneous
  // Symbol.clip 	Convenience fluent method for clip().
  // Symbol.sign 	Convenience fluent method for sign().

  // Symbol creation routines

  /**
   * Creates a symbolic variable with specified name.
   */
  public static Symbol var(String name, Map<String, String> attr, int[] shape,
                         float lrMult, float wdMult, DType dtype, Object init, String stype) {

    mxnet.SymbolHandle out = new mxnet.SymbolHandle();

    Util.check(mxnet.MXSymbolCreateVariable(name, out));

    return new Symbol(out);
  }

  public static Symbol var(String name) {
    return var(name, null, null, 0, 0, null, null, null);
  }

  /**
   * Returns a new symbol of given shape and type, filled with zeros.
   */
  public static Symbol zeros(DType dType, int... shape) { // TODO: ctx missing here?!
    Map<String, String> params = new HashMap<>();
    if (dType != null) {
      params.put("dtype", dType.toString());
    }

    params.put("shape", Util.toTupleParam(shape));

    // TODO: Documentation says to use "cast" (lower case) but it doesn't exist
    Symbol symbol = createAtomicSymbol("_zeros", params);
    compose(null, symbol, new String[]{}, new Symbol[]{});
    return symbol;
  }

  /**
   * Returns a new symbol of given shape and type, filled with ones.
   */
  public static Symbol ones(DType dtype, int... shape) {
    Map<String, String> params = new HashMap<>();
    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }

    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("_ones", params);
    compose(null, symbol, new String[]{}, new Symbol[]{});
    return symbol;
  }

  /**
   * Returns evenly spaced values within a given interval.
   */
  public static Symbol arange(DType dtype, float start, float step, int repeat,
                            float stop) {
    Map<String, String> params = new HashMap<>();

    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }

    params.put("start", Float.toString(start));
    params.put("step", Float.toString(step));
    params.put("repeat", Integer.toString(repeat));
    params.put("stop", Float.toString(stop));


    Symbol symbol = createAtomicSymbol("_arange", params);
    compose(null, symbol, new String[]{}, new Symbol[]{});
    return symbol;
  }



  // Symbol manipulation routines

  // Changing shape and type

  /**
   * Casts all elements of the input to a new type.
   */
  public static Symbol cast(Symbol data, DType type) {
    Map<String, String> params = new HashMap<>();
    params.put("dtype", type.toString());

    // TODO: Documentation says to use "cast" (lower case) but it doesn't exist
    Symbol symbol = createAtomicSymbol("Cast", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[]{data});
    return symbol;
  }

  /**
   * Reshapes the input array.
   */
  public static Symbol reshape(Symbol data, int... shape) {
    Map<String, String> params = new HashMap<>();
    params.put("shape", Util.toTupleParam(shape));

    // TODO: ``Reshape`` is deprecated, use ``reshape``
    Symbol symbol = createAtomicSymbol("Reshape", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[]{data});
    return symbol;
  }

  /**
   * Flattens the input array into a 2-D array by collapsing the higher dimensions.
   */
  public static Symbol flatten(Symbol data) {
    return createSymbol(NameManager.current().get(null, "flatten"), "Flatten", data);
  }

  /**
   * Inserts a new axis of size 1 into the array shape
   */
  public static Symbol expandDims(Symbol data, int axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Integer.toString(axis));

    Symbol symbol = createAtomicSymbol("expand_dims", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }



  // Expanding elements

  /**
   * Broadcasts the input array to a new shape.
   */
  public static Symbol broadcastTo(Symbol data, int... shape) {
    Map<String, String> params = new HashMap<>();
    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("broadcast_to", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Broadcasts the input array over particular axes.
   */
  public static Symbol broadcastAxes(Symbol data, int[] size, int... axis) {
    // TODO: NDArray is called broadcastAxis
    Map<String, String> params = new HashMap<>();
    params.put("size", Util.toTupleParam(size));
    params.put("axis", Util.toTupleParam(axis));

    Symbol symbol = createAtomicSymbol("broadcast_axis", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Repeats elements of an array.
   */
  public static Symbol repeat(Symbol data, int repeat, int axis) {
    Map<String, String> params = new HashMap<>();
    params.put("repeats", Integer.toString(repeat));
    params.put("axis", Integer.toString(axis));

    Symbol symbol = createAtomicSymbol("repeat", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Repeats the whole array multiple times.
   */
  public static Symbol tile(Symbol data, int... reps) {

    Map<String, String> params = new HashMap<>();
    params.put("reps", Util.toTupleParam(reps));

    Symbol symbol = createAtomicSymbol("tile", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Pads an input array with a constant or edge values of the array.
   */
  public static void pad() {
    // TODO: Implement
  }



  // Rearranging elements

  /**
   * Permutes the dimensions of an array.
   */
  public static Symbol transpose(Symbol data, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axes", Util.toTupleParam(axis));

    Symbol symbol = createAtomicSymbol("transpose", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Interchanges two axes of an array.
   */
  public static Symbol swapaxes(Symbol data, int dim1, int dim2) {
    Map<String, String> params = new HashMap<>();
    params.put("dim1", Integer.toString(dim1));
    params.put("dim2", Integer.toString(dim1));

    Symbol symbol = createAtomicSymbol("SwapAxis", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Reverses the order of elements along given axis while preserving array shape.
   */
  public static Symbol flip(Symbol data, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));

    Symbol symbol = createAtomicSymbol("reverse", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }



  // Joining and splitting symbols

  /**
   * Joins input arrays along a given axis.
   */
  public static void concat() {
    // TODO: Implement
  }

  /**
   * Splits an array along a particular axis into multiple sub-arrays.
   */
  public static void split(Symbol split) {
    // TODO: Implement
  }



  // Indexing routines

  /**
   * Slices a contiguous region of the array.
   */
  public static Symbol slice(Symbol data, int[] begin, int end[]) {
    Map<String, String> params = new HashMap<>();
    params.put("begin", Util.toTupleParam(begin));
    params.put("end", Util.toTupleParam(end));

    Symbol symbol = createAtomicSymbol("slice", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Slices along a given axis.
   */
  public static Symbol sliceAxis(Symbol data, int begin, int end, int axis) {
    Map<String, String> params = new HashMap<>();
    params.put("begin", Integer.toString(begin));
    params.put("end", Integer.toString(end));
    params.put("axis", Integer.toString(axis));

    Symbol symbol = createAtomicSymbol("slice_axis", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Takes elements from an input array along the given axis.
   */
  public static Symbol take(Symbol data, Symbol indices, int axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Integer.toString(axis));

    Symbol symbol = createAtomicSymbol("take", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Takes elements from a data batch.
   */
  public static void batchTake(Symbol data, Symbol indices) {
    // TODO: Implement
  }

  /**
   * Returns a one-hot array.
   */
  public static Symbol oneHotEncode(Symbol data) {
    return createSymbol("_onehot_encode", data);
  }

  /**
   * Picks elements from an input array according to the input indices along the given axis.
   */
  public static void pick() {
    // TODO: Implement
  }

  /**
   * Given three ndarrays, condition, x, and y, return an ndarray with the elements from x or y,
   * depending on the elements from condition are true or false.
   */
  public static void where() { // TODO: Missing in NDArray
    // TODO: Implement
  }



  // Arithmetic operations

  // TODO: Align these names with NDArray ?!?!

  /**
   * Returns element-wise sum of the input arrays with broadcasting.
   */
  public static Symbol broadcastAdd(Symbol lhs, Symbol rhs) {
    return createSymbol("broadcast_add", lhs, rhs);
  }

  /**
   * Returns element-wise difference of the input arrays with broadcasting.
   */
  public static Symbol broadcastSub(Symbol lhs, Symbol rhs) {
    return createSymbol("broadcast_sub", lhs, rhs);
  }

  /**
   * Returns element-wise product of the input arrays with broadcasting.
   */
  public static Symbol broadcastMul(Symbol lhs, Symbol rhs) {
    return createSymbol("broadcast_mul", lhs, rhs);
  }

  /**
   * Returns element-wise division of the input arrays with broadcasting.
   */
  public static Symbol broadcastDiv(Symbol lhs, Symbol rhs) {
    return createSymbol("broadcast_div", lhs, rhs);
  }

  /**
   * Returns element-wise modulo of the input arrays with broadcasting.
   */
  public static Symbol broadcastMod(Symbol lhs, Symbol rhs) {
    return createSymbol("broadcast_mod", lhs, rhs);
  }

  /**
   * Numerical negative of the argument, element-wise.
   */
  public static Symbol negative(Symbol data) {
    return createSymbol("negative", data);
  }

  /**
   * Returns the reciprocal of the argument, element-wise.
   */
  public static void reciprocal() { // TODO: Missing in NDArray
    // TODO: Implement
  }

  /**
   * Dot product of two arrays.
   */
  public static Symbol dot(Symbol lhs, Symbol rhs, boolean transposeA, boolean transposeB) {
    Map<String, String> params = new HashMap<>();
    params.put("transpose_a", Boolean.toString(transposeA));
    params.put("transpose_b", Boolean.toString(transposeB));

    Symbol symbol = createAtomicSymbol("dot", params);
    compose(null, symbol, new String[]{"lhs", "rhs"}, new Symbol[] {lhs, rhs});
    return symbol;
  }

  /**
   * Batchwise dot product.
   */
  public static Symbol batchFot(Symbol lhs, Symbol rhs, boolean transposeA, boolean transposeB) {
    Map<String, String> params = new HashMap<>();
    params.put("transpose_a", Boolean.toString(transposeA));
    params.put("transpose_b", Boolean.toString(transposeB));

    Symbol symbol = createAtomicSymbol("batch_dot", params);
    compose(null, symbol, new String[]{"lhs", "rhs"}, new Symbol[] {lhs, rhs});
    return symbol;
  }

  /**
   * Adds all input arguments element-wise.
   */
  public static void addN(Symbol... data) {
    // TODO: Implement
  }



  //  Trigonometric functions

  /**
   *  Computes the element-wise sine of the input array.
   */
  public static Symbol sin(Symbol data) {
    return createSymbol("sin", data);
  }

  /**
   * Computes the element-wise cosine of the input array.
   */
  public static Symbol cos(Symbol data) {
    return createSymbol("cos", data);
  }

  /**
   * Computes the element-wise tangent of the input array.
   */
  public static Symbol tan(Symbol data) {
    return createSymbol("tan", data);
  }

  /**
   * Returns element-wise inverse sine of the input array.
   */
  public static Symbol arcsin(Symbol data) {
    return createSymbol("arcsin", data);
  }

  /**
   * Returns element-wise inverse cosine of the input array.
   */
  public static Symbol arccos(Symbol data) {
    return createSymbol("arccos", data);
  }

  /**
   * Returns element-wise inverse tangent of the input array.
   */
  public static Symbol arctan(Symbol data) {
    return createSymbol("arctan", data);
  }

  /**
   * Given the “legs” of a right triangle, returns its hypotenuse.
   */
  public static Symbol hypot(Symbol lhs, Symbol rhs) {
    return createSymbol("_hypot", lhs, rhs);
    // TODO: ONLY in Symbol
  }

  /**
   * Returns the hypotenuse of a right angled triangle, given its “legs” with broadcasting.
   */
  public static Symbol broadcastHypot(Symbol lhs, Symbol rhs) {
    return createSymbol("broadcast_hypot", lhs, rhs);
    // TODO: ONLY in Symbol
  }

  /**
   * Converts each element of the input array from radians to degrees.
   */
  public static Symbol degrees(Symbol data) {
    return createSymbol("degrees", data);
  }

  /**
   * Converts each element of the input array from degrees to radians.
   */
  public static Symbol radians(Symbol data) {
    return createSymbol("radians", data);
  }



  //  Hyperbolic functions
  /**
   * Returns the hyperbolic sine of the input array, computed element-wise.
   */
  public static Symbol sinh(Symbol data) {
    return createSymbol("sinh", data);
  }

  /**
   * Returns the hyperbolic cosine of the input array, computed element-wise.
   */
  public static Symbol cosh(Symbol data) {
    return createSymbol("cosh", data);
  }

  /**
   * Returns the hyperbolic tangent of the input array, computed element-wise.
   */
  public static Symbol tanh(Symbol data) {
    return createSymbol("tanh", data);
  }

  /**
   * Returns the element-wise inverse hyperbolic sine of the input array, computed element-wise.
   */
  public static Symbol arcsinh(Symbol data) {
    return createSymbol("arcsinh", data);
  }

  /**
   * Returns the element-wise inverse hyperbolic cosine of the input array, computed element-wise.
   */
  public static Symbol arccosh(Symbol data) {
    return createSymbol("arccosh", data);
  }

  /**
   * Returns the element-wise inverse hyperbolic tangent of the input array, computed element-wise.
   */
  public static Symbol arctanh(Symbol data) {
    return createSymbol("arctanh", data);
  }



  //  Reduce functions

  /**
   * Computes the sum of array elements over given axes.
   */
  public static Symbol sum(Symbol data, boolean keepdims, boolean exclude, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));
    params.put("keepdims", Boolean.toString(keepdims));
    params.put("exclude", Boolean.toString(exclude));

    Symbol symbol = createAtomicSymbol("sum", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Computes the sum of array elements over given axes treating Not a Numbers (NaN) as zero.
   */
  public static Symbol nansum(Symbol data, boolean keepdims, boolean exclude, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));
    params.put("keepdims", Boolean.toString(keepdims));
    params.put("exclude", Boolean.toString(exclude));

    Symbol symbol = createAtomicSymbol("nansum", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Computes the product of array elements over given axes.
   */
  public static Symbol prod(Symbol data, boolean keepdims, boolean exclude, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));
    params.put("keepdims", Boolean.toString(keepdims));
    params.put("exclude", Boolean.toString(exclude));

    Symbol symbol = createAtomicSymbol("prod", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Computes the product of array elements over given axes treating Not a Numbers (NaN) as one.
   */
  public static Symbol nanprod(Symbol data, boolean keepdims, boolean exclude, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));
    params.put("keepdims", Boolean.toString(keepdims));
    params.put("exclude", Boolean.toString(exclude));

    Symbol symbol = createAtomicSymbol("nanprod", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Computes the mean of array elements over given axes.
   */
  public static Symbol mean(Symbol data, boolean keepdims, boolean exclude, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));
    params.put("keepdims", Boolean.toString(keepdims));
    params.put("exclude", Boolean.toString(exclude));

    Symbol symbol = createAtomicSymbol("mean", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Computes the max of array elements over given axes.
   */
  public static Symbol max(Symbol data, boolean keepdims, boolean exclude, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));
    params.put("keepdims", Boolean.toString(keepdims));
    params.put("exclude", Boolean.toString(exclude));

    Symbol symbol = createAtomicSymbol("max", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Computes the min of array elements over given axes.
   */
  public static Symbol min(Symbol data, boolean keepdims, boolean exclude, int... axis) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Util.toTupleParam(axis));
    params.put("keepdims", Boolean.toString(keepdims));
    params.put("exclude", Boolean.toString(exclude));

    Symbol symbol = createAtomicSymbol("min", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Flattens the input array and then computes the l2 norm.
   */
  public static Symbol norm(Symbol data) {
    return createSymbol("norm", data);
  }



  //  Rounding

  /**
   * Returns element-wise rounded value to the nearest integer of the input.
   */
  public static Symbol round(Symbol data) {
    return createSymbol("round", data);
  }

  /**
   * Returns element-wise rounded value to the nearest integer of the input.
   */
  public static Symbol rint(Symbol data) {
    return createSymbol("rint", data);
  }

  /**
   * Returns element-wise rounded value to the nearest integer towards zero of the input.
   */
  public static Symbol fix(Symbol data) {
    return createSymbol("fix", data);
  }

  /**
   * Returns element-wise floor of the input.
   */
  public static Symbol floor(Symbol data) {
    return createSymbol("floor", data);
  }

  /**
   * Returns element-wise ceiling of the input.
   */
  public static Symbol ceil(Symbol data) {
    return createSymbol("ceil", data);
  }

  /**
   * Return the element-wise truncated value of the input.
   */
  public static Symbol trunc(Symbol data) {
    return createSymbol("trunc", data);
  }



  //  Exponents and logarithms

  /**
   * Returns element-wise exponential value of the input.
   */
  public static Symbol exp(Symbol data) {
    return createSymbol("exp", data);
  }

  /**
   * Returns exp(x) - 1 computed element-wise on the input.
   */
  public static Symbol expm1(Symbol data) {
    return createSymbol("expm1", data);
  }

  /**
   * Returns element-wise Natural logarithmic value of the input.
   */
  public static Symbol log(Symbol data) {
    return createSymbol("log", data);
  }

  /**
   * Returns element-wise Base-10 logarithmic value of the input.
   */
  public static Symbol log10(Symbol data) {
    return createSymbol("log10", data);
  }

  /**
   * Returns element-wise Base-2 logarithmic value of the input.
   */
  public static Symbol log2(Symbol data) {
    return createSymbol("log2", data);
  }

  /**
   * Returns element-wise log(1 + x) value of the input.
   */
  public static Symbol log1p(Symbol data) {
    return createSymbol("log1p", data);
  }



  //  Powers

  /**
   * Returns result of first array elements raised to powers from second array, element-wise with broadcasting.
   */
  public static Symbol broadcast_power(Symbol lhs, Symbol rhs) {
    return createSymbol("_power", lhs, rhs);
  }

  /**
   * Returns element-wise square-root value of the input.
   */
  public static Symbol sqrt(Symbol data) {
    return createSymbol("sqrt", data);
  }

  /**
   * Returns element-wise inverse square-root value of the input.
   */
  public static Symbol rsqrt(Symbol data) {
    return createSymbol("rsqrt", data);
  }

  /**
   * Returns element-wise cube-root value of the input.
   */
  public static Symbol cbrt(Symbol data) {
    // TODO: Does that work ?!?!!
    return createSymbol("cbrt", data);
  }

  /**
   * Returns element-wise inverse cube-root value of the input.
   */
  public static Symbol rcbrt(Symbol data) {
    // TODO: Does that work ?!?!!
    return createSymbol("rcbrt", data);
  }

  /**
   * Returns element-wise squared value of the input.
   */
  public static Symbol square(Symbol data) {
    return createSymbol("square", data);
  }



  //  Logic functions

  /**
   * Returns the result of element-wise equal to (==) comparison operation with broadcasting.
   */
  public static Symbol broadcast_equal(Symbol lhs, Symbol rhs) {
    Symbol symbol = createAtomicSymbol("broadcast_equal", null);
    compose(null, symbol, new String[]{"lhs", "rhs"}, new Symbol[] {lhs, rhs});
    return symbol;
  }

  /**
   * Returns the result of element-wise not equal to (!=) comparison operation with broadcasting.
   */
  public static Symbol broadcast_not_equal(Symbol lhs, Symbol rhs) {
    Symbol symbol = createAtomicSymbol("broadcast_not_equal", null);
    compose(null, symbol, new String[]{"lhs", "rhs"}, new Symbol[] {lhs, rhs});
    return symbol;
  }

  /**
   * Returns the result of element-wise greater than (>) comparison operation with broadcasting.
   */
  public static Symbol broadcast_greater(Symbol lhs, Symbol rhs) {
    Symbol symbol = createAtomicSymbol("broadcast_greater", null);
    compose(null, symbol, new String[]{"lhs", "rhs"}, new Symbol[] {lhs, rhs});
    return symbol;
  }

  /**
   * Returns the result of element-wise greater than or equal to (>=) comparison operation with broadcasting.
   */
  public static Symbol broadcast_greater_equal(Symbol lhs, Symbol rhs) {
    Symbol symbol = createAtomicSymbol("broadcast_greater_equal", null);
    compose(null, symbol, new String[]{"lhs", "rhs"}, new Symbol[] {lhs, rhs});
    return symbol;
  }

  /**
   * Returns the result of element-wise lesser than (<) comparison operation with broadcasting.
   */
  public static Symbol broadcast_lesser(Symbol lhs, Symbol rhs) {
    Symbol symbol = createAtomicSymbol("broadcast_lesser", null);
    compose(null, symbol, new String[]{"lhs", "rhs"}, new Symbol[] {lhs, rhs});
    return symbol;
  }

  /**
   * Returns the result of element-wise lesser than or equal to (<=) comparison operation with broadcasting.
   */
  public static Symbol broadcastLesserEqual(Symbol lhs, Symbol rhs) {
    Symbol symbol = createAtomicSymbol("broadcast_lesser_equal", null);
    compose(null, symbol, new String[]{"lhs", "rhs"}, new Symbol[] {lhs, rhs});
    return symbol;
  }



  //  Random sampling

  /**
   * Draw random samples from a uniform distribution.
   */
  public static Symbol randomUniform(DType dtype, float low, float high, int... shape) {
    Map<String, String> params = new HashMap<>();

    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }

    params.put("low", Float.toString(low));
    params.put("high", Float.toString(high));
    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("_random_uniform", params);
    compose(null, symbol, new String[]{}, new Symbol[] {});
    return symbol;
  }

  /**
   * Draw random samples from a normal (Gaussian) distribution.
   */
  public static Symbol randomNormal(DType dtype, float loc, float scale, int... shape) {
    Map<String, String> params = new HashMap<>();

    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }

    params.put("loc", Float.toString(loc));
    params.put("scale", Float.toString(scale));
    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("_random_normal", params);
    compose(null, symbol, new String[]{}, new Symbol[] {});
    return symbol;
  }

  /**
   * Draw random samples from a gamma distribution.
   */
  public static Symbol randomGamma(DType dtype, float alpha, float beta, int... shape) {
    Map<String, String> params = new HashMap<>();

    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }

    params.put("alpha", Float.toString(alpha));
    params.put("beta", Float.toString(beta));
    params.put("shape", Util.toTupleParam(shape));


    Symbol symbol = createAtomicSymbol("_random_gamma", params);
    compose(null, symbol, new String[]{}, new Symbol[] {});
    return symbol;
  }

  /**
   * Draw random samples from an exponential distribution.
   */
  public static Symbol randomExponential(DType dtype, float lam, int... shape) {
    Map<String, String> params = new HashMap<>();

    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }

    params.put("lam", Float.toString(lam));
    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("_random_exponential", params);
    compose(null, symbol, new String[]{}, new Symbol[] {});
    return symbol;
  }

  /**
   * Draw random samples from a Poisson distribution.
   */
  public static Symbol randomPoisson(DType dtype, float lam, int... shape) {
    Map<String, String> params = new HashMap<>();
    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }
    params.put("lam", Float.toString(lam));
    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("_random_poisson", params);
    compose(null, symbol, new String[]{}, new Symbol[] {});
    return symbol;
  }

  /**
   * Draw random samples from a negative binomial distribution.
   */
  public static Symbol randomNegativeBinomial(DType dtype, int k, float p, int... shape) {
    Map<String, String> params = new HashMap<>();
    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }
    params.put("k", Integer.toString(k));
    params.put("p", Float.toString(p));
    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("_random_negative_binomial", params);
    compose(null, symbol, new String[]{}, new Symbol[] {});
    return symbol;
  }

  /**
   * Draw random samples from a generalized negative binomial distribution.
   */
  public static Symbol randomGeneralizedNegativeBinomial(DType dtype, float mu,
                                                       float alpha, int... shape) {
    Map<String, String> params = new HashMap<>();
    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }
    params.put("mu", Float.toString(mu));
    params.put("alpha", Float.toString(alpha));
    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("_random_generalized_negative_binomial", params);
    compose(null, symbol, new String[]{}, new Symbol[] {});
    return symbol;
  }

  /**
   * Concurrent sampling from multiple uniform distributions on the intervals given by [low,high).
   */
  public static Symbol sampleUniform(DType dtype, Symbol low, Symbol high, int... shape) {
    Map<String, String> params = new HashMap<>();

    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }

    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("sample_uniform", params);
    compose(null, symbol, new String[]{"low", "high"}, new Symbol[] {low, high});
    return symbol;
  }

  /**
   * Concurrent sampling from multiple normal distributions with parameters mu (mean) and sigma (standard deviation).
   */
  public static Symbol sampleNormal(DType dtype, Symbol mu, Symbol sigma, int... shape) {
    Map<String, String> params = new HashMap<>();

    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }

    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("sample_normal", params);
    compose(null, symbol, new String[]{"mu", "sigma"}, new Symbol[] {mu, sigma});
    return symbol;
  }

  /**
   * Concurrent sampling from multiple gamma distributions with parameters alpha (shape) and beta (scale).
   */
  public static Symbol sampleGamma(DType dtype, Symbol alpha, Symbol beta, int... shape) {
    Map<String, String> params = new HashMap<>();

    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }

    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("sample_gamma", params);
    compose(null, symbol, new String[]{"alpha", "beta"}, new Symbol[] {alpha, beta});
    return symbol;
  }

  /**
   * Concurrent sampling from multiple exponential distributions with parameters lambda (rate).
   */
  public static Symbol sampleExponential(DType dtype, Symbol lam, int... shape) {
    Map<String, String> params = new HashMap<>();
    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }
    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("sample_exponential", params);
    compose(null, symbol, new String[]{"lam"}, new Symbol[] {lam});
    return symbol;
  }

  /**
   * Concurrent sampling from multiple Poisson distributions with parameters lambda (rate).
   */
  // TODO: Second one with lam as type NDArray ?!?!
  public static Symbol samplePoisson(DType dtype, Symbol lam, int... shape) {
    Map<String, String> params = new HashMap<>();
    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }
    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("sample_poisson", params);
    compose(null, symbol, new String[]{"lam"}, new Symbol[] {lam});
    return symbol;
  }

  /**
   * Concurrent sampling from multiple negative binomial distributions with parameters k (failure limit) and p (failure probability).
   */
  public static Symbol sampleNegativeBinomial(DType dtype, Symbol k, Symbol p, int... shape) {
    Map<String, String> params = new HashMap<>();
    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }
    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("sample_negative_binomial", params);
    compose(null, symbol, new String[]{"k", "p"}, new Symbol[] {k, p});
    return symbol;
  }

  /**
   * Concurrent sampling from multiple generalized negative binomial distributions
   * with parameters mu (mean) and alpha (dispersion).
   */
  public static Symbol sampleGeneralizedNegativeBinomial(DType dtype, Symbol mu, int... shape) {
    Map<String, String> params = new HashMap<>();
    if (dtype != null) {
      params.put("dtype", dtype.toString());
    }
    params.put("shape", Util.toTupleParam(shape));

    Symbol symbol = createAtomicSymbol("sample_generalized_negative_binomial", params);
    compose(null, symbol, new String[]{"mu"}, new Symbol[] {mu});
    return symbol;
  }

  //  mxnet.random.seed 	Seeds the random number generators in MXNet.



  //  Sorting and searching

  /**
   * Returns a sorted copy of an input array along the given axis.
   */
  public static Symbol sort(Symbol data, int axis, boolean isAscend) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Integer.toString(axis));
    params.put("is_ascend", Boolean.toString(isAscend));

    Symbol symbol = createAtomicSymbol("sort", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Returns the top k elements in an input array along the given axis.
   */
  // TODO: move the enum to separate file !?!?!?!!?
  public static Symbol topk(Symbol data, int axis, int k, NDArray.TopkReturnType returnType, boolean isAscend) {

    Map<String, String> params = new HashMap<>();
    params.put("axis", Integer.toString(axis));
    params.put("k", Integer.toString(axis));
    params.put("ret_typ", returnType.value);
    params.put("is_ascend", Boolean.toString(isAscend));

    Symbol symbol = createAtomicSymbol("topk", params);
    compose(null, symbol, new String[]{"data"}, new Symbol[] {data});
    return symbol;
  }

  /**
   * Returns the indices that would sort an input array along the given axis.
   *
   * This function performs sorting along the given axis and returns an array of indices
   * having same shape as an input array that index data in sorted order.
   */
  public static Symbol argsort(Symbol symbol, int axis, boolean isAscend) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Integer.toString(axis));
    params.put("is_ascend", Boolean.toString(isAscend));

    Symbol argsortSymbol = createAtomicSymbol("argsort", params);
    compose(null, symbol, new String[]{"data"},
        new Symbol[] {symbol});
    return argsortSymbol;
  }


  /**
   * Returns indices of the maximum values along an axis.
   *
   * In the case of multiple occurrences of maximum values, the indices corresponding to
   * the first occurrence are returned.
   */
  public static Symbol argmax(Symbol data, int axis, boolean keepDims) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Integer.toString(axis));
    params.put("keepdims", Boolean.toString(keepDims));

    Symbol result = createAtomicSymbol("argmax", params);
    compose(null, result, new String[]{"data"},
        new Symbol[] {data});
    return result;
  }

  /**
   * Returns indices of the minimum values along an axis.
   *
   * In the case of multiple occurrences of minimum values, the indices corresponding to the
   * first occurrence are returned.
   */
  public static Symbol argmin(Symbol data, int axis, boolean keepDims) {
    Map<String, String> params = new HashMap<>();
    params.put("axis", Integer.toString(axis));
    params.put("keepdims", Boolean.toString(keepDims));

    Symbol result = createAtomicSymbol("argmin", params);
    compose(null, result, new String[] {"data"}, new Symbol[] {data});
    return result;
  }



  //  Linear Algebra

  /**
   * Performs general matrix multiplication and accumulation.
   */
  public static Symbol linalgGemm(Symbol a, Symbol b, Symbol c, boolean transposeA,
                                  boolean transposeB, double alpha, double beta) {
    Map<String, String> params = new HashMap<>();
    params.put("transpose_a", Boolean.toString(transposeA));
    params.put("transpose_b", Boolean.toString(transposeB));
    params.put("alpha", Double.toString(alpha));
    params.put("beta", Double.toString(beta));

    Symbol symbol = createAtomicSymbol("_linalg_gemm", null);
    compose(null, symbol, new String[]{"A", "B", "C"}, new Symbol[] {a, b, c});
    return symbol;
  }

  /**
   * Performs general matrix multiplication.
   */
  public static Symbol linalgGemm2(Symbol a, Symbol b, Symbol c, boolean transposeA,
                                 boolean transposeB, double alpha) {
    Map<String, String> params = new HashMap<>();
    params.put("transpose_a", Boolean.toString(transposeA));
    params.put("transpose_b", Boolean.toString(transposeB));
    params.put("alpha", Double.toString(alpha));

    Symbol symbol = createAtomicSymbol("_linalg_gemm2", null);
    compose(null, symbol, new String[]{"A", "B", "C"}, new Symbol[] {a, b, c});
    return symbol;
  }

  /**
   * Performs Cholesky factorization of a symmetric positive-definite matrix.
   */
  public static Symbol linalgPotrf(Symbol data) {
    return createSymbol("_linalg_potrf", data);
  }

  /**
   * Performs matrix inversion from a Cholesky factorization.
   */
  public static Symbol linalgPotri(Symbol data) {
    return createSymbol("_linalg_potri", data);
  }

  /**
   * Performs multiplication with a triangular matrix.
   */
  public static Symbol linalgTrmm(Symbol a, Symbol b, boolean transpose, boolean rightside,
                                 double alpha) {
    Map<String, String> params = new HashMap<>();
    params.put("transpose", Boolean.toString(transpose));
    params.put("rightside", Boolean.toString(rightside));
    params.put("alpha", Double.toString(alpha));

    Symbol symbol = createAtomicSymbol("_linalg_trmm", null);
    compose(null, symbol, new String[]{"A", "B"}, new Symbol[] {a, b});
    return symbol;
  }

  /**
   * Solves matrix equations involving a triangular matrix.
   */
  public static Symbol linalgTrsm(Symbol a, Symbol b, boolean transpose, boolean rightside,
                                 double alpha) {
    Map<String, String> params = new HashMap<>();
    params.put("transpose", Boolean.toString(transpose));
    params.put("rightside", Boolean.toString(rightside));
    params.put("alpha", Double.toString(alpha));

    Symbol symbol = createAtomicSymbol("_linalg_trsm", null);
    compose(null, symbol, new String[]{"A", "B"}, new Symbol[] {a, b});
    return symbol;
  }

  /**
   * Computes the sum of the logarithms of all diagonal elements in a matrix.
   */
  public static Symbol linalgSumlogdiag(Symbol data) {
    return createSymbol("_linalg_sumlogdiag", data);
  }



  //  Miscellaneous

  /**
   * Returns element-wise maximum of the input elements.
   */
  public static Symbol maximum(Symbol lhs, Symbol rhs) {
    Symbol symbol = createAtomicSymbol("_maximum", null);
    compose(null, symbol, new String[]{"lhs", "rhs"}, new Symbol[] {lhs, rhs});
    return symbol;
  }

  /**
   * Returns element-wise minimum of the input elements.
   */
  public static Symbol minimum(Symbol lhs, Symbol rhs) {
    Symbol symbol = createAtomicSymbol("_minimum", null);
    compose(null, symbol, new String[]{"lhs", "rhs"}, new Symbol[] {lhs, rhs});
    return symbol;
  }

  /**
   * Returns element-wise maximum of the input arrays with broadcasting.
   */
  public static Symbol broadcastMaximum(Symbol lhs, Symbol rhs) {
    Symbol symbol = createAtomicSymbol("broadcast_maximum", null);
    compose(null, symbol, new String[]{"lhs", "rhs"}, new Symbol[] {lhs, rhs});
    return symbol;
  }

  /**
   * Returns element-wise minimum of the input arrays with broadcasting.
   */
  public static Symbol broadcastMinimum(Symbol lhs, Symbol rhs) {
    Symbol symbol = createAtomicSymbol("broadcast_minimum", null);
    compose(null, symbol, new String[]{"lhs", "rhs"}, new Symbol[] {lhs, rhs});
    return symbol;
  }

  /**
   * Clips (limits) the values in an array.
   */
  public static Symbol clip(Symbol data, int min, int max) {

    Map<String, String> params = new HashMap<>();
    params.put("a_min", Integer.toString(min));
    params.put("a_max", Integer.toString(max));

    Symbol result = createAtomicSymbol("clip", params);
    compose(null, result, new String[] {"data"}, new Symbol[] {data});
    return result;
  }

  /**
   * Returns element-wise absolute value of the input.
   */
  public static Symbol abs(Symbol data) {
    return createSymbol("abs", data);
  }

  /**
   * Returns element-wise sign of the input.
   */
  public static Symbol sign(Symbol data) {
    return createSymbol("sign", data);
  }

  /**
   * Returns the gamma function (extension of the factorial function to the reals) , computed element-wise on the input array.
   */
  public static Symbol gamma(Symbol data) {
    return createSymbol("gamma", data);
  }

  /**
   * Returns element-wise log of the absolute value of the gamma function of the input.
   */
  public static Symbol gammaln(Symbol data) {
    return createSymbol("gammaln", data);
  }



  //  Neural network - Basic

  /**
   * Applies a linear transformation: Y=XWT+b
   */
  public static Symbol fullyConnected(Symbol data, Symbol weight, Symbol bias, int numHidden,
                                    // boolean noBias, TODO: Support this!
                                    Boolean flatten) {
    Map<String, String> params = new HashMap<>();
    params.put("num_hidden", Integer.toString(numHidden));

    if (flatten != null) {
      params.put("flatten", flatten.toString());
    }

    Map<String, Symbol> args = new HashMap<>();
    args.put("data", data);
    args.put("weight", weight);
    args.put("bias", bias);

    Symbol result = createAtomicSymbol("FullyConnected", params);
    compose(NameManager.current().get(null, "fullyconnected"), result, args);
    return result;
  }

  public static Symbol fullyConnected(Symbol data, int numHidden) {
    return fullyConnected(data, null, null, numHidden, null);
  }

  /**
   * Compute N-D convolution on (N+2)-D input.
   */
  public static Symbol convolution(Symbol data, int[] kernel, int numFilter) {
    // TODO: Implement
    return null;
  }

  /**
   * Applies an activation function element-wise to the input.
   */
  public static Symbol activation(Symbol data, NDArray.ActivationType actType) {
    Map<String, String> params = new HashMap<>();
    params.put("act_type", actType.value);

    Symbol result = createAtomicSymbol("Activation", params);

    compose(NameManager.current().get(null, "activation"), result, new String[] {"data"},
        new Symbol[] {data});
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

  public enum SoftmaxNormalizationType {
    BATCH("batch"),
    NULL("null"),
    VALID("valid");

    final String value;

    SoftmaxNormalizationType(String value) {
      this.value = value;
    }
  }

  /**
   * Computes the gradient of cross entropy loss with respect to softmax output.
   */
  public static Symbol softmaxOutput(Symbol data, Symbol label, float gradScale, float ignoreLabel,
                                   boolean multiOutput, boolean useIgnore, boolean preserveShape,
                                   SoftmaxNormalizationType normalization,
                                   boolean outGrad) {

    Map<String, String> params = new HashMap<>();
    params.put("grad_scale", Float.toString(gradScale));
    params.put("ignore_label", Float.toString(ignoreLabel));
    params.put("multi_output", Boolean.toString(multiOutput));
    params.put("use_ignore", Boolean.toString(useIgnore));
    params.put("preserve_shape", Boolean.toString(preserveShape));

    if (normalization != null) {
      params.put("normalization", normalization.value);
    }

    params.put("out_grad", Boolean.toString(outGrad));

    Symbol result = createAtomicSymbol("SoftmaxOutput", params);

    Map<String, Symbol> args = new HashMap<>();
    args.put("data", data);
    args.put("label", label);

    compose("softmax", result, args);

    return result;
  }

  /**
   * Applies the softmax function.
   */
  public static void softmax() {
    // TODO: Implement
  }

  /**
   * Computes the log softmax of the input.
   */
  public static void  logSoftmax() {
    // TODO: Implement
  }



  //  Neural network - More

  /**
   * Applies correlation to inputs.
   */
  public static void correlation() {
    // TODO: Implement
  }

  /**
   * Computes 2D transposed convolution (aka fractionally strided convolution) of the input tensor.
   */
  public static void  deconvolution() {
    // TODO: Implement
  }

  /**
   * Applies a recurrent layer to input.
   */
  public static void rnn() {
    // TODO: Implement
  }

  /**
   * Maps integer indices to vector representations (embeddings).
   */
  public static void embedding() {
    // TODO: Implement
  }

  /**
   * Applies Leaky rectified linear unit activation element-wise to the input.
   */
  public static void leakyRelu() {
    // TODO: Implement
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
  public static void  lrn() {
    // TODO: Implement
  }

  /**
   * Performs region of interest(ROI) pooling on the input array.
   */
  public static void  roiPooling() {
    // TODO: Implement
  }

  /**
   * Applies softmax activation to input.
   */
  public static void  softmaxActivation() {
    // TODO: Implement
  }

  /**
   * Applies dropout operation to input array.
   */
  public static void  dropout() {
    // TODO: Implement
  }

  /**
   * Applies bilinear sampling to input feature map.
   */
  public static void  bilinearSampler() {
    // TODO: Implement
  }

  /**
   * Generates 2D sampling grid for bilinear sampling.
   */
  public static void  gridGenerator() {
    // TODO: Implement
  }

  /**
   * Performs nearest neighbor/bilinear up sampling to inputs.
   */
  public static void  upSampling() {
    // TODO: Implement
  }

  /**
   * Applies a spatial transformer to input feature map.
   */
  public static void  spatialTransformer() {
    // TODO: Implement
  }

  /**
   * Computes and optimizes for squared loss during backward propagation.
   */
  public static void  linearRegressionOutput() {
    // TODO: Implement
  }

  /**
   * Applies a logistic function to the input.
   */
  public static void  logisticRegressionOutput() {
    // TODO: Implement
  }

  /**
   * Computes mean absolute error of the input.
   */
  public static void  maeRegressionOutput() {
    // TODO: Implement
  }

  /**
   * Computes support vector machine based transformation of the input.
   */
  public static void  svmOutput() {
    // TODO: Implement
  }

  /**
   * Calculate cross entropy of softmax output and one-hot label.
   */
  public static void  softmaxCrossEntropy() {
    // TODO: Implement
  }

  /**
   * Calculate Smooth L1 Loss(lhs, scalar) by summing
   */
  public static void  smoothL1(Symbol data, float scalar) {
    // TODO: Implement
  }

  /**
   * Apply a sparse regularization to the output a sigmoid activation function.
   */
  public static void  identityAttachKLSparseReg() {
    // TODO: Implement
  }

  /**
   * Make your own loss function in network construction.
   */
  public static void  makeLoss() {
    // TODO: Implement
  }

  /**
   * Stops gradient computation.
   */
  public static void  blockGrad() {
    // TODO: Implement
  }

  /**
   * Apply a custom operator implemented in a frontend language (like Python).
   */
  public static void  custom() {
    // TODO: Implement
  }

  public void dispose() {
    mxnet.MXSymbolFree(getHandle());
    handle = null;
  }

  @Override
  public String toString() {
    if (handle != null) {
      PointerPointer debugString = new PointerPointer(1);

      Util.check(mxnet.MXSymbolPrint(getHandle(), debugString));

      return debugString.getString(0);
    }
    else {
      return "NULL";
    }
  }
}
