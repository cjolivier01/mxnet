package org.apache.mxnet.javacpp.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

// TODO: Currently the build must be run twice, first to generate mxnet.java and then to compile it

@Properties(target = "org.apache.mxnet.javacpp.mxnet", value = {
    @Platform(value = {"linux-x86"}, compiler = "cpp11", define = {"DMLC_USE_CXX11 1", "MSHADOW_USE_CBLAS 1", "MSHADOW_IN_CXX11 1"},
        include = {"mxnet/c_api.h", "mxnet/c_predict_api.h", },
        link = "mxnet", resource = {"include", "lib"}, includepath = {"/usr/local/cuda/include/",
        "/System/Library/Frameworks/vecLib.framework/", "/System/Library/Frameworks/Accelerate.framework/"}, linkpath = "/usr/local/cuda/lib/") })
public class mxnet implements InfoMapper {
  public void map(InfoMap infoMap) {
    infoMap.put(new Info("MXNET_EXTERN_C", "MXNET_DLL").cppTypes().annotations())
        .put(new Info("NDArrayHandle").valueTypes("NDArrayHandle").pointerTypes("PointerPointer", "@Cast(\"NDArrayHandle*\") @ByPtrPtr NDArrayHandle"))

        .put(new Info("FunctionHandle").annotations("@Const").valueTypes("FunctionHandle").pointerTypes("PointerPointer", "@Cast(\"FunctionHandle*\") @ByPtrPtr FunctionHandle"))
        .put(new Info("AtomicSymbolCreator").valueTypes("AtomicSymbolCreator").pointerTypes("PointerPointer", "@Cast(\"AtomicSymbolCreator*\") @ByPtrPtr AtomicSymbolCreator"))
        .put(new Info("SymbolHandle").valueTypes("SymbolHandle").pointerTypes("PointerPointer", "@Cast(\"SymbolHandle*\") @ByPtrPtr SymbolHandle"))
        .put(new Info("AtomicSymbolHandle").valueTypes("AtomicSymbolHandle").pointerTypes("PointerPointer", "@Cast(\"AtomicSymbolHandle*\") @ByPtrPtr AtomicSymbolHandle"))
        .put(new Info("ExecutorHandle").valueTypes("ExecutorHandle").pointerTypes("PointerPointer", "@Cast(\"ExecutorHandle*\") @ByPtrPtr ExecutorHandle"))
        .put(new Info("DataIterCreator").valueTypes("DataIterCreator").pointerTypes("PointerPointer", "@Cast(\"DataIterCreator*\") @ByPtrPtr DataIterCreator"))
        .put(new Info("DataIterHandle").valueTypes("DataIterHandle").pointerTypes("PointerPointer", "@Cast(\"DataIterHandle*\") @ByPtrPtr DataIterHandle"))
        .put(new Info("KVStoreHandle").valueTypes("KVStoreHandle").pointerTypes("PointerPointer", "@Cast(\"KVStoreHandle*\") @ByPtrPtr KVStoreHandle"))
        .put(new Info("RecordIOHandle").valueTypes("RecordIOHandle").pointerTypes("PointerPointer", "@Cast(\"RecordIOHandle*\") @ByPtrPtr RecordIOHandle"))
        .put(new Info("RtcHandle").valueTypes("RtcHandle").pointerTypes("PointerPointer", "@Cast(\"RtcHandle*\") @ByPtrPtr RtcHandle"))
        .put(new Info("OptimizerCreator").valueTypes("OptimizerCreator").pointerTypes("PointerPointer", "@Cast(\"OptimizerCreator*\") @ByPtrPtr OptimizerCreator"))
        .put(new Info("OptimizerHandle").valueTypes("OptimizerHandle").pointerTypes("PointerPointer", "@Cast(\"OptimizerHandle*\") @ByPtrPtr OptimizerHandle"));
  }
}