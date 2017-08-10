package org.apache.mxnet;

import org.apache.mxnet.javacpp.mxnet;

public class Util {

  static final int TRUE = 0;
  static final int FALSE = 1;

  static void check(int status) {
    if (status == -1) {
      throw new MXNetError(mxnet.MXGetLastError().getString());
    }
  }

  static String toTupleParam(int... axis) {
    StringBuilder axisParam = new StringBuilder();
    axisParam.append('(');
    for (int a : axis) {
      axisParam.append(a + ",");
    }
    axisParam.append(')');

    return axisParam.toString();
  }
}
