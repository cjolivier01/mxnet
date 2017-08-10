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

/**
 * Data-type of the array’s elements.
 */
public class DType {

  public static final DType FLOAT_16 = new DType(2);
  public static final DType FLOAT_32 = new DType(0);
  public static final DType FLOAT_64 = new DType(1);
  public static final DType UINT_8 = new DType(3);
  public static final DType INT_32 = new DType(4);

  private final int dtype;

  DType(int dtype) {
    if (dtype < 0 || dtype > 4) {
      throw new IllegalArgumentException("dtype is too small/large: " + dtype);
    }

    this.dtype = dtype;
  }

  int getDType() {
    return dtype;
  }

  public int numberOfBytes() {
    switch (dtype) {
      case 0:
        return 4;
      case 1:
        return 8;
      case 2:
        return 2;
      case 3:
        return 1;
      case 4:
        return 4;
      default:
        throw new IllegalStateException("Invalid data type!");
    }
  }

  public String toString() {
    switch (dtype) {
      case 0:
        return "float32";
      case 1:
        return "float64";
      case 2:
        return "float16";
      case 3:
        return "uint8";
      case 4:
        return "int32";
      default:
        throw new IllegalStateException("Invalid data type!");
    }
  }

  @Override
  public int hashCode() {
    return dtype;
  }

  @Override
  public boolean equals(Object o) {
    if (o == this) {
      return true;
    }

    if (o instanceof DType) {
      DType s = (DType) o;

      return dtype == s.dtype;
    }

    return false;
  }
}
