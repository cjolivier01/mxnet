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
 * A context describes the device type and ID on which computation should be carried on.
 * MXNet can run operations on CPU and different GPUs.
 *
 * How to run MXNet on multiple CPU/GPUs <http://mxnet.io/how_to/multi_devices.html>`
 * for more details.
 */
public class Context {

  private static final int CPU_DEVICE_ID = 1;
  private static final int CPU_PINNED_DEVICE_ID = 3;
  private static final int GPU_DEVICE_ID = 2;

  private final int deviceType;
  private final int deviceId;

  Context(int deviceType, int deviceId) {

    if (deviceType < 1 || deviceType > 3) {
      throw new IllegalArgumentException("deviceType must be 1, 2 or 3");
    }

    if (deviceId < 0) {
      throw new IllegalArgumentException("deviceId must be >= 0");
    }

    this.deviceType = deviceType;
    this.deviceId = deviceId;
  }

  int getDeviceType() {
    return deviceType;
  }

  int getDeviceId() {
    return deviceId;
  }

  String getDeviceTypeName() {
    switch (getDeviceType()) {
      case CPU_DEVICE_ID:
        return "cpu";
      case CPU_PINNED_DEVICE_ID:
        return "cpu_pinned";
      case GPU_DEVICE_ID:
        return "gpu";
      default:
        throw new IllegalStateException("Unexpected device type: " + getDeviceType());
    }
  }

  /**
   * @return context string in the format [cpu|gpu|cpu_pinned](n), as used for imperative calls.
   */
  String toContextString() {
    return String.format("%s(%d)", getDeviceTypeName(), getDeviceId());
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof Context) {
      Context ctx = (Context) obj;
      return toContextString().equals(ctx.toContextString());
    }

    return false;
  }

  @Override
  public int hashCode() {
    return toContextString().hashCode();
  }

  @Override
  public String toString() {
    // This is for debugging and might change in the future,
    // e.g. include more details about the devices, e.g. cpu model
    // or gpu model, cuda versions, etc.
    return toContextString();
  }

  public static Context cpu() {
    return new Context(CPU_DEVICE_ID, 0);
  }

  public static Context cpuPinned() {
    return new Context(CPU_DEVICE_ID, 0);
  }

  public static Context gpu(int deviceId) {
    return new Context(GPU_DEVICE_ID, deviceId);
  }
}
