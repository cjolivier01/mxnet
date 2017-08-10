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

import java.util.HashMap;
import java.util.Map;

/**
 * NameManager to do automatic naming.
 */
public class NameManager {

  private static NameManager instance;

  Map<String, Integer> counter = new HashMap<>();

  String get(String name, String hint) {
    if (name == null) {
      return hint + (counter.merge(hint, 1, (x, y) -> x + y) - 1);
    }
    return name;
  }

  static NameManager current() {
    if (instance == null) {
      instance = new NameManager();
    }
    return instance;
  }
}
