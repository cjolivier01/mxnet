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
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <sstream>

/*!
 * \brief Some string utility functions that aren't specific to tuning
 */
struct StringUtil {
  /*!
   * \brief Terim whitespace from beninning and end of string
   * \param s String to trimp
   * \return reference to the modified string. This is the same std::string object as what was
   *         supplied in the parameters
   */
  static std::string &trim(std::string *s) {
    s->erase(s->begin(), std::find_if(s->begin(), s->end(), [](int ch) {
      return !std::isspace(ch);
    }));
    s->erase(std::find_if(s->rbegin(), s->rend(), [](int ch) {
      return !std::isspace(ch);
    }).base(), s->end());
    return *s;
  }
  static void replaceAll(std::string& str, const std::string& from, const std::string& to) {
    if(from.empty())
      return;
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
      str.replace(start_pos, from.length(), to);
      start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
  }};

#define REPLACE_TOKEN "%ITEM%"

int main(int argc, char **argv) {
  if(argc != 4) {
    std::cerr << "Usage:" << std::endl
              << argv[0]
              << " <input file> <template file> <output file>" << std::endl;
  }
  std::ifstream input, tmpl;
  input.open(argv[1]);
  if(input.fail()) {
    std::cerr << "Error opening file: \"" << argv[1] << "\": " << strerror(errno) << std::endl;
    return errno;
  }
  tmpl.open(argv[2]);
  if(tmpl.fail()) {
    std::cerr << "Error opening file: \"" << argv[2] << "\": " << strerror(errno) << std::endl;
    return errno;
  }

  std::ofstream output;
  output.open(argv[3], std::ios_base::trunc|std::ios_base::out);
  if(output.fail()) {
    std::cerr << "Error opening file: \"" << argv[3] << "\": " << strerror(errno) << std::endl;
    return errno;
  }

  std::stringstream tmpl_buff;
  while(!tmpl.eof()) {
    std::string line;
    tmpl >> line;
    tmpl_buff << line;
  }
  tmpl.close();

  std::string tmpl_string = tmpl_buff.str();
  StringUtil::trim(&tmpl_string);

  while(!input.eof()) {
    std::string token;
    input >> token;
    StringUtil::trim(&token);
    if(!token.empty()) {
      std::string replaced = tmpl_string;
      StringUtil::replaceAll(replaced, REPLACE_TOKEN, token);
      output << replaced << std::endl;
      if(output.fail()) {
        std::cerr << "Error writing file: \"" << argv[3] << "\": " << strerror(errno) << std::endl;
        return errno;
      }
    }
  }
  output.close();
  input.close();

  return 0;
}