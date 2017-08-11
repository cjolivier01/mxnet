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

/*!
 * \file infer_graph_attr_pass.cc
 * \brief infer graph shape, dtype, and storage type
 */

#include <mxnet/op_attr_types.h>
#include <mxnet/graph_attr_types.h>
#include <common/utils.h>
#include "./exec_pass.h"
#ifndef NDEBUG
#include "../../tests/cpp/include/test_util.h"
#endif

namespace mxnet {
namespace exec {

template<typename AttrType, typename FInfer>
bool ApplyOpInferAttr(const nnvm::Graph& g,
                      const FInfer& finfer,
                      const NodeAttrs& attrs,
                      const uint32_t nid,
                      std::vector<AttrType>* in_attrs,
                      std::vector<AttrType>* out_attrs) {
  return finfer(attrs, in_attrs, out_attrs);
}

template<>
bool ApplyOpInferAttr<int, FInferStorageType>(const nnvm::Graph& g,
                                              const FInferStorageType& finfer,
                                              const NodeAttrs& attrs,
                                              const uint32_t nid,
                                              std::vector<int>* in_attrs,
                                              std::vector<int>* out_attrs) {
  const ContextVector& ctxes = g.GetAttr<ContextVector>("context");
  return finfer(attrs, ctxes[nid], in_attrs, out_attrs);
}

//struct Node {
//  /*! \brief pointer to the source node */
//  const nnvm::Node* source;
//  /*! \brief inputs to the node */
//  array_view<NodeEntry> inputs;
//  /*! \brief control flow dependencies to the node */
//  array_view<uint32_t> control_deps;
//};

//class Node {
// public:
//  /*! \brief The attributes in the node. */
//  NodeAttrs attrs;
//  /*! \brief inputs to this node */
//  std::vector<NodeEntry> inputs;
//  /*!
//   * \brief Optional control flow dependencies
//   *  Gives operation must be performed before this operation.
//   */
//  std::vector<NodePtr> control_deps;
//  /*! \brief destructor of node */
//  ~Node();
//  /*! \return operator in this node */
//  inline const Op* op() const;
//  /*!
//   * \brief return whether node is placeholder variable.
//   *  This is equivalent to op == nullptr
//   * \return whether node is placeholder input variable
//   */
//  inline bool is_variable() const;
//  /*! \return number of outputs from this node */
//  inline uint32_t num_outputs() const;
//  /*! \return number of inputs from this node */
//  inline uint32_t num_inputs() const;
//  /*!
//   * \brief create a new empty shared_ptr of Node.
//   * \return a created empty node.
//   */
//  static NodePtr Create();
//};

class GraphDumper {
  template<typename Pair>
  static std::string dict_item_to_string(const Pair &pair) {
    std::stringstream ss;
    const std::string &key = pair.first;
    std::string val = pair.second;
    if (key == "__storage_type__") {
      val = common::stype_string(atoi(val.c_str()));
    }
    ss << key << " = " << val;
    return ss.str();
  }

 public:
  static std::string tabbit(const size_t t) {
    std::stringstream ss;
    for (size_t i = 0; i < t; ++i) {
      ss << "  ";
    }
    return ss.str();
  }

  static std::ostream &print(std::ostream *os, const nnvm::NodeEntry &entry, size_t tabs = 0) {
    *os << tabbit(tabs) << "nnvm::NodeEntry: "
        << "index: " << entry.index
        << ", version: " << entry.version
        << std::endl;
    if(entry.node) {
      print(os, *entry.node, tabs + 1);
    }
    return *os;
  }

  static std::ostream &print(std::ostream *os,
                            const nnvm::IndexedGraph::NodeEntry &entry,
                            size_t tabs = 0) {
    return *os << tabbit(tabs) << "IndexedGraph::NodeEntry "
               << "node_id: " << entry.node_id
               << ", index: " << entry.index
               << ", version: " << entry.version
               << std::endl;
  }

  static std::ostream &print(std::ostream *os, const nnvm::NodeAttrs &attrs, size_t tabs = 0) {
    *os << tabbit(tabs) << "nnvm::NodeAttrs" << std::endl;
    ++tabs;
    if(!attrs.name.empty()) {
      *os << tabbit(tabs) << "name: " << attrs.name << std::endl;
    }
    if (attrs.op) {
      *os << tabbit(tabs) << "OP: " << attrs.op->name << std::endl;
    }
    if(!attrs.dict.empty()) {
      *os << tabbit(tabs) << "dict: " << std::endl;
      for (auto mm : attrs.dict) {
        *os << tabbit(tabs + 1) << dict_item_to_string(mm) << std::endl;
      }
    }
    return *os;
  }

  static std::ostream &print(std::ostream *os,
                            const nnvm::Node &node,
                            size_t tabs = 0) {
    *os << tabbit(tabs) << "nnvm::Node" << std::endl;
    ++tabs;
    if (const nnvm::Op *op = node.op()) {
      *os << tabbit(tabs) << "OP: " << op->name << std::endl;
    }
    print(os, node.attrs, tabs + 1);
    if (!node.inputs.empty()) {
      *os << tabbit(tabs) << "inputs: " << std::endl;
      for (const nnvm::NodeEntry &ne : node.inputs) {
        print(os, ne, tabs + 1);
      }
    }
    if (!node.control_deps.empty()) {
      *os << tabbit(tabs) << "control_deps: ";
      for (size_t i = 0, n = node.control_deps.size(); i < n; ++i) {
        *os << tabbit(tabs + 1) << "control_deps[ 0 - " << n << "]: ";
        print(os, *node.control_deps[i], tabs + 2);
      }
    }
    return *os;
  }

  static std::ostream &print(std::ostream *os,
                            const nnvm::IndexedGraph::Node &node,
                            size_t tabs = 0) {
    *os << tabbit(tabs) << "nnvm::IndexedGraph::Node" << std::endl;
    ++tabs;
    if (node.source) {
      *os << tabbit(tabs) << "source:" << std::endl;
      print(os, *node.source, tabs + 1);
    }
    if (node.inputs.size()) {
      *os << tabbit(tabs) << "inputs: " << std::endl;
      for (const nnvm::IndexedGraph::NodeEntry &ne : node.inputs) {
        print(os, ne, tabs + 1);
      }
    }
    if (node.control_deps.size()) {
      *os << tabbit(tabs) << "control_deps: ";
      for (size_t i = 0, n = node.control_deps.size(); i < n; ++i) {
        if (i) {
          std::cout << ", ";
        }
        *os << node.control_deps[i];
      }
      *os << std::endl;
    }
    return *os;
  }

  static std::ostream &print(std::ostream *os, const nnvm::IndexedGraph &idx, size_t tabs = 0) {
    *os << tabbit(tabs) << "nnvm::IndexedGraph: " << std::endl;
    ++tabs;
    for (size_t i = 0, n = idx.num_node_entries(); i < n; ++i) {
      const nnvm::IndexedGraph::Node &node = idx[i];
      print(os, node, tabs + 1);
    }
    if (!idx.input_nodes().empty()) {
      std::cout << tabbit(tabs) << "input_nodes:" << std::endl;
      for (size_t i = 0, n = idx.input_nodes().size(); i < n; ++i) {
        const uint32_t input_node_id = idx.input_nodes()[i];
        print(os, idx[input_node_id], tabs + 2);
      }
    }
    if (!idx.outputs().empty()) {
      std::cout << tabbit(tabs) << "outputs:" << std::endl;
      for (size_t i = 0, n = idx.outputs().size(); i < n; ++i) {
        const nnvm::IndexedGraph::NodeEntry &node_entry = idx.outputs()[i];
        print(os, node_entry, tabs + 2);
      }
    }
    return *os;
  }
};

/*!\brief
 * This is a duplicate of the InferAttr function in nnvm with minor modification
 * to support inferring storage type whose function signature is different from
 * shape/type inference functions'. The nnvm InferAttr will be deprecated
 * in the future. Please use interfaces InferShape, InferType, and InferStorageType
 * to call this function.
 */
template<typename AttrType, typename FInferType, typename IsNone, typename FDefault>
static nnvm::Graph InferAttr(nnvm::Graph &&ret,
                             const AttrType empty_val,
                             const char* infer_name,
                             const char* input_name,
                             const char* attr_key_name,
                             const char* attr_name,
                             const char* unknown_name,
                             IsNone fis_none,
                             FDefault fdefault,
                             bool backward_identity_assign) {
  using nnvm::IndexedGraph;
  using nnvm::Op;
  using AttrVector = std::vector<AttrType>;
  using dmlc::any;

  const IndexedGraph& idx = ret.indexed_graph();

  //GraphDumper::print(&std::cout, idx) << std::flush;

  bool trace = false;
  if(!strcmp(infer_name, "FInferStorageType")) {
    std::cout << "Inferring: " << infer_name << std::endl;
    std::cout << GraphDumper::tabbit(1) << "Input name: " << input_name << std::endl;
    std::cout << GraphDumper::tabbit(1) << "Attr name: " << attr_name << std::endl;
    trace = true;
  }

  static const nnvm::OpMap<FInferType>& finfer_callback = Op::GetAttr<FInferType>(infer_name);
  static const nnvm::OpMap<bool>& is_backward = Op::GetAttr<nnvm::TIsBackward>("TIsBackward");

  // gradient function, used to get node correspondence.
  static const nnvm::OpMap<nnvm::FGradient>& fgrad = Op::GetAttr<nnvm::FGradient>("FGradient");

  // reshape attribute vector
  AttrVector attrVector;
  if (ret.attrs.count(attr_name) != 0) {
    attrVector = ret.MoveCopyAttr<AttrVector>(attr_name);
  } else {
    attrVector.resize(idx.num_node_entries(), empty_val);
  }

  if (ret.attrs.count(input_name) != 0) {
    const auto& attr_args = ret.GetAttr<AttrVector>(input_name);
    CHECK_LE(attr_args.size(), idx.input_nodes().size())
        << "More provided " << attr_name << "s than number of arguments.";
    for (size_t i = 0; i < attr_args.size(); ++i) {
      attrVector[idx.entry_id(idx.input_nodes()[i], 0)] = attr_args[i];
    }
    // erase the provided arguments
    ret.attrs.erase(input_name);
  }

  // get the attribute hints
  std::string infer_hints_key = std::string(attr_name) + "_hints";
  if (ret.attrs.count(infer_hints_key)) {
    nnvm::NodeEntryMap<AttrType> attr_hints =
      ret.GetAttr<nnvm::NodeEntryMap<AttrType>>(infer_hints_key);
    for (const auto& kv : attr_hints) {
      nnvm::NodeEntry e = kv.first;
      if (idx.exist(e.node.get())) {
        attrVector[idx.entry_id(kv.first)] = kv.second;
      }
    }
  }

  std::string infer_attr_key;
  if (ret.attrs.count(attr_key_name) != 0) {
    infer_attr_key = ret.GetAttr<std::string>(attr_key_name);
    // erase the provided arguments
    ret.attrs.erase(attr_key_name);
  }
  // Temp space for attribute inference.
  std::vector<AttrType> i_attr, o_attr;

  // inference step function for nid
  auto infer_step = [&](uint32_t nid, bool last_iter) {
    const nnvm::IndexedGraph& _idx = idx;
    const nnvm::IndexedGraph::Node& inode = _idx[nid];
    if(trace) {
      std::cout << "==============================" << std::endl;
      GraphDumper::print(&std::cout, inode);
      std::cout << "==============================" << std::endl << std::flush;
    }
    const size_t num_inputs = inode.inputs.size();
    const size_t num_outputs = inode.source->num_outputs();
    if (inode.source->is_variable()) {
      // Variable node. No operator. Only one output entry.
      if(trace) {
        std::cout << "IS A VARIABLE" << std::endl << std::flush;
      }
      CHECK(inode.source->op() == nullptr);
      CHECK_EQ(num_outputs, 1U);
      const uint32_t out_entry_id = _idx.entry_id(nid, 0);
      if (infer_attr_key.length() != 0 && fis_none(attrVector[out_entry_id])) {
        auto it = inode.source->attrs.dict.find(infer_attr_key);
        if (it != inode.source->attrs.dict.end()) {
          std::istringstream is(it->second);
          CHECK(is >> attrVector[out_entry_id]) << "Invalid attribute";
        }
      }
    } else if (is_backward.get(inode.source->op(), false) &&
               inode.control_deps.size() && backward_identity_assign) {
      CHECK_GE(inode.control_deps.size(), 1U)
        << "BackwardOp need to have control_deps to its forward op";
      if(trace) {
        std::cout << "IS BACKWARD" << std::endl << std::flush;
      }
      const IndexedGraph::Node& fnode = _idx[inode.control_deps[0]];
      nnvm::NodePtr fwd_ptr = inode.source->control_deps[0];
      CHECK(fwd_ptr->op() != nullptr) << "Forward op cannot be a variable";
      // use gradient function to find out the correspondence.
      std::vector<nnvm::NodeEntry> ograd(fwd_ptr->num_outputs());
      for (size_t i = 0; i < ograd.size(); ++i) {
        ograd[i].index = static_cast<uint32_t>(i);
      }
      // input gradient list
      auto igrad = fgrad[fwd_ptr->op()](fwd_ptr, ograd);
      const nnvm::Node* igrad_node = nullptr;
      // Input gradient assignement
      for (size_t i = 0; i < igrad.size(); ++i) {
        if (igrad[i].node->op() == inode.source->op()) {
          uint32_t eid = _idx.entry_id(nid, igrad[i].index);
          if (fis_none(attrVector[eid])) {
            attrVector[eid] = attrVector[_idx.entry_id(fnode.inputs[i])];
          } else {
            CHECK_EQ(attrVector[eid], attrVector[_idx.entry_id(fnode.inputs[i])])
                << "Backward attribute inconsistent with the forward attribute";
          }
          if (igrad_node == nullptr) {
            igrad_node = igrad[i].node.get();
          } else {
            CHECK(igrad_node == igrad[i].node.get());
          }
        }
      }
      // out grad entries
      CHECK(igrad_node != nullptr)
        << "Cannot find matching backward op for " << inode.source->attrs.name;
      for (size_t i = 0; i < igrad_node->inputs.size(); ++i) {
        const nnvm::NodeEntry& e = igrad_node->inputs[i];
        if (e.node == nullptr) {
          uint32_t eid = _idx.entry_id(inode.inputs[i]);
          if (fis_none(attrVector[eid])) {
            attrVector[eid] = attrVector[_idx.entry_id(inode.control_deps[0], e.index)];
          }
        }
      }
    } else {
      bool forward_known = true;
      // Forward operator inference.
      if(trace) {
        std::cout << "IS FORWARD" << std::endl << std::flush;
      }

      // Inputs
      i_attr.resize(num_inputs, empty_val);
      for (uint32_t i = 0; i < i_attr.size(); ++i) {
        const nnvm::IndexedGraph::NodeEntry& node_entry = inode.inputs[i];
        if(trace) {
          std::cout << "input: " << std::endl;
          GraphDumper::print(&std::cout, _idx[node_entry.node_id], 1);
        }
        i_attr[i] = attrVector[_idx.entry_id(node_entry)];
        if (fis_none(i_attr[i])) {
          forward_known = false;
        }
      }

      // Outputs
      o_attr.resize(num_outputs, empty_val);
      for (uint32_t i = 0; i < o_attr.size(); ++i) {
        o_attr[i] = attrVector[_idx.entry_id(nid, i)];
        if(trace) {
          std::cout << "output attribute so far: " << o_attr[i] << std::endl;
        }
        if (fis_none(o_attr[i])) {
          forward_known = false;
        }
      }

      if(trace) {
        std::cout << "i_attr: ";
        for(size_t i = 0, n = i_attr.size(); i < n; ++i) {
          if(i) {
            std::cout << ", ";
          }
          std::cout << i;
        }
        std::cout << std::endl;
        std::cout << "o_attr: ";
        for(size_t i = 0, n = o_attr.size(); i < n; ++i) {
          if(i) {
            std::cout << ", ";
          }
          std::cout << i;
        }
        std::cout << std::endl;
      }

      // Inference
      auto finfer = finfer_callback.get(inode.source->op(), fdefault);
      if (!forward_known) {
        if(trace) {
          std::cout << "So far, forward isn't known..." << std::endl;
        }
        if (finfer != nullptr) {
          // Call inference function of the operator.
          try {
            forward_known = ApplyOpInferAttr(ret, finfer, inode.source->attrs,
                                             nid, &i_attr, &o_attr);
            if(trace) {
              if(!forward_known) {
                std::cout << "Forward is still not known..." << std::endl;
              }
            }
          } catch (const std::exception& e) {
            throw dmlc::Error("Error in operator " + inode.source->attrs.name + ": " + e.what());
          }
        } else {
          CHECK(!last_iter)
              << "Attribute " << infer_name
              << " is not registed by op " << inode.source->op()->name
              << " we are not able to complete the inference because of this";
        }
      }
      if(trace) {
        std::cout << "final i_attr: ";
        for(size_t i = 0, n = i_attr.size(); i < n; ++i) {
          if(i) {
            std::cout << ", ";
          }
          std::cout << i;
        }
        std::cout << std::endl;
        std::cout << "final o_attr: ";
        for(size_t i = 0, n = o_attr.size(); i < n; ++i) {
          if(i) {
            std::cout << ", ";
          }
          std::cout << i;
        }
        std::cout << std::endl;
      }
      // Save to the result map.
      for (uint32_t i = 0; i < num_inputs; ++i) {
        const nnvm::IndexedGraph::NodeEntry& node_entry = inode.inputs[i];
        attrVector[_idx.entry_id(node_entry)] = i_attr[i];
      }
      for (uint32_t i = 0; i < num_outputs; ++i) {
        attrVector[_idx.entry_id(nid, i)] = o_attr[i];
      }
      if(trace) {
        std::cout << "final attrVector: ";
        for (size_t i = 0, n = attrVector.size(); i < n; ++i) {
          if (i) {
            std::cout << ", ";
          }
          std::cout << i;
        }
        std::cout << std::endl;
      }
    }
  };

  size_t last_num_unknown;
  size_t num_unknown = attrVector.size();
  int i = 0;
  do {
    if (i % 2 == 0) {
      for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
        infer_step(nid, false);
      }
    } else {
      // backward inference
      for (uint32_t i = idx.num_nodes(); i != 0; --i) {
        infer_step(i - 1, false);
      }
    }
    last_num_unknown = num_unknown;
    num_unknown = 0;
    for (size_t j = 0; j < idx.num_node_entries(); ++j) {
      if (fis_none(attrVector[j])) {
        ++num_unknown;
      }
    }
    ++i;
  } while (num_unknown > 0 && last_num_unknown > num_unknown);
  // set the attributes
  ret.attrs[attr_name] = std::make_shared<any>(std::move(attrVector));
  // number of nodes who knows the attribute.
  ret.attrs[unknown_name] = std::make_shared<any>(num_unknown);
  return ret;
}

// inference fucntion for same type
inline bool SameType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
  int def_v = -1;
  for (int v : *oattr) {
    if (v != -1) {
      def_v = v; break;
    }
  }
  if (def_v == -1) {
    for (int v : *iattr) {
      if (v != -1) {
        def_v = v; break;
      }
    }
  }
  if (def_v == -1) return false;
  for (int& v : *oattr) {
    v = def_v;
  }
  for (int& v : *iattr) {
    v = def_v;
  }
  return true;
}

// assigning default type N to both input and output attrs with value -1
template <int default_val, int none>
inline bool DefaultType(const nnvm::NodeAttrs& attrs,
                        const Context& ctx,
                        std::vector<int> *iattr,
                        std::vector<int> *oattr) {
  // TODO(junwu): check whether need to use ctx
  for (int& v : *oattr) {
    if (v == none) v = default_val;
  }
  for (int& v : *iattr) {
    if (v == none) v = default_val;
  }
  return true;
}

nnvm::Graph InferShape(nnvm::Graph graph,
                       nnvm::ShapeVector shape_inputs,
                       const std::string& shape_attr_key) {
  using dmlc::any;
  if (shape_inputs.size() != 0) {
    graph.attrs["shape_inputs"] = std::make_shared<any>(std::move(shape_inputs));
  }
  if (shape_attr_key.length() != 0) {
    graph.attrs["shape_attr_key"] = std::make_shared<any>(std::move(shape_attr_key));
  }
  return InferAttr<nnvm::TShape, nnvm::FInferShape>(
      std::move(graph), nnvm::TShape(),
      "FInferShape", "shape_inputs", "shape_attr_key",
      "shape", "shape_num_unknown_nodes",
      [](const nnvm::TShape& s) { return s.ndim() == 0 || s.Size() == 0; },
      nullptr, true);
}

nnvm::Graph InferType(nnvm::Graph graph,
                      nnvm::DTypeVector dtype_inputs,
                      const std::string& dtype_attr_key) {
  using dmlc::any;
  if (dtype_inputs.size() != 0) {
    graph.attrs["dtype_inputs"] = std::make_shared<any>(std::move(dtype_inputs));
  }
  if (dtype_attr_key.length() != 0) {
    graph.attrs["dtype_attr_key"] = std::make_shared<any>(std::move(dtype_attr_key));
  }
  return InferAttr<int, nnvm::FInferType>(
      std::move(graph), -1,
      "FInferType", "dtype_inputs", "dtype_attr_key",
      "dtype", "dtype_num_unknown_nodes",
      [](const int t) { return t == -1; },
      SameType, true);
}

nnvm::Graph InferStorageType(nnvm::Graph graph,
                             StorageTypeVector storage_type_inputs,
                             const std::string& storage_type_attr_key) {

  for(auto stype : storage_type_inputs) {
    std::cout << "Input storage type: " << common::stype_string(stype)
      << std::endl << std::flush;
  }
  std::cout << "storage_type_attr_key: " << storage_type_attr_key << std::endl << std::flush;

  using dmlc::any;
  if (storage_type_inputs.size() != 0) {
    graph.attrs["storage_type_inputs"] = std::make_shared<any>(std::move(storage_type_inputs));
  }
  if (storage_type_attr_key.length() != 0) {
    graph.attrs["storage_type_attr_key"] = std::make_shared<any>(std::move(storage_type_attr_key));
  }
  // for storage type, the backward attr is not necessarily the same as it's correspondence
  const int kDefaultStorage = 0;
  return InferAttr<int, FInferStorageType>(
      std::move(graph), -1,
      "FInferStorageType", "storage_type_inputs", "storage_type_attr_key",
      "storage_type", "storage_type_num_unknown_nodes",
      [](const int t) { return t == -1; },
      DefaultType<kDefaultStorage, -1>, false);
}

}  // namespace exec
}  // namespace mxnet
