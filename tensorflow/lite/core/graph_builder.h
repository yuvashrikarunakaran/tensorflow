/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_CORE_GRAPH_BUILDER_H_
#define TENSORFLOW_LITE_CORE_GRAPH_BUILDER_H_

// This is an EXPERIMENTAL API to programatically build TFLite graphs. It may
// change or be removed at any time. Use it at your own risk.

#include <array>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {

namespace graph_builder {

using OwningErasedPtr = std::unique_ptr<void, void (*)(void*)>;

class InterpreterInfo;
struct Helper;

// Represent a tensor in the TFLite graph.
//
// Copiable but you shouldn't create such an object by yourself. Use the
// `NewInput` family of functions with a Graph for that.
//
// Each tensor is attached to a particular graph. Don't mix tensors created by
// different graphs in operations.
//
// ```cpp
// Tensor a = graph.NewInput(kTfLiteInt32);
// Tensor b = NewInput(graph, kTfLiteFloat32);
// auto [c, d] = NewInputs<2>(graph, kTfLiteFloat32);
// ```
class Tensor {
 public:
  Tensor(const Tensor&) = default;
  Tensor& operator=(const Tensor&) = default;

 private:
  Tensor(InterpreterInfo* builder, int tensor_idx, int graph_idx)
      : builder_(builder), tensor_idx_(tensor_idx), graph_idx_(graph_idx) {}

  friend class Helper;

  InterpreterInfo* builder_;
  int tensor_idx_;
  int graph_idx_;
};

// Respresents a subgraph in the TFLite interpreter.
//
// Copiable but you shouldn't create such an object by yourself. Use the
// `NewGraph` function with a GraphBuilder for that.
//
// ```cpp
// Graph a = builder.NewGraph();
// ```
class Graph {
 public:
  // Returns a new tensors that will be an input to the graph.
  Tensor NewInput(TfLiteType type);

 private:
  Graph(InterpreterInfo* builder, int graph_idx)
      : builder_(builder), graph_idx_(graph_idx) {}

  friend class Helper;

  InterpreterInfo* builder_;
  int graph_idx_;
};

// Allows building a TFLite interpreter programatically.
//
// ```cpp
// GraphBuilder builder;
// Graph grap = builder.NewGraph();
//
// auto [in1, in2] = NewInputs<2>(kTfLiteInt32);
// Tensor sum = Add(in1, in2);
// Tensor abs1 = Abs(in1)
// Tensor out = Mul(sum, abs1);
// MarkOuput(out);
//
// builder.Build(interpreter);
// ```
class GraphBuilder {
 public:
  GraphBuilder();
  Graph NewGraph();

  void Build(Interpreter& interpreter);

 private:
  friend class Helper;

  OwningErasedPtr impl_;
};

// Marks the given tensor as an output of the graph it is attached to.
void MarkOutput(Tensor tensor);

// Marks the given tensors as outputs of the graph they are attached to.
inline void MarkOutputs(const std::vector<Tensor>& tensors) {
  for (const Tensor& t : tensors) {
    MarkOutput(t);
  }
}

// Marks the given tensors as outputs of the graph they are attached to.
inline void MarkOutputs(std::initializer_list<Tensor> tensors) {
  for (const Tensor& t : tensors) {
    MarkOutput(t);
  }
}

// Marks the given tensors as outputs of the graph they are attached to.
template <class... Ts>
void MarkOutputs(Ts... tensors) {
  (MarkOutput(tensors), ...);
}

// Returns a new input for the given graph.
inline Tensor NewInput(Graph graph, TfLiteType type) {
  return graph.NewInput(type);
}

template <size_t... Is>
std::array<Tensor, sizeof...(Is)> NewInputs(std::index_sequence<Is...>,
                                            Graph graph, TfLiteType type) {
  return std::array<Tensor, sizeof...(Is)>{((void)Is, graph.NewInput(type))...};
}

// Returns an array of N inputs for the given graph.
template <size_t N>
std::array<Tensor, N> NewInputs(Graph graph, TfLiteType type) {
  return NewInputs(std::make_index_sequence<N>{}, graph, type);
}

// Creates an ABS operation with `tensor` as the input and returns the tensor
// representing the result.
Tensor Abs(Tensor tensor);

// Creates an ADD operation with `lhs` and `rhs` as the inputs and returns the
// tensor representing the result.
Tensor Add(Tensor lhs, Tensor rhs);

// Creates an MUL operation with `lhs` and `rhs` as the inputs and returns the
// tensor representing the result.
Tensor Mul(Tensor lhs, Tensor rhs);

// Creates a TRANSPOSE operation with `tensor` and `permutation` as the inputs
// and returns the tensor representing the result.
Tensor Transpose(Tensor tensor, Tensor permutation);

// Creates a STABLEHLO_COMPOSITE operation named `name` and falling back to
// `subgraph`.
//
// `inputs` is associated to the subgraph inputs.
//
// Returns a list of tensors representing the subgraph outputs.
std::vector<Tensor> StableHLOComposite(const char* name, const Graph& subgraph,
                                       const std::vector<Tensor>& inputs);

}  // namespace graph_builder
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_GRAPH_BUILDER_H_
