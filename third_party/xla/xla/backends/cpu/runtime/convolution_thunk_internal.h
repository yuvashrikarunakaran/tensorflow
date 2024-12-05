/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_THUNK_INTERNAL_H_
#define XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_THUNK_INTERNAL_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

#include "xla/backends/cpu/runtime/concurrency.h"
#include "xla/tsl/framework/convolution/eigen_spatial_convolutions.h"  // IWYU pragma: keep
#include "tsl/platform/logging.h"

#define EIGEN_USE_THREADS
#include "Eigen/Core"
#include "Eigen/ThreadPool"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu::internal {

// Returns in 'out_data' (assumes to be zero-initialized) image patch in storage
// order (width, height, depth), constructed from patches in 'col_data', which
// is required to be in storage order (in_width * in_height, filter_width,
// filter_height, in_depth). Based on TF implementation by Yangqing Jia (jiayq).
// TODO(adambanas): The original implementation implicitly rotates the kernel by
// 180 degrees, but to be backwards compatible, we cannot do that in XLA. This
// results in counterintuitive operations on col_data, which is also 15-20%
// slower. Try alternative approaches (e.g. rotate kernel before matrix
// multiplication in the calling function).
template <typename T>
void Pack2DPatches(const T* col_data, const int depth, const int height,
                   const int width, const int filter_h, const int filter_w,
                   const int pad_top, const int pad_bottom, const int pad_left,
                   const int pad_right, const int stride_h, const int stride_w,
                   T* __restrict out_im_data) {
  int w_patches_number =
      (width + filter_w - pad_left - pad_right - 2) / stride_w + 1;
  int h_patches_number =
      (height + filter_h - pad_top - pad_bottom - 2) / stride_h + 1;

  const int filter_spatial_size = filter_h * filter_w;

  int w_patch_begin = pad_left - filter_w + 1;
  col_data += depth * (filter_spatial_size - 1);
  for (int w = 0; w < w_patches_number; ++w) {
    int h_patch_begin = pad_top - filter_h + 1;
    for (int h = 0; h < h_patches_number; ++h) {
      // This loop body covers 1 output patch, at all depths, accounting for
      // padding. The next line is always a pointer to the first element of the
      // new output patch. Notice in case of less-than-full padding, the pointer
      // can point to an element outside the image, but such elements will be
      // skipped by the inner if (so no write occurs).
      T* out_im_patch_data =
          out_im_data + (w_patch_begin * height + h_patch_begin) * depth;

      for (int iw = w_patch_begin; iw < w_patch_begin + filter_w; ++iw) {
        for (int ih = h_patch_begin; ih < h_patch_begin + filter_h; ++ih) {
          // This loop body covers 1 spatial point with coordinates (iw, ih)
          // in the output buffer, at all depths
          if (iw >= 0 && iw < width && ih >= 0 && ih < height) {
            for (int i = 0; i < depth; ++i) {
              out_im_patch_data[i] += col_data[i];
            }
          }
          out_im_patch_data += depth;
          col_data -= depth;
        }
        // Jump over remaining number of depth.
        out_im_patch_data += depth * (height - filter_h);
      }

      col_data += 2 * depth * filter_spatial_size;
      h_patch_begin += stride_h;
    }
    w_patch_begin += stride_w;
  }
}

// This implementation is based on TF algorithm with parallel contraction.
// TODO(adambanas): There are other variants of this algorithm, 10% performance
// improvement was observed on 1D case when not using parallel contraction.
// Explore these alternatives.
// TODO(adambanas): Add support for feature group count.
template <typename EigenDevice, typename ScalarType>
void EigenTransposedConv2D(
    const EigenDevice& device, ScalarType* out, ScalarType* lhs,
    ScalarType* rhs, Eigen::Index input_batch, Eigen::Index input_x,
    Eigen::Index input_y, Eigen::Index input_channels, Eigen::Index kernel_x,
    Eigen::Index kernel_y, Eigen::Index kernel_channels,
    Eigen::Index kernel_filters, Eigen::Index output_x, Eigen::Index output_y,
    Eigen::Index padding_x_before, Eigen::Index padding_x_after,
    Eigen::Index padding_y_before, Eigen::Index padding_y_after,
    Eigen::Index lhs_x_dilation, Eigen::Index lhs_y_dilation,
    std::function<void()> done_callback, bool use_thunk_runtime) {
  // TODO(adambanas): Current custom conv algorithm doesn't support both
  // multiple input channels and multiple output channels (i.e. kernel_filters)
  // at the same time.
  CHECK(input_channels == 1 || kernel_filters == 1);

  typedef Eigen::TensorMap<Eigen::Tensor<ScalarType, 2, Eigen::RowMajor>,
                           Eigen::Unaligned>
      TensorMap;
  typedef Eigen::TensorMap<Eigen::Tensor<const ScalarType, 2, Eigen::RowMajor>,
                           Eigen::Aligned>
      ConstTensorMap;

  // Total spatial dimensions.
  const int input_image_size = input_x * input_y;
  const int output_image_size = output_x * output_y;
  // Kernel dimensions per input channel.
  const int kernel_total_size = kernel_x * kernel_y * kernel_filters;

  // Intermediate buffer
  std::vector<ScalarType> col_buffer;
  col_buffer.resize(input_batch * input_image_size * kernel_total_size);
  ScalarType* col_buffer_data = col_buffer.data();

  // Initialize output to zero.
  ScalarType* out_data = out;
  std::fill(out_data,
            out_data + input_batch * output_image_size * kernel_filters,
            ScalarType(0.0f));

  // Initialize contraction dims (we need to transpose 'B' below).
  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_dims;
  contract_dims[0].first = 1;
  contract_dims[0].second = 1;

  // Compute intermediate results (convolution matrix) into col_buffer.
  TensorMap C(col_buffer_data, input_batch * input_image_size,
              kernel_total_size);

  ConstTensorMap A(lhs, input_batch * input_image_size, input_channels);
  ConstTensorMap B(rhs, kernel_total_size, input_channels);

  // Use concurrent execution if we have a thread pool device.
  constexpr bool use_thread_pool =
      std::is_same_v<EigenDevice, Eigen::ThreadPoolDevice>;

  // For thunk runtime, `done_callback` must be provided only if we use a thread
  // pool device. This check is not true for classic runtime which does not
  // support async execution.
  if (use_thunk_runtime) {
    CHECK_EQ(use_thread_pool, static_cast<bool>(done_callback));  // Crash OK
  }

  const int input_offset = input_image_size * kernel_total_size;
  const int output_offset = output_image_size * kernel_filters;

  // Pack the calculated patches into the output buffer.
  // NOTE: The ownership of the col_buffer is transferred to the lambda without
  // data copy or reallocation. Thanks to that, col_buffer_data pointer remains
  // valid, and that is important because 'C' matrix is referencing it. We need
  // to make sure this lambda is never copied, otherwise col_buffer won't
  // contain contraction results at the time lambda is called.
  auto pack_patches = [=, col_buffer = std::move(col_buffer)]() {
    // Using local pointers to buffers, because lambda is not mutable.
    const ScalarType* col_buffer_data = col_buffer.data();
    ScalarType* local_out_data = out_data;

    // TODO(adambanas): Run this part in parallel.
    for (int image_id = 0; image_id < input_batch; ++image_id) {
      Pack2DPatches<ScalarType>(
          col_buffer_data, kernel_filters, output_y, output_x, kernel_y,
          kernel_x, padding_y_before, padding_y_after, padding_x_before,
          padding_x_after, lhs_y_dilation, lhs_x_dilation, local_out_data);

      col_buffer_data += input_offset;
      local_out_data += output_offset;
    }

    // If done callback is provided, we need to call it after all the work is
    // done.
    if (done_callback) {
      done_callback();
    }
  };

  if (done_callback) {
    // Schedule the work in the thread pool and return.
    C.device(device, std::move(pack_patches)) = A.contract(B, contract_dims);
  } else {
    // Run synchronously in the current thread.
    C.device(device) = A.contract(B, contract_dims);
    pack_patches();
  }
}

inline bool CanUseCustomTransposedConv(
    Eigen::Index input_channels, Eigen::Index kernel_filters,
    Eigen::Index x_stride, Eigen::Index y_stride, Eigen::Index lhs_x_dilation,
    Eigen::Index lhs_y_dilation, Eigen::Index rhs_x_dilation,
    Eigen::Index rhs_y_dilation, Eigen::Index feature_group_count) {
  return (lhs_x_dilation > 1 || lhs_y_dilation > 1) && rhs_x_dilation == 1 &&
         rhs_y_dilation == 1 && (input_channels == 1 || kernel_filters == 1) &&
         feature_group_count == 1 && x_stride == 1 && y_stride == 1;
}

// Algorithm that works for all types of 2D convolutions. Even though it works
// for transposed convolutions, the custom algorithm should be used whenever
// applicable, because it is faster.
template <typename EigenDevice, typename ScalarType>
void EigenGenericConv2D(
    const EigenDevice& device, ScalarType* out, ScalarType* lhs,
    ScalarType* rhs, Eigen::Index input_batch, Eigen::Index input_x,
    Eigen::Index input_y, Eigen::Index input_channels, Eigen::Index kernel_x,
    Eigen::Index kernel_y, Eigen::Index kernel_channels,
    Eigen::Index kernel_filters, Eigen::Index output_x, Eigen::Index output_y,
    Eigen::Index x_stride, Eigen::Index y_stride, Eigen::Index padding_x_before,
    Eigen::Index padding_x_after, Eigen::Index padding_y_before,
    Eigen::Index padding_y_after, Eigen::Index lhs_x_dilation,
    Eigen::Index lhs_y_dilation, Eigen::Index rhs_x_dilation,
    Eigen::Index rhs_y_dilation, Eigen::Index feature_group_count,
    std::function<void()> done_callback, bool use_thunk_runtime) {
  const Eigen::TensorMap<Eigen::Tensor<const ScalarType, 4, Eigen::RowMajor>,
                         Eigen::Aligned>
      input(lhs, input_batch, input_x, input_y, input_channels);

  const Eigen::TensorMap<Eigen::Tensor<const ScalarType, 4, Eigen::RowMajor>,
                         Eigen::Aligned>
      kernel(rhs, kernel_x, kernel_y, kernel_channels, kernel_filters);

  Eigen::TensorMap<Eigen::Tensor<ScalarType, 4, Eigen::RowMajor>,
                   Eigen::Aligned>
      output(out, input_batch, output_x, output_y, kernel_filters);

  Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<Eigen::Index>(1, 0);

  Eigen::DSizes<Eigen::Index, 5> input_reshaped_dims;
  input_reshaped_dims[0] = input_batch;
  input_reshaped_dims[1] = input_x;
  input_reshaped_dims[2] = input_y;
  input_reshaped_dims[3] = feature_group_count;
  input_reshaped_dims[4] = input_channels / feature_group_count;

  Eigen::DSizes<Eigen::Index, 5> output_reshaped_dims;
  output_reshaped_dims[0] = input_batch;
  output_reshaped_dims[1] = output_x;
  output_reshaped_dims[2] = output_y;
  output_reshaped_dims[3] = feature_group_count;
  output_reshaped_dims[4] = kernel_filters / feature_group_count;

  // Molds the output of the patch extraction code into a 2d tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  //   kernels
  // - the second dimension (dims[1]): everything else
  Eigen::DSizes<Eigen::Index, 2> pre_contract_dims;
  pre_contract_dims[0] = output_y * output_x * input_batch;
  pre_contract_dims[1] = kernel_channels * kernel_y * kernel_x;

  // Molds the output of the contraction into the shape expected by the user:
  Eigen::DSizes<Eigen::Index, 4> post_contract_dims;
  post_contract_dims[0] = input_batch;
  post_contract_dims[1] = output_x;
  post_contract_dims[2] = output_y;
  post_contract_dims[3] = kernel_filters / feature_group_count;

  Eigen::DSizes<Eigen::Index, 3> kernel_dims;
  kernel_dims[0] = kernel_channels * kernel_y * kernel_x;
  kernel_dims[1] = feature_group_count;
  kernel_dims[2] = kernel_filters / feature_group_count;

  // Constructs convolution and output expressions for a given group index.
  auto convolve_group = [=](int64_t i) {
    // The row and column dimensions must be flipped when passed to Eigen.
    auto convolved =
        input.reshape(input_reshaped_dims)
            .chip(i, 3)
            .extract_image_patches(
                kernel_y, kernel_x, y_stride, x_stride, rhs_y_dilation,
                rhs_x_dilation, lhs_y_dilation, lhs_x_dilation,
                padding_y_before, padding_y_after, padding_x_before,
                padding_x_after, static_cast<ScalarType>(0.0f))
            .reshape(pre_contract_dims)
            .contract(kernel.reshape(kernel_dims).chip(i, 1), contract_dims)
            .reshape(post_contract_dims);
    auto output_reshaped = output.reshape(output_reshaped_dims).chip(i, 3);
    return std::make_pair(output_reshaped, convolved);
  };

  // Use concurrent execution if we have a thread pool device.
  constexpr bool use_thread_pool =
      std::is_same_v<EigenDevice, Eigen::ThreadPoolDevice>;

  // For thunk runtime, `done_callback` must be provided only if we use a thread
  // pool device. This check is not true for classic runtime which does not
  // support async execution.
  if (use_thunk_runtime) {
    CHECK_EQ(use_thread_pool, static_cast<bool>(done_callback));  // Crash OK
  }

  if constexpr (use_thread_pool) {
    // Although we schedule at most one tasks for each thread, individual
    // convolution might also schedule more tasks into the same thread pool.
    auto max_tasks = static_cast<Eigen::Index>(device.numThreads());
    auto task_size = Eigen::numext::div_ceil(feature_group_count, max_tasks);
    auto num_tasks = Eigen::numext::div_ceil(feature_group_count, task_size);

    if (use_thunk_runtime) {
      ScheduleAll(&device, num_tasks, [=, &device](Eigen::Index task_index) {
        Eigen::Index start = task_index * task_size;
        Eigen::Index end = std::min(start + task_size, feature_group_count);
        for (Eigen::Index i = start; i < end; ++i) {
          auto [output, convolved] = convolve_group(i);
          output.device(device, done_callback) = convolved;
        }
      });
    } else {
      Eigen::Barrier barrier(num_tasks);
      ScheduleAll(
          &device, num_tasks, [=, &device, &barrier](Eigen::Index task_index) {
            Eigen::Index start = task_index * task_size;
            Eigen::Index end = std::min(start + task_size, feature_group_count);
            for (Eigen::Index i = start; i < end; ++i) {
              auto [output, convolved] = convolve_group(i);
              output.device(device) = convolved;
            }
            barrier.Notify();
          });
      barrier.Wait();
    }

  } else {
    // Convolve all feature groups sequentially in the caller thread.
    for (Eigen::Index i = 0; i < feature_group_count; ++i) {
      auto [output, convolved] = convolve_group(i);
      output.device(device) = convolved;
    }
  }
}

// TODO(ezhulenev): Make internal implementation a private static method of
// ConvolutionThunk (for consistency with DotThunk). Today we keep it as a
// free function to use it in the legacy XLA CPU runtime.
template <typename EigenDevice, typename ScalarType>
void EigenConv2D(const EigenDevice& device, ScalarType* out, ScalarType* lhs,
                 ScalarType* rhs, Eigen::Index input_batch,
                 Eigen::Index input_x, Eigen::Index input_y,
                 Eigen::Index input_channels, Eigen::Index kernel_x,
                 Eigen::Index kernel_y, Eigen::Index kernel_channels,
                 Eigen::Index kernel_filters, Eigen::Index output_x,
                 Eigen::Index output_y, Eigen::Index x_stride,
                 Eigen::Index y_stride, Eigen::Index padding_x_before,
                 Eigen::Index padding_x_after, Eigen::Index padding_y_before,
                 Eigen::Index padding_y_after, Eigen::Index lhs_x_dilation,
                 Eigen::Index lhs_y_dilation, Eigen::Index rhs_x_dilation,
                 Eigen::Index rhs_y_dilation, Eigen::Index feature_group_count,
                 std::function<void()> done_callback, bool use_thunk_runtime) {
  if (CanUseCustomTransposedConv(input_channels, kernel_filters, x_stride,
                                 y_stride, lhs_x_dilation, lhs_y_dilation,
                                 rhs_x_dilation, rhs_y_dilation,
                                 feature_group_count)) {
    EigenTransposedConv2D(
        device, out, lhs, rhs, input_batch, input_x, input_y, input_channels,
        kernel_x, kernel_y, kernel_channels, kernel_filters, output_x, output_y,
        padding_x_before, padding_x_after, padding_y_before, padding_y_after,
        lhs_x_dilation, lhs_y_dilation, done_callback, use_thunk_runtime);
  } else {
    EigenGenericConv2D(
        device, out, lhs, rhs, input_batch, input_x, input_y, input_channels,
        kernel_x, kernel_y, kernel_channels, kernel_filters, output_x, output_y,
        x_stride, y_stride, padding_x_before, padding_x_after, padding_y_before,
        padding_y_after, lhs_x_dilation, lhs_y_dilation, rhs_x_dilation,
        rhs_y_dilation, feature_group_count, done_callback, use_thunk_runtime);
  }
}

template <typename EigenDevice, typename ScalarType>
void EigenConv3D(const EigenDevice& device, ScalarType* out, ScalarType* lhs,
                 ScalarType* rhs, Eigen::Index input_batch,
                 Eigen::Index input_x, Eigen::Index input_y,
                 Eigen::Index input_z, Eigen::Index input_channels,
                 Eigen::Index kernel_x, Eigen::Index kernel_y,
                 Eigen::Index kernel_z, Eigen::Index kernel_channels,
                 Eigen::Index kernel_filters, Eigen::Index output_x,
                 Eigen::Index output_y, Eigen::Index output_z,
                 Eigen::Index x_stride, Eigen::Index y_stride,
                 Eigen::Index z_stride, Eigen::Index padding_x_before,
                 Eigen::Index padding_x_after, Eigen::Index padding_y_before,
                 Eigen::Index padding_y_after, Eigen::Index padding_z_before,
                 Eigen::Index padding_z_after, Eigen::Index lhs_x_dilation,
                 Eigen::Index lhs_y_dilation, Eigen::Index lhs_z_dilation,
                 Eigen::Index rhs_x_dilation, Eigen::Index rhs_y_dilation,
                 Eigen::Index rhs_z_dilation, Eigen::Index feature_group_count,
                 std::function<void()> done_callback) {
  using ConstTType =
      Eigen::TensorMap<Eigen::Tensor<const ScalarType, 5, Eigen::RowMajor>,
                       Eigen::Aligned>;
  const ConstTType input(lhs, input_batch, input_x, input_y, input_z,
                         input_channels);

  const ConstTType kernel(rhs, kernel_x, kernel_y, kernel_z, kernel_channels,
                          kernel_filters);

  Eigen::TensorMap<Eigen::Tensor<ScalarType, 5, Eigen::RowMajor>,
                   Eigen::Aligned>
      output(out, input_batch, output_x, output_y, output_z, kernel_filters);

  Eigen::DSizes<Eigen::Index, 6> input_reshaped_dims;
  input_reshaped_dims[0] = input_batch;
  input_reshaped_dims[1] = input_x;
  input_reshaped_dims[2] = input_y;
  input_reshaped_dims[3] = input_z;
  input_reshaped_dims[4] = feature_group_count;
  input_reshaped_dims[5] = input_channels / feature_group_count;

  Eigen::DSizes<Eigen::Index, 6> output_reshaped_dims;
  output_reshaped_dims[0] = input_batch;
  output_reshaped_dims[1] = output_x;
  output_reshaped_dims[2] = output_y;
  output_reshaped_dims[3] = output_z;
  output_reshaped_dims[4] = feature_group_count;
  output_reshaped_dims[5] = kernel_filters / feature_group_count;

  Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<Eigen::Index>(1, 0);

  // Molds the output of the patch extraction code into a 2d tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  //   kernels
  // - the second dimension (dims[1]): everything else
  Eigen::DSizes<Eigen::Index, 2> pre_contract_dims;
  pre_contract_dims[0] = output_x * output_y * output_z * input_batch;
  pre_contract_dims[1] = kernel_channels * kernel_x * kernel_y * kernel_z;

  // Molds the output of the contraction into the shape expected by the user:
  Eigen::DSizes<Eigen::Index, 5> post_contract_dims;
  post_contract_dims[0] = input_batch;
  post_contract_dims[1] = output_x;
  post_contract_dims[2] = output_y;
  post_contract_dims[3] = output_z;
  post_contract_dims[4] = kernel_filters / feature_group_count;

  Eigen::DSizes<Eigen::Index, 3> kernel_dims;
  kernel_dims[0] = kernel_channels * kernel_x * kernel_y * kernel_z;
  kernel_dims[1] = feature_group_count;
  kernel_dims[2] = kernel_filters / feature_group_count;

  for (Eigen::Index i = 0; i < feature_group_count; ++i) {
    // The dimension order must be flipped when passed to Eigen.
    auto input_chip = input.reshape(input_reshaped_dims).chip(i, 4);
    auto patches =
        Eigen::TensorVolumePatchOp<Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::Dynamic, decltype(input_chip)>(
            input_chip, kernel_z, kernel_y, kernel_x, z_stride, y_stride,
            x_stride, rhs_z_dilation, rhs_y_dilation, rhs_x_dilation,
            lhs_z_dilation, lhs_y_dilation, lhs_x_dilation, padding_z_before,
            padding_z_after, padding_y_before, padding_y_after,
            padding_x_before, padding_x_after, static_cast<ScalarType>(0.0f));

    auto convolved =
        patches.reshape(pre_contract_dims)
            .contract(kernel.reshape(kernel_dims).chip(i, 1), contract_dims)
            .reshape(post_contract_dims);

    auto output_reshaped = output.reshape(output_reshaped_dims).chip(i, 4);
    if (done_callback) {
      output_reshaped.device(device, done_callback) = convolved;
    } else {
      output_reshaped.device(device) = convolved;
    }
  }
}

// Extern Conv2D template for all supported devices and data types.
#define CONV2D_EXTERN_TEMPLATE(DEVICE, SCALAR_TYPE)                        \
  extern template void EigenConv2D<DEVICE, SCALAR_TYPE>(                   \
      const DEVICE& device, SCALAR_TYPE* out, SCALAR_TYPE* lhs,            \
      SCALAR_TYPE* rhs, Eigen::Index input_batch, Eigen::Index input_x,    \
      Eigen::Index input_y, Eigen::Index input_channels,                   \
      Eigen::Index kernel_x, Eigen::Index kernel_y,                        \
      Eigen::Index kernel_channels, Eigen::Index kernel_filters,           \
      Eigen::Index output_x, Eigen::Index output_y, Eigen::Index x_stride, \
      Eigen::Index y_stride, Eigen::Index padding_x_before,                \
      Eigen::Index padding_x_after, Eigen::Index padding_y_before,         \
      Eigen::Index padding_y_after, Eigen::Index lhs_x_dilation,           \
      Eigen::Index lhs_y_dilation, Eigen::Index rhs_x_dilation,            \
      Eigen::Index rhs_y_dilation, Eigen::Index feature_group_count,       \
      std::function<void()> done_callback, bool use_thunk_runtime)

CONV2D_EXTERN_TEMPLATE(Eigen::DefaultDevice, Eigen::half);
CONV2D_EXTERN_TEMPLATE(Eigen::DefaultDevice, float);
CONV2D_EXTERN_TEMPLATE(Eigen::ThreadPoolDevice, Eigen::half);
CONV2D_EXTERN_TEMPLATE(Eigen::ThreadPoolDevice, float);

#undef CONV2D_EXTERN_TEMPLATE

// Extern Conv3D template for all supported devices and data types.
#define CONV3D_EXTERN_TEMPLATE(DEVICE, SCALAR_TYPE)                            \
  extern template void EigenConv3D<DEVICE, SCALAR_TYPE>(                       \
      const DEVICE& device, SCALAR_TYPE* out, SCALAR_TYPE* lhs,                \
      SCALAR_TYPE* rhs, Eigen::Index input_batch, Eigen::Index input_x,        \
      Eigen::Index input_y, Eigen::Index input_z, Eigen::Index input_channels, \
      Eigen::Index kernel_x, Eigen::Index kernel_y, Eigen::Index kernel_z,     \
      Eigen::Index kernel_channels, Eigen::Index kernel_filters,               \
      Eigen::Index output_x, Eigen::Index output_y, Eigen::Index output_z,     \
      Eigen::Index x_stride, Eigen::Index y_stride, Eigen::Index z_stride,     \
      Eigen::Index padding_x_before, Eigen::Index padding_x_after,             \
      Eigen::Index padding_y_before, Eigen::Index padding_y_after,             \
      Eigen::Index padding_z_before, Eigen::Index padding_z_after,             \
      Eigen::Index lhs_x_dilation, Eigen::Index lhs_y_dilation,                \
      Eigen::Index lhs_z_dilation, Eigen::Index rhs_x_dilation,                \
      Eigen::Index rhs_y_dilation, Eigen::Index rhs_z_dilation,                \
      Eigen::Index feature_group_count, std::function<void()> done_callback)

CONV3D_EXTERN_TEMPLATE(Eigen::DefaultDevice, Eigen::half);
CONV3D_EXTERN_TEMPLATE(Eigen::DefaultDevice, float);
CONV3D_EXTERN_TEMPLATE(Eigen::ThreadPoolDevice, Eigen::half);
CONV3D_EXTERN_TEMPLATE(Eigen::ThreadPoolDevice, float);

#undef CONV3D_EXTERN_TEMPLATE

}  // namespace xla::cpu::internal

#define CONV2D_INSTANTIATE_TEMPLATE(DEVICE, SCALAR_TYPE)                   \
  template void xla::cpu::internal::EigenConv2D<DEVICE, SCALAR_TYPE>(      \
      const DEVICE& device, SCALAR_TYPE* out, SCALAR_TYPE* lhs,            \
      SCALAR_TYPE* rhs, Eigen::Index input_batch, Eigen::Index input_x,    \
      Eigen::Index input_y, Eigen::Index input_channels,                   \
      Eigen::Index kernel_x, Eigen::Index kernel_y,                        \
      Eigen::Index kernel_channels, Eigen::Index kernel_filters,           \
      Eigen::Index output_x, Eigen::Index output_y, Eigen::Index x_stride, \
      Eigen::Index y_stride, Eigen::Index padding_x_before,                \
      Eigen::Index padding_x_after, Eigen::Index padding_y_before,         \
      Eigen::Index padding_y_after, Eigen::Index lhs_x_dilation,           \
      Eigen::Index lhs_y_dilation, Eigen::Index rhs_x_dilation,            \
      Eigen::Index rhs_y_dilation, Eigen::Index feature_group_count,       \
      std::function<void()> done_callback, bool use_thunk_runtime)

#define CONV3D_INSTANTIATE_TEMPLATE(DEVICE, SCALAR_TYPE)                       \
  template void xla::cpu::internal::EigenConv3D<DEVICE, SCALAR_TYPE>(          \
      const DEVICE& device, SCALAR_TYPE* out, SCALAR_TYPE* lhs,                \
      SCALAR_TYPE* rhs, Eigen::Index input_batch, Eigen::Index input_x,        \
      Eigen::Index input_y, Eigen::Index input_z, Eigen::Index input_channels, \
      Eigen::Index kernel_x, Eigen::Index kernel_y, Eigen::Index kernel_z,     \
      Eigen::Index kernel_channels, Eigen::Index kernel_filters,               \
      Eigen::Index output_x, Eigen::Index output_y, Eigen::Index output_z,     \
      Eigen::Index x_stride, Eigen::Index y_stride, Eigen::Index z_stride,     \
      Eigen::Index padding_x_before, Eigen::Index padding_x_after,             \
      Eigen::Index padding_y_before, Eigen::Index padding_y_after,             \
      Eigen::Index padding_z_before, Eigen::Index padding_z_after,             \
      Eigen::Index lhs_x_dilation, Eigen::Index lhs_y_dilation,                \
      Eigen::Index lhs_z_dilation, Eigen::Index rhs_x_dilation,                \
      Eigen::Index rhs_y_dilation, Eigen::Index rhs_z_dilation,                \
      Eigen::Index feature_group_count, std::function<void()> done_callback)

#endif  // XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_THUNK_INTERNAL_H_
