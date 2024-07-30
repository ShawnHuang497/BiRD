// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

/*This code is copied fron NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#pragma once // NOLINT

#include "paddle/extension.h"

#include <cuda.h>         // NOLINT
#include <cuda_runtime.h> // NOLINT

#define DEFAULT_THROW(NAME, TYPE)                                              \
  default:                                                                     \
    do {                                                                       \
      PD_THROW(#NAME, " not implemented for '", TYPE, "'");                    \
    } while (0);                                                               \
    break

#define DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT, NAME, ...) \
  switch (TYPEIN) {                                                            \
  case paddle::DataType::FLOAT32: {                                            \
    using scalar_t_in = float;                                                 \
    switch (TYPEOUT) {                                                         \
    case paddle::DataType::FLOAT32: {                                          \
      using scalar_t_out = float;                                              \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case paddle::DataType::FLOAT16: {                                          \
      using scalar_t_out = phi::dtype::float16;                                \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case paddle::DataType::BFLOAT16: {                                         \
      using scalar_t_out = phi::dtype::bfloat16;                               \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
      DEFAULT_THROW(NAME, TYPEOUT);                                            \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  case paddle::DataType::FLOAT16: {                                            \
    using scalar_t_in = phi::dtype::float16;                                   \
    using scalar_t_out = phi::dtype::float16;                                  \
    __VA_ARGS__;                                                               \
    break;                                                                     \
  }                                                                            \
  case paddle::DataType::BFLOAT16: {                                           \
    using scalar_t_in = phi::dtype::bfloat16;                                  \
    using scalar_t_out = phi::dtype::bfloat16;                                 \
    __VA_ARGS__;                                                               \
    break;                                                                     \
  }                                                                            \
    DEFAULT_THROW(NAME, TYPEIN);                                               \
  }

#define WARP_SIZE 32

template <typename T>
__device__ __forceinline__ T WarpShflXor(T value, int laneMask,
                                         int width = WARP_SIZE,
                                         unsigned int mask = 0xffffffff) {
  return __shfl_xor_sync(mask, value, laneMask, width);
}

template <typename T>
__device__ __forceinline__ T WarpShfl(T value, int srcLane,
                                      int width = WARP_SIZE,
                                      unsigned int mask = 0xffffffff) {
  return __shfl_sync(mask, value, srcLane, width);
}

template <typename U>
__device__ void cuWelfordOnlineSum(const U curr,
                                   U &mu,      // NOLINT
                                   U &sigma2,  // NOLINT
                                   U &count) { // NOLINT
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  U delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

template <typename U>
__device__ void cuChanOnlineSum(const U muB, const U sigma2B, const U countB,
                                U &mu,      // NOLINT
                                U &sigma2,  // NOLINT
                                U &count) { // NOLINT
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA * mu + nB * muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

template <typename T, typename U>
__device__ void cuWelfordMuSigma2(const T *__restrict__ vals, const int n1,
                                  const int n2, const int i1,
                                  U &mu,     // NOLINT
                                  U &sigma2, // NOLINT
                                  U *buf) {
  // Assumptions:
  // 1) blockDim.x == WARP_SIZE
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  mu = U(0);
  sigma2 = U(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T *lvals = vals + i1 * n2;
    int l = 4 * thrx;
    for (; l + 3 < n2; l += 4 * numx) {
      for (int k = 0; k < 4; ++k) {
        U curr = static_cast<U>(lvals[l + k]);
        cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
      }
    }
    for (; l < n2; ++l) {
      U curr = static_cast<U>(lvals[l]);
      cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    for (int l = 0; l <= 4; ++l) {
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      U muB = WarpShfl(mu, srcLaneB);
      U countB = WarpShfl(count, srcLaneB);
      U sigma2B = WarpShfl(sigma2, srcLaneB);
      cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U *ubuf = (U *)buf;                 // NOLINT
      U *ibuf = (U *)(ubuf + blockDim.y); // NOLINT
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset &&
            threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y] = mu;
          ubuf[2 * wrt_y + 1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U muB = ubuf[2 * threadIdx.y];
          U sigma2B = ubuf[2 * threadIdx.y + 1];
          U countB = ibuf[threadIdx.y];
          cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1] / U(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WarpShfl(mu, 0);
      sigma2 = WarpShfl(sigma2 / U(n2), 0);
    }
  }
}

template <>
__device__ void cuWelfordMuSigma2(const phi::dtype::float16 *__restrict__ vals,
                                  const int n1, const int n2, const int i1,
                                  float &mu,     // NOLINT
                                  float &sigma2, // NOLINT
                                  float *buf) {
  // Assumptions:
  // 1) blockDim.x == WARP_SIZE
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = 0.0f;
  mu = float(0);     // NOLINT
  sigma2 = float(0); // NOLINT
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const auto *lvals = vals + i1 * n2;
    int l = 8 * thrx;
    if ((((size_t)lvals) & 3) != 0) { // NOLINT
      // 16 bit alignment
      // first thread consumes first point
      if (thrx == 0) {
        float curr = static_cast<float>(lvals[0]);
        cuWelfordOnlineSum(curr, mu, sigma2, count);
      }
      ++l;
    }
    // at this point, lvals[l] are 32 bit aligned for all threads.
    for (; l + 7 < n2; l += 8 * numx) {
      for (int k = 0; k < 8; k += 2) {
        float2 curr = __half22float2(*((__half2 *)(lvals + l + k))); // NOLINT
        cuWelfordOnlineSum(curr.x, mu, sigma2, count);
        cuWelfordOnlineSum(curr.y, mu, sigma2, count);
      }
    }
    for (; l < n2; ++l) {
      float curr = static_cast<float>(lvals[l]);
      cuWelfordOnlineSum(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    for (int l = 0; l <= 4; ++l) {
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      float muB = WarpShfl(mu, srcLaneB);
      float countB = WarpShfl(count, srcLaneB);
      float sigma2B = WarpShfl(sigma2, srcLaneB);
      cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float *ubuf = (float *)buf;                 // NOLINT
      float *ibuf = (float *)(ubuf + blockDim.y); // NOLINT
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset &&
            threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y] = mu;
          ubuf[2 * wrt_y + 1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float muB = ubuf[2 * threadIdx.y];
          float sigma2B = ubuf[2 * threadIdx.y + 1];
          float countB = ibuf[threadIdx.y];
          cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1] / float(n2); // NOLINT
      // don't care about final value of count, we know count == n2
    } else {
      mu = WarpShfl(mu, 0);
      sigma2 = WarpShfl(sigma2 / float(n2), 0); // NOLINT
    }
  }
}

template <typename U> U rsqrt(U v) { return U(1) / sqrt(v); }
template <> float rsqrt(float v) { return rsqrtf(v); }
template <> double rsqrt(double v) { return rsqrt(v); }

namespace { // NOLINT
// This is the un-specialized struct.  Note that we prevent instantiation of
// this struct by putting an undefined symbol in the function body so it won't
// compile.
//  template <typename T>
//  struct SharedMemory
//  {
//      // Ensure that we won't compile any un-specialized types
//      __device__ T *getPointer()
//      {
//          extern __device__ void error(void);
//          error();
//          return NULL;
//      }
//  };
// https://github.com/NVIDIA/apex/issues/246
template <typename T> struct SharedMemory;

template <> struct SharedMemory<float> {
  __device__ float *getPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

} // namespace

template <typename T, typename U, typename V>
__global__ void
cuApplyLayerNorm(V *__restrict__ output_vals, U *__restrict__ mean,
                 U *__restrict__ invvar, const T *__restrict__ vals,
                 const int n1, const int n2, const U epsilon,
                 const V *__restrict__ gamma, const V *__restrict__ beta) {
  // Assumptions:
  // 1) blockDim.x == WARP_SIZE
  // 2) Tensors are contiguous
  //
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U *buf = shared.getPointer();
    U mu, sigma2;
    cuWelfordMuSigma2(vals, n1, n2, i1, mu, sigma2, buf);
    const T *lvals = vals + i1 * n2;
    V *ovals = output_vals + i1 * n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && beta != NULL) {
      for (int i = thrx; i < n2; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = gamma[i] * static_cast<V>(c_invvar * (curr - mu)) + beta[i];
      }
    } else {
      for (int i = thrx; i < n2; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = static_cast<V>(c_invvar * (curr - mu));
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      mean[i1] = mu;
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}

template <typename T, typename U, typename V>
__device__ void cuLoadWriteStridedInputs(
    const int i1_block, const int thr_load_row_off, const int thr_load_col_off,
    const int i2_off, const int row_stride, U *warp_buf1, U *warp_buf2,
    const T *input, const V *dout, const int i1_end, const int n2,
    const U *__restrict__ mean, const U *__restrict__ invvar) {
  int i1 = i1_block + thr_load_row_off;
  if (i1 < i1_end) {
    U curr_mean = mean[i1];
    U curr_invvar = invvar[i1];
    for (int k = 0; k < blockDim.y; ++k) {
      int i2 = i2_off + k;
      int load_idx = i1 * n2 + i2;
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      if (i2 < n2) {
        U curr_input = static_cast<U>(input[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);
        warp_buf1[write_idx] = curr_dout;
        warp_buf2[write_idx] =
            curr_dout * (curr_input - curr_mean) * curr_invvar;
      } else {
        warp_buf1[write_idx] = U(0);
        warp_buf2[write_idx] = U(0);
      }
    }
  } else {
    for (int k = 0; k < blockDim.y; ++k) {
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      warp_buf1[write_idx] = U(0);
      warp_buf2[write_idx] = U(0);
    }
  }
}

template <typename T, typename U, typename V>
__device__ void cuLoadAddStridedInputs(
    const int i1_block, const int thr_load_row_off, const int thr_load_col_off,
    const int i2_off, const int row_stride, U *warp_buf1, U *warp_buf2,
    const T *input, const V *dout, const int i1_end, const int n2,
    const U *__restrict__ mean, const U *__restrict__ invvar) {
  int i1 = i1_block + thr_load_row_off;
  if (i1 < i1_end) {
    U curr_mean = mean[i1];
    U curr_invvar = invvar[i1];
    for (int k = 0; k < blockDim.y; ++k) {
      int i2 = i2_off + k;
      int load_idx = i1 * n2 + i2;
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      if (i2 < n2) {
        U curr_input = static_cast<U>(input[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);
        warp_buf1[write_idx] += curr_dout;
        warp_buf2[write_idx] +=
            curr_dout * (curr_input - curr_mean) * curr_invvar;
      }
    }
  }
}

template <typename T, typename U, typename V>
__global__ void cuComputePartGradGammaBeta(
    const V *__restrict__ dout, const T *__restrict__ input, const int n1,
    const int n2, const U *__restrict__ mean, const U *__restrict__ invvar,
    U epsilon, U *part_grad_gamma, U *part_grad_beta) {
  const int numsegs_n1 =
      (n1 + blockDim.y * blockDim.y - 1) / (blockDim.y * blockDim.y);
  const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
  const int i1_beg = blockIdx.y * segs_per_block * blockDim.y * blockDim.y;
  const int i1_beg_plus_one =
      (blockIdx.y + 1) * segs_per_block * blockDim.y * blockDim.y;
  const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;
  const int row_stride = blockDim.x + 1;
  const int thr_load_col_off = (threadIdx.x * blockDim.y) & (blockDim.x - 1);
  const int thr_load_row_off =
      (threadIdx.x * blockDim.y) / blockDim.x + threadIdx.y * blockDim.y;
  const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
  SharedMemory<U> shared;
  U *buf = shared.getPointer(); // buf has at least blockDim.x * blockDim.y *
                                // blockDim.y + (blockDim.y -
                                // 1)*(blockDim.x/blockDim.y) elements
  U *warp_buf1 = (U *)buf;      // NOLINT
  U *warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
  // compute partial sums from strided inputs
  // do this to increase number of loads in flight
  cuLoadWriteStridedInputs(i1_beg, thr_load_row_off, thr_load_col_off, i2_off,
                           row_stride, warp_buf1, warp_buf2, input, dout,
                           i1_end, n2, mean, invvar);
  for (int i1_block = i1_beg + blockDim.y * blockDim.y; i1_block < i1_end;
       i1_block += blockDim.y * blockDim.y) {
    cuLoadAddStridedInputs(i1_block, thr_load_row_off, thr_load_col_off, i2_off,
                           row_stride, warp_buf1, warp_buf2, input, dout,
                           i1_end, n2, mean, invvar);
  }
  __syncthreads();
  // inter-warp reductions
  // sum within each warp
  U acc1 = U(0);
  U acc2 = U(0);
  for (int k = 0; k < blockDim.y; ++k) {
    int row1 = threadIdx.y + k * blockDim.y;
    int idx1 = row1 * row_stride + threadIdx.x;
    acc1 += warp_buf1[idx1];
    acc2 += warp_buf2[idx1];
  }
  warp_buf1[threadIdx.y * row_stride + threadIdx.x] = acc1;
  warp_buf2[threadIdx.y * row_stride + threadIdx.x] = acc2;
  __syncthreads();
  // sum all warps
  for (int offset = blockDim.y / 2; offset > 1; offset /= 2) {
    if (threadIdx.y < offset) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + offset;
      int idx1 = row1 * row_stride + threadIdx.x;
      int idx2 = row2 * row_stride + threadIdx.x;
      warp_buf1[idx1] += warp_buf1[idx2];
      warp_buf2[idx1] += warp_buf2[idx2];
    }
    __syncthreads();
  }
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.y == 0 && i2 < n2) {
    int row1 = threadIdx.y;
    int row2 = threadIdx.y + 1;
    int idx1 = row1 * row_stride + threadIdx.x;
    int idx2 = row2 * row_stride + threadIdx.x;
    part_grad_beta[blockIdx.y * n2 + i2] = warp_buf1[idx1] + warp_buf1[idx2];
    part_grad_gamma[blockIdx.y * n2 + i2] = warp_buf2[idx1] + warp_buf2[idx2];
  }
}

template <typename U, typename V>
__global__ void
cuComputeGradGammaBeta(const U *part_grad_gamma, const U *part_grad_beta,
                       const int part_size, const int n1, const int n2,
                       V *grad_gamma, V *grad_beta) {
  // sum partial gradients for gamma and beta
  SharedMemory<U> shared;
  U *buf = shared.getPointer();
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i2 < n2) {
    // each warp does sequential reductions until reduced part_size is num_warps
    int num_warp_reductions = part_size / blockDim.y;
    U sum_gamma = U(0);
    U sum_beta = U(0);
    const U *part_grad_gamma_ptr =
        part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
    const U *part_grad_beta_ptr =
        part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
    for (int warp_offset = 0; warp_offset < num_warp_reductions;
         ++warp_offset) {
      sum_gamma += part_grad_gamma_ptr[warp_offset * n2];
      sum_beta += part_grad_beta_ptr[warp_offset * n2];
    }
    // inter-warp reductions
    const int nbsize3 = blockDim.x * blockDim.y / 2;
    for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
      // top half write to shared memory
      if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
        const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
        buf[write_idx] = sum_gamma;
        buf[write_idx + nbsize3] = sum_beta;
      }
      __syncthreads();
      // bottom half sums
      if (threadIdx.y < offset) {
        const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
        sum_gamma += buf[read_idx];
        sum_beta += buf[read_idx + nbsize3];
      }
      __syncthreads();
    }
    // write out fully summed gradients
    if (threadIdx.y == 0) {
      grad_gamma[i2] = sum_gamma;
      grad_beta[i2] = sum_beta;
    }
  }
}

template <typename T, typename U, typename V>
__global__ void cuComputeGradInput(const V *__restrict__ dout,
                                   const T *__restrict__ input, const int n1,
                                   const int n2, const U *__restrict__ mean,
                                   const U *__restrict__ invvar, U epsilon,
                                   const V *gamma, T *grad_input) {
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    const U c_mean = mean[i1];
    const U c_invvar = invvar[i1];
    const T *k_input = input + i1 * n2;
    const V *k_dout = dout + i1 * n2;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_h = static_cast<U>(k_input[l + k]);
          const U c_loss = static_cast<U>(k_dout[l + k]);
          const U gamma_tmp = static_cast<U>(gamma[l + k]);
          sum_loss1 += c_loss * gamma_tmp;
          sum_loss2 += c_loss * gamma_tmp * (c_h - c_mean) * c_invvar;
        }
      }
      for (; l < n2; ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        const U gamma_tmp = static_cast<U>(gamma[l]);
        sum_loss1 += c_loss * gamma_tmp;
        sum_loss2 += c_loss * gamma_tmp * (c_h - c_mean) * c_invvar;
      }
    } else {
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_h = static_cast<U>(k_input[l + k]);
          const U c_loss = static_cast<U>(k_dout[l + k]);
          sum_loss1 += c_loss;
          sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
        }
      }
      for (; l < n2; ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        sum_loss1 += c_loss;
        sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x / 2; mask > 0; mask /= 2) {
      sum_loss1 += WarpShflXor(sum_loss1, mask);
      sum_loss2 += WarpShflXor(sum_loss2, mask);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      SharedMemory<U> shared;
      U *buf = shared.getPointer();
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[2 * wrt_i] = sum_loss1;
          buf[2 * wrt_i + 1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss1 += buf[2 * read_i];
          sum_loss2 += buf[2 * read_i + 1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2 * threadIdx.x] = sum_loss1;
        buf[2 * threadIdx.x + 1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y != 0) {
        sum_loss1 = buf[2 * threadIdx.x];
        sum_loss2 = buf[2 * threadIdx.x + 1];
      }
    }
    // all threads now have the two sums over l
    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T *k_grad_input = grad_input + i1 * n2;
    if (gamma != NULL) {
      for (int l = thrx; l < n2; l += numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss * static_cast<U>(gamma[l]);
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx; l < n2; l += numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss;
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
    // prevent race where buf is written again before reads are done
    __syncthreads();
  }
}

static cudaDeviceProp GetDevicePropImpl() {
  int device = -1;
  PD_CHECK(cudaGetDevice(&device) == cudaSuccess);
  cudaDeviceProp prop;
  PD_CHECK(cudaGetDeviceProperties(&prop, device) == cudaSuccess);
  return prop;
}

static cudaDeviceProp *GetDeviceProp() {
  static auto prop = GetDevicePropImpl();
  return &prop;
}

template <typename T, typename U, typename V>
void HostApplyLayerNorm(V *output, U *mean, U *invvar, const T *input, int n1,
                        int n2, double epsilon, const V *gamma, const V *beta,
                        cudaStream_t stream) {
  const dim3 threads(32, 4, 1);
  const uint64_t maxGridY = GetDeviceProp()->maxGridSize[1];
  const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
  int nshared =
      threads.y > 1 ? threads.y * sizeof(U) + (threads.y / 2) * sizeof(U) : 0;
  cuApplyLayerNorm<<<blocks, threads, nshared, stream>>>(
      output, mean, invvar, input, n1, n2, U(epsilon), gamma, beta);
}

static void cuda_layer_norm(const paddle::Tensor &x,
                            const paddle::Tensor &scale,
                            const paddle::Tensor &bias, int rows, int cols,
                            float epsilon, paddle::Tensor *y,
                            paddle::Tensor *mean, paddle::Tensor *invvar) {
  DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
      x.type(), y->type(), "cuda_layer_norm_kernel",
      HostApplyLayerNorm(
          y->data<scalar_t_out>(), mean->data<float>(), invvar->data<float>(),
          const_cast<scalar_t_in *>(x.data<scalar_t_in>()), rows, cols, epsilon,
          const_cast<scalar_t_out *>(scale.data<scalar_t_out>()),
          const_cast<scalar_t_out *>(bias.data<scalar_t_out>()), x.stream()));
}

template <typename T, typename U, typename V>
void HostLayerNormGradient(const V *dout, const U *mean, const U *invvar,
                           const paddle::Tensor &input, int n1, int n2,
                           const V *gamma, const V *beta, double epsilon,
                           T *grad_input, V *grad_gamma, V *grad_beta,
                           cudaStream_t stream) {
  if (gamma != NULL && beta != NULL) {
    // compute grad_gamma(j) and grad_beta(j)
    const int part_size = 16;
    const dim3 threads2(32, 4, 1);
    const dim3 blocks2((n2 + threads2.x - 1) / threads2.x, part_size, 1);
    const int nshared2_a =
        2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
    const int nshared2_b = threads2.x * threads2.y * sizeof(U);
    const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
    auto place = input.place();
    paddle::Tensor part_grad_gamma =
        paddle::empty({part_size, n2}, paddle::DataType::FLOAT32, place);
    paddle::Tensor part_grad_beta = paddle::empty_like(part_grad_gamma);
    cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
        dout, input.data<T>(), n1, n2, mean, invvar, U(epsilon),
        part_grad_gamma.data<U>(), part_grad_beta.data<U>());

    const dim3 threads3(32, 8, 1);
    const dim3 blocks3((n2 + threads2.x - 1) / threads2.x, 1, 1);
    const int nshared3 = threads3.x * threads3.y * sizeof(U);
    cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
        part_grad_gamma.data<U>(), part_grad_beta.data<U>(), part_size, n1, n2,
        grad_gamma, grad_beta);
  }

  // compute grad_input
  const uint64_t maxGridY = GetDeviceProp()->maxGridSize[1];
  const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
  const dim3 threads1(32, 4, 1);
  int nshared = threads1.y > 1 ? threads1.y * threads1.x * sizeof(U) : 0;
  cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
      dout, input.data<T>(), n1, n2, mean, invvar, U(epsilon), gamma,
      grad_input);
}

static void
cuda_layer_norm_gradient(const paddle::Tensor &x, const paddle::Tensor &scale,
                         const paddle::Tensor &bias, const paddle::Tensor &mean,
                         const paddle::Tensor &invvar, const paddle::Tensor &dy,
                         int rows, int cols, float epsilon,
                         paddle::Tensor *grad_x, paddle::Tensor *grad_scale,
                         paddle::Tensor *grad_bias) {
  DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
      x.type(), scale.type(), "cuda_layer_norm_gradient_kernel",
      HostLayerNormGradient<scalar_t_in, float, scalar_t_out>(
          dy.data<scalar_t_out>(), mean.data<float>(), invvar.data<float>(), x,
          rows, cols, scale.data<scalar_t_out>(), bias.data<scalar_t_out>(),
          epsilon, grad_x->data<scalar_t_in>(),
          grad_scale->data<scalar_t_out>(), grad_bias->data<scalar_t_out>(),
          x.stream()));
}
