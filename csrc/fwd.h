#pragma once
#include <cstdint>

#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>

#include "segment_policy.h"

// Token strides are in elements; zero selects the dense H*D layout.
template <
    int D,
    bool HasStateIn = true,
    bool HasStateOut = true,
    bool StateFP32 = false,
    bool HasCheckpoint = false,
    bool IsVarlen = true,
    typename SeqlenT = int64_t>
void launch_fwd(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    void const* initial_state_ptr,
    float scale,
    void* final_state_ptr,
    void* checkpoint_state_ptr,
    SeqlenT const* checkpoint_offsets_ptr,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    int64_t q_token_stride,
    int64_t k_token_stride,
    int64_t v_token_stride,
    SeqlenT const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    int num_sms,
    cudaStream_t stream
);

// The two kernels of launch_fwd as standalone launches. launch_prepare (K1)
// fills ``workspace`` from q, k, g and beta; launch_recurrence (K2) reads
// that workspace with v and beta and writes ``out`` and the states. Both
// must use the same geometry (total_tiles, T_total, H, N, cu_seqlens).
template <
    int D,
    bool IsVarlen = true,
    typename SeqlenT = int64_t>
void launch_prepare(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    float scale,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    int64_t q_token_stride,
    int64_t k_token_stride,
    SeqlenT const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    cudaStream_t stream
);

template <
    int D,
    bool HasStateIn = true,
    bool HasStateOut = true,
    bool StateFP32 = false,
    bool HasCheckpoint = false,
    bool IsVarlen = true,
    typename SeqlenT = int64_t>
void launch_recurrence(
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    void const* initial_state_ptr,
    void* final_state_ptr,
    void* checkpoint_state_ptr,
    SeqlenT const* checkpoint_offsets_ptr,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    int64_t v_token_stride,
    SeqlenT const* cu_seqlens_ptr,
    int num_sms,
    cudaStream_t stream
);

// launch_fwd with the chunk-parallel recurrence (segment_policy.h): K1, then
// per-segment transfers, an FP32 scan over them and the per-segment outputs.
// The workspace holds the K1 workspace followed by segment_workspace_bytes()
// for the same total_tiles. The tile is K2's V-split tile, so call it only
// where launch_fwd would pick that tile (2 * H * N <= num_sms). The states may
// be absent (nullptr); state_fp32 gives their dtype. cu_seqlens_ptr == nullptr
// selects N fixed-length sequences. segment_chunks > 0 forces the segment
// length (tests); 0 picks it per step on the device.
template <int D, bool HasCheckpoint = false, typename SeqlenT = int64_t>
void launch_fwd_segmented(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    void const* initial_state_ptr,
    float scale,
    void* final_state_ptr,
    bool state_fp32,
    void* checkpoint_state_ptr,
    SeqlenT const* checkpoint_offsets_ptr,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    int64_t q_token_stride,
    int64_t k_token_stride,
    int64_t v_token_stride,
    SeqlenT const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    int segment_chunks,
    int num_sms,
    cudaStream_t stream
);
