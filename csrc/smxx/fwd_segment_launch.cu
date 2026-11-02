#include <type_traits>

#include "fwd.h"
#include "fwd_kernel1.cuh"
#include "fwd_kernel2.cuh"
#include "fwd_kernel_segment.cuh"

// ==================== launch_fwd_segmented ====================
// K1, then the chunk-parallel recurrence (transfer, scan, output); see
// fwd_kernel_segment.cuh. The segment length is picked on the device per step
// unless segment_chunks > 0 forces it.
template <int D, bool HasCheckpoint, typename SeqlenT>
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
) {
    using BF16 = cutlass::bfloat16_t;
    constexpr int CHUNK = 16;
    constexpr int VD = 64;
    constexpr int kInputStages = 3;
    constexpr int kOutputStages = 2;
    using K2L = K2Layouts<D, CHUNK, VD>;
    using WS = WorkspaceSizes<CHUNK, D>;

    // ===== K1 (prepare), exactly as launch_fwd runs it
    if (cu_seqlens_ptr != nullptr) {
        launch_prepare<D, true, SeqlenT>(
            q_ptr, k_ptr, g_bf16_ptr, beta_ptr, scale, workspace_ptr, total_tiles,
            T_total, H, N, q_token_stride, k_token_stride, cu_seqlens_ptr,
            A_log_ptr, dt_bias_ptr, gate_scale, stream);
    } else {
        launch_prepare<D, false, SeqlenT>(
            q_ptr, k_ptr, g_bf16_ptr, beta_ptr, scale, workspace_ptr, total_tiles,
            T_total, H, N, q_token_stride, k_token_stride, cu_seqlens_ptr,
            A_log_ptr, dt_bias_ptr, gate_scale, stream);
    }

    // --- K1 workspace (as launch_fwd lays it out), then the segment buffers
    int64_t const n_ht = int64_t(H) * total_tiles;
    char* ws = reinterpret_cast<char*>(workspace_ptr);
    int64_t const rows = segment_transfer_rows(total_tiles, segment_chunks);
    float* transfer = reinterpret_cast<float*>(ws + n_ht * WS::kPerTile);
    float* start = transfer + rows * H * 2 * D * D;

    SegmentParams<SeqlenT> params;
    params.initial_state = initial_state_ptr;
    params.final_state = final_state_ptr;
    params.checkpoint_state = checkpoint_state_ptr;
    params.checkpoint_offsets = checkpoint_offsets_ptr;
    params.state_fp32 = state_fp32;
    params.transfer = transfer;
    params.start = start;
    params.out = out_ptr;
    params.cu_seqlens = cu_seqlens_ptr;
    params.ws_kd  = reinterpret_cast<BF16 const*>(ws);
    params.ws_qd  = reinterpret_cast<BF16 const*>(ws + n_ht * WS::kKDecayed);
    params.ws_kr  = reinterpret_cast<BF16 const*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed));
    params.ws_gt  = reinterpret_cast<float const*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored));
    params.ws_inv = reinterpret_cast<BF16 const*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kGTotal));
    params.ws_mqk = reinterpret_cast<BF16 const*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kGTotal + WS::kINV));
    params.T_total = T_total;
    params.H = H;
    params.N = N;
    params.total_tiles = total_tiles;
    params.segment_chunks = segment_chunks;
    params.num_sms = num_sms;

    // --- TMA descriptors of the V-split tile (as K2's)
    auto gmem_layout = make_layout(make_shape(H, T_total, D), make_stride(D, D * H, 1));
    auto v_layout = make_layout(
        make_shape(H, T_total, D),
        make_stride(D, v_token_stride ? v_token_stride : int64_t(D) * H, 1));
    auto beta_gmem_layout = make_layout(make_shape(H * T_total));
    Tensor m_v = make_tensor(make_gmem_ptr(v_ptr), v_layout);
    Tensor m_out = make_tensor(make_gmem_ptr(out_ptr), gmem_layout);
    Tensor m_beta = make_tensor(make_gmem_ptr<BF16>(beta_ptr), beta_gmem_layout);
    auto tma_load_v = make_tma_copy(SM90_TMA_LOAD{}, m_v, typename K2L::TMAVOLayout{});
    auto tma_load_beta = make_tma_copy(SM90_TMA_LOAD{}, m_beta, typename K2L::TMABetaSmemLayout{});
    auto tma_store_out = make_tma_copy(SM90_TMA_STORE{}, m_out, typename K2L::TMAVOLayout{});

    constexpr int kOutputThreads = 128 + 2 * 32;  // math warpgroup + load + store warps
    constexpr int kTransferThreads = 128 + 32;    // math warpgroup + load warp
    int const output_smem = sizeof(SegmentStorage<K2L, kInputStages, kOutputStages, true>);
    int const transfer_smem = sizeof(SegmentStorage<K2L, kInputStages, kOutputStages, false>);
    auto transfer_kernel = _flash_kda_fwd_segment_transfer<
        decltype(tma_load_v), decltype(tma_load_beta), decltype(tma_store_out),
        CHUNK, D, kInputStages, kOutputStages, kTransferThreads, VD, SeqlenT>;
    auto output_kernel = _flash_kda_fwd_segment_output<
        decltype(tma_load_v), decltype(tma_load_beta), decltype(tma_store_out),
        CHUNK, D, kInputStages, kOutputStages, kOutputThreads, VD, HasCheckpoint, SeqlenT>;
    auto scan_kernel = _flash_kda_fwd_segment_scan<CHUNK, D, SeqlenT>;
    constexpr int scan_smem = sizeof(ScanSmem<D>);
    cudaFuncSetAttribute(transfer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, transfer_smem);
    cudaFuncSetAttribute(output_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, output_smem);
    cudaFuncSetAttribute(scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, scan_smem);

    // (sequence, segment) slots: every sequence's non-final segments plus its last.
    int const slots = int(rows) + N;

    // ===== transfers (M, C) of every segment but the last of its sequence
    transfer_kernel<<<dim3(H * 4, slots), dim3(kTransferThreads), transfer_smem, stream>>>(
        tma_load_v, tma_load_beta, tma_store_out, params);
    // ===== the start state of every segment after the first
    scan_kernel<<<dim3(H * (D / kScanColumns), N), dim3(kScanThreads), scan_smem, stream>>>(params);
    // ===== the outputs of every segment, from its start state
    output_kernel<<<dim3(H * (D / VD), slots), dim3(kOutputThreads), output_smem, stream>>>(
        tma_load_v, tma_load_beta, tma_store_out, params);
}

#define INSTANTIATE_LAUNCH_FWD_SEGMENTED(D, CKPT, SEQLEN_T) \
    template void launch_fwd_segmented<D, CKPT, SEQLEN_T>( \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, void const*, float, void*, bool, \
        void*, SEQLEN_T const*, cutlass::bfloat16_t*, void*, \
        int, int, int, int, int64_t, int64_t, int64_t, \
        SEQLEN_T const*, float const*, float const*, float, int, int, \
        cudaStream_t);

INSTANTIATE_LAUNCH_FWD_SEGMENTED(128, false, int32_t)
INSTANTIATE_LAUNCH_FWD_SEGMENTED(128, true,  int32_t)
INSTANTIATE_LAUNCH_FWD_SEGMENTED(128, false, int64_t)
INSTANTIATE_LAUNCH_FWD_SEGMENTED(128, true,  int64_t)
