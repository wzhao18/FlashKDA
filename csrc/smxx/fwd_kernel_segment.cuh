#pragma once
// Chunk-parallel recurrence for long sequences.
//
// K2 walks a sequence's 16-token chunks serially in one CTA per (head, V
// slice): one 16K-token sequence with 12 heads runs on 24 CTAs. The
// recurrence is affine in the state,
//
//     S_end = M_seg S_start + C_seg,
//
// where M_seg ([K, K]) depends only on the segment's keys, gates and betas and
// C_seg is the segment's end state from a zero start. So long sequences are cut
// into segments of `segment_length()` chunks (segment_policy.h) that run in
// parallel. After K1 (unchanged, same workspace):
//
//   transfer  every non-final segment runs the K2 tile math on 256 columns:
//             its 128 value columns from a zero state (C) and 128 identity
//             columns with zero values (M); no outputs;
//   scan      per (sequence, head, 16 value columns): S_0 = the initial state
//             (narrowed to BF16 as K2 loads it), S_{s+1} = M_s S_s + C_s, in
//             FP32 FFMA;
//   output    every segment runs the K2 tile math from S_s (FP32, exact;
//             segment 0 from the initial state as K2 loads it), writes its
//             outputs, and the last one stores the final state as K2 does
//             (through BF16). The segment that holds a checkpoint's chunk
//             exports it as K2 does.
//
// Every tile is the K2 V-split tile (BF16 MMA operands, FP32 accumulation,
// FP32 register-resident state), so a sequence that is not split is
// bit-identical to K2. A split one differs only in where the BF16 operand
// rounding of the state falls (M and C are rounded separately instead of
// their sum) and in the FP32 summation order.

#include <type_traits>

#include "utils.cuh"
#include "fwd_kernel2.cuh"
#include "segment_policy.h"

template <class SeqlenT>
struct SegmentParams {
    using BF16 = cutlass::bfloat16_t;
    // [N, H, V, K]; BF16 or FP32 (state_fp32); nullptr when absent.
    void const* initial_state;
    void* final_state;
    // Checkpoints (K2 semantics): the state after the first
    // checkpoint_offsets[seq] tokens (a multiple of 16; 0 = none).
    void* checkpoint_state;
    SeqlenT const* checkpoint_offsets;
    bool state_fp32;
    // Indexed by the dense enumeration of non-final segments (Segment::xfer0).
    float* transfer;  // [rows, H, 2V, K]: a segment's C^T (rows < V), M^T (rows >= V)
    float* start;     // [rows, H, V, K]: the exact start state of the next segment
    BF16* out;
    SeqlenT const* cu_seqlens;  // nullptr: N fixed-length sequences of T_total / N
    BF16 const* ws_kd;
    BF16 const* ws_qd;
    BF16 const* ws_kr;
    float const* ws_gt;
    BF16 const* ws_inv;
    BF16 const* ws_mqk;
    int T_total;
    int H;
    int N;
    int total_tiles;
    int segment_chunks;  // > 0 forces the segment length (tests); 0 picks it per step
    int num_sms;
};

template <class SeqlenT>
CUTLASS_DEVICE int64_t segment_seq_begin(SegmentParams<SeqlenT> const& p, int i) {
    if (p.cu_seqlens != nullptr) {
        return int64_t(p.cu_seqlens[i]);
    }
    return int64_t(i) * (p.T_total / p.N);
}

// The step's segment length, from cu_seqlens on the device (graph safe).
template <int CHUNK, class SeqlenT>
CUTLASS_DEVICE int segment_chunks(SegmentParams<SeqlenT> const& p) {
    if (p.segment_chunks > 0) {
        return p.segment_chunks;
    }
    int64_t total = 0;
    int64_t longest = 0;
    for (int i = 0; i < p.N; ++i) {
        int64_t const tiles =
            (segment_seq_begin(p, i + 1) - segment_seq_begin(p, i) + CHUNK - 1) / CHUNK;
        total += tiles;
        longest = tiles > longest ? tiles : longest;
    }
    return segment_length(longest, total, p.H, p.num_sms);
}

struct Segment {
    int seq;
    int seg;
    int nseg;
    int xfer0;   // dense index of the sequence's segment 0 among non-final segments
    int chunks;  // segment length in chunks
    int tile_base;
    int64_t bos;
    int64_t eos;
};

// Slots enumerate (sequence, segment) pairs densely in sequence order.
template <int CHUNK, class SeqlenT>
CUTLASS_DEVICE bool find_segment(int slot, SegmentParams<SeqlenT> const& p, Segment& s) {
    int const chunks = segment_chunks<CHUNK>(p);
    int acc = 0;
    int tile_base = 0;
    for (int i = 0; i < p.N; ++i) {
        int64_t const bos = segment_seq_begin(p, i);
        int64_t const eos = segment_seq_begin(p, i + 1);
        int const tiles = (int(eos - bos) + CHUNK - 1) / CHUNK;
        // An empty sequence keeps one segment, which stores its initial state as final.
        int const nseg = max(1, (tiles + chunks - 1) / chunks);
        if (slot < acc + nseg) {
            s.seq = i;
            s.seg = slot - acc;
            s.nseg = nseg;
            s.xfer0 = acc - i;
            s.chunks = chunks;
            s.tile_base = tile_base;
            s.bos = bos;
            s.eos = eos;
            return true;
        }
        acc += nseg;
        tile_base += tiles;
    }
    return false;
}

enum SegmentColumns : int {
    kInitialStart = 0,  // value columns from the initial state, as K2 loads it
    kExactStart = 1,    // value columns from an exact FP32 segment start state
    kZeroStart = 2,     // value columns from a zero state: C
    kIdentity = 3,      // identity columns with zero values: M
};

// K2's state as it loads it: BF16, or FP32 narrowed to BF16.
CUTLASS_DEVICE float segment_load_state(void const* state, bool fp32, int64_t idx) {
    using BF16 = cutlass::bfloat16_t;
    if (fp32) {
        return float(BF16(static_cast<float const*>(state)[idx]));
    }
    return float(static_cast<BF16 const*>(state)[idx]);
}

// K2's state as it stores it: through BF16.
CUTLASS_DEVICE void segment_store_state(void* state, bool fp32, int64_t idx, float value) {
    using BF16 = cutlass::bfloat16_t;
    BF16 const narrowed = BF16(value);
    if (fp32) {
        static_cast<float*>(state)[idx] = float(narrowed);
    } else {
        static_cast<BF16*>(state)[idx] = narrowed;
    }
}

// The K2 V-split pipeline buffers without its shared-memory state tile (the
// state stays in registers here); tiles without outputs keep a single, unused
// output stage.
template <class Layouts, int InputStages, int OutputStages, bool kOutputs>
struct SegmentStorage {
    using K2Storage = SharedStorageK2<Layouts, InputStages, OutputStages>;
    typename K2Storage::InputStorage input[InputStages];
    typename K2Storage::OutputStorage output[kOutputs ? OutputStages : 1];
    typename cutlass::PipelineTmaAsync<InputStages>::SharedStorage load_pipeline;
    typename cutlass::PipelineAsync<OutputStages>::SharedStorage store_pipeline;
};

// One CTA's segment: the K2 tile loop on one 64-column slice, specialized on
// whether the tiles write outputs (output kernel) and load values (all but
// the identity columns), and on checkpoint export.
template <
    bool kStoreOut,
    bool kLoadV,
    bool kCheckpoint,
    class TmaLoadV,
    class TmaLoadBeta,
    class TmaStoreOut,
    int CHUNK,
    int D,
    int InputStages,
    int OutputStages,
    int NumThreads,
    int VD,
    class SeqlenT>
CUTLASS_DEVICE void segment_tiles(
    TmaLoadV const& tma_load_v,
    TmaLoadBeta const& tma_load_beta,
    TmaStoreOut const& tma_store_out,
    SegmentParams<SeqlenT> const& p,
    Segment const& sg,
    int const head_idx,
    int const v_idx,
    int const columns,
    bool const last
) {
    using BF16 = cutlass::bfloat16_t;
    using Layouts = K2Layouts<D, CHUNK, VD>;
    using MMALayout = typename Layouts::MMALayout;
    using TransposedMMALayout = typename Layouts::TransposedMMALayout;
    using VOLayout = typename Layouts::VOLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using TransposedStateSmemLayout = typename Layouts::TransposedStateSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using TMAVOLayout = typename Layouts::TMAVOLayout;
    using TMABetaSmemLayout = typename Layouts::TMABetaSmemLayout;
    using SharedStorageT = SegmentStorage<Layouts, InputStages, OutputStages, kStoreOut>;
    int const T_total = p.T_total;
    int const H = p.H;
    int const total_tiles = p.total_tiles;
    int64_t const bos = sg.bos;
    int const seq_len = int(sg.eos - sg.bos);
    int const t_tiles = (seq_len + CHUNK - 1) / CHUNK;
    int const t_begin = sg.seg * sg.chunks;
    int const t_end = min(t_begin + sg.chunks, t_tiles);
    int const tile_base = sg.tile_base;

    extern __shared__ __align__(128) unsigned char shared_mem[];
    SharedStorageT& shared_storage = *reinterpret_cast<SharedStorageT*>(shared_mem);

    constexpr int kWarpSize = 32;
    constexpr int kComputeThreads = 128;
    // Tiles without outputs run no store warp.
    static_assert(NumThreads == kComputeThreads + (kStoreOut ? 2 : 1) * kWarpSize);

    uint32_t const v_bytes = uint32_t(cute::cosize_v<VOLayout>) * uint32_t(sizeof(BF16));
    // Tiles without outputs skip the q_decayed and Mqk operands.
    uint32_t const tma_bytes =
        (kLoadV ? v_bytes : 0u) +
        uint32_t(32) * uint32_t(sizeof(BF16)) +
        uint32_t(cute::cosize_v<MMALayout>) * uint32_t(sizeof(BF16)) * (kStoreOut ? 3 : 2) +
        uint32_t(cute::cosize_v<GTotalLayout>) * uint32_t(sizeof(float)) +
        uint32_t(cute::cosize_v<LMLayout>) * uint32_t(sizeof(BF16)) * (kStoreOut ? 2 : 1);

    int warp_id = cutlass::canonical_warp_idx_sync();
    WarpRole warp_role = WarpRole::NonParticipant;
    if (warp_id < kComputeThreads / kWarpSize) {
        warp_role = WarpRole::MMA;
    } else if (warp_id < kComputeThreads / kWarpSize + 1) {
        warp_role = WarpRole::LOAD_QKG;
    } else if (warp_id < kComputeThreads / kWarpSize + 2) {
        warp_role = WarpRole::STORE;
    }
    bool lane_predicate = cute::elect_one_sync();

    using LoadPipelineState = cutlass::PipelineState<InputStages>;
    using LoadPipeline = cutlass::PipelineTmaAsync<InputStages>;
    LoadPipeline load_pipeline = make_load_pipeline<InputStages>(
        shared_storage.load_pipeline, tma_bytes, warp_role, 1, kComputeThreads);
    using StorePipelineState = cutlass::PipelineState<OutputStages>;
    using StorePipeline = cutlass::PipelineAsync<OutputStages>;
    StorePipeline store_pipeline = make_store_pipeline<OutputStages>(
        shared_storage.store_pipeline, warp_role, kComputeThreads, 1);
    cutlass::pipeline_init_wait(1);

    // --- LOAD warp: v, beta and K1's per-tile operands
    if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
        Tensor g_v = tma_load_v.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_beta = tma_load_beta.get_tma_tensor(make_shape(H * T_total));
        LoadPipelineState load_write = cutlass::make_producer_start_state<LoadPipeline>();
        auto cta_tma_load_v = tma_load_v.get_slice(Int<0>{});
        auto cta_tma_load_beta = tma_load_beta.get_slice(Int<0>{});
        for (int t = t_begin; t < t_end; ++t) {
            load_pipeline.producer_acquire(load_write);
            using LoadBarrierType = typename LoadPipeline::ProducerBarrierType;
            LoadBarrierType* tma_barrier = load_pipeline.producer_get_barrier(load_write);
            int stage = load_write.index();
            int ws_idx = head_idx * total_tiles + tile_base + t;
            if constexpr (kLoadV) {
                auto v_off = g_v.layout()(head_idx, int(bos) + t * CHUNK, v_idx * VD);
                Tensor g_v_tile = make_tensor(g_v.data() + v_off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<VD>{}), stride(g_v.layout())));
                Tensor s_v_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].v.begin()), TMAVOLayout{});
                cute::copy(tma_load_v.with(*tma_barrier),
                    cta_tma_load_v.partition_S(g_v_tile), cta_tma_load_v.partition_D(s_v_tile));
            }
            int beta_linear = head_idx * T_total + (int(bos) + t * CHUNK);
            int beta_aligned = beta_linear & ~7;
            auto beta_off = g_beta.layout()(beta_aligned);
            Tensor g_beta_tile = make_tensor(g_beta.data() + beta_off, BetaSmemLayout{});
            Tensor s_beta_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].beta.begin()), TMABetaSmemLayout{});
            cute::copy(tma_load_beta.with(*tma_barrier),
                cta_tma_load_beta.partition_S(g_beta_tile), cta_tma_load_beta.partition_D(s_beta_tile));
            cute::SM90_BULK_COPY_G2S::copy(
                p.ws_kd + int64_t(ws_idx) * (CHUNK * D), reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].k_decayed.begin(), int32_t(CHUNK * D * sizeof(BF16)));
            if constexpr (kStoreOut) {
                cute::SM90_BULK_COPY_G2S::copy(
                    p.ws_qd + int64_t(ws_idx) * (CHUNK * D), reinterpret_cast<uint64_t*>(tma_barrier),
                    shared_storage.input[stage].q_decayed.begin(), int32_t(CHUNK * D * sizeof(BF16)));
            }
            cute::SM90_BULK_COPY_G2S::copy(
                p.ws_kr + int64_t(ws_idx) * (CHUNK * D), reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].k_restored.begin(), int32_t(CHUNK * D * sizeof(BF16)));
            cute::SM90_BULK_COPY_G2S::copy(
                p.ws_gt + int64_t(ws_idx) * D, reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].g_total.begin(), int32_t(D * sizeof(float)));
            cute::SM90_BULK_COPY_G2S::copy(
                p.ws_inv + int64_t(ws_idx) * (CHUNK * CHUNK), reinterpret_cast<uint64_t*>(tma_barrier),
                shared_storage.input[stage].INV.begin(), int32_t(CHUNK * CHUNK * sizeof(BF16)));
            if constexpr (kStoreOut) {
                cute::SM90_BULK_COPY_G2S::copy(
                    p.ws_mqk + int64_t(ws_idx) * (CHUNK * CHUNK), reinterpret_cast<uint64_t*>(tma_barrier),
                    shared_storage.input[stage].Mqk.begin(), int32_t(CHUNK * CHUNK * sizeof(BF16)));
            }
            ++load_write;
        }
        load_pipeline.producer_tail(load_write);
    }

    // --- MMA warps: the K2 per-tile math on register-resident FP32 state
    if (warp_role == WarpRole::MMA) {
        LoadPipelineState load_read;
        StorePipelineState out_write = cutlass::make_producer_start_state<StorePipeline>();
        int compute_tid = threadIdx.x;
        constexpr int kValueBlocksPerWarp = VD / ((kComputeThreads / kWarpSize) * 16);
        static_assert(kValueBlocksPerWarp == 1);

        // Layout reference only (never dereferenced).
        Tensor resident_s_acc_T = make_tensor(
            make_smem_ptr(shared_storage.input[0].k_decayed.begin()), TransposedStateSmemLayout{});
        auto resident_mma = make_tiled_mma(
            MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{}, Layout<Shape<_1,_1>>{}, Tile<_16,_16,_16>{});
        const int resident_warp_id = compute_tid / kWarpSize;
        const int resident_lane_id = compute_tid % kWarpSize;
        auto resident_thr_mma = resident_mma.get_slice(resident_lane_id);
        Tensor resident_state_ref = local_tile(
            resident_s_acc_T, make_shape(Int<16>{}, Int<16>{}),
            make_coord(0, resident_warp_id * kValueBlocksPerWarp));
        auto resident_c_ref = resident_thr_mma.partition_C(resident_state_ref);
        using ResidentStateFragment = decltype(make_fragment_like<BF16>(
            resident_thr_mma.make_fragment_C(resident_c_ref)));
        using ResidentStateAcc = decltype(resident_thr_mma.make_fragment_C(resident_c_ref));
        constexpr int kResidentStateRowBlocks = D / 16;
        ResidentStateAcc resident_state[kValueBlocksPerWarp][kResidentStateRowBlocks];

        // (k, column) of every resident element: the same partition as the fragments.
        Tensor state_coords = make_identity_tensor(make_shape(Int<D>{}, Int<VD>{}));
        int64_t const head_row = int64_t(sg.seq) * H + head_idx;
        int64_t const xfer = int64_t(sg.xfer0) + sg.seg;  // this segment's transfer
        #pragma unroll
        for (int m = 0; m < kResidentStateRowBlocks; ++m) {
            #pragma unroll
            for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
                Tensor coord_block = local_tile(
                    state_coords, make_shape(Int<16>{}, Int<16>{}),
                    make_coord(m, resident_warp_id * kValueBlocksPerWarp + bi));
                Tensor coords = resident_thr_mma.partition_C(coord_block);
                #pragma unroll
                for (int i = 0; i < size(coords); ++i) {
                    int const k = get<0>(coords(i));
                    int const column = v_idx * VD + get<1>(coords(i));
                    float value = 0.f;
                    if (columns == kInitialStart) {
                        if (p.initial_state != nullptr) {
                            value = segment_load_state(
                                p.initial_state, p.state_fp32, (head_row * D + column) * D + k);
                        }
                    } else if (columns == kExactStart) {
                        value = p.start[(((xfer - 1) * H + head_idx) * D + column) * D + k];
                    } else if (columns == kIdentity) {
                        value = k == column ? 1.f : 0.f;
                    }
                    resident_state[bi][m](i) = value;
                }
            }
        }

        // A checkpoint in this segment splits its tiles in two runs; the
        // checkpoint is exported between them, outside the tile loop, so
        // steps without checkpoints run the plain loop. As K2: an offset that
        // is not a positive multiple of CHUNK within the sequence exports nothing.
        int t_mid = t_end;
        bool export_checkpoint = false;
        if constexpr (kCheckpoint) {
            int64_t const checkpoint_offset = int64_t(p.checkpoint_offsets[sg.seq]);
            int64_t const tile = checkpoint_offset / CHUNK;  // tiles before the checkpoint
            if (checkpoint_offset > 0 && checkpoint_offset % CHUNK == 0 &&
                tile > t_begin && tile <= t_end) {
                t_mid = int(tile);
                export_checkpoint = true;
            }
        }

        constexpr int kRuns = kCheckpoint ? 2 : 1;
        for (int run = 0, lo = t_begin, hi = t_mid; run < kRuns; ++run, lo = t_mid, hi = t_end) {
        for (int t = lo; t < hi; ++t) {
            load_pipeline.consumer_wait(load_read);
            int load_stage = load_read.index();
            int out_stage = out_write.index();

            Tensor v_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].v.begin()), VOLayout{});
            Tensor beta_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].beta.begin()), BetaSmemLayout{});
            int beta_smem_offset = (head_idx * T_total + int(bos) + t * CHUNK) & 7;
            Tensor out_tile = make_tensor(make_smem_ptr(shared_storage.output[kStoreOut ? out_stage : 0].out.begin()), VOLayout{});

            Tensor k_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_decayed.begin()), MMALayout{});
            Tensor q_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].q_decayed.begin()), MMALayout{});
            Tensor g_total = make_tensor(make_smem_ptr(shared_storage.input[load_stage].g_total.begin()), GTotalLayout{});
            Tensor INV = make_tensor(make_smem_ptr(shared_storage.input[load_stage].INV.begin()), LMLayout{});
            Tensor Mqk = make_tensor(make_smem_ptr(shared_storage.input[load_stage].Mqk.begin()), LMLayout{});

            // Layout reference for the B fragments only; never dereferenced.
            Tensor s_acc = make_tensor(make_smem_ptr(shared_storage.input[0].k_decayed.begin()), StateSmemLayout{});

            // The K2 tile: v_sub, v_beta, U=INV@v, out=q@s, out+=Mqk@U, state update.
            {
            Tensor k_restored_t = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), TransposedMMALayout{});

            auto mma = make_tiled_mma(
                MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                Layout<Shape<_1,_1>>{},
                Tile<_16,_16,_16>{}
            );

            const int warp_id = compute_tid / 32;
            const int lane_id = compute_tid % 32;
            const int group_id = (lane_id / 4) % 8;

            ThrMMA thr_mma = mma.get_slice(lane_id);

            auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(lane_id);
            auto smem_tiled_copy_A_T = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_copy_A_T   = smem_tiled_copy_A_T.get_thread_slice(lane_id);
            auto smem_tiled_load_C  = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_load_C    = smem_tiled_load_C.get_slice(lane_id);
            auto smem_tiled_store_C = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
            auto smem_thr_store_C   = smem_tiled_store_C.get_slice(lane_id);

            Tensor A_ref = local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor B_ref = local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor C_ref = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));

            Tensor tCrAi_k = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_k_view = smem_thr_copy_A.retile_D(tCrAi_k);
            auto tCrA_k = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrAi_q = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_q_view = smem_thr_copy_A.retile_D(tCrAi_q);
            auto tCrA_q = thr_mma.partition_fragment_A(A_ref);

            auto tCrB = thr_mma.partition_fragment_B(B_ref);

            auto tCrC_ref = thr_mma.partition_C(C_ref);

            using AccFragT = decltype(thr_mma.make_fragment_C(tCrC_ref));
            using SFragT = decltype(make_fragment_like<BF16>(thr_mma.make_fragment_C(tCrC_ref)));
            using AFragT = decltype(thr_mma.partition_fragment_A(A_ref));
            using BFragT_u = decltype(thr_mma.partition_fragment_B(B_ref));

            AccFragT u_acc[kValueBlocksPerWarp], out_acc[kValueBlocksPerWarp];
            #pragma unroll
            for (int i = 0; i < kValueBlocksPerWarp; ++i) { u_acc[i] = thr_mma.make_fragment_C(tCrC_ref); clear(u_acc[i]); }
            #pragma unroll
            for (int i = 0; i < kValueBlocksPerWarp; ++i) { out_acc[i] = thr_mma.make_fragment_C(tCrC_ref); clear(out_acc[i]); }

            // ======== Phase 1: Dual GEMM k@s and q@s ========
            constexpr int K_BLOCKS = decltype(cute::size<1>(k_decayed))::value / 16;

            {
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_k_view);
            // Tiles that write no outputs (the transfer columns) skip q@s and Mqk@U.
            if constexpr (kStoreOut) {
                copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                    local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_q_view);
            }

            #pragma unroll
            for (int k = 0; k < K_BLOCKS; ++k) {
                cute::transform(tCrAi_k, tCrA_k, cute::identity{});
                if constexpr (kStoreOut) {
                    cute::transform(tCrAi_q, tCrA_q, cute::identity{});
                }
                movm_transpose_c_to_b_16x16(
                    narrow_state<ResidentStateFragment>(resident_state[0][k]), tCrB);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[0]);
                if constexpr (kStoreOut) {
                    gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[0]);
                }

                if (k + 1 < K_BLOCKS) {
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_k_view);
                    if constexpr (kStoreOut) {
                        copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                            local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_q_view);
                    }
                }
            }
            }

            // ======== Phase 2: Cast out (keep in regs), load v/INV/beta ========
            SFragT out_bf16[kValueBlocksPerWarp];
            if constexpr (kStoreOut) {
                #pragma unroll
                for (int i = 0; i < kValueBlocksPerWarp; ++i)
                    cute::transform(out_acc[i], out_bf16[i], [] __device__ (float x) { return BF16(x); });
            }

            SFragT v_bf16[kValueBlocksPerWarp];
            #pragma unroll
            for (int i = 0; i < kValueBlocksPerWarp; ++i) {
                if constexpr (kLoadV) {
                    Tensor v_block = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id * kValueBlocksPerWarp + i));
                    copy(smem_tiled_load_C, smem_thr_load_C.partition_S(v_block), smem_thr_load_C.retile_D(v_bf16[i]));
                } else {
                    // Identity columns carry zero values (their v tile is not loaded).
                    clear(v_bf16[i]);
                }
            }

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(INV), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            BF16 beta0 = BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + group_id))));
            BF16 beta1 = BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + group_id + 8))));
            // Rows past the sequence end hold another sequence's beta and v.
            // Zero them by select: 0 * NaN is NaN.
            const int tail_len = min(CHUNK, seq_len - t * CHUNK);
            const bool row0_valid = group_id < tail_len;
            const bool row1_valid = group_id + 8 < tail_len;

            // ======== Phase 3: u = (v - u) * beta; u = INV @ u (per block) ========
            SFragT u_bf16[kValueBlocksPerWarp];
            uint32_t u_b_regs[4];

            #pragma unroll
            for (int i = 0; i < kValueBlocksPerWarp; ++i) {
                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });

                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        u_bf16[i](c0) = (v_bf16[i](c0) - u_bf16[i](c0)) * beta0;
                        u_bf16[i](c1) = (v_bf16[i](c1) - u_bf16[i](c1)) * beta1;
                    }
                }
                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        if (!row0_valid) u_bf16[i](make_coord(make_coord(a, 0), 0, d)) = BF16::bitcast(0);
                        if (!row1_valid) u_bf16[i](make_coord(make_coord(a, 1), 0, d)) = BF16::bitcast(0);
                    }
                }

                uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16[i](0));
                SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                auto tCrB_u_tmp = thr_mma.partition_fragment_B(B_ref);
                uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_tmp(0));
                b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                clear(u_acc[i]);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_tmp(_,_,Int<0>{}), u_acc[i]);

                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });
            }

            // ======== Phase 4: Load Mqk, MOVM_T -> tCrB_u_arr, Mqk@U + add out ========
            if constexpr (kStoreOut) {
                copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(Mqk), tCrAi_k_view);
                cute::transform(tCrAi_k, tCrA_k, cute::identity{});
            }

            BFragT_u tCrB_u_arr[kValueBlocksPerWarp];

            #pragma unroll
            for (int i = 0; i < kValueBlocksPerWarp; ++i) {
                uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16[i](0));
                SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                tCrB_u_arr[i] = thr_mma.partition_fragment_B(B_ref);
                uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_arr[i](0));
                b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                if constexpr (kStoreOut) {
                    clear(out_acc[i]);
                    gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_arr[i](_,_,Int<0>{}), out_acc[i]);

                    SFragT gemm_bf16;
                    cute::transform(out_acc[i], gemm_bf16, [] __device__ (float x) { return BF16(x); });
                    cute::transform(out_bf16[i], gemm_bf16, out_bf16[i], [] __device__ (BF16 c, BF16 a) { return c + a; });
                }
            }

            // ======== Phase 5: Store out ========
            if constexpr (kStoreOut) {
                store_pipeline.producer_acquire(out_write);
                #pragma unroll
                for (int i = 0; i < kValueBlocksPerWarp; ++i) {
                    Tensor out_block = local_tile(out_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id * kValueBlocksPerWarp + i));
                    copy(smem_tiled_store_C, smem_thr_store_C.retile_S(out_bf16[i]), smem_thr_store_C.partition_D(out_block));
                }
            }

            // ======== Phase 6: state update ========
            // state[D, VD] = state * g_total + k_restored_t[D, 16] @ U[16, VD]
            constexpr int PREFETCH = 1;
            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16;

            Tensor tCrAi_kr = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_kr_view = smem_thr_copy_A_T.retile_D(tCrAi_kr);

            AFragT ring_A_kr[PREFETCH];
            float ring_g0[PREFETCH], ring_g1[PREFETCH];

            #pragma unroll
            for (int i = 0; i < PREFETCH; ++i) {
                Tensor kr_block = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(i, 0));
                copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_block), tCrAi_kr_view);
                cute::transform(tCrAi_kr, ring_A_kr[i], cute::identity{});

                ring_g0[i] = g_total(i * 16 + group_id);
                ring_g1[i] = g_total(i * 16 + group_id + 8);
            }

            #pragma unroll
            for (int m = 0; m < S_M_BLOCKS; ++m) {
                const int slot = m % PREFETCH;

                float g0 = ring_g0[slot];
                float g1 = ring_g1[slot];

                #pragma unroll
                for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
                    clear(u_acc[bi]);
                    gemm(thr_mma, ring_A_kr[slot](_,_,Int<0>{}), tCrB_u_arr[bi](_,_,Int<0>{}), u_acc[bi]);
                }

                if (m + PREFETCH < S_M_BLOCKS) {
                    Tensor kr_next = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, 0));
                    copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_next), tCrAi_kr_view);
                    cute::transform(tCrAi_kr, ring_A_kr[slot], cute::identity{});
                    ring_g0[slot] = g_total((m + PREFETCH) * 16 + group_id);
                    ring_g1[slot] = g_total((m + PREFETCH) * 16 + group_id + 8);
                }

                #pragma unroll
                for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
                    auto& state_fragment = resident_state[bi][m];
                    #pragma unroll
                    for (int a = 0; a < 2; ++a) {
                        #pragma unroll
                        for (int d = 0; d < 2; ++d) {
                            auto c0 = make_coord(make_coord(a, 0), 0, d);
                            auto c1 = make_coord(make_coord(a, 1), 0, d);
                            state_fragment(c0) = fmaf(state_fragment(c0), g0, u_acc[bi](c0));
                            state_fragment(c1) = fmaf(state_fragment(c1), g1, u_acc[bi](c1));
                        }
                    }
                }
            }
            }

            if constexpr (kStoreOut) {
                cutlass::arch::fence_view_async_shared();
                store_pipeline.producer_commit(out_write);
                ++out_write;
            }
            load_pipeline.consumer_release(load_read);
            ++load_read;
        }
        if constexpr (kCheckpoint) {
            // The checkpoint: the state after tile t_mid - 1, through BF16 (K2).
            if (run == 0 && export_checkpoint) {
                #pragma unroll
                for (int m = 0; m < kResidentStateRowBlocks; ++m) {
                    #pragma unroll
                    for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
                        Tensor coord_block = local_tile(
                            state_coords, make_shape(Int<16>{}, Int<16>{}),
                            make_coord(m, resident_warp_id * kValueBlocksPerWarp + bi));
                        Tensor coords = resident_thr_mma.partition_C(coord_block);
                        #pragma unroll
                        for (int i = 0; i < size(coords); ++i) {
                            int const k = get<0>(coords(i));
                            int const column = v_idx * VD + get<1>(coords(i));
                            segment_store_state(
                                p.checkpoint_state, p.state_fp32,
                                (head_row * D + column) * D + k, resident_state[bi][m](i));
                        }
                    }
                }
            }
        }
        }

        // --- end state: final (K2's BF16 store), or this segment's C / M rows
        #pragma unroll
        for (int m = 0; m < kResidentStateRowBlocks; ++m) {
            #pragma unroll
            for (int bi = 0; bi < kValueBlocksPerWarp; ++bi) {
                Tensor coord_block = local_tile(
                    state_coords, make_shape(Int<16>{}, Int<16>{}),
                    make_coord(m, resident_warp_id * kValueBlocksPerWarp + bi));
                Tensor coords = resident_thr_mma.partition_C(coord_block);
                #pragma unroll
                for (int i = 0; i < size(coords); ++i) {
                    int const k = get<0>(coords(i));
                    int const column = v_idx * VD + get<1>(coords(i));
                    float const value = resident_state[bi][m](i);
                    if constexpr (kStoreOut) {
                        if (last && p.final_state != nullptr) {
                            segment_store_state(
                                p.final_state, p.state_fp32, (head_row * D + column) * D + k, value);
                        }
                    } else {
                        int const row = columns == kIdentity ? D + column : column;
                        p.transfer[((xfer * H + head_idx) * 2 * D + row) * D + k] = value;
                    }
                }
            }
        }
    }

    if constexpr (kStoreOut) {
        if (warp_role == WarpRole::STORE && lane_predicate) {
            Tensor g_out = tma_store_out.get_tma_tensor(make_shape(H, T_total, D));
            auto cta_tma_store = tma_store_out.get_slice(Int<0>{});
            StorePipelineState out_read;
            for (int t = t_begin; t < t_end; ++t) {
                store_pipeline.consumer_wait(out_read);
                int stage = out_read.index();
                int actual_len = min(CHUNK, seq_len - t * CHUNK);
                BF16* out_stage_ptr = shared_storage.output[stage].out.begin();
                if (actual_len < CHUNK) {
                    // Manual store for the tail tile, so the next sequence is not overwritten.
                    Tensor s_out = make_tensor(make_smem_ptr(out_stage_ptr), VOLayout{});
                    for (int row = 0; row < actual_len; ++row) {
                        int64_t global_base = (bos + t * CHUNK + row) * H * D + head_idx * D + v_idx * VD;
                        for (int col = 0; col < VD; ++col) {
                            p.out[global_base + col] = s_out(row, col);
                        }
                    }
                } else {
                    auto out_off = g_out.layout()(head_idx, int(bos) + t * CHUNK, v_idx * VD);
                    Tensor g_out_tile = make_tensor(g_out.data() + out_off,
                        make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<VD>{}), stride(g_out.layout())));
                    Tensor s_out_tile = make_tensor(make_smem_ptr(out_stage_ptr), TMAVOLayout{});
                    cute::copy(tma_store_out, cta_tma_store.partition_S(s_out_tile), cta_tma_store.partition_D(g_out_tile));
                    tma_store_arrive();
                }
                tma_store_wait<0>();
                store_pipeline.consumer_release(out_read);
                ++out_read;
            }
        }
    }
}

// Transfers: blockIdx.x = head * 4 + group; groups 0-1 are value columns from
// a zero state (C), groups 2-3 identity columns with zero values (M).
// blockIdx.y enumerates (sequence, segment) slots; every segment but a
// sequence's last runs. No outputs, so four CTAs fit on an SM.
template <
    class TmaLoadV,
    class TmaLoadBeta,
    class TmaStoreOut,
    int CHUNK,
    int D,
    int InputStages,
    int OutputStages,
    int NumThreads,
    int VD,
    class SeqlenT>
__global__ void __launch_bounds__(NumThreads, 4)
_flash_kda_fwd_segment_transfer(
    CUTE_GRID_CONSTANT TmaLoadV const tma_load_v,
    CUTE_GRID_CONSTANT TmaLoadBeta const tma_load_beta,
    CUTE_GRID_CONSTANT TmaStoreOut const tma_store_out,
    CUTE_GRID_CONSTANT SegmentParams<SeqlenT> const p
) {
    static_assert(VD == 64, "segments run the K2 V-split tile");
    int const group = int(blockIdx.x) % 4;
    int const head_idx = int(blockIdx.x) / 4;
    Segment sg;
    if (!find_segment<CHUNK>(int(blockIdx.y), p, sg) || sg.seg + 1 == sg.nseg) {
        return;
    }
    if (group < 2) {
        segment_tiles<false, true, false, TmaLoadV, TmaLoadBeta, TmaStoreOut, CHUNK, D,
                      InputStages, OutputStages, NumThreads, VD>(
            tma_load_v, tma_load_beta, tma_store_out, p, sg, head_idx, group, kZeroStart, false);
    } else {
        segment_tiles<false, false, false, TmaLoadV, TmaLoadBeta, TmaStoreOut, CHUNK, D,
                      InputStages, OutputStages, NumThreads, VD>(
            tma_load_v, tma_load_beta, tma_store_out, p, sg, head_idx, group - 2, kIdentity, false);
    }
}

// Outputs: blockIdx.x = head * 2 + V slice, blockIdx.y = (sequence, segment)
// slot, every segment. Segment 0 starts from the initial state, every later
// one from its scanned FP32 start state; the last one stores the final state.
// HasCheckpoint is a separate instantiation, so steps without checkpoints run
// the plain tile loop.
template <
    class TmaLoadV,
    class TmaLoadBeta,
    class TmaStoreOut,
    int CHUNK,
    int D,
    int InputStages,
    int OutputStages,
    int NumThreads,
    int VD,
    bool HasCheckpoint,
    class SeqlenT>
__global__ void __launch_bounds__(NumThreads, 2)
_flash_kda_fwd_segment_output(
    CUTE_GRID_CONSTANT TmaLoadV const tma_load_v,
    CUTE_GRID_CONSTANT TmaLoadBeta const tma_load_beta,
    CUTE_GRID_CONSTANT TmaStoreOut const tma_store_out,
    CUTE_GRID_CONSTANT SegmentParams<SeqlenT> const p
) {
    static_assert(VD == 64, "segments run the K2 V-split tile");
    int const v_idx = int(blockIdx.x) % 2;
    int const head_idx = int(blockIdx.x) / 2;
    Segment sg;
    if (!find_segment<CHUNK>(int(blockIdx.y), p, sg)) {
        return;
    }
    segment_tiles<true, true, HasCheckpoint, TmaLoadV, TmaLoadBeta, TmaStoreOut, CHUNK, D,
                  InputStages, OutputStages, NumThreads, VD>(
        tma_load_v, tma_load_beta, tma_store_out, p, sg, head_idx, v_idx,
        sg.seg == 0 ? kInitialStart : kExactStart, sg.seg + 1 == sg.nseg);
}

// Scan: S_{s+1} = M_s S_s + C_s for one (sequence, head, 16 value columns),
// serially over the sequence's segments, in FP32 FFMA. A thread owns 8 rows
// (k) x 4 columns (v) of one quarter of the inner dimension (j); the quarters
// are summed in a fixed order. M_s^T and C_s are staged in shared memory with
// bulk copies while the previous step computes.
constexpr int kScanColumns = 16;
constexpr int kScanSplits = 4;
constexpr int kScanThreads = 256;

template <int D>
struct ScanSmem {
    float m[2][D][D];                              // M^T: [j][k]
    float c[2][kScanColumns][D];                   // C^T: [v][k]
    // S: [j][v], rows padded to 20 floats (16-byte aligned): the reduction's
    // column writes then conflict 4-way instead of 16-way.
    float s[D][kScanColumns + 4];
    float partial[kScanSplits][kScanColumns][D];   // [split][v][k]
    alignas(8) cutlass::arch::ClusterTransactionBarrier full[2];
};

template <int CHUNK, int D, class SeqlenT>
__global__ void __launch_bounds__(kScanThreads)
_flash_kda_fwd_segment_scan(CUTE_GRID_CONSTANT SegmentParams<SeqlenT> const p) {
    static_assert(D == 128 && kScanThreads == (D / 8) * (kScanColumns / 4) * kScanSplits);
    constexpr int kColumnBlocks = D / kScanColumns;
    constexpr int kSplitRows = D / kScanSplits;
    int const head_idx = int(blockIdx.x) / kColumnBlocks;
    int const column0 = (int(blockIdx.x) % kColumnBlocks) * kScanColumns;
    int const seq = int(blockIdx.y);
    int const H = p.H;

    // The sequence's segments (the same enumeration as find_segment).
    int const chunks = segment_chunks<CHUNK>(p);
    int xfer0 = 0;
    int nseg = 0;
    for (int i = 0; i <= seq; ++i) {
        int const tiles =
            int(segment_seq_begin(p, i + 1) - segment_seq_begin(p, i) + CHUNK - 1) / CHUNK;
        int const n = max(1, (tiles + chunks - 1) / chunks);
        if (i < seq) {
            xfer0 += n - 1;
        } else {
            nseg = n;
        }
    }
    if (nseg <= 1) {
        return;
    }

    extern __shared__ __align__(16) unsigned char scan_raw[];
    ScanSmem<D>& sm = *reinterpret_cast<ScanSmem<D>*>(scan_raw);

    int const tid = threadIdx.x;
    int const lane = tid % 32;
    int const warp = tid / 32;
    int const kq = lane % 16;                      // k = 4 kq + {0..3} and 64 + 4 kq + {0..3}
    int const vq = lane / 16 + 2 * (warp % 2);     // v = 4 vq .. 4 vq + 3
    int const split = warp / 2;                    // j in [32 split, +32)

    // One thread stages M_s^T (64 KB) and C_s's columns (8 KB) with bulk copies.
    if (tid == 0) {
        sm.full[0].init(1);
        sm.full[1].init(1);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
    auto stage = [&](int s, int buf) {
        if (tid == 0) {
            float const* rows = p.transfer + (int64_t(xfer0 + s) * H + head_idx) * 2 * D * D;
            constexpr uint32_t kMBytes = D * D * sizeof(float);
            constexpr uint32_t kCBytes = kScanColumns * D * sizeof(float);
            sm.full[buf].arrive_and_expect_tx(kMBytes + kCBytes);
            auto* bar = reinterpret_cast<uint64_t*>(&sm.full[buf]);
            cute::SM90_BULK_COPY_G2S::copy(rows + int64_t(D) * D, bar, &sm.m[buf][0][0], kMBytes);
            cute::SM90_BULK_COPY_G2S::copy(rows + int64_t(column0) * D, bar, &sm.c[buf][0][0], kCBytes);
        }
    };

    stage(0, 0);
    // S_0: the initial state as K2 loads it (zero when absent); stored [k][v].
    {
        int64_t const base = ((int64_t(seq) * H + head_idx) * D + column0) * D;
        for (int i = tid; i < kScanColumns * D; i += kScanThreads) {
            sm.s[i % D][i / D] = p.initial_state != nullptr
                ? segment_load_state(p.initial_state, p.state_fp32, base + i)
                : 0.f;
        }
    }
    for (int s = 0; s + 1 < nseg; ++s) {
        int const buf = s & 1;
        sm.full[buf].wait((s >> 1) & 1);
        __syncthreads();  // sm.s holds S_s; every thread has left step s - 1
        if (s + 2 < nseg) {
            stage(s + 1, buf ^ 1);  // buffer buf ^ 1 was last read in step s - 1
        }
        float acc[8][4];
        #pragma unroll
        for (int a = 0; a < 8; ++a) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                acc[a][b] = 0.f;
            }
        }
        #pragma unroll 8
        for (int jj = 0; jj < kSplitRows; ++jj) {
            int const j = split * kSplitRows + jj;
            float4 const m0 = *reinterpret_cast<float4 const*>(&sm.m[buf][j][4 * kq]);
            float4 const m1 = *reinterpret_cast<float4 const*>(&sm.m[buf][j][64 + 4 * kq]);
            float4 const x = *reinterpret_cast<float4 const*>(&sm.s[j][4 * vq]);
            float const mv[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
            float const xv[4] = {x.x, x.y, x.z, x.w};
            #pragma unroll
            for (int a = 0; a < 8; ++a) {
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    acc[a][b] = fmaf(mv[a], xv[b], acc[a][b]);
                }
            }
        }
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            *reinterpret_cast<float4*>(&sm.partial[split][4 * vq + b][4 * kq]) =
                make_float4(acc[0][b], acc[1][b], acc[2][b], acc[3][b]);
            *reinterpret_cast<float4*>(&sm.partial[split][4 * vq + b][64 + 4 * kq]) =
                make_float4(acc[4][b], acc[5][b], acc[6][b], acc[7][b]);
        }
        __syncthreads();  // partials written; every read of sm.s and of this buffer done
        float* dst = p.start + ((int64_t(xfer0 + s) * H + head_idx) * D + column0) * D;
        #pragma unroll
        for (int e = tid; e < kScanColumns * D; e += kScanThreads) {
            int const v = e / D;
            int const k = e % D;
            float sum = sm.partial[0][v][k];
            #pragma unroll
            for (int q = 1; q < kScanSplits; ++q) {
                sum += sm.partial[q][v][k];
            }
            sum += sm.c[buf][v][k];
            dst[e] = sum;
            sm.s[k][v] = sum;
        }
        // The next step's first barrier orders these writes before its reads.
    }
}
