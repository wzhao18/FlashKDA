#pragma once
// Chunk-parallel recurrence: the split policy and the workspace it needs.
//
// The serial recurrence (K2) walks a sequence's 16-token chunks in order, in
// one CTA per (sequence, head, V slice). A long sequence with few heads then
// runs on a few dozen CTAs while the rest of the GPU idles. The recurrence is
// affine in the state, S_end = M S_start + C, so a long sequence can be cut
// into segments that run in parallel and are joined by an FP32 scan over the
// segments' transfers (M, C). See smxx/fwd_kernel_segment.cuh.
//
// Everything here is integer arithmetic so that the host (dispatch, workspace
// sizing) and the device (per-step segment length, read from cu_seqlens so a
// captured CUDA graph stays valid for other sequence lengths) agree exactly.

#include <cstdint>

#include <cutlass/cutlass.h>

// Segments are at least this many 16-token chunks (sizes the workspace).
constexpr int kSegmentMinChunks = 48;
// At most this many segments per sequence.
constexpr int kSegmentMaxSegments = 11;
// Segment length meaning "do not split".
constexpr int kSegmentUnsplit = 1 << 28;

// Cost model in CTA-chunks (one CTA running one 16-token chunk), measured on
// GB300 with 12 heads and the V-split tile (two CTAs per head):
//   serial: ~0.54 us per chunk of the longest sequence while its CTAs fit on
//           the GPU, ~1.9 CTA-chunks/us per SM when they do not;
//   split:  3x the tile work (the C and M transfers, then the outputs) at
//           ~2.4 CTA-chunks/us per SM, plus ~30 us of scan and launches.
// Split iff  0.54 L > 3 W / (2.4 S) + 30  with W = 2 H total (the serial
// throughput bound W / (1.9 S) is always below the split cost, so it never
// decides). Scaled by 1900 S to integers:
//     1026 L S > 2375 W + 57000 S.
CUTLASS_HOST_DEVICE bool segment_split_pays(
    int64_t longest_chunks, int64_t total_chunks, int64_t H, int64_t num_sms) {
    int64_t const work = 2 * H * total_chunks;
    return 1026 * longest_chunks * num_sms > 2375 * work + 57000 * num_sms;
}

// Segment length (in chunks) for a step with `total_chunks` chunks whose
// longest sequence has `longest_chunks`. The longest runs get ~SMs / H - 1
// segments: the transfer (4 H CTAs per segment, four per SM) and output
// (2 H CTAs per segment, two per SM) launches then stay within one wave.
CUTLASS_HOST_DEVICE int segment_length(
    int64_t longest_chunks, int64_t total_chunks, int64_t H, int64_t num_sms) {
    if (!segment_split_pays(longest_chunks, total_chunks, H, num_sms)) {
        return kSegmentUnsplit;
    }
    int64_t segments = num_sms / H - 1;
    segments = segments < 2 ? 2 : segments;
    segments = segments > kSegmentMaxSegments ? kSegmentMaxSegments : segments;
    int64_t const chunks = (total_chunks + segments - 1) / segments;
    return int(chunks < kSegmentMinChunks ? kSegmentMinChunks : chunks);
}

// Host gate: can a step of at most `total_chunks` chunks split at all? The
// split condition grows with the longest sequence, which is at most the
// total, so this is a necessary condition for every cu_seqlens of the shape.
inline bool segment_split_possible(int64_t total_chunks, int64_t H, int64_t num_sms) {
    return segment_split_pays(total_chunks, total_chunks, H, num_sms);
}

// Upper bound on the non-final segments of a step with at most `total_chunks`
// chunks: every sequence's non-final segments are full, of at least
// min(segment_chunks) chunks. `segment_chunks > 0` forces the length (tests).
inline int64_t segment_transfer_rows(int64_t total_chunks, int64_t segment_chunks) {
    int64_t const min_chunks = segment_chunks > 0 ? segment_chunks : kSegmentMinChunks;
    return total_chunks / min_chunks;
}

// Workspace bytes of the chunk-parallel recurrence, after the K1 workspace:
// per non-final segment and head, its transfer [2D, D] (C^T rows, then M^T
// rows) and the start state [D, D] of the next segment, FP32.
inline int64_t segment_workspace_bytes(
    int64_t total_chunks, int64_t H, int64_t D, int64_t segment_chunks) {
    return segment_transfer_rows(total_chunks, segment_chunks) * H * 3 * D * D * 4;
}
