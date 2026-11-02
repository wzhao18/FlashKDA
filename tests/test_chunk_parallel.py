"""Chunk-parallel recurrence (``csrc/smxx/fwd_kernel_segment.cuh``).

On sm_10x, ``fwd`` cuts long sequences into segments that run in parallel and
joins them with an FP32 scan over the segments' affine state transfers.

- A sequence it does not split runs the K2 V-split tile math in K2's order:
  outputs and states are bit-identical to ``torch_ref`` and to the serial
  recurrence.
- A split sequence moves where the BF16 operand rounding of the state falls;
  its error against an FP64 token-by-token reference must match the serial
  recurrence's own error.
- The segment length is picked on the device from ``cu_seqlens``, so one
  captured CUDA graph must serve steps with other sequence lengths.

``_segment_chunks`` (tests only): ``> 0`` forces segments of that many
16-token chunks, ``-1`` the serial recurrence, ``0`` the automatic policy.
"""

from __future__ import annotations

import math

import pytest
import torch


if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 10:
    pytest.skip(
        "the chunk-parallel recurrence runs on sm_10x", allow_module_level=True
    )

import flash_kda
from torch_ref import torch_ref


D = 128
H = 4
LOWER_BOUND = -5.0
SERIAL = -1
AUTO = 0
UNSPLIT = 1 << 20  # one segment longer than any sequence here


def _cu(lens, dtype=torch.int32):
    offsets = [0]
    for n in lens:
        offsets.append(offsets[-1] + n)
    return torch.tensor(offsets, dtype=dtype, device="cuda")


def _inputs(lens, *, seed=0, batch=None, cu_dtype=torch.int32, state=torch.float32):
    """Varlen inputs for ``lens``, or ``batch`` fixed-length sequences of
    ``lens[0]`` tokens (no cu_seqlens)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if batch is None:
        B, T, N = 1, sum(lens), len(lens)
    else:
        B, T, N = batch, lens[0], batch

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    return dict(
        q=randn(B, T, H, D).bfloat16(),
        k=randn(B, T, H, D).bfloat16(),
        v=randn(B, T, H, D).bfloat16(),
        # Long-memory gates (per-token decay ~1 - 1e-4 .. e^-2.5): the state
        # carries across segment boundaries.
        g=(randn(B, T, H, D) * 2 - 6).bfloat16(),
        beta=randn(B, T, H).bfloat16(),
        A_log=torch.rand(H, device="cuda", generator=gen),
        dt_bias=torch.rand(H, D, device="cuda", generator=gen),
        initial_state=(randn(N, H, D, D) * 0.1).to(state),
        cu_seqlens=_cu(lens, cu_dtype) if batch is None else None,
    )


def _num_sequences(x):
    return x["initial_state"].shape[0]


def _run(
    x,
    segment_chunks,
    *,
    checkpoint_offsets=None,
    workspace=None,
    out=None,
    final=None,
    checkpoint=None,
):
    """``torch.ops.flash_kda.fwd`` (the op takes checkpoints) on NaN-filled
    outputs; returns (out, final[, checkpoint])."""
    B, T = x["q"].shape[:2]
    N = _num_sequences(x)
    if out is None:
        out = torch.full_like(x["v"], math.nan)
    if final is None:
        final = torch.full_like(x["initial_state"], math.nan)
    if checkpoint_offsets is not None and checkpoint is None:
        checkpoint = torch.zeros_like(x["initial_state"])
    if workspace is None:
        workspace = torch.empty(
            flash_kda.get_workspace_size(B * T, H, N, _segment_chunks=segment_chunks),
            dtype=torch.uint8,
            device="cuda",
        )
    torch.ops.flash_kda.fwd(
        x["q"], x["k"], x["v"], x["g"], x["beta"], D**-0.5, out, workspace,
        x["A_log"], x["dt_bias"], LOWER_BOUND, x["initial_state"], final,
        x["cu_seqlens"], checkpoint, checkpoint_offsets, segment_chunks,
    )  # fmt: skip
    if checkpoint_offsets is not None:
        return out, final, checkpoint
    return out, final


def _bounds(x):
    if x["cu_seqlens"] is not None:
        return x["cu_seqlens"].tolist()
    B, T = x["q"].shape[:2]
    return [b * T for b in range(B + 1)]


def _reference(x, checkpoint_offsets=None):
    """FP64 KDA, token by token, from the initial state as the kernels load it
    (BF16-narrowed); final state [N, H, V, K]. With ``checkpoint_offsets``,
    also the state after that many tokens of each sequence (zero where the
    offset is 0)."""
    q, k, v, g = (x[n].flatten(0, 1).double() for n in ("q", "k", "v", "g"))
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    decay = LOWER_BOUND * torch.sigmoid(
        torch.exp(x["A_log"].double())[None, :, None] * (g + x["dt_bias"].double())
    )
    beta = torch.sigmoid(x["beta"].flatten(0, 1).double())
    out = torch.empty_like(v)
    finals = []
    checkpoints = torch.zeros_like(x["initial_state"], dtype=torch.float64)
    offsets = checkpoint_offsets.tolist() if checkpoint_offsets is not None else None
    bounds = _bounds(x)
    for n in range(len(bounds) - 1):
        S = x["initial_state"][n].bfloat16().double().transpose(-1, -2).clone()
        for t in range(bounds[n], bounds[n + 1]):
            if offsets is not None and offsets[n] and t == bounds[n] + offsets[n]:
                checkpoints[n] = S.transpose(-1, -2)
            S *= torch.exp(decay[t])[:, :, None]
            kv = torch.einsum("hk,hkv->hv", k[t], S)
            S += beta[t][:, None, None] * k[t][:, :, None] * (v[t] - kv)[:, None, :]
            out[t] = D**-0.5 * torch.einsum("hk,hkv->hv", q[t], S)
        finals.append(S.transpose(-1, -2))
    out = out.view_as(x["v"])
    if offsets is not None:
        return out, torch.stack(finals), checkpoints
    return out, torch.stack(finals)


def _rel(x, y):
    return ((x.double() - y.double()).norm() / y.double().norm()).item()


@pytest.mark.parametrize("state", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))
@pytest.mark.parametrize(
    "lens,batch,cu_dtype",
    [
        ([0, 1, 0, 17, 0], None, torch.int32),  # empty and one-token sequences
        ([64, 0, 33], None, torch.int64),
        ([17], 2, torch.int64),  # fixed length, B = 2
    ],
    ids=("varlen_int32", "varlen_int64", "batched"),
)
def test_unsplit_matches_torch_ref(lens, batch, cu_dtype, state):
    """One segment per sequence is K2's tile loop: bit-identical to torch_ref,
    the final state of an empty sequence included."""
    x = _inputs(lens, batch=batch, cu_dtype=cu_dtype, state=state)
    out, final = _run(x, UNSPLIT)
    ref_out = torch.full_like(x["v"], math.nan)
    ref_final = torch.full_like(x["initial_state"], math.nan)
    torch_ref(
        x["q"], x["k"], x["v"], x["g"], x["beta"], D**-0.5, ref_out,
        A_log=x["A_log"], dt_bias=x["dt_bias"], lower_bound=LOWER_BOUND,
        initial_state=x["initial_state"].clone(), final_state=ref_final,
        cu_seqlens=x["cu_seqlens"],
    )  # fmt: skip
    assert torch.equal(out, ref_out)
    assert torch.equal(final, ref_final)


def test_policy_unsplit_is_serial():
    """A batch the automatic policy keeps whole is bit-identical to the serial
    recurrence."""
    x = _inputs([300, 1000, 17, 700])
    out, final = _run(x, AUTO)
    serial_out, serial_final = _run(x, SERIAL)
    assert torch.equal(out, serial_out)
    assert torch.equal(final, serial_final)


@pytest.mark.parametrize("state", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))
@pytest.mark.parametrize(
    "lens,batch,segment_chunks",
    [
        ([1024], None, 8),  # 8 segments
        ([300, 1000, 17, 700, 0], None, 4),  # tails, a 2-chunk sequence, an empty one
        ([1024], 2, 8),  # fixed length, B = 2
        ([2048], None, AUTO),  # the automatic policy splits a lone long sequence
    ],
    ids=("one", "varlen", "batched", "auto"),
)
def test_split_matches_reference(lens, batch, segment_chunks, state):
    x = _inputs(lens, seed=1, batch=batch, state=state)
    serial_out, serial_final = _run(x, SERIAL)
    ref_out, ref_final = _reference(x)
    out, final = _run(x, segment_chunks)
    assert not out.isnan().any() and not final.isnan().any()
    # The split moves the BF16 rounding of the state, not its precision: the
    # error against FP64 matches the serial recurrence's.
    serial_err = (_rel(serial_out, ref_out), _rel(serial_final, ref_final))
    err = (_rel(out, ref_out), _rel(final, ref_final))
    assert err[0] <= 1.02 * serial_err[0], (err, serial_err)
    assert err[1] <= 1.02 * serial_err[1], (err, serial_err)
    # And it must actually have split.
    assert not torch.equal(out, serial_out)
    assert _rel(out, serial_out) < 1e-2


def test_strided_qkv():
    """Packed Q/K/V (token stride 3 H D) through the split path match dense inputs."""
    x = _inputs([300, 1000, 17, 700], seed=4)
    packed = torch.stack((x["q"], x["k"], x["v"]), dim=2)
    x["q"], x["k"], x["v"] = (t.contiguous() for t in packed.unbind(2))
    expected = _run(x, 4)
    x["q"], x["k"], x["v"] = packed.unbind(2)
    actual = _run(x, 4)
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)


def test_in_place_state_stays_serial():
    """With final_state aliasing initial_state, the split would race the first
    segment's read of a row against the last segment's write: such calls keep
    the serial recurrence."""
    x = _inputs([2048], seed=5)
    serial_out, serial_final = _run(x, SERIAL)
    state = x["initial_state"].clone()
    x["initial_state"] = state
    out, final = _run(x, AUTO, final=state)
    assert torch.equal(out, serial_out)
    assert torch.equal(final, serial_final)
    with pytest.raises(RuntimeError, match="segment_chunks"):
        _run(x, 8, final=x["initial_state"])


def test_cuda_graph_replays_other_lengths():
    """The policy and the segment enumeration read cu_seqlens on the device: a
    graph captured for one split must replay another (same T, N) correctly."""
    first, second = [2048, 0], [1500, 548]
    x = _inputs(first, seed=2)
    T, N = sum(first), len(first)
    workspace = torch.empty(
        flash_kda.get_workspace_size(T, H, N), dtype=torch.uint8, device="cuda"
    )
    out = torch.empty_like(x["v"])
    final = torch.empty_like(x["initial_state"])
    _run(x, AUTO, workspace=workspace, out=out, final=final)  # warm up
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(x, AUTO, workspace=workspace, out=out, final=final)
    x["cu_seqlens"].copy_(_cu(second))
    graph.replay()
    expected_out, expected_final = _run(x, AUTO)
    torch.cuda.synchronize()
    assert torch.equal(out, expected_out)
    assert torch.equal(final, expected_final)
    # The replayed step split, too.
    serial_out, _ = _run(x, SERIAL)
    assert not torch.equal(out, serial_out)


def _checkpoint_offsets(lens):
    """A checkpoint at a 16-token boundary inside every sequence long enough,
    0 elsewhere."""
    offsets = [(n - 1) // 32 * 16 if n > 32 else 0 for n in lens]
    return torch.tensor(offsets, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize(
    "lens,segment_chunks",
    [
        ([300, 1000, 17, 700], AUTO),  # kept whole: bit-identical to serial
        ([2048, 1000, 17], 4),  # split: checkpoints inside and on segment edges
    ],
    ids=("unsplit", "split"),
)
def test_checkpoint(lens, segment_chunks):
    """Intermediate checkpoints: kept whole they are bit-identical to the
    serial recurrence; split, the checkpoint's error against FP64 matches its."""
    x = _inputs(lens, seed=3)
    offsets = _checkpoint_offsets(lens)
    if segment_chunks:
        # One checkpoint exactly on a segment boundary (the end of segment 1).
        offsets[0] = 2 * segment_chunks * 16
    out, final, checkpoint = _run(x, segment_chunks, checkpoint_offsets=offsets)
    serial_out, serial_final, serial_checkpoint = _run(
        x, SERIAL, checkpoint_offsets=offsets
    )
    if not segment_chunks:
        assert torch.equal(out, serial_out)
        assert torch.equal(final, serial_final)
        assert torch.equal(checkpoint, serial_checkpoint)
        return
    _, _, ref_checkpoint = _reference(x, offsets)
    has = offsets > 0
    err = _rel(checkpoint[has], ref_checkpoint[has])
    serial_err = _rel(serial_checkpoint[has], ref_checkpoint[has])
    assert err <= 1.02 * serial_err, (err, serial_err)
    assert not torch.equal(checkpoint, serial_checkpoint)
    # Rows without a checkpoint are left untouched.
    assert torch.equal(checkpoint[~has], torch.zeros_like(checkpoint[~has]))
