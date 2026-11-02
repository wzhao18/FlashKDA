import torch

from . import _C  # noqa: F401


def get_workspace_size(T_total, H, N=1, *, _segment_chunks=0):
    """Workspace bytes for ``fwd`` on the current CUDA device.

    On sm_10x this includes room for the chunk-parallel recurrence when a
    sequence of ``T_total`` tokens with ``H`` heads can be split. ``_segment_chunks``
    is for tests only and must match the value passed to ``fwd``.
    """
    return torch.ops.flash_kda.get_workspace_size(T_total, H, N, _segment_chunks)


def fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    workspace=None,
    *,
    _segment_chunks=0,
):
    """FlashKDA forward (Flash Kimi Delta Attention).

    Args:
        q (torch.Tensor): Query, bf16, shape ``[B, T, H, K]``.
        k (torch.Tensor): Key, bf16, shape ``[B, T, H, K]``.
        v (torch.Tensor): Value, bf16, shape ``[B, T, H, V]``.
        g (torch.Tensor): Gate before activation, bf16, shape ``[B, T, H, K]``.
        beta (torch.Tensor): Beta logits (pre-activation; sigmoid is applied
            internally), bf16, shape ``[B, T, H]``.
        scale (float): Scaling factor.
        out (torch.Tensor): Output buffer, bf16, shape ``[B, T, H, V]``. Written
            in place.
        A_log (torch.Tensor): Log-gate parameter, fp32, shape ``[H]``.
        dt_bias (torch.Tensor): Gate bias, fp32, shape ``[H, K]``.
        lower_bound (float): Gate lower bound, expected in ``[-5.0, 0]``.
        initial_state (torch.Tensor, optional): Initial recurrent state, bf16
            or fp32. Shape ``[B, H, V, K]`` for batched mode, or ``[N, H, V, K]``
            for varlen mode. ``None`` means start from zero.
        final_state (torch.Tensor, optional): Output buffer for the final
            recurrent state. Same dtype/shape rules as ``initial_state``.
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths, int32
            or int64, shape ``[N+1]``. When provided, ``B`` must be 1.
        workspace (torch.Tensor, optional): Reusable uint8 workspace of at
            least ``get_workspace_size(B * T, H, N)`` bytes. Allocated
            automatically when omitted.
        _segment_chunks (int): Tests only. ``0`` (default) picks the
            recurrence automatically; ``> 0`` forces the chunk-parallel
            recurrence with segments of this many 16-token chunks; ``< 0``
            forces the serial recurrence.
    Notes:
        * Currently requires ``K = V = 128``.
        * Q/K/V may have a padded token stride, with contiguous H/D rows,
          16-byte-aligned row starts, and a uniform stride across B*T.
          Beta may be strided. Other input and output tensors must be contiguous.
        * On sm_10x, long sequences with few heads run the chunk-parallel
          recurrence: each sequence is cut into segments that run in
          parallel and are joined by an FP32 scan over the segments' state
          transfers. It is chosen automatically, per step on the device from
          ``cu_seqlens`` (CUDA-graph safe). A sequence that is not split is
          bit-identical to the serial recurrence; a split one rounds the
          state's BF16 MMA operand at different points, with the same error
          against an FP64 reference.
    """
    B, T_seq, H = q.shape[:3]
    N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B
    if workspace is None:
        with torch.cuda.device(q.device):
            workspace_size = get_workspace_size(
                B * T_seq, H, N, _segment_chunks=_segment_chunks
            )
        workspace = torch.empty(workspace_size, dtype=torch.uint8, device=q.device)

    torch.ops.flash_kda.fwd(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        out,
        workspace,
        A_log,
        dt_bias,
        lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        segment_chunks=_segment_chunks,
    )
