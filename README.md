# FlashKDA

FlashKDA: Flash Kimi Delta Attention — high-performance KDA kernels built on CUTLASS

## News

- **2026-04-22** — Deep-Dive Blog: the design decisions behind FlashKDA v1, read it [here](docs/20260420-flashkda-v1-deep-dive.md).

## Requirements
- SM90 and above
- CUDA 12.9 and above
- PyTorch 2.10 and above

## Installation
```bash
git clone https://github.com/MoonshotAI/FlashKDA.git flash-kda
cd flash-kda
git submodule update --init --recursive
pip install -v --no-build-isolation .
```

By default, the build detects the current CUDA device and compiles for that architecture. For wheel or CI builds, compile all supported architectures explicitly:

```bash
FLASH_KDA_CUDA_ARCHS=all pip install -v --no-build-isolation .
```

Supported values are `auto` (default), `all`, or a comma-separated arch list such as `90a,100a`.

## Using FlashKDA as an FLA backend

Once installed, FlashKDA is auto-dispatched from `flash-linear-attention`'s `chunk_kda`. See [fla-org/flash-linear-attention#852](https://github.com/fla-org/flash-linear-attention/pull/852) for integration details.

**Requirements**

1. Install `flash-linear-attention >= 0.5.0`:
   ```bash
   pip install -U flash-linear-attention
   ```
2. Call `chunk_kda` under `torch.inference_mode()` 
   ```python
   import torch
   from fla.ops.kda import chunk_kda

   with torch.inference_mode():
       out, final_state = chunk_kda(
           q=q, k=k, v=v, g=g, beta=beta,
           scale=scale,
           initial_state=h0,
           output_final_state=True,
           use_gate_in_kernel=True,
           use_qk_l2norm_in_kernel=True,
           use_beta_sigmoid_in_kernel=True,
           safe_gate=True,
           A_log=A_log, dt_bias=dt_bias,
           lower_bound=lower_bound,
           transpose_state_layout=True,
           cu_seqlens=cu_seqlens,
       )
   ```

**Opt out:** set `FLA_FLASH_KDA=0` to fall back to the Triton path.

**Debug dispatch:** add `logging.basicConfig(level=logging.INFO)` to see `[FLA Backend] kda.chunk_kda -> flashkda` on hit, or `... rejected: <reason>` on miss.

## Performance

See [BENCHMARK_H20.md](BENCHMARK_H20.md).

## Tests

```bash
bash tests/test.sh
```

- `tests/test_fwd.py` — correctness tests (exact match against the torch reference; compared with `flash-linear-attention`)
- `tests/test_vsplit.py` — bitwise coverage for the experimental V-split K2 path
- `tests/test_chunk_parallel.py` — the chunk-parallel recurrence (sm_10x): bitwise against the torch reference when a sequence is not split, error against an FP64 reference when it is, CUDA-graph replay, checkpoints


## Kernel API

### `flash_kda.fwd`

```python
flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
              initial_state=None, final_state=None, cu_seqlens=None,
              workspace=None)
```

**Parameters:**

| Parameter | Dtype | Shape | Description |
|---|---|---|---|
| `q` | bf16 | `[B, T, H, K]` | Query |
| `k` | bf16 | `[B, T, H, K]` | Key |
| `v` | bf16 | `[B, T, H, V]` | Value |
| `g` | bf16 | `[B, T, H, K]` | Gate before activation |
| `beta` | bf16 | `[B, T, H]` | Beta logits (pre-activation; sigmoid applied internally) |
| `scale` | float | scalar | scaling factor |
| `out` | bf16 | `[B, T, H, V]` | Output tensor |
| `A_log` | fp32 | `[H]` | Log-gate parameter |
| `dt_bias` | fp32 | `[H, K]` | Gate bias |
| `lower_bound` | float | scalar | Gate lower bound (range from -5.0 to 0) |
| `initial_state` | bf16/fp32/None | `[B, H, V, K]` or `[N, H, V, K]` | (optional) Initial recurrent state |
| `final_state` | bf16/fp32/None | `[B, H, V, K]` or `[N, H, V, K]` | (optional, output) Final recurrent state |
| `cu_seqlens` | int32/int64 | `[N+1]` | (optional) Cumulative sequence lengths for variable-length batching |
| `workspace` | uint8/None | `[workspace_size]` | (optional) Reusable workspace; allocated automatically when omitted |

- Currently requires `K = V = 128`.
- `initial_state` / `final_state` accept `None` (stateless), bf16, or fp32 tensors. When both are provided, their dtypes must match.
- When `cu_seqlens` is provided, `B` must be 1, `T` is the total length across all sequences, and `initial_state` / `final_state` have shape `[N, H, V, K]`.
- When `cu_seqlens` is `None`, each batch element is treated as an independent sequence, and the state shape is `[B, H, V, K]`.
- K2 automatically splits the value dimension into two `V=64` CTAs when
  `2 * H * N <= SM count`, targeting workloads with insufficient CTA-level
  parallelism.
- `workspace` must hold at least `get_workspace_size(B * T, H, N)` bytes for the
  current device; omit it to have one allocated.

### Chunk-parallel recurrence (sm_10x)

K2 walks each sequence's 16-token chunks serially, in one CTA per (sequence,
head, V slice), so a long sequence with few heads (for example 12 heads, a
96-head layer under 8-way tensor parallelism) keeps only a few dozen SMs busy.
The recurrence is affine in the state, `S_end = M S_start + C`, so on sm_10x
`fwd` cuts such sequences into segments that run in parallel:

1. K1 (prepare) runs unchanged.
2. Transfer: every non-final segment runs the K2 tile math on 256 columns:
   its 128 value columns from a zero state (`C`) and 128 identity columns with
   zero values (`M`).
3. Scan: per (sequence, head), `S_{s+1} = M_s S_s + C_s` in FP32.
4. Output: every segment runs the K2 tile math from its exact FP32 start
   state; the last one stores the final state, and the one holding a
   checkpoint exports it, as K2 does.

The choice is automatic and needs no host synchronization: `fwd` uses it when
`2 * H * N <= SM count` (where K2 would run its V-split tile) and a step of
the shape can benefit, and the segment length is then picked per step on the
device from `cu_seqlens`, so a captured CUDA graph stays valid for other
sequence lengths. Batches of shorter sequences stay whole. Calls whose
`final_state` or `checkpoint_state` aliases `initial_state`, workspaces sized
without the segment buffers, and other architectures run the serial K2.

Precision: a sequence that is not split is bit-identical to K2. A split one
keeps the FP32 state and the BF16 MMA operands, but rounds `M` and `C` to BF16
separately where K2 rounds their combination, so it is not bitwise equal to K2;
its error against an FP64 reference matches K2's.

Prepare + recurrence on GB300, 12 heads, FP32 state (CUDA events). Measured on
the out-of-tree build these kernels were ported from (`-DNDEBUG`, int32
`cu_seqlens`, dense Q/K/V); not yet re-measured with this package's build:

| tokens per sequence | serial (us) | chunk-parallel (us) |
|---|---|---|
| 16384 | 697.6 | 318.3 |
| 12288 | 529.9 | 253.4 |
| 8192 | 362.5 | 192.9 |
| 2 x 8192 | 388.7 | 324.1 |
| 4096 | 194.5 | 123.6 |
| 2048 | 103.8 | 97.2 |
| 4 x 3072 (kept whole) | 184.0 | 180.6 |

## Development

To set up IntelliSense (clangd) for the CUDA/C++ sources, run:

```bash
bash setup_clangd.sh
```

This generates a `.clangd` file with the correct repository paths and installs the global clangd `config.yaml` to `~/.config/clangd/`.

## Citation

```bibtex
@misc{flashkda2026,
      title={FlashKDA: Flash Kimi Delta Attention},
      author={Yutian Chen, Zhiyuan Li, Yucheng Wang, Ming Wei},
      year={2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/MoonshotAI/FlashKDA}},
}
```
