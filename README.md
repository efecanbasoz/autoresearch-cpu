# autoresearch-cpu

CPU adaptation of Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) — autonomous ML research agents on commodity hardware, no GPU required.

The original autoresearch requires an NVIDIA GPU (tested on H100). This fork removes that requirement entirely, letting you run autonomous ML research experiments on any machine with a CPU. Perfect for learning, experimentation, and tinkering without cloud GPU costs.

## What changed vs original

| Area | Original | This fork |
|------|----------|-----------|
| Attention | Flash Attention 3 (`kernels` package) | PyTorch native `scaled_dot_product_attention` |
| Mixed precision | bf16 autocast (CUDA) | fp32 (bf16 autocast [causes ~400x backward slowdown on CPU](https://github.com/pytorch/pytorch/issues/)) |
| Compilation | `torch.compile` with CUDA backend | Disabled on CPU (inductor backend crashes) |
| Device | Hardcoded `cuda` | Auto-detected via `get_device()` — works on both CPU and CUDA |
| Defaults | DEPTH=8, SEQ_LEN=2048, BATCH=128 | DEPTH=4, SEQ_LEN=512, BATCH=8 (tuned for CPU) |
| Time budget | 5 min | 30 min (compensates for slower hardware) |
| Sliding window | SSSL pattern via FA3 | Full causal attention (no sliding window) |
| Dependencies | `torch[cu128]` + `kernels` | `torch` (CPU, no CUDA index) |

All changes are backward-compatible — if a CUDA GPU is detected, the code uses CUDA paths automatically.

## Requirements

- **CPU**: Any x86_64 processor (tested on Intel Core Ultra 7 265, 20 cores)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Disk**: ~2GB for dataset + dependencies
- **Python**: 3.10+
- **No GPU needed**

## Quick start

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py --num-shards 4

# 4. Run a training experiment (~40 min: 30 min training + eval)
uv run train.py
```

## Benchmark (Intel Core Ultra 7 265, 20 cores, 128GB RAM)

| Metric | Value |
|--------|-------|
| val_bpb | 1.955 |
| Training time | 307s (5 min budget) |
| Step time | ~12s |
| Throughput | ~5,000 tok/sec |
| Peak memory | 2.2 GB |
| Parameters | 11.5M |
| Steps completed | 32 |

With the default 30 min TIME_BUDGET, expect ~140 steps and better val_bpb (~1.5-1.7).

## How it works

Same as the original — three files:

- **`prepare.py`** — data prep, tokenizer, dataloader, evaluation metric. Not modified by the agent.
- **`train.py`** — GPT model, Muon+AdamW optimizer, training loop. **The agent edits this file.**
- **`program.md`** — instructions for the AI agent.

Point your AI agent (Claude, GPT, Codex, etc.) at `program.md` and let it run experiments autonomously. Each experiment modifies `train.py`, trains for the time budget, checks `val_bpb`, and keeps or discards the change.

## Tuning for your hardware

If training is too slow or you want faster iteration:

| Parameter | File | Default | Try |
|-----------|------|---------|-----|
| `DEPTH` | train.py | 4 | 2 (smaller model, faster steps) |
| `MAX_SEQ_LEN` | prepare.py | 512 | 256 (4x less attention compute) |
| `DEVICE_BATCH_SIZE` | train.py | 8 | 4 or 2 (less memory per step) |
| `TIME_BUDGET` | prepare.py | 1800 | 300 (quick 5 min test) |
| `TOTAL_BATCH_SIZE` | train.py | 2^16 | 2^14 (fewer grad accum steps) |

For much smaller hardware, consider using [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) dataset — lower entropy text produces better results with tiny models.

## Key findings

**bf16 autocast on CPU is broken for training.** `torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)` causes ~400x backward pass slowdown. Forward pass is fine, but autograd backward is catastrophically slow. This fork uses fp32 with `contextlib.nullcontext()` instead.

**torch.compile inductor crashes on CPU.** The inductor backend fails during C++ code generation for this model architecture. This fork skips compilation on CPU entirely. Training still works in eager mode.

## License

MIT — same as the original.

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START -->
<a href="https://github.com/sirkhet-dev"><img src="https://github.com/sirkhet-dev.png" width="60px" alt="sirkhet-dev" /></a>
<!-- ALL-CONTRIBUTORS-LIST:END -->

## Credits

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — the original project
- [nanochat](https://github.com/karpathy/nanochat) — the training code this is based on

## Other platform forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS MLX)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD ROCm)
