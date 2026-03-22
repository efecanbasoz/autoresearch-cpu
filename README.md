# autoresearch-cpu

> CPU adaptation of Karpathy's autoresearch — autonomous ML research on commodity hardware, no GPU required.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)

The original [autoresearch](https://github.com/karpathy/autoresearch) requires an NVIDIA GPU (tested on H100). This fork removes that requirement entirely, letting you run autonomous ML research experiments on any machine with a CPU. Perfect for learning, experimentation, and tinkering without cloud GPU costs.

---

## Table of Contents

- [Features](#features)
- [What Changed vs Original](#what-changed-vs-original)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Benchmark](#benchmark)
- [How It Works](#how-it-works)
- [Tuning for Your Hardware](#tuning-for-your-hardware)
- [Key Findings](#key-findings)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Zero GPU requirement** — Runs on any x86_64 CPU
- **Auto-detection** — Uses CUDA if available, falls back to CPU seamlessly
- **CPU-tuned defaults** — Optimized batch size, sequence length, and time budget for CPU
- **Backward compatible** — All changes work on both CPU and CUDA
- **Minimal dependencies** — `torch+cpu` only, zero NVIDIA packages needed

---

## What Changed vs Original

| Area | Original | This fork |
|------|----------|-----------|
| Attention | Flash Attention 3 (`kernels` package) | PyTorch native `scaled_dot_product_attention` |
| Mixed precision | bf16 autocast (CUDA) | fp32 (bf16 autocast [causes ~400x backward slowdown on CPU](https://github.com/pytorch/pytorch/issues/)) |
| Compilation | `torch.compile` with CUDA backend | Disabled on CPU (inductor backend crashes) |
| Device | Hardcoded `cuda` | Auto-detected via `get_device()` — works on both CPU and CUDA |
| Defaults | DEPTH=8, SEQ_LEN=2048, BATCH=128 | DEPTH=4, SEQ_LEN=512, BATCH=8 (tuned for CPU) |
| Time budget | 5 min | 30 min (compensates for slower hardware) |
| Sliding window | SSSL pattern via FA3 | Full causal attention (no sliding window) |
| Dependencies | `torch[cu128]` + `kernels` | `torch+cpu` (explicit CPU-only index, zero NVIDIA packages) |

All changes are backward-compatible — if a CUDA GPU is detected, the code uses CUDA paths automatically.

---

## Requirements

- **CPU**: Any x86_64 processor (tested on Intel Core Ultra 7 265, 20 cores)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Disk**: ~2GB for dataset + dependencies
- **Python**: 3.10+
- **No GPU needed**

---

## Quick Start

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

---

## Benchmark

Intel Core Ultra 7 265, 20 cores, 128GB RAM:

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

---

## How It Works

Same as the original — three files:

- **`prepare.py`** — data prep, tokenizer, dataloader, evaluation metric. Not modified by the agent.
- **`train.py`** — GPT model, Muon+AdamW optimizer, training loop. **The agent edits this file.**
- **`program.md`** — instructions for the AI agent.

Point your AI agent (Claude, GPT, Codex, etc.) at `program.md` and let it run experiments autonomously. Each experiment modifies `train.py`, trains for the time budget, checks `val_bpb`, and keeps or discards the change.

---

## Tuning for Your Hardware

If training is too slow or you want faster iteration:

| Parameter | File | Default | Try |
|-----------|------|---------|-----|
| `DEPTH` | train.py | 4 | 2 (smaller model, faster steps) |
| `MAX_SEQ_LEN` | prepare.py | 512 | 256 (4x less attention compute) |
| `DEVICE_BATCH_SIZE` | train.py | 8 | 4 or 2 (less memory per step) |
| `TIME_BUDGET` | prepare.py | 1800 | 300 (quick 5 min test) |
| `TOTAL_BATCH_SIZE` | train.py | 2^16 | 2^14 (fewer grad accum steps) |

For much smaller hardware, consider using [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) dataset — lower entropy text produces better results with tiny models.

---

## Key Findings

**bf16 autocast on CPU is broken for training.** `torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)` causes ~400x backward pass slowdown. Forward pass is fine, but autograd backward is catastrophically slow. This fork uses fp32 with `contextlib.nullcontext()` instead.

**torch.compile inductor crashes on CPU.** The inductor backend fails during C++ code generation for this model architecture. This fork skips compilation on CPU entirely. Training still works in eager mode.

---

## Security Notes

- **Tokenizer cache integrity**: The tokenizer is stored as a Python pickle at `~/.cache/autoresearch/tokenizer/tokenizer.pkl`. A SHA-256 hash is written alongside it at creation time and verified on every load. Do not import tokenizer caches from untrusted sources.
- **Data provenance**: Dataset shards are downloaded from HuggingFace over HTTPS. No content-integrity checksums are verified beyond TLS.
- **Autonomous execution**: The `program.md` workflow instructs AI agents to run indefinitely with local git operations. Always run on a dedicated branch in an isolated environment.

---

## Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Test with `uv run train.py` to verify
4. Commit your changes
5. Push to the branch and open a Pull Request

---

## License

[Apache-2.0](./LICENSE) — same as the original.

---

## Credits

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — the original project
- [nanochat](https://github.com/karpathy/nanochat) — the training code this is based on

## Other Platform Forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS MLX)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD ROCm)
