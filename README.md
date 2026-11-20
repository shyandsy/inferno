# inferno

> A from-scratch LLM inference engine - rebuilding vLLM's core ideas from zero to learn AI infra.
> An [epistem lab](https://github.com/EpistemLab) project. *epistem knows; inferno runs.*

## Why

Modern LLM serving engines - vLLM, SGLang, TensorRT-LLM - are fast because of a handful of systems ideas: paged KV cache, continuous batching, custom kernels. Those ideas are buried under a million-plus lines of code, easy to *use* and hard to actually *understand*.

'inferno' rebuilds them from scratch, one at a time, on a small model I can reason about end-to-end. Not a production engine - a learning engine, built in public.

## Approach

**Make it talk, then make it fast.** Every milestone is a self-contained, runnable step that adds one real capability. Each one lands first in **Python** - for clarity and as a correctness oracle - then gets rewritten in *C++/CUDA* for performance. Correctness is pinned to a reference: every version must still match Hugging Face GPT-2's output, token for token.

## Roadmap

### Phase 1 - Engine (reference backend, correctness first)

- [ ] **M1** - Minimal forward pass: hand-written GPT-2 forward, greedy decode, `backend` interface defined
- [ ] **M2** - KV cache
- [ ] **M3** - Paged KV Cache + Continuous batching *(the heart of vLLM)*

### Phase 2 - CUDA backend (single GPU performance)

- [ ] **M4** - CUDA backend: Python-C++-CUDA bridge, baseline kernels, first `nsys` profile
- [ ] **M5** - Bottleneck analysis + operator fusion (fuse RMSNorm/residual/activation, paged-attention kernel, report speedup **%** of SOL)

### Phase 3 - Scale & multi-GPU

- [ ] **M6** - Scale-up & stress: move to a 7-8B model, drive high concurrency/long context; surface compute-bound GEMM (MFU) & KV pressure; quantization as the memory-bound lever
- [ ] **M7** — Multi-GPU inference: tensor parallelism, NCCL collectives, compute/comm overlap

### Phase 4 - Platform backend

- [ ] **M8** — RK3588 backend: same interface, ARM/NPU implementation (engine untouched)

## Non-goals

Not production-grade (no auth, fault tolerance, or multi-node serving). Not broad
model coverage — one or two architectures, not a model zoo. Not trying to beat
vLLM — benchmarks are for learning, not winning. Inference only — no training
infra; distributed skills come from multi-GPU *inference*, not training. No later
phase before the earlier ones are solid; no single codebase pretending to be
optimal on every platform.

## Status

**M1 in progress.** Currently CPU-only; GPU work starts at M4.

## Stack

- **Engine / control plane** — Python
- **Reference backend** — PyTorch (correctness oracle; runs on CPU)
- **CUDA backend** — C++/CUDA custom kernels (NVIDIA GPU; `nsys` / `ncu`)
- **Platform backend** — RK3588 (ARM + RKNN NPU), Phase 4
- **Models** — GPT-2 (124M) as oracle; a small modern model (Qwen2.5-0.5B / Llama-3.2-1B, GQA) for mechanisms & profiling; a 7–8B model for scale & multi-GPU

## License

[MIT](LICENSE)