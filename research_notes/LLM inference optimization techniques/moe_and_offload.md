# MoE inference and CPU–GPU hybrid / offloaded execution (catalog, through Oct 2026)

Target box: RTX 4090 24 GB (sm_89), PCIe 3.0 x16 (~13 GB/s), i7-10700K (8C, AVX2 only, no AVX-512/AMX), 64 GB DDR4-3600 (~29 GB/s copy), slow USB disk or NVMe. Models: gpt-oss-120b (~63 GB, 128 experts, top-4), Qwen3.6-35B-A3B (256 experts, top-8), Gemma 4 26B-A4B, GLM-5.3-Flash (320B-A18B, ~93+ GB at IQ1).

Slot vocabulary used below (proposed for the plugin engine): **PLACE** (weight residency planner: which tensors/experts live in VRAM/RAM/disk), **CPU-EXP** (CPU expert GEMV/GEMM kernel), **GPU-MOE** (GPU grouped/fused MoE kernel, permute/unpermute), **XFER** (H2D transfer and prefetch engine, streams, pinned buffers), **CACHE** (GPU expert cache and eviction policy), **PRED** (next-expert predictor), **SCHED** (CPU/GPU overlap, pipeline, per-batch CPU-vs-GPU decision), **ROUTE** (lossy router changes: top-k reduction, skipping, rerouting), **COMPRESS** (quantization, pruning, merging of experts), **LOAD** (disk/mmap/NVMe loader), **SPEC** (speculative decoding interacting with offload).

Date tags: [YYYY-MM] = first public version (arXiv v1, PR merge, or venue).

## 1. Systems for running MoE / large models across CPU + GPU + disk

### Takeaway
Two families exist: (a) "compute where the weight lives" hybrids (llama.cpp/ik_llama.cpp `--n-cpu-moe`/`-ot`, KTransformers, Fiddler, HybriMoE, DALI, Apex, FreeToken), which suit low-batch decode on a host with decent RAM bandwidth; and (b) "move the weight to the GPU" offloaders (Mixtral-offloading, MoE-Infinity, ProMoE, HOBBIT, FlexGen, ZeRO-Inference, MoE-Lightning, MoE-Gen), which suit fast links, high cache-hit rates, or large-batch prefill and throughput. In 2026 the leading practical engines for consumer boxes are llama.cpp (GPU expert cache `--moe-cache-mib` merged Oct 7 2026), ik_llama.cpp, KTransformers, and the newer Strata fork. The academic frontier is moving to hybrid hit/miss execution (GPU-cached experts run on the GPU, misses run on the CPU), bandwidth-adaptive splitting, and stream-loading prefill.

### Cited Findings

**Production / open-source engines**
- **llama.cpp `--cpu-moe` / `--n-cpu-moe N`** [2025-08]: keeps the MoE expert tensors of the first N layers in system RAM, where the CPU computes them; attention, shared tensors and the remaining experts stay on the GPU. Slots: PLACE, CPU-EXP. The maintainers' gpt-oss guide reports ~30 t/s at zero context on an RTX 5090 with `--n-cpu-moe 21` for gpt-oss-120b, and ~20 t/s with all MoE layers on the CPU — [llama.cpp Discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396); explainer: [dev.to: Understanding MoE offloading](https://dev.to/someoddcodeguy/understanding-moe-offloading-5co6)
- **gpt-oss-120b on consumer hardware is RAM-bandwidth-bound.** One user went to 25+ tok/s only after fixing RAM that was running at 2000 MT/s instead of its rated 4800 MT/s. The author says token generation "scales with memory bandwidth" when experts sit in system RAM — [carteakey blog](https://carteakey.dev/blog/local-inference/optimizing-gpt-oss-120b-local-inference/). I found no clean RTX 4090 + DDR4 + `--n-cpu-moe` measurement for gpt-oss-120b, and an estimator's "~13 tok/s on 4090" is not a measurement — [modelfit.io](https://modelfit.io/best-hardware-for/gpt-oss-120b/)
- **llama.cpp `-ot` / `--override-tensor` regex placement**: assigns individual tensors (for example `ffn_.*_exps`) to a CPU or GPU buffer. ik_llama.cpp added it Feb 25 2025 (PR 232) — [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp). Slot: PLACE.
- **llama.cpp `--moe-cache-mib N` GPU expert cache** [merged 2026-10-07, PR #29887]: an LRU cache, one entry per expert, held in VRAM for experts that live in host memory. MUL_MAT_ID runs on the GPU and only cache misses are uploaded. It is used only for batches of ≤32 tokens; larger batches take the regular "copy weights to GPU" offload path. The author suggests about 10% of total expert size, and there is a `--moe-cache-mib auto` option. Ported from the qvac-fabric MoE cache. Author benchmarks used Qwen3.8-Flash-Next Q4_0 (93.7 GiB, 65.4 GiB of experts) on PCIe 4.0 x16 with an EPYC 7742 host:
  - RTX 4090: 25.0 → 39.4 t/s (`--fit --moe-cache-mib 6544`, 72% hit rate); 40.7 t/s (`-cmoe --moe-cache-mib 11000`, 1.62x, 77% hit rate).
  - RTX 5090: 30.8 → 67.8 t/s (2.20x, 89% hit rate).
  - AMD R9700 at IQ1_M: 11.0 → 27.6 t/s at 16 GB of cache. Prompt processing was unchanged.
  - Some ROCm/Vulkan users saw slowdowns.
  - A commenter estimated that break-even against CPU-only decode is about an 85% hit rate on PCIe 4.0 x8 for GLM-5.3-Flash. This is a community estimate.
  - A reviewer noted that the cache silently disables itself if the budget cannot hold `n_expert_used` slots.
  - Slots: CACHE, XFER, GPU-MOE. Sources: [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887); multi-GPU split in [PR #30112](https://github.com/ggml-org/llama.cpp/pull/30112); design discussion in [Discussion #24528 (RFC: hybrid hit/miss execution)](https://github.com/ggml-org/llama.cpp/discussions/24528); flag docs in the [Debian llama-server man page](https://manpages.debian.org/unstable/llama.cpp-tools/llama-server.1.en.html)
- **The cache depends heavily on capacity (V100, PCIe Gen3 x4 at 3.3 GB/s).** llama.cpp `--cpu-moe --moe-cache-mib`:
  - Flat speed down to 20 GB of cache (83% of experts, 99.1% hit rate).
  - 16 GB: 96.4% hit rate, but 4x the PCIe traffic, and about 23 t/s.
  - 8 GB: about 5 t/s.
  - The cache path cost about 10% versus whole-model-on-GPU, with 50 graph splits per token versus 2 for Strata.
  - Source: [Strata issue #3082 (2026-10-18)](https://github.com/Niko1221/Strata/issues/3082)
- **Hybrid hit/miss fork of llama.cpp** [2026]: computes cached experts on the GPU and missed experts on the CPU, aimed at slow PCIe links. It reports +31% token generation on an 8 GB GPU — [DavideWasTaken/llama.cpp](https://github.com/DavideWasTaken/llama.cpp)
- **ik_llama.cpp** (Iwan Kawrakow's fork): improved CPU matmul for AVX2 and NEON, IQK/trellis quants (IQ1_KT…IQ4_KT, CPU-capable). Dated MoE and offload features:
  - Fused FFN ops for MoE (PR 229, 2025-02-23).
  - `-ot` (PR 232, 2025-02-25).
  - Smart Expert Reduction `-ser`, a dynamic top-k reduction (PR 239, 2025-03-01).
  - Better MoE TG/PP on CUDA (PRs 248, 283).
  - GPU offload policy control: whether ops on RAM-resident tensors are offloaded to the GPU (PR 405, 2025-05-12).
  - "Better GPU offload strategy for MoE models when using hybrid" (PR 520).
  - Run-time repacking `-rtr` into row-interleaved CPU layouts. The README warns it can force matmuls onto the CPU, and K-quants have no CUDA row-interleaved path.
  - MLA/FlashMLA-3 for DeepSeek, described as "fastest CPU-only inference for DeepSeek".
  - Auto-fit of offloaded tensors (PRs 1501/1504).
  - Port of the GPU expert cache ([PR #2644](https://github.com/ikawrakow/ik_llama.cpp/pull/2644)).
  - Sources: [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp), [daily.dev summary](https://daily.dev/posts/ikawrakow-ik-llama-cpp-llama-cpp-fork-with-additional-sota-quants-and-improved-performance-esem3uuzj)
- **ik_llama.cpp vs mainline (anecdotal):** about 2x faster CPU inference for Qwen3.5 until a mainline speedup PR landed, after which the TG gap narrowed to ~10% — [HN comment](https://news.ycombinator.com/item?id=47296534)
- **KTransformers** (Tsinghua MADSys + Approaching.AI; SOSP 2025): attention and shared experts run on the GPU, routed experts on the CPU. Components:
  - AMX-specialized CPU kernels and an asynchronous CPU-GPU task scheduler.
  - **Expert Deferral**: overlaps CPU expert work with subsequent GPU work. It raises CPU utilization from typically below 75% to almost 100%, adds up to 1.45x throughput, and costs ≤0.5% accuracy, so it is lossy-ish.
  - Reported speedups over prior systems: 4.62–19.74x prefill, 1.25–4.09x decode.
  - Sources: [SOSP'25 paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf), [project page](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/), [repo](https://github.com/kvcache-ai/ktransformers)
  - The CPU kernels were upstreamed into SGLang as "GPU TP + CPU/GPU hybrid EP" (Oct 2025) — [LMSYS blog](https://lmsys.org/blog/2025-10-22-KTransformers), [SGLang issue #11425](https://github.com/sgl-project/sglang/issues/11425)
  - The OSDI'26 paper cites KTransformers at ~22 tok/s decode on INT4 rerouted DeepSeek-V3/R1 and 16 tok/s on DeepSeek-V3 INT4 — [OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf)
- **Strata** [2026] (MIT, hybridized llama.cpp fork; "Project Maya" forks Strata): targets giant MoEs across GPU/RAM/SSD, with kernels rewritten for specific architectures such as Qwen3.8-Flash-Next, plus an MTP drafter (`--spec`).
  - Community benchmark on a V100-PCIe 32 GB at Gen3 x4 with Coder IQ1_M: 98–101 t/s versus 41–43 t/s for llama.cpp whole-model-on-GPU; prompt reading about 2.9x faster. Without MTP it is still about 1.5x faster.
  - The "everyday" mode keeps 95% of experts on the GPU and 5% in RAM, at 85–92 t/s.
  - Self-reported, one model on one card; the techniques are not documented publicly.
  - Sources: [Strata issue #3082](https://github.com/Niko1221/Strata/issues/3082), [Medium comparison](https://medium.com/data-science-in-your-pocket/strata-vs-llama-cpp-the-best-way-to-run-a-125b-llm-on-your-gaming-pc-7e4afe3c1760), [MindStudio on Project Maya](https://www.mindstudio.ai/blog/project-maya-fork-strata-llama-cpp)
- **HF Accelerate big-model offload** (`device_map="auto"`, CPU/disk offload): moves layer weights to the GPU on demand. Simple but slow and latency-oriented — [Accelerate big model inference docs](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
- **DeepSpeed ZeRO-Inference** [2022-09]: keeps weights in CPU memory or NVMe and streams them layer by layer to the GPU. Throughput-oriented (large batch) — [DeepSpeed blog](https://www.deepspeed.ai/2022/09/09/zero-inference.html); [DeepSpeed-Inference paper arXiv 2207.00032](https://arxiv.org/abs/2207.00032)

**Academic systems (CPU compute vs transfer, hybrid)**
- **Fiddler** [2024-02; ICLR'25]: computes missed experts on the CPU instead of moving their weights over PCIe, because moving activations is far cheaper than moving weights. The v1 abstract reports uncompressed Mixtral-8x7B (>90 GB) at >3 tok/s on a single 24 GB GPU, an order of magnitude over prior offloaders. Slots: SCHED, CPU-EXP. Lossless — [arXiv 2402.07033](https://arxiv.org/abs/2402.07033), [repo efeslab/fiddler](https://github.com/efeslab/fiddler)
- **HybriMoE** [2025-04; DAC'25]: hybrid CPU-GPU scheduling and cache management for MoE, with dynamic intra-layer CPU/GPU task assignment, impact-driven prefetch and score-based caching. Built on KTransformers — [arXiv 2504.05897](https://arxiv.org/abs/2504.05897)
- **DALI** [2026-02]: workload-aware offloading on local PCs. It solves expert placement as a 0-1 integer program with a runtime greedy assignment, uses residual-based prefetching (predicting high-workload experts from inter-layer residuals), and a workload-aware cache replacement that exploits temporal correlation. The abstract claims prefill and decode speedups without numbers — [arXiv 2602.03495](https://arxiv.org/abs/2602.03495)
- **Apex** ("Efficient CPU-GPU collaborative MoE inference via Adaptive Expert Assignment") [2026-10]: assigns GPU-missed experts between the CPU and GPU, and speculatively predicts high-load experts one layer ahead so transfers start early. Up to 39.5% (avg 31.1%) decode throughput gain over the strongest baseline — [arXiv 2610.23006](https://arxiv.org/abs/2610.23006)
- **FreeToken** [2026-08]: argues that static placement misses most routed traffic on consumer hardware and that consumer CPU memory bandwidth is too small to absorb misses alone. It divides miss work between the CPU and the GPU-over-PCIe adaptively by bandwidth — [arXiv 2608.16157](https://arxiv.org/pdf/2608.16157)
- **RapidMoE** [2026-10]: adaptive residual offloading with importance arbitration that adjusts the critical expert set at runtime — [arXiv 2610.01265](https://arxiv.org/abs/2610.01265)
- **Cloud-grade SLOs for local MoE (OSDI'26, Tsinghua)** [2026-06]. Techniques:
  - **Stream-loading prefill (SLP)**: streams expert weights to the GPU for prefill, reaching 1,200 tok/s and 32K-token prompts in under 30 s.
  - **Distributed SLP + SmallEP**: 1,800 tok/s on 2x RTX 5090.
  - Intra-node prefill/decode disaggregation with zero-copy shared weights.
  - Dual-batch attention/MoE overlap: <15% latency increase, +50% throughput.
  - AVX-512 FP8 GEMV kernel: 4–5x lower CPU latency.
  - Fine-grained CPU parallelism: 28 tok/s on INT4 DeepSeek-V3 and 21.5 tok/s on intact FP8 V3, on dual-socket CPUs.
  - Sources: [OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf), [arXiv 2606.10493](https://arxiv.org/html/2606.10493). A third-party AI review flagged missing methodology for the throughput numbers — [pith.science](https://pith.science/paper/2606.10493)
- **CoX-MoE** [2026-05]: coalesced expert execution with AMX CPU-GPU co-execution (not usable without AMX) — [arXiv 2605.17889](https://arxiv.org/pdf/2605.17889)
- **Task scheduling for LLM inference on single moderate-GPU systems** [2024-11] — [arXiv 2411.15715](https://arxiv.org/pdf/2411.15715)
- **MoE-Lightning** [2024-11; ASPLOS'25]: throughput-oriented batch inference on memory-constrained GPUs. It uses the CGOPipe CPU-GPU-I/O pipeline with paged weights and a Hierarchical Roofline Model that searches the policy. Up to 10.3x throughput over prior offloaders for Mixtral 8x7B on a single T4 (16 GB) — [arXiv 2411.11217](https://arxiv.org/abs/2411.11217)
- **MoE-Gen** [2025-03]: high-throughput offline MoE generation on one GPU. It accumulates tokens in host memory and runs each module at a very large batch ("module-based batching") — [arXiv 2503.09716](https://arxiv.org/abs/2503.09716)
- **Klotski** [2025-02]: expert-aware multi-batch pipeline that overlaps loading of attention and expert weights with computation across batches, for throughput — [arXiv 2502.06888](https://arxiv.org/abs/2502.06888)
- **FlexGen** [2023-03; ICML'23]: LP-based policy search over GPU/CPU/disk placement of weights, KV and activations, with a zig-zag block schedule and 4-bit group-wise compression of weights and KV. Throughput-oriented, reporting up to 100x higher max throughput than prior offloaders for OPT-175B on one T4 — [arXiv 2303.06865](https://arxiv.org/abs/2303.06865)
- **PowerInfer** [2023-12; SOSP'24]: splits neurons into hot (GPU-resident) and cold (CPU-computed) using activation locality, with neuron-aware sparse operators and online predictors. It requires ReLU-sparse models (ReLU-fied, TurboSparse). Average 13.20 tok/s and peak 29.08 tok/s on a single RTX 4090, 18% below an A100 — [arXiv 2312.12456](https://arxiv.org/abs/2312.12456), [repo SJTU-IPADS/PowerInfer](https://github.com/SJTU-IPADS/PowerInfer)
- **PowerInfer-2** [2024-06]: smartphone version with neuron-cluster-level pipelining and flash I/O. Reports up to 29.2x over prior frameworks and TurboSparse-Mixtral-47B at 11.68 tok/s on a phone — [arXiv 2406.06282](https://arxiv.org/abs/2406.06282)
- **LLM in a Flash** (Apple) [2023-12]: runs models up to 2x DRAM from flash. It uses predicted FFN sparsity to load only the needed rows, a sliding "windowing" cache of recent neurons, and row-column bundling for larger contiguous reads. Reports 4–5x (CPU) and 20–25x (GPU) over naive loading — [arXiv 2312.11514](https://arxiv.org/abs/2312.11514)
- **MoNDE** [2024-05]: near-data computing. Cold experts run on a memory-side NDP device (CXL-style) and hot experts on the GPU. Needs special hardware — [arXiv 2405.18832](https://arxiv.org/abs/2405.18832)
- **Lina** [ATC'23]: in inference, dynamically re-allocates resources and replicates experts by popularity for distributed MoE — [USENIX ATC'23](https://www.usenix.org/conference/atc23/presentation/li-jiamin)
- **CommitMoE** [AAAI'26]: "fallback-free" offloaded MoE inference under GPU memory limits — [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/39454/43415)

### Inferences
- On this box, decode with experts in DDR4 is bounded by roughly (CPU-resident expert bytes per token) / 29 GB/s. Example for gpt-oss-120b at MXFP4: hidden = intermediate = 2880, so each expert has 3×2880×2880 ≈ 24.9M params ≈ 13 MB at ~4.25 bpw. Top-4 over 36 layers gives ≈1.9 GB of expert weights per token. With all experts on the CPU that is a ~15 tok/s ceiling. With about 30% of experts resident in VRAM (≈18 GB of experts), CPU traffic falls to ~1.3 GB per token, a ~22 tok/s ceiling before attention, synchronization and kernel inefficiency. This is consistent with the community 10–25 tok/s range.
- PCIe 3.0 x16 (13 GB/s) is about 2.2x slower per byte than the DDR4 copy bandwidth (29 GB/s). So for decode, uploading a missed expert to the GPU costs more than computing it in place on the CPU. Fiddler-style "compute where it lives" plus hybrid hit/miss is the right default; a pure upload-on-miss cache only pays off at high hit rates.
- Naive break-even for a pure upload-on-miss cache versus CPU compute is h ≈ 1 − 13/29 ≈ 55%, before overheads; the community estimate is about 85% on a PCIe 4.0 x8 link.
- A hybrid (hits on GPU, misses on CPU) is never worse than either pure strategy if scheduling overlaps.
- For prefill (large ubatch), the opposite holds. Streaming the ~59 GB of gpt-oss-120b experts once per ubatch over 13 GB/s takes ~4.5 s. At ubatch 4096 that is a PCIe ceiling of roughly 900 tok/s.
- On the CPU side, 8 AVX2 cores cannot approach that (~10 GFLOP per token at 5.1B active). So prefill should stream weights to the GPU with large `-ub` (llama.cpp's ≥32-token offload path, the OSDI'26 SLP). A per-batch-size crossover switch is mandatory.
- Qwen3.6-35B-A3B (~20 GB at Q4) and Gemma 4 26B-A4B (~15 GB at Q4) can be mostly or fully VRAM-resident, so offload matters mainly for gpt-oss-120b and GLM-5.3-Flash.
- GLM-5.3-Flash at ~93 GB IQ1 exceeds the 88 GB of RAM + VRAM. It therefore needs disk paging (hopeless on USB, marginal on NVMe) or a resident-subset method (REAP pruning or dynamic REAP; see section 2).
- AMX- and AVX-512-specific kernels (KTransformers AMX, OSDI'26 FP8 GEMV, CoX-MoE) do not transfer to the i7-10700K. Their scheduling ideas (async CPU/GPU overlap, expert deferral, SLP, dual-batch overlap) do transfer.

### Gaps
- No verified RTX 4090 + PCIe 3.0 + DDR4 measurements for gpt-oss-120b, Qwen3.6-35B-A3B, or GLM-5.3-Flash with `--n-cpu-moe` or `--moe-cache-mib` were found; all cache numbers above are PCIe 4.0 or Gen3 x4 V100.
- Strata's internals (caching, prediction, kernels) are undocumented in the sources found.
- The KTransformers Expert Deferral mechanics were only available at abstract level; my reading is that the CPU result of some experts is consumed one layer late, which explains the small accuracy cost. Verify in the PDF.
- Fiddler's later versions report different headline numbers (smaller speedups versus stronger baselines); only the v1 claim is given here.
- The MoE-Lightning, Klotski, MoE-Gen, HybriMoE, DALI and RapidMoE numbers were not re-fetched this session. Entries rely on abstracts, and numbers were omitted where I could not confirm them.

## 2. Expert caching, prefetching, prediction, and expert-level compression

### Takeaway
Caching (LRU/LFU and workload-aware eviction) plus next-layer prediction (gate lookahead, residual-based, learned predictors) are mature, lossless ideas. The most consequential 2026 development for a 24 GB box is lossy-but-adaptive resident-set management. **Dynamic REAP** (Dettmers, ICLR 2026, bitsandbytes2) keeps a fixed resident subset of experts, masks routing to it, and swaps 4 experts per layer every 256 tokens asynchronously. Combined with per-layer mixed precision, it fits a 125B-class MoE in 24 GB, and it was evaluated on Qwen3.6-35B-A3B and GLM-5.3-Flash. Static REAP pruning (Cerebras) is the zero-runtime-cost alternative. Dynamic top-k reduction (ik_llama `-ser`, AdapMoE, top-p routing) trades quality for less traffic.

### Cited Findings

**Caching and eviction**
- **Mixtral-offloading** (Eliseev & Mazur) [2023-12]: LRU expert cache in VRAM, speculative prefetch of next-layer experts by applying the next layer's gate to the current hidden state, and HQQ mixed quantization. Lossless apart from the quantization; latency-oriented — [arXiv 2312.17238](https://arxiv.org/abs/2312.17238), [repo dvmazur/mixtral-offloading](https://github.com/dvmazur/mixtral-offloading). LRU is the usual baseline policy — [PreScope arXiv 2509.23638](https://arxiv.org/html/2509.23638v1)
- **MoE-Infinity** [2024-01]: "sparsity-aware expert cache" for personal machines. It traces request-level expert activation (an Expert Activation Matrix) and prefetches and caches by request-level frequency. Slots: CACHE, PRED — [arXiv 2401.14361](https://arxiv.org/abs/2401.14361)
- **Diff-MoE** [SC'25]: priority-driven differential expert caching for batched inference. It splits prior work into prefetch-based and cache-based offloading — [ACM DL](https://dl.acm.org/doi/10.1145/3712285.3759903)
- **Least-Stale eviction / SpecMD**: a PyTorch-hook framework that composes caching, prefetch, miss-handling and routing policies. A 2026 reproducibility paper warns about replay semantics and workload contamination in cache evaluations — [arXiv 2608.07911](https://arxiv.org/pdf/2608.07911)
- **Ideas from the llama.cpp cache discussion** (not implemented): frequency-based admission instead of plain LRU, and reusing the compute buffer as cache space during generation — [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887)

**Prediction and prefetch**
- **Pre-gated MoE** [2023-08; ISCA'24]: modifies the model so each block's gate selects the experts of the next block. Prefetch then overlaps fully, but it needs fine-tuning — [arXiv 2308.12066](https://arxiv.org/abs/2308.12066)
- **ProMoE** [2024-10]: learned predictor that prefetches experts in a sliding window ("stride prefetch") on consumer GPUs. Lossless — [arXiv 2410.22134](https://arxiv.org/abs/2410.22134); summary in [PreScope](https://arxiv.org/html/2509.23638v1)
- **PreScope** [2025-09]: prefetching-focused offloaded MoE inference for resource-constrained settings — [arXiv 2509.23638](https://arxiv.org/html/2509.23638v1)
- **SpecPrefetch** [2026-07]: parameter-efficient expert prefetching for sparse MoE — [arXiv 2607.24787](https://arxiv.org/pdf/2607.24787)
- **DALI residual-based prefetching** [2026-02] — [arXiv 2602.03495](https://arxiv.org/abs/2602.03495). **Apex one-layer-ahead speculative high-load prediction** [2026-10] — [arXiv 2610.23006](https://arxiv.org/abs/2610.23006)
- **Single-layer lookahead accuracy of ~84–91%** is cited by an aggregator. Treat it as unverified — [emergentmind](https://www.emergentmind.com/topics/expert-offloading)
- **SiDA-MoE** [2023-10; MLSys'24]: an offline-trained hash function predicts experts per token, so experts can be offloaded or prefetched ahead of time — [arXiv 2310.18859](https://arxiv.org/abs/2310.18859)
- **Read-ME** [2024-10; NeurIPS'24]: refactors a dense model into an MoE with a router decoupled from the backbone. This enables pre-gating and lookahead scheduling — [arXiv 2410.19123](https://arxiv.org/abs/2410.19123)
- **ExpertFlow** [2024-10]: routing-path predictor plus token scheduling, to cut expert swaps — [arXiv 2410.17954](https://arxiv.org/abs/2410.17954)
- **fMoE** [2025-02]: fine-grained expert offloading driven by semantic and trajectory "expert maps" — [arXiv 2502.05370](https://arxiv.org/abs/2502.05370)
- **eMoE** [2025-03]: task-aware, memory-efficient MoE inference that predicts routing periodically — [arXiv 2503.06823](https://arxiv.org/abs/2503.06823)
- **DAOP** [2025-01]: data-aware offloading with predictive pre-calculation and per-sequence CPU/GPU expert allocation — [arXiv 2501.10375](https://arxiv.org/html/2501.10375v1)
- **AdapMoE** [2024-08]: adaptive gating (variable number of experts), adaptive prefetching, and DP-based cache allocation across layers — [arXiv 2408.10284](https://arxiv.org/abs/2408.10284)
- **EdgeMoE** [2023-08]: expert-wise bit-width adaptation plus predictive expert preloading on edge devices — [arXiv 2308.14352](https://arxiv.org/abs/2308.14352)
- **HOBBIT** [2024-11]: mixed-precision expert offloading. On a miss it loads a low-precision copy of less-important experts, cutting transfer bytes at a small quality cost, with token-, layer- and sequence-level policies — [arXiv 2411.01433](https://arxiv.org/abs/2411.01433). It is also cited as a prefetch+cache system in [Dettmers 2026](https://timdettmers.com/papers/runtime-dynamic-compression.pdf)
- **SwapMoE** [2023-08]: keeps a small, dynamically swapped set of "virtual experts" resident, for a bounded memory budget — [arXiv 2308.15030](https://arxiv.org/abs/2308.15030)
- **FloE** [2025-05]: on-the-fly MoE inference on memory-constrained GPUs that combines expert compression with sparse prediction. Reports 48.7x over DeepSpeed-MII on an RTX 3090, though against a weak baseline — [arXiv 2505.05950](https://arxiv.org/pdf/2505.05950v1)
- **ACM Computing Surveys "Survey on Inference Optimization Techniques for MoE Models"** covers Pre-gated MoE, MoE-Infinity, ProMoE and others — [ACM DL 10.1145/3794845](https://dl.acm.org/doi/10.1145/3794845)

**Runtime dynamic compression / dynamic REAP (Dettmers, ICLR 2026)** [2026-09]
- **Joint per-layer quantization and expert removal.** It allocates quantization level and expert removal jointly per layer, using "iterative sensitivity probing" (repeated probe-and-reallocate near the edge of instability, solved with a Lagrangian dual, perplexity-per-byte). It uses a HIGGS codebook quant ladder {0.7, 1, 1.125, …, 3, 4, 5} bpw (Hadamard rotation + vector quantization) — [paper PDF](https://timdettmers.com/papers/runtime-dynamic-compression.pdf)
- **Dynamic REAP mechanics.** Routing is restricted to a resident set R_i by masking the router logits. Non-resident experts keep accumulating unmasked routing probability, tracked with short- and long-term EMAs.
  - Every 256 routed tokens, non-residents are paired against residents by rank. If at least 4 pairs show an incoming score more than 5% above the outgoing score, 4 experts are swapped; an anti-thrashing rule blocks a swapped expert until the next checkpoint.
  - Experts are prepared asynchronously from CPU RAM or NVMe and swapped in at token boundaries. The paper says this "incurs practically no runtime speed overhead" and neither raises resident memory nor makes tokens wait.
  - Source: [paper PDF](https://timdettmers.com/papers/runtime-dynamic-compression.pdf)
- **Results.**
  - 0.4–1 bpw better than Unsloth dynamic quants at equal WikiText-2 perplexity. Qwen3.6-35B-A3B: Unsloth at 2.2 bpw ≈ bnb2 at 1.5 bpw. Qwen3.8-Flash-Next: 2.7 ≈ 1.7. GLM-5.3-Flash: 2.8 ≈ 2.5, and the paper calls GLM-5.3-Flash "very difficult to quantize" and its compression "not very successful".
  - A 125B-class model fits in 24 GB resident and a 550B-class model in 96 GB.
  - At 50% residency (128 of 256 experts) on Qwen3.6-35B-A3B, dynamic REAP keeps routing-mass coverage at 92.5–97.0% across all 25 cross-domain pairs, where static REAP collapses below 5% on some pairs. Dynamic beats static even when static is fitted to the test set.
  - Evaluation is perplexity-only (KLD correlation r≈0.98–0.99).
  - Source: [paper PDF](https://timdettmers.com/papers/runtime-dynamic-compression.pdf)
- **Release:** announced as integrated into bitsandbytes, private beta from Sept 22 2026 — [Dettmers on X](https://x.com/Tim_Dettmers/status/2102416553316647354); [paper notes issue](https://github.com/AkihikoWatanabe/paper_notes/issues/6699)

**Pruning, merging, skipping, top-k reduction (lossy, slot COMPRESS/ROUTE)**
- **REAP** (Cerebras, ICLR 2026) [2025-10]: a one-shot expert pruning criterion based on router gate values times expert activation norms. It beats merging on generative tasks, and is near-lossless on code for Qwen3-Coder-480B and Kimi-K2 at 50% pruning. Checkpoints are on HF (GLM-4.5-Air-REAP-82B-A12B, GLM-4.6-REAP-218B-A32B, MiniMax-M2-REAP variants at 25/30/40%). A March 2026 fix renormalizes top-k logits in the saliency calculation — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v1), [repo CerebrasResearch/reap](https://github.com/CerebrasResearch/reap), [Cerebras blog](https://www.cerebras.ai/blog/reap), [HF GLM-4.6-REAP](https://huggingface.co/cerebras/GLM-4.6-REAP-218B-A32B)
- **"Not All Experts are Equal"** [2024-02]: post-training expert pruning plus dynamic expert skipping during inference — [arXiv 2402.14800](https://arxiv.org/abs/2402.14800)
- **MC-SMoE** [2023-10; ICLR'24]: merges experts guided by routing statistics, then compresses them — [arXiv 2310.01334](https://arxiv.org/abs/2310.01334). **HC-SMoE** [2024-10]: retraining-free hierarchical clustering merge — [arXiv 2410.08589](https://arxiv.org/abs/2410.08589)
- **Mixture Compressor (MC)** [ICLR'25]: mixed-precision expert quantization plus dynamic pruning — [ICLR poster](https://iclr.cc/virtual/2025/poster/28749)
- **Benchmarking composable compression in MoE** [2026-08] — [arXiv 2608.21693](https://arxiv.org/pdf/2608.21693). **Pruning and distilling MoE into dense LMs** [2026-05] — [arXiv 2605.28207](https://arxiv.org/pdf/2605.28207)
- **Dynamic top-p routing ("Harder Tasks Need More Experts")** [2024-03]: a variable number of experts per token. It needs training — [arXiv 2403.07652](https://arxiv.org/abs/2403.07652)
- **Lynx** [2024-11]: dynamic, batch-aware expert selection that cuts the number of active experts at serving time — [arXiv 2411.08982](https://arxiv.org/abs/2411.08982)
- **ik_llama.cpp Smart Expert Reduction (`-ser`)** [2025-03]: runtime top-k reduction for DeepSeek-style models — [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp)
- **KTransformers Expert Deferral**: ≤0.5% accuracy drop for up to 1.45x — [SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- **QMoE** [2023-10]: sub-1-bit compression of trillion-parameter MoEs with a custom decoding kernel — [arXiv 2310.16795](https://arxiv.org/abs/2310.16795). **MoQE** [2023-10]: low-bit expert-only quantization (experts are robust to it) — [arXiv 2310.02410](https://arxiv.org/abs/2310.02410)
- **Shared-expert placement**: DeepSeekMoE introduced always-active shared experts — [arXiv 2401.06066](https://arxiv.org/abs/2401.06066). Hybrid systems pin attention and shared experts on the GPU and send routed experts to the CPU — [KTransformers coverage, noze.it](https://www.noze.it/en/insights/ktransformers-hybrid-cpu-gpu-inference/); [OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf)

### Inferences
- **Dynamic REAP fits this box's constraints.** Swaps happen every 256 tokens. 4 experts × N layers × expert size is a small trickle that fits easily within PCIe 3.0 or even NVMe bandwidth, and it needs no expert compute on the AVX2 CPU. It is the most promising route for GLM-5.3-Flash, which cannot be fully held in RAM + VRAM. The caveat is that the paper reports weak results specifically on GLM-5.3-Flash.
- **A runtime "resident-subset + masked routing" slot generalizes well.** It covers static REAP, dynamic REAP, and residency-aware top-k reduction. Pair it with a quality-eval harness (KLD vs reference) because these methods are lossy.
- **Prediction and prefetch help less here than on fast-link systems.** At 13 GB/s, a prefetched expert (about 13 MB for gpt-oss-120b) takes ~1 ms per expert. That is comparable to a whole layer's GPU time at batch 1, so prefetch can hide only a few experts per layer. It works best combined with hybrid execution: prefetched hits run on the GPU, unpredicted misses on the CPU.
- **Cache hit rates are workload-dependent and drift within a document**, which motivates frequency+recency eviction or EMA-based scoring (dynamic REAP's EMAs, DALI temporal correlation) over plain LRU.

### Gaps
- No public numbers were found for dynamic REAP decode throughput on an RTX 4090. The paper gives perplexity and coverage, not tok/s. bitsandbytes2 is in private beta, and its GGUF/llama.cpp interop is unknown.
- No source in this session gives ik_llama `-ser` quality and speed numbers.
- Predictor accuracy for gpt-oss-120b, Qwen3.6 and Gemma 4 routers specifically was not found.
- I could not verify the arXiv ID or details of "SpecMoEOff" (speculative decoding plus MoE offloading). Recall, unverified.

## 3. Data movement: overlap, pinned memory, streams, GDS/NVMe, mmap, transfer compression, CPU-vs-GPU cost model

### Takeaway
On PCIe 3.0 the transfer engine matters mostly for prefill (weight streaming at large ubatch) and for cache fills. For decode, the cost model says compute-on-CPU usually beats transfer. Mandatory building blocks: pinned (page-locked) staging buffers; separate copy and compute streams with events (double buffering); a batch-size threshold for CPU-compute vs GPU-stream; minimal graph splits and syncs per token; and NVMe/mmap only for cold-start or resident-subset swapping.

### Cited Findings
- **llama.cpp batch threshold.** It switches between paths at a batch threshold: ≤32 tokens use the expert cache or CPU compute; larger batches use the "copy host weights to GPU" offload path. Prefill does not benefit from the cache — [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887)
- **Pageable vs pinned uploads.** A reviewer raised that pageable-memory uploads on HIP can bottleneck the cache path. Observed transfer rates ranged from roughly 3 to 35 GB/s across setups — [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887)
- **Graph splits are a per-token cost.** 50 graph splits per token for llama.cpp with the expert cache versus 2 in Strata contributed to a ~10% overhead — [Strata issue #3082](https://github.com/Niko1221/Strata/issues/3082)
- **ik_llama.cpp offload policy and repacking.** It exposes a GPU offload policy to choose whether ops on RAM-resident tensors are offloaded to the GPU (PR 405). `-rtr` repacking can unintentionally pin matmuls to the CPU — [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp)
- **Stream-loading prefill** (stream expert weights to the GPU during prefill) reached 1,200 tok/s prefill. Dual-batch attention/MoE overlap keeps the CPU and GPU both busy under concurrency — [OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf)
- **CPU-GPU synchronization overhead and low CPU utilization are the main hybrid bottlenecks.** KTransformers addresses them with an async scheduler and Expert Deferral — [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/)
- **CGOPipe** pipelines CPU compute, GPU compute and I/O with paged weights. HRM, a roofline over GPU/CPU/PCIe bandwidths, picks the policy and is effectively a learned or analytical cost model — [MoE-Lightning arXiv 2411.11217](https://arxiv.org/abs/2411.11217)
- **FlexGen** formulates placement and schedule as an LP over GPU/CPU/disk with transfer compression (4-bit weights and KV) — [arXiv 2303.06865](https://arxiv.org/abs/2303.06865)
- **Bandwidth-adaptive splitting of miss work between CPU and PCIe** — [FreeToken](https://arxiv.org/pdf/2608.16157). **0-1 ILP placement with runtime greedy** — [DALI](https://arxiv.org/abs/2602.03495)
- **Flash/NVMe streaming**: windowing plus row-column bundling for larger contiguous reads — [LLM in a Flash](https://arxiv.org/abs/2312.11514). **NVMe weight offload** — [ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html). **Dynamic REAP prepares experts from CPU RAM or NVMe asynchronously** — [Dettmers 2026](https://timdettmers.com/papers/runtime-dynamic-compression.pdf)
- **Mixed-precision transfer**: loading low-bit copies of less-critical missed experts — [HOBBIT](https://arxiv.org/abs/2411.01433); **expert bit-width adaptation** — [EdgeMoE](https://arxiv.org/abs/2308.14352)
- **GPUDirect Storage (cuFile)**: direct NVMe→GPU DMA bypassing the CPU bounce buffer — [NVIDIA GDS docs](https://docs.nvidia.com/gpudirect-storage/). **Windows DirectStorage** offers an NVMe→GPU path with GPU decompression — [microsoft/DirectStorage](https://github.com/microsoft/DirectStorage)

### Inferences
- **Cost-model slot (SCHED).** Model for each MoE layer and batch size b:
  - CPU path: t_cpu ≈ max(bytes_cpu/BW_ddr(≈29 GB/s eff.), FLOPs/CPU_peak).
  - GPU-stream path: t_gpu ≈ bytes_missing/BW_pcie(≈13) + FLOPs/GPU.
  - Hybrid: max(t_cpu(misses), t_gpu(hits)) + sync cost.
  - For b ≲ 32, CPU or hybrid wins on this box. For b ≳ a few hundred, streaming wins.
  - The crossover should be auto-tuned per model, since the agent can measure it directly.
- **Practical PCIe 3.0 x16 rates need pinned memory.** ~13 GB/s requires pinned host memory. Pinning ~45 GB of the 64 GB RAM can strain Windows or Linux; a smaller pinned ring buffer with a CPU memcpy into it costs DDR bandwidth as well.
- **Graph splits and synchronizations per token** are a first-class tunable on this box. Each CPU↔GPU handoff costs tens of microseconds of latency, and at 36–92 layers this adds up. Strata's 2 splits per token versus 50 shows the headroom.
- **GPUDirect Storage limits.** GDS is Linux-oriented and, to my knowledge, officially targets data-center/workstation GPUs (GeForce runs in compatibility mode). It is unlikely to help here. DirectStorage is the Windows-native option. Disk paths matter only for GLM-5.3-Flash and for cold start; from a slow USB disk, mmap paging during decode would be catastrophic.
- **Transfer compression** (lower-bit expert copies for upload, HOBBIT-style) directly increases effective PCIe bandwidth, so it is relevant on PCIe 3.0. Dequantization must happen on the GPU.

### Gaps
- GDS support on GeForce/Windows was not verified this session; the claim above is from background knowledge.
- No measured pinned-vs-pageable numbers for this exact box were found.

## 4. Expert-parallel / routing kernels (including multi-GPU)

### Takeaway
On a single GPU what matters is the fused or grouped MoE GEMM (vLLM fused_moe Triton, SGLang, MegaBlocks block-sparse, ScatterMoE scatter-gather) and the cost of token permutation. Multi-GPU EP machinery (DeepEP, EPLB) is mostly irrelevant to one 4090. EPLB's ideas (replicating hot experts, popularity statistics) map onto expert-cache admission. Aux-loss-free balancing (DeepSeek-V3) produces flatter expert popularity, which lowers cache hit rates for offloading.

### Cited Findings
- **MegaBlocks** [2022-11; MLSys'23]: reformulates MoE as block-sparse GEMMs, so no token dropping or capacity-factor padding is needed — [arXiv 2211.15841](https://arxiv.org/abs/2211.15841)
- **ScatterMoE** [2024-03]: fused scatter/gather MoE kernels in Triton that avoid padding and extra copies — [arXiv 2403.08245](https://arxiv.org/abs/2403.08245)
- **vLLM fused_moe**: Triton grouped-GEMM kernels with sorted token alignment (moe_align_block_size) and permute/unpermute; tunable per-GPU config JSONs — [vLLM fused_moe source](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe)
- **SGLang hybrid EP with KTransformers CPU kernels** — [LMSYS blog](https://lmsys.org/blog/2025-10-22-KTransformers)
- **DeepEP**: all-to-all dispatch/combine kernels for expert parallelism (NVLink/RDMA, low-latency decode mode) — [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP). **EPLB**: replicates heavy-load experts and packs them across GPUs using load statistics — [deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB)
- **Aux-loss-free load balancing** (bias term on routing scores) — [arXiv 2408.15664](https://arxiv.org/abs/2408.15664); used in DeepSeek-V3 — [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- **Tutel**: adaptive parallelism and pipelining switching for MoE at runtime — [arXiv 2206.03382](https://arxiv.org/abs/2206.03382)
- **SmallEP expert parallelism** across 2 consumer GPUs for prefill — [OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf)
- **Mixture of Lookup Experts (MoLE)**: experts re-parameterized into lookup tables at inference, so offloaded experts cost a lookup instead of weight transfer. It requires a model trained this way — [arXiv 2503.15798](https://arxiv.org/abs/2503.15798)

### Inferences
- **Batch-1 decode on the 4090 uses GEMV-style fused MoE.** With top-4 or top-8 at batch 1, each expert sees one token. The GPU-MOE slot should have a GEMV-specialized path (one launch over all selected experts, MUL_MAT_ID-style) and a grouped-GEMM path for prefill. The agent can autotune Triton/CUDA tile configs on sm_89.
- **Capacity factors and token dropping** are training- and serving-scale concepts. For local inference, use dropless kernels.
- **Flatter routing hurts caching.** Models trained with strong balancing (aux-loss-free, DeepSeek-style) give flatter per-expert popularity over long windows. Within-document locality still exists (dynamic REAP coverage of 92.5–97% at 50% residency on Qwen3.6), so recency- and EMA-based caches still work.

### Gaps
- No kernel-level benchmarks for MXFP4 MoE GEMV on sm_89 (gpt-oss-120b's native format; the 4090 has no FP4 tensor cores, so it dequantizes to FP16/INT8) were collected.

## 5. Ranking: which techniques most likely help THIS box (PCIe 3.0, DDR4 ~29 GB/s, AVX2 8-core, RTX 4090)

### Takeaway
In priority order:
1. Static placement tuning (attention + shared + as many experts as fit in VRAM; `-ot` / `--n-cpu-moe`), with a per-batch CPU-vs-stream switch (stream for prefill).
2. Hybrid hit/miss execution: GPU-resident or cached experts on the GPU, misses computed on the CPU, never uploaded on miss during decode.
3. A frequency- or EMA-aware GPU expert cache sized from leftover VRAM.
4. Strong AVX2 CPU expert kernels (ik_llama.cpp-class) with tight CPU/GPU overlap and minimal graph splits.
5. For GLM-5.3-Flash and other >88 GB models: a resident subset via REAP or dynamic REAP plus low-bit mixed precision.
6. Lossy knobs (top-k reduction, expert deferral) as last-mile, quality-gated options.
7. Speculative decoding only if draft and verify reuse resident experts (DraftExpert insight, Strata MTP).

### Cited Findings
- Decode with experts in RAM scales with memory bandwidth — [carteakey blog](https://carteakey.dev/blog/local-inference/optimizing-gpt-oss-120b-local-inference/)
- GPU expert cache: 1.57–1.62x on a 4090 at 72–77% hit rate (PCIe 4.0); a 1.3x gain at a 33% hit rate (R9700); losses on some backends and setups — [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887)
- Hit rate collapses as the cache shrinks below the working set (V100 sweep) — [Strata issue #3082](https://github.com/Niko1221/Strata/issues/3082)
- Hybrid hit/miss on a slow link: +31% on an 8 GB GPU — [DavideWasTaken fork](https://github.com/DavideWasTaken/llama.cpp)
- Consumer CPU bandwidth alone cannot absorb misses; split them by bandwidth — [FreeToken](https://arxiv.org/pdf/2608.16157). Adaptive miss assignment plus one-layer-ahead prediction gives +31% avg — [Apex](https://arxiv.org/abs/2610.23006)
- Stream-loading prefill — [OSDI'26](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf)
- Dynamic REAP fits 125B-class in 24 GB, with high cross-domain coverage; weak on GLM-5.3-Flash — [Dettmers 2026](https://timdettmers.com/papers/runtime-dynamic-compression.pdf)
- REAP static pruning is near-lossless at 50% on code for large models — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v1)
- Speculative decoding under offload must count newly loaded experts, not drafted tokens — [DraftExpert arXiv 2607.24434](https://arxiv.org/html/2607.24434v1)
- Strata's MTP drafter gave about a 1.5x extra on top of its kernels on a slow link — [Strata issue #3082](https://github.com/Niko1221/Strata/issues/3082)
- Not applicable (hardware): AMX kernels ([KTransformers](https://github.com/kvcache-ai/ktransformers), [CoX-MoE](https://arxiv.org/pdf/2605.17889)), the AVX-512 FP8 GEMV ([OSDI'26](https://www.usenix.org/system/files/osdi26-wang-wenxin.pdf)), and NDP/CXL ([MoNDE](https://arxiv.org/abs/2405.18832))

### Inferences
Ranked list with expected regime and per-model notes:

1. **[PLACE] VRAM residency planner (Tier 1, lossless).**
   - Put attention, KV, embeddings, shared experts and dense layers on the GPU; fill the remaining VRAM with experts, chosen by measured popularity (not simply the first N layers) and in their native quant.
   - gpt-oss-120b: the main lever.
   - Qwen3.6-35B-A3B and Gemma 4 26B-A4B: at Q4 they likely fit entirely in 24 GB with modest context, so offloading matters little; the gains there come from GPU kernels.
   - Expected gain: each 10% of expert bytes moved to VRAM removes ~10% of the CPU DDR traffic per token.
2. **[SCHED] Per-batch path switch (Tier 1, lossless).**
   - Decode (b ≤ ~32): CPU/hybrid. Prefill: stream experts over PCIe with ubatch 2048–4096+.
   - Prefill is PCIe-bound at about 900 tok/s for gpt-oss-120b in theory; the 8-core AVX2 CPU alone is ~an order of magnitude slower.
3. **[SCHED+CACHE] Hybrid hit/miss execution with a GPU expert cache (Tier 1, lossless).**
   - Cached experts run on the GPU; misses run on the CPU in parallel; cache fills happen asynchronously off the critical path.
   - Avoids the PCIe-3.0 penalty of upload-on-miss (13 vs 29 GB/s). Use EMA/frequency admission to prevent thrashing.
4. **[CPU-EXP] AVX2-optimized quantized expert GEMV (Tier 1, lossless).**
   - ik_llama.cpp-class kernels, quant formats with cheap AVX2 dequantization, run-time repacking for CPU-only tensors, and NUMA-free thread pinning on 8 cores.
   - Goal: reach ≥80% of 29 GB/s effective. Decode is bandwidth-bound, so efficiency is everything.
5. **[SCHED] Overlap and sync minimization (Tier 1–2, lossless).**
   - Async CPU/GPU task queue, fewer graph splits, CUDA graphs on the GPU side.
   - KTransformers-style async scheduling; Expert Deferral is optional and lossy.
6. **[COMPRESS] Lower-bit experts (Tier 2, lossy).**
   - Mixed-precision per layer and expert (Unsloth-style dynamic quants, bnb2 HIGGS ladder).
   - Fewer bytes per token over DDR directly raises tok/s, and more experts fit in VRAM. For gpt-oss-120b, MXFP4 is native, so further quantization is riskier.
7. **[COMPRESS+ROUTE] Resident subset: REAP (static) or dynamic REAP (Tier 2 for GLM-5.3-Flash, lossy).**
   - The only realistic way to run GLM-5.3-Flash (≥93 GB) without disk paging on 64 GB + 24 GB.
   - Dynamic REAP's 256-token swap cadence is cheap even over NVMe. Avoid running from the USB disk.
8. **[PRED+XFER] Next-layer expert prediction and prefetch (Tier 2, lossless).**
   - Gate lookahead or residual-based. Useful mainly to pre-warm the GPU cache; on PCIe 3.0 it can hide only a few experts per layer.
9. **[ROUTE] Top-k reduction / expert skipping (Tier 3, lossy).**
   - `-ser`-style, or overriding expert_used_count. Linear traffic savings, quality-gated.
10. **[SPEC] Speculative decoding (Tier 3).**
    - MTP or self-draft: helps only if the verify batch hits mostly resident experts. On offloaded MoE, the verify pass touches the union of experts across draft tokens, so gains are smaller than on dense models (DraftExpert).
11. **[LOAD] mmap/NVMe/DirectStorage (Tier 3).**
    - Cold start and dynamic-REAP swaps only. GDS is likely unavailable on GeForce/Windows.

Other notes:
- Quick upper-bound check for the agent: tok/s ≤ 29 GB/s ÷ (CPU-resident active expert bytes per token). For gpt-oss-120b, that is ~15 tok/s with all experts on the CPU and ~22 tok/s with ~30% resident. Hybrid caching plus overlap is what can push past that, by making more of the per-token traffic GPU-local.
- These systems have seen on-par setups go both ways (slowdowns on ROCm/Vulkan, V100 collapse below working set), so every technique should be accepted only after an A/B on the actual box with a fixed prompt set and KLD/perplexity checks.

### Gaps
- No measurements on PCIe 3.0 + DDR4 + AVX2 for any of the target models were found, so the tier ordering rests on bandwidth arithmetic plus PCIe 4.0 results.
- Router-locality statistics (cache hit rate vs cache size) for gpt-oss-120b, Qwen3.6-35B-A3B, Gemma 4 26B-A4B and GLM-5.3-Flash were not found. These determine how much tiers 3 and 8 can deliver.
- Quality impact of REAP or dynamic REAP on gpt-oss-120b and Gemma 4 is unknown; the Dettmers paper reports GLM-5.3-Flash as hard to compress.
