# Model compression for serving, system/hardware techniques, and an RTX 4090 + i7-10700K + WSL2 hardware reference

Scope note: written 2026-10-23. Each technique carries a date: the arXiv or first-publication month. Sources come in two groups. Some were fetched or searched in this session and are marked **[verified]**. The others are canonical paper links from prior knowledge that I did not re-fetch this session, marked **[canonical, not re-fetched]**. For those, the URL and the gist should be right, but check exact numbers before quoting them. Slot names are suggestions for the user's plugin engine: `weights-format`, `expert-selection/router`, `ffn-kernel`, `attention`, `lm-head/sampler`, `layer-schedule`, `offload-scheduler`, `allocator`, `loader`, `cpu-kernel`, `runtime/launch`, `os-config`.

---

## (a) Model-compression techniques usable at serving time

### Takeaway
For this box (CPU-offloaded MoE, bound by memory and PCIe bandwidth), these compression levers have the most leverage. All of them reduce the bytes moved per token, which is what limits decode here, rather than the FLOPs computed:
- **MoE expert pruning or merging** (REAP, REAM, NAEE, HC-SMoE, Sub-MoE). This shrinks the CPU-resident expert pool, so more experts fit in VRAM or pinned RAM.
- **Dynamic expert skipping or top-k reduction.** This cuts the expert bytes read per token.
- **Activation sparsity on the CPU-side expert GEMV** (TEAL, CATS, R-Sparse). Skipping weight columns that would multiply near-zero activations reduces DRAM reads.
- **Layer dropping.**

2:4 sparse tensor cores exist on Ada and give at most 2x on matmul throughput, but the end-to-end gains measured elsewhere are modest (about 1.1–1.3x). They mainly help compute-bound GPU prefill, not offloaded decode. Distillation and low-rank factorization (SVD) are offline, quality-costly options; treat them as "produce a different model" rather than as runtime slots.

### Cited Findings

#### A1. MoE expert pruning and merging (slot: `weights-format` offline, plus `expert-selection/router`)
- **REAP (Router-weighted Expert Activation Pruning)**, arXiv 2510.13999, Oct 2025, ICLR 2026. It is one-shot expert pruning that scores each expert by router gate values times expert activation norms, aiming to minimize a reconstruction-error bound. The authors argue that merging loses the router's fine-grained control and so carries "irreducible error", which is why pruning beats merging on generative tasks. Tested on sparse MoEs from 20B to 1T parameters, it was "near-lossless" on code generation for Qwen3-Coder-480B and Kimi-K2 at 50% expert pruning. The official code is at github.com/CerebrasResearch/reap; March 2026 updates added layer-wise calibration so large models can be pruned on a single GPU, plus a fix that renormalizes top-k router logits. **[verified]** — [arXiv](https://arxiv.org/html/2510.13999v2); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10006871); [Repo](https://github.com/CerebrasResearch/reap)
- **REAM (Router-weighted Expert Activation Merging)**, arXiv 2604.04356, Apr 2026. It groups experts and merges their weights instead of deleting them. It reports a trade-off between multiple-choice and generative quality that depends on the calibration mix, and was evaluated at 25% and 50% expert reduction on Qwen3 and GLM-4.5. **[verified]** — [alphaxiv](https://www.alphaxiv.org/abs/2604.04356v1); [arXiv HTML](https://arxiv.org/html/2604.04356v1)
- **Sub-MoE (subspace expert merging)**, AAAI 2026. It reports beating prior pruning and merging methods on Mixtral, DeepSeek and Qwen MoEs, keeping about 86% of original zero-shot performance on Mixtral-8x7B at 50% expert reduction. **[verified]** — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/39464)
- **NAEE ("Not All Experts are Equal")**, arXiv 2402.14800, Feb 2024. It does task-specific expert pruning with an enumeration search, plus **dynamic expert skipping** at inference: when the second expert's gate weight is small relative to the top-1, that expert is skipped. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2402.14800)
- **MC-SMoE** (merge, then compress), arXiv 2310.01334, Oct 2023, ICLR 2024. It groups experts by routing statistics, merges each group, then low-rank-compresses the result. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2310.01334)
- **HC-SMoE** (retraining-free hierarchical clustering of experts by output similarity), arXiv 2410.08589, Oct 2024. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2410.08589)
- Earlier merging work showed merging ahead of pruning on perplexity and multiple-choice benchmarks. REAP disputes that this carries over to generative tasks. **[verified, as summarized in REAP related work]** — [arXiv](https://arxiv.org/html/2510.13999v2)

#### A2. Activation sparsity (slot: `ffn-kernel` / `cpu-kernel`, plus a calibration step)
- **TEAL**, arXiv 2408.14690, Aug 2024, ICLR 2025. Training-free magnitude thresholding applied to hidden states throughout the model. It reaches 40–50% model-wide sparsity with little degradation on Llama-2, Llama-3 and Mistral (7B–70B). Batch-1 decode speedups were 1.53x at 40% and 1.8x at 50%, using custom sparse GEMV kernels in GPT-Fast with CUDA graphs and torch.compile on A100 and A6000. Gains were smaller on Llama-3-8B, partly because its large LM head is not sparsified. The method is compatible with weight quantization. **[verified]** — [arXiv](https://arxiv.org/pdf/2408.14690); [Repo](https://github.com/FasterDecoding/TEAL)
- **Deja Vu**, arXiv 2310.17157, Oct 2023, ICML 2023. A small learned predictor picks the attention heads and MLP neurons to compute for each token ("contextual sparsity"). **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2310.17157)
- **CATS**, arXiv 2404.08763, Apr 2024. Thresholds the SiLU gate output in gated MLPs, about 50% sparsity, training-free, with a custom kernel. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2404.08763)
- **ReLU-fication / "ReLU Strikes Back"**, arXiv 2310.04564, Oct 2023. Swaps the activation to ReLU and fine-tunes, recovering high activation sparsity. Follow-ups: **ProSparse** ([arXiv 2402.13516](https://arxiv.org/abs/2402.13516), Feb 2024) and **Turbo Sparse** ([arXiv 2406.05955](https://arxiv.org/abs/2406.05955), Jun 2024, dReLU, applied to Mixtral too). These require continued training. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2310.04564)
- **Q-Sparse**, arXiv 2407.10969, Jul 2024. Top-K activation sparsification trained end to end; the authors present it as complementary to low-bit weights. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2407.10969)
- **R-Sparse**, arXiv 2504.19449, Apr 2025, ICLR 2025. Rank-aware, training-free activation sparsity that applies to all linear layers, not just the MLP. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2504.19449)
- **PowerInfer** (hot/cold neuron split: hot neurons on GPU, cold neurons computed on CPU, driven by activation sparsity), arXiv 2312.12456, Dec 2023. It targets exactly the consumer-GPU-plus-CPU setting (it was benchmarked on an RTX 4090). A related paper, **LLM in a flash** ([arXiv 2312.11514](https://arxiv.org/abs/2312.11514), Dec 2023), uses sparsity prediction to load only the needed rows from flash storage. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2312.12456)

#### A3. Unstructured and semi-structured (N:M) weight sparsity (slot: `weights-format` + `ffn-kernel`)
- **SparseGPT**, arXiv 2301.00774, Jan 2023. One-shot pruning to 50–60% unstructured sparsity, or to 2:4/4:8, using an OBS-style weight update. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2301.00774)
- **Wanda**, arXiv 2306.11695, Jun 2023. Prunes by weight magnitude times input-activation norm, with no weight update; supports 2:4. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2306.11695)
- **MaskLLM**, arXiv 2409.17481, Sep 2024, NeurIPS 2024. Learns 2:4 masks end to end; better quality than one-shot 2:4 but needs training compute. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2409.17481)
- **Sparse-Marlin** (2:4 sparse plus INT4 kernel; covered in the MARLIN paper), arXiv 2408.11743, Aug 2024. Mixed-precision GEMM with 2:4 support for Ampere and Ada, from the IST Austria/Neural Magic line used in vLLM. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2408.11743)
- **Flash-LLM**, arXiv 2309.10285, Sep 2023. Unstructured-sparse matmul on tensor cores using a "load as sparse, compute as dense" approach. Its benefit is reduced memory traffic for skinny (decode) GEMMs. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2309.10285)
- **PyTorch semi-structured (2:4) sparsity** (`to_sparse_semi_structured`, with cuSPARSELt or CUTLASS backends) requires PyTorch 2.1+ and compute capability 8.0+, so sm_89 qualifies. The tutorial reports 1.3x on BERT inference (F1 86.92 to 86.48 after sparse fine-tuning). PyTorch also reports a 10% end-to-end gain on Segment Anything and 1.3x on ViT-L MLP training, both on an A100. The theoretical ceiling is 2x. **[verified]** — [PyTorch tutorial](https://pytorch.org/tutorials/prototype/semi_structured_sparse.html); [PyTorch blog](https://pytorch.org/blog/accelerating-neural-network-training/)
- The Ada whitepaper lists 4090 sparse tensor rates at exactly twice the dense rates for FP8, FP16, BF16, TF32, INT8 and INT4. For example, FP8 with FP16 accumulate is 660.6 TFLOPS dense and 1321.2 sparse. **[verified]** — [NVIDIA Ada whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- **SlideSparse** (arXiv 2603.05232, Mar 2026) evaluates sparse tensor cores on A100, H100, B200, RTX 4090, RTX 5080 and DGX Spark. It notes that sparse tensor cores support only the 2:4 format, and its headline result is 1.33x end-to-end on A100 INT8. I did not obtain its RTX 4090 figures. **[verified, snippet-level]** — [arXiv](https://arxiv.org/html/2603.05232)

#### A4. Structured pruning, layer dropping and width reduction (slot: `weights-format` offline; layer dropping can be a runtime `layer-schedule` slot)
- **LLM-Pruner**, arXiv 2305.11627, May 2023. Removes coupled structures (heads and channels) by gradient importance, then recovers with LoRA. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2305.11627)
- **SliceGPT**, arXiv 2401.15024, Jan 2024, ICLR 2024. Applies orthogonal rotations and then deletes rows and columns, shrinking the embedding dimension by up to about 25–30%. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2401.15024)
- **FLAP**, arXiv 2312.11983, Dec 2023. Retraining-free structured pruning based on fluctuation, with bias compensation. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2312.11983)
- **ShortGPT** (drop layers with low "Block Influence"), arXiv 2403.03853, Mar 2024. **LaCo** (layer collapse; [arXiv 2402.11187](https://arxiv.org/abs/2402.11187), Feb 2024). **"The Unreasonable Ineffectiveness of the Deeper Layers"** ([arXiv 2403.17887](https://arxiv.org/abs/2403.17887), Mar 2024), which finds that a large fraction of deep layers can be dropped, with a small QLoRA "heal". **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2403.03853)
- **Sheared LLaMA** (targeted structured pruning plus continued pretraining), arXiv 2310.06694, Oct 2023. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2310.06694)
- **Minitron** (prune depth, width, heads and MLP, then distill), arXiv 2407.14679, Jul 2024; the follow-up "Minitron approach in practice" is [arXiv 2408.11796](https://arxiv.org/abs/2408.11796), Aug 2024. NVIDIA's pipeline for producing smaller dense models. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2407.14679)

#### A5. Dynamic depth and early exit (slot: `layer-schedule`, together with speculative decoding)
- **CALM** (Confident Adaptive Language Modeling: per-token early exit with calibrated confidence), arXiv 2207.07061, Jul 2022, NeurIPS 2022. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2207.07061)
- **LayerSkip** (train with layer dropout and an early-exit loss, then self-speculative decoding: early layers draft, the full model verifies), arXiv 2404.16710, Apr 2024, ACL 2024. Requires a model trained or fine-tuned with that recipe. **[canonical, not re-fetched]** — [arXiv](https://arxiv.org/abs/2404.16710)

#### A6. Low-rank factorization (slot: `weights-format` offline)
- **ASVD** (activation-aware SVD), arXiv 2312.05821, Dec 2023. **SVD-LLM** (truncation-aware whitening plus a layer-wise update), arXiv 2403.07378, Mar 2024. Both are training-free. At moderate compression ratios they lose more quality than 4-bit quantization at the same size, so they are mostly useful in combination with quantization or on the KV projections. **[canonical, not re-fetched]** — [ASVD](https://arxiv.org/abs/2312.05821); [SVD-LLM](https://arxiv.org/abs/2403.07378)

#### A7. Distillation into smaller or denser models (offline, so it produces a different model rather than a slot)
- Minitron (A4) is the main recipe for distilling a pruned model. **MiniLLM** ([arXiv 2306.08543](https://arxiv.org/abs/2306.08543), Jun 2023) uses reverse-KL distillation for generative LMs. **[canonical, not re-fetched]**

#### A8. Lossless weight compression (slot: `loader` / `weights-format`)
- **DFloat11**, arXiv 2504.11651, Apr 2025. Entropy-codes the BF16 exponent bits, bringing models to about 70% of BF16 size with bit-exact output, and decompresses on the GPU with a custom kernel. **ZipNN**, arXiv 2411.05239, Nov 2024. Lossless Huffman-style compression of model files, aimed at storage and transfer. **[canonical, not re-fetched]** — [DFloat11](https://arxiv.org/abs/2504.11651); [ZipNN](https://arxiv.org/abs/2411.05239)

### Inferences
- **Which lever matters most on this box.** The CPU-side expert pool is the bottleneck: decode reads the active experts from DDR4 at roughly 29–45 GB/s and cannot hold all of them in 24 GB of VRAM. Expert pruning at 25–50% (REAP for generative quality, REAM if multiple-choice quality matters) has a direct structural payoff. A 50% pruned gpt-oss-120b-class model might fit most hot experts in VRAM plus pinned RAM, which would change the offload regime entirely. Quality risk depends on the task, and REAP's claims are strongest for code. Re-calibrate on the user's own workload mix.
- **Dynamic top-k reduction or expert skipping** (NAEE-style, or thresholding the gate weight of the k-th expert) is a cheap runtime `expert-selection` slot. It needs no retraining, the agent can tune it with a gate threshold, and it cuts CPU expert bytes per token roughly in proportion to the experts skipped. Pair it with a quality eval gate.
- **Activation sparsity on CPU expert GEMVs** (TEAL or CATS style: skip weight columns whose input is near zero) needs a weight layout that makes skipped columns contiguous, i.e. column-major or blocked storage per input channel. With 4-bit block-quantized weights, skipping works only at quantization-block granularity, which limits the gain. Expect well under the 1.5–1.8x seen on GPUs in fp16.
- **2:4 sparsity matters little for offloaded MoE decode.** Decode is memory-bound, and 2:4 saves only 50% of values at the cost of about 12.5–25% metadata. It also stacks awkwardly with 4-bit block quantization (Sparse-Marlin is the exception). It is more relevant to GPU-side prefill of attention and shared experts.
- **lm_head / vocabulary.** Large-vocabulary models make the LM head a large fixed per-token read. As I recall from the model config (not verified here), gpt-oss uses vocab ≈201k and hidden 2880, so the head is about 0.58B params, roughly 1.16 GB in BF16. Options: quantize the head to INT8 or FP8 (halves the read); trim the vocabulary to the languages and tokens actually used, remapping IDs (lossy for out-of-vocabulary text); compute logits only for a candidate subset (approximate). The TEAL paper explicitly notes that the large Llama-3 LM head limited its gains.
- **Lossless compression is mostly a disk-loading lever** on a 240 MB/s USB disk. Entropy coding helps BF16 (exponent redundancy) but likely yields little on already-quantized MXFP4 or Q4 data, which is near maximum entropy; benchmark before building it.

### Gaps
- I found no measured RTX 4090 end-to-end numbers for 2:4 sparse LLM inference; the SlideSparse 4090 figures were not obtained.
- I found no source on vocabulary or lm_head pruning methods specific to 2025–2026; the technique is described above from first principles only.
- I did not verify configurations (expert counts, vocab, hidden size) for Qwen3.6-35B-A3B, Gemma 4 26B-A4B or GLM-5.3-Flash; these are names from the user context and I could not confirm them here.
- NAEE, MC-SMoE and HC-SMoE did not appear in this session's searches; their links come from prior knowledge.
- I did not check whether pruned (REAP) checkpoints of gpt-oss-120b or the other target models exist publicly as of Oct 2026.

---

## (b) System and hardware-level techniques (allocators, pinned/unified memory, I/O and loading, OS and CPU tuning, GPU power, WSL2)

### Takeaway
On this box, **WSL2 is the biggest system-level constraint**:
- The WSL2 VM gets only **50% of host RAM by default** (about 32 GB of 64 GB), which is too small to hold gpt-oss-120b's experts.
- **Pinned (page-locked) memory is capped by Windows**, typically around 50% of system RAM, and NVIDIA says it is lower under WSL2.
- **`cudaHostRegister` has been reported not to work on WSL2**, while `cudaHostAlloc` does.
- **Full managed (unified) memory is unavailable**, so oversubscription and prefetching through `cudaMallocManaged` are not a viable slot.
- **Persistence and compute modes cannot be set** through nvidia-smi inside WSL.
- **Files on /mnt/c (9P/DrvFs) load several times slower** than files on the VM's ext4.
- The default `autoMemoryReclaim=dropCache` may evict the page cache holding mmapped models.

Fixes, in priority order:
1. Raise `memory=` in `.wslconfig`.
2. Keep models on ext4 inside the VM, or on NVMe.
3. Use explicitly allocated pinned staging buffers (`cudaHostAlloc`) in a ring, rather than pinning the whole model.
4. Use CUDA graphs to amortize the higher launch overhead on WDDM.

### Cited Findings

#### B1. WSL2-specific facts
- `.wslconfig` defaults: `memory` = "50% of total memory on Windows"; `processors` = same as the number of Windows logical processors; `swap` = "25% of memory size on Windows rounded up to the nearest GB"; experimental `autoMemoryReclaim` default `dropCache` ("cached memory will be reclaimed immediately"; other options `gradual` and `disabled`); `sparseVhd` default false; `virtiofs` is an experimental alternative for Windows filesystem shares. The page was updated 2026-10-16. Note that some settings (`networkingMode`, `debugConsole` and others) are Windows 11 only; the user runs Windows 10 19045. **[verified]** — [Microsoft Learn: WSL config](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
- WSL `[gpu]` uses "para-virtualization" to reach the Windows GPU, and GPU libraries live at `/usr/lib/wsl/lib`. **[verified]** — [Microsoft Learn: WSL config](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
- NVIDIA CUDA on WSL known limitations:
  - "Full Managed Memory Support is not available on Windows native", so it is not available on WSL2 either. Managed-memory apps may see "reduced performance and high system memory usage", and concurrent CPU/GPU access is unsupported.
  - Pinned system memory available to applications is limited.
  - NVML/nvidia-smi: GPU utilization and active-process queries are unsupported, and modifiable state ("ECC, compute mode, persistence mode") won't be supported.
  - Profilers are listed as not available yet or in preview.
  - Only WDDM mode is supported; TCC is not.

  **[verified]** — [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- Pinned memory limit: NVIDIA forum staff say the cap is set by Windows (typically ~50% of system memory, "not always 50%"), that WSL2 limits are likely lower because of the extra WSL2-to-WDDM machinery, and that requests to raise it are unlikely to be actionable. A PyTorch user hit pin_memory out-of-memory errors while using well under half of 64 GB in WSL. **[verified]** — [NVIDIA forum: 50% cudaHostAlloc limit](https://forums.developer.nvidia.com/t/change-limit-of-50-for-cudahostalloc-pinned-memory-on-windows-10-11/228235/8)
- `cudaHostRegister` was reported failing on WSL2 while `cudaHostAlloc` works (forum thread, Jan 2024). It may have changed in newer drivers. **[verified]** — [NVIDIA forum](https://forums.developer.nvidia.com/t/cudahostregister-not-supported-on-wsl2/279429)
- vLLM disables pinned memory by default on WSL2 and lets you opt in with `VLLM_WSL2_ENABLE_PIN_MEMORY=1`; on kernels older than 4.19.121 it is disabled entirely. **[verified]** — [vLLM CUDA platform docs](https://docs.vllm.ai/en/latest/api/vllm/platforms/cuda/)
- NVIDIA blog and forum post on improving CUDA performance on WSL2 (launch latency and other enhancements). **[verified link exists; content not fetched]** — [NVIDIA forum: Leveling up CUDA performance on WSL2](https://forums.developer.nvidia.com/t/leveling-up-cuda-performance-on-wsl2-with-new-enhancements/186219)
- File location: guides report that /mnt/c (9P) is 3–5x slower than native ext4 for model loading and recommend keeping models on the distro's ext4 filesystem. One anecdotal post says mmap of huge files over 9P fails or is slow; treat that as unconfirmed. **[verified, secondary sources]** — [mljourney WSL2 Ollama guide](https://mljourney.com/how-to-run-ollama-on-wsl2-windows-subsystem-for-linux/); [bestllmfor WSL2 vs native](https://bestllmfor.com/guide/ollama-wsl2-vs-windows-natif/)

#### B2. Model loading: mmap vs read vs direct I/O, file formats, disk compression
- GGUF was designed so weights can be mmapped directly without read() or copy, which is why llama.cpp defaults to mmap (llama.cpp issue #91 discussion). **[verified, snippet-level]** — [llama.cpp issue #91](https://github.com/ggerganov/llama.cpp/issues/91)
- Counter-evidence: on DGX Spark (native Linux), loading gpt-oss-120b in llama.cpp took 1 min 44 s with mmap versus 56 s with `--no-mmap`, so mmap can be the bottleneck even without 9P. **[verified]** — [NVIDIA forum: slow mmap on DGX Spark](https://forums.developer.nvidia.com/t/very-slow-mmap-on-dgx-spark-that-affects-model-loading-questions-to-nvidia/349886.md)
- GPUDirect Storage (GDS): officially tested GPUs in early release notes were data-center and Quadro cards (T4, A10, A100, V100, Quadro P6000), with no GeForce. A forum user reports an RTX 3060 works only in **compatibility mode**, which falls back to a CPU bounce buffer through cuFile without nvidia-fs.ko. Since CUDA 12.2 (GDS 1.7), non-O_DIRECT file descriptors are supported; before that only O_DIRECT was. **[verified]** — [GDS O_DIRECT guide](https://docs.nvidia.com/gpudirect-storage/o-direct-guide/index.html); [GDS forum: supported storage](https://forums.developer.nvidia.com/t/gpudirect-storage-supported-storage/200193); [GDS release notes](https://docs.nvidia.com/gpudirect-storage/pdf/release-notes.pdf)

#### B3. GPU memory allocators (slot: `allocator`)
- PyTorch `expandable_segments` (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) reserves a contiguous virtual range and maps physical pages on demand through CUDA VMM. It reduces fragmentation from block splitting and merging, but peak memory still depends on allocation order and page mapping has runtime overhead. Introduced in PR #96995 (2023). **[verified]** — [PyTorch devlog Jun 2026: CUDA caching allocator fragmentation](https://docs.pytorch.org/devlogs/eager/2026-06-01-cuda-caching-allocator/); [Xema arXiv 2607.11136](https://arxiv.org/pdf/2607.11136)
- CUDA low-level virtual memory management (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`) enables growable buffers without copying. **vAttention** ([arXiv 2405.04437](https://arxiv.org/abs/2405.04437), May 2024) uses it for a contiguous-virtual, paged-physical KV cache as an alternative to PagedAttention ([arXiv 2309.06180](https://arxiv.org/abs/2309.06180)). **[canonical, not re-fetched]** — [NVIDIA blog: Low-level GPU VMM](https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/)

#### B4. CPU/GPU scheduling, synchronization and launch (slot: `runtime/launch`)
- CUDA device flags `cudaDeviceScheduleSpin`, `cudaDeviceScheduleYield`, `cudaDeviceScheduleBlockingSync` and `cudaDeviceScheduleAuto` choose whether the host thread spins (lowest latency, burns a core) or sleeps while waiting on the GPU. On an 8-core CPU that also runs expert GEMVs, a spinning sync thread steals a core. **[canonical, not re-fetched]** — [CUDA Runtime API: device management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)

#### B5. MoE offload systems relevant to the CPU-vs-transfer decision (slot: `offload-scheduler`)
- **Fiddler** (arXiv 2402.07033, Feb 2024) computes offloaded experts on the CPU instead of transferring their weights to the GPU, on the argument that moving activations is far cheaper than moving weights over PCIe. **Mixtral offloading** (Eliseev & Mazur, [arXiv 2312.17238](https://arxiv.org/abs/2312.17238), Dec 2023) uses an LRU expert cache in VRAM plus speculative expert prefetch. **[canonical, not re-fetched]** — [Fiddler](https://arxiv.org/abs/2402.07033)

#### B6. Linux memory and OS knobs
- Transparent Huge Pages modes (`always`, `madvise`, `never`) and `madvise(MADV_HUGEPAGE)`: 2 MB pages cut TLB misses when streaming tens of GB of expert weights. **[canonical, not re-fetched]** — [Linux kernel docs: THP](https://docs.kernel.org/admin-guide/mm/transhuge.html)

#### B7. PCIe transfers
- PCIe Gen3 x16 gives about 11–12 GB/s usable with pinned memory per direction for transfers of 8–16 MB or more. One measured pinned host-to-device result was 11,709.8 MB/s at 32 MB. Pageable transfers are slower because they add a staging copy, and small transfers are much slower because of per-packet and launch overhead. **[verified]** — [NVIDIA forum: optimize H2D](https://forums.developer.nvidia.com/t/optimize-data-transfer-rate-from-host-to-device/51998); [NVIDIA forum: Quadro RTX 8000 bandwidthTest](https://forums.developer.nvidia.com/t/nvidia-quadro-rtx-8000-bandwidthtest-theoretical-max-results/186064)

### Inferences
Each item names the slot or config surface it affects, and how confident I am.

1. **`os-config`: WSL VM memory (high priority).** Set `memory=` in `.wslconfig` to about 56–58 GB, leaving Windows and the driver roughly 6–8 GB. Consider `swap=0`, or a swap file on a fast disk, so the VM does not swap expert weights to the USB disk or C:. With the default 32 GB, a roughly 60 GB+ MoE cannot stay in RAM and will page.
2. **`os-config`: page cache (medium confidence).** Set `autoMemoryReclaim=disabled`, and/or load experts into anonymous memory (`--no-mmap` style read into a malloc or pinned buffer, or mmap plus `mlock`/`MAP_POPULATE`). The docs do not say what triggers `dropCache`, so test whether mmapped weights get re-read from the 240 MB/s disk after idle periods.
3. **`allocator` / `offload-scheduler`: pinned memory budget.** Assume the pinned budget is under 32 GB and probably well below. Design for a fixed set of `cudaHostAlloc` staging buffers (a double or triple ring of a few hundred MB to a few GB) fed by CPU memcpy from pageable expert storage. Do not plan to register the whole 60 GB model as pinned memory, given the cap and the `cudaHostRegister` report. The agent should probe the actual cap at startup (binary-search `cudaHostAlloc` sizes) and record it as a hardware fact.
4. **`allocator`: unified memory.** Do not use `cudaMallocManaged` oversubscription or prefetch on WSL2; NVIDIA documents it as unsupported or slow there. A native Linux boot would restore it, though it is rarely faster than explicit streams for MoE anyway.
5. **`loader`: file placement.**
   - Put models on WSL ext4 (inside ext4.vhdx), not /mnt/*. Better still, attach an NVMe disk to the VM directly with `wsl --mount`, which gives a native Linux filesystem.
   - With the USB disk at 240 MB/s, a cold load of about 61–65 GB takes about 4.5 minutes; the 9P path would likely make it slower.
   - Parallel loading cannot beat the disk's sequential bandwidth, so the main load-time levers are faster storage, keeping weights resident in page cache or a RAM disk, and lossless compression where it actually compresses.
   - GDS is not a real option on a GeForce card under WSL2: compatibility mode at best, which is just a CPU bounce.
6. **`runtime/launch`: launch overhead.** CUDA graphs plus fewer, larger kernels matter more on WSL2/WDDM, where submission goes through paravirtualization, than on native Linux. Measure per-launch latency at startup.
7. **`cpu-kernel` / `os-config`: threads and affinity.**
   - Use about 8 worker threads, one per physical core, for memory-bound expert GEMV. SMT usually adds little once DRAM is saturated and can hurt cache behaviour. Reserve one hardware thread for the GPU-feeding or orchestration thread, and use blocking sync rather than spin if that core is needed for compute.
   - Under WSL2, `sched_setaffinity` pins to vCPUs, and Hyper-V decides the vCPU-to-physical-core mapping, so measure physical-core/SMT topology empirically rather than trusting `lscpu`. Low confidence; I did not verify this.
8. **`os-config`: huge pages.** Allocate expert weight arenas with `madvise(MADV_HUGEPAGE)` or 2 MB-aligned allocations. I did not verify the WSL kernel's THP default; check `/sys/kernel/mm/transparent_hugepage/enabled`.
9. **GPU power and clocks.**
   - Persistence mode cannot be set from WSL, so it is a non-issue (the Windows driver keeps the GPU initialized).
   - Power-limit and clock-lock changes (`nvidia-smi -pl` / `-lgc`) have to be made from the Windows host with admin rights.
   - Decode is memory-bound, so a lower power limit likely costs little decode throughput, while VRAM clock matters more. Prefill is tensor-bound and sensitive to power and clocks.
   - These are inferences from roofline reasoning; I found no source with numbers for this box.
10. **Resizable BAR.** The 4090 supports ReBAR, and Z490 boards generally gained it via BIOS update. I did not verify this in this session. ReBAR lets the CPU map all of VRAM for direct writes; under WSL2 it is unclear whether CUDA exposes or uses it. It is likely irrelevant to cudaMemcpy throughput, which is DMA-engine-driven. Treat it as low priority.
11. **NUMA.** Not applicable on a single-socket 10700K, but a NUMA-aware slot interface should still exist for portability.
12. **Process priority.** Raising nice priority, or the Windows priority of `vmmem`, may reduce jitter from host background tasks. This is speculative and unmeasured.

### Gaps
- No primary measurement of WSL2 vs native Linux CUDA kernel-launch latency or H2D bandwidth for 2025–2026 drivers was found.
- I could not confirm whether `cudaHostRegister` works on current (2026) WSL2 drivers; the report is from Jan 2024.
- The actual pinned-memory cap on a 64 GB Windows 10 host under WSL2 is undocumented; it must be measured.
- What exactly triggers `autoMemoryReclaim=dropCache`, and whether it hits mmapped model pages, is undocumented.
- The WSL2 kernel's default THP mode was not verified.
- I found no source on hardware-accelerated GPU scheduling (HAGS) effects for CUDA under WSL2.
- Windows-native vs WSL2 vs native Linux throughput comparisons for llama.cpp or ik_llama MoE offload were not found in this session.
- No current (2025–2026) GDS support matrix listing GeForce was retrieved.

---

## (c) Hardware-capability reference and rooflines for this box (RTX 4090 / PCIe 3.0 x16 / i7-10700K / DDR4-3600 dual channel)

### Takeaway
The RTX 4090 is very strong in compute (165 TFLOPS BF16 with FP32 accumulate, 330 FP8 with FP32 accumulate, 660 INT8 TOPS dense) and has 1 TB/s of VRAM bandwidth, but this box feeds it through **PCIe 3.0 x16 at about 12 GB/s**. That is roughly 1/80 of VRAM bandwidth and below the CPU's own DRAM bandwidth (about 29–45 GB/s achievable).

Two consequences for batch-1 decode with CPU-resident experts:
- Computing experts **on the CPU** (memory-bound, roughly bytes of active experts ÷ ~30–40 GB/s) beats shipping them to the GPU (÷ ~12 GB/s), by roughly 3x.
- Uploading experts to the GPU only pays off when each expert serves enough tokens: about **≥10–20 tokens per expert per upload**, as in prefill or large batches.

On GeForce Ada, **INT8 tensor throughput is 4x BF16-with-FP32-accumulate**, and FP16 with FP32 accumulate runs at half rate. That makes W8A8/FP8 GPU kernels unusually attractive for prefill.

### Cited Findings

#### C1. RTX 4090 (AD102, sm_89) — NVIDIA Ada whitepaper, Appendix A **[verified]**

| Spec | Value |
| --- | --- |
| SMs | 128 (16,384 CUDA cores, 128 per SM) |
| Tensor cores | 512 (4th gen, 4 per SM) |
| Boost clock | 2520 MHz |
| Peak FP32 (non-tensor) | 82.6 TFLOPS |
| Peak FP16 / BF16 (non-tensor) | 82.6 TFLOPS |
| Peak INT32 (non-tensor) | 41.3 TOPS |
| FP8 tensor, FP16 accumulate | 660.6 TFLOPS dense / 1321.2 sparse |
| FP8 tensor, FP32 accumulate | 330.3 dense / 660.6 sparse (corrected in whitepaper v2.02) |
| FP16 tensor, FP16 accumulate | 330.3 dense / 660.6 sparse |
| FP16 tensor, FP32 accumulate | 165.2 dense / 330.4 sparse |
| BF16 tensor, FP32 accumulate | 165.2 dense / 330.4 sparse |
| TF32 tensor | 82.6 dense / 165.2 sparse |
| INT8 tensor | 660.6 dense / 1321.2 sparse TOPS |
| INT4 tensor | 1321.2 dense / 2642.4 sparse TOPS |
| Memory | 24 GB GDDR6X, 384-bit, 21 Gbps, **1008 GB/s** |
| L2 | **73,728 KB (72 MB)** enabled (full AD102 has 96 MB) |
| L1 / shared memory | 16,384 KB total = 128 KB per SM (unified, configurable) |
| Register file | 32,768 KB total |
| TGP | 450 W |
| PCIe | Gen 4 (on this Z490 platform it runs at Gen 3) |

Sparse figures use the 2:4 sparsity feature. Source: [NVIDIA Ada GPU Architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)

- The RTX 4090 has 72 MB of L2 enabled out of AD102's 96 MB. Chips and Cheese has microbenchmarks of its cache and memory hierarchy. **[verified]** — [Chips and Cheese: Microbenchmarking RTX 4090](https://chipsandcheese.com/p/microbenchmarking-nvidias-rtx-4090); [Wikipedia: Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture))
- Shared memory per SM: the CUDA Programming Guide's compute-capability table lists 100 KB maximum shared memory per SM for the 8.6/8.9-class parts (164 KB for 8.0). Since CC 8.0, 1 KB per SM is reserved by the system, so the per-block opt-in maximum is 99 KB (101,376 bytes, as reported on CC 8.6). Above 48 KB you need dynamic shared memory plus `cudaFuncSetAttribute`. **[verified, partial: the table snippet was garbled, so 8.9 = 100 KB is inferred from the 8.x columns and should be confirmed with `cudaDevAttrMaxSharedMemoryPerMultiprocessor`]** — [NVIDIA forum: max shared memory](https://forums.developer.nvidia.com/t/question-about-max-shared-memory-in-block-and-multiprocessor/283345)

#### C2. What Ada (sm_89) lacks vs Hopper (sm_90) and Blackwell (sm_100/sm_120)
- Hopper introduced the Tensor Memory Accelerator (TMA), asynchronous warpgroup MMA (`wgmma`), thread block clusters and distributed shared memory, and the Transformer Engine. Ada has none of TMA, wgmma or clusters/DSMEM. It uses Ampere-style `mma.sync` and `cp.async` (adding FP8 E4M3/E5M2 to `mma`) plus 2:4 sparse tensor cores. **[canonical, not re-fetched]** — [NVIDIA blog: Hopper architecture in depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- Native FP4/FP6 tensor formats (NVFP4/MXFP4 in hardware) arrived with Blackwell. Ada has FP8 but no FP4 tensor path, so MXFP4 weights (as in gpt-oss) must be dequantized to FP8/FP16/INT8 in-register before `mma`. **[canonical, not re-fetched]** — [NVIDIA Blackwell architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- The Ada whitepaper says the 4th-gen tensor core adds FP8 and that FP8 "halves the data storage requirements and doubles throughput" compared with FP16. **[verified]** — [Ada whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)

#### C3. PCIe 3.0 x16
- Practical rate is about 11–12 GB/s per direction with pinned buffers of 8–16 MB or more; pageable transfers are lower. **[verified]** — [NVIDIA forum](https://forums.developer.nvidia.com/t/optimize-data-transfer-rate-from-host-to-device/51998); [NVIDIA forum: bandwidthTest](https://forums.developer.nvidia.com/t/nvidia-quadro-rtx-8000-bandwidthtest-theoretical-max-results/186064)

#### C4. Intel Core i7-10700K and DDR4
- Intel ark: 8 cores / 16 threads, up to 5.1 GHz, 16 MB cache, **official memory DDR4-2933, 2 channels, Max Memory Bandwidth 45.8 GB/s**, status Discontinued. DDR4-3600 is an XMP overclock beyond the official spec. **[verified]** — [Intel ark i7-10700K](https://www.intel.com/content/www/us/en/products/sku/199335.html)

### Inferences
These are rooflines computed from the cited numbers; the assumptions are stated in each item.

1. **DDR4-3600 dual-channel theoretical bandwidth** is 3600 MT/s × 8 B × 2 channels = **57.6 GB/s**. The user's measured "copy" figure of about 29 GB/s counts each byte read once and written once, so pure-read streaming (what GEMV does) may reach about 35–45 GB/s on this platform. The agent should measure a read-only bandwidth kernel; a STREAM-style "triad/read" figure is the right denominator for CPU expert rooflines.
2. **PCIe 3.0 theoretical bandwidth** is 8 GT/s × 128/130 × 16 lanes ≈ **15.75 GB/s per direction**; practical is about 12 GB/s. That is roughly 1/84 of the 4090's 1008 GB/s VRAM bandwidth and about 1/3 of CPU DRAM read bandwidth.
3. **Platform upgrade path (prior knowledge, verify).** Z490 boards with 10th-gen CPUs run CPU PCIe lanes at Gen3. Many Z490 boards enable PCIe 4.0 x16 with an 11th-gen Rocket Lake CPU (e.g. i9-11900K), which also adds AVX-512 including VNNI. That could double host-to-device bandwidth to about 24–25 GB/s and speed up CPU int8 dot products, at the same 8-core count. Check the specific board's support.
4. **AVX2 compute on the 10700K** (Skylake-derived core, prior knowledge):
   - **FP32:** 2 × 256-bit FMA per cycle per core = 32 FP32 FLOP/cycle/core. At about 4.7 GHz all-core × 8 cores, peak is roughly **1.2 TFLOPS**.
   - **FP16:** there is no native FP16 arithmetic, only F16C conversion, so FP16 compute is no faster than FP32.
   - **INT8:** done with `vpmaddubsw` + `vpmaddwd`. There is no VNNI on Comet Lake, so it needs more instructions per multiply-accumulate. The practical int8 dot-product rate is roughly 2–3x the FP32 rate per cycle.
   - **Practical takeaway:** for batch-1 decode these compute peaks far exceed what DRAM can feed (about 40 GB/s × about 2 ops per weight byte at 8-bit is about 80 GOPS needed), so CPU expert decode is firmly **memory-bound**. Dequantization cost (e.g. MXFP4 or Q4_K to fp32 or int8) can become the limit if kernels are poor, so kernel quality matters.
5. **Batch-1 decode roofline for CPU-resident experts.**
   - Let A be the active expert parameters per token that live on the CPU, at b bits per weight. CPU time per token ≈ A·b/8 ÷ BW_read. Shipping the same experts to the GPU instead costs A·b/8 ÷ 12 GB/s. So CPU compute is about **3x faster than transfer** for single-token decode (Fiddler's rationale).
   - Example, assuming about 4–4.5B active expert parameters at about 4.25 bits (gpt-oss-class, MXFP4; I did not verify the config in this session): about 2.1–2.4 GB per token. That gives roughly **15–19 tok/s ceiling on CPU at 35–40 GB/s**, versus about 5 tok/s if transferring every expert over PCIe. Anything cached in VRAM (attention, shared or dense layers, hot experts) moves out of this budget at 1 TB/s.
6. **Prefill and batch crossover: CPU compute vs PCIe upload, per expert.** Model it this way:
   - An expert has P params at about 4 bits, so P/2 bytes.
   - Upload time ≈ P/2 ÷ 12e9.
   - CPU compute time for T routed tokens ≈ 2·P·T ÷ F_cpu.
   - With an effective F_cpu of about 0.3–0.6 TFLOPS (realistic for quantized GEMM on 8 AVX2 cores), the break-even is T ≈ F_cpu ÷ (4 × 12e9), i.e. about **6–12 tokens per expert**.

   So in prefill or large-batch steps, once more than about 10–20 tokens route to an expert, streaming its weights to the GPU (overlapped with compute, from pinned staging buffers) beats CPU compute. The GPU side is never the limit: 165–660 TFLOPS.
   - Implication for slot design: an `offload-scheduler` that decides **per expert, per step**, based on the routed-token count, whether to compute on the CPU or upload to the GPU. KTransformers, Fiddler and llama.cpp's offload heuristics all approximate this.
7. **Uploading the whole model per prefill pass.** About 60 GB of expert weights at 12 GB/s is about **5 s per full pass**. So prefill chunks need roughly thousands of tokens before full-upload prefill beats hybrid prefill. Pinned-memory caps under WSL2 (section B) may prevent full-rate streaming of all experts, because pageable-to-pinned staging costs another CPU memcpy at DRAM speed.
8. **GPU kernel precision choice on GeForce Ada.**
   - BF16/FP16 with FP32 accumulate runs at only 165 TFLOPS, half the FP16-accumulate rate.
   - FP8 with FP32 accumulate reaches 330, and INT8 reaches 660 TOPS.
   - For compute-bound prefill of GPU-resident layers, prefer FP8 (E4M3) or INT8 W8A8 kernels. FP16 accumulate is an option only if numerically acceptable.
   - For memory-bound decode, the bytes per weight dominate, so 4-bit weights with FP16 or INT8 compute are best.
9. **72 MB L2 (prior-knowledge reasoning).** Small per-layer working sets fit in L2: KV for short contexts, router weights, small dense layers, and staging tiles for expert weights arriving over PCIe. This allows L2-resident reuse across a decode batch. The L2 is not large enough to cache experts across tokens; for example, one gpt-oss expert MLP at about 4 bits is several MB, so only a handful fit.

### Gaps
- I did not confirm sm_89 shared memory per SM (100 KB) and the register and thread limits from a clean copy of the CUDA Programming Guide table; query them on the device.
- AVX2 int8 and FP32 throughput figures for the 10700K come from prior microarchitecture knowledge (uops.info / Intel optimization manual), not fetched this session. The agent should microbenchmark.
- I found no 4090-specific measurement of achieved FP8 or INT8 GEMM TFLOPS versus peak. Community reports suggest real FP8 GEMMs reach a fraction of peak at 450 W, but I have no source.
- Z490 Resizable BAR and PCIe 4.0 support with 11th-gen CPUs varies by board and was not verified.
- Achievable read-only DRAM bandwidth for this exact DDR4-3600 configuration under WSL2 has not been measured; only the user's ~29 GB/s copy number exists.
