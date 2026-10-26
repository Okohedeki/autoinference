# Serving-system architectures, batching/scheduling, parallelism, and the DeepSeek open-source inference stack (as of 2026-10-23)

Target box (for relevance notes): 1x RTX 4090 24 GB (sm_89, Ada), PCIe 3.0 x16 (~12-13 GB/s real), i7-10700K (8C/16T, AVX2, no AVX-512/AMX), ~64 GB DDR4 (~35-45 GB/s), large MoE with CPU expert offload, 1-16 concurrent streams.

Source-handling note: items marked **[fetched]** were verified by reading the primary page in this session. Items marked **[canonical, not re-fetched]** cite the well-known paper/repo URL from prior knowledge; the writer should treat their numbers as the authors' claims and may wish to spot-check. Expected-gain figures are the authors' own reported numbers unless otherwise stated.

---

## Q1. What exactly has DeepSeek open-sourced for inference (through Oct 2026), including the new "DeepSeek inference engine"?

### Takeaway
On **2026-10-20 DeepSeek open-sourced DeepInfer** (Apache-2.0), the production inference framework behind its API. It is a vLLM-derived engine that supports only DeepSeek-V4.1-Flash and only NVIDIA SM100 (Blackwell) and Huawei Ascend 950. It is offline-inference only: no KV-cache system, no PD-disaggregation glue, and no DP load balancer. It **cannot run on an RTX 4090 (sm_89)** as released. Its value for this project is as a design reference: async "zero-overhead" scheduling, confidence-scheduled MTP speculation, SWA bounded replay, batch-invariance, and the slot decomposition. It joins the 2025 Open Infra Week libraries (FlashMLA, DeepEP, DeepGEMM, DualPipe, EPLB, profile-data, 3FS, smallpond), the V3/R1 inference-system write-up, the April 2025 "path to open-sourcing" note, and 2026 kernel libraries (TileKernels, DeepSelect, DeepJIT, DeepGEMM-Ascend) plus deepseek-recipe.

### Cited Findings

**DeepInfer (released 2026-10-20)**
- **Release and licence:** DeepSeek open-sourced DeepInfer on Oct 20, 2026, describing it as the inference framework behind its production services. It is released under Apache 2.0. — [SiliconANGLE, 2026-10-20](https://siliconangle.com/2026/10/20/deepseek-open-sources-deepinfer-the-inference-framework-behind-its-production-services/); [Quartz](https://qz.com/deepseek-open-source-deepinfer-inference-software-ai-102026)
- **Self-description:** The README describes DeepInfer as "the high-performance inference framework that powers DeepSeek's large-scale production services". It says this release "supports offline inference with the core features needed for serving at scale." **[fetched]** — [github.com/deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer)
- **Lineage:** "originally built primarily on vLLM", with additional inspiration from SGLang and TensorRT-LLM. **[fetched]** — [DeepInfer README](https://github.com/deepseek-ai/DeepInfer)
- **Model:** Only DeepSeek-V4.1-Flash is supported. The default checkpoint is `deepseek-ai/DeepSeek-V4.1-Flash-1017`, and `DEEPINFER_HF_MODEL_DIR` can point to a local copy. **[fetched]** — [DeepInfer README](https://github.com/deepseek-ai/DeepInfer)
- **Hardware:** NVIDIA SM100 GPUs (CUDA 13) and Huawei Ascend 950 NPUs (CANN 9.2). Secondary reporting states that Hopper and older are not supported. Other requirements are Linux, GCC 13 or later, and Python 3.12. The build is `export DEEPINFER_ACCELERATOR=nvidia; ./build.sh`, and the example is `examples/offline_inference.py`. **[fetched]** — [DeepInfer README](https://github.com/deepseek-ai/DeepInfer); [web summary of SiliconANGLE/New Stack coverage](https://thenewstack.io/deepseek-deepinfer-huawei-nvidia/)
- **Not included:** The README lists three pieces "required for a complete inference service (not included in this release)":
  - a KV cache system;
  - Prefill/Decode disaggregation ("most required engine functionality is already implemented");
  - load balancing across DP replicas for prefill and decode.
  A third-party write-up phrases the second as "VE/prefill/decode disaggregation", where VE is likely a vision encoder (V4.1 is multimodal). **[fetched]** — [DeepInfer README](https://github.com/deepseek-ai/DeepInfer); [DEV write-up](https://dev.to/tiaforge/deepinfer-whats-actually-inside-deepseeks-newly-open-sourced-inference-engine-35m1)
- **Features listed in the README** **[fetched]** — [DeepInfer README](https://github.com/deepseek-ai/DeepInfer)
  - **Kernels:** FlashMLA, DeepGEMM, DeepEP, TileKernels, DeepSelect, and custom TileLang kernels, under `deepinfer/model_executor/layers/tilelang_ops`.
  - **Optimized model and sampler runtime.**
  - **Asynchronous scheduling:** a "zero-overhead batch scheduler" that overlaps CPU batch scheduling with device execution. It is on by default.
  - **Speculative decoding "DSpark":** MTP draft tokens, with "confidence estimates to schedule verification". It is enabled with `DEEPINFER_ENABLE_MTP_VARLEN=1`, and `DEEPINFER_USE_NEXTN_DECODE` sets the maximum draft length (default 5). Secondary reports quote DeepSeek estimating a **4-7% throughput gain at production load** from this approach ([SiliconANGLE](https://siliconangle.com/2026/10/20/deepseek-open-sources-deepinfer-the-inference-framework-behind-its-production-services/)).
  - **Decoder SWA bounded replay:** replays only a bounded suffix to cut prefill work. The defaults are `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` and `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`. The retained suffix must cover the SWA window plus the maximum backtracking length.
  - **Parallelism:** Attention-DP with balanced context parallelism (CP), and MoE EP with EPLB. No TP is mentioned.
  - **Batch-invariant, deterministic inference:** `DEEPINFER_BATCH_INVARIANT=1`, NVIDIA only. Deterministic sampling also needs `DEEPINFER_ENABLE_MTP_VARLEN=0` and a fixed `seq_seed`.
  - **Structured outputs:** regex, JSON-schema, or EBNF, via `SamplingParams(so_args=...)`.
- **Repo layout:** `csrc/`, `deepinfer/`, `deep_cluster/`, `doc/`, `eval/`, `examples/` (including `sweep/run_sweep.sh`), `scripts/`, `tests/`, `vendor/`, plus `build.sh`, `setup.py`, `pyproject.toml`, and separate requirements files for base, NVIDIA, Ascend, and dev. The page snapshot showed about 21k stars and 4 commits on main. Benchmarks are Pareto plots only, with no numbers in the text: prefill on 4 SM100 GPUs, decode on 64 SM100 GPUs within one scale-up domain. **[fetched]** — [DeepInfer README](https://github.com/deepseek-ai/DeepInfer)

**Other DeepSeek org repos (snapshot 2026-10-23)** **[fetched]** — [github.com/deepseek-ai](https://github.com/deepseek-ai)
- The org has 45 repos. Recently updated:
  - deepseek-harness: a "plugin-based harness", in TypeScript (Oct 23).
  - DeepInfer (Oct 21).
  - DeepEP (Oct 21).
  - DeepGEMM: "GPU BLAS kernel library" (Oct 20).
  - deepseek-recipe: Rust (Oct 19).
  - **DeepJIT**: a "lightweight library for xPU kernel JIT compilation" (Oct 19).
  - FlashMLA (Oct 15).
  - **DeepGEMM-Ascend** (Oct 14).
  - **TileKernels**: a kernel library written in TileLang (Oct 14).
- **deepseek-recipe** (MIT) is "a collection of Rust libraries and Python bindings" for protocol conversion. It converts Messages, Chat Completions, and Responses requests into a Conversation, renders the prompt, tokenizes it, and parses model output back. The caller supplies model inference, tool execution, and HTTP transport. Crates: `deepseek-recipe`, `-core`, `-encoding`, `-image`, `-macros`, and `-python`, plus example `server-rs` and `server-py`. It supports V4 and V4.1. **[fetched]** — [github.com/deepseek-ai/deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)

**V4.1-Flash: the model DeepInfer serves**
- **Paper:** arXiv:2609.19969, "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression", submitted 2026-09-17. **[fetched]** — [arXiv abstract](https://arxiv.org/abs/2609.19969)
  - **Architecture:** a multimodal MoE with a 552B backbone and 1M context, using a Causal Encoder-Decoder (CED). It activates 16B parameters per token in decode and 8B in prefill.
  - **KV compression:** CSA2 (Compressed Sparse Attention 2) with cross-layer KV reuse, plus FP4 KV cache. The global HBM KV is **890 bytes/token, about 1/4 of V4-Flash**. "SWA Bounded Replay" cuts the persistent (SSD or host) KV to about 1/8.
- **Kernel flow:** According to the paper text as surfaced by search, V4.1-Flash relies on:
  - a fused RoPE-attention-RoPE-cast kernel in FlashMLA;
  - Mega-Gate, Mega-mHC, and Mega-MoE kernels in DeepGEMM;
  - TileKernels;
  - a TopK kernel in DeepSelect.
  Only the search snippet was seen, not the PDF body. — [arXiv PDF](https://arxiv.org/pdf/2609.19969)
- **V4 timeline (secondary sources):**
  - V4 family (Pro and Flash) announced 2026-04-24, described as a 1.6T MoE with 1M context ([NYU Shanghai RITS](https://rits.shanghai.nyu.edu/ai/deepseek-releases-v4-open-source-1-6t-moe-with-1m-context)).
  - V4-Flash-0731 on 2026-07-31 ([DataNorth](https://datanorth.ai/news/deepseek-releases-deepseek-v4-flash-0731); [TechNode](https://technode.com/2026/07/31/deepseek-puts-v4-flash-api-into-public-beta)).
  - V4.1-Flash on about 2026-09-10 ([LetsDataScience](https://letsdatascience.com/news/deepseek-updates-flash-model-with-leaner-serving-1cf96356)).

**2025 releases** **[fetched]** — [open-infra-index](https://github.com/deepseek-ai/open-infra-index)
- **Open Source Week, one drop daily from 2025-02-24:**
  - **Day 1, FlashMLA:** MLA decoding kernel for Hopper, BF16, paged KV.
  - **Day 2, DeepEP:** EP all-to-all communication library, with normal (high-throughput) and low-latency modes.
  - **Day 3, DeepGEMM:** FP8 GEMM for dense and MoE, JIT-compiled.
  - **Day 4:** DualPipe (bidirectional pipeline parallelism), EPLB (expert-parallel load balancer), and profile-data (V3/R1 compute-comm overlap traces).
  - **Day 5:** 3FS (Fire-Flyer File System, SSD+RDMA) and smallpond (data processing on 3FS).
  - **Day 6:** the V3/R1 Inference System Overview, a write-up rather than code.
- **April 2025:** "The Path to Open-Sourcing the DeepSeek Inference Engine".
- **May 2025:** ISCA'25 paper "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures" ([arXiv 2505.09343](https://arxiv.org/abs/2505.09343)).
- **V3/R1 Inference System Overview (Day 6)** **[fetched]** — [day_6 write-up](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
  - **Design:** Cross-node EP is used because only 8 of 256 experts are active per token, so a large global batch is needed to give each expert enough rows. FP8 is used for GEMMs and dispatch; BF16 for core MLA and combine.
  - **Prefill unit:** "Routed Expert EP32, MLA/Shared Expert DP32". It spans 4 nodes with 32 redundant routed experts, and each GPU holds 9 routed experts plus 1 shared expert.
  - **Decode unit:** "Routed Expert EP144, MLA/Shared Expert DP144". It spans 18 nodes, and each GPU holds 2 routed experts plus 1 shared expert.
  - **Overlap:** Prefill uses **dual-batch overlap**: two micro-batches alternate so one's communication hides behind the other's compute. Decode splits attention into 2 steps and runs a **5-stage pipeline**.
  - **Three load balancers:**
    - prefill: equalizes core-attention compute and input tokens per GPU;
    - decode: equalizes KV usage and request count per GPU;
    - expert-parallel: minimizes the maximum dispatch-receive load.
  - **Stats, 24 h from 2025-02-27/28:**
    - 226.75 average H800 nodes (278 peak);
    - 608B input tokens, of which 342B (56.3%) hit the **on-disk KV cache**;
    - 168B output tokens at 20-22 tok/s per user, with average KV length 4,989;
    - about 73.7k input tok/s and 14.8k output tok/s per H800 node;
    - theoretical 545% cost-profit margin.
- **Path to Open-Sourcing note (Apr 2025)** **[fetched]** — [open-infra-index/OpenSourcing_DeepSeek_Inference_Engine](https://github.com/deepseek-ai/open-infra-index/tree/main/OpenSourcing_DeepSeek_Inference_Engine)
  - **Why the engine was not released then:**
    - It was based on "an early fork of vLLM from over a year ago", "heavily customized for DeepSeek models".
    - It was "tightly coupled with our internal infrastructure, including cluster management tools".
    - The team had limited maintenance bandwidth.
  - **The plan instead:** extract standalone libraries, contribute optimizations upstream (vLLM named as the base), and sync before launches for "Day-0" community support.
  - **Inference from these facts:** DeepInfer (Oct 2026) reverses the 2025 "won't release" stance, but keeps the "infrastructure-coupled pieces withheld" boundary: the KV store, PD glue, and load balancer are still missing.

**Upstream adoption of DeepSeek techniques (SGLang reproduction, May 2025)** **[fetched]** — [LMSYS blog, 2025-05-05](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
- **Setup:** SGLang reproduced the DeepSeek design on 96 H100s with PD disaggregation, DP-attention plus DP dense FFN plus EP MoE, DeepEP, DeepGEMM, EPLB, and two-batch overlap.
- **Throughput:** 52.3k input and 22.3k output tok/s per node for 2K inputs, about $0.20 per 1M output tokens, and up to 5x vanilla TP16.
- **Phase-specific modes:** DeepEP normal mode suits prefill; low-latency mode suits CUDA-graph decode. PD disaggregation is what lets each phase use its own mode. DeepGEMM likewise uses contiguous layout for prefill and masked layout for decode.
- **EPLB:** 1.49x prefill and 2.54x decode speedups.
- **TBO gains:** 27-35% throughput at the same tokens per device. It also enables 16,384 tokens per device in prefill (vanilla runs out of memory beyond 8,192), for a 40.5% gain.
- **TBO hurts decode at small batches:** -27% at 32 tokens per device, +25.5% at 256, with breakeven around 64-128 tokens per device.

**DeepSeek-V3.2 (2025)** **[canonical, not re-fetched]**
- V3.2-Exp (Sept 2025) introduced DeepSeek Sparse Attention (DSA), a "lightning indexer" plus top-k token selection. It came with sparse FlashMLA kernels and TileLang reference kernels, and had day-0 vLLM and SGLang support. — [github.com/deepseek-ai/DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp); [FlashMLA repo](https://github.com/deepseek-ai/FlashMLA)

### Inferences
- **DeepInfer does not run on the target box.** It needs SM100 or Ascend 950, CUDA 13, FP4/FP8 Blackwell kernels, and an EP/DP multi-GPU design, and it ships only V4.1-Flash. Treat it as an architecture reference and a source of algorithms, not as a runnable baseline.
- **Transferable to a single Ada GPU:**
  - the async/zero-overhead scheduler pattern;
  - confidence-scheduled MTP verification (DSpark), which helps with 1-16 streams;
  - SWA bounded replay and suffix-only prefill for SWA/compressed-attention models;
  - batch-invariant kernels as a correctness oracle when an agent swaps kernels (deterministic outputs make A/B equivalence testing trivial);
  - grammar-constrained decoding as a sampler slot.
- **Possibly portable:** DeepSelect (TopK) and TileKernels are TileLang. TileLang targets multiple GPUs, but sm_89 support for these specific kernels is unverified.
- **The 4-7% DSpark figure is at production (large-batch) load.** At batch 1-16 on a 4090, speculative decoding's relative gain is typically much larger, because decode is memory- or PCIe-bound there.
- **DeepInfer's README feature list is effectively a slot list:** kernels (attention, GEMM, MoE, comm, TopK), model runtime, sampler, scheduler (async), speculative decoder, prefill-replay policy, parallel layout (DP-attn/CP/EP), determinism mode, and structured-output constraint. Its explicit exclusions are the KV store, PD transport, and router/LB.

### Gaps
- Exact internal module boundaries of DeepInfer (scheduler and KV-manager code structure, and how much vLLM v1 or MRV2 code remains) were not inspected. The README does not describe internals; reading `deepinfer/` source is needed.
- The DSpark algorithm details (how confidence is estimated, the verification-batching policy) are not described beyond the README one-liner. No paper was found.
- The exact role of `deep_cluster/` is unknown.
- I could not verify whether DeepSeek released DeepInfer support for any older model (V3/R1/V3.2) or for Hopper. The README says V4.1-Flash only.
- The full The New Stack article body could not be retrieved (truncated).
- The DeepSelect repo was not seen in the first page of the org listing, so whether it is public or vendored is unconfirmed.
- No confirmation of V4-Pro open weights or serving details was found in this session.

---

## Q2. Batching and scheduling techniques catalog (one entry per technique)

### Takeaway
Serving-level throughput gains come from four families:
1. **Batch formation:** continuous batching, chunked prefill, and token budgets.
2. **Phase separation:** disaggregation, or its single-GPU analogs (stall-free mixing, nano-batching).
3. **Memory and preemption policy:** paged KV, swap vs. recompute, prefix caching.
4. **CPU-overhead elimination:** async or overlap scheduling, GPU-side input prep, and CUDA graphs.

At 1-16 streams on one GPU with CPU experts, the dominant levers are (a) continuous batching with chunked prefill, (b) zero-overhead or async scheduling, (c) prefix caching, and (d) speculative decoding. Cluster techniques (PD disaggregation, routing, DP load balancing) mostly reduce to "stream prioritization" policies on one box.

### Cited Findings

Fields per entry: what it does | slot | preconditions | expected gain and regime | quality impact | maturity | sources.

**B1. Static (request-level) batching.** Run a fixed batch until all sequences finish.
- **Slot:** scheduler.
- **Gain:** baseline. It wastes compute on padding and finished sequences.
- **Quality:** none.
- **Maturity:** legacy (HF generate, early FasterTransformer).
- **Source:** discussed as the baseline in Orca — [OSDI'22](https://www.usenix.org/conference/osdi22/presentation/yu). [canonical]

**B2. Continuous / iteration-level batching with selective batching (Orca).** The scheduler re-forms the batch at every decode iteration: finished requests leave and new ones join. Non-attention ops are batched across requests; attention runs per request.
- **Slot:** scheduler plus attention backend (needs a varlen/ragged batch).
- **Gain:** the paper reports up to 36.9x throughput vs. FasterTransformer at the same latency, for GPT-3 175B.
- **Quality:** none.
- **Maturity:** universal (vLLM, SGLang, TRT-LLM "in-flight batching", TGI, llama-server `-np` slots, LMDeploy "persistent batch").
- **Source:** [Orca, OSDI'22](https://www.usenix.org/conference/osdi22/presentation/yu). [canonical]

**B3. PagedAttention / paged KV cache (vLLM).** The KV cache lives in fixed-size blocks with a block table, like OS virtual memory. This nearly eliminates fragmentation and enables copy-on-write sharing (parallel sampling, beam search).
- **Slot:** KV-cache manager plus attention kernel.
- **Gain:** the paper reports 2-4x throughput vs. FasterTransformer and Orca at the same latency, from larger effective batches.
- **Quality:** none.
- **Maturity:** universal.
- **Source:** [Kwon et al., SOSP'23, arXiv 2309.06180](https://arxiv.org/abs/2309.06180). [canonical]

**B4. Chunked prefill (Sarathi) and stall-free scheduling (Sarathi-Serve).** Split long prompts into fixed-size chunks and piggyback decode tokens into each chunked-prefill iteration. A per-iteration token budget bounds iteration latency, so ongoing decodes never stall behind a large prefill ("generation stalls").
- **Slot:** scheduler (token budget) plus attention (mixed prefill/decode batch).
- **Preconditions:** a varlen attention kernel that handles a mixed batch, or POD-style fused kernels.
- **Gain:** Sarathi-Serve reports up to 2.6x serving capacity for Mistral-7B on 1 A100, and up to 3.7x for Yi-34B on 2 A100s, under TBT SLOs vs. vLLM. Sarathi (2023) reported up to 10x decode throughput improvement for LLaMA-13B.
- **Quality:** none.
- **Maturity:** default in vLLM V1, SGLang, TRT-LLM, and DeepSpeed-FastGen ("Dynamic SplitFuse").
- **Single-GPU relevance:** high. Chunk size is the key knob trading TTFT against TPOT. With CPU experts, chunk size also sets the CPU-side GEMM shape (bigger chunks give better AVX2 GEMM efficiency).
- **Sources:** [Sarathi, arXiv 2308.16369](https://arxiv.org/abs/2308.16369); [Sarathi-Serve, OSDI'24, arXiv 2403.02310](https://arxiv.org/abs/2403.02310). [canonical]

**B5. Dynamic SplitFuse (DeepSpeed-FastGen).** Long prompts are split into chunks across iterations, and short prompts are fused to hit a target forward size (token count) every step.
- **Slot:** scheduler.
- **Gain:** the paper claims up to 2.3x effective throughput vs. vLLM.
- **Maturity:** DeepSpeed-MII. The project has since become largely inactive [inference, unverified].
- **Source:** [arXiv 2401.08671](https://arxiv.org/abs/2401.08671). [canonical]

**B6. Unified token-budget scheduler (vLLM V1).** There is no prefill/decode distinction. Each request has `num_computed_tokens` and the scheduler assigns tokens under a budget. This generalizes chunked prefill, prefix caching, and speculative decoding into one loop.
- **Slot:** scheduler.
- **Maturity:** vLLM V1 (2025), now the default.
- **Sources:** [vLLM V1 blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release) [canonical; URL redirect seen this session]; [vLLM docs](https://docs.vllm.ai/).

**B7. Prefill-decode disaggregation (DistServe, Splitwise, Mooncake, TetriInfer, DéjàVu).** Prefill and decode run on separate GPU pools, and the KV cache is transferred between them. This removes interference and allows per-phase parallelism and hardware.
- **Slot:** cluster topology plus KV transfer plus router.
- **Preconditions:** two or more devices and fast interconnect.
- **Sources and gains** [all canonical]:
  - **DistServe:** up to 7.4x more requests or 12.6x tighter SLO — [OSDI'24, arXiv 2401.09670](https://arxiv.org/abs/2401.09670).
  - **Splitwise:** 1.4x throughput at 20% lower cost, or 2.35x throughput at the same power and cost — [ISCA'24, arXiv 2311.18677](https://arxiv.org/abs/2311.18677).
  - **Mooncake:** KVCache-centric disaggregated architecture for Kimi, with a CPU/DRAM/SSD KV pool, early rejection, and up to 525% throughput increase in simulation (paper claim). FAST'25 — [arXiv 2407.00079](https://arxiv.org/abs/2407.00079), [github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake).
  - **TetriInfer:** chunked prefill plus disaggregation plus length-prediction-based decode scheduling — [arXiv 2401.11181](https://arxiv.org/abs/2401.11181).
  - **DéjàVu:** KV streaming for disaggregation, microbatch swapping, and fault tolerance — [ICML'24, arXiv 2403.01876](https://arxiv.org/abs/2403.01876).
  - **DeepSeek production:** EP32 prefill, EP144 decode **[fetched]** — [Day-6 overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md).
- **Maturity:** production in SGLang, vLLM (NIXL/LMCache connectors), NVIDIA Dynamo, llm-d, and Mooncake. Not included in DeepInfer's release.
- **Single-GPU analog:** temporal disaggregation. Alternate prefill-heavy and decode-heavy phases, or run phase-specific kernel configs (e.g., DeepEP normal vs. low-latency, DeepGEMM contiguous vs. masked) chosen per step. On the target box, prefill is CPU-GEMM/PCIe-bound and decode is DDR-bandwidth-bound, so per-phase offload policy is the analog: stream experts to the GPU for big prefill batches, compute on CPU for decode.

**B8. POD-Attention / fused prefill-decode attention.** A single kernel co-schedules prefill and decode attention on the same SMs to overlap compute-bound and memory-bound work in hybrid batches.
- **Slot:** attention backend.
- **Gain:** paper reports up to 59% faster attention (avg 28%) and up to 22% higher end-to-end throughput.
- **Single-GPU relevance:** directly applicable.
- **Source:** [arXiv 2410.18038](https://arxiv.org/abs/2410.18038). [canonical]

**B9. Nano-batching / intra-device parallelism (NanoFlow).** Split the batch into nano-batches and run compute-, memory-, and network-bound ops concurrently on one GPU via SM partitioning and multiple streams.
- **Slot:** executor / op scheduler.
- **Gain:** paper reports 1.91x throughput vs. TensorRT-LLM.
- **Single-GPU relevance:** high. This is the canonical intra-GPU micro-batch overlap, and the analog of TBO for overlapping CPU-expert compute with GPU attention.
- **Source:** [arXiv 2408.12757](https://arxiv.org/abs/2408.12757). [canonical]

**B10. SLO-aware scheduling.** Admission and ordering policies target TTFT/TPOT SLOs (goodput) rather than raw throughput.
- **Slot:** scheduler policy.
- **Examples** [canonical]:
  - DistServe goodput (above).
  - **Andes:** QoE-aware token delivery — [arXiv 2404.16283](https://arxiv.org/abs/2404.16283).
  - **Llumnix:** live migration of requests across instances for load balancing, de-fragmentation, and priorities — [OSDI'24, arXiv 2406.03243](https://arxiv.org/abs/2406.03243).
- **Single-GPU relevance:** priorities and deadlines among 1-16 streams.

**B11. Preemption: swap vs. recompute.** When KV memory runs out, preempt a request. Either swap its KV to CPU RAM (PCIe cost) or drop it and recompute the prefill later (compute cost).
- **Slot:** KV manager plus scheduler policy.
- **Gain:** avoids OOM and enables overcommit. vLLM V1 uses recompute by default.
- **Quality:** none.
- **Single-GPU relevance:** with PCIe 3.0 (~12 GB/s), recompute often beats swap for short contexts, but swap wins for long contexts. A cost model is needed.
- **Source:** [vLLM paper §4.5](https://arxiv.org/abs/2309.06180). [canonical]

**B12. Fairness: Virtual Token Counter (VTC).** Fair-queuing across clients by counting served tokens (weighted input/output), with a provable fairness bound.
- **Slot:** scheduler policy.
- **Source:** [Sheng et al., OSDI'24, arXiv 2401.00588](https://arxiv.org/abs/2401.00588). [canonical]

**B13. Preemptive/MLFQ scheduling (FastServe).** Skip-join multi-level feedback queue at token granularity to reduce head-of-line blocking, with proactive KV swapping.
- **Slot:** scheduler.
- **Gain:** paper reports up to 31.4x throughput improvement at the same latency vs. vLLM (2023).
- **Source:** [arXiv 2305.05920](https://arxiv.org/abs/2305.05920). [canonical]

**B14. Output-length prediction scheduling.** Predict response length to approximate shortest-job-first and pack batches.
- **Slot:** scheduler (plus a predictor model).
- **Sources** [canonical]:
  - Response Length Perception — [arXiv 2305.13144](https://arxiv.org/abs/2305.13144);
  - Learning-to-Rank scheduling — [arXiv 2408.15792](https://arxiv.org/abs/2408.15792);
  - TetriInfer (above).
- **Gain:** latency and throughput improvements under mixed lengths.
- **Quality:** none.
- **Maturity:** research.

**B15. Prefix caching / RadixAttention.** Reuse the KV of shared prefixes across requests. SGLang uses a radix tree with LRU; vLLM V1 uses hash-based blocks with "near-zero overhead" prefix caching on by default.
- **Slot:** KV manager plus scheduler (cache-aware ordering).
- **Gain:** large for multi-turn chat, agents, and few-shot prompts. DeepSeek reports 56.3% of input tokens hit its disk KV cache **[fetched]**.
- **Quality:** none.
- **Maturity:** universal. llama-server has `--cache-reuse` and slot prompt caching.
- **Sources:** [SGLang, arXiv 2312.07104](https://arxiv.org/abs/2312.07104) [canonical]; [Day-6 overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md).

**B16. Cache-aware scheduling ordering.** Order the waiting queue by longest-prefix match (SGLang "LPM" and DFS-weight policies) to maximize cache hits. Preble does the same across distributed GPUs.
- **Slot:** scheduler policy.
- **Sources:** [SGLang paper](https://arxiv.org/abs/2312.07104); [Preble, arXiv 2407.00023](https://arxiv.org/abs/2407.00023). [canonical]

**B17. Tiered KV cache / KV offload (host RAM, SSD).** Spill KV blocks to CPU or disk and reload them on prefix hit.
- **Examples:**
  - CachedAttention/AttentionStore — [arXiv 2403.19708](https://arxiv.org/abs/2403.19708) [canonical];
  - LMCache — [github.com/LMCache/LMCache](https://github.com/LMCache/LMCache) [canonical];
  - Mooncake store;
  - SGLang HiCache;
  - DeepSeek's disk KV cache (3FS-backed) **[fetched]**.
- **Slot:** KV-cache storage tier.
- **Single-GPU relevance:** high for multi-turn agents. 64 GB of DDR4 is contested with MoE experts, so an NVMe tier is likely.

**B18. Overlap / zero-overhead scheduling (SGLang v0.4).** Run CPU scheduling for batch N+1 while the GPU executes batch N, using future-token placeholders. This hides scheduler, radix-cache, and detokenizer CPU time.
- **Slot:** scheduler/executor loop.
- **Gain:** SGLang reported a 1.1x average and up to 1.3x speedup [canonical].
- **Maturity:** default in SGLang. vLLM has `--async-scheduling` (V1 retrofit), and MRV2 is async-first. DeepInfer has it on by default **[fetched]**.
- **Sources:** [LMSYS SGLang v0.4 blog, 2024-12-04](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) [canonical]; [DeepInfer README](https://github.com/deepseek-ai/DeepInfer).
- **Single-GPU relevance:** critical at small batch with Python orchestration. Also required to overlap CPU-expert work.

**B19. GPU-native input prep and async output copy (vLLM Model Runner V2).** **[fetched]**
- **What it does:**
  - Each request gets a stable row in a fixed-size state table.
  - Per-step inputs (`input_ids`, `positions`, `query_start_loc`, `seq_lens`) are built on-GPU with Triton kernels.
  - The sampler is Triton-native (Gumbel-max).
  - Outputs are copied to the host on a separate stream.
  - Rejection-sampling results feed prep kernels directly, so spec decode and async scheduling coexist without syncs.
- **Slot:** model runner.
- **Gain:** +56% output tok/s on Qwen3-0.6B on GB200 (host-bound case); -6.3% TPOT for GLM-4.7-FP8 with MTP.
- **Maturity:** opt-in via `VLLM_USE_V2_MODEL_RUNNER=1`, announced 2026-03-24. As of v0.18.0 it lacks LoRA, EPLB, DBO, linear attention, and logits processors.
- **Source:** [vLLM blog, MRV2](https://vllm.ai/blog/2026-03-24-mrv2).

**B20. CPU-overhead reductions: detokenization and process split.**
- **What it does:** Move the API server, tokenizer, and detokenizer into separate processes, off the GPU loop. vLLM V1 uses an EngineCore process with ZMQ; SGLang uses separate tokenizer, scheduler, and detokenizer processes.
- **Related:** CUDA graphs for decode and torch.compile (vLLM V1 piecewise CUDA graphs).
- **Slot:** frontend/IPC plus executor.
- **Gain:** the vLLM V1 blog claimed up to 1.7x throughput vs. V0 [canonical].
- **Sources:** [vLLM V1 blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release); [SGLang paper](https://arxiv.org/abs/2312.07104). [canonical]

**B21. CUDA graphs and batch-size bucketing.** Capture decode steps for a set of batch sizes and pad to the nearest bucket.
- **Slot:** executor.
- **Gain:** removes kernel-launch overhead (large at batch 1-16).
- **Preconditions:** static shapes and no host sync. CPU-offloaded experts break graphs unless split at offload boundaries.
- **Maturity:** universal. llama.cpp has CUDA graphs for single-sequence decode.
- [general knowledge; no single canonical paper]

**B22. Request routing / load balancing.** Least-load, prefix-cache-aware (consistent hashing on prefix), or KV-aware routing across replicas.
- **Examples:** SGLang router (cache-aware), NVIDIA Dynamo KV-aware router, llm-d inference scheduler, AIBrix.
- **DeepSeek:** three balancers (prefill, decode, EP) **[fetched]**.
- **Slot:** router (cluster). There is no single-GPU analog except stream priority.
- **Sources** [canonical]: [Day-6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md); [Dynamo](https://github.com/ai-dynamo/dynamo); [llm-d](https://github.com/llm-d/llm-d).

**B23. Multi-LoRA serving (S-LoRA, Punica, dLoRA).** Batch requests with different LoRA adapters using custom gather kernels (Punica SGMV; S-LoRA MBGMM/MBGMV), with a unified paged pool for adapters and KV.
- **Slot:** linear-layer kernel plus adapter memory manager.
- **Gain:** S-LoRA serves thousands of adapters with up to 4x throughput vs. vLLM-packed.
- **Quality:** none.
- **Maturity:** vLLM, SGLang, LoRAX, and TRT-LLM.
- **Sources** [canonical]: [S-LoRA, arXiv 2311.03285](https://arxiv.org/abs/2311.03285); [Punica, arXiv 2310.18547](https://arxiv.org/abs/2310.18547).
- **Single-GPU relevance:** low unless the user serves adapters.

**B24. Speculative decoding as a scheduling problem.** Draft-and-verify (MTP, EAGLE, Medusa, n-gram/prompt lookup) is integrated into continuous batching. The serving-level question is how many draft tokens to verify per request under load. DeepInfer's DSpark schedules verification by confidence **[fetched]**.
- **Slot:** speculator plus scheduler (token budget).
- **Gain:** large at low batch, shrinking at high batch (DeepSeek: 4-7% at production load).
- **Quality:** lossless with rejection sampling.
- **Sources:** [DeepInfer README](https://github.com/deepseek-ai/DeepInfer); [EAGLE, arXiv 2401.15077](https://arxiv.org/abs/2401.15077) [canonical].
- Detailed speculative-decoding coverage is presumably in a sibling note.

**B25. Batch-invariant / deterministic inference.** Kernels whose reductions do not depend on batch composition (fixed split-K, fixed attention chunking), so outputs are bitwise identical regardless of co-batched requests.
- **Slot:** kernel policy (GEMM, attention, RMSNorm).
- **Cost:** some throughput.
- **Maturity:** DeepInfer `DEEPINFER_BATCH_INVARIANT=1` **[fetched]**; vLLM and SGLang deterministic modes (2025).
- **Origin:** Thinking Machines, "Defeating Nondeterminism in LLM Inference", Sept 2025 [canonical].
- **Sources:** [DeepInfer](https://github.com/deepseek-ai/DeepInfer); [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
- **Relevance:** very high as a test oracle for an agent swapping kernels.

**B26. Bounded replay / suffix-only re-prefill for SWA and compressed-KV models.** Persist only compressed or global KV. On cache hit, recompute only a bounded suffix (≥ SWA window plus backtrack) instead of storing per-layer SWA KV.
- **Slot:** KV-cache policy plus prefill planner.
- **Gain:** about 1/8 persistent KV footprint for V4.1-Flash; reduced prefill.
- **Preconditions:** a model with SWA plus a global/compressed attention split.
- **Maturity:** new (Sept-Oct 2026).
- **Sources:** [arXiv 2609.19969](https://arxiv.org/abs/2609.19969); [DeepInfer README](https://github.com/deepseek-ai/DeepInfer). **[fetched]**

**B27. Structured / grammar-constrained decoding.** Regex, JSON-schema, or EBNF masks applied in the sampler (XGrammar, Outlines, llguidance, llama.cpp GBNF). Overlapping mask computation with the GPU forward pass is the serving-level optimization.
- **Slot:** sampler/logits processor.
- **Sources:** [DeepInfer README](https://github.com/deepseek-ai/DeepInfer) **[fetched]**; [XGrammar, arXiv 2411.15100](https://arxiv.org/abs/2411.15100) [canonical].

**B28. Hybrid CPU-GPU MoE scheduling (directly relevant).**
- **KTransformers** (SOSP'25, Tsinghua MADSys + Approaching.AI) **[fetched abstract via search]**: AMX-specialized CPU kernels plus an asynchronous CPU-GPU task scheduler. Reports 4.62-19.74x prefill and 1.25-4.09x decode vs. existing hybrid systems. **Expert Deferral** raises CPU utilization from below 75% to nearly 100%, giving up to 1.45x more throughput at ≤0.5% average accuracy drop. Since Oct 2025 the CPU kernels are integrated into SGLang. — [SOSP'25 paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf); [LMSYS blog 2025-10-22](https://lmsys.org/blog/2025-10-22-KTransformers)
- **Fiddler:** execute experts on CPU rather than transferring weights over PCIe when that is cheaper — [arXiv 2402.07033](https://arxiv.org/abs/2402.07033) [canonical].
- **MoE-Lightning:** CPU-GPU-I/O pipelining with a hierarchical roofline model, for batch throughput on memory-constrained GPUs — [arXiv 2411.11217](https://arxiv.org/abs/2411.11217) [canonical].
- **HybriMoE:** hybrid CPU-GPU scheduling and cache management for MoE — [arXiv 2504.05897](https://arxiv.org/abs/2504.05897) [canonical; ID appeared in search results].
- **llama.cpp / ik_llama.cpp:** `--override-tensor` / `--n-cpu-moe` place expert tensors in CPU RAM — [llama.cpp](https://github.com/ggml-org/llama.cpp) [canonical].

### Inferences
- **Highest-value scheduling slots for the target box:**
  1. a token-budget scheduler with chunked prefill (chunk size is a searchable knob);
  2. an async/overlap loop (B18/B19), so Python and CPU-expert dispatch don't serialize with the GPU;
  3. a prefix cache (radix or hash) plus a host/NVMe KV tier;
  4. a speculation policy (draft length per stream, confidence-gated);
  5. preemption policy (swap vs. recompute under a PCIe 3.0 cost model);
  6. a CPU-GPU MoE overlap policy (Expert Deferral, NanoFlow-style micro-batching).
- **Low priority on one box:** disaggregation, routing, DP load balancing, and multi-LoRA.
- **Workload shape:** with 1-16 streams, the decode batch is small, so per-step fixed costs matter disproportionately. Python overhead, kernel launches, host syncs, and PCIe transfer latency for router logits and expert selection all count. This is why async scheduling and CUDA graphs matter more here than in datacenter settings.
- **Searchable vs. stateful knobs:** Plugin boundaries should keep the scheduler policy (pure function: queue state → batch plan) separate from the KV-manager mechanism (block allocation, eviction, tiers), as vLLM V1 does. An agent can then search policies cheaply without touching memory safety.

### Gaps
- No primary-source verification this session of SGLang's exact overlap-scheduler speedup numbers, or of the vLLM V1 1.7x claim (blog fetch redirected and was not re-fetched).
- Recent 2026 academic scheduling papers (OSDI'26/MLSys'26/EuroSys'26) were not surveyed. A USENIX OSDI'26 paper (wang wenxin) appeared in search results with an unknown topic.
- No quantitative data was found on chunked-prefill optimal chunk size for CPU-offloaded MoE on consumer hardware.

---

## Q3. Parallelism strategies, and which have single-GPU analogs

### Takeaway
Multi-device parallelism (TP, PP, SP/CP, EP, DP-attention, DualPipe) is irrelevant on one GPU as such. But the overlap ideas behind DeepSeek's two-batch overlap and 5-stage decode pipeline, NanoFlow, and DualPipe map directly onto overlapping CPU expert compute, PCIe transfers, and GPU attention across micro-batches on a CPU+GPU box. The CPU is effectively a second, slow device in an "expert-parallel" layout. SGLang data shows micro-batch overlap only pays off above a per-device token threshold: decode breakeven is about 64-128 tokens per device in the datacenter case.

### Cited Findings

**P1. Tensor parallelism (Megatron).** Split weight matrices column- and row-wise across GPUs, with an all-reduce per layer.
- **Gain:** lower latency, more memory. Needs NVLink-class bandwidth.
- **Single-GPU analog:** split a layer between CPU and GPU (llama.cpp tensor split). This is generally bad over PCIe 3.0, except for row-split of huge FFNs.
- **Source:** [arXiv 1909.08053](https://arxiv.org/abs/1909.08053). [canonical]

**P2. Pipeline parallelism (GPipe, interleaved 1F1B).** Layers are split into stages, with micro-batches flowing through.
- **Inference:** micro-batches fill the bubbles.
- **Single-GPU analog:** layer-wise CPU/GPU split (llama.cpp `-ngl`), where GPU layers and CPU layers form a 2-stage pipeline. With 2+ micro-batches (streams), both devices can be busy.
- **Sources:** [GPipe, arXiv 1811.06965](https://arxiv.org/abs/1811.06965) [canonical]. vLLM v0.16 supports async scheduling plus PP — [release notes](https://newreleases.io/project/pypi/vllm/release/0.16.0).

**P3. DualPipe (DeepSeek, training).** A bidirectional pipeline that feeds micro-batches from both ends. Forward and backward chunks are overlapped, and each chunk is split into attention / all-to-all dispatch / MLP / all-to-all combine so communication is fully hidden.
- **Note:** training-oriented.
- **Sources:** [github.com/deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe); [DeepSeek-V3 tech report, arXiv 2412.19437](https://arxiv.org/abs/2412.19437); listed as Day 4 in [open-infra-index](https://github.com/deepseek-ai/open-infra-index) **[fetched]**.

**P4. Sequence parallelism / context parallelism / Ring Attention.** Shard the sequence dimension across devices.
- **Examples:**
  - Megatron SP — [arXiv 2205.05198](https://arxiv.org/abs/2205.05198);
  - Ring Attention — [arXiv 2310.01889](https://arxiv.org/abs/2310.01889);
  - LoongServe elastic SP — [arXiv 2404.09526](https://arxiv.org/abs/2404.09526);
  - DeepInfer "balanced CP" with attention-DP **[fetched]**.
- **Single-GPU analog:** split long-context attention between GPU (recent KV) and CPU (old KV), as in KV-offload attention.
- [canonical except DeepInfer]

**P5. Expert parallelism (EP).** Experts are sharded across devices, with all-to-all dispatch and combine (DeepEP).
- **DeepEP modes:**
  - normal: NVLink plus RDMA forwarding, for prefill and training;
  - low-latency: pure RDMA, CUDA-graph compatible, hook-based overlap that occupies no SMs, for decode.
- **Redundant experts and EPLB** fix imbalance: SGLang measured 1.49x prefill and 2.54x decode from EPLB.
- **Single-GPU analog:** CPU-GPU expert placement. Hot experts stay on the GPU; cold experts live in CPU RAM. EPLB-style statistics drive the choice of which experts are "redundant" (GPU-cached).
- **Sources:** [DeepEP](https://github.com/deepseek-ai/DeepEP); [EPLB](https://github.com/deepseek-ai/EPLB); [LMSYS large-scale EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/) **[fetched]**.

**P6. Data-parallel attention (DP-attention).** Each device holds a full attention/MLA replica, with different requests on each, while MoE uses EP. This avoids replicating MLA's compressed KV under TP (MLA has 1 KV head).
- **Used in:** DeepSeek (DP32/DP144), SGLang, vLLM, and DeepInfer.
- **Single-GPU analog:** none, except that attention stays entirely on the GPU while experts are distributed.
- **Sources:** [Day-6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md); [LMSYS](https://lmsys.org/blog/2025-05-05-large-scale-ep/) **[fetched]**.

**P7. Two-batch overlap (TBO / DBO) and DeepSeek's 5-stage decode pipeline.** Split the batch into 2 micro-batches. While micro-batch A does attention or compute, B does dispatch or combine communication. In decode, attention is split into two steps for a 5-stage pipeline.
- **Gains (SGLang):** +27-35% at fixed tokens per device. Decode: -27% at 32 tokens per device, +25.5% at 256.
- **Maturity:** SGLang (`--enable-two-batch-overlap`); vLLM "DBO" (dual-batch overlap; not yet in MRV2 as of v0.18).
- **Single-GPU analog (key):** micro-batch A runs GPU attention plus shared expert while micro-batch B's routed experts run on CPU (AVX2) and PCIe transfers proceed. This is what KTransformers' async CPU-GPU scheduler and Expert Deferral exploit. Expert Deferral defers some experts' results to the next layer to widen the overlap window, at ≤0.5% accuracy cost.
- **Sources:** [Day-6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md); [LMSYS](https://lmsys.org/blog/2025-05-05-large-scale-ep/); [MRV2 blog](https://vllm.ai/blog/2026-03-24-mrv2); [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) **[fetched]**.

**P8. Disaggregated expert parallelism / attention-FFN disaggregation (MegaScale-Infer).** Attention and FFN/expert modules run on separate GPU pools, with ping-pong micro-batch pipelining between them.
- **Single-GPU analog:** the CPU+GPU split, where attention runs on the GPU and experts on the CPU, is literally attention-FFN disaggregation with ping-pong.
- **Source:** [arXiv 2504.02263](https://arxiv.org/abs/2504.02263). [canonical]

**P9. Intra-GPU nano-batching (NanoFlow, see B9).** The only parallelism here that is natively single-GPU: overlap of heterogeneous ops via SM partitioning and streams. [canonical]

### Inferences
- **Treat CPU+GPU as a 2-device heterogeneous system.** The relevant "parallel layout" slot choices are:
  1. which tensors live where (attention, shared expert, and hot experts on the GPU; cold experts on the CPU);
  2. micro-batch count (1 vs. 2 vs. N) and the overlap schedule;
  3. per-phase policy: for prefill with large token counts, stream expert weights over PCIe to compute on the GPU (~12 GB/s means about 1 s per 12 GB); for decode, compute on the CPU;
  4. an expert-cache replacement policy (EPLB statistics as a prior).
- **When TBO pays off:** The SGLang threshold (TBO hurts below ~64-128 tokens per device) suggests a micro-batch overlap analog may hurt at 1-4 streams and help at 8-16 streams plus prefill. The agent should search this per regime rather than fix it.
- **Not applicable on one box:** DualPipe itself (training), TP/SP across devices, and DeepEP's all-to-all. DeepEP's design idea, communication that occupies no compute SMs and uses hook-based completion, maps onto using copy engines / async memcpy streams for PCIe transfers so SMs stay busy.

### Gaps
- No measured data on two-micro-batch CPU/GPU overlap for a specific 4090 + AVX2 desktop configuration.
- KTransformers' published numbers rely heavily on AMX (Xeon). Gains on AVX2-only CPUs like the i7-10700K are unknown from these sources.

---

## Q4. Engine architectures and plugin seams

### Takeaway
Most engines share a common skeleton:
- frontend/API → tokenizer → scheduler (batch plan) → KV-cache manager → model runner/executor → model definition (layers) → kernel backends (attention, GEMM/quant, MoE, comm) → sampler → detokenizer.

The cleanest existing plugin seams are vLLM's: out-of-tree plugin entry points for models and platforms, an attention-backend registry, quantization-method registry, KV connectors, and MRV2's `ModelState`. SGLang's are similar. llama.cpp's seam is the ggml backend and graph scheduler, with tensor-override placement rules.

For the target box (single consumer GPU plus CPU-offloaded MoE), the most relevant engines are llama.cpp/ik_llama.cpp, KTransformers (now also inside SGLang), ExLlamaV3 (GPU-only), and vLLM/SGLang (if the model fits, or with CPU offload). DeepInfer is reference-only.

### Cited Findings (one entry per engine; [canonical] unless marked fetched)

**E1. vLLM (V1 engine, 2025 to present; MRV2, 2026).** Python plus C++/CUDA/Triton.
- **Core design:**
  - EngineCore runs in a separate process (ZMQ IPC) from the API server, tokenizer, and detokenizer.
  - A unified token-budget scheduler.
  - A KV-cache manager with paged blocks and hash-based prefix caching.
  - A worker per device, each with a model runner.
  - torch.compile plus piecewise CUDA graphs.
  - Async scheduling (V1 retrofit; MRV2 async-first) **[fetched for MRV2]**.
- **Seams:**
  - `vllm.general_plugins` / `vllm.platform_plugins` entry points (out-of-tree models, platforms such as Ascend and AMD ATOM);
  - attention backends (FlashAttention, FlashInfer, Triton, FlashMLA);
  - `QuantizationConfig`/`QuantizeMethodBase` registry (GPTQ, AWQ, Marlin, FP8, GGUF, compressed-tensors);
  - model registry;
  - KV connectors (LMCache, NIXL) for PD and offload;
  - logits processors;
  - speculative proposers;
  - MRV2 `ModelState` for model-specific runner logic.
- **Consumer GPU:** strong for models that fit in VRAM, with Marlin kernels for Ampere/Ada. Weak for CPU-offloaded MoE (`--cpu-offload-gb` is a slow weight-streaming approach).
- **DeepInfer is a vLLM descendant** **[fetched]**.
- **Sources:**
  - [vLLM V1 blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release)
  - [MRV2 blog](https://vllm.ai/blog/2026-03-24-mrv2) **[fetched]**
  - [AMD vLLM-ATOM plugin write-up](https://rocm.blogs.amd.com/software-tools-optimization/vllm-atom/README.html) (plugin registration path; plugin handles attention-backend routing while vLLM keeps scheduling and KV management)
  - [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

**E2. SGLang.** Python plus CUDA/Triton, with an sgl-kernel package and a Rust router.
- **Core design:**
  - separate tokenizer, scheduler, and detokenizer processes;
  - RadixAttention prefix cache;
  - zero-overhead overlap scheduler;
  - frontend DSL;
  - PD disaggregation;
  - DP-attention, EP with DeepEP, TBO, EPLB;
  - HiCache tiered KV;
  - KTransformers CPU-expert kernels integrated (Oct 2025).
- **Seams:** attention backends (FlashInfer, FA3, Triton, FlashMLA, TRT-LLM MLA), quantization methods (largely shared with vLLM), model files under `python/sglang/srt/models`, the router, and grammar backends (XGrammar, Outlines, llguidance).
- **Consumer GPU:** good for in-VRAM models. The KTransformers integration gives a hybrid MoE path.
- **Sources:** [SGLang paper](https://arxiv.org/abs/2312.07104); [LMSYS large-scale EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/) **[fetched]**; [LMSYS KTransformers in SGLang](https://lmsys.org/blog/2025-10-22-KTransformers); [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang).

**E3. TensorRT-LLM (NVIDIA).** C++/CUDA with a Python API. It started as a TensorRT engine-build (AOT compile) design and shifted in 2025 to a PyTorch-native backend as default.
- **Features:** in-flight batching, paged KV, chunked prefill, KV reuse, speculative decoding (EAGLE, MTP), and FP8/FP4/INT4-AWQ.
- **Seams:** model definitions, plugins (TensorRT plugins historically), attention backends, and quant modes.
- **Consumer GPU:** supports Ada (FP8 on sm_89). Heavyweight, and no CPU-expert offload.
- **Source:** [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).

**E4. LMDeploy / TurboMind (InternLM).** TurboMind is a C++/CUDA engine derived from FasterTransformer, with a Python PyTorch engine as an alternative backend.
- **Features:** persistent (continuous) batching, blocked KV, KV INT8/INT4 quantization, AWQ W4A16, tensor parallelism.
- **Consumer GPU:** good INT4 kernels, sm_75 and newer.
- **Source:** [github.com/InternLM/lmdeploy](https://github.com/InternLM/lmdeploy).

**E5. llama.cpp / llama-server (ggml-org).** C/C++ on top of the ggml tensor library.
- **Core design:** computation graph plus a backend scheduler (`ggml_backend_sched`) that assigns ops to CUDA, CPU, Vulkan, Metal, SYCL and similar backends by weight placement. Also GGUF quantization (k-quants, i-quants).
- **llama-server:** continuous batching over `-np` parallel slots, a unified KV cache, prompt cache and reuse, speculative decoding (draft model, n-gram), and GBNF grammars.
- **MoE offload:** `--override-tensor` (`-ot`) regex placement, `--n-cpu-moe`.
- **Seams:** ggml backends (new backends plug in at the device/op level), quant types (type traits plus kernels per backend), model architecture (`llama-model.cpp` graph builders), and the sampler chain.
- **Consumer GPU:** excellent and universal; the main CPU+GPU MoE baseline.
- **Source:** [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

**E6. ik_llama.cpp (ikawrakow fork).** A llama.cpp fork focused on CPU and hybrid performance.
- **Features:** new quant types (IQ_K, trellis/IQ*_KT), faster CPU matmul kernels (AVX2/AVX-512/NEON), MLA support for DeepSeek with FlashMLA-style variants, fused MoE ops, runtime repacking, and better hybrid GPU/CPU MoE throughput.
- **Seams:** same as llama.cpp.
- **Relevance:** very high for AVX2 plus 4090 with DeepSeek/Qwen MoE.
- **Source:** [github.com/ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp).

**E7. ExLlamaV2 / ExLlamaV3 / TabbyAPI (turboderp).** Python plus CUDA.
- **Quantization:** EXL2 (V2) uses mixed-bit GPTQ-like quantization. EXL3 (V3) is QTIP-based trellis quantization.
- **Features:** a dynamic generator with continuous batching, paged attention (flash-attn), prefix dedup, and speculative decoding. TabbyAPI is the OpenAI-compatible server.
- **Consumer GPU:** top single-user tok/s for in-VRAM models. Little or no CPU-expert offload (GPU-centric).
- **Sources:** [github.com/turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3); [github.com/theroyallab/tabbyAPI](https://github.com/theroyallab/tabbyAPI).

**E8. KTransformers (kvcache-ai / Tsinghua MADSys).** Python framework with C++ CPU kernels (AMX/AVX-512/AVX2, llamafile-derived) and a YAML-rule-based module injection system that replaces HF modules with optimized ones.
- **Features:** async CPU-GPU scheduler, Expert Deferral, CUDA graphs, and integration into SGLang (Oct 2025).
- **Seams:** the YAML injection rules (regex on module names → replacement class plus device) are themselves a "slot" system.
- **Relevance:** highest for the target box. It is a near-direct precedent for slot swapping. Caveat: the headline performance assumes AMX.
- **Sources:** [github.com/kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers); [SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) **[fetched abstract]**.

**E9. MLC-LLM (TVM / Apache TVM Unity).** An ML-compilation engine. Models are expressed in TVM Relax/TensorIR and compiled per target (CUDA, Vulkan, Metal, WebGPU, mobile). MLCEngine adds continuous batching, paged KV, and speculative decoding.
- **Seams:** compiler passes and schedules (auto-tuning), and model definitions in Python.
- **Consumer GPU:** portable, though kernel performance on CUDA often trails hand-tuned engines [inference].
- **Source:** [github.com/mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm).

**E10. DeepSpeed-FastGen / MII.** Built on the DeepSpeed-Inference v2 "ragged batching" kernels, with Dynamic SplitFuse and a blocked KV cache.
- **Status:** development activity appears to have slowed since 2024 [inference; not verified].
- **Sources:** [arXiv 2401.08671](https://arxiv.org/abs/2401.08671); [github.com/deepspeedai/DeepSpeed-MII](https://github.com/deepspeedai/DeepSpeed-MII).

**E11. Text Generation Inference (Hugging Face).** Rust router/launcher with Python/CUDA model servers. Continuous batching, flash/paged attention, and multiple backends (including TRT-LLM and llama.cpp backends).
- **Status:** I believe HF moved TGI to maintenance mode in late 2025, recommending vLLM/SGLang. **Not verified this session, so treat as a gap.**
- **Source:** [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference).

**E12. Ollama.** Go server. It originally wrapped llama.cpp; since 2025 it has its own Go engine on top of ggml for newer multimodal models, with automatic GPU/CPU layer splitting and model management.
- **Consumer GPU:** convenience over peak throughput.
- **Source:** [github.com/ollama/ollama](https://github.com/ollama/ollama).

**E13. LM Studio.** Closed-source desktop app with pluggable runtimes ("llama.cpp" CUDA/Vulkan/CPU and "MLX" on Apple), plus an OpenAI-compatible server.
- **Seam:** runtime engines are swappable packages.
- **Source:** [lmstudio.ai](https://lmstudio.ai/).

**E14. mistral.rs.** Rust engine on top of candle.
- **Features:** continuous batching, PagedAttention, ISQ (in-situ quantization), GGUF/GPTQ/AWQ/HQQ/FP8, speculative decoding, X-LoRA/LoRA, device mapping across GPU/CPU, and MoE expert offload features.
- **Source:** [github.com/EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs).

**E15. candle (Hugging Face).** A minimalist Rust ML framework (tensor library plus CUDA/Metal/CPU backends). It is a library rather than a serving engine; mistral.rs and others build on it.
- **Source:** [github.com/huggingface/candle](https://github.com/huggingface/candle).

**E16. Apple MLX / mlx-lm.** Array framework for Apple silicon (lazy evaluation, unified memory, Metal kernels). mlx-lm provides generation, quantization, and a server with batching added in 2025.
- **Relevance:** not relevant to CUDA hardware, but a clean design reference.
- **Sources:** [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx); [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm).

**E17. DeepSeek DeepInfer (2026-10-20).** **[fetched]** See Q1 for details.
- **Design:** a vLLM-derived core with SGLang and TRT-LLM ideas, DeepSeek kernel libraries, a TileLang ops layer, async scheduling, DSpark spec decode, SWA bounded replay, DP-attention/CP/EP+EPLB, batch-invariant mode, and structured outputs.
- **Languages:** Python plus C++ (`csrc/`) plus TileLang/CUDA, with Ascend CANN support.
- **Consumer GPU:** none (SM100 or Ascend 950 only).
- **Source:** [github.com/deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer).

**E18. Orchestration layers (not engines).** These sit above engines and handle disaggregation and routing. Not needed for one box.
- NVIDIA Dynamo (March 2025; PD disaggregation, KV-aware routing, NIXL transfer) — [github.com/ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)
- llm-d (Red Hat/Google/IBM; K8s-native vLLM disaggregation) — [github.com/llm-d/llm-d](https://github.com/llm-d/llm-d)
- Mooncake — [github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- AIBrix — [github.com/vllm-project/aibrix](https://github.com/vllm-project/aibrix)

### Inferences
**Proposed slot map** (synthesized from the seams above and from DeepInfer's feature and exclusion list):
1. Frontend/protocol (cf. deepseek-recipe).
2. Tokenizer/detokenizer process.
3. Scheduler policy: batch formation, chunk size, priorities, preemption choice, speculation budget.
4. KV-cache manager: paging, prefix index, eviction, tiers (GPU/host/NVMe), quantized KV.
5. Executor loop: sync vs. async/overlap, CUDA-graph policy, micro-batch count.
6. Model definition / graph builder.
7. Placement policy: which tensors and experts go to GPU vs. CPU (cf. llama.cpp `-ot`, KTransformers YAML rules).
8. Attention backend.
9. Dense GEMM / quant-method backend (GPU).
10. MoE backend: router/top-k, GPU grouped-GEMM, CPU expert kernels (AVX2), expert cache and prefetch, transfer engine.
11. Sampler: including grammar and penalties; speculative verify.
12. Speculator: MTP / EAGLE / n-gram / draft model.
13. Determinism / batch-invariance mode, for the agent's correctness checks.

**Other inferences:**
- **Placement as a rule set:** KTransformers' YAML injection and llama.cpp `-ot` regex are the closest existing precedents for "agent swaps implementations per slot". Both express placement and implementation as declarative rules over module or tensor names, which an agent can search over.
- **vLLM vs. llama.cpp style seams:** vLLM's process split (EngineCore vs. frontend) and MRV2's `ModelState` show how to keep per-model hacks out of the core runner. llama.cpp's backend scheduler shows how to make CPU/GPU op placement a graph-level decision rather than model code.

### Gaps
- Exact 2026 state was not verified this session for:
  - TGI (maintenance mode?);
  - DeepSpeed-MII activity;
  - Ollama's engine split;
  - ExLlamaV3 MoE/CPU offload features;
  - mistral.rs MoE offload specifics;
  - TensorRT-LLM's PyTorch-backend default.
  These descriptions are from prior knowledge (through mid-2026) and should be spot-checked.
- No official vLLM attention-backend registry design doc was found in this session's search.
- Precise language breakdowns per engine (beyond the obvious) were not gathered.
