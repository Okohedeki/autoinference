# Inference optimization techniques in TensorRT-LLM, FlashInfer and NVIDIA's adjacent inference libraries (as of 2026-10-23)

Scope: NVIDIA TensorRT-LLM (PyTorch backend + TRT-LLM Gen kernels; latest tag seen v1.4.0rc1, 2026-10-22), FlashInfer (latest v0.7.2.post1, 2026-10-23), cuDNN frontend SDPA (support matrix at cuDNN 9.18.1, features through 9.27), CUTLASS examples 00–95 (CuTe DSL 4.7 era), NVIDIA Dynamo (1.0 "production-ready", container tag 1.6.0), ModelOpt (latest dated item 2026-10-15), NIXL (2026 README), kvpress (README, paper arXiv 2510.00636).

Method actually used (39 tool calls): TRT-LLM docs feature pages (quantization, feature-combination matrix, speculative-decoding, kvcache, kv-cache-compression, paged-attention-ifb-scheduler, parallel-strategy, attention, sparse-attention, torch_compile_and_piecewise_cuda_graph), the `cpp/tensorrt_llm/kernels` tree, GitHub releases v1.3.0rc28 → v1.4.0rc1, the `docs/source/blogs/tech_blog` index (30 posts); FlashInfer docs API index, flashinfer.ai release-highlights page (v0.6.14 → v0.7.2), v0.7 blog, blog index; cuDNN frontend `docs/operations/Attention.md`; CUTLASS `examples/` tree; Dynamo README + router docs (via search snippets); ModelOpt README; NIXL README; kvpress README. Pages that 404'd or were not opened are listed under Gaps.

Architecture shorthand used in the table: **Ampere** = sm_80/86; **Ada** = sm_89; **Hopper** = sm_90; **Blackwell DC** = sm_100/sm_103 (B200/B300, GB200/GB300); **Blackwell consumer/workstation** = sm_120/121 (RTX PRO 6000, RTX 5090, DGX Spark); **Rubin** = sm_107 (appears in current TRT-LLM/FlashInfer/cuDNN docs; included because the sources list it).

---

## Key Question 1 — Which inference optimization techniques are implemented, per library, with preconditions / regime / claimed gain / status?

### Takeaway
Across the eight NVIDIA sources there are ~300 distinct, individually pluggable techniques; the densest clusters are (a) low-precision GEMM/MoE recipes (FP8 per-tensor/block/rowwise, NVFP4, MXFP4, W4A16/W4A8, with hard per-architecture gates), (b) attention kernel families (FMHA/XQA/MLA/TRT-LLM Gen/PrimTS/cuDNN/CuTe DSL plus seven sparse-attention schemes), (c) KV-cache tiering/compression (host/disk tiers, NVFP4 cold pages, TriAttention, 45 kvpress eviction presses), (d) speculative decoding (8 live modes in TRT-LLM, 2 deprecated), and (e) MoE expert-parallel communication (MegaMoE, one-sided NVLink all-to-all, NCCL-EP/NIXL-EP, DWDP, Helix). Most docs pages give qualitative gains only; numeric claims below are the ones actually printed in the fetched sources.

### Cited Findings (the table)

Slot vocabulary: quant-recipe, placement, expert-cache, offload-scheduler, moe-cpu, moe-gpu, linear-gpu, attn-prefill, attn-decode, recurrent-mixer, fused-elementwise, kv-dtype, kv-select, kv-allocator, kv-tier, sampler, spec, scheduler, executor, loader, os-config, parallelism, engine.

| # | technique | slot(s) | what it does (one line) | preconditions (hardware arch / model features) | regime (prefill/decode/batch size) | claimed gain (with source) | lossless? | status (stable/experimental, version/date) | link |
|---|---|---|---|---|---|---|---|---|---|
| **TensorRT-LLM — quantization recipes & low-precision GEMM** | | | | | | | | | |
| 1 | NVFP4 (W4A4, 16-elem blocks + FP8 scale, FP32 tensor scale) | quant-recipe, linear-gpu | 4-bit weights+activations with two-level block scaling on 5th-gen tensor cores | Blackwell sm_100/103, sm_120, Rubin sm_107; **not** Hopper/Ada/Ampere | both; biggest win decode/memory-bound | Perf overview: DeepSeek-R1-0528 DEP4 FP4 1000/1000 ≈6,463 tok/s/GPU B200, 6,939 GB200 vs 1,627 H200 FP8 [PO] | No (PTQ; ModelOpt claims BF16-matching accuracy on Nemotron 3 Ultra [MO]) | Stable, default PyTorch backend path (quantization.md) | [Q] |
| 2 | MXFP4 (OCP 32-elem blocks, E8M0 scale) | quant-recipe, moe-gpu | MX-format 4-bit weights, used for GPT-OSS experts | Blackwell sm_100/103, sm_120, sm_107; not Hopper | both | not quantified on page | No | Stable for GPT-OSS (quantization.md matrix) | [Q] |
| 3 | FP8 per-tensor (E4M3, static scales) | quant-recipe, linear-gpu | FP8 W/A GEMM with per-tensor scaling | Ada sm_89 + Hopper + Blackwell + Rubin; not Ampere | both | not quantified on page | No (near-lossless PTQ) | Stable | [Q] |
| 4 | FP8 block scaling (128×128 weight / 1×128 act; MXFP8-style on sm_100+) | quant-recipe, linear-gpu | Fine-grained FP8 for DeepSeek-style checkpoints; sm_100/103/107 kernels use MXFP8-style recipe differing from SM90 recipe | Hopper, Blackwell DC sm_100/103, Rubin; **not** sm_120, Ada, Ampere | both | n/q | No | Stable (DeepSeek-R1/V3/V4) | [Q] |
| 5 | FP8 rowwise (per-token act, per-channel weight) | quant-recipe, linear-gpu | Dynamic rowwise FP8 (Llama 3 recipe) | Hopper only | both | n/q | No | Stable, Hopper-only | [Q] |
| 6 | W4A16 AWQ (INT4 weight-only, activation-aware scales) | quant-recipe, linear-gpu | 4-bit weights dequantized to 16-bit in-kernel | Ampere+ (all listed arches incl. sm_120) | small-batch decode (weight-bandwidth-bound) | n/q | No | Stable | [Q] |
| 7 | W4A16 GPTQ | quant-recipe, linear-gpu | 4-bit weight-only via GPTQ Hessian rounding | Ampere+ | small-batch decode | n/q | No | Stable | [Q] |
| 8 | W4A8 AWQ (INT4 weight, FP8 activation) | quant-recipe, linear-gpu | 4-bit weights, 8-bit activations to recover compute-bound throughput | Ada, Hopper, Blackwell DC, Rubin; **not** sm_120 or Ampere | mid-batch | n/q | No | Stable | [Q] |
| 9 | W4A8 GPTQ | quant-recipe, linear-gpu | as #8 with GPTQ weights | same as #8 | mid-batch | n/q | No | Stable | [Q] |
| 10 | W4A16 NVFP4 group-size 32 | quant-recipe, linear-gpu | NVFP4 weights, BF16 activations, 32-elem groups (weight-only FP4) | Blackwell | small-batch decode | n/q | No | New in v1.3.0rc29 (#19091) | [REL] |
| 11 | Nemotron-H NVFP4 on Hopper W4A16 fallback **removed** | quant-recipe | automatic Hopper fallback dropped; NVFP4 Nemotron-H now Blackwell-only | Blackwell | — | — | — | v1.4.0rc1 (#15772) | [REL] |
| 12 | Marlin NVFP4 MoE kernels (+Triton top-k combine) | moe-gpu, quant-recipe | Marlin-style weight-only FP4 GEMM for MoE with Triton combine | Hopper/Ampere-class (Marlin lineage); kernels/marlin | decode, small batch | n/q | No | v1.3.0rc29 (#18999); kernels/marlin exists | [KT] |
| 13 | QServe W4A8 GEMM (per-channel & per-group) | linear-gpu, quant-recipe | QoQ W4A8KV4-style INT4-weight/INT8-act GEMM (qserveGemmPerChannel/PerGroup.cu) | Ampere+/Hopper (INT8 tensor cores) | mid-batch | n/q | No | Present in kernels tree | [KT] |
| 14 | weightOnlyBatchedGemv | linear-gpu | W4/W8 weight-only batched GEMV for tiny M (decode) | Ampere+ | decode, batch ≤ few | n/q | Yes (same math as GEMM) | kernels/weightOnlyBatchedGemv | [KT] |
| 15 | tinygemm2 | linear-gpu | small-M GEMM path | n/s | decode | n/q | Yes | kernels/tinygemm2 | [KT] |
| 16 | CUTLASS kernels (cutlass_kernels, internal_cutlass_kernels; FP8/FP4 GEMM, MoE grouped GEMM, fp8GroupedGemmConfig, splitkGroupGemm, cuda_graph_grouped_gemm) | linear-gpu, moe-gpu | CUTLASS 3.x/4.x instantiations for dense, grouped, split-K and CUDA-graph-safe grouped GEMM | per instantiation (Hopper WS / Blackwell UMMA) | both | n/q | Yes | Stable | [KT] |
| 17 | FP8 per-token activation quant (fp8PerTokenQuant, fusedActivationQuant) | fused-elementwise | fused act→FP8 quantization with per-token scale | Ada+ | both | n/q | No | Stable | [KT] |
| 18 | Fused gated RMSNorm + quant (fusedGatedRMSNormQuant) | fused-elementwise | RMSNorm + gate + FP8/FP4 quant in one kernel | Blackwell for FP4 | both | n/q | No (quant) | Stable | [KT] |
| 19 | RMSNorm + FP4 quant fusion (rmsNormFp4QuantKernels) | fused-elementwise | norm then NVFP4 quantize, single pass | Blackwell | both | n/q | No | Stable | [KT] |
| 20 | Fused concat + FP4/FP8 quant (fusedCatFp4/fusedCatFp8) | fused-elementwise | concat Q/K/latent then quantize | Blackwell (FP4) / Ada+ (FP8) | both | n/q | No | Stable | [KT] |
| 21 | Pre-quant scale kernel (AWQ smoothing) | fused-elementwise | applies AWQ per-channel pre-scale to activations | any | both | n/q | Yes (algebraic) | Stable | [KT] |
| 22 | Fused QK-Norm + RoPE (fusedQKNormRope; DiT split variants) | fused-elementwise | per-head QK RMSNorm + RoPE fused | any | both | n/q | Yes | Stable | [KT] |
| 23 | Fused LayerNorm / group RMSNorm / adaptive LN kernels | fused-elementwise | fusedLayernormKernels, groupRmsNormKernels, fusedAdaptiveLayerNormKernel | any | both | n/q | Yes | Stable | [KT] |
| 24 | arcquantFP4 kernel | quant-recipe | FP4 quantization kernel variant ("arcquant") | Blackwell | — | n/q | No | File present; purpose not documented in fetched sources | [KT] |
| 25 | Inverse-RoPE FP8 quant (inverseRopeFp8QuantKernel) | fused-elementwise | MLA path: undo RoPE and FP8-quantize for cache | Hopper+ | both | n/q | No | Stable | [KT] |
| 26 | MLA context FP8 Q quant fused into absorb-BMM epilogue | fused-elementwise, attn-prefill | removes separate quant kernel on MLA context path | Hopper/Blackwell MLA models | prefill | n/q | No | v1.3.0rc29 (#19082) | [REL] |
| 27 | Deferred fused-GELU MLP NVFP4 scale finalization | fused-elementwise | finalizes FP4 scales lazily in MLP | Blackwell | both | n/q | No | v1.3.0rc29 (#18906) | [REL] |
| 28 | FP8 KV cache (`KvCacheConfig(dtype='fp8')`) | kv-dtype | stores KV as FP8 with per-tensor scale; dequant on the fly in gen kernel | **all** listed arches incl. Ampere | decode (halves KV bandwidth) | n/q | No (near-lossless) | Stable; works for checkpoints that didn't enable it | [Q] |
| 29 | NVFP4 active KV cache (`dtype='nvfp4'`) | kv-dtype | 4-bit KV in the hot GPU cache | Blackwell DC sm_100/103 only; requires FP8 W/A checkpoint quantized with ModelOpt `--kv_cache_quant nvfp4` | decode, long context | n/q | No | Stable but narrow (only FP8 W/A alongside) | [Q] |
| 30 | INT8 KV cache | kv-dtype | 8-bit integer KV with per-tensor scale (attention.md describes FP8/INT8 KV dequant) | Ampere+ | decode | n/q | No | Present in attention.md; absent from PyTorch-backend quant matrix (legacy TRT path) | [ATT] |
| 31 | INT8 SmoothQuant (W8A8) | quant-recipe | per-channel/per-token INT8 with activation smoothing | Ampere+ | both | n/q | No | **Unverified for current PyTorch backend** — not on quantization.md; legacy TRT-engine path | [Q] |
| 32 | MiniMax-M3 NVFP4 KV cache | kv-dtype | model-specific NVFP4 KV for MiniMax-M3 (requires `msa` sparse impl) | Blackwell DC | decode | n/q | No | v1.4.0rc0 (#19422) | [REL] |
| **TensorRT-LLM — MoE compute & MoE communication** | | | | | | | | | |
| 33 | CUTLASS MoE backend (kernels/moe; grouped GEMM) | moe-gpu | expert grouped GEMM with fused routing/permute | Hopper/Blackwell | both | n/q | Yes | Stable; idle workspace reclaimed v1.4.0rc1 (#19981) | [KT] |
| 34 | TRT-LLM Gen MoE (trtllmGenKernels) | moe-gpu | pre-generated Blackwell MoE kernels (BF16, FP8 per-tensor/block, NVFP4/MXFP4) | Blackwell DC sm_100/103 | both; low-latency variants for small batch | n/q here (FlashInfer wraps same kernels) | Yes | Stable | [KT] |
| 35 | CuTe DSL MoE backend (Blackwell/Rubin; Kimi K3 NVFP4 SiTU; SwiGLU clamp; NVFP4 FC1 layout) | moe-gpu | Python-authored CuTe DSL MoE GEMMs | Blackwell DC, Rubin sm_107 | both | n/q | Yes | v1.3.0rc29 (#19003), v1.4.0rc1 (#19488, #20181) | [REL] |
| 36 | MegaMoE (fused dispatch+expert+combine, from FlashInfer) | moe-gpu, parallelism | single fused EP MoE megakernel; cached kernel descriptors; SM107 and multi-node init | Blackwell DC (+Hopper BF16 via FlashInfer) | both | FlashInfer: 1.06–1.32× throughput DeepSeek-V4 Flash/Pro on 1×8 SM100 [F7] | Yes | v1.4.0rc0/rc1 (#19901, #19183) | [REL] |
| 37 | Fast MoE routing for batches ≤ 8 tokens | moe-gpu | specialized routing path for tiny decode batches | any | decode, batch ≤ 8 | n/q | Yes | v1.4.0rc0 (#19682) | [REL] |
| 38 | Finalize-before-combine MoE scheduler path | moe-gpu, scheduler | optional reordering of finalize vs. combine | EP | both | n/q | Yes | v1.4.0rc0 (#19749), opt-in | [REL] |
| 39 | Post-SiLU / SwiGLU clamping in MoE | fused-elementwise | numeric clamp inside fused activation for FP4 stability | Blackwell | both | n/q | Changes numerics slightly | v1.4.0rc0 (#19217) | [REL] |
| 40 | MoE as Dense GEMM (blog24) | moe-gpu | reformulates MoE as dense GEMM for small-batch latency | n/s (blog) | decode, low batch | see blog24 (not opened) | Yes | Tech blog #24 | [TB] |
| 41 | DeepSeek-V3 min-latency fused kernels (dsv3MinLatencyKernels) | moe-gpu, fused-elementwise | hand-fused kernels for DeepSeek-R1/V3 batch-1 latency | Blackwell | decode, batch 1–few | blog01 "Pushing latency boundaries" B200 | Yes | Stable | [KT][TB] |
| 42 | Llama 4 min-latency fused kernels (llama4MinLatencyKernels) | moe-gpu, fused-elementwise | fused kernels for Llama 4 Maverick low-latency decode | Blackwell | decode | NVIDIA claimed 1000+ TPS/user on Maverick (blog06 is the Eagle guide; the 1000 TPS figure **not confirmed in fetched sources**) | Yes | Stable | [KT] |
| 43 | No-aux-loss top-k routing kernels (noAuxTcKernels) | moe-gpu | DeepSeek "noaux_tc" expert selection with bias | DeepSeek-style routers | both | n/q | Yes | Stable | [KT] |
| 44 | MoE top-k funcs / moe_utils | moe-gpu | shared top-k, permute, scatter helpers | any | both | n/q | Yes | Stable | [KT] |
| 45 | Wide-EP (expert slots, replication, load balancing; custom EP comm kernels for GB200 MNNVL) | parallelism, placement | large-EP MoE (DeepSeek-V3/R1, Llama4, Qwen3) across NVL72 | GB200/GB300 MNNVL for custom kernels | high-throughput decode | blog04/08/14 (Scaling EP parts 1–3); Dynamo README: 7× throughput/GPU DeepSeek-R1 GB200 NVL72 vs B200 [DY] | Yes | Stable (examples/wide_ep) | [PAR] |
| 46 | Offline EPLB (static expert placement from historical stats) | placement | precompute replicated/hot-expert placement | Wide-EP | steady traffic | n/q | Yes | Stable | [PAR] |
| 47 | Online EPLB (dynamic, layer-wise weight redistribution) | placement, expert-cache | re-places experts at runtime to real traffic with layer-wise updates | Wide-EP | shifting traffic | n/q | Yes | Stable | [PAR] |
| 48 | One-sided all-to-all over NVLink (blog18; overhauled v1.4.0rc1 #19610) | parallelism | MoE dispatch/combine via NVLink one-sided puts instead of NCCL | NVLink/MNNVL (GB200) | EP decode | blog18 (not opened) | Yes | Stable | [TB][REL] |
| 49 | MoE all-to-all counters published before dispatch | parallelism | reduces sync latency in A2A | EP | decode | n/q | Yes | v1.3.0rc29 (#19425) | [REL] |
| 50 | DWDP — Distributed Weight Data Parallelism (dwdpFabricGather/Tma) | parallelism, placement | shards weights across NVL72 and gathers via fabric per layer, so each GPU runs DP | GB200 NVL72 (MNNVL fabric) | throughput | blog19 (not opened) | Yes | Tech blog #19 | [TB][KT] |
| 51 | Helix parallelism (KV-cache sharding; helixAllToAll/helixKernels) | parallelism | shards KV cache across GPUs for multi-million-token decode | multi-GPU NVLink | long-context decode | blog22 (not opened) | Yes | Feature page; "Untested" with overlap scheduler, "No" with PP and spec decoding [FM] | [FEAT][TB] |
| 52 | Attention DP (`enable_attention_dp`) | parallelism | replicate attention weights, partition KV across ranks; TP for MoE | multi-GPU | large batch | n/q | Yes | Stable | [PAR] |
| 53 | ADP balance strategy (blog10) | scheduler, parallelism | balances requests across attention-DP ranks | attention DP | throughput | blog10 (not opened) | Yes | Tech blog #10 | [TB] |
| 54 | Tensor parallelism | parallelism | shard weights/heads | multi-GPU | small batch, memory-constrained | n/q | Yes | Stable | [PAR] |
| 55 | Pipeline parallelism | parallelism | layer split across GPUs | multi-GPU | large models | n/q | Yes | Stable; "No" with spec decoding [FM] | [PAR] |
| 56 | Expert parallelism / MoE TP / hybrid ETP | parallelism | experts on single GPUs; or slice experts; product must equal TP | MoE | both | n/q | Yes | Stable | [PAR] |
| 57 | Context (sequence) parallelism — Ulysses (ulyssesPermuteScatter/PostUnscatter; recoverFromRingAtten) | parallelism | long-prompt attention split over GPUs (Ulysses head-parallel; ring recovery) | multi-GPU | long prefill | n/q | Yes | Stable; also MiniMax-H3 VAE Ulysses v1.4.0rc1 | [PAR][KT] |
| 58 | Token-sharded TP for Wan (VisualGen) | parallelism | token-dim sharding for video DiT | multi-GPU | DiT | n/q | Yes | v1.4.0rc1 (#19713) | [REL] |
| 59 | Custom all-reduce kernels (customAllReduceKernels; one-shot/two-shot) | parallelism, fused-elementwise | latency-optimized AR vs NCCL | NVLink | decode small msgs | n/q | Yes | Stable | [KT] |
| 60 | AllReduce + residual + RMSNorm (+FP8/FP4 quant) fusion | fused-elementwise, parallelism | torch.compile pass fuses AR, add, norm, quant into one op | TP | decode | n/q | Yes | Stable (torch.compile path) | [TC] |
| 61 | User-buffers AllReduce (`enable_userbuffers`; kernels/userbuffers) | parallelism | AR directly from registered buffers, no copy | NVLink | decode | n/q | Yes | Stable, opt-in | [TC][KT] |
| 62 | MNNVL all-reduce (incl. through Ray orchestrator) | parallelism | multi-node NVLink AR over NVL72 fabric | GB200/GB300 NVL72 | TP across nodes | n/q | Yes | v1.3.0rc29 (#18231) | [REL] |
| 63 | Rubin locality-domain execution | parallelism, os-config | topology-aware execution for Rubin | Rubin sm_107 | — | n/q | Yes | v1.4.0rc1 (#19272) | [REL] |
| 64 | LL128 protocol (ll128Proto.cuh) | parallelism | low-latency 128B comm protocol in custom comm kernels | NVLink | decode | n/q | Yes | File present | [KT] |
| **TensorRT-LLM — attention kernels & backends** | | | | | | | | | |
| 65 | TRT-LLM native attention backend (`TrtllmAttention`) | attn-prefill, attn-decode | default production backend; fused QKV in, optional FP8 out | all | both | n/q | Yes | Stable, default | [ATT] |
| 66 | FlashInfer attention backend | attn-prefill, attn-decode | FlashInfer kernels with IFB + paged KV; also GDN prefill on SM120/121 | per FlashInfer support | both | n/q | Yes | Stable (v1.3.0rc29 #19376 for GDN) | [ATT][REL] |
| 67 | Vanilla attention (reference) | attn-prefill | PyTorch reference w/ linear KV | any | — | — | Yes | not for production | [ATT] |
| 68 | Context FMHA (contextFusedMultiHeadAttention; FlashAttention-2 style, fmhaDispatcher) | attn-prefill | fused prefill attention; vanilla MHA for short seqs | Ampere+ | prefill | n/q | Yes | Stable; static tile scheduler override v1.4.0rc0 (#19873) | [ATT][KT] |
| 69 | FP8 context FMHA / FP8 paged context FMHA | attn-prefill | FP8 inputs in prefill attention | **Ada, Hopper and above** | prefill | n/q | No | Stable; tiled FP8 head-dim-256 on SM120 v1.4.0rc0 (#19825) | [ATT][REL] |
| 70 | XQA generation kernels (xqaDispatcher; blog XQA-kernel.md) | attn-decode | MQA/GQA decode kernel loading KV once per group; `TRTLLM_FORCE_XQA=1` | Hopper+ (originally H100/H200) | decode | blog "XQA-kernel" (not opened; historically ~2.4× throughput Llama-70B H200) | Yes | Stable, on by default for supported configs | [ATT][BL] |
| 71 | Masked MHA generation kernel (decoderMaskedMultiheadAttention) with multi-block (split-KV) mode | attn-decode | decode attention applying bias/RoPE/(de)quant on the fly; splits long KV across blocks when occupancy low | Ampere+ | decode, low batch × long ctx | n/q | Yes | Stable | [ATT][KT] |
| 72 | TRT-LLM Gen FMHA (generated Blackwell attention; CSA2 native trtllm-gen on sm_100) | attn-prefill, attn-decode | pre-generated Blackwell attention kernels incl. sparse/compression variants | Blackwell DC | both | n/q | Yes | Stable | [SPA][KT] |
| 73 | PrimTS attention (dense BF16, mixed BF16/FP8, FP8 context; block-sparse FMHA for VSA/SOL) | attn-prefill | Python-authored (CUTLASS DSL Primitives/Task-Scheduling) attention paths | Blackwell DC | prefill | n/q | Yes/No(FP8) | v1.4.0rc0/rc1 (#18079, #20006) | [REL] |
| 74 | CuTe DSL MLA (Blackwell; packed query heads, compact inputs) | attn-decode | MLA decode written in CuTe DSL | Blackwell DC | decode | n/q | Yes | v1.4.0rc1 (#19338) | [REL] |
| 75 | FlashMLA (kernels/flashMLA) | attn-decode, attn-prefill | DeepSeek FlashMLA port for SM90 (CSA2 backend on SM90) | Hopper | both | n/q | Yes | Stable | [KT][SPA] |
| 76 | MLA kernels (mlaKernels, mlaChunkedPrefill; absorb/non-absorb) | attn-prefill, attn-decode | latent-KV attention with weight absorption for decode, chunked prefill | Hopper/Blackwell; DeepSeek/Kimi/GLM MLA models | both | blog02/03/15 | Yes | Stable | [KT] |
| 77 | NVFP4 MLA KV gather (nvfp4MlaKvCacheGather) + NVFP4 MLA residual switch (DeepSeek-V4) | kv-dtype, attn-decode | FP4 latent KV read path | Blackwell DC | decode | n/q | No | v1.3.0rc29 (#19305) | [REL][KT] |
| 78 | Chunked context / chunked prefill | attn-prefill, scheduler | split long prompts into KV-block-multiple chunks interleaved with decode | paged KV + FMHA | prefill mixed with decode | n/q (bounds TTFT jitter) | Yes | Stable; "Untested" with PP [FM] | [SCH][ATT] |
| 79 | Sliding-window attention / cyclic (rolling-buffer) KV | attn-prefill, attn-decode, kv-allocator | `attention_window_size`; separate pools per window; freed blocks returned to tree | any; Gemma3/GPT-OSS | both | n/q | Yes (model-defined) | Stable; `sink_token_length` deprecated | [ATT][KV] |
| 80 | Attention sinks (GPT-OSS) | attn-decode | learned sink logits in softmax | GPT-OSS | both | blog09 GPT-OSS deploy | Yes (model) | Stable (via GPT-OSS support) — mechanism **not described on attention.md** | [TB] |
| 81 | ALiBi fused in kernel | attn-prefill, attn-decode | slopes computed in-kernel | any | both | n/q | Yes | Stable | [ATT] |
| 82 | RoPE fusion in attention (GPT-NeoX/GPT-J) | fused-elementwise | RoPE applied inside attention kernel | any | both | n/q | Yes | Stable | [ATT] |
| 83 | Relative attention bias kernel (buildRelativeAttentionBias) | attn-prefill | T5-style relative bias | enc-dec | prefill | n/q | Yes | Stable | [KT] |
| 84 | Cross-attention + cross KV cache | attn-prefill, attn-decode | encoder-decoder attention | enc-dec models | both | n/q | Yes | Stable | [ATT] |
| 85 | Beam-search KV via `cache_indirection` | kv-allocator, sampler | per-beam KV path indirection; C++ KVCM V2 beam search | dense models | decode | n/q | Yes | v1.4.0rc0 (#17306) | [ATT][REL] |
| 86 | Packed/unpadded tensors (remove_input_padding) | executor | no padding across batch | any | both | n/q | Yes | Stable | [ATT] |
| 87 | PDL (programmatic dependent launch) on decode edges | executor | overlaps kernel prologue with predecessor epilogue | Hopper+ | decode | n/q | Yes | v1.3.0rc29 (#19446) | [REL][TC] |
| 88 | SageAttention kernels (SM90) — quantized attention (VisualGen) | attn-prefill | INT8/FP8 smoothed QK attention for DiT | Hopper (updated); SM120 Sage via FlashInfer | DiT prefill | blog28 | No | v1.4.0rc0 (#18496) | [REL][TB] |
| 89 | Skip-Softmax Attention (BLASST) | attn-prefill, attn-decode | dynamically skips softmax tiles below threshold; `threshold_scale_factor` or `target_sparsity` | TRT-LLM backend only | long context | blog16/28 (not opened) | No (approximate; sparsity "calibration guidance") | Prototype API | [SPA][TB] |
| 90 | RocketKV (`rocket`) | kv-select, attn-decode | dense prefill, evict prompt KV beyond budget, decode attends Top-K pages | CC ≥ 10.0 (Blackwell); block reuse & chunked prefill off; no disagg | long-context decode | n/q | No | Prototype, training-free | [SPA] |
| 91 | DeepSeek Sparse Attention — DSA indexer + Top-K + sparse MLA (`dsa`) | kv-select, attn-decode | learned MQA indexer scores history, Top-K token select, sparse MLA | DeepSeek-V3.2/GLM-5.x; TRT-LLM backend; optional GVR Top-K on Blackwell (index_topk 512/1024/2048) | both | blog15 | No (model-native) | Prototype API, model-native | [SPA] |
| 92 | DeepSeek-V4 hybrid sparse attention (`deepseek_v4`: SWA + indexer over 4× compressed + dense over 128× compressed/HCA) | kv-select, attn-decode | three-branch attention over compressed histories; compressorKernels | Hopper (needs fp8_ds_mla) or Blackwell; blocks 128/256; beam 1 | long context | blog26 | No (model-native) | Prototype API | [SPA][KT] |
| 93 | CSA2 (DeepSeek-V4.1) | kv-select, attn-decode | SWA + compressed history with shared KV/index; decoder bounded replay | FlashMLA sm_90 / trtllm-gen sm_100 / FlashInfer sm_120; no PP/CP | long context | n/q | No (replay approximation on by default) | Prototype, v1.4.0rc1 | [SPA] |
| 94 | MiniMax-M3 block-sparse GQA (`minimax_m3`; `msa` impl) | kv-select, attn-decode | index branch scores blocks; forced initial+local; Top-K; sparse GQA | `msa`: Blackwell DC, block 128; `triton` legacy | both | FlashInfer MSA: 2.45× geomean over baseline, 11 B200 cases [FR] | No | Prototype | [SPA] |
| 95 | Generic block-sparse attention kernels (BSR / bitmask, block-summary proxies) | attn-prefill, attn-decode | attend to selected KV blocks per head; proxies for unselected | Blackwell DC; head dim 128 | both | n/q | No (proxy approx.) | Kernel capability | [SPA] |
| 96 | Token-sparse MQA/GQA & page-sparse MHA kernels (NVRTC JIT) | attn-decode | attention over precomputed token/page lists | Blackwell DC | decode | n/q | depends on selector | Kernel capability | [SPA] |
| 97 | Sparse MLA kernel (BF16; DeepSeek-V3.2 192/128, V4 512/512) | attn-decode | token-selected latent attention | Hopper → sm_121 | decode | n/q | depends | Kernel capability | [SPA] |
| 98 | GVR V2 — self-sampling exact Top-K (deterministic FP32; "guess-verify-refine") | kv-select, sampler | exact top-k over indexer scores with low overflow overhead | Blackwell | decode | blog29 | Yes (exact) | v1.4.0rc1 (#20100, #20290) | [TB][REL] |
| 99 | Indexer K-cache gather/scatter + indexerTopK kernels | kv-select | maintain indexer side-cache and top-k | DSA models | both | n/q | Yes | Stable | [KT] |
| 100 | GLM-5.3-Flash indexer scoring on BF16 tensor cores | kv-select | indexer logits via tensor cores | Blackwell | both | n/q | Yes | v1.4.0rc1 (#20251) | [REL] |
| 101 | MiniMax-M3 FP8 indexer kernel + select-blocks | kv-select | FP8 indexer scoring and block selection | Blackwell | both | n/q | No (FP8) | Stable | [KT] |
| 102 | Temporal-correlation index reuse for sparse attention (blog21; GVR index reuse for Qwen3.8) | kv-select | reuses Top-K selections across adjacent decode steps | DSA-style models | decode | blog21 (not opened) | No | Tech blog #21; v1.3.0rc28 (#19138) | [TB][REL] |
| 103 | Sparse KV cache config, tier-aware locking, sparse KV offload with GPU metadata publication | kv-tier, kv-select | offload sparse-selected KV with metadata for routing | KVCM V2 | long ctx | n/q | depends | v1.4.0rc0 (#19580, #19821, #19502) | [REL] |
| **TensorRT-LLM — KV cache system & compression** | | | | | | | | | |
| 104 | Paged KV cache (blocks, pools per window-size × head-count) | kv-allocator | fixed-size blocks from pool; `free_gpu_memory_fraction` 0.9 default | all | both | memory vs contiguous (n/q) | Yes | Stable | [KV][SCH] |
| 105 | MQA/GQA-aware KV layout | kv-allocator | blocks hold only per-group KV | GQA models | both | memory reduction (n/q) | Yes | Stable | [KV] |
| 106 | KV block reuse / prefix caching (radix tree; SHA-256 keys) | kv-allocator | share matched prefix blocks across requests | all | prefill (skips recompute) | n/q | Yes | Stable, default on | [KV] |
| 107 | Partial block reuse (`enable_partial_reuse`, `copy_on_partial_reuse`) | kv-allocator | reuse partially matched blocks | all | prefill | n/q | Yes | Stable, default on | [KV] |
| 108 | Prefix-aware scheduling (`enable_prefix_aware_scheduling`) | scheduler | defers duplicate first chunks; adjusts token budgets by reuse estimate | all | prefill | n/q | Yes | Stable, default on | [KV] |
| 109 | Prioritized LRU eviction + retention policies (`TokenRangeRetentionConfig`, `decode_retention_policy`) | kv-allocator | 0–100 priority, LRU tiebreak, durations | all | — | n/q | Yes | Stable | [KV] |
| 110 | SWA endpoint retention (`swa_endpoint_rewind_tokens`) | kv-allocator | priority 70 for SWA blocks near prompt end | KVCM V2, `all_reusable` | — | n/q | Yes | Prototype | [KV] |
| 111 | Configurable SWA retention priority / KV connector prefixes in V2 scheduler budgeting | kv-allocator, scheduler | budget accounting for connector-provided prefixes | KVCM V2 | — | n/q | Yes | v1.4.0rc0 (#17974, #19670) | [REL] |
| 112 | Host (CPU) KV offloading (`host_cache_size`, `cold_tier_offload_min_priority`) | kv-tier | evicted GPU blocks → pinned host, copied back on reuse | all | prefix-heavy, multi-turn | n/q | Yes | Stable | [KV] |
| 113 | Disk KV tier (KVCM V2) + progressive host-tier fill | kv-tier | disk as third tier; fills host tier progressively | KVCM V2 | long-lived sessions | n/q | Yes | v1.4.0rc1 (#19650); disk referenced in compression page | [REL][KC] |
| 114 | Tier-aware `pool_ratio` quotas per layer group | kv-tier | per-layer-group tier weights (hybrid Mamba) | hybrid models | — | n/q | Yes | Stable | [KV] |
| 115 | Variable attention window per layer (`max_attention_window` list) / V2 window derivation from `layer_types` | kv-allocator | separate pools for sliding vs full layers (GPT-OSS, Gemma3/4) | mixed-layer models | both | n/q | Yes | Stable | [KV] |
| 116 | KV cache manager V2 (`use_kv_cache_manager_v2`) | kv-allocator | new C++ manager; default for hybrid Mamba, DeepSeek-V4, GPT-OSS, Gemma3/4, Llama/Llama4; default with Python transceiver in 1.4.0rc1 | — | — | n/q | Yes | Default v1.4.0rc1 (#17495) | [KV][REL] |
| 117 | Branch snapshots (`enable_branch_snapshot`) | kv-allocator | keep SWA window at fork so reuse extends past shared prefix | KVCM V2 | multi-branch agents | n/q | Yes | Prototype | [KV] |
| 118 | Mamba state snapshots (`mamba_state_config` periodic/offset) | kv-allocator, recurrent-mixer | retain recurrent state at offsets for prefix reuse | hybrid SSM models | prefill | n/q | Yes | Stable | [KV] |
| 119 | Cache salting (`cache_salt`) + multimodal UUID keys (BLAKE3) | kv-allocator, os-config | tenant isolation of reuse; MM content hashing | all | — | — | Yes | Stable | [KV] |
| 120 | KV cache events — buffered (`get_kv_cache_events`, `/kv_cache_events`) | kv-allocator, executor | stored/removed/created/updated events for routers | all | — | — | Yes | Stable | [KV] |
| 121 | KV cache events — streaming (ZeroMQ PUB + replay ROUTER) with routing load | kv-allocator, executor | native event stream incl. load; best-effort, no PP/CP | all | — | — | Yes | Prototype → native v1.4.0rc1 (#19440) | [KV][REL] |
| 122 | KV cache connector API (Mooncake connector; KV cache lending API) | kv-tier, executor | pluggable external KV store / transfer backends | KVCM V2 | — | — | Yes | v1.4.0rc1 (#19171, #19722) | [FEAT][REL] |
| 123 | Mooncake store pool + V2 scheduler preemption | kv-tier, scheduler | distributed KV store with preemption-aware scheduling | Mooncake | — | — | Yes | v1.4.0rc0 (#19235) | [REL] |
| 124 | KV reuse tracked by source tier | kv-tier | telemetry/accounting per tier (GPU/host/disk/connector) | KVCM V2 | — | — | Yes | v1.3.0rc28 (#18583) | [REL] |
| 125 | Draft KV budget split weighted by per-request spend | kv-allocator, spec | allocates draft-model KV proportionally | spec decoding | decode | n/q | Yes | v1.4.0rc0 (#19744) | [REL] |
| 126 | Opt-in cached-KV token counts | executor | reports cached token count per request | all | — | — | Yes | v1.3.0rc29 (#19375) | [REL] |
| 127 | Partial KV copy kernel (kvCachePartialCopy) | kv-allocator | copy sub-block ranges (partial reuse / beams) | all | — | — | Yes | Stable | [KT] |
| 128 | NVFP4 cold-page compression (host/disk tiers; nvfp4ColdPageKernels) | kv-tier, kv-dtype | attention KV stored NVFP4 only in cold tiers; GPU stays runtime dtype; non-attention state lossless | Blackwell DC sm_100/103, Rubin; KVCM V2; host or disk cache > 0 | multi-turn / long sessions | more pages per cold byte, fewer bytes moved (n/q) | No (attention KV); non-attention state lossless | Supported; tested Qwen3/3.5, GLM-5.2, DeepSeek-R1/V4 (blog30) | [KC][TB] |
| 129 | RoPE-preserved cold compression (`skip_rope_quantization: true`) | kv-dtype | keep RoPE part FP8/BF16, rest NVFP4 | DeepSeek-V4, GLM-5, Qwen3.5 | — | none claimed (costs storage) | RoPE part yes | Exploratory | [KC] |
| 130 | 2FP4 RoPE residual (`nvfp4_residual_dim=64`, DeepSeek-V4 default) | kv-dtype | two-FP4 encoding of RoPE dims | DeepSeek-V4; blocks 128/256; not with fp8_ds_mla footer layout | — | n/q | No | Exploratory default | [KC] |
| 131 | Mixed-precision NVFP4 active KV (`dtype: mixed_nvfp4_*`, first/last N hot) | kv-dtype | NVFP4 history with BF16/FP8 head and tail tokens | Blackwell | long ctx decode | n/q (per-step conversion cost) | No | Prototype | [KV] |
| 132 | TriAttention (periodic token eviction + physical compaction during generation) | kv-select, kv-allocator | drops low-contribution tokens on a schedule using offline calibration | Blackwell DC/Rubin; KVCM V2; calibration file; not MLA/SSM/DSA/DeepSeek-V4 | long generation | lower KV memory and attention work (n/q) | No | Supported; tested Qwen3, GPT-OSS, Llama 3 | [KC] |
| 133 | One-compression-method-per-instance constraint | os-config | exclusivity rule | — | — | — | — | Documented | [KC] |
| 134 | Prefix-tokenization cache (MiniMax-M3 text prompts) | loader, executor | caches tokenizer output for repeated prefixes | any | prefill | n/q | Yes | v1.4.0rc0 (#19613); feature page | [FEAT][REL] |
| 135 | Unified multimodal encoder caching (fewer output copies) | kv-allocator, executor | caches vision-encoder outputs across requests | VLMs | prefill | n/q | Yes | v1.4.0rc1 (#17876, #19799) | [REL] |
| **TensorRT-LLM — scheduling, executor, runtime** | | | | | | | | | |
| 136 | In-flight (continuous, iteration-level) batching | scheduler | context and generation sequences share an iteration; packed inputs, context first | all | both | n/q | Yes | Stable | [SCH] |
| 137 | Generation-first scheduling + eviction on completion | scheduler | decode requests scheduled before new prefills; finished seqs freed before next iter | all | both | n/q | Yes | Stable | [SCH] |
| 138 | `max_num_tokens` budget (default 8192 since v0.11) | scheduler | caps packed tokens/iter; raises utilization to a plateau then hurts TTFT/TPOT; frees KV memory | all | both | n/q | Yes | Stable | [SCH] |
| 139 | `max_batch_size` / `max_seq_len` runtime tuning | scheduler | runtime caps tunable without rebuild | all | — | n/q | Yes | Stable | [SCH] |
| 140 | Capacity scheduler policies (GUARANTEED_NO_EVICT / MAX_UTILIZATION / STATIC_BATCH) | scheduler | KV-memory-aware admission; MAX_UTILIZATION allows preemption | all | — | n/q | Yes | **Not on fetched page** (deferred to "Additional Options"); known from LLM API — unverified this run | [SCH] |
| 141 | Overlap scheduler (CPU scheduling overlapped with GPU step; kept responsive during disagg KV receives) | scheduler, executor | hides host overhead by preparing iteration N+1 during N | all | decode | n/q | Yes | Stable; v1.4.0rc1 (#19857) | [FEAT][REL][FM] |
| 142 | Generation CUDA graphs (`cuda_graph_config`, capture sizes, padding optional) | executor | replay decode steps; capture list powers of 2 ≤128 then 256-steps to 3072 | all | decode | blog20 "Tuning CUDA Graph batch sizes for higher output throughput" | Yes | Stable | [TC][TB] |
| 143 | Piecewise CUDA graph (prefill/mixed; attention eager, rest captured) | executor | cuts launch overhead for host-bound prefill iterations; padding mandatory | not multi-engine (multimodal) | prefill small-token-count | n/q | Yes | Stable (`prefill_cuda_graph_backend: piecewise`) | [TC] |
| 144 | Breakable prefill CUDA graph (no torch.compile) | executor | capture model body without compile | no LoRA/multimodal/compile | prefill | n/q | Yes | Experimental; validated BF16 Qwen3.5 TP1, NVFP4 DeepSeek multi-GPU | [TC] |
| 145 | torch.compile backend (fullgraph tracing, vertical fusion) | executor, fused-elementwise | compiled decoder forward with ATen-IR fusion passes | ops must trace w/ fake kernels | both | n/q | Yes | Stable | [TC] |
| 146 | `compile_only_piecewise_graphs` | executor | compile only piecewise-eligible forwards → faster startup | piecewise backend; DecoderModelForCausalLM | startup | no perf impact measured Qwen3-8B FP8 TP1/2 [TC] | Yes | v1.4.0rc1 (#18366), opt-in | [TC][REL] |
| 147 | Auto multi-stream (`max_num_streams`; DAG + critical-path cost) | executor | overlaps independent ops (router GEMM ∥ SiLU-mul+FP4 quant) | any | both | n/q | Yes | Stable | [TC] |
| 148 | Re-inplace optimization | executor | removes auto_functionalize copies | torch.compile | both | n/q | Yes | Stable | [TC] |
| 149 | Autotuner (kernel/tactic selection; autotune forwards via compiled callable) | engine | per-shape kernel choice | all | — | n/q | Yes | Stable | [TC] |
| 150 | Runtime sleep / wakeup (release GPU memory, resume) | executor, os-config | frees weights/KV for co-location; wake fast | all | — | — | Yes | v1.4.0rc1 (#19016, #19604); sleep-wakeup.md | [FEAT][REL] |
| 151 | Runtime memory checkpointing | executor | snapshot/restore runtime memory | — | — | — | Yes | Prototype (feature page) | [FEAT] |
| 152 | Ray orchestrator | executor, os-config | Ray-based multi-process orchestration (MNNVL AR supported) | — | — | — | Yes | Prototype | [FEAT] |
| 153 | ModelExpress (MX) checkpoint loading | loader | streams weights GPU→GPU via NIXL/NVLink | NIXL | startup | Dynamo README: 7× faster startup DeepSeek-V3 H200 [DY] | Yes | Feature page | [FEAT][DY] |
| 154 | Checkpoint loading framework (HF/custom loaders, static FP8 Cosmos w/o requant) | loader | pluggable weight loaders/mappers | — | startup | — | Yes | Stable | [FEAT][REL] |
| 155 | Sub-agent routing | executor, scheduler | routes sub-agent calls within a serving instance (agentic workloads) | — | — | blog23/26 | Yes | Feature page | [FEAT][TB] |
| 156 | Disaggregated serving (context/generation servers; KV transceiver; Python transceiver default) | executor, parallelism | split prefill and decode fleets; KV transfer over UCX/NIXL/MPI | multi-GPU/node | both | blog05 | Yes | Stable; "Yes" with all spec modes, "No" with logits post-processor [FM] | [FEAT][TB] |
| 157 | LoRA / multi-LoRA (kernels/lora; DoRA scaling) | linear-gpu, executor | batched adapter GEMMs, per-request adapter | all | both | n/q | Yes | Stable; "Untested" with EP/Helix/ADP/disagg/spec [FM] | [FEAT][KT] |
| 158 | Guided decoding (XGrammar / llguidance; logitsBitmask kernel) | sampler | constrained JSON/grammar via token bitmask | all | decode | blog12 combining guided + speculative | Yes (constraint) | Stable | [FEAT][KT][TB] |
| 159 | Logits post-processor / post-processing hook | sampler, executor | user callbacks on logits / outputs | all | decode | — | Yes | Stable; "No" with disagg & spec [FM] | [FEAT][FM] |
| 160 | Torch sampler (batched; per-request seeds; fusedSampling kernels) | sampler | GPU batched top-k/top-p/temperature with seeds | all | decode | n/q; lower per-response streaming overhead v1.4.0rc1 (#20244) | Yes | Stable | [KT][REL][FM] |
| 161 | Beam search | sampler | multi-beam decoding with KV indirection | dense models (V2 C++) | decode | — | Yes | Stable | [REL] |
| 162 | Additional outputs (hidden states, logprobs, context logits) | executor | optional extra tensors per request | all | — | — | Yes | Feature page | [FEAT] |
| 163 | Embeddings (encoder-only models) | executor | BERT-class batched embedding serving | — | prefill | — | Yes | Feature page | [FEAT] |
| 164 | Startup-stage timing metrics + sampled iteration stats | executor | telemetry for tuning | all | — | — | Yes | v1.4.0rc0 (#18060, #19717) | [REL] |
| 165 | uvloop pinning for trtllm-serve streaming | os-config | restores streaming throughput | — | — | regression fix | Yes | v1.4.0rc1 (#19992) | [REL] |
| 166 | Inference-time compute implementation (blog13) | executor, spec | scaffolding for test-time compute (parallel sampling/verification) | — | — | blog13 (not opened) | — | Tech blog #13 | [TB] |
| 167 | Agentic serving: trace replay + job-level metrics (blog27); joint agent/TRT-LLM optimization (blog23) | executor | evaluation + co-design for multi-call agent jobs | — | — | blogs (not opened) | — | Tech blogs #23, #27 | [TB] |
| **TensorRT-LLM — speculative decoding** | | | | | | | | | |
| 168 | Draft-Target (two-model) | spec | small draft model proposes, target verifies; tokenizer must match | any | low batch only ("speedups only observable at low batch sizes") | n/q | Yes (distribution-preserving) | Stable; examples set `disable_overlap_scheduler=True` | [SD] |
| 169 | EAGLE-3 (linear) | spec | trained EAGLE-3 head on target hidden states | EAGLE-3 checkpoints (Llama 3/4, gpt-oss-120b, Qwen3); PyTorch backend supports **only** Eagle3 | low batch | blog06 Llama4 Maverick Eagle guide; blog11 GPT-OSS Eagle3 | Yes | Stable | [SD][TB] |
| 170 | EAGLE-3 dynamic tree (`use_dynamic_tree`, `dynamic_tree_max_topK`, `max_total_draft_tokens`) | spec | multi-candidate tree per layer | not SWA or MLA models (DeepSeek, gpt-oss) | low batch | n/q | Yes | Stable; Torch sampler greedy-only for MTP dynamic tree [FM] | [SD][FM] |
| 171 | MTP (native multi-token-prediction heads) | spec | uses model's own MTP modules; `num_nextn_predict_layers` = `max_draft_len` | DeepSeek, Qwen3.8/3.5 MoE, Step-3.x, Nemotron-H (quantized MTP heads rc0) | low batch | blog02 DeepSeek-R1 MTP; draft-loop overhead reduced rc0 (#19785) | Yes | Stable | [SD][TB][REL] |
| 172 | MTP relaxed acceptance (`relaxed_topk`, `relaxed_delta`) | spec | accept near-top drafts during reasoning "thinking" phase | reasoning models | low batch | n/q | **No** (changes distribution) | Stable, optional | [SD] |
| 173 | PARD (parallel draft, mask tokens) | spec | target-independent draft predicts K tokens in one pass | PARD draft model | low batch | n/q | Yes | Stable | [SD] |
| 174 | DFlash / DFlash 2 (target hidden states as cross-attn context) | spec | parallel multi-token draft conditioned on `target_layer_ids`; backends VANILLA (any), TRTLLM (sm_100/103), FA4 (sm_90) | DFlash draft model | low batch | n/q | Yes | Stable; DFlash 2 v1.3.0rc28 | [SD][REL] |
| 175 | LiLiCorr (DFlash scorer correlating per-position top-k) | spec | correlation scorer before verification; `max_draft_len ≤ block_size-1` | matching vocab/embedding width; non-causal ckpt | low batch | n/q (acceptance may drop with stochastic sampling) | Yes (target distribution preserved) | v1.4.0rc1 (#19856) | [SD][REL] |
| 176 | DSpark (MLA-backed standalone drafter; ragged verification w/ confidence policy + calibrated scoring; Helix verification) | spec | sparse-attention drafter for Kimi K3 / MLA models | MLA models; Helix groups | low batch | n/q | Policy-dependent | v1.3.0rc29 (#19040), v1.4.0rc1 (#19707, #19818, #20071) | [REL] |
| 177 | NGram / prompt-lookup (`NGram`) + auto-enablement heuristics (blog07) | spec | prefix→continuation map from prompt/output; no model | none | low batch, repetitive text | blog07 perf analysis | Yes | **Deprecated**, removal Jan 2027 → use SA | [SD][TB] |
| 178 | Suffix Automaton (SA) speculation (`SADecodingConfig`, `sa_spec_threshold`=4; `use_sa_spec` combinable with MTP/Eagle3/PARD) | spec | GPU suffix-automaton pattern matcher proposes drafts on exact repetition | none | low batch; code/agent loops | n/q ("very accurate on exact repetition") | Yes | Stable | [SD] |
| 179 | User-provided `Drafter` | spec | custom drafting hook | — | — | — | — | **Deprecated**, removal Jan 2027 | [SD] |
| 180 | One-model speculative acceptance with per-request seeds | spec, sampler | single-engine spec with seeded sampling | — | decode | — | Yes | v1.4.0rc1 (#19719, #20147) | [REL] |
| 181 | Medusa / ReDrafter / Lookahead (legacy TensorRT-engine backend) | spec | multi-head / RNN-draft / Jacobi lookahead speculation | TRT engine flow | low batch | ModelOpt: Medusa up to 1.9× Llama 3.1 [MO] | Yes | **Not in PyTorch-backend spec page**; legacy status unverified this run | [SD][MO] |
| 182 | Guided + speculative decoding combination (blog12) | spec, sampler | grammar mask applied to draft + verify | — | decode | blog12 (not opened) | Yes | Tech blog #12 | [TB] |
| **TensorRT-LLM — recurrent / hybrid mixers & misc kernels** | | | | | | | | | |
| 183 | Mamba/Mamba2 SSM kernels (selectiveScan, mamba2MTPSSMCache, mambaConv1d, causalConv1d) | recurrent-mixer | selective scan / SSD prefill + state update, conv1d, MTP-aware SSM cache | Nemotron-H, hybrid models | both | n/q | Yes | Stable | [KT] |
| 184 | KDA decode (kdaDecode; non-MTP decode kernel; prefill avoids cuBLAS fallback) | recurrent-mixer | Kimi Delta Attention recurrent decode | Kimi K3 | decode | n/q | Yes | v1.3.0rc29 (#18938), v1.4.0rc1 (#20084) | [KT][REL] |
| 185 | Kimi K3 attention-residual RMSNorm fusion (kimiK3AttnRes) | fused-elementwise | fuses residual + norm in K3 attention block | Kimi K3 | both | n/q | Yes | v1.3.0rc28 (#19182) | [KT][REL] |
| 186 | RG-LRU kernel (lruKernel) | recurrent-mixer | gated linear recurrence (Griffin/RecurrentGemma) | — | both | n/q | Yes | Stable | [KT] |
| 187 | mHC kernels (mhcKernels) | fused-elementwise | fused pre/post kernels for hyper-connection-style residual mixing (also in FlashInfer `mhc_pre_big_fuse`) | — | both | n/q | Yes | Present; semantics not documented in fetched pages | [KT][FD] |
| 188 | DeepSeek-V4 block table / Q-norm kernels; MiniMax-M3 paged attention for dense layers | attn-decode | model-specific paged/normalization kernels | Blackwell | both | n/q | Yes | v1.4.0rc1 (#20159) | [KT][REL] |
| 189 | DiT fused kernels (QK-norm+RoPE, gate+resid+norm+shift/scale, split-norm) and Cache-DiT | fused-elementwise, engine | VisualGen fusions and feature caching for diffusion transformers | DiT models | diffusion steps | blog25/28 | Cache-DiT: No | Beta (VisualGen) | [KT][FEAT] |
| 190 | VisualGen CUDA graphs / quantized attention / sparse attention (VSA, SOL) / parallel tiled VAE | engine, attn-prefill | video-generation specific runtime features | Blackwell for FP8 Sage | DiT | blog25 (NVL72 scaling), blog28 | mixed | Beta | [FEAT][REL] |
| 191 | Encode-Prefill-Decode (EPD) disaggregation with EVS for VLMs | executor, parallelism | separate encoder stage; efficient video sampling | Nemotron VL | prefill | n/q | Yes | v1.4.0rc0 (#19648, #19702) | [REL] |
| 192 | Delay-stream / global-timer / cumsum / lookup utility kernels | executor | timing, prefix-sum, embedding lookup | — | — | — | Yes | Present | [KT] |
| 193 | Logits bitmask kernel (logitsBitmask) | sampler | apply grammar bitmask to vocab logits | — | decode | — | Yes | Stable | [KT] |
| **FlashInfer — attention** | | | | | | | | | |
| 194 | Single/batch decode with paged KV (FA2-style, split-KV, `plan()/run()`) | attn-decode | decode attention with work-balanced split-KV; CUDA-graph-safe plan/run | Ampere (sm_80) → Blackwell; FA2 modules JIT-only for SM75 | decode | FlashInfer blog 2024 (not opened) | Yes | Stable | [FD][FR] |
| 195 | Batch prefill — ragged and paged (FA2 / FA3 on Hopper) | attn-prefill | variable-length prefill over ragged or paged KV | Ampere+; FA3 Hopper | prefill | — | Yes | Stable | [FD] |
| 196 | `trtllm_batch_decode/prefill_with_kv_cache` (TRT-LLM Gen backend) | attn-decode, attn-prefill | wraps TRT-LLM Gen Blackwell FMHA cubins | Blackwell DC (sm_100/103), Rubin (FP16 softmax, sparse compression v0.6.18) | both | — | Yes | Stable | [FD][FR] |
| 197 | XQA batch decode (`xqa_batch_decode_with_kv_cache`; ragged Q + per-row SWA for spec) | attn-decode | GQA decode kernel; spec-decode friendly | Hopper/Blackwell; SM120/121 (v0.6.16) | decode | — | Yes | Stable | [FD][FR] |
| 198 | cuDNN prefill/decode backends (`cudnn_batch_prefill_with_kv_cache`; auto-routes 2–4-token GQA decode rows to cuDNN on B200/B300) | attn-prefill, attn-decode | cuDNN SDPA engines via frontend[cutedsl] ≥1.31 | Blackwell DC; fp16/bf16, head 128; `FLASHINFER_DECODE_AUTO_CUDNN=0` to pin FA2 | spec verification decode | — | Yes | v0.7.2 (Oct 15 2026) | [FR] |
| 199 | Cascade / shared-prefix attention (`merge_state(s)`, recursive attention states) | attn-decode, kv-select | compute shared-prefix attention once, merge with per-request suffix | any | batched decode w/ common prefix | Cascade blog (Feb 2024; speedup not re-fetched) | Yes | Stable | [FD][FB] |
| 200 | MLA paged attention wrapper (`BatchMLAPagedAttentionWrapper`: TRTLLM-GEN / XQA / CuTe DSL backends; `backend="auto"` on SM100; `trtllm_prefill_with_kv_cache_mla`) | attn-decode, attn-prefill | DeepSeek-style latent attention; backend planned in `plan()` | Hopper → Rubin; MLA auto on SM103/107 (v0.7.2) | both | — | Yes | v0.7.1/0.7.2 | [FD][FR] |
| 201 | Hopper FP8 KV MLA (native SM90) | kv-dtype, attn-decode | FP8 latent cache on H100/H200 | Hopper | decode | "bit-identical to BF16 path" (v0.6.14) [FR] | claimed yes | v0.6.14 (Jul 2 2026) | [FR] |
| 202 | Sparse MLA (DeepSeek-V3.2/V4/V4.1, GLM-5.1; dual FP8+FP4 cache; GLM53-NoPE 528-byte rows) | kv-select, attn-decode | token-selected latent attention incl. NVFP4 cache on SM120/121 and `cake` backend on B200/B300 | Hopper → sm_121; opt-in | decode/prefill | 528-byte rows cut per-token KV storage 19.5% vs 656-byte (v0.7.1) [FR] | No (sparse) | v0.6.14 → v0.7.2 | [FR] |
| 203 | Attention scores / indexer logits (`fp8_paged_mqa_logits`, `fp4_paged_mqa_logits`, schedule) | kv-select | DSA indexer scoring kernels, paged FP8/MXFP4 | Blackwell | both | — | No (FP8/FP4) | v0.7.0 | [FD][FR] |
| 204 | Block-sparse / variable-block-sparse attention (`BlockSparseAttentionWrapper`, VSA, SM120 Sage block-sparse, proxy-compensated Sol-Attn) | attn-prefill | attention over BSR block masks; video sparse attention; FP8 Sage | SM100/103, SM120 | prefill (video/diffusion) | — | No | v0.6.15–0.7.0 | [FD][FR] |
| 205 | MSA ops (MiniMax-M3 sparse attention: `msa_sparse_attention`, `msa_topk_select`, `msa_proxy_score`) | kv-select, attn-decode | indexer + top-k + sparse GQA | SM100/103; SM120/121 (NVFP4 indexer, FP8/NVFP4 KV) | both | 2.45× geomean over baseline, 11 B200 cases [FR] | No | v0.6.16–0.7.0 | [FD][FR] |
| 206 | POD-Attention (`PODWithPagedKVCacheWrapper`, batched) | attn-prefill, attn-decode | runs prefill and decode in one fused kernel (SM-level co-scheduling) | Ampere+ | mixed batches | — | Yes | Stable | [FD] |
| 207 | Cake FMHA (`cake_batch_decode/context_with_kv_cache`; Cake NVFP4 warp decode) | attn-decode, attn-prefill | CuTe DSL "cake" attention family | Blackwell DC | both | — | Yes | Experimental | [FD] |
| 208 | NVFP4 attention (SM120; score-slot reuse for Cosmos); asymmetric VO-split NVFP4 prefill (Gemma 4 qk512/vo256) | attn-prefill | 4-bit Q/K attention paths | sm_120/121, Blackwell DC | prefill | — | No | v0.6.18–0.7.0 | [FR] |
| 209 | CuTe DSL GQA decode with sliding window + attention sinks; HCA decode | attn-decode | Python-authored decode with SWA/sink support; DeepSeek-V4 HCA | Blackwell | decode | — | Yes | v0.6.15/0.6.18 | [FR][FD] |
| 210 | Logits soft-cap, custom masks, ALiBi, variable window — JIT attention variant templates | attn-prefill, attn-decode | customizable attention via JIT templates (v0.2 design) | any | both | FlashInfer 0.2 blog (Dec 2024) | Yes | Stable | [FB][FD] |
| 211 | Prims-TS open attention kernels (FMHA prefill/decode, MLA decode; paged block-sparse, variable window, unified plan/run) | attn-prefill, attn-decode | TRT-LLM-Gen-equivalent kernels as readable Python (CUTLASS DSL 4.7) | Blackwell DC (B200 accuracy targeted; B300 pending) | both | — | Yes | Experimental, v0.7.0 | [F7][FR] |
| 212 | FMHAv2 head_dim 256/512 + SWA on SM120/121 (Gemma 4) | attn-prefill | large-head-dim prefill | sm_120/121 | prefill | — | Yes | v0.6.14 | [FR] |
| 213 | Speculative-verification attention entry points (Kimi K3 spec query length ≤8; XQA ragged Q) | attn-decode, spec | multi-query-row decode for draft verification | Blackwell | decode | — | Yes | v0.6.16–0.7.0 | [FR] |
| 214 | Green contexts (`flashinfer.green_ctx`; uGPU localization; CUTLASS ex. 95) | os-config, executor | partition SMs for co-scheduled prefill/decode or comm/compute | Hopper+ | mixed | — | Yes | Stable | [FD][CX] |
| **FlashInfer — sampling, norm, RoPE, activation, paging, quantization, top-k** | | | | | | | | | |
| 215 | Sorting-free top-k / top-p / min-p sampling (`*_sampling_from_probs`, renorm/mask helpers) | sampler | rejection-based sampling without full sort | any | decode | Sorting-free sampling blog (Mar 10 2025; numbers not re-fetched) | Yes (exact distribution) | Stable | [FD][FB] |
| 216 | `chain_speculative_sampling` | sampler, spec | verifies draft chain with target probs in one kernel | any | decode | — | Yes | Stable | [FD] |
| 217 | `LogitsPipe` logits processor (Temperature/Softmax/TopK/TopP/MinP/Sample) | sampler | composable fused logits pipeline | any | decode | — | Yes | Stable | [FD] |
| 218 | Cake sampling (top-k/top-p + routing) | sampler, moe-gpu | CuTe DSL sampling/routing kernels | Blackwell | decode | — | Yes | Experimental | [FD] |
| 219 | `top_k` / `top_k_varlen` (radix, guess-verify-refine, CUB fallback; page-table/ragged transforms) | kv-select, sampler | exact top-k for indexer scores; Rubin support | Blackwell/Rubin | decode | — | Yes | v0.6.18–0.7.0 | [FD][FR] |
| 220 | RMSNorm family (`rmsnorm`, `fused_add_rmsnorm`, `gemma_rmsnorm`, `layernorm`, `fused_qk_rmsnorm_rope`, `gated_residual_rmsnorm`, Cake training RMSNorm) | fused-elementwise | fused residual+norm (+RoPE) | any | both | — | Yes | Stable | [FD] |
| 221 | RMSNorm + FP4 quantization (cute_dsl) | fused-elementwise | norm then NVFP4 quant | Blackwell | both | — | No | Experimental | [FD] |
| 222 | RoPE kernels (`apply_rope`, `apply_llama31_rope`, cos/sin cache, `rope_quantize_fp8`, `mla_rope_quantize_fp8`) | fused-elementwise | RoPE + optional FP8 quant fused | any | both | — | Yes/No | Stable | [FD] |
| 223 | Activation kernels (`silu_and_mul`, `gelu_tanh_and_mul`, `gelu_and_mul`, `silu_and_mul_mxfp8_quantize`) | fused-elementwise | gated activation (+MXFP8 quant) | any; MXFP8 on Blackwell/Rubin | both | — | Yes/No | Stable | [FD][FR] |
| 224 | Paged KV append (`append_paged_kv_cache`, MLA variant, NVFP4 append; fused RoPE+NVFP4 quant+paged insert writers) | kv-allocator, kv-dtype | write new KV into pages incl. 4-bit | any; NVFP4 on Blackwell | both | — | Yes/No | Stable / v0.7.2 | [FD][FR] |
| 225 | Quantization kernels (FP8, NVFP4, MXFP4, MXFP8; NVFP4 KV; packbits; block-scale interleave; `NVFP44Over6Config`) | quant-recipe, fused-elementwise | activation/weight quant + layout prep | Blackwell for FP4 | both | — | No | Stable; NVFP44Over6 v0.7.1 | [FD][FR] |
| 226 | MLA KV pack / concat (`concat_mla_k`, `concat_mla_kv_quant_fp8`) | fused-elementwise | pack latent + RoPE parts with FP8 | Hopper+ | both | — | No | Stable | [FD] |
| **FlashInfer — GEMM, MoE, communication** | | | | | | | | | |
| 227 | Dense GEMM (`mm_bf16`, `mm_fp8`, `mm_fp4`, `mm_mxfp8`, `gemm_fp8_nt_groupwise`, `fp8_linear`, `fp4_linear`) with autotuned backends (CUTLASS, cuDNN, CuTe DSL, TRT-LLM Gen) | linear-gpu | multi-backend quantized GEMM | FP4: Blackwell; FP8 groupwise: Hopper/Blackwell (known-issue M≤32 on SM100/103) | both | — | Yes | Stable | [FD][FR] |
| 228 | `tgv_gemm_sm100` (small-M GEMM) / `tinygemm_bf16` | linear-gpu | latency GEMM for decode M | Blackwell / any | decode | — | Yes | Stable | [FD] |
| 229 | `mm_bf16_fp4` W4A16 GEMM (cuDNN + CuTe DSL) | linear-gpu, quant-recipe | FP4 weight-only GEMM | SM100/103, SM120/121 | decode | — | No | v0.6.14/0.6.18 | [FR] |
| 230 | SVDQuant NVFP4 GEMM (CUTLASS) | linear-gpu, quant-recipe | low-rank + FP4 for diffusion/LLM | Blackwell DC, Rubin | both | — | No | v0.6.18 | [FR] |
| 231 | Fused-epilogue GEMMs (prims_ts): FP8/NVFP4 + SwiGLU; QKV + QK-norm + RoPE; `tmem_overlap` | linear-gpu, fused-elementwise | epilogue fusion of activation / norm / RoPE | B200/B300/Rubin | both | — | Yes | v0.7.2 opt-in | [FR] |
| 232 | Grouped GEMM (`grouped_mm_bf16/fp8/mxfp8/fp4`; Ampere/Hopper section; CuTe DSL grouped GEMM on Blackwell) | moe-gpu, linear-gpu | batched expert GEMM | Ampere → Blackwell | both | — | Yes | Stable | [FD] |
| 233 | Fused MoE — TRT-LLM Gen backend (FP8 shared-expert fusion; DeepSeek-V4 `hash_topk`; fused FP8 blockwise megakernel for batch ≤ 8) | moe-gpu | Blackwell generated MoE kernels | Blackwell DC | decode small batch for megakernel | — | Yes | v0.6.15 | [FD][FR] |
| 234 | Fused MoE — CUTLASS backend (new runners; BF16/W4A16 on SM90; GeluTanh) | moe-gpu | CUTLASS grouped-GEMM MoE | Hopper/Blackwell | both | — | Yes | v0.7.0 | [FR] |
| 235 | Fused MoE — cuTile backend (BF16, NVFP4) | moe-gpu | tile-IR authored MoE | Blackwell | both | — | Yes | v0.7.0 | [FR] |
| 236 | Fused MoE — CuTe DSL backends (SM90 BF16; NVFP4 W4A16 & W4A4 sharing one expert layout; GeGLU-tanh, SiTU) | moe-gpu | Python-authored MoE GEMMs | Hopper, Blackwell DC, SM120/121 | both | W4A16 1.50× W4A4 at 1 token (DeepSeek EP8 shape, B200); W4A4 wins large-batch prefill [FR]; DGX Spark W4A16 1.36× Marlin at 1 token (v0.7.1) | Yes | v0.6.18–0.7.1 | [FR] |
| 237 | Fused MoE — cuDNN Frost grouped GEMM (BF16, MXFP8, NVFP4, MXFP8-act/MXFP4-wt) | moe-gpu | cuDNN-backed experts on Rubin (BF16 also SM120); needs PyTorch 2.14, CUDA 13.4+ | Rubin sm_107, SM120 | both | — | Yes | v0.7.2 | [FR] |
| 238 | B12x fused MoE (`b12x_fused_moe`, `B12xMoEWrapper`; MXFP4/NVFP4/W4A16; cooperative persistent launches, tensor-core decode path) | moe-gpu | MoE for consumer/workstation Blackwell and DGX Spark | sm_120/121 | both | — | No (quant) | v0.6.17/0.6.18 | [FR] |
| 239 | AlphaMoE router + NVFP4/FP8 MoE | moe-gpu | alternative router/expert kernel set | Blackwell DC | both | — | Yes | Experimental | [FD] |
| 240 | MonoMoE (single-kernel block-FP8 MoE) | moe-gpu | one persistent kernel for whole MoE layer | Hopper sm_90a | both | — | Yes | Experimental | [FD] |
| 241 | BGMV multi-LoRA fused MoE | moe-gpu, linear-gpu | batched LoRA inside MoE experts | — | both | — | Yes | Experimental | [FD] |
| 242 | Prims-TS fused MoE (TRT-LLM Gen MoE as Python: BF16, FP8 per-tensor/block, NVFP4/MXFP4) | moe-gpu | open re-implementation of Gen MoE | Blackwell DC | both | — | Yes | Experimental v0.7.0 | [F7][FR] |
| 243 | Unified MoE API (`QuantConfig` separate weight/activation/output formats → W4A8 = MXFP4 w + MXFP8 a; shared experts; MXINT4; routing variants `FromLogits`, `TopKSigmoid`, per-tensor routed FP8, pre-routed FP4) | moe-gpu, engine | single API over all MoE backends and quant combos | per backend | both | — | depends | Official v0.7.0 | [FR] |
| 244 | `MoEEpLayer` split path (dispatch → expert → combine) over NCCL-EP / NIXL-EP / NVLink one-sided & two-sided / CAKE via `SplitConfig`; CUDA-graph capturable | parallelism, moe-gpu | composable EP communication backends; rank-mask fault tolerance | multi-GPU; NCCL-EP validated 4×B200 in vLLM | both | — | Yes | v0.6.15 → 0.7.2 | [FR][F7] |
| 245 | MegaMoE fused EP MoE (symmetric memory overlap; `deep_gemm_mega`, `nvfp4_cutedsl`, `mxfp8_cutedsl`; H100 BF16 up to 32 ranks; SM120 W4A8; Rubin NVFP4/MXFP8) | moe-gpu, parallelism | single fused kernel overlapping all-to-all with expert compute | Hopper (BF16), Blackwell DC, SM120, Rubin | both; prefill result in vLLM PR #54049 (unmerged) | 1.06–1.20× DeepSeek-V4-Flash, 1.19–1.32× V4-Pro (1×8 SM100) [F7]; CuTe DSL NVFP4 up to 1.89× `deep_gemm_mega` at 8192 tok/rank GB200 EP4 [FR] | Yes | v0.6.16 → 0.7.2 (MegaMoE blog Sep 22 2026) | [F7][FR] |
| 246 | Hopper EP MoE pull FP8 / push backends (fused FC1 epilogue drops ~1 GiB per-rank buffer) | moe-gpu, parallelism | H100 expert-parallel MoE | Hopper; CUDA 12.8+ | both | 562 TFLOPS/rank at 384-expert geometry [FR] | Yes | v0.6.18 | [FR] |
| 247 | Expert sharding via `expert_map` (SM120/121 W4A16) | placement | shard experts across small GPUs | sm_120/121 | both | — | Yes | v0.7.1 | [FR] |
| 248 | TRT-LLM AllReduce + fusion (AR + residual + RMSNorm + dynamic per-token FP8; `checkpoint_prepare/restore` workspaces) | parallelism, fused-elementwise | fused TP collective | NVLink | decode | — | Yes | Stable; v0.6.15 | [FD][FR] |
| 249 | MNNVL all-reduce fusion (CuTe DSL BF16; AR+residual+RMSNorm+optional MoE finalize) | parallelism | multi-node NVLink fused AR | GB300 TP8/TP16 profiles | decode | — | Yes | v0.6.18 | [FR] |
| 250 | `quantized_all_reduce` FP8 two-shot | parallelism | FP8 payload AR | SM90+, NVSwitch | decode | "halves transfer volume" [FR] | No | v0.6.16 | [FR] |
| 251 | PCIe IPC AllReduce / AllGather / ReduceScatter (`PcieIpcAllReduceWorkspace`, 2/4/8 ranks) | parallelism | collectives for PCIe-only boxes | PCIe GPUs | decode | — | Yes | v0.7.0 | [FR][FD] |
| 252 | vLLM custom AllReduce (comm) | parallelism | ported vLLM AR | NVLink | decode | — | Yes | Stable | [FD] |
| 253 | Fused all-gather + matmul (`backend="cake"`, up to TP8) | parallelism, linear-gpu | overlapped AG-GEMM | Blackwell | both | — | Yes | v0.7.0 | [FR] |
| 254 | Ulysses sequence-parallel all-to-all (fused NVLink-P2P over CUDA IPC, NCCL fallback) | parallelism | head-parallel long-context attention | multi-GPU | prefill | — | Yes | v0.6.17 | [FR] |
| 255 | MoE all-to-all, DCP all-to-all, mixed communication utilities | parallelism | dispatch/combine and decode-context-parallel A2A | multi-GPU | both | — | Yes | Stable | [FD] |
| 256 | Confidential-computing multicast-free IPC workspace | os-config, parallelism | AR fusion without multicast for CC mode | Hopper/Blackwell CC | decode | — | Yes | v0.6.16 | [FR] |
| **FlashInfer — recurrent mixers, diffusion, tooling** | | | | | | | | | |
| 257 | GDN (Gated DeltaNet) prefill/decode (`gdn_prefill`, `gdn2_prefill`, `gdp_prefill`, fused decode step; CuTe DSL rewrite; context-parallel delta rule; `chunk_gated_delta_rule2`) | recurrent-mixer | linear-attention mixer for Qwen3.5/3.6 | Hopper (prefill/CP), SM120 | both | ~20–25% GDN prefill speedup from CuTe DSL rewrite (v0.6.14) [FR] | Yes | v0.6.14–0.7.1 | [FR][FD] |
| 258 | KDA (Kimi Delta Attention) `kda_prefill` / `kda_decode` / `fused_kda_decode` / `packed_fused_kda_decode` (conv + KDA + gated RMSNorm fused) | recurrent-mixer | Kimi K3 recurrent mixer | SM100 (fused), SM10x packed, SM120 recurrent prefill | both | 1.33× vLLM fused kernel at 1 row on B200 (geomean 1.13×) [FR]; +1.8–3.3% output throughput w/ 6 drafts on 8×B300 (v0.7.1) | Yes | v0.6.18–0.7.1 | [FR] |
| 259 | cuDNN linear attention backend (KDA, GDN prefill on SM100) | recurrent-mixer | cuDNN-executed chunked delta rule | Blackwell DC; CUDA 12 builds | prefill | — | Yes | v0.7.1 | [FR] |
| 260 | Mamba kernels (`selective_state_update` w/ hardware stochastic rounding, `checkpointing_ssu`, `ssd_combined_fwd`, `replayssm_materialize`; CuTe DSL Mamba2 SSD on Rubin) | recurrent-mixer | SSM prefill/decode for Nemotron-H | B200/GB300, Rubin | both | — | Yes | v0.7.0–0.7.2 | [FR][FD] |
| 261 | mHC kernels (`mhc_post`, `mhc_pre_big_fuse(_with_prenorm)`) | fused-elementwise | fused hyper-connection pre/post ops | — | both | — | Yes | Stable | [FD] |
| 262 | Diffusion ops (MiniMax-H3 MXFP8 pre-attention, out-proj, varlen attention; DiT fused norm/residual) | fused-elementwise | video-DiT kernels | B200/B300 | diffusion | — | mixed | v0.7.0 | [FD][FR] |
| 263 | Autotuner v2 (`autotune_v2()`; eager vs CUDA-graph `MeasurementPolicy`; persistent cross-restart cache; untuned default as candidate) | engine | selects kernel configs with cached results | any | — | median regret 1.390%→0.418% eager (276 workloads), 0.0565%→0.00646% CUDA-graph (281); >100%-regret cases 59→0; vLLM B200 Qwen3-8B-FP8 TP2 tuning 128 s→1 s on restart, startup 410.7 s→165.3 s [F7][FR] | Yes | Opt-in v0.7.0 | [F7][FR] |
| 264 | Experimental namespace / `@flashinfer_experimental_api` / `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS` | engine | opt-in incubation with 4-week graduation target | — | — | — | — | v0.7.0 policy | [F7] |
| 265 | JIT compile + cubin/jit-cache wheels per architecture; CuTe DSL disk cache (3–30 ms reload) | engine, loader | runtime kernel generation with caching; wheel 1.94 GB→1.02 GB | — | startup | [FR] | Yes | v0.6.16–0.7.0 | [FR] |
| 266 | `fi_trace` / Trace Apply (operation schema extraction & replay) | engine | capture op schemas from a serving run to re-tune/replay | — | — | — | Yes | Docs | [FD] |
| 267 | FlashInfer-Bench (kernel benchmark/eval loop; `bench_gpu_time` CUDA events/graphs/CUPTI) | engine | standardized kernel benchmarking | — | — | blog Oct 21 2025 | — | Stable | [FB][FD] |
| 268 | JAX via TVM FFI | engine | FlashInfer kernels inside JAX (Gemma 3 tutorial) | — | — | — | Yes | Docs | [FD] |
| **cuDNN frontend — SDPA** | | | | | | | | | |
| 269 | Fused SDPA forward (FP16/BF16; MHA/MQA/GQA; BHSD/BSHD/interleaved/THD layouts) | attn-prefill, attn-decode | FlashAttention-class fused attention engines | sm_80+ (CUDA 12.x); head dim mult. of 8; Ampere/Ada prefill ≤256, decode ≤128; Hopper ≤256 incl. 192/128; B200/B300 prefill ≤256 decode ≤128 | both | — | Yes | Stable since cuDNN 8.9.3 | [CD] |
| 270 | FP8 SDPA fwd/bwd (E4M3/E5M2 with descale/scale/amax) | attn-prefill | 8-bit attention | Hopper or newer (datacenter Blackwell; **not** consumer Blackwell); head dim mult. of 16; fwd causal+padding, bwd causal only | prefill | — | No | 9.1+ (bottom-right causal FP8 on SM100+ in 9.7; deterministic FP8 Blackwell 9.19) | [CD] |
| 271 | Paged attention (K/V page tables; packed page tables; optional max KV len) | attn-decode, kv-allocator | non-contiguous KV via block tables | sm_80+; forward only | decode | — | Yes | 9.5 (paged), 9.7 (paged+ragged), 9.10.2 (packed tables) | [CD] |
| 272 | Ragged / packed (THD) sequences, `cu_seq_len`, ragged offset multiplier, `max_total_seq_len` | attn-prefill | variable-length batches without padding | sm_80+ (Ampere/Ada ragged in 9.18); GQA ragged 9.6 | prefill | — | Yes | 9.6–9.24 | [CD] |
| 273 | Causal (top-left / bottom-right), sliding-window (left bound), band mask (left+right), padding mask | attn-prefill, attn-decode | mask variants in-kernel | sm_80+ | both | — | Yes | 9.2 (SWA), 9.3 (BR causal) | [CD] |
| 274 | Additive bias (broadcast), ALiBi (right bound 0), 128×128 block mask (UNIFIED) | attn-prefill | bias/mask variants | sm_80+; block mask UNIFIED | prefill | — | Yes | 8.9.6 / 9.14 | [CD] |
| 275 | Softcap, `score_mod` callback (fwd/bwd), attention sinks | attn-prefill, attn-decode | arbitrary score modification and sink tokens | sm_80+; sinks 9.13+ | both | — | Yes | 9.13 | [CD] |
| 276 | UNIFIED SDPA implementation (vs COMPOSITE/AUTO); unfused FMA softmax on SM100; FP8 output as FP16/BF16 on Blackwell | attn-prefill | newer unified engine | 9.13.1+; SM100 unfused FMA 9.21 | both | — | Yes | 9.13.1+ | [CD] |
| 277 | FROST engines (cc 10.7 / Rubin): softmax precision, pre-folded scale, fused sigmoid epilogue gate (d=256), base-2 stats | attn-prefill, fused-elementwise | Rubin-specific SDPA engines with fused gating | Rubin sm_107 | both | — | Yes | 9.27 | [CD] |
| 278 | DeepSeek head-dim configs (d_qk=192, d_v=128; 256/256 bwd on Blackwell) | attn-prefill | MLA-shaped attention | Hopper, Blackwell | both | — | Yes | 9.11/9.19 | [CD] |
| 279 | Dropout (Philox / custom mask / RNG dump), `generate_stats`, QKClip stats, deterministic modes | attn-prefill | training-side features also usable for stats | sm_80+; deterministic bwd SM100 9.19+ | — | — | Yes | 9.x | [CD] |
| 280 | Execute-time shape overrides | executor | run graph at smaller extent than declared (avoid re-plan) | tile-aligned lengths | both | — | Yes | Experimental | [CD] |
| **CUTLASS 3.x/4.x examples relevant to LLM inference** | | | | | | | | | |
| 281 | Hopper warp-specialized GEMM / collective builder / epilogue swizzle (ex. 48–50) | linear-gpu | TMA + WGMMA persistent GEMM | Hopper sm_90a | both | — | Yes | Stable | [CX] |
| 282 | Hopper FP8 WS GEMM (54); FP8 blockwise-scaling GEMM (67) and grouped (68) | linear-gpu, moe-gpu | FP8 dense/grouped with 128-block scales (DeepSeek recipe) | Hopper | both | — | No | Stable | [CX] |
| 283 | Hopper mixed-input GEMM (55: W4A16/W8A16 with scales/zeros) and grouped mixed-dtype (69) | linear-gpu, moe-gpu | weight-only quant GEMM, dequant in mainloop | Hopper | decode | — | No | Stable | [CX] |
| 284 | Hopper grouped GEMM (57) / pointer-array batched (56) | moe-gpu | MoE expert GEMMs with per-group problem sizes | Hopper | both | — | Yes | Stable | [CX] |
| 285 | Hopper GEMM with top-k + softmax epilogue (61) | moe-gpu, fused-elementwise | router GEMM fused with top-k/softmax | Hopper | both | — | Yes | Stable | [CX] |
| 286 | Hopper GEMM with weight prefetch (63) | linear-gpu | prefetch weights to hide latency at small M | Hopper | decode | — | Yes | Stable | [CX] |
| 287 | Stream-K GEMM (47 Ampere, 74 Blackwell) | linear-gpu | work decomposition fixing tail effects | Ampere / Blackwell | both | — | Yes | Stable | [CX] |
| 288 | Ada FP8 GEMM (58), Ada FP8 grouped (64), Ada FP8 blockwise (94) | linear-gpu, moe-gpu | FP8 on sm_89 | Ada sm_89 | both | — | No | Stable | [CX] |
| 289 | Hopper sparse GEMM (62), Blackwell sparse (83), narrow-precision sparse (84), GeForce sparse (80) | linear-gpu | 2:4 structured sparsity GEMM | Hopper / Blackwell DC / sm_120 | both | — | No (needs sparse weights) | Stable | [CX] |
| 290 | Blackwell GEMM + collective builder + preferred cluster (70, 71, 73) | linear-gpu | UMMA/TMEM persistent GEMM | Blackwell DC sm_100a | both | — | Yes | Stable | [CX] |
| 291 | Blackwell narrow-precision block-scaled GEMM (72: NVFP4/MXFP4/MXFP6/MXFP8) | linear-gpu | FP4/FP6/FP8 block-scaled tensor-core GEMM | Blackwell DC | both | — | No | Stable | [CX] |
| 292 | Blackwell blockwise FP8 GEMM (81) and GeForce blockwise (87) | linear-gpu | DeepSeek-style blockwise FP8 on Blackwell | sm_100 / sm_120 | both | — | No | Stable | [CX] |
| 293 | Blackwell grouped GEMM (75), Blackwell MoE GEMM (92) | moe-gpu | expert GEMMs on Blackwell | Blackwell DC | both | — | Yes | Stable | [CX] |
| 294 | Blackwell mixed-dtype GEMM (86) | linear-gpu | W4A16-style on Blackwell | Blackwell DC | decode | — | No | Stable | [CX] |
| 295 | Blackwell GeForce GEMM (79) | linear-gpu | sm_120 (consumer) GEMM path | sm_120 | both | — | Yes | Stable | [CX] |
| 296 | SM103 FP4 "ultra" GEMM (89) and grouped (90) | linear-gpu, moe-gpu | B300-specific FP4 throughput path | sm_103 (B300) | both | — | No | Stable | [CX] |
| 297 | FP4 GEMV (91) | linear-gpu | decode-M=1 FP4 kernel | Blackwell | decode batch 1 | — | No | Stable | [CX] |
| 298 | Blackwell FMHA (77), Hopper FMHA (88), Blackwell low-latency GQA (93), Ampere fused MHA (41) | attn-prefill, attn-decode | CUTLASS reference attention kernels incl. decode-latency GQA | per arch | both | — | Yes | Stable | [CX] |
| 299 | Distributed GEMM (65 Hopper, 82 Blackwell) | parallelism, linear-gpu | TP GEMM with fused communication | multi-GPU NVLink | both | — | Yes | Example | [CX] |
| 300 | GEMM with green context (95) | os-config, linear-gpu | SM-partitioned GEMM | Blackwell | — | — | Yes | Example | [CX] |
| 301 | Emulated BF16x9 GEMM (78) | linear-gpu | FP32-accuracy via BF16 tensor cores | Blackwell | — | — | Yes (approx FP32) | Example | [CX] |
| 302 | CuTe DSL (examples/python; CUTLASS DSL 4.7 used by FlashInfer Prims-TS and TRT-LLM CuTe DSL MoE/MLA) | engine | Python kernel authoring with JIT to CUDA | Hopper/Blackwell/Rubin | — | — | Yes | Stable (4.x) | [CX][F7] |
| **NVIDIA Dynamo** | | | | | | | | | |
| 303 | Disaggregated prefill/decode (separately scalable pools; SGLang/TRT-LLM/vLLM) | executor, parallelism | split-phase serving | multi-GPU | both | 7× throughput/GPU DeepSeek-R1 GB200 NVL72 vs B200 (InferenceX); "750×" GB300 (InferenceXv2, as printed) [DY] | Yes | Dynamo 1.0 production | [DY] |
| 304 | KV-aware router (`--router-mode kv`; cost = prefill(overlap × `kv-overlap-score-weight`, default 1.0) + decode load; `router-temperature` softmax, default 0) | scheduler, kv-select | routes to worker maximizing prefix overlap vs load | any backend | TTFT-sensitive | 2× faster TTFT Qwen3-Coder 480B (Baseten) [DY] | Yes | Stable | [DR][DY] |
| 305 | Router radix tree fed by KV events; snapshot to NATS object store at 1,000,000 msgs | scheduler | global prefix index from worker events | KV events enabled | — | — | Yes | Stable | [DR] |
| 306 | ApproxKvIndexer (`--no-kv-events`; TTL 120 s; no state persistence) | scheduler | predicts cache state from routing decisions | any | — | — | Yes (approximate index) | Stable | [DR] |
| 307 | Router modes round-robin / random / kv / direct; Gateway API (GAIE EPP) K8s plugin | scheduler | routing topologies | — | — | — | Yes | Stable | [DY][DR] |
| 308 | SLA-based Planner (profiling + autoscaling prefill/decode pools) | scheduler, os-config | scales pools to TTFT/ITL targets at min TCO | K8s | — | 80% fewer SLA breaches, 5% lower TCO (Alibaba APSARA 2025) [DY] | Yes | Stable | [DY] |
| 309 | AISimulate + DGDR zero-config deploy (model+hardware+SLA YAML) | os-config | offline config search then deploy | K8s | — | — | Yes | Beta | [DY] |
| 310 | KV Block Manager (KVBM: GPU→CPU→SSD→remote; S3/Azure blob tiers) | kv-tier, offload-scheduler | multi-tier KV offload | TRT-LLM, vLLM (SGLang in progress) | multi-turn | — | Yes | Stable | [DY] |
| 311 | Global KV events (cluster-wide cache visibility) | kv-allocator, scheduler | event bus for routers | — | — | — | Yes | Stable | [DY] |
| 312 | Multimodal E/P/D disaggregation with embedding cache | executor | separate encoder stage + cache | VLMs | prefill | 30% faster TTFT on image workloads [DY] | Yes | New in 1.0 | [DY] |
| 313 | Agentic hints (priority, expected output length, speculative prefill, session metadata; SGLang sub-agent KV isolation) | scheduler | per-request scheduling hints | — | — | — | Yes | New in 1.0 | [DY] |
| 314 | ModelExpress weight streaming (NIXL/NVLink GPU→GPU) | loader | cold-start acceleration | NIXL | startup | 7× faster startup DeepSeek-V3 H200 [DY] | Yes | Stable | [DY] |
| 315 | Grove (topology-aware gang scheduling on NVL72) | os-config | K8s operator for rack-scale placement | GB200 NVL72 | — | — | Yes | Stable | [DY] |
| 316 | Fault tolerance (canary health checks, in-flight request migration) | executor | migrates requests on worker failure | — | — | — | Yes | Stable | [DY] |
| 317 | Service discovery (K8s CRDs/EndpointSlices, file-based, etcd/NATS) | os-config | — | — | — | — | — | Stable | [DY] |
| **ModelOpt (TensorRT Model Optimizer)** | | | | | | | | | |
| 318 | PTQ recipes (FP8, NVFP4, MXFP4, INT4 AWQ, W4A8; export to TRT-LLM/vLLM/SGLang HF checkpoints) | quant-recipe | calibrate and export quantized checkpoints | deploy arch per TRT-LLM matrix (#1–#9) | — | "2×–4× compression preserving quality" [MO]; Qwen3.6 W4A4: 1.30× vLLM throughput vs BF16, 3.1× smaller | No | Stable (Oct 15 2026 Nemotron 3.5 NVFP4 ckpt) | [MO] |
| 319 | NVFP4 local-Hessian weight scales | quant-recipe | Hessian-aware block-scale selection | Blackwell deploy | — | "improved NVFP4 accuracy" (no figure) | No | Sep 9 2026 | [MO] |
| 320 | NVFP4 scale learning | quant-recipe | trains block scales to recover accuracy | Blackwell deploy | — | n/q | No | Sep 16 2026 | [MO] |
| 321 | QAT (quantization-aware training) | quant-recipe | fine-tune with fake-quant | — | — | n/q | No | Stable (Sep 2025 blog) | [MO] |
| 322 | QAD (quantization-aware distillation) | quant-recipe | distill to recover NVFP4/W4A4 loss | — | — | n/q | No | Oct 15 2026 | [MO] |
| 323 | AutoQuantize (mixed-precision per-layer search) | quant-recipe, placement | automatic per-layer format assignment under constraint | — | — | n/q | No | Aug 24 2026 | [MO] |
| 324 | AutoQuant KV (per-layer KV format search + GEMM composition) | kv-dtype, quant-recipe | chooses FP8/NVFP4 KV per layer | Blackwell for NVFP4 KV | decode | Qwen3-8B case study (no figures) | No | Oct 8 2026 | [MO] |
| 325 | NVFP4 KV-cache export (`--kv_cache_quant nvfp4` with `--quant fp8`) | kv-dtype | produces checkpoints TRT-LLM needs for active NVFP4 KV | Blackwell DC | decode | — | No | Stable | [Q][MO] |
| 326 | Minitron pruning (+ distillation) | placement, engine | depth/width pruning then distill | — | — | Bielik Minitron 7B: 33% smaller, 50% faster, 90% quality; Nemotron-3-Nano prune+distill+FP8: 2.6× vLLM throughput, 2.6× memory [MO] | No | Stable | [MO] |
| 327 | Puzzletron (heterogeneous pruning / NAS for LLM & VLM) | placement, engine | per-block architecture search | — | — | n/q | No | May 13 2026 | [MO] |
| 328 | Speculative-decoding head training (Medusa; EAGLE-3 training in `examples/speculative_decoding` — EAGLE3 **not named on README**) | spec | train draft heads for TRT-LLM/vLLM | — | low batch | Medusa up to 1.9× Llama 3.1 [MO] | Yes (at inference) | Stable | [MO] |
| 329 | 2:4 sparsity | quant-recipe, linear-gpu | structured sparse weights | Ampere+ sparse tensor cores | both | n/q | No | Stable | [MO] |
| 330 | Diffusers / ONNX / Windows (RTX) quantization | quant-recipe | FP8/INT8 for diffusion & ONNX RT | RTX / Ada+ | — | Adobe: 60% lower diffusion latency, 40% lower TCO [MO] | No | Stable | [MO] |
| **NIXL** | | | | | | | | | |
| 331 | NIXL agent P2P transfer abstraction (memory: CPU/GPU; storage: file/block/object) with ETCD metadata | kv-tier, executor | one API for KV/weight movement across heterogeneous paths | Linux only; CUDA 12/13; GDRCopy "necessary for maximum performance" | disagg KV transfer, offload | no numbers on README (NIXLBench/KVBench provided) | Yes | Stable (UCX 1.23 tested) | [NX] |
| 332 | NIXL backends: UCX (primary; also ROCm), GDS/GDS_MT (cuFile), GPUNETIO (DOCA), LIBFABRIC, POSIX, OBJ, AZURE_BLOB, HF3FS, MOONCAKE, GUSLI, UCCL | kv-tier | pluggable transports/storage | per plugin deps | — | Dynamo news: 19× faster TTFT Dell PowerScale with NIXL (third-party) [DY] | Yes | Stable (plugin maturity varies) | [NX] |
| 333 | NIXL-EP (MoE dispatch/combine over NIXL; rank-mask fault tolerance; CUDA-graph) | parallelism, moe-gpu | EP communication backend in FlashInfer `MoEEpLayer` | multi-node | both | — | Yes | FlashInfer v0.6.15–0.7.2 | [FR] |
| 334 | NIXLBench / KVBench | engine | transfer benchmarking tools | — | — | — | — | Stable | [NX] |
| **kvpress (NVIDIA; HF-transformers KV compression "presses")** | | | | | | | | | |
| 335 | RandomPress | kv-select | random scores (baseline) | HF transformers; any GPU | prefill-time eviction | — | No | Stable | [KP] |
| 336 | KnormPress | kv-select | inverse key-norm scoring | — | — | — | No | Stable | [KP] |
| 337 | SnapKVPress | kv-select | attention from last few queries | — | — | — | No | Stable | [KP] |
| 338 | ExpectedAttentionPress | kv-select | expected attention during generation | — | — | — | No | Stable | [KP] |
| 339 | StreamingLLMPress | kv-select | keep initial + recent tokens | — | — | — | No | Stable | [KP] |
| 340 | TOVAPress | kv-select | last-query attention averaged across heads | — | — | — | No | Stable | [KP] |
| 341 | ObservedAttentionPress | kv-select | average attention observed in prefill | needs eager attention | — | — | No | Stable | [KP] |
| 342 | QFilterPress | kv-select | SVD projection of queries approximates attention | — | — | — | No | Stable | [KP] |
| 343 | PyramidKVPress | kv-select, placement | more budget to lower layers | — | — | — | No | Stable | [KP] |
| 344 | LagKVPress | kv-select | lag-relative KV info; query/attention-free | — | — | — | No | Stable | [KP] |
| 345 | KeyDiffPress | kv-select | evict by key similarity | — | — | — | No | Stable | [KP] |
| 346 | NonCausalAttnPress | kv-select | non-causal chunked attention scores | — | — | — | No | Stable | [KP] |
| 347 | LeverageScorePress | kv-select | approximate statistical leverage (preserves outliers) | — | — | — | No | Stable | [KP] |
| 348 | CompactorPress | kv-select | blends NonCausalAttn + Leverage by ratio | — | — | — | No | Stable | [KP] |
| 349 | CURPress | kv-select | CUR decomposition pruning of K and V | — | — | — | No | Stable | [KP] |
| 350 | KVzapPress / FastKVzipPress / KVzipPress | kv-select | KVzip context-reconstruction redundancy (exact: multiple forward passes; zap/fast: surrogate/gating) | — | — | — | No | Stable | [KP] |
| 351 | CapPress | kv-select | query-aware capacity via log-det leverage proxy | — | — | — | No | Stable | [KP] |
| 352 | DropKVPress | kv-select | output deviation if removed | — | — | — | No | Stable | [KP] |
| 353 | ThinKPress | kv-dtype, kv-select | compress key **channel** dimensions | — | — | — | No | Stable | [KP] |
| 354 | SimLayerKVPress | kv-select, placement | StreamingLLM on "lazy" layers | — | — | — | No | Stable | [KP] |
| 355 | DuoAttentionPress | kv-select, placement | full cache for retrieval heads, streaming for others | needs head labels | — | — | No | Stable | [KP] |
| 356 | FinchPress | kv-select | SnapKV-like with dynamic window + key re-rotation | — | — | — | No | Stable | [KP] |
| 357 | KVgradPress | kv-select | input×gradient attribution (one backward per chunk) | — | — | — | No | Stable | [KP] |
| 358 | RestoreKVPress | kv-select | 8 LoRA-encoded restore tokens before KVzip pruning | trained restore tokens | — | — | No | Stable | [KP] |
| 359 | KVComposePress | kv-select | align per-head selections into composite tokens (2× KV during prefill) | — | — | — | No | Stable | [KP] |
| 360 | AdaKVPress / LUKVPress / PerLayerCompressionPress | kv-select, placement | head-wise budget / layer+head budget curves / per-layer ratios | wrappers | — | — | No | Stable / experimental (PerLayer) | [KP] |
| 361 | ComposedPress / KeyRerotationPress | kv-select | chain presses; re-rotate RoPE after pruning | — | — | — | Rerotation: yes | Stable | [KP] |
| 362 | ChunkKVPress / EntropyGatedChunkKVPress / ChunkPress / BlockPress | kv-select | chunk- or block-wise selection for coherence / uniform compression | — | — | — | No | Stable | [KP] |
| 363 | CriticalKVPress / CriticalAdaKVPress | kv-select | refine scores with L1 norm of Wo·V, two-stage | — | — | — | No | Stable | [KP] |
| 364 | DecodingPress / CompressionRatioDecodingPress / PrefillDecodingPress | kv-select | periodic compression during generation; separate prefill/decode presses | — | decode | — | No | Stable | [KP] |
| 365 | DMSPress | kv-select | threshold eviction (trained evictors from DMS paper not included) | — | both | — | No | Stable | [KP] |
| 366 | CAMPress / MergingPress | kv-select | merge evicted KV into kept neighbors (cosine-similar) | — | decode | — | No | Stable | [KP] |

Reference links (used in the `link` column):

[Q]: https://nvidia.github.io/TensorRT-LLM/features/quantization.html
[FM]: https://nvidia.github.io/TensorRT-LLM/features/feature-combination-matrix.html
[SD]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md
[KV]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md
[KC]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kv-cache-compression.md
[SCH]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/paged-attention-ifb-scheduler.md
[PAR]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md
[ATT]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/attention.md
[SPA]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/sparse-attention.md
[TC]: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/torch_compile_and_piecewise_cuda_graph.md
[FEAT]: https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/features
[KT]: https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels
[REL]: https://github.com/NVIDIA/TensorRT-LLM/releases
[TB]: https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/blogs/tech_blog
[BL]: https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/blogs
[PO]: https://nvidia.github.io/TensorRT-LLM/1.3.0rc22/developer-guide/perf-overview.html
[FD]: https://docs.flashinfer.ai/
[FR]: https://flashinfer.ai/releases/
[F7]: https://flashinfer.ai/2026/09/22/flashinfer-v07.html
[FB]: https://flashinfer.ai/
[CD]: https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md
[CX]: https://github.com/NVIDIA/cutlass/tree/main/examples
[DY]: https://github.com/ai-dynamo/dynamo
[DR]: https://docs.nvidia.com/dynamo/latest/architecture/kv_cache_routing.html
[MO]: https://github.com/NVIDIA/TensorRT-Model-Optimizer
[NX]: https://github.com/ai-dynamo/nixl
[KP]: https://github.com/NVIDIA/kvpress

**Count: 366 rows** (TensorRT-LLM 193; FlashInfer 75; cuDNN 12; CUTLASS 22; Dynamo 15; ModelOpt 13; NIXL 4; kvpress 32 — kvpress presses that are pure wrappers were grouped 2–3 per row where the README groups them; everything else is one technique per row). "n/q" = the fetched source states the feature but prints no quantitative gain; "—" = not applicable or not stated.

### Inferences
- The architecture gate matters more than the library: NVFP4/MXFP4 anything (rows 1–2, 10, 12, 29, 128–131, 203, 208, 225, 229–231, 236, 291, 296–297) is Blackwell-or-newer; FP8 block-scaling is Hopper + datacenter Blackwell + Rubin but **not** sm_120; FP8 rowwise is Hopper-only; W4A8 excludes sm_120 and Ampere; FP8 KV is the only quantized-KV option that reaches Ampere [Q].
- Both TRT-LLM and FlashInfer have converged on the same Blackwell kernel substrate — TRT-LLM Gen cubins, now mirrored as readable Python "PrimTS/Prims-TS" kernels on CUTLASS DSL 4.7 — and on FlashInfer-hosted MegaMoE for expert-parallel MoE; a maxbox plugin for "MegaMoE" can therefore target vLLM/SGLang/TRT-LLM through one FlashInfer API [F7][REL].
- TRT-LLM docs are explicit that every speculative mode only pays off at low batch sizes and cannot be toggled dynamically [SD]; CUDA-graph capture-size tuning (blog20) and `max_num_tokens` (default 8192) are the two most-documented throughput knobs [SCH][TC].
- KV-cache tiering is now a three-tier (GPU/host/disk) system with per-tier lossy compression only in cold tiers (NVFP4 cold pages) and a hard rule of one compression method per instance [KC]; Dynamo's KVBM extends the same idea to SSD/S3/Azure [DY].
- Deprecations with dates: NGram and user-provided drafters removed Jan 2027 (replaced by Suffix Automaton) [SD]; `trtllm_custom_all_reduce`, `BatchDecodeMlaWithPagedKVCacheWrapper`, `QuantVariant` removed in FlashInfer v0.7.0 [FR].

### Gaps (unverified items)
- **Numeric claims not re-verified in this run** (pages not opened): TRT-LLM tech blogs 01–30 bodies (DeepSeek-R1 B200 latency/throughput records, MTP, Scaling EP parts 1–3, disagg, Llama 4 Maverick Eagle guide, NGram auto-enablement, ADP balance, skip-softmax, one-sided A2A, DWDP, CUDA-graph batch tuning, Helix, MoE-as-dense-GEMM, DeepSeek-V4, GVR V2, KV compression); the "Llama 4 Maverick 1000 TPS/user" and "DeepSeek-R1 world record" figures did **not** surface in search — only perf-overview numbers (row 1) are cited. FlashInfer cascade and sorting-free-sampling blog speedups likewise not re-fetched.
- **TRT-LLM pages not opened**: overlap-scheduler.md, lora.md, guided-decoding.md, sampling.md, disagg-serving.md, helix.md, kv-cache-connector.md, sleep-wakeup.md, model-express.md, subagent-routing.md, prefix-tokenization-cache.md, ray-orchestrator.md, long-sequence.md, additional-outputs.md, post-processor-hook.md, embeddings.md, visualgen pages — rows 141, 150–163 are therefore sourced only from the features index + release notes + feature matrix.
- **Capacity scheduler policies** (GUARANTEED_NO_EVICT / MAX_UTILIZATION / STATIC_BATCH), **INT8 SmoothQuant**, **INT8 KV** on the PyTorch backend, and **Medusa/ReDrafter/Lookahead** status: not on the fetched 2026 pages; likely legacy TensorRT-engine-only. Treat rows 30, 31, 140, 181 as unverified.
- **TRT-LLM kernel tree**: the GitHub listing was truncated between `minimaxM3Fp8IndexerKernel.h` and the second page; directory list is complete (29 dirs) but a few file names in the gap may be missing. `arcquantFP4`, `mhcKernels`, `kimiK3AttnRes` purposes are inferred from names only.
- **TRT-LLM Gen** sub-tree (`trtllmGenKernels`) contents and **TRT-LLM releases before v1.3.0rc28** (0.x → 1.2 perf items) were not enumerated (release page truncated at 100k chars).
- **FlashInfer**: GitHub releases API body was unusable (1.5 MB); release-highlights page covers v0.6.14 → v0.7.2 only, so v0.3–v0.5 items (e.g., first Blackwell/TRT-LLM Gen integration, FP4 MoE blog) are not dated here. MegaMoE blog URL 404'd (content taken from the v0.7 blog instead). The MLSys 2025 paper was not fetched.
- **cuDNN**: decode-specific fused kernels beyond SDPA (e.g., cuDNN "linear attention" runtime) are known only through FlashInfer release notes.
- **CUTLASS**: example list is from the directory names only; CuTe DSL version/date for CUTLASS 4.x not confirmed from the CUTLASS repo (only "CUTLASS DSL 4.7" via FlashInfer).
- **Dynamo**: router cost formula and knobs come from search snippets of docs (v0.7.1 / v0.9.x / latest) — the two direct docs URLs 404'd; planner internals (load predictors, correction factors), KVBM tier policies and NIXL-based KV transfer details were not opened.
- **ModelOpt**: `examples/llm_ptq/README.md` and the "choosing quant methods" guide 404'd; per-format hardware/regime guidance is therefore taken from the TRT-LLM quantization matrix; MXFP4, INT4 AWQ, SmoothQuant, W4A8, EAGLE3 are not named on the ModelOpt README although known to exist in the repo.
- **NIXL**: no version number or benchmark numbers on README; S3 is not named (OBJ plugin presumably covers it).
- **kvpress**: no version/date; RULER leaderboard numbers not included in README.

---

## Key Question 2 — What are the plugin seams of each library (where would a maxbox plugin hook in)?

### Takeaway
Every library exposes a small number of configuration-level or Python-API-level seams where one technique can be swapped for another without rebuilding: TRT-LLM via `LlmArgs`/YAML sub-configs and backend-string selectors; FlashInfer via per-op `backend=` arguments, wrapper classes with `plan()/run()`, the unified MoE `QuantConfig`/`SplitConfig`, and the experimental namespace; Dynamo via router/planner CLI flags and the KVBM/connector interfaces; ModelOpt via quant-config recipes; NIXL via transport plugins; kvpress via composable `Press` objects.

### Cited Findings
- **TensorRT-LLM (PyTorch backend)** — seams visible in the fetched docs: `KvCacheConfig` (dtype `auto|fp8|nvfp4|mixed_nvfp4_*`, `enable_block_reuse`, `enable_partial_reuse`, `host_cache_size`, `cold_tier_offload_min_priority`, `use_kv_cache_manager_v2`, `block_reuse_config`, `mamba_state_config`, `kv_events_config`, `event_buffer_max_size`) [KV]; `kv_cache_compression` config (one method per instance: NVFP4 cold-page with RoPE options, or TriAttention) [KC]; `sparse_attention_config` (`rocket|dsa|deepseek_v4|csa2|minimax_m3|skip_softmax`) [SPA]; attention backend string (`TRTLLM|FLASHINFER|VANILLA`, plus DFlash `VANILLA|TRTLLM|FA4`) [ATT][SD]; `decoding_type` YAML (`MTP|Eagle3|NGram|DraftTarget|PARD|DFlash|SA`) with `use_sa_spec` composition and a `Drafter`/`ResourceManager` hook (deprecated Jan 2027) [SD]; `cuda_graph_config`, `prefill_cuda_graph_backend: piecewise`, `torch_compile_config` (`enable_userbuffers`, `max_num_streams`, `compile_only_piecewise_graphs`) [TC]; `scheduler_config.enable_prefix_aware_scheduling`, `max_num_tokens`, `max_batch_size` [SCH][KV]; parallel knobs `tensor_parallel_size`, `moe_expert_parallel_size`, `moe_tensor_parallel_size`, `enable_attention_dp`, Wide-EP offline/online EPLB [PAR]; KV cache connector API (Mooncake connector, lending API) and KV event streams (ZMQ) for external routers [KV][REL]; logits post-processor / post-processing hook / guided-decoding backend (XGrammar, llguidance) [FEAT][FM]; checkpoint loaders (HF, ModelExpress) [FEAT]. The feature-combination matrix is itself a compatibility oracle a planner can read (e.g., spec decoding × PP = No; logits post-processor × disagg = No; Helix × overlap scheduler = Untested) [FM].
- **FlashInfer** — per-op `backend=` selectors (`"auto"|"fa2"|"fa3"|"trtllm-gen"|"xqa"|"cute-dsl"|"cudnn"|"cake"|"cutlass"|"cutile"|"prims_ts"`; env `FLASHINFER_DECODE_AUTO_CUDNN`), wrapper objects with `plan()/run()` for CUDA-graph safety (`BatchPrefill/Decode/MLA/POD/BlockSparse` wrappers) [FD][FR]; unified MoE `QuantConfig` (weight/activation/output format), `ExpertConfig`, routing variants, `MoEEpLayer` + `SplitConfig` (NCCL-EP / NIXL-EP / NVLink / CAKE) [FR][F7]; `LogitsPipe` composable sampler; `autotune_v2()` with `MeasurementPolicy` and persistent cache; `flashinfer.experimental` namespace + `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS` for opt-in graduation; JIT attention variant templates (custom masks/soft-cap) and `fi_trace`/Trace Apply for schema capture; `green_ctx` for SM partitioning; JAX via TVM FFI [FD][F7].
- **cuDNN frontend** — graph API with `sdpa` node attributes (implementation `AUTO|COMPOSITE|UNIFIED`, masks, page tables, ragged offsets, `score_mod`, softcap, sinks) and engine-selection heuristics; consumed by FlashInfer and TRT-LLM as a backend [CD][FR].
- **CUTLASS** — templated collective builders (Hopper WS, Blackwell UMMA), epilogue visitor trees (top-k/softmax, bias, quant), grouped/ptr-array problem descriptors, Stream-K scheduler, and CuTe DSL Python authoring (used by FlashInfer Prims-TS and TRT-LLM CuTe DSL kernels) [CX][F7].
- **Dynamo** — frontend CLI/env: `--router-mode {round-robin,random,kv,direct}`, `--kv-overlap-score-weight` (`DYN_KV_OVERLAP_SCORE_WEIGHT`), `--router-temperature` (`DYN_ROUTER_TEMPERATURE`), `--no-kv-events` (ApproxKvIndexer), snapshot threshold; Planner SLA targets; DGDR YAML (model, hardware, SLA); KVBM tier config (GPU/CPU/SSD/S3/Azure); backend adapters for SGLang/TRT-LLM/vLLM; Gateway API EPP plugin; per-request agentic hints [DY][DR].
- **ModelOpt** — quant config recipes (FP8/NVFP4/MXFP4/INT4-AWQ/W4A8, KV-cache quant flags such as `--kv_cache_quant nvfp4`), AutoQuantize/AutoQuant-KV search, QAT/QAD trainers, pruning (Minitron/Puzzletron), speculative-head training, unified HF checkpoint export to TRT-LLM/vLLM/SGLang [MO][Q].
- **NIXL** — agent API with backend plugins (UCX, GDS, GPUNETIO, LIBFABRIC, POSIX, OBJ, AZURE_BLOB, HF3FS, MOONCAKE, GUSLI, UCCL), ETCD metadata exchange, Python/Rust bindings; used as TRT-LLM/Dynamo KV transfer and as FlashInfer `NIXL-EP` [NX][FR].
- **kvpress** — `Press` objects (scorer presses + wrapper presses `ComposedPress`, `AdaKVPress`, `PerLayerCompressionPress`, `DecodingPress`, `PrefillDecodingPress`) applied as HF forward hooks via a `KVPressTextGenerationPipeline`; compression ratio is the single shared knob [KP].

### Inferences
- For maxbox, the lowest-friction seams are configuration strings rather than code: TRT-LLM's YAML sub-configs and FlashInfer's `backend=` arguments can both be driven from a machine profile (arch, NVLink/MNNVL presence, PCIe-only, memory) without touching kernels.
- FlashInfer's experimental namespace plus Autotuner v2's persistent cache is effectively a built-in "search for best engine per machine" loop, so a maxbox plugin could consume its cache rather than re-measure.

### Gaps
- Exact Python class/argument names for TRT-LLM `SparseAttentionConfig`, `SADecodingConfig`, `KvCacheConfig.kv_cache_compression`, and the connector interface were taken from the docs pages' prose; signatures were not verified against source.
- Dynamo's KVBM and connector Python/Rust interfaces, and planner config schema, were not opened.
