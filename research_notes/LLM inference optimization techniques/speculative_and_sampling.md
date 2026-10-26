# Speculative / Parallel Decoding and Sampling / Decoding Kernels for LLM Inference (catalog, coverage to Oct 2026)

Scope note for the report writer. Target system: one RTX 4090 (24 GB, sm_89), PCIe 3.0 x16 (~12-13 GB/s real), i7-10700K (AVX2, 8C), ~64 GB DDR4 (~40-45 GB/s real), large MoE models with routed experts kept in system RAM (llama.cpp `-cmoe`/`-ot` style: experts are *computed on the CPU*, not streamed to the GPU). Decode is bound by reading expert weights from DDR4.

Entry format used below: **Name (date)** — what it does | slot | preconditions | reported speedup and regime | lossless or lossy | maturity | sources.
"Slot" names the plugin slot in the user's engine: `drafter` (proposes tokens), `draft-head` (extra layers attached to the target), `verifier` (tree/chain verification, acceptance rule), `spec-controller` (draft length/tree shape policy), `sampler` (logits to token), `logits-processor` (penalties, grammar masks), `scheduler` (batching interaction).

Source-confidence note: entries marked "(abstract)" cite headline numbers from the paper's arXiv abstract as I know it; those classic 2022-2025 papers were not re-fetched this session (tool budget), so the report writer should treat their numbers as "as reported by the authors". Items fetched or searched this session are cited to the page actually read.

---

## Q1. Draft-model methods (separate small model proposes, big model verifies)

### Takeaway
Classic draft-then-verify is lossless (rejection sampling keeps the target distribution exactly), gives about 2-3x on dense models at batch 1 when draft and target share a tokenizer and family, and has the most extreme payoffs in *offloading* regimes (SpecInfer, Sequoia, SpecExec, TriForce report 3-18x when the target is streamed over PCIe). On the user's CPU-expert MoE setup, a separate draft model is the weakest option: it costs VRAM, and the target-side verification cost grows with the union of experts touched.

### Cited Findings
- **Blockwise Parallel Decoding (Stern et al., Nov 2018)** — first "predict k future tokens with extra heads, verify with the base model, keep the longest agreeing prefix" scheme, greedy only | draft-head + verifier | needs trained heads | ~2x iteration reduction on MT (abstract) | lossless under greedy | historical | [arXiv 1811.03115](https://arxiv.org/abs/1811.03115)
- **Speculative Decoding (Leviathan, Kalman, Matias; Google; Nov 2022)** — a small model drafts γ tokens; the target scores all γ+1 positions in one pass; a modified rejection-sampling rule accepts a prefix so outputs are distributed exactly as the target | drafter + verifier | same-tokenizer draft model | 2-3x on T5-XXL (abstract) | lossless | foundational, in every engine | [arXiv 2211.17192](https://arxiv.org/abs/2211.17192)
- **Speculative Sampling (Chen et al.; DeepMind; Feb 2023)** — same algorithm, developed independently, for Chinchilla-70B with a 4B draft | drafter + verifier | 2-2.5x in a distributed setup (abstract) | lossless | foundational | [arXiv 2302.01318](https://arxiv.org/abs/2302.01318)
- **BiLD, Big Little Decoder (Feb 2023)** — the small model generates until unsure, then the big model takes over or rolls back. Fallback and rollback policies | drafter + spec-controller | lossy, with a tunable tolerance (abstract) | [arXiv 2302.07863](https://arxiv.org/abs/2302.07863)
- **SpecInfer (May 2023)** — several small speculative models or boosted drafts build a *token tree*. The target verifies the whole tree in one pass using a topology-aware causal mask ("tree attention"), with multi-step stochastic verification | drafter + verifier(tree) | 1.5-2.8x distributed, **2.6-3.5x for offloading-based inference** (abstract) | lossless | research, now a standard idea | [arXiv 2305.09781](https://arxiv.org/abs/2305.09781)
- **Staged Speculative Decoding (Spector & Ré, Aug 2023)** — restructures drafts into a tree and adds a second stage that speculates on the draft model itself, for small-batch on-device use | drafter cascade | ~3.16x on GPT-2-L at batch 1 (abstract) | lossless | research | [arXiv 2308.04623](https://arxiv.org/abs/2308.04623)
- **DistillSpec (Oct 2023)** — knowledge distillation of the draft model, on-policy, with a divergence chosen to maximise acceptance | drafter training | 10-45% over standard SD (abstract) | lossless at inference | research recipe | [arXiv 2310.08461](https://arxiv.org/abs/2310.08461)
- **Online Speculative Decoding (Oct 2023)** — continually distils the draft on live query traffic to track the query distribution | drafter training (online) | lossless | research | [arXiv 2310.07177](https://arxiv.org/abs/2310.07177)
- **SpecTr (Oct 2023)** — multi-draft verification posed as optimal transport; k i.i.d. drafts per step, with a provably better acceptance rule | verifier | lossless | research | [arXiv 2310.15141](https://arxiv.org/abs/2310.15141)
- **Cascade Speculative Drafting (Dec 2023)** — vertical cascade (draft-of-draft, ending in a statistical bigram model) plus horizontal cascade (smaller drafters for later, low-value positions) | drafter cascade | lossless | research | [arXiv 2312.11462](https://arxiv.org/abs/2312.11462)
- **Sequoia (Feb 2024)** — dynamic programming finds the optimal tree size and shape for the hardware, with a sampling-without-replacement verifier that stays robust across temperatures | spec-controller + verifier | up to ~4x for 7B on A100; **offloading Llama2-70B on L40: 0.56 s/token, ~9.96x over their optimized offloading baseline** (abstract) | lossless | research code | [arXiv 2402.12374](https://arxiv.org/abs/2402.12374)
- **Recursive Speculative Decoding (Feb 2024)** — tree drafts sampled without replacement | verifier | lossless | [arXiv 2402.14160](https://arxiv.org/abs/2402.14160)
- **TriForce (Apr 2024)** — hierarchical speculation for long context: a tiny model drafts for the target with a retrieval-based partial KV cache, which in turn drafts for the full target | drafter cascade | **2x RTX 4090 offloading Llama2-7B-128K: 0.108 s/token, 7.78x over their optimized offloading system** (abstract) | lossless | research | [arXiv 2404.11912](https://arxiv.org/abs/2404.11912)
- **SpecExec (Jun 2024)** — "speculative execution": the draft builds a very large most-likely-continuation tree (thousands of tokens), and the target verifies it in one pass. Designed for **consumer GPUs with RAM offloading** (tested on RTX 4090-class) | drafter + verifier(tree) | 70B on consumer GPU with offloading: 4-6 tok/s with 4-bit, 2-3 tok/s with 16-bit; 10.6x / 18.7x over autoregressive offloading (abstract) | lossless | research code | [arXiv 2406.02532](https://arxiv.org/abs/2406.02532)
- **Speculative Cascades (Google, May 2024)** — mixes cascades (defer to big model) with speculative decoding through a deferral rule | lossy and tunable | [arXiv 2405.19261](https://arxiv.org/abs/2405.19261)
- **QSpec / quantized self-draft (Oct 2024)** — the draft is a low-precision copy of the same model, verified at higher precision | drafter | lossless relative to the high-precision model | [arXiv 2410.11305](https://arxiv.org/abs/2410.11305)
- **PARD (AMD, Apr 2025)** — PARallel Draft: one target-independent draft model, adapted to predict several masked future tokens in a single forward pass. One draft serves a whole model family | drafter | lossless | open code | [arXiv 2504.18583](https://arxiv.org/abs/2504.18583)
- **llama.cpp `draft-simple`** — classic draft model in llama.cpp. Flags `-md/--spec-draft-model`, `-hfd`, `-ngld` (draft layers on GPU), `-cmoed/-ncmoed` (keep a MoE draft's experts on CPU), and draft KV types `-ctkd/-ctvd`. Default `--spec-draft-n-max 3` | drafter | lossless (greedy exact-match caveats below) | production | [llama.cpp docs/speculative.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)
- Consumer-GPU anecdote: with a 0.8B external drafter on a dense Qwen 27B on one RTX 3090, the drafter config was **−29.8%** (slower than plain decoding) | [hackmd thc1006 RTX 3090 study](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090)

### Inferences
- The offloading papers (SpecInfer, Sequoia, SpecExec, TriForce) are the closest analogue to the user's bandwidth-starved regime. Their big multipliers come from verifying very large trees per weight read, which works when the whole weight set is read anyway (dense offload). With CPU-resident top-k MoE experts, a larger tree touches more distinct experts, so the SpecExec-style "giant tree" logic does not transfer directly. Tree width has to be traded against the expert union (see Q7).
- A separate draft model is the lowest-priority drafter for the user. Heads (MTP, EAGLE-3, DFlash, DSpark) reuse the target's hidden state, are cheap in VRAM, and get much higher acceptance.

### Gaps
- No found measurements of classic draft-model SD on llama.cpp with CPU-offloaded MoE experts on a 4090.

---

## Q2. Heads and self-drafting (Medusa, EAGLE family, MTP, layer-skip)

### Takeaway
Methods that draft from the target's own hidden states dominate in production (EAGLE-3, and native MTP for models that ship it). For the user's models, native MTP heads (Qwen3.6, Gemma 4, GLM nextn) and EAGLE-3 drafts are the practical options in llama.cpp (`draft-mtp`, `draft-eagle3`). Acceptance is 75-90% for the first MTP token, but on small-active-parameter MoE the end-to-end gain is modest (about 1.17x reported for Qwen3.6-35B-A3B on a large GPU) and can be negative on a 24 GB card.

### Cited Findings
- **Medusa-1 / Medusa-2 (Jan 2024)** — K extra decoding heads on the last hidden state, each predicting token t+k; candidates are combined into a tree and verified with tree attention. Introduced **typical acceptance**, a lossy, faster acceptance rule | draft-head + verifier(tree) | heads trained on a frozen backbone (M-1) or jointly (M-2) | >2.2x (M-1), 2.3-3.6x (M-2) (abstract) | lossless with rejection sampling, lossy with typical acceptance | mature, in vLLM, TRT-LLM and others | [arXiv 2401.10774](https://arxiv.org/abs/2401.10774), [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)
- **Hydra (Feb 2024)** — sequentially dependent Medusa heads (each head sees the earlier draft tokens) | draft-head | ~1.31x throughput over Medusa (abstract) | lossless | [arXiv 2402.05109](https://arxiv.org/abs/2402.05109)
- **EAGLE-1 (Jan 2024)** — a one-layer autoregressive drafter at the *feature* (second-to-top layer) level, fed the token sequence shifted by one to remove sampling uncertainty | draft-head | 2.7-3.5x on LLaMA2-Chat 70B (abstract) | lossless | mature | [arXiv 2401.15077](https://arxiv.org/abs/2401.15077), [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)
- **EAGLE-2 (Jun 2024)** — context-aware *dynamic draft trees*: the drafter's confidence approximates acceptance, so the tree is expanded and reranked per step | draft-head + spec-controller | 3.05-4.26x (abstract) | lossless | mature | [arXiv 2406.16858](https://arxiv.org/abs/2406.16858)
- **EAGLE-3 (Mar 2025)** — drops feature prediction, fuses low, mid and high-layer target features, and trains with "training-time test" (multi-step rollouts) so acceptance scales with data | draft-head | up to ~6.5x; ~1.38x throughput at batch 64 in SGLang (abstract) | lossless | **de facto industrial standard** | [arXiv 2503.01840](https://arxiv.org/abs/2503.01840); described as the industrial standard in [SpecForge](https://arxiv.org/pdf/2603.18567)
- **SpecForge / SpecBundle (sgl-project, Mar 2026)** — open training framework for EAGLE-3 drafts; up to 9.9x faster EAGLE-3 training for Qwen3-235B-A22B; SpecBundle drafts give up to 4.48x end-to-end on SGLang | tooling | [arXiv 2603.18567](https://arxiv.org/pdf/2603.18567), [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge)
- **llama.cpp `draft-eagle3`** — a one-layer draft that reads target hidden states. The GGUF must be converted with `--target-model-dir` and is trained for one specific target | [llama.cpp docs/speculative.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)
- **HASS (Aug 2024)** — "harmonized" EAGLE training: context alignment between training and decoding plus distillation focused on the top-k tokens | draft-head training | lossless | [arXiv 2408.15766](https://arxiv.org/abs/2408.15766)
- **GLIDE with CAPE (Feb 2024)** — the draft reuses the target's KV cache (cross-attention to it), and confidence-based proposal expansion | drafter | lossless | [arXiv 2402.02082](https://arxiv.org/abs/2402.02082)
- **Falcon (Dec 2024)** — semi-autoregressive drafting with coupled sequential glancing distillation and a "decoding tree" | draft-head | lossless | [arXiv 2412.12639](https://arxiv.org/abs/2412.12639)
- **Clover / Clover-2 (2024)** — RNN-based regressive connections between Medusa-style heads, plus an attention decoder | draft-head | lossless | [arXiv 2405.00263](https://arxiv.org/abs/2405.00263), [arXiv 2408.00264](https://arxiv.org/abs/2408.00264)
- **ReDrafter, Recurrent Drafter (Apple, Mar 2024)** — RNN drafter on target hidden states, with beam search over drafts and dynamic tree attention; distilled. Integrated in TensorRT-LLM and MLX | draft-head | lossless | production at Apple/NVIDIA | [arXiv 2403.09919](https://arxiv.org/abs/2403.09919)
- **Speculative Streaming (Apple, Feb 2024)** — fuses drafting into the target by fine-tuning with multi-stream attention, so there are no extra heads | needs fine-tune | [arXiv 2402.11131](https://arxiv.org/abs/2402.11131)
- **Multi-token prediction training (Gloeckle et al., Meta, Apr 2024)** — trains n independent output heads on a shared trunk. 4-token models decode up to 3x faster with self-speculation (abstract) | needs pretraining with MTP | [arXiv 2404.19737](https://arxiv.org/abs/2404.19737)
- **DeepSeek-V3 MTP (Dec 2024)** — sequential MTP modules (each a transformer block plus projection, sharing the embedding and head) used as a training objective and reused for speculation. Second-token acceptance 85-90%, about 1.8x TPS (tech report) | draft-head (native) | model must ship MTP weights | lossless | production (SGLang, vLLM) | [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- **llama.cpp `draft-mtp`** — uses the model's built-in MTP heads. Added in PR #22673; one guide says it was merged 16 May 2026 | [llama.cpp docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md), [mer.vin guide](https://mer.vin/2026/05/run-qwen-3-6-mtp-in-llama-cpp-faster-local-inference-with-built-in-speculative-decoding/)
  - Official ggml-org MTP GGUFs exist for Qwen3.6-27B and **Qwen3.6-35B-A3B** — [mer.vin](https://mer.vin/2026/05/run-qwen-3-6-mtp-in-llama-cpp-faster-local-inference-with-built-in-speculative-decoding/)
  - MTP acceptance is about 75-90% on MTP-trained targets. At 128K context: decode 90 to 95 tok/s (+5%) with 76% acceptance at n=2, but **prefill fell 22%** (3,840 to 2,978 tok/s) because of a D2H/H2D transfer per ubatch. MTP is **silently disabled with `nParallel > 1` and with `--cache-reuse`**; it needs ~500 MB of extra compute buffer | [HF space "MTP vs the others"](https://victor-mtp-on-hf-endpoints.static.hf.space/index.html)
  - Qwen3.6-35B-A3B Q8_0 on an RTX PRO 6000 (fully in VRAM): 193.36 to 225.48 tok/s with MTP (**1.17x**); the gain is attributed to only 3B active parameters making baseline decode cheap | [JarvisLabs](https://jarvislabs.ai/blog/qwen36-mtp-llamacpp-rtxpro6000) (via search summary)
  - Dense Qwen 27B Q4 on one RTX 3090, llama.cpp `draft-mtp`: baseline 41.55 tok/s; MTP n-max 2: 66.39 (+59.8%); n-max 3: 63.29 (+52.3%); n-max 5: 54.89 (+32.1%). Each extra verified position cost about 0.25-0.29 of a plain decode step | [hackmd RTX 3090 study](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090)
  - Known issue: `draft-mtp` with `--spec-draft-n-max 3` changes deterministic output on Qwen3.6 27B (issue #23302). Image input reportedly crashes with MTP on | [llama.cpp issue #23302](https://github.com/ggml-org/llama.cpp/issues/23302), [mer.vin](https://mer.vin/2026/05/run-qwen-3-6-mtp-in-llama-cpp-faster-local-inference-with-built-in-speculative-decoding/)
- **Gemma 4 MTP drafters (Google, 2026)** — separate small drafter checkpoints ("assistant_model" in HF Transformers) that predict several tokens autoregressively. Google claims up to 3x with no quality loss, measured on LiteRT-LM, MLX, HF Transformers and vLLM; Apache 2.0, on HF, Kaggle and Ollama | drafter/draft-head | lossless | [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/), [Gemma MTP docs](https://ai.google.dev/gemma/docs/mtp/mtp)
- **"Your LLM Knows the Future" (Apple, Jul 2025)** — fine-tunes an AR model to fill in mask tokens for future positions (gated LoRA plus a sampler head), giving MTP without pretraining | needs fine-tune | [arXiv 2507.11851](https://arxiv.org/abs/2507.11851)
- **Self-Speculative / Draft & Verify (Sep 2023)** — the draft is the target with some layers skipped (layer set found by Bayesian optimization) | drafter = target subset | no extra weights | up to ~1.99x (abstract) | lossless | [arXiv 2309.08168](https://arxiv.org/abs/2309.08168)
- **LayerSkip (Meta, Apr 2024)** — training recipe with layer dropout and an early-exit loss; self-speculation drafts with early layers and verifies with the rest, reusing the KV cache | needs a LayerSkip-trained checkpoint | up to ~2.16x (abstract) | lossless | [arXiv 2404.16710](https://arxiv.org/abs/2404.16710)
- **Kangaroo (Apr 2024)** — a shallow sub-network plus a small adapter as drafter, with early-exit drafting | ~1.68x on Spec-Bench with 88.7% fewer extra params than Medusa (abstract) | lossless | [arXiv 2404.18911](https://arxiv.org/abs/2404.18911)
- **SWIFT (Oct 2024)** — on-the-fly self-speculation that adaptively picks layers to skip, with no training | 1.3-1.6x (abstract) | lossless | [arXiv 2410.06916](https://arxiv.org/abs/2410.06916)
- **DraftExpert (Jul 2026)** — MoE self-speculation: the drafter keeps the shared expert plus the top-1 routed expert plus one small trained "draft expert" per layer that stays in accelerator memory (see Q7) | [alphaXiv 2607.24434](https://www.alphaxiv.org/abs/2607.24434v1)

### Inferences
- For Qwen3.6-35B-A3B and GLM-5.3-Flash (nextn), MTP is the zero-training, already-shipped option. For Gemma 4 26B-A4B, the shipped MTP drafters are the analogue. For gpt-oss-120b (no native MTP), an EAGLE-3 draft is the realistic head; NVIDIA has published EAGLE-3 drafts for gpt-oss, but I did not verify this in this session.
- Self-speculation by skipping *experts* rather than layers (DraftExpert-style: shared plus top-1 expert, all in VRAM) suits the user's topology especially well. Drafting then touches no DDR4 at all, and only verification pays the expert-read cost.

### Gaps
- No measured acceptance or throughput for MTP or EAGLE-3 on gpt-oss-120b, Gemma 4 26B-A4B or GLM-5.3-Flash under CPU expert offload.
- Could not confirm whether llama.cpp `draft-mtp` supports GLM's nextn layer and Gemma 4's separate MTP GGUF under the same code path.

---

## Q3. Draft-free methods (Jacobi/lookahead, n-gram/prompt lookup, retrieval, suffix trees)

### Takeaway
Draft-free methods cost no VRAM and almost no compute. They win big on repetitive, copy-heavy or agentic workloads (code edits, RAG, summarization, re-emitting tool output) and lose on open-ended text. In llama.cpp they are `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod` and `ngram-cache`. The docs note that **MoE models need long n-gram drafts** to pay off.

### Cited Findings
- **Jacobi decoding (Santilli et al., May 2023)** — treats greedy decoding as a fixed-point system and iterates all positions in parallel | no extra model | lossless (greedy) | small gains alone | [arXiv 2305.10427](https://arxiv.org/abs/2305.10427)
- **Lookahead Decoding (Hao AI Lab, Feb 2024)** — Jacobi iteration with a 2D window that generates n-grams, an n-gram pool, and parallel verification of candidates in the same forward pass | drafter(self) + verifier | ~1.8x on MT-Bench, up to 4x on code with multi-GPU strong scaling (abstract) | lossless | open code | [arXiv 2402.02057](https://arxiv.org/abs/2402.02057), [hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding)
- **CLLMs, Consistency LLMs (Mar 2024)** — fine-tune the target to map any Jacobi trajectory point to the fixed point, which makes Jacobi decoding converge fast | needs fine-tune | 2.4-3.4x (abstract) | lossless relative to the fine-tuned model | [arXiv 2403.00835](https://arxiv.org/abs/2403.00835)
- **Prompt Lookup Decoding (Saxena, Nov 2023)** — match the last n tokens against the prompt and draft the continuation found there; in HF Transformers as `prompt_lookup_num_tokens`, and in vLLM as `[ngram]` | drafter | lossless | mature | [apoorvumang/prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding)
- **REST, Retrieval-based SD (Nov 2023)** — drafts come from a datastore (suffix array over a corpus) and are verified as a tree | drafter (retrieval) | 1.62-2.36x on 7B/13B (abstract) | lossless | [arXiv 2311.08252](https://arxiv.org/abs/2311.08252)
- **ANPD, Adaptive N-gram Parallel Decoding (Apr 2024)** — an adaptive n-gram module built from context in real time, plus multi-level n-grams | up to 3.67x on LLaMA (abstract) | lossless | [arXiv 2404.08698](https://arxiv.org/abs/2404.08698)
- **Ouroboros (Feb 2024)** — phrase-level drafting and reuse of verification-rejected candidates, without training | lossless | [arXiv 2402.13720](https://arxiv.org/abs/2402.13720)
- **SuffixDecoding (Snowflake, Nov 2024; v2 2025)** — suffix trees over the prompt and past outputs, with an adaptive speculation length set by match confidence; aimed at agentic and repetitive workloads; integrated into vLLM / Arctic Inference | drafter | lossless | production (Arctic Inference) | [arXiv 2411.04975](https://arxiv.org/abs/2411.04975)
- **llama.cpp draftless family** (all from [docs/speculative.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)):
  - `ngram-simple`: most recent earlier match of the last n tokens, drafts the m tokens that followed (defaults n=12, m=48, min-hits 1).
  - `ngram-map-k`: hash map of n-grams in the context; drafts only after a key is seen min-hits times; tracks per-n-gram acceptance (defaults 12/48/1).
  - `ngram-map-k4v`: experimental; tracks up to 4 continuations per key and drafts the dominant one.
  - `ngram-mod`: LCG-hashed n-gram to next-token pool (~16 MB), **shared across all server slots**; variable draft length; defaults n-match 24, n-min 48, n-max 64. Docs: "MoE models need long drafts", while dense models can use smaller n-min/n-max. `--spec-default` turns on ngram-mod.
  - `ngram-cache`: n-gram statistics, optionally loaded from external files.
  - Types can be combined (`--spec-type ngram-mod,ngram-map-k4v`). When a draft model is combined with a draftless type, the draftless one takes precedence.
- Measured: ngram-cache at n=2, 128K context: decode 90 to 75 tok/s (**−17%**), 32% acceptance. N-gram acceptance ranges about 30-90% by workload and drops sharply on non-repetitive output | [HF space](https://victor-mtp-on-hf-endpoints.static.hf.space/index.html)
- Sample stats in the docs: ngram-mod acceptance 0.703 (90/128); mixed setup 0.576 (171/297). Illustrative, not benchmarks | [llama.cpp docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)

### Inferences
- For MoE with CPU experts, a long n-gram draft that is accepted almost entirely (copying a file or tool output) is the best case. Neighbouring tokens in copied spans tend to reuse experts, and the draft costs nothing. The llama.cpp docs advice ("MoE needs long drafts") is consistent with verification having a high fixed cost (the expert union) that only long accepted runs amortize.
- A sensible plugin default is a *hybrid controller*: ngram-mod when a match exists, otherwise MTP/EAGLE/DFlash. llama.cpp already gives draftless types precedence when combined.

### Gaps
- No public measurements of llama.cpp ngram-* on CPU-offloaded MoE.

---

## Q4. Newer methods: DFlash, DSpark, diffusion/parallel drafting, 2025-2026

### Takeaway
2026's main shift is to **parallel (non-autoregressive) drafters** that emit a whole block in one forward pass, conditioned on target hidden states. DFlash (ICML 2026, block diffusion) reports more than 6x lossless and about 2.4x EAGLE-3's speedup. DSpark (Jul 2026) adds a lightweight sequential "Markov/RNN" head to fight suffix decay, plus confidence-scheduled verification length. Both are in llama.cpp (`draft-dflash`, `draft-dspark`). DSpark in llama.cpp currently supports only Qwen3 backbones, with Gemma 4 planned.

### Cited Findings
- **DFlash: Block Diffusion for Flash Speculative Decoding (Chen, Liang, Liu; arXiv Feb 2026; ICML 2026)** — a lightweight block-diffusion drafter generates a block of 16 tokens (10 for Llama-3.1-8B) in **one forward pass**. Hidden features from 5 target layers are fused by a projection and injected into the K/V projections of every draft layer. Default 5 draft layers (8 for Qwen3-Coder-30B-A3B). Embedding and LM head are shared and frozen. Trained on ~800K target-generated samples, 6 epochs | drafter (parallel) | needs a per-target trained drafter | Qwen3-4B/8B average 4.9x / 4.86x (τ≈6.5) greedy on H200 with Transformers, ~4.0-4.2x at T=1, peak ~6.1x; about 2.4x EAGLE-3's speedup. SGLang on B200: Qwen3-8B Math500 5.1x at concurrency 1 falling to 2.8x at 32; **Qwen3-Coder-30B-A3B (MoE) HumanEval 3.5x to 3.1x**. EAGLE-3 (tree 60) fell to 0.6x at concurrency 32 while DFlash held 1.6x | lossless | open: [z-lab/dflash](https://github.com/z-lab/dflash), [HF z-lab/dflash](https://hf.co/collections/z-lab/dflash) | [arXiv 2602.06036](https://arxiv.org/html/2602.06036v1), [ICML 2026](https://icml.cc/virtual/2026/poster/64301)
- **llama.cpp `draft-dflash`** — block-diffusion draft, several layers, receives target hidden states; GGUF converted with `--target-model-dir`. `--spec-draft-n-max` is clamped to the trained block size | [llama.cpp docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)
  - "DFlash2" on a llama.cpp branch (PR #27342), dense Qwen 27B, RTX 3090: n-max 4: 63.13 tok/s (+51.9%); n-max 7: 50.95 (+22.6%), against a 41.55 baseline. MTP n-max 2 was slightly better (+59.8%) | [hackmd RTX 3090 study](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090)
  - DGX Spark (vLLM) Qwen3.6-35B-A3B: DFlash about 80 tok/s vs MTP about 65 tok/s (user report) | [NVIDIA forums](https://forums.developer.nvidia.com/t/qwen-qwen3-6-35b-a3b-and-fp8-has-landed/366822?page=9)
  - Intel blog on DFlash for Qwen3.6 on Core Ultra Series 3 (CPU/iGPU) | [HF blog](https://huggingface.co/blog/ofirzaf/intel-dflash-ptl) (not read in detail)
- **DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation (Xin Cheng + 32 co-authors, arXiv 6 Jul 2026)** — a parallel drafting backbone coupled with a lightweight sequential module (intra-block dependency) to reduce "suffix decay" in acceptance. **Confidence-scheduled verification** sets the verification length per request from estimated prefix-survival probabilities and engine throughput profiles, so batch capacity is not spent on doomed tokens | drafter + spec-controller | improves accepted length over AR and parallel drafters (abstract; numbers not retrieved) | lossless | [arXiv 2607.05147](https://arxiv.org/abs/2607.05147), [HF papers](https://huggingface.co/papers/2607.05147)
- **llama.cpp `draft-dspark`** — "DFlash plus a semi-autoregressive Markov head that biases each block position based on the previous token". `--spec-draft-conf-min P` truncates a block where predicted acceptance falls below P. **Only Qwen3 backbones for now; Gemma4 planned** | [llama.cpp docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)
  - Liquid AI LFM2.5-DSpark drafts (1.2B, 2.6B, 8B-A1B), upstreamed in llama.cpp and SGLang; up to 3.18x on GPU, 2.87x on device. LFM2.5-8B-A1B (MoE) got only **1.18x on M4 Max vs 2.54x on H100**, attributed to llama.cpp's MoE implementation | [Liquid AI blog](https://www.liquid.ai/blog/lfm2-5-dspark) (via search summary)
  - llama.cpp release b10228 reportedly added DeepSeek-V4-Flash's embedded spec head, claimed to be DSpark rather than MTP (PR #25784, third-party claim) | [search summary of dev.to post](https://dev.to/breachprotocol/llamacpp-shipped-dspark-for-deepseek-v4-flash-and-almost-everyone-called-it-the-wrong-name-37ed) (page returned 404 on fetch; unverified)
- **DPara, "When Parallel Drafter Meets Parallel Speculative Decoding" (Sep 2026)** — parallel speculative decoding (draft and verify overlapped) driven by DSpark-style parallel drafters | [arXiv 2609.27396](https://arxiv.org/pdf/2609.27396)
- **"Fail Fast, Win Big" (Dec 2025)** — diffusion LLMs as drafters, rethinking draft strategy | [arXiv 2512.20573](https://arxiv.org/pdf/2512.20573)
- **Multimodal SD with diffusion parallel drafting survey/diagnosis (Aug 2026)** | [arXiv 2608.20743](https://arxiv.org/pdf/2608.20743)
- **PEARL, Parallel spEculative decoding with Adaptive dRaft Length (Aug 2024)** — pre-verify and post-verify overlap drafting with verification (draft runs concurrently with target) | spec-controller | lossless | [arXiv 2408.11850](https://arxiv.org/abs/2408.11850)
- **Lookahead Reasoning (Jun 2025) / SpecReason (Apr 2025)** — step-level speculation for reasoning models: a small model drafts whole reasoning steps, accepted on semantic equivalence | lossy (semantic) | [arXiv 2506.19830](https://arxiv.org/abs/2506.19830), [arXiv 2504.07891](https://arxiv.org/abs/2504.07891)

### Inferences
- Parallel drafters suit the user's hardware because the draft costs one GPU forward pass regardless of block length, so drafting is nearly free. The binding cost is verification on CPU experts. DSpark's confidence truncation (`--spec-draft-conf-min`) is the knob that limits verification width to likely-accepted tokens, which directly bounds the expert union. That makes it a prime search dimension for the agent.
- DFlash's measured MoE result (Qwen3-Coder-30B-A3B: 3.5x, fully on a B200) shows parallel drafting can help MoE when experts are resident. It says nothing about CPU-offloaded experts.

### Gaps
- No DFlash/DSpark drafts were confirmed for Qwen3.6-35B-A3B, Gemma 4 26B-A4B, gpt-oss-120b or GLM-5.3-Flash. (Qwen3.6 DFlash evidently exists given the DGX Spark and Intel reports; repo not checked.)
- No DSpark headline numbers retrieved from the paper itself.

---

## Q5. Verification, acceptance rules, draft-length control, batch interaction, measurement methodology

### Takeaway
Verification is its own slot: chain vs tree (tree attention masks), exact rejection sampling vs typical or greedy acceptance, and a draft-length/tree-shape controller. Speedups fall with batch size and concurrency, sometimes below 1x (EAGLE-3 big trees at 0.6x at concurrency 32). Adaptive controllers (Cascade, SpecDec++, DSpark confidence scheduling, SmartSpec goodput) avoid the slowdowns. "Lossless" in theory is often *not* bit-exact in practice: batched verification changes kernel numerics.

### Cited Findings
- **Tree attention / topology-aware causal mask** — verify many candidate branches in one pass by flattening the tree and masking each token to its ancestors (SpecInfer; Medusa; EAGLE-2 dynamic trees) | verifier | [SpecInfer](https://arxiv.org/abs/2305.09781), [Medusa](https://arxiv.org/abs/2401.10774)
- **Lossless speculative (rejection) sampling** — accept draft token x with prob min(1, p(x)/q(x)); on rejection, resample from norm(max(0, p−q)) | verifier | [Leviathan](https://arxiv.org/abs/2211.17192), [Chen](https://arxiv.org/abs/2302.01318)
- **Typical acceptance (Medusa)** — accept if the target probability exceeds an entropy-dependent threshold; faster, lossy | [arXiv 2401.10774](https://arxiv.org/abs/2401.10774)
- **Multi-draft optimal-transport verification (SpecTr)**, **sampling without replacement (Sequoia, RSD)** | [SpecTr](https://arxiv.org/abs/2310.15141), [Sequoia](https://arxiv.org/abs/2402.12374)
- **Block Verification (Sun et al., Mar 2024)** — verifies the draft block jointly rather than token by token; provably optimal among lossless verifiers and never worse than token-wise | verifier | lossless | [arXiv 2403.10444](https://arxiv.org/abs/2403.10444)
- **Traversal Verification (May 2025)** — leaf-to-root tree verification that keeps more of the tree | verifier | lossless | [arXiv 2505.12398](https://arxiv.org/abs/2505.12398)
- **Dynamic draft length / tree shape**: SpecDec++ (stop drafting via a learned acceptance-probability head, May 2024) [arXiv 2405.19715](https://arxiv.org/abs/2405.19715); EAGLE-2 confidence trees [arXiv 2406.16858](https://arxiv.org/abs/2406.16858); OPT-Tree (Jun 2024) [arXiv 2406.17276](https://arxiv.org/abs/2406.17276); Sequoia hardware-aware DP [arXiv 2402.12374](https://arxiv.org/abs/2402.12374); DSpark confidence scheduling [arXiv 2607.05147](https://arxiv.org/abs/2607.05147); llama.cpp `--spec-draft-p-min` (greedy-draft probability floor), `--spec-draft-p-split`, `--spec-draft-n-min/max`, `--spec-draft-conf-min` [llama.cpp docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)
- **Cascade (utility-driven SD for MoE, Jun 2025; MICRO-58)** — for MoE, draft tokens collectively activate more weights, raising data movement and verification time 2-3x and causing **slowdowns up to 1.5x**. A runtime controller uses "speculation utility" (token gain / verification cost), toggles speculation and tunes K. In vLLM it caps slowdown at 5% and gives +7-14% throughput over static K | spec-controller | [arXiv 2506.20675](https://arxiv.org/abs/2506.20675)
- **Batch-size interaction**: EAGLE-3 still ~1.38x at batch 64 in SGLang (abstract) [arXiv 2503.01840](https://arxiv.org/abs/2503.01840). DFlash vs EAGLE-3 at concurrency 1 to 32 [arXiv 2602.06036](https://arxiv.org/html/2602.06036v1). **MagicDec (Aug 2024)**: for long contexts, SD helps even at large batch because the KV-cache read dominates [arXiv 2408.11049](https://arxiv.org/abs/2408.11049). **SmartSpec/goodput (Jun 2024)**: picks speculation length per step to maximise goodput in vLLM [arXiv 2406.14066](https://arxiv.org/abs/2406.14066). **BASS (Apr 2024)**: batched SD with ragged tensors [arXiv 2404.15778](https://arxiv.org/abs/2404.15778).
- **Cost per extra verified position** on a 3090 (dense 27B, Q4, MMVQ kernels) was about **0.25-0.29 of a plain decode step**, which is why n-max 2 beat larger drafts | [hackmd](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090)
- **Exactness in practice**: on llama.cpp, 92-100% of requests diverged from serial greedy output by 1,600 tokens under MTP/DFlash2 [hackmd](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090). Issue #23302 reports MTP changing deterministic output [GitHub](https://github.com/ggml-org/llama.cpp/issues/23302). llama.cpp docs: stochastic sampling may differ across implementations even with a fixed seed; use greedy when exact match is required [docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)
- **Measurement methodology**: report mean accepted length τ (tokens per target forward) *and* wall-clock speedup, per task category, at fixed temperature and batch. Spec-Bench is the common benchmark [Xia et al. survey 2401.07851](https://arxiv.org/pdf/2401.07851). llama.cpp prints `draft acceptance rate` and per-implementation `#gen drafts / #acc drafts / #gen tokens / #acc tokens / dur(b,g,a)`. It has **SPEED-Bench** (`tools/server/bench/speed-bench`) for baseline vs spec runs, and `--spec-synth-rates` / `--spec-synth-len` to inject synthetic acceptance and measure the cost of verification width independently of drafter quality [llama.cpp docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)

### Inferences
- For an agent-driven search, `--spec-synth-len` is valuable. It lets the agent first measure the *verification-cost curve* (time per step vs verify width K) on the actual MoE+CPU setup, then compute the break-even acceptance for each K before trying real drafters.
- Correctness tests in the plugin harness should not use bit-exact match against serial greedy decoding. Use distribution-level or per-step equivalence checks, or a batch-invariant kernel mode.

### Gaps
- No verified data on how llama.cpp CPU MoE kernels scale with verify width (ubatch of 2-8 tokens) on AVX2.

---

## Q6. Sampling, logits-processing and constrained-decoding kernels

### Takeaway
At batch 1 on a memory-bound MoE, sampling cost is small next to the forward pass but not zero for large vocabularies (gpt-oss ~200K, Gemma ~262K, Qwen ~152K): a CPU sort-based top-p plus GPU-to-CPU logits copy can cost tenths of a ms to ms. GPU-side sorting-free sampling (FlashInfer; llama.cpp `--backend-sampling`) and fast grammar engines (XGrammar, llguidance) keep these overheads near zero. Penalties and DRY are cheap if applied only to the token window.

### Cited Findings
- **FlashInfer sorting-free sampling (Mar 2025)** — inverse-transform sampling with CUB reduce/scan, plus **rejection sampling with a pivot** for top-k / top-p / min-p in one fused kernel with no sort. **Dual-pivot rejection sampling** (v0.2.3) at least halves the candidate range per round, so rounds are bounded at O(log 1/ε). "Top-K first" and "Joint" top-k+top-p modes; `top_p_renorm_probs` / `top_k_renorm_probs`. Cuts total sampling time by more than 50% in vLLM on H100; used in vLLM, SGLang, MLC-LLM. The same rejection idea applies to chain/tree speculative sampling | sampler | [FlashInfer blog](https://flashinfer.ai/2025/03/10/sampling.html)
- **llama.cpp backend sampling** — `--backend-sampling` runs supported target samplers on the GPU backend; draft sampling has `--spec-draft-backend-sampling`. Falls back to CPU for unsupported samplers/layouts; not supported with tensor split | sampler | [llama.cpp docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)
- **Truncation samplers** (sampler slot; all lossy by design): top-k; nucleus/top-p [Holtzman et al. 1904.09751](https://arxiv.org/abs/1904.09751); typical sampling [2202.00666](https://arxiv.org/abs/2202.00666); eta/epsilon sampling [2210.15191](https://arxiv.org/abs/2210.15191); Mirostat [2007.14966](https://arxiv.org/abs/2007.14966); **min-p** [2407.01082](https://arxiv.org/abs/2407.01082); top-nσ [2411.07641](https://arxiv.org/abs/2411.07641). llama.cpp also has XTC and DRY samplers ([llama.cpp tools/main README](https://github.com/ggml-org/llama.cpp/tree/master/tools/main)).
- **Interaction with speculation**: exact rejection sampling needs the *post-processed* target distribution p and draft q at every drafted position. Truncation samplers and penalties must be applied identically to the k+1 verify positions (llama.cpp applies the target sampler chain at verify time) | [llama.cpp docs](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)
- **XGrammar (mlc-ai, Nov 2024)** — CFG-constrained decoding that splits vocabulary tokens into context-independent ones (precomputed bitmask) and context-dependent ones (checked at runtime with a persistent stack), overlapped with GPU compute. Up to 100x faster than prior engines, near-zero end-to-end overhead (abstract); default in vLLM/SGLang | logits-processor | [arXiv 2411.15100](https://arxiv.org/abs/2411.15100), [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)
- **Outlines (Jul 2023)** — regex/JSON-schema to FSM indexed over the vocabulary, so per-token mask lookup is O(1); large FSMs mean slow compile | logits-processor | [arXiv 2307.09702](https://arxiv.org/abs/2307.09702), [dottxt-ai/outlines](https://github.com/dottxt-ai/outlines)
- **llguidance (Microsoft/guidance)** — Rust Earley-parser plus lexer engine, fast per-token masks, no heavy precompute. Optional llama.cpp backend (build flag `LLAMA_LLGUIDANCE`); also used by vLLM and OpenAI-style structured output | logits-processor | [guidance-ai/llguidance](https://github.com/guidance-ai/llguidance)
- **GBNF (llama.cpp grammars)** — native grammar format with a CPU stack-based matcher; JSON-schema to GBNF converter | logits-processor | [llama.cpp grammars README](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
- Other constrained-decoding research: SynCode [2403.01632](https://arxiv.org/abs/2403.01632); DOMINO (subword-aligned constraints, also *speculates* forced tokens) [2403.06988](https://arxiv.org/abs/2403.06988)
- **Grammar + speculation**: forced or deterministic grammar spans (JSON keys, braces) can be emitted as free "drafts" (DOMINO-style jump-forward; SGLang's compressed-FSM jump-forward decoding) | drafter | [DOMINO](https://arxiv.org/abs/2403.06988); [SGLang paper 2312.07104](https://arxiv.org/abs/2312.07104)
- **Beam search** — keeps B hypotheses, multiplying KV and compute by B. Rarely used for chat LLMs; SD can be combined with it, but most engines don't support that | sampler/scheduler | (general; no session source)
- **Decoding-time quality methods with extra compute**: contrastive decoding (needs an amateur model, ~2x forward) [2210.15097](https://arxiv.org/abs/2210.15097); DoLa (layer-contrast, cheap) [2309.03883](https://arxiv.org/abs/2309.03883)

### Inferences
- On the user's box, the sampler slot should keep logits on the GPU and return only the token id. For gpt-oss's ~200K vocab, copying fp32 logits is ~0.8 MB per token over PCIe 3.0 (~0.07 ms), plus a CPU sort. Small, but it adds up across spec verify positions (k+1 rows).
- Repetition, frequency and presence penalties are O(window) gather/scatter on logits; negligible on GPU. DRY is more expensive (it scans for sequence repeats) but still CPU-cheap compared with a ~50-100 ms MoE decode step.

### Gaps
- No session-verified per-token microbenchmarks of llama.cpp CPU vs backend sampling for 150-260K vocabularies.

---

## Q7. Does speculation help MoE, especially with experts offloaded to CPU/RAM?

### Takeaway
Results are mixed and depend on regime. (a) With experts resident in fast memory at medium batch, MoE can benefit *more* than dense models (MoESD, up to 2.29x). (b) In serving, the union of experts activated by draft tokens raises verification cost 2-3x and can make SD a net slowdown, up to 1.5x (Cascade). (c) With offloaded experts, naive SD "often slowed inference". Expansion-aware designs (DraftExpert, 1.45x) and SD-as-prefetch-signal (MoE-SpeQ, SP-MoE, SPICE, MOE-SPAC) recover gains. (d) A community RTX 3090 study found **every llama.cpp spec configuration lost to plain decoding on Qwen3.6-35B-A3B**, attributed to the expert-union cost. The user's exact regime (experts computed on a DDR4 CPU) is the least favourable of these.

### Cited Findings
- **MoESD (Tsinghua/Huawei, NeurIPS 2025)** — under medium batch sizes, MoE benefits more from SD than dense models; proposes a "target efficiency" metric; up to 2.29x for Qwen2-57B-A14B | [arXiv 2505.19645](https://arxiv.org/html/2505.19645v2), [NeurIPS poster](https://nips.cc/virtual/2025/poster/119061)
- **Cascade (Jun 2025, MICRO-58)** — draft tokens collectively activate more weights; verification time up 2-3x; slowdowns up to 1.5x; a utility-driven controller limits slowdown to 5% | [arXiv 2506.20675](https://arxiv.org/abs/2506.20675)
- **SpecMoEOff (Aug 2025)** — SD to increase per-expert work when experts are *transferred* CPU to GPU; roofline shows the MoE layer bound by CPU-GPU transfer. CPU chunked attention (MKL) for verification; EAGLE drafter on GPU; optimizer picks draft length. Mixtral-8x7B FP16 on an **RTX 4090D** (23 GB/s CPU-GPU) and an A30: up to 2.5x decode throughput over prior offloading; SD alone gives 1.45-1.53x. Throughput rises then falls as draft length grows from 1 to 10. *Large-batch regime that assumes all experts are loaded anyway* | [arXiv 2508.21706](https://arxiv.org/html/2508.21706v1)
- **MoE-SpeQ (Nov 2025)** — a small quantized on-device draft predicts which experts upcoming tokens need, so they are prefetched from host memory; up to 2.34x over SOTA offloading for Phi-MoE | [arXiv 2511.14102](https://arxiv.org/html/2511.14102v1)
- **SP-MoE (Oct 2025)** — speculative decoding plus expert prefetching | [arXiv 2510.10302](https://arxiv.org/html/2510.10302v2)
- **MOE-SPAC (Mar 2026)** — SD as a lookahead signal for memory management on edge devices | [arXiv 2603.09983](https://arxiv.org/pdf/2603.09983) (abstract-level only)
- **SPICE (Aug 2026)** — speculative prefetching with a draft aligned to the target MoE, targeting the PCIe bottleneck | [emergentmind 2608.21240](https://www.emergentmind.com/papers/2608.21240) (abstract-level only)
- **DraftExpert (Jul 2026)** — expansion-aware self-speculation for offloaded MoE (CPU-to-GPU and Flash-to-NPU). Drafter = shared expert + top-1 routed expert + one small resident trained draft expert per layer. Dynamic truncation stops drafting when a low-confidence token would pull in many new experts (cuts verification cost ~40% vs fixed length). Background prefetch of predicted experts (86-88% hit rate). DeepSeek-V2-Lite and Moonlight-16B-A3B: **1.45x average decode speedup**, 84-87% acceptance. The paper says standard SD *often slowed* offloaded inference, and simple shared+top-r drafting had <50% acceptance | [alphaXiv 2607.24434](https://www.alphaxiv.org/abs/2607.24434v1)
- **Community, RTX 3090 llama.cpp**: "every llama.cpp configuration tested lost to plain decoding" on Qwen3.6-35B-A3B. The author's hypothesis is the expert-union cost (several positions activate a larger union of experts that short drafts can't amortize), stated as consistent-with rather than proven | [hackmd](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090)
- **Fully-resident MoE**: Qwen3.6-35B-A3B MTP only 1.17x on an RTX PRO 6000 [JarvisLabs](https://jarvislabs.ai/blog/qwen36-mtp-llamacpp-rtxpro6000); LFM2.5-8B-A1B DSpark 1.18x on M4 Max vs 2.54x on H100, blamed on llama.cpp MoE [Liquid AI](https://www.liquid.ai/blog/lfm2-5-dspark); DFlash Qwen3-Coder-30B-A3B 3.5x on B200 SGLang [DFlash paper](https://arxiv.org/html/2602.06036v1)

### Inferences
- **Cost model for the user's regime (inference, not measured).** A decode step costs about (GPU part: attention, dense/shared layers, ~fixed) + (CPU part ∝ bytes of distinct experts touched). With verify width K+1, the CPU part scales with |∪ experts over K+1 tokens|, between 1x (perfect overlap) and (K+1)x (disjoint). SD wins only if E[accepted+1] > (step cost at K+1)/(step cost at 1). Because the CPU part dominates, break-even needs high acceptance *and* high expert overlap between adjacent tokens. Models with fewer routed experts per token (gpt-oss: top-4 of 128) or strong expert locality fare better than wide-top-k models. The agent should measure expert-union growth per layer for real text on each model; llama.cpp's `--spec-synth-len` makes this direct.
- A secondary effect could favour SD even on CPU experts: the CPU expert GEMM for 2-4 tokens may still be memory-bound (AVX2 FMA throughput far exceeds 40 GB/s × 2 flops/byte at Q4), so experts shared between tokens are almost free to compute. Per-step fixed overheads (graph launch, PCIe sync of activations twice per layer, sampling) also amortize.
- High-value plugin ideas for this rig: (1) expansion-aware drafting and truncation (DraftExpert / DSpark `conf-min` / Cascade utility controller); (2) router-aware tree pruning: prefer draft branches whose predicted experts overlap the current union; (3) use draft tokens as an expert-prefetch or cache-pinning signal for a VRAM expert cache (MoE-SpeQ / SPICE style), where ~10-14 GB of spare VRAM could cache hot experts; (4) n-gram drafting on copy-heavy spans, where long accepted runs amortize the union.

### Gaps
- No published measurement for llama.cpp's specific topology (experts *computed* on CPU, not streamed) with speculation on a 4090 + DDR4 system; the 3090 study gives no numbers for its MoE runs.
- No per-model data on expert overlap between consecutive tokens for Qwen3.6-35B-A3B, Gemma 4 26B-A4B, gpt-oss-120b or GLM-5.3-Flash.

---

## Q8. llama.cpp spec-type mapping (pinned v0.7.0) and consumer-GPU numbers

### Takeaway
Every llama.cpp `--spec-type` maps onto a catalog entry above. Consumer-GPU evidence: MTP and DFlash give +50-60% on a *dense* 27B fully in VRAM on a 3090; MoE gains are small (1.17x resident) or negative (3090, Qwen3.6-35B-A3B); n-gram can be −17% on non-repetitive text. Default draft length is 3, and short drafts (n-max 2-4) were best on consumer GPUs.

### Cited Findings
- Mapping (from [llama.cpp docs/speculative.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md)): `none`; `draft-simple` = Leviathan/Chen classic; `draft-eagle3` = EAGLE-3; `draft-mtp` = native MTP heads (DeepSeek-V3-style); `draft-dflash` = DFlash block diffusion; `draft-dspark` = DSpark (DFlash + Markov head, confidence truncation, Qwen3-only); `ngram-simple` = prompt-lookup; `ngram-map-k` / `ngram-map-k4v` = hashed n-gram maps with hit thresholds (ANPD-like); `ngram-mod` = shared-pool hashed n-gram; `ngram-cache` = n-gram statistics (lookup-decoding).
- Combinations are allowed; draftless types take precedence over a draft model; `--spec-default` enables `ngram-mod` | same source
- Consumer data points: RTX 3090 dense 27B: MTP +59.8% (n=2), DFlash2 +51.9% (n=4), external 0.8B draft −29.8%; Qwen3.6-35B-A3B: all configs lost [hackmd](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090). Unknown GPU at 128K context: MTP +5% decode / −22% prefill; ngram-cache −17% [HF space](https://victor-mtp-on-hf-endpoints.static.hf.space/index.html). RX 9070 XT: Qwen 3.6 27B at 46 tok/s with llama.cpp + MTP (title claim only) [piefed](https://piefed.jeena.net/post/394664). A 4090 commenter: dense Qwen 3.6-27B 25-40 tok/s llama.cpp Q4 vs 50+ with vLLM MTP (anecdotal, via search summary).

### Inferences
- For the agent's search space per model: {none, draft-mtp n∈1..4, draft-dflash/dspark if a draft exists (n up to block size, conf-min sweep), ngram-mod / ngram-map-k with long n-min, and ngram + head hybrids}, each measured with SPEED-Bench on the user's actual workload mix. Expect the baseline (`none`) to be a strong competitor for MoE-on-CPU.
- MTP being silently disabled with `--parallel > 1` or `--cache-reuse` means harness configs must check the spec stats line, not just the flags.

### Gaps
- Could not verify a llama.cpp release literally tagged "v0.7.0 (Oct 2026)"; findings are from the current master docs.
- No RTX 4090 + CPU-offloaded-MoE speculative benchmarks found anywhere.
