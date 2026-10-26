# AutoInference: A Slotted Inference Engine Tuned by Search

**Status:** draft specification, pre-implementation. Nothing here is built yet.

AutoInference is an LLM inference engine whose every stage is a replaceable plugin, plus a search loop that finds the fastest correct set of plugins for a given machine, model and workload. The search is Monte Carlo in spirit: it samples a change, plugs it in, verifies it, times it, keeps it if it wins, and writes every trial to a shared record. The record is what makes the system improve over time: on a machine or model it has not seen, the search starts from what won on the most similar setups instead of from scratch.

The engine is hardware-agnostic. It has no built-in target machine; the machine is an input, described by a measured profile. Kernel leaderboards are one use of it: each leaderboard problem is a fixed machine plus a fixed slot, and the same search, harness and record apply.

The technique catalog distilled from `research_notes/` supplies the search space. Its job is coverage: an LLM agent asked to "make inference faster" reaches for the handful of techniques it remembers, while the catalog enumerates roughly a thousand, each tagged with the slot it fills, the hardware it needs, whether it is lossless, and the reference implementation it must beat.

---

## 1. The claim this project has to prove

Generic inference libraries choose kernels, placements and policies with defaults tuned for a small set of datacenter configurations. The claim is that, across a diverse population of machines (different accelerators, memory sizes, interconnects, CPU instruction sets and operating systems) and models (dense, MoE, hybrid recurrent), a per-setup search over the catalog finds configurations that are substantially faster than library defaults at equal output quality, and that the winning configurations differ from one setup to another.

Both halves matter. If tuned configurations are not much faster, the project has no value. If they are faster but identical everywhere, a better default would do and the search is unnecessary. The first milestones (section 11) are designed to test this cheaply before heavy investment.

**Success measures:**

- Speedup of the tuned configuration over each library's default, per (machine, model, workload), with the quality gate passed.
- Diversity of winners: how often the best configuration on one setup is not the best on another.
- Warm-start value: trials needed to reach a given fraction of the final speedup, with and without the record.
- External scores on public kernel benchmarks, under their own rules.

---

## 2. System overview

```
             ┌────────────────────────── Technique catalog ─────────────────────────┐
             │  techniques · implementations · validation targets · tactics          │
             └───────────────┬──────────────────────────────────────┬───────────────┘
                             │ candidates per slot                  │ technique → plugin
                             ▼                                      ▼
  Machine profile ──► ┌──────────────┐   proposals   ┌────────────────────────────┐
  Model + workload ─► │   Searcher   │ ────────────► │ Plugin builder (LLM agent) │
  Record (priors) ──► │ (Monte Carlo)│ ◄──────────── │  writes / adapts plugins   │
                      └──────┬───────┘   new plugins └────────────────────────────┘
                             │ configuration to try
                             ▼
                      ┌──────────────┐   staged gates   ┌─────────────────────────┐
                      │  Core engine │ ───────────────► │ Harness: verify + time  │
                      │ (slot host)  │ ◄─────────────── │ (isolated subprocess)   │
                      └──────────────┘   verdict, times └───────────┬─────────────┘
                                                                    │ every trial
                                                                    ▼
                                                            ┌──────────────┐
                                                            │    Record    │
                                                            └──────────────┘
```

| Component | Responsibility |
|---|---|
| Core engine | Loads a model description, lays it out as a graph of slot instances, runs inference by calling whichever plugin fills each slot. Owns nothing performance-critical itself. |
| Plugins | Implementations of a slot: kernels, policies, offline transforms or system settings. Each ships a manifest (section 4). |
| Technique catalog | Structured knowledge base built from the research notes (section 6). |
| Searcher | Proposes configurations, schedules evaluations, decides what to keep (section 7). |
| Plugin builder | LLM agents that turn a catalog technique into a working plugin for a slot, or adapt an existing one, when no implementation exists for the target platform. |
| Harness | Runs each candidate in isolation through staged correctness and timing gates (section 8). |
| Machine profiler | Measures the machine at startup and produces the profile that keys the record (section 9). |
| Record | Append-only store of every trial and its outcome, plus derived priors (section 9). |

---

## 3. The slot model

### 3.1 Principles

1. **Mechanism is separate from policy.** A kernel is a pure function with a declared shape and dtype contract. A policy is a pure function from observed runtime state to a plan. They are searched and verified differently.
2. **Slots are device-neutral.** A slot names an operation (`linear`, `attn-decode`), not a device. Device is a property of the plugin and a decision of the `placement` slot. The same model can run one `moe-experts` instance on an accelerator and another on the CPU, each filled by a different plugin.
3. **A slot has many instances.** One model graph contains many instances of each slot, distinguished by layer type, phase (prefill or decode), device and shape bucket. Plugins are assigned to *instance groups*, so sliding-window attention layers and full attention layers can use different kernels.
4. **Fusion boundaries are searchable.** A plugin may claim a contiguous span of slot instances (for example norm, then QKV projection, then rotary embedding). Granularity is a search dimension, not a fixed property of the engine, because isolated kernel wins often fail to survive composition.
5. **Every slot has a reference plugin.** A straightforward, correct, slow implementation in high precision serves as the oracle for correctness checks.
6. **Specialization is legal when declared.** A plugin may hard-code shapes, head counts or dimensions, provided its manifest declares exactly what it assumes. The harness then fuzzes only within those declared bounds.

### 3.2 Slot kinds

| Kind | What a plugin is | Interface shape | Default verification gate |
|---|---|---|---|
| `kernel` | A compute routine for one operation (or a fused span) on one device | `run(inputs, outputs, workspace, stream) → status`, plus optional `prepare(weights)` done once, untimed | Numerical tolerance against the reference, calibrated per dtype |
| `policy` | A decision function the runtime consults | `plan(state) → plan`; pure, deterministic given state | Output equivalence against the reference policy (lossless policies) |
| `transform` | An offline change to weights or model files | `apply(model) → model'`; runs once before serving | Quality gate: divergence from the reference model on held-out prompts, plus task evals |
| `config` | A system or runtime setting | `apply(environment)`; may require restart | Output equivalence; no correctness risk beyond that |

### 3.3 Parts and slots

The decomposition below comes from reading the source of four production engines (DeepInfer, llama.cpp, vLLM and SGLang) and recording every subsystem that exists as its own module, class family or registry in at least one of them. A **part** is such a subsystem. A **slot** is the unit the search assigns a plugin to; most parts contain one slot, and the op library contains many.

A part was kept only if it has a stable interface to its neighbours and can be verified on its own. Parts present in only one engine were kept when they exist there as a distinct subsystem (for example explicit placement in llama.cpp, or a draft-length planner in DeepInfer's scheduler).

**Serving.** Infrastructure the search uses but does not tune.

| Part | Slots | Kind | Responsibility |
|---|---|---|---|
| API server | none | infrastructure | HTTP and CLI entry points, request validation, streaming. |
| Renderer | none | infrastructure | Chat templates, tokenization, incremental detokenization, output parsing. |
| Multimodal front | `mm-preprocess`, `mm-encoder` | kernel | Media decoding and preprocessing; the vision or audio encoder forward pass; encoder output cache. |
| Replica coordinator | `replica-balance` | policy | Spreads requests across engine replicas. Multi-replica only. |

**Control**

| Part | Slots | Kind | Responsibility |
|---|---|---|---|
| Scheduler | `scheduler` | policy | Admission, continuous batching, chunked prefill, priorities, preemption. Produces a frozen batch plan each step. |
| Overlap | `overlap` | policy | Whether and how host scheduling of step N+1 overlaps device execution of step N, and how lagging host state is corrected. |
| Speculation planner | `spec-planner` | policy | Chooses draft length or turns speculation off per request, from acceptance statistics and measured verify cost. |

**Memory**

| Part | Slots | Kind | Responsibility |
|---|---|---|---|
| KV and state allocator | `kv-allocator`, `kv-dtype` | policy, kernel | Pages or slots for each attention type (full, sliding window, recurrent state, hybrid); storage format of cached entries. |
| Prefix cache | `prefix-cache` | policy | Finding, sharing and evicting reusable prefixes. Kept separate from the allocator. |
| KV tiers | `kv-tier`, `kv-select` | policy | Host and disk tiers, transfer connectors, sparse selection or eviction of cached entries. |
| Activation memory | `workspace` | policy | Allocation and reuse of activation and scratch buffers. |

**Execution**

| Part | Slots | Kind | Responsibility |
|---|---|---|---|
| Input preparation | `input-prep` | kernel, policy | Turns a batch plan into what kernels consume: positions, page tables, attention metadata. |
| Model runner | none | infrastructure | Holds the model and drives one forward step. |
| Executor | `executor` | policy | Launch strategy, graph capture or compilation, streams, synchronization mode, threads. |
| Placement | `placement` | policy | Device or buffer for each tensor, by declarative rules over tensor names. |
| Expert residency | `expert-cache`, `expert-balance` | policy | Which experts stay on fast memory, and rebalancing expert load across devices. |
| Data movement | `transfer`, `offload-scheduler` | kernel, policy | Copies between devices; whether to compute where data lives or move it. |
| Parallelism | `parallelism`, `collective` | policy, kernel | Splitting the model across devices; collective operations. Multi-device only. |

**Model**

| Part | Slots | Kind | Responsibility |
|---|---|---|---|
| Architecture registry | none | infrastructure | Maps a model's configuration to a graph of slot instances. |
| Weight loader | `loader` | config | Reads weight files into memory: mmap or direct read, staging. |
| Weight format | `weight-format`, `weight-rotation`, `expert-prune` | transform | Per-tensor numeric format, offline rotations, opt-in lossy expert reduction. |
| Adapters | `adapter` | kernel | LoRA and similar low-rank adapters applied to linear and expert layers. |
| Op library | see below | kernel | The model's arithmetic. |

The op library is where most kernel slots live. Each op family is a separate slot, and each slot has instance groups (section 3.4):

| Op slot | Contract (in → out) |
|---|---|
| `embedding` | token ids → hidden states |
| `linear` | activations, weights in some format → activations |
| `norm` | hidden states → normalized hidden states (optionally fused with residual add) |
| `rotary` | queries, keys, positions → rotated queries and keys |
| `kv-write` | new keys and values, slot mapping → updated cache |
| `attn-prefill` | queries, cache view, mask spec → outputs; per layer type |
| `attn-decode` | queries, cache view, mask spec → outputs; per layer type, including tree masks for verification |
| `recurrent-mixer` | input, state → output, new state; chunked prefill and recurrent decode |
| `activation` | gated or plain activation functions |
| `moe-router` | hidden states → expert ids and weights |
| `moe-dispatch` | tokens, routing → tokens grouped by expert (and across devices) |
| `moe-experts` | grouped tokens, expert weights → expert outputs |
| `moe-combine` | expert outputs, routing weights → combined hidden states |
| `lm-head` | hidden states → logits |

**Decoding**

| Part | Slots | Kind | Responsibility |
|---|---|---|---|
| Sampler | `sampler`, `logits-processor` | kernel | Penalties, temperature, top-k, top-p and similar, then token selection. |
| Structured output | `grammar` | kernel, policy | Grammar compilation and per-step token masks. |
| Speculative drafter | `spec-drafter` | kernel, policy | Proposes draft tokens (draft model, extra heads, n-gram or suffix lookup, the model's own multi-token heads). |
| Speculative verifier | `spec-verifier` | kernel | Accepts or rejects drafts while keeping the output distribution exact. |

**Platform**

| Part | Slots | Kind | Responsibility |
|---|---|---|---|
| Platform layer | none | infrastructure | Device discovery and per-device defaults for every other part. |
| System config | `os-config` | config | Memory limits, huge pages, power and clock settings, file locations. |
| Observability | none | infrastructure | Per-part timers and counters. The search reads its signals from here. |

**Out of scope for plugins:** architecture changes that require retraining (new attention types, trained sparsity, retrofitted layers). The catalog records them so the engine can host such models when they exist, but the search never proposes them.

### 3.4 Slot instances and instance groups

The engine derives slot instances from the model description. Each instance carries a key:

```
(slot, layer_type, phase, device, shape_bucket)
```

Shape buckets are coarse ranges (for example tokens-per-step or tokens-per-expert powers of two) so that a plugin choice generalizes across nearby shapes. An **instance group** is a set of instances that share one plugin assignment. By default, instances with the same key form a group; the searcher may split or merge groups.

---

## 4. Plugin manifest

Every plugin ships a manifest. The harness enforces it; the searcher reads it to decide what is legal.

| Field | Meaning |
|---|---|
| `plugin_id`, `version` | Stable identity. |
| `technique_id` | Link to the catalog entry it implements. |
| `claims` | The slot or contiguous span of slots it fills. A span plugin replaces every instance in the span. |
| `kind` | kernel, policy, transform or config. |
| `devices` | Device classes it runs on. |
| `requires` | Capability predicates over the machine profile (instruction sets, tensor-core dtypes, memory features, minimum shared memory, OS). |
| `accepts` / `provides` | Layout and format tokens it consumes and produces (section 5). |
| `assumes` | Shape and parameter bounds it is specialized for. Anything not listed must be handled or rejected cleanly. |
| `params` | Tunable parameter space (tile sizes, thresholds, cache sizes) with types and ranges. |
| `lossless_class` | bit-exact, tolerance-exact, distribution-preserving, or lossy-gated. |
| `gate` | Which verification gates apply, if stricter than the slot default. |
| `deps` | External libraries it calls, with versions. Calls outside this list are forbidden at runtime. |
| `memory` | Declared workspace and persistent memory as a function of shape. Measured peaks above the declaration fail the plugin. |
| `provenance`, `licence` | Source of the code and its licence. |

A plugin must be able to report `unsupported` for a shape outside its `assumes` bounds rather than produce wrong output. The engine then falls back to the next plugin in the instance group's fallback chain, ending at the reference plugin.

---

## 5. Compatibility between slots

Slots constrain one another. The KV storage format limits which attention kernels can read it; speculative verification needs attention plugins that support tree masks and recurrent plugins that can roll state back; any plugin that grows its memory use changes what `placement` can keep resident. These interactions are what make per-slot greedy search get stuck.

Compatibility is expressed as **tokens**, not as hand-written pairwise rules:

- A plugin `provides` tokens describing the data it produces (for example `kv:fp8-e4m3:paged:page=N`, `weights:int4-group:g=N:layout=L`).
- A plugin `accepts` a set of tokens it can consume.
- A configuration is legal only if, for every edge in the slot graph, the producer's token is in the consumer's accepted set.
- Capability tokens (`supports:tree-mask`, `supports:state-rollback`, `graph-capturable`) are checked the same way against slots that need them.

The searcher uses these constraints in two ways: it never proposes illegal configurations, and when it changes a producer it can propose a **joint move** that also swaps the consumers that no longer accept the new token.

---

## 6. Technique catalog

The catalog is built from the twenty note files in `research_notes/`. The notes hold roughly three thousand rows, but many describe the same mechanism under different engine-specific names; a deduplication pass is expected to leave on the order of a thousand distinct techniques grouped into roughly 150 families.

### 6.1 Record types

| Type | What it is |
|---|---|
| Technique | A mechanism or policy, independent of any implementation. |
| Implementation | One concrete code path for a technique: engine or library, version, entry point, status, source. Used as incumbent or as porting material. |
| Validation target | A public benchmark problem with its shapes, tolerances and timing rules. |
| Tactic | A recurring trick attached to a technique family (a load pattern, a tiling choice), often harvested from competition write-ups. |

### 6.2 Technique fields

`technique_id`, `canonical_name`, `aliases[]`, `family`, `primary_slot`, `secondary_slots[]`, `kind`, `lossless_class`, `requires` (the same capability predicates plugins use), `model_features[]` (which model properties make it applicable, such as MoE, sliding window, recurrent layers), `training_required`, `phase`, `claimed_gains[]` (each with value, hardware, workload and source, so a gain measured on one machine is never applied to another), `evidence_grade`, `implementations[]`, `validation_targets[]`, `licence`, `source_rows[]`.

Machine-specific judgements in the earlier reports (relevance to one box, whether it runs on one GPU generation) are replaced by `requires` predicates evaluated against whatever machine profile is current.

### 6.3 Deduplication

The dedup key is a mechanism fingerprint rather than a name:

```
(primary_slot, operation, numeric format or algorithm, granularity, phase)
```

After fingerprinting, passes merge normalized aliases, attach research papers to the engine implementations that cite them (the paper becomes evidence, the engine code becomes the implementation), and fold library implementations into the technique entries they implement.

### 6.4 From technique to plugin

A technique becomes searchable only when a plugin exists for it on the current platform. Three routes, in order of preference:

1. **Wrap** an existing implementation from a library or engine (for example an attention backend, a quantized GEMM, a fused MoE kernel) behind the slot contract.
2. **Port** an implementation that exists for a different platform, using it as the specification.
3. **Generate** a new plugin with an LLM agent from the technique description, the slot contract and the reference plugin.

Generated plugins carry higher risk and always pass the full gate stack, including the audit step in section 8.3.

---

## 7. Search

### 7.1 State and moves

- **Configuration:** an assignment of `(plugin, params)` to every instance group, plus the group partition itself.
- **Incumbent:** the best verified configuration so far for the current (machine, model, workload).
- **Moves** the searcher can propose:
  - swap the plugin of one instance group;
  - change parameters of one plugin;
  - split or merge instance groups;
  - replace a span of slots with a fused plugin, or break a fused plugin back into parts;
  - a joint move across slots coupled by compatibility tokens or by shared memory budget.

### 7.2 Sampling

Proposals are sampled, not enumerated. The sampling weights come from three sources:

1. **Priors from the record:** how often each plugin won on similar setups, for the same slot and instance key.
2. **Catalog metadata:** evidence grade, claimed gains on comparable hardware, lossless class.
3. **Live profile data:** which slot instances dominate measured step time right now. Time spent there is where a move can pay.

A per-slot bandit (Thompson sampling over plugins, with parameter search inside a chosen plugin) is the starting design. It is simple, still Monte Carlo, and improves as the record grows. More structured search (tree search over joint moves, learned cost models) can replace it later without changing the rest of the system.

### 7.3 Staged evaluation

Trials are expensive, so each candidate passes a sequence of increasingly costly stages and is dropped at the first failure:

| Stage | Cost | What it checks |
|---|---|---|
| 0. Static | negligible | Manifest predicates against the profile; compatibility tokens; forbidden calls. |
| 1. Unit correctness | seconds | Plugin against the reference on fuzzed inputs within its declared bounds. |
| 2. Microbenchmark | seconds | Plugin time on the instance's real shapes, under the timing protocol. Candidates slower than the incumbent by a margin stop here. |
| 3. Composed correctness | tens of seconds | The full configuration, under graph capture where applicable, against a reference run. |
| 4. End-to-end timing | minutes | Real model, real workload, interleaved A/B against the incumbent. |
| 5. Quality gate | minutes to hours | Only for lossy plugins: divergence and task evaluations, including long-context retrieval for KV-related slots. |

Budget across candidates uses successive halving: many candidates reach stage 2, few reach stage 4.

### 7.4 Acceptance

A candidate replaces the incumbent only if its end-to-end gain exceeds the measured noise band with statistical confidence, from interleaved repeated runs. The searcher may keep a separate exploration state that accepts slightly worse configurations (annealing-style) to escape local optima, but the published incumbent only ever improves.

### 7.5 Escalation order

The search spends budget in this order by default, because the cost and the risk grow down the list:

1. `config` and `policy` slots using existing plugins (lossless, cheap to verify).
2. Kernel swaps between existing wrapped implementations.
3. Ported or generated kernels.
4. Lossy transforms and policies, only when the user enables them.

---

## 8. Measurement and verification

### 8.1 Timing protocol

Public kernel benchmarks have converged on one protocol, and the harness adopts it so that local scores predict external ones:

- each candidate runs in its own subprocess with a timeout;
- device-side timing events on a single declared stream, with detection of work launched on other streams;
- warmup iterations, then a fixed number of timed iterations, repeated across several trials; report the median and spread;
- cache flush between iterations when measuring cold-weight behaviour;
- clocks and temperature locked or at least logged; trials discarded when they drift;
- interleaved A/B ordering for end-to-end comparisons;
- peak memory recorded for every run.

### 8.2 Gates by lossless class

| Class | Gate |
|---|---|
| bit-exact | Identical outputs to the reference under a deterministic, batch-invariant reference mode. |
| tolerance-exact | Outputs within tolerances calibrated by running the reference in the slot dtype against a higher-precision reference; several hidden input distributions. |
| distribution-preserving | Statistical tests on token distributions given fixed logits (speculative decoding, batching changes), not exact text match. |
| lossy-gated | Divergence from the reference model on held-out prompts, task evaluations, long-context retrieval, multi-instruction adherence. Perplexity alone is not sufficient. |

### 8.3 Anti-cheating rules

Generated code has a documented history of passing checks without being correct. The harness enforces:

- hidden input distributions the plugin author never sees (scaled, negated, near-zero, special values);
- shape fuzzing within declared bounds;
- fresh output buffers poisoned before every call, and varying input addresses, to catch stale or cached outputs;
- no calls to libraries outside the manifest's `deps`; static scan plus runtime interception;
- comparison against the strongest available incumbent, never a weak baseline;
- composed testing with neighbouring slots and under graph capture, not only standalone;
- real routing traces for data-dependent operations, since random inputs misrepresent expert selection and cache hit rates;
- assertions that the optimization actually engaged, read from runtime counters (cache hit rate, acceptance rate, graph splits) rather than from flags;
- an LLM review of every winning generated plugin as a second gate, never the only one.

---

## 9. Machine profile and the record

### 9.1 Machine profile

Measured at startup and cached, re-measured when the environment changes. Specifications from vendor sheets are recorded but never used in place of measurements.

- Devices: type, count, memory capacity, compute capability or feature flags.
- Measured bandwidths: device memory, host memory (read-only streaming, not copy), each host-device link, storage that holds model files.
- Measured compute peaks per dtype that the device supports.
- Launch and synchronization latency.
- Host CPU: core counts, instruction set extensions, cache sizes, NUMA layout.
- Limits: pinned-memory cap, memory available to the process, virtualization layer.
- Software: OS, driver and runtime versions, compiler versions.

### 9.2 Record schema

One row per trial:

| Field | Content |
|---|---|
| `profile_id` | Hash of the machine profile, with the full profile stored once. |
| `model_id` | Model architecture and weight format. |
| `workload_id` | Workload description: context lengths, batch or concurrency, prefill/decode mix. |
| `configuration` | Full plugin and parameter assignment, plus instance group partition. |
| `move` | What changed relative to the parent configuration. |
| `stage_reached`, `verdict` | Where it stopped and why. |
| `timings` | Per-stage measurements with spread. |
| `counters` | Runtime counters (hit rates, acceptance, peak memory, splits). |
| `quality` | Gate results where applicable. |
| `code_hashes` | Exact plugin versions used. |

### 9.3 Warm start

For a new setup, the searcher finds the most similar previous setups by distance over profile features (measured bandwidths and compute, not names), model features (layer types, expert counts, sizes) and workload features. It seeds the initial incumbent from their best configurations, filtered for legality on the new profile, and seeds the bandit priors from their per-slot win rates. The cold-start fallback is library defaults plus the reference plugins.

---

## 10. External benchmarks

Each public benchmark is treated as a fixed machine profile plus a fixed slot instance (or a whole engine, for serving tracks). An adapter maps the benchmark's problem definition onto a slot contract, runs the search against the benchmark's own reference and rules, and exports a submission in its format.

| Benchmark family | Adapter |
|---|---|
| Task-file benchmarks with a Python reference and `generate_input` / `check` hooks (GPU MODE reference kernels and challenges built on the same judge) | One adapter for the task-file layout, plus optional checks some challenges add: graph capturability and an untimed weight-preparation step. |
| Subprocess benchmarks with calibrated tolerances and speed-of-light scoring (SOL-ExecBench, FastKernels, KernelBench-Verified, FlashInfer-Bench) | One adapter implementing their shared protocol: subprocess isolation, cache flush, fixed warmup/timed counts, hidden distributions, peak memory, reward-hack scans. |
| Serving benchmarks that score a patched engine end to end | Out of scope for the first adapters; revisit once the engine itself is competitive. |

Final scores require running on the benchmark's own hardware. Local runs on other hardware give correctness and relative ranking only. Dataset licences differ (some attach obligations to training on submissions), so the catalog records the licence for every validation target and the plugin builder respects it.

---

## 11. Milestones

| Milestone | Delivers | Proves |
|---|---|---|
| M0 Spec | This document, the canonical slot list and the manifest format. | Shared vocabulary. |
| M1 Host (MVP) | Core engine with the full part structure and reference plugins, running one small dense model and one small MoE model end to end (section 12). | The slot decomposition holds for real models. |
| M2 Measurement | Machine profiler, harness with the timing protocol and stages 0–4, record store. | Timings are stable enough to compare. |
| M3 Wrapped search | Incumbent plugins wrapped from existing libraries for the main slots, and the bandit searcher. Run on at least three different machines. | The core claim: tuned beats default, and winners differ by setup. |
| M4 Warm start | Similarity search over the record and seeded priors. | Fewer trials to reach a good configuration. |
| M5 Plugin builder | LLM agents that port and generate plugins from catalog entries, behind the full gate stack. | Coverage beyond what libraries ship. |
| M6 Benchmarks | Both benchmark adapters; first external submissions. | External validation. |
| M7 Lossy | Lossy slots behind quality gates, opt-in. | Larger gains where users accept approximation. |

The catalog build (section 6) runs in parallel from M1 onward, because M3 needs it to choose which incumbents to wrap first.

---

## 12. M1: the MVP engine

The first build is an engine that is correct and slow, with the full part structure from section 3.3 in place. Its purpose is to prove the decomposition on real models and to give every later plugin a reference to be checked against. Performance is explicitly not a goal.

### 12.1 Scope

- **Language and runtime:** Python 3.11+ and PyTorch only. It runs on CPU on Windows, macOS and Linux, and on CUDA or Apple MPS when present, with no compiled extensions.
- **Models:** the Qwen3 dense architecture and the Qwen3 MoE architecture. Correctness tests use tiny randomly initialized configurations of both, so they need no downloads and run on CPU. The real-weight check uses the smallest Qwen3 dense checkpoint; a real MoE checkpoint small enough for one consumer GPU follows once the MoE path passes the random-weight tests.
- **Precision:** FP32 and BF16 weights. No quantized formats.
- **Single device, single replica.**

### 12.2 How parts are defined and swapped

Each part and each op slot has three things:

1. **An interface**, defined as a Python `Protocol` with typed inputs and outputs.
2. **A registry**, keyed by slot name, that maps implementation names to classes. Implementations register themselves with a decorator.
3. **A reference implementation**, written for readability and obvious correctness: plain PyTorch, explicit loops where they make the logic clearer, FP32 accumulation.

An engine configuration file selects an implementation for every slot. Op slots can be overridden per instance group by matching on the instance key (layer type, phase, device). Swapping a part means editing the configuration; no other code changes. Every implementation declares the same manifest fields as section 4, even where the MVP does not yet enforce them, so the format is exercised from the start.

The engine resolves the configuration once at startup into an **op table**: for each slot instance, the chosen implementation. Model code never imports an implementation directly; it calls through the op table. This is what lets one layer use one attention kernel and the next layer another.

### 12.3 Data passed between parts

| Type | Produced by | Consumed by | Content |
|---|---|---|---|
| `Request` | API server, renderer | scheduler | Token ids, sampling parameters, stop conditions. |
| `BatchPlan` | scheduler | input preparation, model runner | Frozen per-step plan: which sequences run, their new token ids, phase per sequence, page tables. |
| `ForwardInputs` | input preparation | model runner, op slots | Flattened token ids, positions, cache slot mapping, attention metadata per layer type. |
| `ForwardOutput` | model runner, sampler | scheduler | Logits for the positions that need them, then sampled token ids. |
| `StepEvents` | scheduler | renderer, API server | New tokens and finished requests, for streaming and detokenization. |

### 12.4 One engine step

1. The scheduler admits waiting requests within its token budget, asks the KV allocator for pages, and emits a `BatchPlan`.
2. Input preparation turns the plan into `ForwardInputs`.
3. The model runner executes the architecture's slot graph through the op table: embedding, then per layer norm, linear projections, rotary, KV write, attention (prefill or decode by phase), output projection, residual, norm, and either the MLP or the MoE sequence (router, dispatch, experts, combine); then the final norm and LM head.
4. The sampler applies logits processors and selects tokens.
5. The scheduler commits progress, frees pages of finished requests, and emits `StepEvents`.

Observability wraps every slot call with a timer and counter when enabled, so each run reports where time went, per slot and per instance group.

### 12.5 What the MVP implements

| Part | MVP implementations |
|---|---|
| API server | Command-line generation and a minimal OpenAI-compatible chat endpoint. |
| Renderer | Tokenizer and chat template from the model's own tokenizer files; incremental detokenization. |
| Scheduler | First-come-first-served continuous batching with chunked prefill. |
| Overlap | `none`. |
| KV and state allocator | Paged allocator for full-attention layers; BF16 or FP32 cache. |
| Prefix cache | `none`, then a block-hash prefix cache. |
| Input preparation | Reference implementation on the host. |
| Executor | Eager execution. |
| Placement | Everything on one device. |
| Weight loader | Safetensors reader. |
| Op library | A reference implementation for every op slot used by Qwen3 dense and MoE. A second implementation for at least three slots (for example `attn-decode` through PyTorch's fused attention, `linear` through the library call, `moe-experts` grouped by expert instead of looped per token) so swapping is tested on real alternatives. |
| Sampler | Greedy, temperature, top-k, top-p; repetition penalty as a logits processor. |
| Speculative drafter, verifier, planner | Interfaces only, with `off` as the single implementation. |
| Observability | Per-slot timers and counters, printed as a summary or written as JSON. |

Every other part from section 3.3 exists as an interface with no implementation, or with a single trivial one, so later work adds implementations without changing the structure.

### 12.6 Verification

- **Reference match:** for tiny random configurations of both architectures, the engine's logits match Hugging Face transformers within FP32 tolerance, and greedy generation matches token for token.
- **Real weights:** greedy generation on the smallest Qwen3 dense checkpoint matches Hugging Face transformers token for token for a set of prompts.
- **Swap equivalence:** for each slot with two implementations, swapping through configuration leaves greedy output identical and logits within tolerance.
- **Batching equivalence:** generating several requests together gives the same tokens as generating each alone.
- **Platforms:** the CPU test suite passes on Windows, macOS and Linux.

### 12.7 Done when

All of section 12.6 passes, a run can print its per-slot timing summary, and adding a new implementation of any op slot takes one new file plus a configuration line.

---

## 13. Platform support

The orchestration layer (engine core, searcher, harness, record) must run on Windows, macOS and Linux. Plugins declare the platforms they support through `requires`; the searcher only proposes plugins legal on the current profile, so a machine with no supported accelerator still runs the CPU reference and CPU plugins.

Expected backend coverage, to be confirmed during M1:

| Platform | Accelerator backends | CPU backends |
|---|---|---|
| Linux | CUDA, ROCm | x86 and Arm SIMD |
| Windows | CUDA native; Linux backends through WSL2 where native builds are impractical | x86 SIMD |
| macOS | Metal | Arm SIMD |

Platform-specific measurements (pinned-memory limits under virtualization, unified-memory behaviour) belong in the machine profile, not in hard-coded branches.

---

## 14. Open decisions

1. **Host language and graph format.** Python orchestration with plugins in compiled languages and kernel DSLs is the working assumption. The model description format (custom, or derived from an existing model library's configs) is undecided.
2. **Own engine or wrap existing engines.** The working assumption is a minimal engine of our own, with existing engines' kernels wrapped as plugins. Wrapping whole engines limits which slots can actually be swapped.
3. **Granularity of instance groups.** How fine the default shape buckets are, and whether the searcher can split groups freely or only along predefined axes.
4. **Searcher algorithm beyond the bandit baseline.** Tree search, evolutionary search or learned cost models, and when to switch.
5. **Similarity metric for warm starts.** Which profile, model and workload features carry weight, and whether the weights are learned from the record.
6. **Lossy scope.** Whether lossy slots are in scope for the first public results or deferred entirely.
7. **Record sharing.** Whether records from different users' machines can be pooled, and under what privacy rules.
8. **Plugin languages.** Which kernel languages the plugin builder targets first, given the portability needed for section 12.

---

## 15. Source material

`research_notes/` holds the raw research this spec draws on: technique tables organized by optimization area (`LLM inference optimization techniques/`) and by source engine, venue or benchmark (`Inference techniques by source/`). Those files become catalog input in M1–M3. They contain hardware-specific observations and claimed gains from many different machines; the catalog keeps each gain with the hardware it was measured on and never transfers a number from one platform to another.
