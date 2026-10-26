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

### 3.3 Canonical slot list

This list replaces the two partially overlapping vocabularies in the earlier reports. Device-specific slot names (separate CPU and GPU MoE slots, for example) are merged into device-neutral ones.

**Model compute (kernels)**

| Slot | Contract (in → out) | Notes |
|---|---|---|
| `linear` | activations, weight in some format → activations | Covers dense projections and the LM head. Instance groups split by weight format, device and batch bucket (GEMV-like vs GEMM-like). |
| `moe-experts` | routed tokens, selected expert weights → combined outputs | Covers grouped GEMM, per-expert GEMV, gather/scatter and finalize. Instance groups by device and tokens-per-expert bucket. |
| `router` | hidden states → expert ids and weights | Softmax/sigmoid, top-k, grouping. Lossless unless the plugin changes k (then lossy, see `expert-prune`). |
| `attn-prefill` | Q, K, V or cache, mask spec → output, cache writes | Per layer type: full, sliding window, sinks, gated output, latent (MLA-style), sparse-indexed. |
| `attn-decode` | Q, cache view, mask spec → output | Same per-layer-type variants; also tree masks for speculative verification. |
| `recurrent-mixer` | input, state → output, state' | Linear-attention, delta-rule and state-space layers. Separate chunked-prefill and recurrent-decode instance groups. |
| `fused-elementwise` | element-wise or reduction chains | Norms, residual adds, rotary embedding, activation-gating, quantize/dequantize of activations. Usually claims spans adjacent to other slots. |
| `sampler` | logits, sampling params, optional grammar mask → token ids | Includes penalties and constrained decoding. |
| `collective` | tensors across devices → reduced or exchanged tensors | All-reduce, all-to-all. Only instantiated on multi-device machines. |

**Memory and data movement**

| Slot | Kind | Contract | Notes |
|---|---|---|---|
| `placement` | policy | tensor names, phase, profile → device or buffer per tensor | Declarative rules over tensor and module names. Re-planned whenever another slot changes memory demand. |
| `expert-cache` | policy | expert ids per step → hit or miss, admission, eviction | Must assert it is actually engaged (hit rate logged). |
| `offload-scheduler` | policy | routed counts per layer and step → compute-where-resident vs move-to-device, overlap plan | The crossover point between computing on the host and streaming weights is a learned parameter, not a constant. |
| `transfer` | kernel + policy | copy requests → staged, overlapped copies | Pinned staging buffers, copy-engine use, prefetch distance. |
| `kv-dtype` | kernel | KV write/read in a storage format | Defines the cache layout tokens consumed by attention plugins (section 5). |
| `kv-select` | policy | cache, attention statistics → retained or selected entries | Eviction and sparse retrieval. Lossy unless the plugin proves otherwise. |
| `kv-allocator` | policy | sequences, budget → block tables | Paged, per-layer-type pools, prefix sharing. |
| `kv-tier` | policy | prefix blocks → device, host or disk tiers | Prefix persistence and reuse. |
| `loader` | config | files → resident weights | mmap vs direct read, staging, file location. |

**Runtime control**

| Slot | Kind | Contract | Notes |
|---|---|---|---|
| `scheduler` | policy | queue state → batch plan | Token budget, chunked prefill, priorities, preemption. Chunk size also sets kernel shape buckets. |
| `executor` | policy | slot graph → launches | Graph capture mode, launch grouping, sync mode, CPU thread affinity, host/device overlap. |
| `spec-drafter` | kernel + policy | context, hidden states → draft tokens | Draft models, extra heads, n-gram and suffix drafting, the model's own multi-token heads. |
| `spec-verifier` | kernel | drafts, target logits → accepted tokens | Acceptance rule; must keep the output distribution exact. |
| `spec-controller` | policy | acceptance statistics, verify cost → draft length or off | Speculation can be a net loss on some setups, so "off" is always a candidate. |
| `os-config` | config | environment → system settings | Memory limits, huge pages, power and clock settings, file system choice. |

**Offline transforms**

| Slot | Kind | Contract | Notes |
|---|---|---|---|
| `weight-format` | transform | high-precision tensors (+ calibration) → encoded tensors per tensor | Per-tensor format choice. The cheapest format to decode differs by device, so this couples to `placement`. |
| `weight-rotation` | transform | weights → rotated weights | Hadamard or learned rotations folded offline; affects quantization error, not runtime cost. |
| `expert-prune` | transform or policy | router, experts → reduced expert set or k | Lossy by definition; opt-in only and gated on quality. |

**Out of scope for plugins:** architecture changes that require retraining (new attention types, trained sparsity, retrofitted layers). The catalog records them so that the engine can host such models when they exist, but the search never proposes them.

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
| M1 Host | Core engine with reference plugins for every model-compute slot, running one small dense model and one small MoE model end to end. | The slot decomposition holds for real models. |
| M2 Measurement | Machine profiler, harness with the timing protocol and stages 0–4, record store. | Timings are stable enough to compare. |
| M3 Wrapped search | Incumbent plugins wrapped from existing libraries for the main slots, and the bandit searcher. Run on at least three different machines. | The core claim: tuned beats default, and winners differ by setup. |
| M4 Warm start | Similarity search over the record and seeded priors. | Fewer trials to reach a good configuration. |
| M5 Plugin builder | LLM agents that port and generate plugins from catalog entries, behind the full gate stack. | Coverage beyond what libraries ship. |
| M6 Benchmarks | Both benchmark adapters; first external submissions. | External validation. |
| M7 Lossy | Lossy slots behind quality gates, opt-in. | Larger gains where users accept approximation. |

The catalog build (section 6) runs in parallel from M1 onward, because M3 needs it to choose which incumbents to wrap first.

---

## 12. Platform support

The orchestration layer (engine core, searcher, harness, record) must run on Windows, macOS and Linux. Plugins declare the platforms they support through `requires`; the searcher only proposes plugins legal on the current profile, so a machine with no supported accelerator still runs the CPU reference and CPU plugins.

Expected backend coverage, to be confirmed during M1:

| Platform | Accelerator backends | CPU backends |
|---|---|---|
| Linux | CUDA, ROCm | x86 and Arm SIMD |
| Windows | CUDA native; Linux backends through WSL2 where native builds are impractical | x86 SIMD |
| macOS | Metal | Arm SIMD |

Platform-specific measurements (pinned-memory limits under virtualization, unified-memory behaviour) belong in the machine profile, not in hard-coded branches.

---

## 13. Open decisions

1. **Host language and graph format.** Python orchestration with plugins in compiled languages and kernel DSLs is the working assumption. The model description format (custom, or derived from an existing model library's configs) is undecided.
2. **Own engine or wrap existing engines.** The working assumption is a minimal engine of our own, with existing engines' kernels wrapped as plugins. Wrapping whole engines limits which slots can actually be swapped.
3. **Granularity of instance groups.** How fine the default shape buckets are, and whether the searcher can split groups freely or only along predefined axes.
4. **Searcher algorithm beyond the bandit baseline.** Tree search, evolutionary search or learned cost models, and when to switch.
5. **Similarity metric for warm starts.** Which profile, model and workload features carry weight, and whether the weights are learned from the record.
6. **Lossy scope.** Whether lossy slots are in scope for the first public results or deferred entirely.
7. **Record sharing.** Whether records from different users' machines can be pooled, and under what privacy rules.
8. **Plugin languages.** Which kernel languages the plugin builder targets first, given the portability needed for section 12.

---

## 14. Source material

`research_notes/` holds the raw research this spec draws on: technique tables organized by optimization area (`LLM inference optimization techniques/`) and by source engine, venue or benchmark (`Inference techniques by source/`). Those files become catalog input in M1–M3. They contain hardware-specific observations and claimed gains from many different machines; the catalog keeps each gain with the hardware it was measured on and never transfers a number from one platform to another.
