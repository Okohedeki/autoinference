# AutoInference

An LLM inference engine in which every part is a swappable plugin: the scheduler, the KV cache, input preparation, the sampler, and each op of the forward pass (attention, linear layers, norms, the MoE router, dispatch, experts and combine, and so on). A configuration file chooses the implementation for every slot, and per-layer or per-phase overrides let different layers use different kernels.

The long-term goal is a search loop that finds the fastest correct combination of plugins for a given machine, model and workload. The design is in [`reports/AutoInference design spec.md`](reports/AutoInference%20design%20spec.md).

**Status: M1 MVP.** The engine is correct and deliberately slow. Every slot has a readable reference implementation that later implementations are checked against.

## What works

- Qwen3 dense and Qwen3 MoE checkpoints in safetensors format.
- Continuous batching with chunked prefill, a paged KV cache, preemption by recompute, and an optional block-hash prefix cache.
- Greedy, temperature, top-k, top-p and repetition-penalty sampling.
- A command line and a minimal OpenAI-compatible `/v1/chat/completions` endpoint, streaming included.
- Per-slot timing (`--profile`) and a report of which implementation filled each slot (`--describe`).
- Runs on CPU on Windows, macOS and Linux, and on CUDA or Apple MPS when available. It is pure Python on PyTorch, with no compiled extensions.

## Setup

Python 3.11 or newer is required.

**Linux and macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

For CUDA, install the PyTorch build matching your driver first (see pytorch.org), then run the install above. On Apple Silicon the default PyTorch wheel includes MPS.

## Usage

```bash
autoinference generate Qwen/Qwen3-0.6B --chat --prompt "Name three primary colors." --max-tokens 64
```

The model argument is a local checkpoint directory or a Hugging Face repo id, which is downloaded to the Hugging Face cache on first use.

Swap implementations through a configuration file. Two are bundled: `reference.toml`, where every slot uses its reference implementation, and `torch.toml`, which uses PyTorch-native ops where they exist and turns on prefix caching:

```bash
autoinference generate Qwen/Qwen3-0.6B --config torch.toml --chat --prompt "Hello" --profile --describe
```

Serve an OpenAI-compatible endpoint:

```bash
autoinference serve Qwen/Qwen3-0.6B --config torch.toml --port 8000
```

List every slot and its registered implementations:

```bash
autoinference parts
```

## Bio service

`autoinference bio` hosts biology models as a service. It is built for batch campaigns, such as scoring many small molecules or binders against one protein target.

- **Funnel campaigns.** A campaign is a list of stages. Each stage runs one model on the survivors of the previous stage and keeps the best of them, so expensive models only see promising candidates. A cost estimate, stage by stage, is available before anything runs. It switches from default per-candidate timings to measured ones once a backend has run.
- **Trust checks.** Every predicted pose is checked for clashes, overlapping atoms and a ligand that is not touching the protein. Optional consensus backends re-score the final survivors and report how well the models agree.
- **Tuning on assay data.** After lab testing, send the outcomes back. The service fits a re-scorer on every model score those candidates received, plus an isotonic calibration to the observed hit rate. Later campaigns that name the tuned model get a calibrated `p_active` for each candidate. Models and data stay in the workspace.
- **Reproducible runs.** Each job records a manifest (spec digest, seed, backend fingerprints, environment, output digest). `verify` replays a job and checks that the output digest is identical. Submissions, results, tunings and replays go into a hash-chained audit log.
- **Agents.** An MCP server exposes the same operations as tools: estimate, submit, poll, results, tune and verify.

Backends:

| Backend | What it is | Needs |
|---|---|---|
| `boltz2` | Boltz-2 co-folding and affinity, through the `boltz` command | `pip install boltz` (may be a separate environment), a GPU |
| `rdkit-druglike` | QED drug-likeness filter for small molecules | `pip install rdkit` |
| `synthetic-screen`, `synthetic-pose` | Seeded stand-ins for a cheap filter and a pose model, for demos and tests | nothing |

```bash
pip install -e ".[bio]"
autoinference bio backends
autoinference bio estimate examples/bio/funnel.json
autoinference bio run examples/bio/funnel.json --root bio-workspace
autoinference bio verify <job-id> --root bio-workspace
autoinference bio tune <job-id> assay.csv --root bio-workspace      # CSV: id,active  or  id,value with --threshold
autoinference bio serve --root bio-workspace --port 8100             # HTTP API under /v1
autoinference bio mcp --root bio-workspace                           # MCP over stdio
```

To use Boltz-2 installed in another environment, pass its executable with `--boltz-command`, or set `AUTOINFERENCE_BOLTZ`. On Windows, add `"extra_args": "--no_kernels --num_workers 0"` to the stage's `options`, because Boltz's fused kernels are Linux-only. All candidates in a batch go to one `boltz predict` call. The target's alignment is computed on the first batch and reused for every later one.

HTTP endpoints: `GET /v1/backends`, `POST /v1/estimate`, `POST /v1/campaigns`, `GET /v1/jobs`, `GET /v1/jobs/<id>`, `GET /v1/jobs/<id>/results?top=N`, `POST /v1/jobs/<id>/tune` (`{"labels": {"<id>": 1}}`), `POST /v1/jobs/<id>/verify`, `GET /v1/audit`.

Cost figures use `--gpu-hour-price`, which defaults to a placeholder of 1.0. Set it to your actual rental price.

## Configuration

```toml
[engine]
device = "auto"        # cpu, cuda, mps or auto
dtype = "bfloat16"
page_size = 16

[parts]
scheduler = "fcfs"
prefix-cache = "block-hash"

[ops]
default = "reference"  # used by any op slot not listed
linear = "torch"
attn-decode = "torch-sdpa"

[[ops.override]]       # later overrides win; match on layer_type, phase, device or layers
slot = "attn-decode"
layers = [0, 1]
impl = "reference"
```

## Adding an implementation

Write a class that follows the slot's interface (`src/autoinference/ops/interfaces.py` for ops, `src/autoinference/parts/interfaces.py` for other parts) and register it:

```python
from autoinference import Manifest, register

@register("norm", "my-rmsnorm", Manifest(kind="kernel", lossless_class="tolerance-exact"))
class MyRMSNorm:
    def __call__(self, x, weight, eps):
        ...
```

Import the module before building the engine, then select it with `norm = "my-rmsnorm"` under `[ops]`. The swap-equivalence tests show how to check it against the reference.

## Tests

```bash
pytest
```

The suite builds tiny random Qwen3 and Qwen3 MoE checkpoints and checks that:

- logits and greedy output match Hugging Face transformers;
- every alternative implementation matches the reference;
- batching, chunked prefill, preemption and prefix caching leave outputs unchanged.

It needs no downloads and runs on CPU on any OS. To also check a real checkpoint:

```bash
AUTOINFERENCE_REAL_MODEL=Qwen/Qwen3-0.6B pytest tests/test_real_weights.py
```

(On Windows PowerShell: `$env:AUTOINFERENCE_REAL_MODEL="Qwen/Qwen3-0.6B"; pytest tests/test_real_weights.py`.)

## Layout

```
src/autoinference/
  engine.py         builds every part from the configuration and runs the step loop
  config.py         configuration and per-instance op resolution
  registry.py       slot registry and implementation manifests
  types.py          data passed between parts
  observability.py  per-slot timers and counters
  ops/              op-library slots: interfaces, op table, reference and PyTorch implementations
  parts/            scheduler, KV allocator, prefix cache, input prep, sampler, loader, renderer, stubs
  models/           architecture registry and the Qwen3 dense / MoE graph
  bio/              bio service: backends, campaigns, trust checks, tuning, provenance, HTTP and MCP
  server.py, cli.py
examples/bio/       example campaign spec
tests/
reports/            design specification
research_notes/     source material for the technique catalog
```
