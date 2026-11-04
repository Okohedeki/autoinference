# Platform specification: serverless inference for non-LLM models

Status: proposal, v0.1. Decisions marked **(decide)** need an owner's answer before the matching work starts; everything else is the recommended default.

## 1. What we are building

A hosted platform that runs open, specialised AI models (structure prediction, force fields, robot policies, world models, forecasters, segmenters and so on) behind one API, with the same three product surfaces an inference platform needs:

1. **Model APIs.** A catalog of open models, each with a typed schema, a billing unit and a measured cost. One call to run a model, a queue for long jobs, a playground per model, an MCP tool per model.
2. **Serverless deployment.** Customers deploy their own model through the same contract the catalog uses: an image, a handler, a schema. We run it with scale-to-zero, metrics and logs.
3. **Fine-tuning.** Per model family, fitting on the customer's data (assay outcomes, labelled scans, their own trajectories). Our version of a compute product until we own hardware.

The platform differentiators, which apply to every model rather than one domain:

- **Funnels.** Multi-stage campaigns that spend expensive models only on candidates that cheap models liked.
- **Trust checks.** Output validation per output type (pose geometry, energy conservation, segmentation sanity, forecast bounds), consensus between models, calibration against outcomes.
- **Tuning.** Re-scorers fitted on customer outcomes, used in later runs.
- **Provenance.** A manifest per run (inputs, model version, weights digest, seed, environment), deterministic replay, and a hash-chained audit log.
- **Agents.** Every operation exposed as MCP tools over a remote server, with proper authorization.

Domains (17): biology; genetics and single-cell; chemistry and materials; robotics; world models; decision models; autonomous; earth, weather and climate; engineering simulation and design; medical imaging; lab and microscopy imaging; cybersecurity; energy and grid; agriculture; particle physics, astronomy and fusion; neuroscience and BCI; document AI. The catalog itself lives in `models/` (section 6), not in this document.

Not in scope: generative media (image, video, music, speech), general-purpose LLM serving, recommenders and ads, embeddings as a product. General perception models (segmentation, depth, detection) are hosted only as building blocks inside pipelines for the domains above.

## 2. Architecture

```
                       Cloudflare (DNS, TLS, WAF, rate limits, cache)
                                       |
          +----------------------------+-----------------------------+
          |                            |                             |
   api.<domain>                 app.<domain>                   mcp.<domain>
   control plane (FastAPI)      dashboard + playgrounds        MCP remote server
   on Modal, 1 warm container   Next.js on Cloudflare          (same control plane)
          |
          |  Postgres (Neon)            R2 (object storage)        Stripe (billing)
          |  jobs, keys, usage, audit   inputs, outputs, weights   credits, invoices
          |
   +------+----------------------------------------------------------+
   |                        data plane (GPU workers)                 |
   |  Modal: one class per model, own image, weights baked in,       |
   |         scale to zero, GPU fallback list, container cap         |
   |  RunPod Serverless: same handler, cheaper for big batches       |
   |  BYOC (later): same images in the customer's AWS/GCP account    |
   +-----------------------------------------------------------------+
```

The control plane is small, stateless and CPU-only. The data plane is one container per model. Nothing of value lives on a developer machine or inside a container: inputs and outputs are in R2, state is in Postgres, secrets are in the providers' secret stores.

### 2.1 Why these choices

| Layer | Choice | Why | Alternatives considered |
|---|---|---|---|
| Edge | **Cloudflare** | DNS, TLS, WAF and per-key rate limits in one place; R2 has no egress fees, which matters for multi-GB genomics and imaging outputs; Pages hosts the dashboard | AWS CloudFront + S3 (egress cost), Vercel for the dashboard (fine, but a second vendor) |
| Control plane | **FastAPI on Modal** with one always-warm container | No servers to run; same account and secrets as the workers; a plain ASGI app, so it moves to Fly.io or Kubernetes unchanged when needed | Cloudflare Workers (Python support is immature and workers cannot call Modal directly), Fly.io (fine, adds a vendor) |
| Database | **Postgres on Neon** | Serverless, scales to zero, database branches per environment and per pull request | Supabase (bundles auth and storage we get elsewhere), RDS (overkill now) |
| Object storage | **Cloudflare R2** | S3-compatible, zero egress, presigned URLs, multipart upload for large files | S3, GCS |
| Auth | **WorkOS AuthKit** **(decide)** | OAuth sign-in (Google, GitHub, email), SSO/SAML for enterprise customers from day one, an OAuth 2.1 authorization server for MCP; free up to a large monthly-user count | Clerk (developer-friendly, SSO costs more), Auth0 (pricier), self-hosted (Better Auth; more to run) |
| API keys | **Our own**, hashed in Postgres | Standard for SDK access; scoped per project with rate limits and spend caps | — |
| Billing | **Stripe** with metered usage and prepaid credits | Usage-based invoices for cards; prepaid credit packs for customers who buy in advance; enterprise invoices by hand | Lago, Orb (good, later) |
| GPU workers | **Modal** first, **RunPod Serverless** second | Covered in the compute plan: per-second billing and scale-to-zero now, cheaper batch capacity when volume justifies it | Baseten, Replicate (more managed, more expensive), bare clouds (ops burden) |
| Observability | **OpenTelemetry** from the control plane and workers to **Grafana Cloud**; **Sentry** for errors | Free tiers cover us for a long time; traces per request across edge, control plane and worker | Axiom, Datadog (expensive) |
| Dashboard | **Next.js** on Cloudflare Pages | Playgrounds generated from model schemas; usage, keys, jobs | — |

Idle cost of the whole stack before customers: Cloudflare, Neon, WorkOS, Grafana and Sentry at zero on free tiers; the warm control-plane container is the only fixed GPU-vendor cost, in the tens of dollars a month. GPU time is only spent when a job runs.

### 2.2 Environments

`dev`, `staging` and `prod`, each with its own Modal environment, Neon branch, R2 bucket and Stripe mode. Secrets live in Modal Secrets (workers and control plane), Cloudflare (edge and dashboard) and GitHub Actions (deploys). No secret is ever committed; a pre-commit secret scan enforces it.

Deploys: a push to `main` runs tests, deploys the control plane and changed model images to `staging`, and promotes to `prod` on a tag. Model images are built once per model version and reused across environments.

## 3. API

REST over JSON, versioned under `/v1`. Every response carries a `request_id`. Errors are `{"error": {"type", "message", "param"}}` with the usual status codes. Idempotency through an `Idempotency-Key` header on every write.

### 3.1 Authentication

- **API keys** (`Authorization: Bearer <key>`): the SDK and server-to-server path. Keys are scoped to a project, created in the dashboard or through the API, stored as a hash, shown once. Each key carries rate limits and an optional spend cap.
- **OAuth 2.1** sessions for the dashboard, playgrounds and the MCP server. The MCP server follows the MCP authorization specification (authorization server discovery, PKCE, dynamic client registration), so any MCP client can connect without manual key handling.
- **Webhook signatures**: outgoing webhooks are signed with a per-project secret.

### 3.2 Models

```
GET  /v1/models                          catalog: id, domain, version, unit, price, hardware, license
GET  /v1/models/{model}                  schema: typed inputs and outputs, limits, example request
POST /v1/models/{model}/run              synchronous; for calls that finish within 60 s
POST /v1/models/{model}/queue            asynchronous; returns a request id at once
GET  /v1/requests/{id}                   status: queued, running, succeeded, failed, cancelled; progress; units
GET  /v1/requests/{id}/result            outputs, with signed URLs for files
POST /v1/requests/{id}/cancel
POST /v1/models/{model}/estimate         units and price before running, from measured timings
```

The queue request accepts `webhook_url`, `priority` and `provider` hints. Results and files are retained for a project-configurable period (default 30 days), then deleted; the manifest and audit entries are kept.

Model ids are `<domain>/<name>` with an optional pinned version: `biology/boltz-2`, `biology/boltz-2@2.2.1`. Unpinned ids resolve to the current version; the manifest always records the resolved version and the weights digest.

### 3.3 Files

```
POST /v1/files                           presigned upload URL (multipart for large files)
GET  /v1/files/{id}                      metadata and a signed download URL
```

Inputs reference files by id. Workers read inputs and write outputs through presigned R2 URLs, so bulk data never passes through the control plane.

### 3.4 Workflows

The cross-model operations that the differentiators need:

```
POST /v1/workflows/campaigns             a funnel: stages of models, keep rules, consensus, tuned model
POST /v1/workflows/campaigns/{id}/tune   fit a re-scorer on outcomes
POST /v1/workflows/verify/{request}      replay and compare output digests
GET  /v1/audit                           the project's hash-chained log, verifiable offline
```

A campaign is the current bio funnel generalised: any model whose output has a sortable score can be a stage.

### 3.5 Deployments (customer models)

```
POST /v1/deployments                     image, handler entry point, schema, hardware, scaling
GET  /v1/deployments/{id}                status, replicas, recent errors
POST /v1/deployments/{id}/run|queue      same shape as catalog models
```

A deployment is a catalog entry that only its project can see. Same runtime contract, same metrics, same billing by GPU seconds.

### 3.6 Account

```
GET  /v1/usage                           units and cost by model, by day
GET  /v1/keys · POST /v1/keys · DELETE /v1/keys/{id}
GET  /v1/projects/{id}/webhooks · POST ...
```

### 3.7 MCP

`mcp.<domain>` serves the same operations as tools over streamable HTTP: `list_models`, `describe_model`, `run_model`, `queue_model`, `request_status`, `request_result`, `estimate`, `submit_campaign`, `tune`, `verify`. Each catalog model also appears as its own tool, generated from its schema, so an agent sees `fold_protein` rather than a generic `run_model` with a string argument.

## 4. SDKs

Generated from the OpenAPI document, which is itself generated from the model schemas, so a new catalog model needs no SDK change.

- **Python** first: `client.run("biology/boltz-2", protein=..., ligand=...)`, `client.queue(...)`, `request.wait()`, `client.upload(path)`, `client.campaigns.submit(spec)`. Typed inputs per model via generated TypedDicts. Async variants.
- **TypeScript** second, same surface.
- **CLI** (`autoinference` today) grows `models`, `run`, `queue`, `files` and `keys` commands over the same SDK.

## 5. Data model

Postgres tables (project-scoped, every row carries `project_id`):

| Table | Holds |
|---|---|
| `organizations`, `projects`, `members` | tenancy; WorkOS organization ids |
| `api_keys` | hash, prefix, scopes, rate limit, spend cap, last used |
| `models`, `model_versions` | catalog entries, schemas, image digests, weights digests, unit, price, hardware |
| `deployments` | customer models: same fields plus owning project and scaling settings |
| `requests` | one row per run or queue call: model version, inputs digest, status, provider, provider job id, units, GPU seconds, timings, manifest |
| `files` | R2 key, size, content type, digest, expiry |
| `campaigns`, `campaign_stages` | funnel specs and per-stage reports |
| `tuned_models` | fitted re-scorers and their metrics |
| `usage_events` | one per billable unit batch, synced to Stripe meters |
| `credits` | prepaid balance ledger |
| `audit_log` | append-only, hash-chained; `verify` recomputes the chain |

Retention: request rows and audit entries are kept; outputs in R2 expire per project policy. The local SQLite-and-files workspace in the current bio service remains the development and test mode behind the same storage interface, so the test suite needs no cloud account.

## 6. Model runtime contract

Every model, from the catalog or a customer, is a directory:

```
models/<domain>/<name>/
  model.yaml        id, version, domain, license, unit, price, hardware, image recipe, schema
  handler.py        load() once per container; run(inputs, ctx) per request
  checks.py         optional trust checks on outputs
  tests/            a tiny input that must produce a known output digest
```

`model.yaml` declares:

- **schema**: JSON Schema for inputs and outputs, with file-typed fields (`pdb`, `fasta`, `vcf`, `nifti`, `tiff`, `npz`, …) that the API turns into upload and download URLs and the playground turns into widgets.
- **unit**: what one billable unit is (`structure`, `1k_residues`, `1k_md_steps`, `1k_actions`, `series`, `page`, …) and how the handler reports units consumed.
- **hardware**: GPU types in fallback order, memory, timeout, concurrency.
- **image**: base image, pip or uv requirements, system packages, weight downloads (with digests) baked at build time.
- **license** and **provenance**: the weights' license, source URL and digest, checked by CI before a model can be listed publicly.

The handler is plain Python with no provider SDK in it. Thin adapters wrap it for Modal (`@app.cls`) and RunPod (`handler(job)`), which is what `cloud.py` already does for Boltz-2. A handler writes files to a work directory; the adapter uploads them to R2 and returns signed URLs.

Trust checks run in the control plane after the worker returns, so they are the same regardless of provider.

## 7. Billing

- **Per unit per model.** Price per unit is set from measured GPU seconds per unit on the cheapest GPU that meets the model's needs, times the provider's price, times a margin. The estimate endpoint quotes it before the run.
- **GPU seconds** for deployments and for anything without a natural unit.
- **Prepaid credits** decrement per request; a request whose estimate exceeds the balance is refused. **Metered billing** for card customers through Stripe meters, invoiced monthly. **Enterprise** by invoice, with a committed spend.
- Every request stores the units, GPU seconds, provider and price it was billed at, so invoices are reproducible from the `requests` table alone.

## 8. Observability and operations

- Traces: one trace per request across edge, control plane and worker; GPU seconds, cold-start time, queue wait and model time as spans.
- Metrics per model version: p50/p95 latency, cold-start rate, failure rate, cost per unit, units per day. These drive both the status page and the pricing.
- Logs: worker stdout and stderr captured per request and retained for a short period, viewable in the dashboard.
- Alerts: failure rate per model, queue depth, provider errors, spend anomalies.
- Status page per model (public).

## 9. Security and compliance

- Tenancy enforced at the query layer: every table is project-scoped and every query filters on it.
- Secrets never in images or the repository; weights are public and their digests are recorded.
- Data in transit over TLS; at rest encrypted by R2 and Neon. Customer inputs are deleted on expiry; manifests keep only digests.
- Audit log is customer-exportable and verifiable without us.
- Workers run with network access limited to R2, the control plane and the model's own alignment or data services (for Boltz, the MSA server) **(decide)**: whether to run our own MSA server rather than send sequences to a public one. Pharma customers will ask.
- Roadmap: SOC 2 Type I once there are paying customers; BYOC deployment (our images in the customer's cloud account) for customers whose data cannot leave.

## 10. Build order

1. **Control plane skeleton.** FastAPI on Modal, Neon schema, R2 presigned files, API keys, request queue backed by Modal `spawn` with status written back by the worker. The current bio service's local mode stays as the test mode.
2. **Model contract.** `model.yaml`, handler, adapters; move Boltz-2 onto it; one model from each of the other sixteen domains to prove the contract is generic.
3. **Auth and dashboard.** WorkOS sign-in, projects, keys, usage, generated playgrounds.
4. **Billing.** Stripe meters and credits; estimate gating.
5. **SDKs and MCP.** OpenAPI generation, Python SDK, remote MCP server with OAuth.
6. **Workflows.** Campaigns, tuning, verify and audit, generalised from the bio service.
7. **Deployments.** Customer models through the same contract.
8. **Fill the catalog**, with licenses checked by CI.

Decisions needed before step 1: product name and domain; WorkOS versus Clerk; whether the platform lives in this repository (`platform/`, `models/`, `sdk/`, `web/` beside `src/autoinference/`) or in a new one.
