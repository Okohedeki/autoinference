# Inference market map

A tool for finding open positions in the AI inference market. It embeds every known inference offering, searches the embedding space for voids that sit inside the market but far from every product, decodes each void back into a concrete product description, and ranks the results.

## How it works

1. **Offerings.** [`data/offerings.csv`](data/offerings.csv) lists about 300 inference products. Each one has a description and seven facets: layer, modality, deployment, customer, vertical, pricing and differentiator. An eighth facet, *supply*, says whose hardware the inference runs on: the vendor's data centers, the customer's own hardware, end-user devices, a network edge, pooled crowd GPUs, or other providers' capacity. Supply is derived from layer and deployment, with a few overrides. The allowed values for every facet are in [`facets.py`](facets.py).
2. **Embedding.** Each offering is embedded twice: once from its own description, and once from sentences generated from the facets it covers. The two vectors are averaged. The default model is Qwen3-Embedding-0.6B.
3. **Candidates.** The tool generates 30,000 facet combinations made from proven parts. Half are *mutations*: a real offering's exact pattern with one facet changed, such as who supplies the compute or who buys it. The other half are recombinations in which every pair of facets already occurs together in some real product, except at most one pair, which allows one new twist. These candidates are the vocabulary for decoding.
4. **Void search.** Gradient ascent maximizes the soft minimum cosine distance to every offering. The point is constrained to a convex mixture of three real offerings, so it stays "between" products instead of drifting into nonsense.
5. **Decoding.** Each void is mapped to the nearest candidate combination. It is written up as a title, a description, and a pitch that names the offering whose pattern it copies and the twist that sets it apart.
6. **Ranking.** By default, ideas are compared across the whole market by how they work, and industry is ignored. The score is a weighted geometric mean of:
   - **emptiness:** the percentile distance to the nearest offering;
   - **proven pattern:** how much of the idea already exists somewhere else;
   - **buildability:** whether a small team can build the layer.

   A new mechanism in a crowded layer counts as an opening, so crowding is reported but does not lower the score.

   With `--industries`, the tool instead looks for proven patterns moved into new industries. It adds an **adoption gap** term, which measures how low AI use is in the sector, using the Census Business Trends and Outlook Survey in [`data/adoption.csv`](data/adoption.csv). It also lowers the score in crowded verticals. [`data/checked.csv`](data/checked.csv) records industry openings already researched by hand; the page shows their verdicts as labels.

## Setup

Python 3.11 or newer.

**Linux and macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The tool runs on CUDA, Apple MPS or CPU. On CPU, embedding the 30,000 candidates takes a few minutes the first time. Embeddings are cached in `cache/`, so later runs only embed new text.

## Run

```bash
python pipeline.py
python render.py
```

- `pipeline.py` writes `out/ideas.json` (the ranked ideas, with every score component) and `out/map.json` (2D positions from UMAP).
- `render.py` bakes the map data into `out/market_map.html`, a single interactive page.

Useful flags:

- `--model Qwen/Qwen3-Embedding-8B` uses a sharper embedding model.
- `--candidates` sets the size of the decoding vocabulary.
- `--starts` sets the number of void-search restarts.
- `--min-plausibility` sets how much of a recombination must already exist.

If your Hugging Face cache sits on a nearly full drive, set `HF_HUB_CACHE` to a folder elsewhere before running.

## Extending the dataset

Add rows to `data/offerings.csv`, using only values listed in `facets.py`. Separate multiple values with `;`. The pipeline refuses unknown facet values. The ideas are only as good as this list: a vertical that looks empty may only be missing from the data, so check the top ideas by hand before acting on them.
