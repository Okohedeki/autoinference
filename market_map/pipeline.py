"""Find open positions in the AI inference market by searching for voids in embedding space.

1. Embed every known offering (its description plus sentences rendered from its facets).
2. Sample many facet combinations that make product sense and embed them the same way. These are the
   vocabulary for decoding: any point in the space can be turned back into the nearest combination.
3. Search for voids: points inside the market (convex mixtures of a few real offerings) that are as
   far as possible from every offering, found by gradient ascent on a soft minimum distance.
4. Decode each void to its nearest facet combination, then score the result on novelty, whether the
   pattern is proven in some other vertical, how low AI adoption is in its sector, buildability for a
   small team, and whether anyone already serves it directly.

Run:  python pipeline.py            (writes out/ideas.json and out/map.json)
"""

import argparse
import csv
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch

from embed import Embedder
from facets import ADOPTION_FALLBACK, BUILDABILITY, FACETS, describe, is_compatible, supply_of

HERE = Path(__file__).parent
PATTERN_FACETS = ["layer", "supply", "modality", "deployment", "customer", "pricing"]
SELF_HOSTED_LAYERS = {"engine", "edge-runtime", "appliance", "optimization", "serverless-api", "platform"}


def load_offerings(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for facet, vocab in FACETS.items():
            if facet == "supply":
                continue
            row[facet] = [v.strip() for v in row[facet].split(";") if v.strip()]
            unknown = [v for v in row[facet] if v not in vocab]
            if unknown:
                raise ValueError(f"{row['name']}: unknown {facet} {unknown}")
        row["supply"] = supply_of(row)
    return rows


def load_adoption(path: Path) -> tuple[dict[str, float], str]:
    if not path.exists():
        return dict(ADOPTION_FALLBACK), "built-in estimate"
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    shares = dict(ADOPTION_FALLBACK)
    shares.update({r["vertical"]: float(r["share"]) for r in rows})
    return shares, rows[0].get("source", "") if rows else "built-in estimate"


def combos_of(row: dict, cap: int, rng: random.Random) -> list[dict]:
    """All facet combinations an offering covers (capped), used to place it in the space."""
    every = [dict(zip(FACETS, values)) for values in itertools.product(*(row[f] for f in FACETS))]
    valid = [c for c in every if is_compatible(c)] or every
    return rng.sample(valid, cap) if len(valid) > cap else valid


NON_VERTICAL = [f for f in FACETS if f != "vertical"]
FACET_PAIRS = list(itertools.combinations(NON_VERTICAL, 2))


def attested_pairs(offerings: list[dict]) -> set:
    """Every pair of facet values (ignoring vertical) that at least one real offering combines."""
    seen = set()
    for row in offerings:
        for a, b in FACET_PAIRS:
            seen.update((a, x, b, y) for x in row[a] for y in row[b])
    return seen


def plausibility(combo: dict, attested: set) -> float:
    """Share of the combination's non-vertical facet pairs that some real product already combines."""
    return sum((a, combo[a], b, combo[b]) in attested for a, b in FACET_PAIRS) / len(FACET_PAIRS)


def sample_candidates(n: int, offerings: list[dict], min_plausibility: float, industries: bool,
                      rng: random.Random) -> list[dict]:
    """Facet combinations built from proven parts.

    Half are mutations: a real offering's exact pattern with one thing changed. With `industries` that
    one thing is the vertical; otherwise it is any facet except the vertical, such as who supplies the
    compute or who the customer is. The rest are random combinations whose facet pairs are almost all
    attested, allowing one new twist.
    """
    attested = attested_pairs(offerings)
    seen, out = set(), []

    def add(combo):
        key = tuple(combo[f] for f in FACETS)
        if key not in seen and is_compatible(combo):
            seen.add(key)
            out.append(combo)

    for row in offerings:
        for combo in combos_of(row, 12, rng):
            if industries:
                for vertical in FACETS["vertical"]:
                    add({**combo, "vertical": vertical})
            else:
                for facet in NON_VERTICAL:
                    for value in FACETS[facet]:
                        add({**combo, facet: value})
    out = rng.sample(out, min(len(out), n // 2))
    seen = {tuple(c[f] for f in FACETS) for c in out}
    tries = 0
    while len(out) < n and tries < 200 * n:
        tries += 1
        combo = {f: rng.choice(list(FACETS[f])) for f in FACETS}
        if not industries:
            combo["vertical"] = "horizontal"
        if plausibility(combo, attested) >= min_plausibility:
            add(combo)
    return out


def find_voids(E: torch.Tensor, n_starts: int, mix: int, steps: int, tau: float, seed: int) -> torch.Tensor:
    """Maximize the soft minimum cosine distance to every offering over mixtures of a few offerings.

    Each start picks `mix` random offerings and learns weights on the simplex over them, so a void is
    always "between" real products and stays inside the market rather than drifting into nonsense.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = E.shape[0]
    support = torch.stack([torch.randperm(n, generator=g)[:mix] for _ in range(n_starts)]).to(E.device)
    logits = torch.zeros(n_starts, mix, device=E.device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    for _ in range(steps):
        w = torch.softmax(logits, dim=1)
        v = torch.nn.functional.normalize((w.unsqueeze(-1) * E[support]).sum(1), dim=-1)
        dist = 1 - v @ E.T  # (starts, offerings)
        soft_min = -tau * torch.logsumexp(-dist / tau, dim=1)
        opt.zero_grad()
        (-soft_min.sum()).backward()
        opt.step()
    with torch.no_grad():
        w = torch.softmax(logits, dim=1)
        voids = torch.nn.functional.normalize((w.unsqueeze(-1) * E[support]).sum(1), dim=-1)
    return voids, support, w.detach()


SHORT = {
    "layer": {"chip": "chip", "gpu-cloud": "GPU cloud", "serverless-api": "model API", "engine": "inference engine",
              "edge-runtime": "edge runtime", "appliance": "inference appliance", "platform": "deployment platform",
              "gateway": "model gateway", "optimization": "model optimizer", "observability": "inference monitor"},
    "modality": {"llm": "LLM", "vision": "vision", "speech": "speech", "video": "video", "embeddings": "retrieval",
                 "multimodal": "multimodal", "code": "code", "science": "science"},
}


def lower_first(text: str) -> str:
    """Lowercase a sentence's first letter unless it starts an acronym (API, KV, LLM)."""
    return text if text[1:2].isupper() else text[:1].lower() + text[1:]


PHRASES = {
    "layer": lambda v: f"sell it as {FACETS['layer'][v]}",
    "supply": lambda v: f"run it {FACETS['supply'][v]}",
    "modality": lambda v: f"serve {FACETS['modality'][v]}",
    "deployment": lambda v: f"deliver it {FACETS['deployment'][v]}",
    "customer": lambda v: f"sell to {FACETS['customer'][v]}",
    "pricing": lambda v: f"charge through {FACETS['pricing'][v]}",
    "differentiator": lambda v: f"compete on {FACETS['differentiator'][v]}",
}


def title(combo: dict) -> str:
    vertical = "" if combo["vertical"] == "horizontal" else FACETS["vertical"][combo["vertical"]].split(" and ")[0].capitalize() + " "
    where = {"on-prem": "on-prem ", "edge": "edge ", "device": "on-device ", "dedicated": "private ", "cloud": ""}[combo["deployment"]]
    source = {"crowd": "crowd-sourced ", "network-edge": "network-edge ", "third-party": "brokered ",
              "user-device": "", "customer-hw": "", "datacenter": ""}[combo["supply"]]
    customer = {"consumers": " for consumers", "smb": " for small businesses", "oem": " for device makers",
                "regulated": " for regulated orgs", "developers": "", "enterprise": ""}[combo["customer"]]
    return f"{vertical}{source}{where}{SHORT['modality'][combo['modality']]} {SHORT['layer'][combo['layer']]}{customer}".strip()


def pitch(combo: dict, template: dict | None, adoption: float, nearest: list[str]) -> str:
    """Turn a decoded combination into a grounded pitch: the proven pattern, where it goes, who is close."""
    vertical = FACETS["vertical"][combo["vertical"]]
    parts = []
    if template:
        parts.append(f"{template['name']} proves the pattern: {lower_first(template['description'])}")
        changes = [PHRASES[f](combo[f]) for f in PATTERN_FACETS + ["differentiator"] if combo[f] not in template[f]]
        if changes:
            parts.append("The twist: " + "; ".join(changes) + ".")
    if combo["vertical"] != "horizontal":
        parts.append(f"Bring it to {vertical}, where about {adoption:.0%} of firms use AI, for "
                     f"{FACETS['customer'][combo['customer']]} running {FACETS['modality'][combo['modality']]}.")
    parts.append(f"Nearest existing offerings in embedding space: {', '.join(nearest)}.")
    return " ".join(parts)


def load_checked(path: Path) -> dict:
    """Openings already researched by hand, keyed by (layer, modality, vertical). Taken ones are skipped."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {(r["layer"], r["modality"], r["vertical"]): {"verdict": r["verdict"], "note": r["note"]}
                for r in csv.DictReader(f)}


def overlap(a: dict, row: dict, facets: list[str]) -> float:
    return sum(a[f] in row[f] for f in facets) / len(facets)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offerings", type=Path, default=HERE / "data" / "offerings.csv")
    ap.add_argument("--adoption", type=Path, default=HERE / "data" / "adoption.csv")
    ap.add_argument("--industries", action="store_true",
                    help="look for industry transplants scored on sector AI adoption, instead of new mechanisms")
    ap.add_argument("--checked", type=Path, default=HERE / "data" / "checked.csv",
                    help="hand-checked openings; taken ones are left out of the ranking")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--model", default=None, help="embedding model (default depends on hardware)")
    ap.add_argument("--candidates", type=int, default=30000, help="facet combinations used for decoding")
    ap.add_argument("--starts", type=int, default=4000, help="random starts for the void search")
    ap.add_argument("--ideas", type=int, default=60)
    ap.add_argument("--min-plausibility", type=float, default=14 / 15,
                    help="share of non-vertical facet pairs that must exist in some real product")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    offerings = load_offerings(args.offerings)
    adoption, adoption_source = load_adoption(args.adoption)
    embedder = Embedder(args.model, cache_dir=HERE / "cache")
    print(f"{len(offerings)} offerings, embedding with {embedder.model_name} on {embedder.device}")

    # Offerings: half their own words, half the facet sentences they cover.
    desc_vecs = embedder.embed([r["description"] for r in offerings])
    facet_vecs = []
    for row in offerings:
        facet_vecs.append(embedder.embed([describe(c) for c in combos_of(row, 12, rng)]).mean(0))
    off = desc_vecs + np.stack(facet_vecs)
    off /= np.linalg.norm(off, axis=1, keepdims=True)

    if not args.industries:  # place offerings by how they work, not who they sell to
        for row in offerings:
            row["sold_into"], row["vertical"] = row["vertical"], ["horizontal"]
    candidates = sample_candidates(args.candidates, offerings, args.min_plausibility, args.industries, rng)
    attested = attested_pairs(offerings)
    cand = embedder.embed([describe(c) for c in candidates], batch_size=128)
    print(f"{len(candidates)} candidate combinations embedded")

    device = embedder.device if embedder.device != "mps" else "cpu"
    E = torch.from_numpy(off).to(device)
    C = torch.from_numpy(cand).to(device)

    # How crowded each candidate's neighborhood is: distance to its nearest real offering.
    cand_nn = (1 - (C @ E.T)).min(1).values.cpu().numpy()
    off_nn = 1 - (E @ E.T).fill_diagonal_(-1).max(1).values.cpu().numpy()

    voids, support, weights = find_voids(E, args.starts, mix=3, steps=300, tau=0.02, seed=args.seed)
    void_dist = (1 - voids @ E.T).min(1).values
    order = torch.argsort(void_dist, descending=True).cpu().numpy()

    max_adoption = max(adoption.values())
    names = {r["name"]: j for j, r in enumerate(offerings)}
    checked = load_checked(args.checked)
    ideas, used_keys = [], set()
    nearest_cand = torch.topk(voids @ C.T, k=8, dim=1)
    for vi in order:
        for ci, sim in zip(nearest_cand.indices[vi].tolist(), nearest_cand.values[vi].tolist()):
            combo = candidates[ci]
            key = (combo["layer"], combo["modality"], combo["vertical"])
            if not args.industries:
                key = (combo["layer"], combo["modality"], combo["supply"])
            if key in used_keys:
                continue
            used_keys.add(key)
            idea_vec = C[ci]
            sims = (E @ idea_vec).cpu().numpy()
            neighbors = np.argsort(-sims)[:5]

            # Proven pattern: the closest offering in another vertical that shares the same layer.
            same_layer = [
                (overlap(combo, r, PATTERN_FACETS), j) for j, r in enumerate(offerings)
                if combo["layer"] in r["layer"] and (combo["vertical"] not in r["vertical"] or not args.industries)
            ]
            overlap_best, template = max(same_layer, default=(0.0, None))
            proven = 0.5 * plausibility(combo, attested) + 0.5 * overlap_best
            direct = [
                r["name"] for r in offerings
                if combo["layer"] in r["layer"] and combo["vertical"] in r["vertical"]
                and combo["deployment"] in r["deployment"] and combo["modality"] in r["modality"]
            ]
            vertical_players = [
                r["name"] for r in offerings if combo["vertical"] in r["vertical"]
            ]
            novelty = float((cand_nn < cand_nn[ci]).mean())  # percentile among all candidates
            gap = 1 - adoption[combo["vertical"]] / max_adoption
            build = BUILDABILITY[combo["layer"]]
            same_lane = [n for n in vertical_players if combo["layer"] in offerings[names[n]]["layer"]]
            if args.industries:
                score = (max(novelty, 1e-3) ** 0.35) * (max(proven, 0.2) ** 0.2) * (max(gap, 0.05) ** 0.25) * (build ** 0.2)
                score *= 0.5 if direct else 1.0
                score *= float(np.exp(-0.15 * len(same_lane) - 0.02 * len(vertical_players)))  # crowded verticals
            else:
                # A new mechanism in a crowded layer is fine, so crowding is reported but not penalized.
                score = (max(novelty, 1e-3) ** 0.5) * (max(proven, 0.2) ** 0.25) * (build ** 0.25)
            ideas.append({
                "facets": combo,
                "title": title(combo),
                "description": describe(combo),
                "pitch": pitch(combo, offerings[template] if template is not None else None,
                               adoption[combo["vertical"]], [offerings[j]["name"] for j in neighbors[:3]]),
                "score": round(score, 4),
                "novelty": round(novelty, 3),
                "nn_distance": round(float(cand_nn[ci]), 4),
                "void_distance": round(float(void_dist[vi]), 4),
                "decode_error": round(1 - sim, 4),
                "proven": round(proven, 2),
                "template": offerings[template]["name"] if template is not None else None,
                "adoption": adoption[combo["vertical"]],
                "adoption_gap": round(gap, 3),
                "buildability": build,
                "direct_competitors": direct,
                "vertical_players": vertical_players,
                "same_lane": same_lane,
                "neighbors": [offerings[j]["name"] for j in neighbors],
                "void_between": [offerings[j]["name"] for j in support[vi].tolist()],
                "void_weights": [round(x, 3) for x in weights[vi].tolist()],
                "checked": checked.get(key),
                "self_hostable": combo["layer"] in SELF_HOSTED_LAYERS and combo["deployment"] in {"on-prem", "edge", "device", "dedicated"},
                "_vec": idea_vec.cpu().numpy(),
            })
            break
        if len(ideas) >= args.ideas * 3:
            break

    ideas.sort(key=lambda d: -d["score"])
    ideas = ideas[: args.ideas]
    for rank, idea in enumerate(ideas, 1):
        idea["rank"] = rank

    # 2D map: offerings, ideas and a sample of the candidate space as context.
    import umap

    dust_idx = rng.sample(range(len(candidates)), min(6000, len(candidates)))
    stack = np.concatenate([off, np.stack([i["_vec"] for i in ideas]), cand[dust_idx]])
    xy = umap.UMAP(n_neighbors=30, min_dist=0.5, spread=1.5, metric="cosine", random_state=args.seed).fit_transform(stack)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    n_off, n_ideas = len(offerings), len(ideas)

    args.out.mkdir(parents=True, exist_ok=True)
    for i, idea in enumerate(ideas):
        idea["xy"] = [round(float(v), 4) for v in xy[n_off + i]]
        idea.pop("_vec")
    map_data = {
        "model": embedder.model_name,
        "mode": "industries" if args.industries else "mechanisms",
        "candidates": len(candidates),
        "adoption_source": adoption_source,
        "offerings": [
            {
                "name": r["name"], "description": r["description"], "region": r["region"],
                **{f: r[f] for f in FACETS},
                "nn_distance": round(float(off_nn[j]), 4),
                "xy": [round(float(v), 4) for v in xy[j]],
            }
            for j, r in enumerate(offerings)
        ],
        "ideas": ideas,
        "dust": [
            {"xy": [round(float(v), 4) for v in xy[n_off + n_ideas + k]], "d": round(float(cand_nn[ci]), 3),
             "layer": candidates[ci]["layer"], "vertical": candidates[ci]["vertical"]}
            for k, ci in enumerate(dust_idx)
        ],
        "adoption": adoption,
    }
    (args.out / "map.json").write_text(json.dumps(map_data))
    (args.out / "ideas.json").write_text(json.dumps(ideas, indent=2))
    print(f"wrote {n_ideas} ideas to {args.out}")
    for idea in ideas[:15]:
        print(f"{idea['rank']:>2}  {idea['score']:.3f}  {idea['description']}")


if __name__ == "__main__":
    main()
