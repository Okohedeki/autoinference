"""Closed vocabularies for every offering facet, plus the per-value weights the ranking uses.

Every offering in data/offerings.csv and every generated idea is described with these facets, so a
point found in embedding space can always be decoded back to a concrete facet combination.
"""

LAYER = {
    "chip": "custom AI chips and accelerator hardware",
    "gpu-cloud": "rentable GPU cloud capacity",
    "serverless-api": "a serverless pay-per-use model API",
    "engine": "an inference engine or serving runtime",
    "edge-runtime": "an on-device or edge inference runtime",
    "appliance": "a turnkey on-site inference appliance",
    "platform": "a managed model deployment platform",
    "gateway": "a routing gateway across models and providers",
    "optimization": "model compression and optimization tooling",
    "observability": "inference monitoring and evaluation tooling",
}

MODALITY = {
    "llm": "language models",
    "vision": "computer vision models",
    "speech": "speech and audio models",
    "video": "video generation models",
    "embeddings": "embedding and retrieval models",
    "multimodal": "multimodal models",
    "code": "code models",
    "science": "biology and chemistry models such as protein, molecule and genomics models",
    "recsys": "recommendation and ranking models",
    "tabular": "tabular models such as gradient-boosted trees for scoring and fraud",
    "timeseries": "time-series forecasting and anomaly detection models",
    "graph": "graph neural network models",
    "simulation": "physics simulation surrogate models",
    "earth": "weather, climate and earth-observation models",
    "pointcloud": "lidar, 3D and point-cloud models",
    "policy": "robot control and reinforcement learning policy models",
}

DEPLOYMENT = {
    "cloud": "hosted in the public cloud",
    "dedicated": "on dedicated single-tenant capacity",
    "on-prem": "installed on the customer's own servers",
    "edge": "running at edge sites near the work",
    "device": "running on end-user devices",
}

CUSTOMER = {
    "developers": "developers and startups",
    "enterprise": "large enterprises",
    "regulated": "regulated organizations",
    "smb": "small and mid-sized businesses",
    "consumers": "consumers",
    "oem": "device makers",
}

VERTICAL = {
    "horizontal": "any industry",
    "healthcare": "healthcare",
    "finance": "finance and insurance",
    "legal": "legal services",
    "manufacturing": "manufacturing",
    "agriculture": "agriculture",
    "construction": "construction and trades",
    "retail": "retail and restaurants",
    "government": "government and defense",
    "education": "education",
    "logistics": "transportation and logistics",
    "energy": "energy and utilities",
    "media": "media and entertainment",
    "telecom": "telecommunications",
    "automotive": "automotive",
}

PRICING = {
    "per-token": "usage-based pricing",
    "gpu-hour": "per-GPU-hour pricing",
    "subscription": "a flat subscription",
    "license": "a software license",
    "hardware": "a hardware purchase",
    "open-source": "open source with paid support",
}

DIFFERENTIATOR = {
    "speed": "the lowest latency",
    "cost": "the lowest cost",
    "privacy": "data privacy and sovereignty",
    "ease": "ease of use",
    "breadth": "the widest model choice",
    "scale": "massive scale",
    "efficiency": "energy and hardware efficiency",
    "customization": "domain customization",
}

# Who owns the hardware the inference runs on. A new supply model in a crowded layer is often the
# whole idea, so this facet matters more than the industry a product is sold into.
SUPPLY = {
    "datacenter": "on compute the vendor owns or rents in data centers",
    "customer-hw": "on hardware the customer already owns",
    "user-device": "on the end user's own phone or computer",
    "network-edge": "on a telecom or CDN edge network",
    "crowd": "on spare GPUs pooled from many independent owners",
    "third-party": "on other providers' capacity, resold or routed",
}

FACETS = {
    "layer": LAYER,
    "supply": SUPPLY,
    "modality": MODALITY,
    "deployment": DEPLOYMENT,
    "customer": CUSTOMER,
    "vertical": VERTICAL,
    "pricing": PRICING,
    "differentiator": DIFFERENTIATOR,
}

# How realistic each layer is for a small team to build and sell within about a year.
# Chips and GPU clouds need large capital; software layers do not.
BUILDABILITY = {
    "chip": 0.05,
    "gpu-cloud": 0.25,
    "serverless-api": 0.6,
    "engine": 0.75,
    "edge-runtime": 0.7,
    "appliance": 0.6,
    "platform": 0.8,
    "gateway": 0.9,
    "optimization": 0.8,
    "observability": 0.9,
}

# Share of US firms using AI to produce goods or services, by sector (0-1). Approximate values in
# the range of the US Census Bureau Business Trends and Outlook Survey; see data/adoption.csv for
# the figures and source actually loaded at run time. These are only a fallback.
ADOPTION_FALLBACK = {
    "horizontal": 0.10,
    "healthcare": 0.09,
    "finance": 0.15,
    "legal": 0.20,
    "manufacturing": 0.07,
    "agriculture": 0.03,
    "construction": 0.04,
    "retail": 0.06,
    "government": 0.08,
    "education": 0.15,
    "logistics": 0.05,
    "energy": 0.07,
    "media": 0.25,
    "telecom": 0.25,
    "automotive": 0.07,
}

# Combinations that make no product sense; the idea generator skips them.
INCOMPATIBLE = [
    ({"layer": "chip"}, {"customer": "consumers"}),
    ({"layer": "chip"}, {"pricing": "per-token"}),
    ({"layer": "chip"}, {"pricing": "subscription"}),
    ({"layer": "gpu-cloud"}, {"deployment": "device"}),
    ({"layer": "gpu-cloud"}, {"pricing": "hardware"}),
    ({"layer": "serverless-api"}, {"deployment": "device"}),
    ({"layer": "serverless-api"}, {"pricing": "hardware"}),
    ({"layer": "appliance"}, {"deployment": "cloud"}),
    ({"layer": "appliance"}, {"pricing": "per-token"}),
    ({"layer": "gateway"}, {"pricing": "hardware"}),
    ({"layer": "observability"}, {"pricing": "hardware"}),
    ({"layer": "edge-runtime"}, {"deployment": "cloud"}),
    ({"deployment": "device"}, {"pricing": "gpu-hour"}),
    ({"deployment": "cloud"}, {"pricing": "hardware"}),
    ({"customer": "consumers"}, {"pricing": "license"}),
    ({"customer": "oem"}, {"deployment": "cloud"}),
    ({"modality": "science"}, {"customer": "consumers"}),
    ({"modality": "simulation"}, {"customer": "consumers"}),
    ({"modality": "tabular"}, {"customer": "consumers"}),
]


# Who actually buys in each vertical; a "consumer agriculture" or "startup government" idea is noise.
VERTICAL_CUSTOMERS = {
    "healthcare": {"regulated", "enterprise", "smb"},
    "finance": {"regulated", "enterprise", "smb"},
    "legal": {"smb", "enterprise", "regulated"},
    "manufacturing": {"enterprise", "smb", "oem"},
    "agriculture": {"enterprise", "smb", "oem"},
    "construction": {"enterprise", "smb"},
    "retail": {"enterprise", "smb", "oem"},
    "government": {"regulated"},
    "education": {"enterprise", "smb", "consumers"},
    "logistics": {"enterprise", "smb", "oem"},
    "energy": {"enterprise", "regulated"},
    "media": {"developers", "enterprise", "consumers", "smb"},
    "telecom": {"enterprise"},
    "automotive": {"oem", "enterprise", "consumers"},
}


# Where each supply model can physically serve from.
SUPPLY_DEPLOYMENTS = {
    "datacenter": {"cloud", "dedicated"},
    "customer-hw": {"on-prem", "edge", "device"},
    "user-device": {"device"},
    "network-edge": {"edge"},
    "crowd": {"cloud", "edge", "device"},
    "third-party": {"cloud", "dedicated"},
}

# Offerings whose supply the deployment rule below gets wrong.
SUPPLY_OVERRIDES = {
    "Vast.ai": ["crowd"], "Salad": ["crowd"], "io.net": ["crowd"], "Chutes": ["crowd"], "Hyperbolic": ["crowd"],
    "Cloudflare Workers AI": ["network-edge"], "Akamai Inference Cloud": ["network-edge"],
    "Fastly AI Accelerator": ["network-edge"], "AT&T / Verizon edge AI": ["network-edge"],
    "SoftBank AI-RAN": ["network-edge"], "Nokia ARC-Pro cell-site AI": ["network-edge"],
}


def supply_of(row: dict) -> list[str]:
    """Derive an offering's supply facet from its layer and deployments, with a few known exceptions."""
    if row["name"] in SUPPLY_OVERRIDES:
        return SUPPLY_OVERRIDES[row["name"]]
    if "gateway" in row["layer"]:
        return ["third-party"]
    out = []
    for d in row["deployment"]:
        s = {"cloud": "datacenter", "dedicated": "datacenter", "on-prem": "customer-hw",
             "edge": "customer-hw", "device": "user-device"}[d]
        if s not in out:
            out.append(s)
    return out


def is_compatible(combo: dict[str, str]) -> bool:
    allowed = VERTICAL_CUSTOMERS.get(combo.get("vertical"))
    if allowed and combo.get("customer") and combo["customer"] not in allowed:
        return False
    if "supply" in combo and combo.get("deployment") not in SUPPLY_DEPLOYMENTS[combo["supply"]]:
        return False
    for a, b in INCOMPATIBLE:
        if all(combo.get(k) == v for k, v in a.items()) and all(combo.get(k) == v for k, v in b.items()):
            return False
    return True


def describe(combo: dict[str, str]) -> str:
    """Render a facet combination as the same kind of sentence the offerings are written in."""
    vertical = "" if combo["vertical"] == "horizontal" else f" in {VERTICAL[combo['vertical']]}"
    return (
        f"{LAYER[combo['layer']].capitalize()} for {MODALITY[combo['modality']]}, "
        f"{DEPLOYMENT[combo['deployment']]} {SUPPLY[combo['supply']]}, sold to {CUSTOMER[combo['customer']]}{vertical} "
        f"through {PRICING[combo['pricing']]}, competing on {DIFFERENTIATOR[combo['differentiator']]}."
    )
