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
    "science": "scientific and biology models",
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

FACETS = {
    "layer": LAYER,
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


def is_compatible(combo: dict[str, str]) -> bool:
    allowed = VERTICAL_CUSTOMERS.get(combo.get("vertical"))
    if allowed and combo.get("customer") and combo["customer"] not in allowed:
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
        f"{DEPLOYMENT[combo['deployment']]}, sold to {CUSTOMER[combo['customer']]}{vertical} "
        f"through {PRICING[combo['pricing']]}, competing on {DIFFERENTIATOR[combo['differentiator']]}."
    )
