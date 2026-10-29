"""Bake out/map.json into a single self-contained HTML page (out/market_map.html)."""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=HERE / "out" / "map.json")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "market_map.html")
    args = ap.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    # Keep the page light: the map needs positions and display fields, not every diagnostic.
    for o in data["offerings"]:
        o.pop("nn_distance", None)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    template = (HERE / "web" / "template.html").read_text(encoding="utf-8")
    args.out.write_text(template.replace("/*__DATA__*/null", payload), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
