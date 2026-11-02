"""Check the UPT release without downloading screenshots (Python 3.9+)."""
import argparse
import csv
import hashlib
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = {
    "screens.csv": ["vins_id", "platform", "vins_image_path", "component_count", "reference_fold"],
    "components.csv": ["vins_id", "component_id", "x_min", "y_min", "x_max", "y_max"],
    "annotations.csv": ["vins_id", "component_id", "rating_final", "rating_rounded"],
}
HISTOGRAM = [2146, 1204, 1882, 2138, 1594, 787]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_table(path, fields):
    with path.open(encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        require(reader.fieldnames == fields, f"unexpected header: {path.name}")
        rows = list(reader)
    require(all(None not in r and None not in r.values() for r in rows),
            f"malformed row: {path.name}")
    return rows


def index(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        require(key not in result, f"duplicate key: {key}")
        result[key] = row
    return result


def load_release(directory, vins_root=None, check_hashes=True):
    directory = Path(directory)
    if check_hashes:
        manifest = (directory / "SHA256SUMS").read_text(encoding="ascii").splitlines()
        expected = [f"{hashlib.sha256((directory / name).read_bytes()).hexdigest()}  {name}"
                    for name in sorted(SCHEMAS)]
        require(manifest == expected, "data checksum mismatch")
    tables = {name: read_table(directory / name, fields) for name, fields in SCHEMAS.items()}
    screens = {k[0]: v for k, v in index(tables["screens.csv"], ["vins_id"]).items()}
    keys = ["vins_id", "component_id"]
    components = index(tables["components.csv"], keys)
    labels = index(tables["annotations.csv"], keys)
    require(len(screens) == 800 and len(components) == 9751, "wrong release dimensions")
    require(components.keys() == labels.keys(), "component/label keys differ")
    sizes = Counter(k[0] for k in components)
    require(set(sizes) == screens.keys(), "screen/component keys differ")
    folds, platforms, hist = Counter(), Counter(), Counter()
    for sid, row in screens.items():
        require(re.fullmatch(r"(Android|IMG)_[0-9]+", sid), f"invalid VINS ID: {sid}")
        platform, subset = ("Android", "Android") if sid.startswith("Android_") else ("iOS", "iphone")
        ref = f"{subset}/JPEGImages/{sid}.jpg"
        require(row["platform"] == platform and row["vins_image_path"] == ref, f"bad source: {sid}")
        require(int(row["component_count"]) == sizes[sid], f"wrong component count: {sid}")
        fold = int(row["reference_fold"])
        require(fold in range(5), f"unknown fold: {sid}")
        platforms[platform] += 1
        folds[fold] += 1
        if vins_root:
            require((Path(vins_root) / ref).is_file(), f"missing VINS source: {ref}")
    require(platforms == {"Android": 422, "iOS": 378}, "wrong platform counts")
    require(folds == {n: 160 for n in range(5)}, "wrong reference fold sizes")
    for key, row in components.items():
        require(re.fullmatch(r"[A-Za-z]+_[0-9]+(?:_[A-Za-z]+_[0-9]+)*", key[1]),
                f"invalid component ID: {key}")
        x, y, x2, y2 = (float(row[f]) for f in ("x_min", "y_min", "x_max", "y_max"))
        require(0 <= x < x2 <= 1 and 0 <= y < y2 <= 1, f"invalid box: {key}")
        val, band = float(labels[key]["rating_final"]), int(labels[key]["rating_rounded"])
        require(math.isfinite(val) and 0 <= val <= 5 and band == round(val), f"invalid rating: {key}")
        hist[band] += 1
    require([hist[n] for n in range(6)] == HISTOGRAM, "rating distribution differs from the paper")
    return screens, components, labels


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--vins-root", type=Path, help="optional local VINS_Dataset directory")
    args = p.parse_args()
    screens, components, labels = load_release(args.data, args.vins_root)
    print(f"OK: {len(screens)} screens, {len(components)} component boxes, {len(labels)} annotations")
    print("Rounded rating counts (0–5):", *HISTOGRAM)
