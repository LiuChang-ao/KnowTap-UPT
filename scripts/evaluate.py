"""Score out-of-fold predictions or run a screenshot-free geometry reference baseline."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from validate_release import ROOT, index, load_release, read_table, require

PRED_FIELDS = ["vins_id", "component_id", "reference_fold", "prediction"]


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(order)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i, j):
            out[order[k]] = (i + j + 1) / 2
        i = j
    return out


def metrics(ys, ps):
    ybar = mean(ys)
    total = sum((y - ybar) ** 2 for y in ys)
    r2 = 1 - sum((y - p) ** 2 for y, p in zip(ys, ps)) / total if total else None
    yr, pr = ranks(ys), ranks(ps)
    ybar, pbar = mean(yr), mean(pr)
    den = math.sqrt(sum((v - ybar) ** 2 for v in yr) * sum((v - pbar) ** 2 for v in pr))
    rho = sum((y - ybar) * (p - pbar) for y, p in zip(yr, pr)) / den if den else None
    return {"srcc": rho, "r2": r2}


def summarize(scores, labels, screens):
    require(scores.keys() == labels.keys(), "predictions must cover every release key exactly once")
    groups = defaultdict(list)
    for key, pred in scores.items():
        require(math.isfinite(pred), f"nonfinite prediction: {key}")
        groups[int(screens[key[0]]["reference_fold"])].append((float(labels[key]["rating_final"]), pred))
    result = []
    for fold in range(5):
        y, p = zip(*groups[fold])
        result.append(dict(fold=fold, n=len(y), **metrics(y, p)))
    summary = {}
    for metric in ("srcc", "r2"):
        values = [r[metric] for r in result]
        summary[metric] = ({"mean": mean(values), "sample_sd": stdev(values)}
                           if None not in values else {"mean": None, "sample_sd": None})
    return {"folds": result, "summary": summary}


def geometry(components, labels, screens):
    """A small demonstrator, not the paper's full constructed-feature model."""
    import numpy as np

    keys = sorted(components)
    x, y, folds = [], [], []
    for k in keys:
        row = components[k]
        l, t, r, b = (float(row[f]) for f in ("x_min", "y_min", "x_max", "y_max"))
        w, h = r - l, b - t
        x.append([1., w, h, w * h, (l + r) / 2, (t + b) / 2, int(screens[k[0]]["component_count"])])
        y.append(float(labels[k]["rating_final"]))
        folds.append(int(screens[k[0]]["reference_fold"]))
    x, y, folds = map(np.asarray, (x, y, folds))
    scores = {}
    for f in range(5):
        held = folds == f
        coef = np.linalg.lstsq(x[~held], y[~held], rcond=None)[0]
        scores.update((keys[i], float(p)) for i, p in zip(np.flatnonzero(held), x[held] @ coef))
    return scores


def write_scores(path, scores, screens):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(PRED_FIELDS)
        for (sid, cid), pred in sorted(scores.items()):
            writer.writerow([sid, cid, screens[sid]["reference_fold"], pred])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--geometry-baseline", action="store_true")
    mode.add_argument("--predictions", type=Path, help="one held-out prediction per released key")
    p.add_argument("--write-predictions", type=Path, help="optional path for geometry output; must be new")
    args = p.parse_args()
    screens, components, labels = load_release(args.data)
    if args.geometry_baseline:
        scores = geometry(components, labels, screens)
        if args.write_predictions:
            write_scores(args.write_predictions, scores, screens)
    else:
        require(not args.write_predictions, "--write-predictions is for the geometry example")
        table = index(read_table(args.predictions, PRED_FIELDS), ["vins_id", "component_id"])
        require(table.keys() == labels.keys(), "unknown or missing prediction keys")
        scores = {}
        for k, r in table.items():
            require(int(r["reference_fold"]) == int(screens[k[0]]["reference_fold"]),
                    f"prediction's fold does not match its screen: {k}")
            scores[k] = float(r["prediction"])
    print(json.dumps(summarize(scores, labels, screens), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
