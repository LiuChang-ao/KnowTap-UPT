"""Project the final per-screen consensus CSVs into the public, image-free schema."""
import argparse
import csv
import hashlib
import io
from pathlib import Path

from validate_release import SCHEMAS, load_release, read_table, require


def serialize(fields, rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True, help="final per-screen author CSV directory")
    p.add_argument("--folds", type=Path, required=True, help="author's imgId,fold_id assignment CSV")
    p.add_argument("--output", type=Path, required=True, help="new or empty output directory")
    args = p.parse_args()
    require(not args.output.exists() or
            (args.output.is_dir() and not any(args.output.iterdir())),
            "refusing to overwrite an existing dataset")
    assignments = read_table(args.folds, ["imgId", "fold_id"])
    folds = {r["imgId"]: int(r["fold_id"]) for r in assignments}
    require(len(folds) == len(assignments), "duplicate UI in fold assignments")
    components, ratings, screens = [], [], []
    for source in sorted(args.annotations.glob("*.csv"), key=lambda f: f.stem):
        sid = source.stem
        require(sid in folds, f"screen not in reference split: {sid}")
        with source.open(encoding="utf-8-sig", newline="") as fp:
            reader = csv.DictReader(fp)
            fields = {"imgId", "objId", "ratingInt", "ratingFinal", "x", "y", "x2", "y2"}
            require(fields <= set(reader.fieldnames or []), f"missing source column: {source.name}")
            rows = sorted(reader, key=lambda r: r["objId"])
        platform, subset = ("Android", "Android") if sid.startswith("Android_") else ("iOS", "iphone")
        screens.append(dict(vins_id=sid, platform=platform, vins_image_path=f"{subset}/JPEGImages/{sid}.jpg",
                            component_count=len(rows), reference_fold=folds[sid]))
        for row in rows:
            require(row["imgId"] == sid, f"filename and screen ID disagree: {source.name}")
            key = dict(vins_id=sid, component_id=row["objId"])
            components.append(dict(**key, x_min=row["x"], y_min=row["y"],
                                   x_max=row["x2"], y_max=row["y2"]))
            # Preserve the decimal strings in the source; never quantize the continuous target.
            ratings.append(dict(**key, rating_final=row["ratingFinal"], rating_rounded=row["ratingInt"]))
    require({r["vins_id"] for r in screens} == set(folds), "screen/fold sets differ")
    tables = {"screens.csv": screens, "components.csv": components, "annotations.csv": ratings}
    files = {name: serialize(SCHEMAS[name], rows) for name, rows in tables.items()}
    manifest = "".join(f"{hashlib.sha256(files[name]).hexdigest()}  {name}\n" for name in sorted(files))
    args.output.mkdir(exist_ok=True, parents=True)
    for name, content in files.items():
        (args.output / name).write_bytes(content)
    (args.output / "SHA256SUMS").write_bytes(manifest.encode("ascii"))
    load_release(args.output)
    print(f"Wrote {len(ratings)} annotations in {len(screens)} screens to {args.output}")


if __name__ == "__main__":
    main()
