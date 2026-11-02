# KnowTap: User-Perceived Tappability (UPT) dataset

UPT contains visually perceived tappability annotations for **9,751 components on 800 mobile UI screens**. A target describes how strongly a component *looks* tappable in its screen context; it does not measure touch accuracy or whether an element actually responds to a tap.

**Paper:** Chang'ao Liu, Chengyi Shen, Zhitong Cui, Yao Wang, and Shijian Luo. [KnowTap: A Knowledge-Driven Tool for Fine-Grained Tappability Assessment and Design Iteration of Mobile UIs](https://doi.org/10.1145/3830398.3830517). UIST 2026.

**Source screens:** Obtain the original images from the [official VINS dataset](https://github.com/sbunian/VINS#vins-dataset). This repository contains no screenshots, crops, OCR transcripts, embeddings, or image archives. Paths in the index are relative to the extracted `VINS_Dataset` directory.

| File | Rows | Contents |
| --- | ---: | --- |
| [screens.csv](data/screens.csv) | 800 | VINS screen ID, platform, upstream image path, component count and reference fold |
| [components.csv](data/components.csv) | 9,751 | Retained component IDs and normalized bounding boxes |
| [annotations.csv](data/annotations.csv) | 9,751 | Final continuous training target and its rounded display value |
| [SHA256SUMS](data/SHA256SUMS) | 3 | SHA-256 digest of each released CSV |

Join the two component tables on **both** `vins_id` and `component_id`. To retrieve the corresponding screen, join on `vins_id` and resolve `vins_image_path` under your own VINS download. See the [data dictionary and provenance](DATA.md) for coordinate conventions, merged components, label construction and filtering.

## Quick start

From the repository root, run the release checks with Python 3.9 or newer; no additional packages or images are needed:

```sh
python scripts/validate_release.py
```

To also check that all 800 referenced images exist in an extracted VINS distribution:

```sh
python scripts/validate_release.py --vins-root /path/to/VINS_Dataset
```

The validator checks the file hashes, unique keys, one-to-one joins, bounding boxes, score range, rounded label counts and screen-level folds. Example of reading one annotated box using only the standard library:

```python
import csv

def rows(name):
    with open(f"data/{name}.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

labels = {(r["vins_id"], r["component_id"]): r for r in rows("annotations")}
box = rows("components")[0]
target = labels[box["vins_id"], box["component_id"]]["rating_final"]
print(box["vins_id"], box["component_id"], box["x_min"], box["y_min"], target)
```

## Reproducibility

The [evaluation guide](EVALUATION.md) shows how to score held-out predictions, run a geometry-only example, or train a reference RGB + target-mask ResNet-18 using a local VINS download. These scripts are release implementations; they are not the historical KnowTap training pipeline. No checkpoint or claim of reproducing the paper's reported training runs is included.

The CSVs are ready to use as distributed. The [export script](scripts/export_annotations.py) documents the field projection from the authors' final per-screen tables; it is not needed to use the dataset.

## Attribution and licenses

Please cite [KnowTap](https://doi.org/10.1145/3830398.3830517) when using UPT. When using the underlying screens, also credit Bunian et al., *VINS: Visual Search for Mobile User Interface Design* ([official project](https://github.com/sbunian/VINS)). Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).

The authors' UPT annotations and original metadata are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); retain attribution and indicate changes. The new scripts are licensed under MIT. Neither license grants rights to the upstream screens, app designs or other third-party assets. See [LICENSE.md](LICENSE.md) for the scope of each license.
