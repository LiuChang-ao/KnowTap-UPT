# Data dictionary and provenance

This release is a projection of the final retained (D1) tables used in the KnowTap analyses. The initial sample contained 1,000 VINS screens. Candidate components came from leaf elements, with very small and very large components filtered during preprocessing. The study excluded 137 screens for segmentation problems and 63 disputed screens for which the pre-labels would have required disproportionate correction. Only the 800 retained screens and their 9,751 final components are released. The disputed screens tended to be denser; the sample should not be treated as representative of all mobile interfaces.

## Identifiers and geometry

| Column | Definition |
| --- | --- |
| `vins_id` | Case-sensitive VINS screenshot filename stem, including leading zeros (e.g., `IMG_0630` in the upstream naming scheme). Treat IDs as strings. |
| `component_id` | ID in the authors' cleaned component list; unique **within a screen**. It is not a global ID or a VINS XML row number. |
| `x_min`, `y_min`, `x_max`, `y_max` | Left, top, right, bottom relative to screen width and height, respectively. Origin: upper left; x grows right, y grows down. These are corners, not width and height. |
| `platform`, `vins_image_path` | Android or iOS, and the relative path in the official VINS distribution. The final sample includes 422 Android screens and 378 iPhone screens. |
| `component_count` | Number of retained components on the screen. This counts cleaned components, not every element in the upstream VINS annotation. |
| `reference_fold` | Zero-based held-out fold in the accompanying five-way, screen-level assignment. All components of a screen share one fold (see [evaluation notes](EVALUATION.md)). |

For an image of width `W` and height `H`, the pixel box is `(x_min*W, y_min*H, x_max*W, y_max*H)`. The release stores the source coordinate values without rounding. There are 5,151 components on Android screens and 4,600 on iOS screens.

## Labels

The annotation protocol started with GPT-4o pre-labels followed by independent correction by three crowdworker groups on a 0–5 scale. Use **`rating_final`** (the source `ratingFinal`) as the continuous regression target. The descriptive counts below are computed from this score with `round(float(rating_final))` in Python (ties to even). Individual votes cannot be recovered from the final target.

Component merging matters: **672 retained rows** combine an icon and a text element. Their final bounding box is the union of the two part boxes and their score is the mean of the two part consensus scores. This was checked against the pre-merge table for all 672 rows. As a result, **350 targets are not multiples of one third** (up to floating-point precision). They are present in the analysis data and have been preserved; do not round them to reconstruct three integer votes. For example, the retained `Android_1 / Icon_12_Text_25` has `rating_final = 3.5`.

| Rounded value | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Components | 2,146 | 1,204 | 1,882 | 2,138 | 1,594 | 787 |

## Source checks

- The release uses the final per-screen consensus CSVs (`clean_ocr_v2_structure_merge_csvs`), not GPT pre-labels, model predictions, detector training data, or augmented copies.
- The retained keys and numeric fields match the previous cleaned OCR export; the later export updated OCR text, which is not distributed here.
- Every released key, box and continuous target was checked against the companion per-screen JSON and the modeling feature table. The human target column in the camera-ready model-comparison table independently matches every released target.
- Each of the 800 retained JPEG files matched the SHA-256 digest of the image identified by its path in the local VINS distribution. No pixels were included in the export.

To regenerate these three CSVs **from the authors' local derived tables**, use the command below. Raw VINS alone does not contain the human consensus labels. The output directory must be new or empty. This step does not read or export images.

```sh
python scripts/export_annotations.py --annotations /path/to/clean_ocr_v2_structure_merge_csvs --folds /path/to/group_kfold_splits_v1/fold_assignments.csv --output rebuilt-data
```
