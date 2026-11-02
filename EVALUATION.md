# Grouped evaluation

Use `rating_final` as the target. A component-level random split can put elements from the same screen in both training and validation. The reference assignment in [screens.csv](data/screens.csv) keeps each UI in exactly one of five held-out folds (160 screens each). It comes from the project’s saved `fold_assignments.csv`. It was checked against the saved training/validation lists, but has not been independently established as the exact assignment of every model in the paper.

## Score your predictions

Write one **held-out** prediction per component with this exact CSV header:

```csv
vins_id,component_id,reference_fold,prediction
```

Use predictions on the **0–5 target scale**. Train on the other four screen folds; fit any learned transforms, including dimensionality reduction, on those training folds only. The scorer checks keys, coverage, finite values and fold IDs; a prediction file alone cannot prove how a model was trained.

```sh
python scripts/evaluate.py --predictions path/to/oof_predictions.csv
```

It reports Spearman rank correlation (average ranks for ties) and R² per fold, followed by the unweighted fold mean and sample standard deviation. Undefined statistics, such as correlation with a constant predictor, are reported as `null`.

## A runnable geometry baseline

This example uses least-squares regression on box width, height, area, center coordinates and the number of retained components on that screen. It needs NumPy but no images.

```sh
python scripts/evaluate.py --geometry-baseline --write-predictions outputs/geometry.csv
python scripts/evaluate.py --predictions outputs/geometry.csv
```

This is an example of the split and scoring procedure, not the paper’s random forest with design, text and deep features.

## Reference ResNet-18 baseline

The optional [RGB + mask script](scripts/resnet18_mask.py) implements the adapted baseline described in the paper: full-screen RGB in [0,1], a fourth channel containing the binary target mask, a scalar regression head, and MSE on the target divided by five. Predictions are converted back to the 0–5 scale. The script requires PyTorch, torchvision, Pillow and NumPy. Obtain the screens yourself from the [official VINS project](https://github.com/sbunian/VINS#vins-dataset).

```sh
python scripts/resnet18_mask.py --vins-root /path/to/VINS_Dataset --smoke
python scripts/resnet18_mask.py --vins-root /path/to/VINS_Dataset --device cuda --output outputs/mask_oof.csv
python scripts/evaluate.py --predictions outputs/mask_oof.csv
```

Random initialization is the default. With `--imagenet`, torchvision may download its ImageNet weights; the fourth input channel is initialized from the mean of the three RGB-channel weights, and all layers remain trainable. Run outputs are not part of the dataset. The resize, optimizer, training length and seed in this standalone example are reference choices, not verified settings from the historical experiments. A smoke run exercises one training and one held-out batch and does not produce benchmark results.
