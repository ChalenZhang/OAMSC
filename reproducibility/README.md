# Model Evaluation

The evaluation bundle contains three Faster R-CNN EMA detectors, code, and
configurations. Evaluation requires local test images and annotations, not
the style-transferred training bank.

## Model Selection

Architecture, class order, and evaluation settings are in
[models.json](models.json); training settings are in
[training_config.json](training_config.json).

| Model ID | Saved EMA updates | Selection | Reference mAP50 (%) |
| --- | ---: | --- | --- |
| `cityscapes-21-ema` | 20,088 | Fixed endpoint | BDD daytime-clear 44.37; Fog 55.57; Rain 63.08 |
| `realdrivesim-21-ema-validation` | 19,000 | Best on sampled target validation data | Cityscapes 43.46; BDD daytime-all 30.88 |
| `realdrivesim-21-ema-endpoint` | 19,500 | Fixed endpoint | Not recorded |

Reference scores are recorded evaluations of the indicated checkpoints.

## Prepare Test Data

Install [requirements-evaluation.txt](requirements-evaluation.txt), using a
CUDA-compatible PyTorch build for GPU evaluation. Each source directory below
contains the corresponding validation images under `images/` and original
annotations under `labels/`, as expected by the included converters. BDD100K
annotations are per-image JSON; Cityscapes annotations are polygon JSON with
their original city subdirectories. Download them from the official dataset
providers under their terms; [original papers and data links](../data/README.md#original-datasets)
are listed in the data guide. Test data is not bundled.

```bash
python code/prepare_fasterrcnn_bdd100k.py \
  --dataset-root /path/to/BDD100K-Validation \
  --output-dir /path/to/Prepared-Test-Data \
  --name bdd_daytime_clear --timeofday daytime --weather clear

python code/prepare_fasterrcnn_bdd100k.py \
  --dataset-root /path/to/BDD100K-Validation \
  --output-dir /path/to/Prepared-Test-Data \
  --name bdd_daytime_all --timeofday daytime --weather all

python code/prepare_fasterrcnn_cityscapes_foggy.py \
  --dataset-root /path/to/Foggy-Cityscapes-Validation \
  --output-dir /path/to/Prepared-Test-Data --name foggy --beta 0.02

python code/prepare_fasterrcnn_cityscapes_rain.py \
  --dataset-root /path/to/Rainy-Cityscapes-Validation \
  --output-dir /path/to/Prepared-Test-Data --name rainy \
  --alpha 0.02 --beta all --dropsize all --pattern all

python code/prepare_fasterrcnn_cityscapes_val.py \
  --dataset-root /path/to/Cityscapes-Validation \
  --output-dir /path/to/Prepared-Test-Data --name cityscapes
```

## Evaluate

Extract the bundle and run commands from its root, where `weights/`, `code/`,
and `reproducibility/` reside. The evaluator loads all detector tensors locally
and does not fetch pretrained backbone weights.

```bash
python code/evaluate_released_model.py \
  --model cityscapes-21-ema --benchmark bdd-daytime-clear \
  --data /path/to/Prepared-Test-Data/bdd_daytime_clear_coco.json \
  --out /path/to/Evaluation-Results/cityscapes-bdd.csv
```

Use `foggy-cityscapes` with `foggy_coco.json` and `rainy-cityscapes` with
`rainy_coco.json` for the other Cityscapes-source rows. Use
`realdrivesim-21-ema-validation` with `cityscapes` / `cityscapes_coco.json` or
`bdd-daytime-all` / `bdd_daytime_all_coco.json` for the RealDriveSim rows.
The alternative `realdrivesim-21-ema-endpoint` is evaluated the same way.

For a two-image CPU test, add `--device cpu --workers 0 --limit-images 2`.
Use the full split and listed filters for reference metrics. Raw AP values
are fractions; multiply by 100 for percentages.

## Minimal Statistics

- [reference_metrics.csv](statistics/reference_metrics.csv): five recorded
  cross-domain evaluations, including classwise AP50 and exact subset counts.
- [style_counts.csv](statistics/style_counts.csv): retained counts for Origin
  and 20 canonical styles, with missing-view counts for both sources.
- [training_epochs.csv](statistics/training_epochs.csv): recorded epoch loss
  and learning rate for the two complete training runs.
- [corruption_by_severity.csv](statistics/corruption_by_severity.csv) and
  [corruption_summary.csv](statistics/corruption_summary.csv): the 76 conditions
  and their summaries, keeping seven-class AP50, seven-class AP50:95, and
  car-only AP50:95 separate. Average the five severities, then the 15 corruption
  means; rPC is `100 * mPC / Clean`.

Recompute these summaries without models or test data:

```bash
python code/verify_reproduction_statistics.py
```

Cityscapes-source corruption results, separated by metric scope:

| Metric | Clean (%) | mPC (%) | rPC (%) |
| --- | ---: | ---: | ---: |
| Seven-class AP50 | 64.38 | 41.15 | 63.91 |
| Seven-class AP50:95 | 37.49 | 22.97 | 61.28 |
| Car-only AP50:95 | 52.66 | 36.54 | 69.40 |

The corruption generator includes the full input path in its random seed;
changing paths can change stochastic corruption outputs.

To run the corruption workflow on authorized local images:

```bash
python code/generate_cityscapes_corruption_dataset.py \
  --data /path/to/Prepared-Test-Data/cityscapes_coco.json \
  --out-root /path/to/Cityscapes-C --workers 4
python code/evaluate_cityscapes_corruption_precomputed.py \
  --precomputed-root /path/to/Cityscapes-C --project weights \
  --manifest reproducibility/corruption_models.csv --weights-type ema \
  --workers 4 --out-dir /path/to/Evaluation-Results/corruption
```

## Distribution

[Cityscapes License agreement item 3](https://www.cityscapes-dataset.com/license/)
expressly permits qualifying abstract derivatives, including trained models,
provided they do not allow recovery of the dataset or similar content.
The Cityscapes-trained weights remain subject to its non-commercial terms.
RealDriveSim source attribution and modifications are retained in the
[attribution notice](../data/RealDriveSim-Multi-Style/ATTRIBUTION.md).
