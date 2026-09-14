# OAMSC

Code and data for Origin-Anchored Multi-Style Consistency (OAMSC).

## Models and Evaluation

- **[Download models and evaluation code](https://github.com/ChalenZhang/OAMSC/releases/download/models-v1.0/OAMSC-Evaluation-v1.0.zip)**
- **[Download minimal statistics](https://github.com/ChalenZhang/OAMSC/releases/download/models-v1.0/OAMSC-Statistics-v1.0.zip)**

The evaluation bundle provides three EMA detectors, code, and fixed settings.
Supply the test data locally to rerun evaluation without commercial style
images. The statistics bundle contains aggregate experimental records.
See the [evaluation guide](reproducibility/README.md) for
commands, model selection, and evaluation protocols.

## Dataset Download

**[Open the RealDriveSim Multi-Style Dataset release](https://github.com/ChalenZhang/OAMSC/releases/tag/data-v1.0)**

The dataset files are stored under **Assets** on that release page, not in
the repository tree. The release provides 119,983 generated images across
20 styles together with matching seven-class YOLO labels. The original
RealDriveSim images are not duplicated.

Download every style with [GitHub CLI](https://cli.github.com/):

```bash
mkdir -p /path/to/OAMSC-downloads
gh release download data-v1.0 \
  --repo ChalenZhang/OAMSC \
  --pattern 'OAMSC-RealDriveSim-Multi-Style-v1.0--*.tar.part-*' \
  --dir /path/to/OAMSC-downloads \
  --skip-existing
```

To download one style, replace `Oil-Painting` with a name from
[`style_names.txt`](data/RealDriveSim-Multi-Style/style_names.txt):

```bash
STYLE=Oil-Painting
gh release download data-v1.0 \
  --repo ChalenZhang/OAMSC \
  --pattern "OAMSC-RealDriveSim-Multi-Style-v1.0--*-${STYLE}.tar.part-*" \
  --dir /path/to/OAMSC-downloads \
  --skip-existing
```

Each style is a split tar stream. Concatenate its numbered parts and extract
it as follows; running the loop for multiple styles merges them into the same
canonical dataset root.

```bash
mkdir -p /path/to/OAMSC-data
for first in /path/to/OAMSC-downloads/*.tar.part-001; do
  prefix=${first%.part-001}
  cat "${prefix}".part-* | tar -xf - -C /path/to/OAMSC-data
done
```

The extracted files appear under:

```text
/path/to/OAMSC-data/OAMSC-Full-Multi-Style-Dataset/
  RealDriveSim-Multi-Style/
    <Style>/images/
    <Style>/labels/
```

## Repository Contents

- [`code/`](code/): training, data preparation, and evaluation scripts.
- [`data/RealDriveSim-Multi-Style/`](data/RealDriveSim-Multi-Style/): dataset
  conventions and source attribution.
- [`data/Cityscapes-Multi-Style/`](data/Cityscapes-Multi-Style/): local
  reproduction guide. Cityscapes-derived images are not redistributed.

## Setup

```bash
python -m venv /path/to/Virtual-Environment
source /path/to/Virtual-Environment/bin/activate
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with the local CUDA runtime when needed.
The principal entry points are documented in [`code/README.md`](code/README.md).

## Data Conventions

Images and labels from the same scene share a filename stem. Labels use
normalized YOLO rows:

```text
class_id x_center y_center width height
```

Class identifiers `0` through `6` denote bicycle, bus, car, motorcycle,
person, rider, and truck. Preparation scripts convert these records to COCO
format for Faster R-CNN.

## Source and Terms

The released images are appearance-transferred derivatives of the public
[RealDriveSim](https://realdrivesim.github.io/) dataset, which is licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The source
and OAMSC modifications are recorded in the
[attribution notice](data/RealDriveSim-Multi-Style/ATTRIBUTION.md).

Cityscapes images and inherited annotations are not distributed. Authorized
users can follow the [local reproduction guide](data/Cityscapes-Multi-Style/README.md).
The Cityscapes-trained detector is provided separately under the conditions
described in the [model distribution notes](reproducibility/README.md#distribution).

This repository is provided for non-commercial academic research and
scholarly exchange. Third-party materials remain subject to their original
terms. Please report substantiated copyright, privacy, licensing, or other
rights concerns through the [issue tracker](https://github.com/ChalenZhang/OAMSC/issues).
The maintainers will review them promptly and, where appropriate, correct or
remove affected material.
