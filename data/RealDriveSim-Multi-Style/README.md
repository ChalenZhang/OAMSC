# RealDriveSim Multi-Style Dataset

Model weights and evaluation commands are in the
[evaluation guide](../../reproducibility/README.md).

## Download

Download the `data-v1.0` release using the commands in the
[root download section](../../README.md#dataset-download).

The release contains 119,983 manually accepted style-transferred images and
matching seven-class YOLO labels across 20 styles. They were generated from
6,000 randomly selected RealDriveSim RGB scenes: 3,480 Adverse Weather and
2,520 Normal Weather scenes. Original images are available from the official
RealDriveSim project and are not duplicated in this release.

Each style is stored as numbered pieces named:

```text
OAMSC-RealDriveSim-Multi-Style-v1.0--<index>-<Style>.tar.part-<number>
```

Download all pieces for a style, then concatenate and extract them in numeric
order:

```bash
cat OAMSC-RealDriveSim-Multi-Style-v1.0--01-Oil-Painting.tar.part-* \
  | tar -xf -
```

The extracted style directory contains `images/`, matching `labels/`, and a
short provenance notice. Correspondence is defined by the shared filename
stem. The canonical class and style names are listed in
[`classes.txt`](classes.txt) and [`style_names.txt`](style_names.txt).

## Attribution

These files are appearance-transferred derivatives of
[RealDriveSim](https://realdrivesim.github.io/), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). OAMSC modifications
include source-scene selection, image-to-image style transfer, manual
acceptance and regeneration, seven-class label mapping, and canonical English
naming. Retain the [full attribution notice](ATTRIBUTION.md) when sharing the
derivatives.
