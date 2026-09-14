# Cityscapes Multi-Style Reproduction Guide

Model weights and evaluation commands are in the
[evaluation guide](../../reproducibility/README.md).

## Why Images Are Not Included

This repository intentionally distributes no Cityscapes Origin image,
style-transferred image, or inherited annotation. The official
[Cityscapes Terms and Conditions](https://www.cityscapes-dataset.com/license/)
contain two directly relevant restrictions:

- License agreement, item 3: "That you do not distribute this dataset or
  modified versions."
- Terms of Use, Section 4.2, prohibits making protected dataset contents
  accessible to third parties and applies the same rule to modified or
  derived works when the dataset can be reconstructed or derived from them.

Obtain images and annotations from Cityscapes under its terms and keep
style-transferred views local unless the rights holder authorizes distribution.

## Local Reproduction Procedure

1. Register at the official Cityscapes portal, accept its current terms, and
   download the required images and annotations directly to an authorized
   local machine.
2. Use an image-to-image editing model. The style bank was generated with
   Seedream; GPT-based or comparable image editors are alternatives for
   local generation.
3. For each directory, substitute its canonical style name into the minimal
   prompt: `Change the visual style of this image to <STYLE_NAME>.`
   A stricter optional form is: `Change the visual style of this image to
   <STYLE_NAME> while preserving every object, camera geometry, road layout,
   framing, spatial position, and image dimensions.`
4. Preserve the source filename stem. Store each accepted generated image
   under `<STYLE_NAME>/images/` and the corresponding authorized local label
   under `<STYLE_NAME>/labels/`.
5. Compare every generated view side by side with its Origin. Reject and
   regenerate any output containing a missing, duplicated, displaced,
   deformed, or newly introduced object; altered camera or road geometry;
   changed crop, padding, or resolution; obvious hallucination; unreadable
   content; or file corruption.
6. Record failed or unavailable views instead of inventing correspondence.
   The preparation scripts tolerate missing styles while preserving the
   scene and style identifiers of retained files.

The [RealDriveSim multi-style release](../RealDriveSim-Multi-Style/README.md)
provides distributable examples of the same generation and acceptance
procedure. Questions about the procedure or rights concerns can be submitted
to the repository maintainers through the
[GitHub issue tracker](https://github.com/ChalenZhang/OAMSC/issues).

## Canonical Layout

```text
Cityscapes-Multi-Style/
  README.md
  classes.txt
  style_names.txt
  <Style>/
    images/
    labels/
  ...
```

Create one `<Style>` directory for every entry in `style_names.txt`. This is
the same per-style image/label organization used by the downloadable
RealDriveSim archive, while the Cityscapes contents remain local.
