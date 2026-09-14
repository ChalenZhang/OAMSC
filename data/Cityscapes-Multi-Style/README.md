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

## Generation Models and Style Bank

We constructed the offline style bank using the commercial Seedream
[3.0](https://arxiv.org/abs/2504.11346),
[4.0](https://arxiv.org/abs/2509.20427),
[4.5](https://seed.bytedance.com/en/seedream4_5), and
[5.0 Lite](https://seed.bytedance.com/en/seedream5_0_lite) models.
Each source image serves as the photographic Origin and is rendered in
20 styles, with five styles from each family:

| Family | Styles |
| --- | --- |
| Painting and Fine Art | Oil Painting, Sketching, Line Art, Crayon Drawing, Chinese Painting |
| Anime and Comics | Ghibli Anime, American Comics, Chibi Comics, Painterly Anime, Pixel Art |
| Film and Games | 3D Modeling, Post-Apocalyptic, Science-Fiction, Cyberpunk, AAA Game Scene |
| Materials and Crafts | Paper Cutting, Stained Glass, Building Blocks, Collage, Textile Art |

Every generated view is manually compared with its Origin. Only accepted
views that preserve object identity, location, geometry, and annotation
correspondence enter the bank; they inherit the original bounding boxes and
classes. Generation and curation are completed offline, with no image
generator invoked during detector training or inference.

## Local Reproduction Procedure

1. Register at the official Cityscapes portal, accept its current terms, and
   download the required images and annotations directly to an authorized
   local machine. Generate the training bank from source training images,
   keeping evaluation images separate.
2. Use an image-conditioned editor from the Seedream family or an
   alternative below. Supply the Origin as the image input and a target
   style as the text instruction. Start each style from the Origin, not
   from another stylized output.
3. Apply a prompt below for each of the 20 canonical styles. Keep the
   original framing and image dimensions so the source boxes remain valid.
4. Compare every generated view side by side with its Origin. Reject and
   regenerate any output containing a missing, duplicated, displaced,
   deformed, or newly introduced object; altered camera or road geometry;
   changed crop, padding, or resolution; obvious hallucination; unreadable
   content; or file corruption.
5. Preserve the source filename stem. Store each accepted generated image
   under `<STYLE_NAME>/images/` and its unchanged authorized local label
   under `<STYLE_NAME>/labels/`. Retain the original image and label in
   `Origin/images/` and `Origin/labels/`.
6. Record failed or unavailable views instead of inventing correspondence.
   The preparation scripts tolerate missing styles while preserving the
   scene and style identifiers of retained files. Convert the accepted bank
   with [the COCO preparation script](../../code/prepare_fasterrcnn_fixed_all_scenes.py)
   before following the [training commands](../../code/README.md).

The Faster R-CNN pipeline uses an ImageNet-pretrained ResNet-101-FPN.
Each training group contains one Origin and three accepted styles with a
shared geometric transform. All four images supervise the student;
styled student features align with the EMA teacher's Origin features at
global FPN and GT-RoI levels.

## Prompt Templates

For local regeneration, substitute a canonical display name such as
`Oil Painting`, `Sketching`, or `Cyberpunk` into this minimal prompt:

```text
Change the visual style of this image to <STYLE_NAME>.
```

For stronger appearance-only instructions, use:

```text
Change only the visual style of the supplied image to <STYLE_NAME>.
Preserve all objects, their identities, counts, positions, sizes, and
silhouettes. Keep the camera viewpoint, road layout, framing, and image
dimensions unchanged. Modify only texture, palette, and rendering medium.
Do not add, remove, duplicate, move, or deform any object.
```

Use spaces in display names and hyphens in directory names, for example
`Oil Painting` in the prompt and `Oil-Painting/images/` on disk.
These instructions do not replace the full manual acceptance check.

## Alternative Image Editors

The same Origin-plus-prompt procedure can also be used with:

| Model | Use |
| --- | --- |
| [GPT Image 2.5 Sunburst](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst) | Supply the Origin and the style instruction to image editing. |
| [FLUX.2 pro](https://docs.bfl.ai/flux_2/flux2_image_editing) (`flux-2-pro`) | Use single-reference editing with the Origin as the input image. |
| [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | Run the available weights locally with an image input and an editing prompt. |

Apply the same full manual inspection and regeneration procedure.
Use a local deployment for restricted images; a hosted editor requires
permission covering disclosure to its provider.

The [RealDriveSim multi-style release](../RealDriveSim-Multi-Style/README.md)
provides distributable examples of the same generation and acceptance
procedure. Questions about the procedure or rights concerns can be submitted
to the repository maintainers through the
[GitHub issue tracker](https://github.com/ChalenZhang/OAMSC/issues).

## Canonical Layout

```text
/path/to/Cityscapes-Multi-Style/
  README.md
  classes.txt
  style_names.txt
  Origin/
    images/<scene>.<ext>
    labels/<scene>.txt
  Oil-Painting/
    images/<scene>.<ext>
    labels/<scene>.txt
  <Style>/
    images/
    labels/
  ...
```

Create one `<Style>` directory for every entry in `style_names.txt`, in
addition to `Origin`. The list contains 20 styles and excludes Origin. This is
the same per-style image/label organization used by the downloadable
RealDriveSim archive, while the Cityscapes contents remain local.
