# Code

The main entry points are:

- `evaluate_released_model.py`: evaluate a tensor-only EMA checkpoint with the
  fixed protocol in the [reproduction guide](../reproducibility/README.md).
- `verify_reproduction_statistics.py`: check published aggregate calculations
  without models or test images.
- `train_fasterrcnn_resnet101.py`: Faster R-CNN OAMSC training with grouped
  scenes, synchronized geometry, EMA teacher references, and global and
  GT-RoI consistency losses.
- `train_yolov8x_oamsc.py`: YOLOv8x compatibility training.
- `prepare_fasterrcnn_fixed_all_scenes.py`: Cityscapes multi-style COCO
  preparation.
- `prepare_fasterrcnn_realdrivesim_fixed_all_scenes.py`: RealDriveSim
  multi-style COCO preparation.
- `prepare_fasterrcnn_bdd100k.py`, `prepare_fasterrcnn_cityscapes_foggy.py`,
  and `prepare_fasterrcnn_cityscapes_rain.py`: target-domain preparation.
- `evaluate_*.py`: Faster R-CNN, YOLO, weather, and corruption evaluation.

Run any script with `--help` for its complete interface. Dataset paths,
checkpoints, outputs, and GPU selections are command-line parameters. Replace
`/path/to/...` in examples with your local locations.

The Faster R-CNN reference preparation and training sequence begins with:

```bash
python code/prepare_fasterrcnn_fixed_all_scenes.py \
  --dataset-root /path/to/Cityscapes-Multi-Style \
  --output-dir /path/to/Prepared-Cityscapes \
  --expected-scenes 2975

torchrun --standalone --nproc_per_node 2 \
  code/train_fasterrcnn_resnet101.py \
  --data /path/to/Prepared-Cityscapes/train_coco.json \
  --epochs 27 --batch-size 16 --lr 0.01 \
  --scene-grouped-batches --origin-anchor-groups \
  --styles-per-anchor 3 --sync-scene-hflip \
  --consistency-mode origin_teacher --ema-teacher \
  --ema-decay 0.9998 --consistency-weight 0.10 \
  --target-consistency-weight 0.05 \
  --consistency-ramp-start 2000 --consistency-ramp-end 6000
```

Model-specific settings are listed in
[`training_config.json`](../reproducibility/training_config.json).
