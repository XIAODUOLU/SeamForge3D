# SeamForge3D

SeamForge3D learns weld-seam fields from fused `XYZ + Normal` point clouds and
recovers ordered 3D robot trajectories. The current reference dataset contains
a base plate, a vertical web, and two open fillet seams. It supports offline NPZ
generation and dynamic raycast generation with sensor/pose/noise augmentation.

The network predicts seam heat, metric centerline offset, unoriented tangent,
part embedding, endpoint heat, and junction heat. A topology-aware graph stage
performs offset voting, tangent graph construction, open/closed/junction tracing,
gap bridging, B-spline/arc-length resampling, and optional dense-cloud surface
intersection refinement.

## Backbone policy

`randlanet` is the Intel/CPU default and wraps the complete official Open3D-ML
encoder-decoder. `utonia` wraps the official Utonia/PTv3 model and adds its
official PTv3 decoder blocks so dense per-point features are returned. Selectors
are configuration-only; no locally reimplemented substitute backbone is used.
See [docs/backbone_selection.md](docs/backbone_selection.md).

## Reproducible deployment

```bash
chmod +x deploy.sh
./deploy.sh --profile intel  # Intel Ultra / CPU: seamforge3d-intel
./deploy.sh --profile cuda   # NVIDIA training: seamforge3d-cuda + Utonia
```

The profiles are intentionally separate because Open3D 0.19 ML requires the
PyTorch 2.2 ABI, while the pinned Utonia stack uses PyTorch 2.5/CUDA 12.4.
The CUDA profile pins Utonia commit `da776a0bd3a48c6df83ac2ae0e27b26141cc7e31`.
The downloaded checkpoint is ignored by Git; its expected SHA-256 is documented
in `weights/README.md`. Use it only under the project's commercial authorization.

## Data, training, and inference

```bash
conda activate seamforge3d-intel
python tools/generate_dataset.py --config configs/overfit_two_meshes.yaml --visualizations 2 --overwrite
python tools/train.py --config configs/overfit_two_meshes.yaml --device cpu
python tools/train.py --config configs/base.yaml --device cuda --backbone utonia --checkpoint weights/utonia.pth
python tools/infer.py --config configs/overfit_two_meshes.yaml \
  --model outputs/overfit_balanced_scene/last.pt \
  --input outputs/overfit_dataset/train/scene_000002.npz \
  --output outputs/inference --device cpu --inference-votes 8 \
  --probability-threshold 0.95
pytest -q
```

`--device auto|cpu|cuda` controls runtime placement. For a strict memorization
audit, `tools/train.py` also accepts `--train-scenes 1 --train-start-index 2`.
Production probability thresholds must be calibrated on the target validation
set; the two-mesh audit uses `0.95` to inspect the learned centerline ridge.

Generated inspection artifacts include labeled PLY/PNG scenes, prediction and
GT trajectory overlays, CSV/JSON trajectories, metric JSON, checkpoints, and a
training plot made with:

```bash
python tools/plot_training.py \
  --metrics outputs/overfit_balanced_scene/metrics.jsonl \
  --output outputs/overfit_balanced_scene/training_curves.png
```
