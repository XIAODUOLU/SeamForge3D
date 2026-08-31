# Backbone selection for Intel deployment

## Decision

The default backbone is Open3D-ML RandLA-Net. Utonia remains an optional CUDA
training backbone selected with `model.backbone.name: utonia` or
`tools/train.py --backbone utonia`.

## Comparison

| Backbone | Dense encoder-decoder | CPU path | NVIDIA-only dependencies | Current role |
|---|---:|---:|---:|---|
| Open3D-ML RandLA-Net | Yes | Yes | No | Default Intel/CPU model |
| PointNeXt-S | Yes | Not production-ready in official stack | FPS/ball-query extensions commonly use CUDA | Future GPU benchmark |
| Utonia/PTv3 | Decoder added using official PTv3 module | No supported production path | spconv; FlashAttention recommended | CUDA pretrained benchmark |
| LitePT | Yes | PointROPE has fallback, full stack still uses spconv | spconv | Not selected |
| PointTransformerX | Paper describes portable CPU support | Claimed | No | Revisit after official code release |

RandLA-Net was selected because it is an established per-point segmentation
encoder-decoder, uses random sampling rather than expensive farthest-point
sampling, supports large point sets, and is shipped by Open3D-ML with a CPU
runtime. The adapter does not reimplement or reduce the backbone. It prepares
the documented hierarchy/neighbor inputs and consumes the complete model's
dense output as the shared feature field for all SeamForge heads.

Open3D's implementation uses BatchNorm momentum 0.01. The Intel profile uses
per-scan BatchNorm statistics at inference (Dropout remains disabled), which is
stable for the thousands of points in one scan and avoids a dependency on stale
running statistics when production scan distributions change. This behavior is
configurable with `randlanet.use_batch_stats_in_eval`.
Inference also averages multiple deterministic random-sampling votes, matching
RandLA-Net's coverage strategy; `randlanet.inference_votes` controls the
accuracy/latency trade-off.

PointNeXt-S remains a valuable accuracy comparison. The official OpenPoints
implementation is built around FPS, ball query, and interpolation operators;
available packaged alternatives explicitly compile CUDA operators. It should
not be the default until an audited Intel implementation/export is available.

## Environment compatibility

Open3D 0.19 ML bindings require PyTorch 2.2.x. Utonia uses PyTorch 2.5/CUDA
12.4. `deploy.sh` therefore creates separate reproducible profiles:

```bash
./deploy.sh --profile intel
./deploy.sh --profile cuda
```

This separation prevents binary ABI conflicts from being hidden in a single
environment.
