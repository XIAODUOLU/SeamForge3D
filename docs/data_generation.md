# Procedural workpieces and viewpoint-aware truth

## Families

`assembly_generator: procedural` creates two physical workpieces: a randomized
base plate and one randomized contacting profile. Supported families are:

- `t_joint`: thin vertical web, two open seams;
- `triangular_prism`: extruded triangular profile, two open seams;
- `cylinder`: upright cylinder, one closed circular seam;
- `cone`: upright cone, one closed circular seam;
- `i_beam`: three-section I profile, two open flange seams;
- `l_angle`: angle profile, two open footprint seams;
- `box`: upright rectangular workpiece, two open seams.

Dimensions, profile yaw, global rigid pose, camera placement, sensor noise, and
dropout are sampled independently. `balanced_family_sampling: true` deterministically
cycles the selected family from the scene seed so large offline and dynamic
datasets do not accidentally under-sample a family.

## Viewpoint modes

- `view_mode: multi`: fuse `num_views` randomly placed cameras.
- `view_mode: single`: exactly one camera.
- `view_mode: mixed`: use one camera with `single_view_probability`, otherwise
  draw `num_views` cameras.

Camera ranges are controlled by `azimuth_deg`, `elevation_deg`,
`camera_distance`, and `fov_deg`. To reproduce one physical viewpoint, use
degenerate ranges such as `[45, 45]`, set `assembly_rotation_mode: none`, and
disable pose noise. See `configs/fixed_view_example.yaml`.

## Excluding unobserved CAD truth

When `filter_invisible_seams: true`, full CAD trajectories are not used directly
as supervision. Each sampled seam point must have a nearby observed raw point
from **both** members of its `part_pair`. This uses the post-raycast cloud, so it
incorporates occlusion, field-of-view clipping, grazing-angle dropout, random
point dropout, and patch dropout.

Short unsupported gaps can be closed with `visibility_max_gap_points`. Supported
runs shorter than `min_visible_seam_length` are discarded. A partially observed
closed loop becomes one or more open trajectories. A scene with no supported
trajectory is retained as a hard negative with a stable empty-GT schema.

Every NPZ retains both representations:

- `full_seam_*`: complete CAD trajectory and `full_seam_visible_fraction`;
- `seam_*`: only view-supported trajectories used by labels and metrics.

Generated label PNGs plot supervised GT in black and complete CAD trajectories
as thin grey dashed curves, making removal decisions visually auditable.

## Offline and dynamic use

Generate the default 512/64/64 train/val/test split:

```bash
CONFIG=configs/large_procedural.yaml bash scripts/generate_data.sh
```

For training-time generation set `data.dynamic: true`. The dataset changes its
seed every epoch while keeping validation/test fixed. Offline generation is
recommended initially because it permits label inspection and exact experiment
reproduction. The default large split can consume several gigabytes because raw
point clouds are retained for final dense refinement.
