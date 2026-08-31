import numpy as np

from seamforge3d.postprocess.reconstruct import ReconstructionConfig, SeamGraphReconstructor


def test_two_open_seams_reconstruct_as_two_components():
    rng = np.random.default_rng(2)
    x = np.linspace(-0.08, 0.08, 161)
    first = np.column_stack((x, np.full_like(x, -0.008), np.zeros_like(x)))
    second = np.column_stack((x, np.full_like(x, 0.008), np.zeros_like(x)))
    coord = np.concatenate((first, second)).astype(np.float32)
    coord += rng.normal(0, 0.00015, coord.shape)
    prediction = {
        "seam_logit": np.full(len(coord), 8.0, dtype=np.float32),
        "seam_offset": np.zeros_like(coord),
        "seam_tangent": np.tile([1.0, 0.0, 0.0], (len(coord), 1)).astype(np.float32),
        "part_embed": np.r_[np.tile([[0.0, 0.0]], (len(first), 1)), np.tile([[2.0, 2.0]], (len(second), 1))].astype(np.float32),
    }
    config = ReconstructionConfig(graph_radius=0.004, bridge_gap=0.006, merge_voxel=0.001, min_component_length=0.05)
    trajectories = SeamGraphReconstructor(config)(coord, prediction)
    assert len(trajectories) == 2
    assert all(trajectory.topology == "open" for trajectory in trajectories)
    assert all(len(trajectory.points) > 50 for trajectory in trajectories)

