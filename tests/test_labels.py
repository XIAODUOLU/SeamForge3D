import numpy as np

from seamforge3d.geometry.assets import create_seed_assembly
from seamforge3d.labels import build_seam_labels


def test_exact_seam_points_have_zero_distance_and_correct_tangent():
    assembly = create_seed_assembly(41)
    points = np.concatenate([seam.points for seam in assembly.seams])
    labels = build_seam_labels(points, assembly.seams, sigma=0.004, supervision_radius=0.012)
    assert np.max(labels.distance) < 1e-7
    assert np.min(labels.heat) > 0.999
    assert np.max(np.linalg.norm(labels.offset, axis=1)) < 1e-7
    assert np.allclose(np.abs(labels.tangent[:, 0]), 1.0)


def test_off_seam_label_geometry():
    assembly = create_seed_assembly(101)
    points = np.array([[0.0, -0.004, 0.003], [0.0, 0.030, 0.020]], dtype=np.float32)
    labels = build_seam_labels(points, assembly.seams, sigma=0.004, supervision_radius=0.012)
    assert np.isclose(labels.distance[0], 0.003, atol=1e-6)
    assert np.allclose(labels.offset[0], [0, 0, -0.003], atol=1e-6)
    assert labels.heat[0] > 0
    assert labels.heat[1] == 0
    assert np.all(labels.offset[1] == 0)

