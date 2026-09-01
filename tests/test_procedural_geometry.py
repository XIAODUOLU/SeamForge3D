import numpy as np

from seamforge3d.geometry.assets import PROCEDURAL_FAMILIES, SeamCurve, create_procedural_assembly
from seamforge3d.geometry.visibility import filter_visible_seams
from seamforge3d.labels import build_seam_labels, encode_seams


def test_all_procedural_families_are_valid_and_have_seams():
    for family_index, family in enumerate(PROCEDURAL_FAMILIES):
        assembly = create_procedural_assembly(np.random.default_rng(100 + family_index), families=(family,))
        assert assembly.family == family
        assert len(assembly.meshes) == 2
        assert len(assembly.seams) >= 1
        for mesh in assembly.meshes:
            assert np.isfinite(mesh.vertices).all()
            assert mesh.triangles.min() >= 0
            assert mesh.triangles.max() < len(mesh.vertices)
        for seam in assembly.seams:
            assert len(seam.points) >= 2
            assert np.linalg.norm(np.diff(seam.points, axis=0), axis=1).sum() > 0.01


def test_visibility_keeps_only_two_part_supported_run():
    x = np.linspace(-0.1, 0.1, 101, dtype=np.float32)
    seam_a = SeamCurve(np.column_stack((x, np.zeros_like(x), np.zeros_like(x))), "open", (0, 1), 0)
    seam_b = SeamCurve(np.column_stack((x, np.full_like(x, 0.04), np.zeros_like(x))), "open", (0, 1), 1)
    supported_x = x[x <= 0.0]
    part_a = np.column_stack((supported_x, np.zeros_like(supported_x), np.full_like(supported_x, 0.001)))
    part_b = np.column_stack((supported_x, np.full_like(supported_x, 0.001), np.zeros_like(supported_x)))
    raw_coord = np.vstack((part_a, part_b)).astype(np.float32)
    instance = np.r_[np.zeros(len(part_a), dtype=np.int32), np.ones(len(part_b), dtype=np.int32)]
    visible, fraction = filter_visible_seams(
        (seam_a, seam_b), raw_coord, instance, support_radius=0.003,
        min_visible_length=0.02, max_gap_points=0,
    )
    assert len(visible) == 1
    assert visible[0].topology == "open"
    assert 0.45 <= fraction[0] <= 0.55
    assert fraction[1] == 0.0
    assert visible[0].points[:, 0].max() <= 0.003


def test_empty_visible_truth_has_stable_schema_and_negative_labels():
    encoded = encode_seams(())
    assert encoded["seam_points"].shape == (0, 3)
    assert encoded["seam_part_pairs"].shape == (0, 2)
    points = np.zeros((8, 3), dtype=np.float32)
    labels = build_seam_labels(points, (), sigma=0.004, supervision_radius=0.012)
    assert np.all(labels.heat == 0)
    assert np.all(labels.nearest_trajectory == -1)
