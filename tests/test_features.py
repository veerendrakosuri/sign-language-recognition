import numpy as np
import pytest

from signlang.features import baseline_summary, normalize, mirror, prepare, StablePrediction


def sample():
    p = np.zeros((8, 75, 3), dtype=np.float32)
    m = np.zeros((8, 75), dtype=bool)
    p[:, 11] = [.3, .4, .1]
    p[:, 12] = [.7, .4, .1]
    p[:, 33:54] = [.2, .3, .01]
    m[:, [11, 12]] = True
    m[:, 33:54] = True
    return p, m


def test_missing_hands_stay_zero_after_centering_and_augmentation():
    p, m = sample()
    n = normalize(p, m)
    assert np.all(n[:, 54:] == 0)
    assert np.allclose(n[:, [11, 12], :2].mean(axis=1), 0)
    result = prepare(p, m, rng=np.random.default_rng(42))
    assert result.shape == (32, 225)
    assert np.all(result.reshape(32, 75, 3)[:, 54:] == 0)


def test_missing_shoulders_interpolated_and_all_missing_rejected():
    p, m = sample()
    m[3:5, [11, 12]] = False
    p[3:5, [11, 12]] = 0
    assert np.isfinite(normalize(p, m)).all()
    m[:, [11, 12]] = False
    with pytest.raises(ValueError, match="shoulders"):
        normalize(p, m)


def test_mirror_swaps_anatomical_sides_and_is_involution():
    p, m = sample()
    p = normalize(p, m)
    flipped, fm = mirror(p, m)
    assert not fm[:, 33:54].any()
    assert fm[:, 54:].all()
    assert np.allclose(flipped[:, 54:, 0], -p[:, 33:54, 0])
    restored, rm = mirror(flipped, fm)
    assert np.allclose(restored, p)
    assert np.array_equal(rm, m)


def test_stability_clears_on_low_confidence_and_changed_sign():
    s = StablePrediction(.8, 2)
    assert s.update("book", .9) is None
    assert s.update("book", .9) == "book"
    assert s.update("book", .2) is None
    assert s.update("book", .9) is None
    assert s.update("drink", .9) is None


def test_image_translation_and_scale_invariance():
    p, m = sample()
    moved = p.copy()
    moved[..., :2] = moved[..., :2] * 1.4 + .2
    moved[..., 2] *= 1.4
    moved[~m] = 0
    assert np.allclose(normalize(p, m), normalize(moved, m), atol=1e-6)


def test_observed_mean_excludes_missing_landmark_frames():
    p, m = sample()
    m[:4, 33:54] = False
    p[:4, 33:54] = 0
    zero_mean = baseline_summary(p, m, frames=8, summary="zero_mean")
    observed = baseline_summary(p, m, frames=8, summary="observed_mean")
    assert np.allclose(observed[99:162], zero_mean[99:162] * 2)
    assert np.all(observed[162:] == 0)
    with pytest.raises(ValueError, match="summary"):
        baseline_summary(p, m, summary="unknown")
