"""Shared offline/live preprocessing. Missing observations stay exactly zero."""
import numpy as np

POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)] + list(zip(range(11, 33, 2), range(12, 33, 2)))


def normalize(points, valid):
    points = np.asarray(points, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if points.ndim != 3 or points.shape[1:] != (75, 3) or len(points) == 0:
        raise ValueError("Expected non-empty points with shape (frames, 75, 3)")
    if valid.shape != points.shape[:2] or not np.isfinite(points).all():
        raise ValueError("Invalid mask or non-finite landmark values")
    width = np.linalg.norm(points[:, 11, :2] - points[:, 12, :2], axis=1)
    good = valid[:, 11] & valid[:, 12] & (width > 1e-4)
    if not good.any():
        raise ValueError("No frame has two usable shoulders; keep upper body in view")
    # Interpolate only the body reference, never invent missing hand observations.
    t = np.arange(len(points))
    center = (points[:, 11] + points[:, 12]) / 2
    center = np.stack([np.interp(t, t[good], center[good, d]) for d in range(3)], axis=1)
    scale = np.median(width[good])
    out = points.copy()
    out[:, :, :2] -= center[:, None, :2]
    out[:, :33, 2] -= center[:, None, 2]
    # Hand z is wrist-relative; pose z has a different origin. Do not mix them.
    out /= scale
    out[~valid] = 0
    return out.astype(np.float32)


def resample(sequence, frames=32):
    if frames < 2 or len(sequence) == 0:
        raise ValueError("Need a non-empty sequence and at least two output frames")
    idx = np.linspace(0, len(sequence) - 1, frames).round().astype(int)
    return sequence[idx]


def mirror(points, valid):
    out, mask = points.copy(), valid.copy()
    out[..., 0] *= -1  # Coordinates have already been centered.
    for left, right in POSE_PAIRS:
        out[:, [left, right]] = out[:, [right, left]]
        mask[:, [left, right]] = mask[:, [right, left]]
    out[:, 33:54], out[:, 54:75] = out[:, 54:75].copy(), out[:, 33:54].copy()
    mask[:, 33:54], mask[:, 54:75] = mask[:, 54:75].copy(), mask[:, 33:54].copy()
    out[~mask] = 0
    return out, mask


def prepare(points, valid, frames=32, rng=None, mirror_probability=0.0):
    seq = normalize(points, valid)
    mask = valid.copy()
    if rng is not None:
        angle = rng.uniform(-0.08, 0.08)
        c, s = np.cos(angle), np.sin(angle)
        seq[..., :2] = seq[..., :2] @ np.array([[c, s], [-s, c]], dtype=np.float32)
        seq *= rng.uniform(0.9, 1.1)
        if rng.random() < mirror_probability:
            seq, mask = mirror(seq, mask)
        seq[~mask] = 0
    return resample(seq, frames).reshape(frames, 225)


class StablePrediction:
    def __init__(self, threshold=0.75, repeats=3):
        if not 0 <= threshold <= 1 or repeats < 1:
            raise ValueError("Threshold must be in [0,1] and repeats >= 1")
        self.threshold, self.repeats = threshold, repeats
        self.reset()

    def reset(self):
        self.label, self.count = None, 0

    def update(self, label, probability):
        if probability < self.threshold:
            self.reset()
            return None
        self.count = self.count + 1 if label == self.label else 1
        self.label = label
        return label if self.count >= self.repeats else None
