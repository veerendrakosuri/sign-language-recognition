"""Check real detector asset/API compatibility without accessing a camera."""
import numpy as np
from signlang.landmarks import Extractor

with Extractor("models") as extractor:
    for timestamp in (0, 40):
        points, valid = extractor.frame(np.zeros((480, 640, 3), np.uint8), timestamp)
        assert points.shape == (75, 3) and valid.shape == (75,)
        assert np.isfinite(points).all()
        assert not valid.any(), "Blank image unexpectedly produced landmarks"
print("MediaPipe pose/hand assets load and process blank frames successfully")
