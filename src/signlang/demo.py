import time
from collections import deque

import cv2
import numpy as np
import torch

from .features import prepare, StablePrediction
from .landmarks import Extractor
from .training import load_gru


def webcam(args):
    if args.predict_every < 1 or args.window_seconds <= 0:
        raise ValueError("predict-every and window-seconds must be positive")
    model, pack = load_gru(args.checkpoint, args.device)
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    stable = StablePrediction(args.threshold, args.repeats)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Cannot open camera. Close other camera apps or try --camera 1.")
    buffer = deque()
    started, previous = time.perf_counter(), time.perf_counter()
    step, fps, shown, confidence = 0, 0.0, None, 0.0
    try:
        with Extractor(args.models) as extractor, torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.perf_counter()
                p, m = extractor.frame(frame, (now - started) * 1000)
                usable = bool(m[11] and m[12] and m[33:].any())
                if usable:
                    buffer.append((now, p, m))
                    while buffer and now - buffer[0][0] > args.window_seconds:
                        buffer.popleft()
                else:
                    buffer.clear()
                    stable.reset()
                    shown = None
                ready = len(buffer) >= 2 and buffer[-1][0] - buffer[0][0] >= args.window_seconds * .85
                if ready and step % args.predict_every == 0:
                    points = np.stack([v[1] for v in buffer])
                    masks = np.stack([v[2] for v in buffer])
                    try:
                        x = prepare(points, masks, pack["frames"])
                        prob = torch.softmax(model(torch.from_numpy(x[None]).to(args.device)), 1)[0]
                        confidence, index = prob.max(0)
                        confidence = float(confidence)
                        shown = stable.update(pack["labels"][int(index)], confidence)
                    except ValueError:
                        shown = None
                        stable.reset()
                if not ready:
                    shown = None
                # Mirror only the preview; model sees the same orientation as offline extraction.
                preview = cv2.flip(frame, 1)
                for point in p[m]:
                    cv2.circle(preview, (int((1-point[0])*preview.shape[1]), int(point[1]*preview.shape[0])), 2, (0,220,120), -1)
                after = time.perf_counter()
                instantaneous = 1 / max(after - previous, 1e-6)
                fps = instantaneous if step == 0 else 0.9 * fps + 0.1 * instantaneous
                previous = after
                caption = f"{shown} ({confidence:.0%})" if shown else ("Hold upper body and hands in view" if not usable else "Collecting / uncertain")
                cv2.putText(preview, caption, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
                cv2.putText(preview, f"FPS {fps:.1f} | isolated ASL | Q quit", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
                cv2.imshow("Sign language recognition", preview)
                step += 1
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
