import csv
import json
import platform
import random
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .data import digest, load_split, require_training_classes, write_json
from .features import baseline_summary, prepare
from .model import SignGRU


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_directory(path, args):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{path} is not empty. Choose a fresh --output for each run.")
    path.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(manifest_sha256=digest(args.manifest), python=platform.python_version(),
                  torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__)
    write_json(path / "config.json", config)
    return path


def get_labels(args):
    return json.loads(Path(args.labels).read_text(encoding="utf-8"))


def train_baseline(args):
    from threadpoolctl import threadpool_limits
    labels = get_labels(args)
    _, y, ids, raw = load_split(args.manifest, args.cache, "train", args.frames)
    require_training_classes(y, labels)
    _, vy, vids, valid_raw = load_split(args.manifest, args.cache, "val", args.frames)
    x = np.stack([baseline_summary(p, m, args.frames, args.summary) for p, m in raw])
    vx = np.stack([baseline_summary(p, m, args.frames, args.summary) for p, m in valid_raw])
    out = run_directory(args.output, args)
    best, best_score, scores = None, -1, []
    # Fixed native thread count avoids borderline predictions changing with BLAS scheduling.
    with threadpool_limits(limits=1):
        for c in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000,
                                   class_weight="balanced", random_state=args.seed))
            model.fit(x, y)
            score = float(model.score(vx, vy))
            scores.append(dict(C=c, summary=args.summary, validation_top1=score))
            if score > best_score:
                best, best_score = model, score
    joblib.dump(dict(model=best, labels=labels, frames=args.frames, summary=args.summary),
                out / "baseline.joblib")
    write_json(out / "validation.json", scores)
    write_json(out / "sample_ids.json", dict(train=ids, val=vids))
    print(f"Baseline validation top-1: {best_score:.4f}")


class Sequences(Dataset):
    def __init__(self, raw, y, frames, seed, mirror_probability):
        self.raw, self.y, self.frames = raw, y, frames
        self.rng = np.random.default_rng(seed)
        self.mirror_probability = mirror_probability

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        p, m = self.raw[index]
        return prepare(p, m, self.frames, self.rng, self.mirror_probability), int(self.y[index])


def train_gru(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size and patience must be positive")
    if not 0 <= args.mirror_probability <= 1:
        raise ValueError("mirror-probability must be in [0,1]")
    seed_all(args.seed)
    labels = get_labels(args)
    _, y, ids, raw = load_split(args.manifest, args.cache, "train", args.frames)
    require_training_classes(y, labels)
    vx, vy, vids, _ = load_split(args.manifest, args.cache, "val", args.frames)
    out = run_directory(args.output, args)
    device = torch.device(args.device)
    spec = dict(classes=len(labels), hidden=args.hidden, layers=2, dropout=0.3)
    model = SignGRU(**spec).to(device)
    counts = np.bincount(y, minlength=len(labels))
    weights = len(y) / (len(labels) * counts)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train = DataLoader(Sequences(raw, y, args.frames, args.seed, args.mirror_probability),
                       batch_size=args.batch_size, shuffle=True, num_workers=0,
                       generator=torch.Generator().manual_seed(args.seed))
    valid = DataLoader(list(zip(torch.from_numpy(vx), torch.from_numpy(vy))), batch_size=args.batch_size)
    best, stale, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for bx, by in train:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(by)
        model.eval()
        correct = 0
        with torch.inference_mode():
            for bx, by in valid:
                correct += int((model(bx.to(device)).argmax(1).cpu() == by).sum())
        score = correct / len(vy)
        row = dict(epoch=epoch, train_loss=total_loss / len(y), validation_top1=score)
        history.append(row)
        print(row)
        if score > best:
            best, stale = score, 0
            torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                            model=spec, labels=labels, frames=args.frames,
                            epoch=epoch, validation_top1=score), out / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_json(out / "sample_ids.json", dict(train=ids, val=vids))


def load_gru(checkpoint, device="cpu"):
    pack = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = SignGRU(**pack["model"])
    model.load_state_dict(pack["state_dict"])
    return model.to(device).eval(), pack
