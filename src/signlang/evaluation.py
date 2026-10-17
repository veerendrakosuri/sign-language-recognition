import csv
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .data import digest, load_split, read_manifest, write_json
from .features import baseline_summary
from .training import load_gru


def metrics(y, probabilities, labels):
    if len(y) == 0 or probabilities.shape != (len(y), len(labels)):
        raise ValueError("Expected one probability row per sample and one column per class")
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite predictions")
    order = np.argsort(-probabilities, axis=1, kind="stable")
    predicted = order[:, 0]
    k = min(5, len(labels))
    return dict(samples=len(y), classes=len(labels), top1=float(np.mean(predicted == y)),
                top5=float(np.mean(np.any(order[:, :k] == y[:, None], axis=1))), top_k=k,
                macro_f1=float(f1_score(y, predicted, labels=list(range(len(labels))),
                                       average="macro", zero_division=0))), predicted


def evaluate(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a fresh evaluation --output directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.kind == "gru":
        model, pack = load_gru(args.checkpoint, args.device)
    else:
        # joblib uses pickle: use only checkpoints you created or trust.
        pack = joblib.load(args.checkpoint)
        model = pack["model"]
    labels = pack["labels"]
    current_labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    if labels != current_labels:
        raise ValueError("Checkpoint labels differ from manifest labels")
    x, y, ids, raw = load_split(args.manifest, args.cache, args.split, pack["frames"])
    if args.kind == "gru":
        with torch.inference_mode():
            probabilities = np.concatenate([torch.softmax(model(torch.from_numpy(x[i:i+64]).to(args.device)), 1)
                                            .cpu().numpy() for i in range(0, len(x), 64)])
    else:
        probabilities = np.zeros((len(x), len(labels)), np.float32)
        summary = pack.get("summary", "zero_mean")
        features = np.stack([baseline_summary(p, m, pack["frames"], summary) for p, m in raw])
        probabilities[:, model.classes_.astype(int)] = model.predict_proba(features)
    result, predicted = metrics(y, probabilities, labels)
    result.update(split=args.split, model=args.kind, checkpoint_sha256=digest(args.checkpoint),
                  manifest_sha256=digest(args.manifest), evaluated_video_ids=ids)
    expected = [r for r in read_manifest(args.manifest) if r["split"] == args.split]
    result["expected_samples"] = len(expected)
    result["coverage"] = len(y) / len(expected)
    result["missing_classes"] = [labels[i] for i in range(len(labels)) if i not in set(y.tolist())]
    write_json(out / "metrics.json", result)
    write_json(out / "classification_report.json", classification_report(
        y, predicted, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0))
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "true_gloss", "predicted_gloss", "confidence"])
        for i, vid in enumerate(ids):
            writer.writerow([vid, labels[y[i]], labels[predicted[i]], float(probabilities[i].max())])
    matrix = confusion_matrix(y, predicted, labels=list(range(len(labels))))
    with (out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/predicted"] + labels)
        writer.writerows([[label] + row.tolist() for label, row in zip(labels, matrix)])
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * .18), max(6, len(labels) * .18)))
    display = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(display, vmin=0, vmax=1, cmap="Blues")
    ax.set(xlabel="Predicted sign", ylabel="True sign", title=f"{args.kind.upper()} - {args.split} (row normalized)")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    summary = (f"# {args.kind.upper()} evaluation\n\nSplit: {args.split}. "
               f"Evaluated {len(y)}/{len(expected)} official instances.\n\n"
               "| Top-1 | Top-k | k | Macro F1 |\n|---:|---:|---:|---:|\n"
               f"| {result['top1']:.4f} | {result['top5']:.4f} | {result['top_k']} | {result['macro_f1']:.4f} |\n\n"
               "Missing videos and classes change comparability; inspect metrics.json.\n")
    (out / "results.md").write_text(summary, encoding="utf-8")
    print(summary)
