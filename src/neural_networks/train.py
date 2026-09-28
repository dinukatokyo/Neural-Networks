"""Reproducible training, untouched holdout evaluation, and saved demo assets.

Run from the project root: python -m neural_networks.train
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from neural_networks.model import ChurnNN, FEATURES

ROOT = Path(__file__).resolve().parents[2]


def clean_data(path):
    raw = pd.read_csv(path)
    required = FEATURES + ["Churn"]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = raw[required].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if not set(data.Churn.unique()).issubset({0, 1}) or data.Churn.nunique() != 2:
        raise ValueError("Churn must contain both binary labels 0 and 1.")
    # Identical observations must not appear on both sides of a random split.
    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    return data, {"raw_rows": len(raw), "invalid_rows_removed": len(raw) - before,
                  "duplicate_rows_removed": before - len(data), "clean_rows": len(data)}


def evaluate(y, probabilities, threshold=0.5):
    p = np.asarray(probabilities).ravel()
    predicted = (p >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    # Thin plot data only; metrics use every observation.
    indices = np.unique(np.linspace(0, len(fpr) - 1, min(180, len(fpr))).astype(int))
    return {"rows": len(y), "churn_rate": float(np.mean(y)), "threshold": threshold,
            "accuracy": float(accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "log_loss": float(log_loss(y, p)),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
            "roc": {"fpr": fpr[indices].tolist(), "tpr": tpr[indices].tolist()}}


def run_training(data_path=None, output_dir=None, epochs=20, batch_size=512, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    data_path = Path(data_path or ROOT / "data/train.csv")
    output_dir = Path(output_dir or ROOT / "artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    data, cleaning = clean_data(data_path)
    trainval, test = train_test_split(data, test_size=0.2, stratify=data.Churn, random_state=seed)
    train, val = train_test_split(trainval, test_size=0.2, stratify=trainval.Churn, random_state=seed)
    scaler = StandardScaler().fit(train[FEATURES])

    def tensors(frame):
        return (torch.tensor(scaler.transform(frame[FEATURES]), dtype=torch.float32),
                torch.tensor(frame.Churn.to_numpy(), dtype=torch.float32).reshape(-1, 1))

    x_train, y_train = tensors(train)
    x_val, y_val = tensors(val)
    x_test, y_test = tensors(test)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = ChurnNN()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            probabilities = model(inputs)
            loss = criterion(probabilities, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(inputs)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), y_val).item()
        history.append({"epoch": epoch, "training_loss": total_loss / len(train),
                        "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch:02d}/{epochs}: train={history[-1]['training_loss']:.4f} "
              f"validation={val_loss:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_p = model(x_test).numpy().ravel()
    baseline = LogisticRegression(max_iter=1000, random_state=seed).fit(x_train.numpy(), train.Churn)
    baseline_p = baseline.predict_proba(x_test.numpy())[:, 1]
    majority_p = np.full(len(test), float(train.Churn.mean()))
    metrics = {"neural_network": evaluate(test.Churn, test_p),
               "logistic_regression": evaluate(test.Churn, baseline_p),
               "majority_baseline": evaluate(test.Churn, majority_p)}
    external_path = data_path.parent / "test.csv"
    external_info = None
    if external_path.exists():
        external, external_info = clean_data(external_path)
        # Exclude feature vectors present in the original data from this secondary check.
        seen = pd.MultiIndex.from_frame(data[FEATURES])
        overlaps = pd.MultiIndex.from_frame(external[FEATURES]).isin(seen)
        external_info["overlapping_feature_rows_removed"] = int(overlaps.sum())
        external = external.loc[~overlaps].copy()
        if len(external) and external.Churn.nunique() == 2:
            xe, _ = tensors(external)
            with torch.no_grad():
                ep = model(xe).numpy().ravel()
            metrics["external_test_file"] = evaluate(external.Churn, ep)
            metrics["external_logistic_regression"] = evaluate(external.Churn, baseline.predict_proba(xe.numpy())[:, 1])
    metadata = {"features": FEATURES, "seed": seed, "epochs": epochs, "best_epoch": best_epoch,
                "batch_size": batch_size, "cleaning": cleaning,
                "splits": {"train": len(train), "validation": len(val), "test": len(test)},
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                "feature_min": train[FEATURES].min().to_dict(),
                "feature_max": train[FEATURES].max().to_dict(),
                "feature_median": train[FEATURES].median().to_dict(),
                "external_cleaning": external_info, "threshold": 0.5,
                "parameter_count": sum(p.numel() for p in model.parameters())}
    torch.save(model.state_dict(), output_dir / "model.pt")
    for name, value in [("metadata", metadata), ("metrics", metrics), ("history", history)]:
        (output_dir / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    predictions = test[FEATURES + ["Churn"]].copy()
    predictions["probability"] = test_p
    predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)
    # Real held-out examples, selected by score for a quick presentation.
    examples = predictions.sort_values("probability").iloc[[0, len(predictions)//2, -1]].copy()
    examples["example"] = ["Lower score", "Middle score", "Higher score"]
    examples.to_csv(output_dir / "examples.csv", index=False)
    print(json.dumps({k: {m: v for m, v in result.items() if m not in ['roc', 'confusion_matrix']}
                      for k, result in metrics.items()}, indent=2), flush=True)
    return model, metadata, history, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    run_training(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
