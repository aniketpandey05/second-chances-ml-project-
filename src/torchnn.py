"""
A neural network in PyTorch, with explicit backpropagation.

WHY THIS EXISTS, given sklearn's MLPClassifier is already in the comparison
-------------------------------------------------------------------------
Two things sklearn genuinely cannot do, both of which matter on this dataset:

1. CLASS-WEIGHTED LOSS. `MLPClassifier` has no `class_weight` parameter - you
   can check with `inspect.signature`. That is exactly why it posts the best
   accuracy in the study (0.628) and nearly the worst balanced accuracy (0.373):
   with Adoption at 51.6% of records it has no mechanism to resist predicting
   the majority class. `nn.CrossEntropyLoss(weight=...)` fixes that directly,
   and the improvement in balanced accuracy is measurable.

2. LEARNED EMBEDDINGS. One-hot encoding treats the 50 breed groups as 50
   mutually orthogonal directions - Labrador is exactly as far from Golden
   Retriever as it is from Siamese. An embedding layer learns a dense vector per
   category from the outcome signal, so breeds that behave alike end up near
   each other. This is the standard treatment of categorical inputs in deep
   learning and there is no sklearn equivalent.

The training loop is written out rather than hidden behind `.fit()`, so the
forward pass, the loss, `loss.backward()` and the optimiser step are all visible
- which is the part of the syllabus this is meant to demonstrate.

CPU is sufficient: the whole thing trains in under two minutes.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data import FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")
ARRAYS = os.path.join(ROOT, "results", "arrays")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class ShelterNet(nn.Module):
    """
    One embedding per categorical field, concatenated, then a small MLP.

    Embedding width follows the usual heuristic min(16, (cardinality+1)//2):
    a field with two values gets one dimension, breed_group with 50 gets 16.
    """

    def __init__(self, cardinalities: list[int], n_classes: int,
                 hidden=(128, 64), dropout=0.3):
        super().__init__()
        dims = [max(1, min(16, (c + 1) // 2)) for c in cardinalities]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, d) for c, d in zip(cardinalities, dims)])
        self.emb_dim = sum(dims)

        layers, prev = [], self.emb_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):                       # x: (batch, n_fields) int64
        parts = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.mlp(torch.cat(parts, dim=1))


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------
def encode_ordinal(df: pd.DataFrame):
    """Each categorical field -> integer codes. Returns (X, y, cardinalities, classes)."""
    cols, cards = [], []
    for f in FEATURES:
        cat = pd.Categorical(df[f].astype(str))
        cols.append(cat.codes.astype(np.int64))
        cards.append(len(cat.categories))
    X = np.column_stack(cols)
    classes = sorted(df["outcome"].unique())
    lookup = {c: i for i, c in enumerate(classes)}
    y = df["outcome"].map(lookup).to_numpy(dtype=np.int64)
    return X, y, cards, classes


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_model(X_tr, y_tr, X_va, y_va, cards, n_classes, class_weights,
                epochs=40, batch_size=1024, lr=1e-3, patience=6, seed=42,
                verbose=True):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ShelterNet(cards, n_classes).to(device)
    # THE point of using torch here: a per-class weight on the loss.
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=batch_size, shuffle=True)
    Xv = torch.tensor(X_va).to(device)
    yv = torch.tensor(y_va).to(device)

    history, best, best_state, stale = [], -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimiser.zero_grad()          # 1. clear last step's gradients
            logits = model(xb)             # 2. forward pass
            loss = criterion(logits, yb)   # 3. weighted cross-entropy
            loss.backward()                # 4. BACKPROPAGATION
            optimiser.step()               # 5. gradient descent step

            running += loss.item() * len(xb)
        train_loss = running / len(X_tr)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = criterion(val_logits, yv).item()
            val_pred = val_logits.argmax(1).cpu().numpy()
        val_bal = balanced_accuracy_score(y_va, val_pred)
        val_f1 = f1_score(y_va, val_pred, average="macro", zero_division=0)
        history.append({"epoch": epoch + 1, "train_loss": round(train_loss, 5),
                        "val_loss": round(val_loss, 5),
                        "val_balanced_acc": round(float(val_bal), 5),
                        "val_macro_f1": round(float(val_f1), 5)})

        if val_f1 > best + 1e-4:
            best, stale = val_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"      early stop at epoch {epoch+1} "
                          f"(best val macro-F1 {best:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), device


def predict(model, X, device, batch_size=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size]).to(device)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# evaluation, matching supervised.py's schema so the rows merge
# ---------------------------------------------------------------------------
def _row(name, protocol, y_true, y_pred, n_train, fit_s, pred_s):
    return {
        "protocol": protocol, "model": name,
        "train_rows": n_train, "test_rows": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 5),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 5),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 5),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 5),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 5),
        "fit_seconds": round(fit_s, 2), "predict_seconds": round(pred_s, 2),
    }


def _evaluate(X_tr, y_tr, X_te, y_te, cards, classes, protocol, weighted,
              seed, verbose):
    name = ("Torch NN (embeddings, weighted)" if weighted
            else "Torch NN (embeddings, unweighted)")
    if verbose:
        print(f"[torch:{protocol}] {name:<34} train {len(X_tr):>7,} ...")

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=seed, stratify=y_tr)

    if weighted:
        counts = np.bincount(y_fit, minlength=len(classes)).astype(float)
        counts[counts == 0] = 1.0
        w = len(y_fit) / (len(classes) * counts)      # inverse frequency
    else:
        w = np.ones(len(classes))

    t0 = time.perf_counter()
    model, history, device = train_model(X_fit, y_fit, X_val, y_val, cards,
                                         len(classes), w, seed=seed,
                                         verbose=verbose)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = predict(model, X_te, device)
    pred_s = time.perf_counter() - t0

    row = _row(name, protocol, y_te, y_pred, len(X_fit), fit_s, pred_s)
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_te, y_pred, labels=labels,
                                                 zero_division=0)
    per_class = [{"protocol": protocol, "model": name, "outcome": classes[i],
                  "support": int(s[i]), "precision": round(float(p[i]), 5),
                  "recall": round(float(r[i]), 5), "f1": round(float(f[i]), 5)}
                 for i in labels]

    slug = "torch_nn_weighted" if weighted else "torch_nn_unweighted"
    np.save(os.path.join(ARRAYS, f"cm_{protocol}_{slug}.npy"),
            confusion_matrix(y_te, y_pred, labels=labels))

    if verbose:
        print(f"      acc={row['accuracy']:.4f} bal_acc={row['balanced_accuracy']:.4f} "
              f"macroF1={row['macro_f1']:.4f} mcc={row['mcc']:.4f} ({fit_s:.1f}s)")
    return row, per_class, history, slug


def run(df: pd.DataFrame, seed=42, temporal_cutoff=2021, verbose=True):
    """Both protocols, weighted and unweighted, so the weighting is isolated."""
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(ARRAYS, exist_ok=True)

    X, y, cards, classes = encode_ordinal(df)
    years = df["year"].to_numpy()
    rows, per_class, curves = [], [], []

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    for weighted in (True, False):
        r, pc, hist, slug = _evaluate(X_tr, y_tr, X_te, y_te, cards, classes,
                                      "random", weighted, seed, verbose)
        rows.append(r); per_class.extend(pc)
        curves.append(hist.assign(protocol="random", variant=slug))

    tr, te = years <= temporal_cutoff, years > temporal_cutoff
    if tr.sum() > 50 and te.sum() > 50:
        for weighted in (True, False):
            r, pc, hist, slug = _evaluate(X[tr], y[tr], X[te], y[te], cards,
                                          classes, "temporal", weighted, seed, verbose)
            rows.append(r); per_class.extend(pc)
            curves.append(hist.assign(protocol="temporal", variant=slug))

    summary = pd.DataFrame(rows)
    pc_df = pd.DataFrame(per_class)
    curve_df = pd.concat(curves, ignore_index=True)

    summary.to_csv(os.path.join(TABLES, "torch_summary.csv"), index=False)
    pc_df.to_csv(os.path.join(TABLES, "torch_per_class.csv"), index=False)
    curve_df.to_csv(os.path.join(TABLES, "torch_training_curves.csv"), index=False)

    names_path = os.path.join(ARRAYS, "model_names.json")
    names = {}
    if os.path.exists(names_path):
        with open(names_path) as fh:
            names = json.load(fh)
    names.update({"torch_nn_weighted": "Torch NN (embeddings, weighted)",
                  "torch_nn_unweighted": "Torch NN (embeddings, unweighted)"})
    with open(names_path, "w") as fh:
        json.dump(names, fh, indent=2)

    return summary, pc_df, curve_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load, stratified_sample

    df = stratified_sample(load(verbose=False), 60_000)
    s, pc, curves = run(df)
    pd.set_option("display.width", 220)
    print()
    print(s.to_string(index=False))
