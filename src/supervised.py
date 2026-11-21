"""
The five taught classifiers predicting shelter outcome from intake attributes.

Two protocols, and the difference between them is a result:

  RANDOM      stratified train/test split. The usual thing, and optimistic.
  TEMPORAL    train on <= 2021, test on >= 2022. Shelter policy changes over
              time (Austin's intake collapsed from 19,775 in 2019 to 9,778 in
              2020), so a random split lets the model learn period-specific
              policy and quietly take credit for it.

Every model sees the SAME test set. What differs is how many training rows each
is allowed, because the algorithms have very different complexity:

    Decision Tree / Naive Bayes / MLP   near-linear       -> full training set
    Linear SVM                          ~O(n d)           -> capped
    KNN                                 O(n_train n_test) -> capped
    RBF SVM                             O(n^2)-O(n^3)     -> capped hard

That cap table is reported with the results. It is the reason an RBF SVM cannot
be trained on every record while a decision tree can.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_text

from data import FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")
ARRAYS = os.path.join(ROOT, "results", "arrays")


def slugify(name: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in name.lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def encode(df: pd.DataFrame):
    """One-hot the categorical features. Returns (X, y, encoder, feature_names)."""
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    X = enc.fit_transform(df[FEATURES])
    y = df["outcome"].to_numpy()
    return X, y, enc, list(enc.get_feature_names_out(FEATURES))


def model_zoo(seed=42):
    """(name, estimator, max_train_rows). None means the whole training set."""
    return [
        ("Decision Tree",
         DecisionTreeClassifier(criterion="entropy", max_depth=12,
                                min_samples_leaf=20, class_weight="balanced",
                                random_state=seed),
         None),
        ("Naive Bayes (Bernoulli)",
         BernoulliNB(),
         None),
        ("KNN (k=15)",
         KNeighborsClassifier(n_neighbors=15, algorithm="brute", n_jobs=-1),
         20_000),
        ("SVM (linear)",
         LinearSVC(C=0.5, dual="auto", max_iter=3000, class_weight="balanced",
                   random_state=seed),
         60_000),
        ("SVM (RBF)",
         SVC(C=2.0, gamma="scale", kernel="rbf", cache_size=1000,
             class_weight="balanced", random_state=seed),
         10_000),
        ("Neural Net (MLP, backprop)",
         MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                       batch_size=512, learning_rate_init=1e-3, max_iter=60,
                       early_stopping=True, n_iter_no_change=6, random_state=seed),
         None),
    ]


def _cap(X, y, cap, seed):
    if cap is None or len(X) <= cap:
        return X, y, len(X)
    rng = np.random.default_rng(seed)
    frac = cap / len(X)
    keep = []
    for c in np.unique(y):
        pool = np.flatnonzero(y == c)
        want = max(2, min(len(pool), int(round(len(pool) * frac))))
        keep.append(rng.choice(pool, want, replace=False))
    idx = np.concatenate(keep)
    rng.shuffle(idx)
    return X[idx], y[idx], len(idx)


def _evaluate(name, est, X_tr, y_tr, X_te, y_te, labels, cap, seed,
              protocol, save_arrays, verbose):
    Xf, yf, used = _cap(X_tr, y_tr, cap, seed)
    if verbose:
        print(f"[sup:{protocol}] {name:<28} train {used:>7,} ...", end="", flush=True)

    t0 = time.perf_counter()
    est.fit(Xf, yf)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_hat = est.predict(X_te)
    pred_s = time.perf_counter() - t0

    row = {
        "protocol": protocol,
        "model": name,
        "train_rows": used,
        "test_rows": len(X_te),
        "accuracy": round(accuracy_score(y_te, y_hat), 5),
        "balanced_accuracy": round(balanced_accuracy_score(y_te, y_hat), 5),
        "macro_f1": round(f1_score(y_te, y_hat, average="macro", zero_division=0), 5),
        "weighted_f1": round(f1_score(y_te, y_hat, average="weighted", zero_division=0), 5),
        "mcc": round(matthews_corrcoef(y_te, y_hat), 5),
        "fit_seconds": round(fit_s, 2),
        "predict_seconds": round(pred_s, 2),
    }

    p, r, f, s = precision_recall_fscore_support(y_te, y_hat, labels=labels,
                                                 zero_division=0)
    per_class = [{"protocol": protocol, "model": name, "outcome": labels[i],
                  "support": int(s[i]), "precision": round(float(p[i]), 5),
                  "recall": round(float(r[i]), 5), "f1": round(float(f[i]), 5)}
                 for i in range(len(labels))]

    if save_arrays:
        os.makedirs(ARRAYS, exist_ok=True)
        cm = confusion_matrix(y_te, y_hat, labels=labels)
        np.save(os.path.join(ARRAYS, f"cm_{protocol}_{slugify(name)}.npy"), cm)

    if verbose:
        print(f" acc={row['accuracy']:.4f} macroF1={row['macro_f1']:.4f} "
              f"fit={fit_s:.1f}s")
    return row, per_class


def tree_rules(df: pd.DataFrame, feature_names, max_depth=4, seed=42) -> str:
    """
    A shallow decision tree rendered as text, so its paths can be set beside the
    association rules. Two independent routes to the same kind of structure.
    """
    X, y, _, names = encode(df)
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth,
                                  min_samples_leaf=50, class_weight="balanced",
                                  random_state=seed).fit(X, y)
    text = export_text(tree, feature_names=names, max_depth=max_depth)
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "decision_tree_rules.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def run(df: pd.DataFrame, seed=42, temporal_cutoff=2021, verbose=True):
    """Both protocols. Returns (summary_df, per_class_df)."""
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(ARRAYS, exist_ok=True)

    X, y, _, feature_names = encode(df)
    labels = sorted(pd.unique(y))
    years = df["year"].to_numpy()

    summary, per_class = [], []

    # ---- protocol 1: random stratified split ------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    if verbose:
        print(f"[sup:random] train {X_tr.shape} test {X_te.shape}")
    for name, est, cap in model_zoo(seed):
        row, pc = _evaluate(name, est, X_tr, y_tr, X_te, y_te, labels, cap, seed,
                            "random", True, verbose)
        summary.append(row)
        per_class.extend(pc)

    # ---- protocol 2: temporal holdout -------------------------------------
    tr_mask, te_mask = years <= temporal_cutoff, years > temporal_cutoff
    if te_mask.sum() > 50 and tr_mask.sum() > 50:
        if verbose:
            print(f"[sup:temporal] train <= {temporal_cutoff}: {tr_mask.sum():,} | "
                  f"test > {temporal_cutoff}: {te_mask.sum():,}")
        for name, est, cap in model_zoo(seed):
            row, pc = _evaluate(name, est, X[tr_mask], y[tr_mask], X[te_mask],
                                y[te_mask], labels, cap, seed, "temporal", True, verbose)
            summary.append(row)
            per_class.extend(pc)
    elif verbose:
        print("[sup:temporal] skipped - not enough rows on one side of the cutoff")

    summary = pd.DataFrame(summary)
    per_class = pd.DataFrame(per_class)
    summary.to_csv(os.path.join(TABLES, "supervised_summary.csv"), index=False)
    per_class.to_csv(os.path.join(TABLES, "supervised_per_class.csv"), index=False)

    with open(os.path.join(ARRAYS, "outcome_labels.json"), "w") as fh:
        json.dump(list(labels), fh)
    with open(os.path.join(ARRAYS, "model_names.json"), "w") as fh:
        json.dump({slugify(n): n for n, _, _ in model_zoo(seed)}, fh, indent=2)

    tree_rules(df, feature_names, seed=seed)
    return summary, per_class


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load, stratified_sample

    df = stratified_sample(load(verbose=False), 25_000)
    s, pc = run(df)
    pd.set_option("display.width", 220)
    print()
    print(s.to_string(index=False))
