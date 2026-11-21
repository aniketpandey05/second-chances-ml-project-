"""
All five taught clustering algorithms on categorical animal attributes.

The methodological spine of the project lives here.

Every feature is categorical, so Euclidean distance is not meaningful. The right
dissimilarity is HAMMING: the fraction of the 11 attributes on which two animals
differ. With 11 attributes it takes only 12 distinct values, and that is a gift
rather than a problem - it makes eps directly interpretable:

    eps = 2/11  ->  "animals differing in at most 2 attributes are neighbours"

K-Medoids, Hierarchical, DBSCAN and OPTICS all accept a precomputed
dissimilarity matrix, so all four run on Hamming. K-Means CANNOT: it computes
means, which requires Euclidean geometry, and the mean of categorical vectors is
not an animal. It is included deliberately, on one-hot data, as the contrast
that shows why medoid-based clustering exists. The medoid of a cluster is a real
animal you can look up; a centroid is not.

The headline experiment is the ATYPICALITY TEST. DBSCAN and OPTICS label
low-density points as noise - animals whose attribute combination is unusual.
The question is whether those animals fare worse. No labels are used to find
them, only to score them afterwards.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import OneHotEncoder

from data import FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")
ARRAYS = os.path.join(ROOT, "results", "arrays")


# ---------------------------------------------------------------------------
# dissimilarity
# ---------------------------------------------------------------------------
def codes(df: pd.DataFrame) -> np.ndarray:
    """Categorical values -> integer codes, shape (n, n_features)."""
    return np.column_stack([
        pd.Categorical(df[f].astype(str)).codes.astype(np.int16) for f in FEATURES
    ])


def hamming_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Pairwise Hamming dissimilarity: the fraction of attributes that differ.
    Computed feature by feature so only one n x n boolean temp exists at a time.
    """
    c = codes(df)
    n, p = c.shape
    D = np.zeros((n, n), dtype=np.float32)
    for j in range(p):
        col = c[:, j]
        D += (col[:, None] != col[None, :])
    D /= p
    np.fill_diagonal(D, 0.0)
    return D


def onehot(df: pd.DataFrame) -> np.ndarray:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    return enc.fit_transform(df[FEATURES])


# ---------------------------------------------------------------------------
# K-Medoids (alternating / Park-Jun), precomputed dissimilarity
# ---------------------------------------------------------------------------
def kmedoids(D: np.ndarray, k: int, max_iter=60, seed=42):
    """
    Assign to nearest medoid, then replace each medoid by the member minimising
    total in-cluster dissimilarity. Needs only the distance matrix, which is
    exactly why it works on categorical data and K-Means does not.

    Returns (labels, medoid_indices, total_cost).
    """
    rng = np.random.default_rng(seed)
    n = len(D)
    medoids = rng.choice(n, size=k, replace=False)
    labels = np.full(n, -1)

    for _ in range(max_iter):
        new_labels = D[:, medoids].argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = np.flatnonzero(labels == j)
            if len(members) == 0:
                medoids[j] = rng.integers(n)
                continue
            sub = D[np.ix_(members, members)]
            medoids[j] = members[sub.sum(axis=0).argmin()]

    labels = D[:, medoids].argmin(axis=1)
    cost = float(D[np.arange(n), medoids[labels]].sum())
    return labels, medoids, cost


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _sil(D, labels, max_n=4000, seed=42):
    mask = labels >= 0
    if mask.sum() < 10 or len(np.unique(labels[mask])) < 2:
        return None
    idx = np.flatnonzero(mask)
    if len(idx) > max_n:
        idx = np.random.default_rng(seed).choice(idx, max_n, replace=False)
        idx = np.sort(idx)
    if len(np.unique(labels[idx])) < 2:
        return None
    return float(silhouette_score(D[np.ix_(idx, idx)], labels[idx], metric="precomputed"))


MIN_NOISE_FOR_RATE = 30     # below this a "rate" is just noise about noise


def atypicality(y: np.ndarray, labels: np.ndarray) -> dict:
    """
    THE headline test. Compare outcomes for noise points against clustered ones.
    Noise is found without labels; labels only score it afterwards.

    `reliable` guards against the trap this walked into once already: a setting
    that produces 3 noise points will happily report a 66.7% euthanasia rate,
    which is two animals. Only rows with >= MIN_NOISE_FOR_RATE noise points are
    reported as findings; the rest are kept for completeness and marked.
    """
    noise = labels == -1
    clustered = ~noise
    base = {"noise_n": int(noise.sum()), "clustered_n": int(clustered.sum())}
    if noise.sum() == 0 or clustered.sum() == 0:
        return {**base, "euth_rate_noise": None, "euth_rate_clustered": None,
                "euth_rate_ratio": None, "adopt_rate_noise": None,
                "adopt_rate_clustered": None, "reliable": False}
    en = float((y[noise] == "Euthanasia").mean())
    ec = float((y[clustered] == "Euthanasia").mean())
    return {
        **base,
        "euth_rate_noise": round(100 * en, 3),
        "euth_rate_clustered": round(100 * ec, 3),
        "euth_rate_ratio": round(en / ec, 3) if ec > 0 else None,
        "adopt_rate_noise": round(100 * float((y[noise] == "Adoption").mean()), 3),
        "adopt_rate_clustered": round(100 * float((y[clustered] == "Adoption").mean()), 3),
        "reliable": bool(noise.sum() >= MIN_NOISE_FOR_RATE),
    }


def score(name, D, y, labels, seconds, extra=None) -> dict:
    h, c, _ = homogeneity_completeness_v_measure(y, labels)
    n_clusters = len(set(labels[labels >= 0]))
    noise = int((labels == -1).sum())
    row = {
        "algorithm": name,
        "n_rows": len(labels),
        "n_clusters": n_clusters,
        "noise_points": noise,
        "noise_pct": round(100 * noise / len(labels), 3),
        "ARI": round(float(adjusted_rand_score(y, labels)), 5),
        "NMI": round(float(normalized_mutual_info_score(y, labels)), 5),
        "homogeneity": round(float(h), 5),
        "completeness": round(float(c), 5),
        "silhouette": None,
        "seconds": round(seconds, 2),
    }
    s = _sil(D, labels)
    row["silhouette"] = None if s is None else round(s, 5)
    if extra:
        row.update(extra)
    return row


def composition(df: pd.DataFrame, labels: np.ndarray, medoids=None) -> pd.DataFrame:
    """Per-cluster size, dominant outcome, purity, and the medoid animal."""
    rows = []
    y = df["outcome"].to_numpy()
    for cl in sorted(set(labels)):
        m = labels == cl
        if m.sum() == 0:
            continue
        vals, cnts = np.unique(y[m], return_counts=True)
        rec = {
            "cluster": int(cl),
            "size": int(m.sum()),
            "dominant_outcome": str(vals[cnts.argmax()]),
            "purity": round(float(cnts.max() / m.sum()), 4),
            "euthanasia_pct": round(100 * float((y[m] == "Euthanasia").mean()), 3),
            "median_stay_days": round(float(df.loc[m, "length_of_stay_days"].median()), 2),
        }
        if medoids is not None and 0 <= cl < len(medoids):
            med = df.iloc[medoids[cl]]
            rec["medoid_animal"] = (f"{med.animal_type}, {med.breed_group}, "
                                    f"{med.colour_group}, {med.age_band}, "
                                    f"{med.intake_type}")
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("size", ascending=False)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(df: pd.DataFrame, n_metric=6000, n_hier=4000, k=8, seed=42, verbose=True):
    """
    n_metric : rows for K-Medoids / DBSCAN / K-Means (n x n matrix in memory)
    n_hier   : rows for Hierarchical / OPTICS (both heavier than O(n^2) in practice)
    """
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(ARRAYS, exist_ok=True)
    from data import stratified_sample

    big = stratified_sample(df, n_metric, seed)
    small = stratified_sample(df, n_hier, seed)

    t0 = time.perf_counter()
    D_big = hamming_matrix(big)
    D_small = hamming_matrix(small)
    if verbose:
        print(f"[clu] Hamming matrices {D_big.shape} and {D_small.shape} in "
              f"{time.perf_counter()-t0:.1f}s "
              f"({D_big.nbytes/1e6:.0f} + {D_small.nbytes/1e6:.0f} MB)")
        vals = np.unique(D_big)
        print(f"[clu] distinct distance values: {len(vals)} "
              f"(1 step = 1 of {len(FEATURES)} attributes differing)")

    y_big = big["outcome"].to_numpy()
    y_small = small["outcome"].to_numpy()
    rows, atyp = [], []

    # ---- K-Means on one-hot: the deliberate contrast ----------------------
    Xb = onehot(big)
    t0 = time.perf_counter()
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xb)
    rows.append(score(f"K-Means (one-hot, k={k})", D_big, y_big, lab,
                      time.perf_counter() - t0,
                      {"dissimilarity": "Euclidean on one-hot"}))
    if verbose:
        print(f"[clu] k-means      ARI={rows[-1]['ARI']:.4f}")

    # ---- K-Medoids on Hamming --------------------------------------------
    t0 = time.perf_counter()
    lab_md, medoids, cost = kmedoids(D_big, k, seed=seed)
    rows.append(score(f"K-Medoids (Hamming, k={k})", D_big, y_big, lab_md,
                      time.perf_counter() - t0,
                      {"dissimilarity": "Hamming", "cost": round(cost, 2)}))
    composition(big, lab_md, medoids).to_csv(
        os.path.join(TABLES, "kmedoids_composition.csv"), index=False)
    if verbose:
        print(f"[clu] k-medoids    ARI={rows[-1]['ARI']:.4f} cost={cost:.1f}")

    # ---- Hierarchical -----------------------------------------------------
    for link in ("complete", "average"):
        t0 = time.perf_counter()
        lab_h = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                        linkage=link).fit_predict(D_small)
        rows.append(score(f"Hierarchical ({link}, Hamming)", D_small, y_small, lab_h,
                          time.perf_counter() - t0, {"dissimilarity": "Hamming"}))
        if verbose:
            print(f"[clu] hier/{link:<9} ARI={rows[-1]['ARI']:.4f}")
    # Ward needs Euclidean, so it gets one-hot - another instance of the same point
    t0 = time.perf_counter()
    lab_w = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(onehot(small))
    rows.append(score("Hierarchical (ward, one-hot)", D_small, y_small, lab_w,
                      time.perf_counter() - t0, {"dissimilarity": "Euclidean on one-hot"}))
    np.save(os.path.join(ARRAYS, "dendrogram_linkage.npy"),
            linkage(onehot(small)[:1200], method="ward"))

    # ---- DBSCAN: eps in units of "attributes allowed to differ" -----------
    p = len(FEATURES)
    best = None
    for n_diff in (1, 2, 3):
        eps = (n_diff + 0.5) / p          # strictly between n_diff and n_diff+1
        t0 = time.perf_counter()
        lab_db = DBSCAN(eps=eps, min_samples=10, metric="precomputed").fit_predict(D_big)
        dt = time.perf_counter() - t0
        a = atypicality(y_big, lab_db)
        rows.append(score(f"DBSCAN (<={n_diff} attrs differ)", D_big, y_big, lab_db, dt,
                          {"dissimilarity": "Hamming", "eps_attrs": n_diff,
                           "min_samples": 10}))
        atyp.append({"algorithm": f"DBSCAN (<={n_diff} attrs differ)", **a})
        if verbose:
            print(f"[clu] dbscan<={n_diff}    clusters={rows[-1]['n_clusters']:<4} "
                  f"noise={rows[-1]['noise_pct']:5.2f}%  "
                  f"euth noise/clustered = {a['euth_rate_noise']}% / {a['euth_rate_clustered']}%")
        if best is None or rows[-1]["n_clusters"] > 1:
            best = (n_diff, lab_db)
    if best:
        composition(big, best[1]).to_csv(
            os.path.join(TABLES, "dbscan_composition.csv"), index=False)

    # ---- OPTICS -----------------------------------------------------------
    t0 = time.perf_counter()
    op = OPTICS(min_samples=10, metric="precomputed", cluster_method="xi", xi=0.05)
    lab_op = op.fit_predict(D_small.astype(np.float64))
    dt = time.perf_counter() - t0
    a = atypicality(y_small, lab_op)
    rows.append(score("OPTICS (Hamming, xi)", D_small, y_small, lab_op, dt,
                      {"dissimilarity": "Hamming", "min_samples": 10}))
    atyp.append({"algorithm": "OPTICS (Hamming, xi)", **a})
    reach = op.reachability_[op.ordering_]
    np.save(os.path.join(ARRAYS, "optics_reachability.npy"), reach)
    np.save(os.path.join(ARRAYS, "optics_ordering_outcome.npy"),
            y_small[op.ordering_].astype(object), allow_pickle=True)
    if verbose:
        print(f"[clu] optics       clusters={rows[-1]['n_clusters']} "
              f"noise={rows[-1]['noise_pct']:.2f}%  "
              f"euth noise/clustered = {a['euth_rate_noise']}% / {a['euth_rate_clustered']}%")

    summary = pd.DataFrame(rows)
    atypical = pd.DataFrame(atyp)
    summary.to_csv(os.path.join(TABLES, "clustering_summary.csv"), index=False)
    atypical.to_csv(os.path.join(TABLES, "atypicality_test.csv"), index=False)
    return summary, atypical


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load

    df = load(verbose=False)
    s, a = run(df, n_metric=4000, n_hier=2500)
    pd.set_option("display.width", 240)
    print()
    print(s.to_string(index=False))
    print()
    print(a.to_string(index=False))
