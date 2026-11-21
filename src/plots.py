"""
Report figures.

Design rules held throughout, so the figures read as one system:
  * magnitude  -> single-hue sequential blue ramp (heatmaps)
  * identity   -> fixed categorical order, never cycled; legend for >= 2 series
  * polarity   -> diverging blue/orange about a neutral zero line
  * NO dual-axis charts; different scales get separate panels
  * recessive grid and axes, thin marks, selective direct labels
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")
ARRAYS = os.path.join(ROOT, "results", "arrays")
PLOTS = os.path.join(ROOT, "results", "plots")

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#2a78d6", "#256abf", "#184f95", "#0d366b"])

OUTCOME_ORDER = ["Adoption", "Transfer", "Return to Owner", "Euthanasia", "Other"]


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2.0, "lines.markersize": 5, "font.size": 9,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def _save(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    path = os.path.join(PLOTS, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] {name}")
    return path


def _csv(name):
    p = os.path.join(TABLES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _npy(name, **kw):
    p = os.path.join(ARRAYS, name)
    return np.load(p, **kw) if os.path.exists(p) else None


# ---------------------------------------------------------------- 1 outcomes
def fig_outcomes(df):
    counts = df["outcome"].value_counts().reindex(
        [o for o in OUTCOME_ORDER if o in df["outcome"].unique()]).dropna()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.barh(counts.index[::-1], counts.values[::-1], color=SERIES[0], height=0.62)
    ax.set_xlabel("animals")
    ax.set_title(f"Shelter outcomes, {len(df):,} dogs and cats")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    for i, v in enumerate(counts.values[::-1]):
        ax.annotate(f"{v:,}  ({100*v/len(df):.1f}%)", (v, i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
    ax.set_xlim(right=counts.max() * 1.28)
    return _save(fig, "01-outcomes.png")


# ------------------------------------------------------------------ 2 rules
def fig_rules(top):
    if top is None or top.empty:
        return None
    d = top.sort_values("lift").tail(16)
    outcomes = [o for o in OUTCOME_ORDER if o in d["outcome"].unique()]
    cmap = {o: SERIES[i] for i, o in enumerate(outcomes)}
    fig, ax = plt.subplots(figsize=(9.0, 0.42 * len(d) + 1.6))
    ax.barh(range(len(d)), d["lift"], color=[cmap[o] for o in d["outcome"]], height=0.64)
    ax.set_yticks(range(len(d)), d["antecedent"], fontsize=7.5)
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1.0)
    ax.set_xlabel("lift  (1.0 = no association)")
    ax.set_title("Strongest association rules, by lift")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    for i, (v, n) in enumerate(zip(d["lift"], d["n_animals"])):
        ax.annotate(f"{v:.2f}  (n={n:,})", (v, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7, color=INK_2)
    ax.set_xlim(right=d["lift"].max() * 1.3)
    handles = [plt.Line2D([], [], color=cmap[o], lw=6) for o in outcomes]
    ax.legend(handles, outcomes, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncols=len(outcomes))
    return _save(fig, "02-rules-lift.png")


# -------------------------------------------------- 3 apriori vs fp-growth
def fig_rule_scaling(scale, sweep):
    if scale is None:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    a1.plot(scale["n_transactions"], scale["Apriori_seconds"], marker="o",
            color=SERIES[0], label="Apriori")
    a1.plot(scale["n_transactions"], scale["FP-Growth_seconds"], marker="s",
            color=SERIES[1], label="FP-Growth")
    a1.set_xlabel("transactions"); a1.set_ylabel("seconds")
    a1.set_title("Cost of mining the same itemsets")
    a1.grid(True); a1.set_axisbelow(True); a1.legend(loc="upper left")

    if sweep is not None:
        a2.plot(sweep["min_support"], sweep["n_itemsets"], marker="o", color=SERIES[2])
        a2.set_xscale("log"); a2.set_yscale("log")
        # explicit ticks at the sampled thresholds: matplotlib's automatic log
        # minor ticks collide into an unreadable smear at this range
        xs = sorted(sweep["min_support"])
        a2.set_xticks(xs, [f"{v:g}" for v in xs])
        a2.minorticks_off()
        ys = sweep["n_itemsets"]
        a2.set_yticks([1000, 2000, 5000, 10000], ["1k", "2k", "5k", "10k"])
        a2.set_ylim(ys.min() * 0.7, ys.max() * 1.4)
        a2.set_xlabel("minimum support"); a2.set_ylabel("frequent itemsets")
        a2.set_title("Choosing the support threshold")
        a2.grid(True); a2.set_axisbelow(True)
    fig.suptitle("Association rule mining cost", x=0.01, ha="left",
                 fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, "03-rule-mining-cost.png")


# ------------------------------------------------- 4 random vs temporal
def fig_protocols(summary):
    if summary is None or summary.empty:
        return None
    piv = summary.pivot_table(index="model", columns="protocol", values="macro_f1")
    piv = piv.sort_values(piv.columns[0], ascending=True)
    y = np.arange(len(piv)); h = 0.34
    fig, ax = plt.subplots(figsize=(8.0, 0.56 * len(piv) + 1.8))
    for i, proto in enumerate(piv.columns):
        ax.barh(y + (i - 0.5) * (h + 0.02), piv[proto], height=h,
                color=SERIES[i], label=f"{proto} split")
    ax.set_yticks(y, piv.index)
    ax.set_xlabel("macro F1")
    ax.set_title("Honest evaluation costs you accuracy you never had")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)
    return _save(fig, "04-protocol-comparison.png")


# ----------------------------------------------------- 5 accuracy vs balance
def fig_accuracy_trap(summary):
    if summary is None or summary.empty:
        return None
    d = summary[summary["protocol"] == "random"].sort_values("accuracy")
    if d.empty:
        d = summary.sort_values("accuracy")
    y = np.arange(len(d)); h = 0.26
    fig, ax = plt.subplots(figsize=(8.0, 0.56 * len(d) + 1.8))
    for i, (col, lab) in enumerate([("accuracy", "Accuracy"),
                                    ("balanced_accuracy", "Balanced accuracy"),
                                    ("macro_f1", "Macro F1")]):
        ax.barh(y + (i - 1) * (h + 0.02), d[col], height=h, color=SERIES[i], label=lab)
    ax.set_yticks(y, d["model"])
    ax.set_xlim(0, 1.0); ax.set_xlabel("score")
    ax.set_title("The same models, scored three ways")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=3)
    return _save(fig, "05-accuracy-trap.png")


# --------------------------------------------------------- 6 per-class F1
def fig_per_class(per_class):
    if per_class is None or per_class.empty:
        return None
    d = per_class[per_class["protocol"] == "random"]
    piv = d.pivot_table(index="model", columns="outcome", values="f1")
    order = [o for o in OUTCOME_ORDER if o in piv.columns]
    piv = piv[order]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(1.6 + 1.05 * len(order), 0.5 * len(piv) + 1.9))
    im = ax.imshow(piv.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
    ax.set_yticks(range(len(piv)), piv.index)
    ax.set_title("Per-class F1: the rare outcomes are where models fail")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}", (j, i), ha="center", va="center",
                            fontsize=7.5, color="#ffffff" if v > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("F1", fontsize=8)
    cb.outline.set_visible(False)
    return _save(fig, "06-per-class-f1.png")


# ------------------------------------------------------ 7 confusion matrices
def fig_confusion(protocol="random"):
    labels = _npy("outcome_labels.json".replace(".json", ".npy"))
    path = os.path.join(ARRAYS, "outcome_labels.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        labels = json.load(fh)
    names = {}
    np_path = os.path.join(ARRAYS, "model_names.json")
    if os.path.exists(np_path):
        with open(np_path) as fh:
            names = json.load(fh)

    files = sorted(f for f in os.listdir(ARRAYS) if f.startswith(f"cm_{protocol}_"))
    if not files:
        return None
    cols = min(3, len(files)); rows = int(np.ceil(len(files) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.9 * cols, 3.6 * rows), squeeze=False)
    for ax, f in zip(axes.ravel(), files):
        cm = np.load(os.path.join(ARRAYS, f)).astype(float)
        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
        slug = f[len(f"cm_{protocol}_"):-4]
        ax.set_title(names.get(slug, slug.replace("_", " ").title()), fontsize=9)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
        ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("actual", fontsize=8)
        ax.tick_params(length=0)
    for ax in axes.ravel()[len(files):]:
        ax.axis("off")
    fig.suptitle(f"Confusion matrices, row-normalised ({protocol} split)",
                 x=0.01, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, f"07-confusion-{protocol}.png")


# ------------------------------------------------------- 8 atypicality test
def fig_atypicality(atyp):
    if atyp is None or atyp.empty:
        return None
    d = atyp.dropna(subset=["euth_rate_noise", "euth_rate_clustered"])
    if "reliable" in d.columns:
        # a "rate" computed from a handful of noise points is not a finding
        d = d[d["reliable"].astype(bool)]
    if d.empty:
        return None
    d = d.assign(label=d["algorithm"] + "\n(" + d["noise_n"].astype(str) + " atypical)")
    y = np.arange(len(d)); h = 0.34
    fig, ax = plt.subplots(figsize=(8.4, 0.62 * len(d) + 1.9))
    ax.barh(y - 0.5 * (h + 0.02), d["euth_rate_clustered"], height=h,
            color=SERIES[0], label="typical (clustered)")
    ax.barh(y + 0.5 * (h + 0.02), d["euth_rate_noise"], height=h,
            color=SERIES[1], label="atypical (noise)")
    ax.set_yticks(y, d["label"], fontsize=7.5)
    ax.set_xlabel("euthanasia rate (%)")
    ax.set_title("Animals with unusual attribute combinations fare worse")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    for i, (n, c, r) in enumerate(zip(d["euth_rate_noise"], d["euth_rate_clustered"],
                                      d["euth_rate_ratio"])):
        ax.annotate(f"{n:.2f}%  ({r:.2f}x)", (n, y[i] + 0.5 * (h + 0.02)),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=INK_2)
    ax.set_xlim(right=d["euth_rate_noise"].max() * 1.75)
    # inside the axes: the bars are short, so the right half is empty, and an
    # above-axes legend collides with the title
    ax.legend(loc="center right")
    return _save(fig, "08-atypicality.png")


# ------------------------------------------------------------ 9 dendrogram
def fig_dendrogram(n_clusters=6):
    Z = _npy("dendrogram_linkage.npy")
    if Z is None:
        return None
    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    cut = Z[-(n_clusters - 1), 2]
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=cut,
               above_threshold_color=MUTED)
    ax.axhline(cut, color=INK_2, linestyle="--", linewidth=1.0)
    ax.set_ylabel("Ward linkage distance"); ax.set_xlabel("animals (1,200-row sample)")
    ax.set_title("Hierarchical clustering of animal profiles")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    return _save(fig, "09-dendrogram.png")


# -------------------------------------------------------------- 10 the claims
def fig_claims(head):
    if head is None or head.empty:
        return None
    d = head.sort_values("adoption_gap_pp")
    colours = [SERIES[1] if v < 0 else SERIES[0] for v in d["adoption_gap_pp"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 0.62 * len(d) + 2.0),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(d))
    a1.barh(y, d["adoption_gap_pp"], color=colours, height=0.6)
    a1.axvline(0, color=AXIS, linewidth=1.0)
    a1.set_yticks(y, d["claim"], fontsize=8)
    a1.set_xlabel("adoption rate gap (percentage points)")
    a1.set_title("What actually changes an animal's chances")
    a1.xaxis.grid(True); a1.set_axisbelow(True)
    for i, v in enumerate(d["adoption_gap_pp"]):
        a1.annotate(f"{v:+.1f}", (v, i), xytext=(6 if v >= 0 else -6, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=7.5, color=INK_2)
    pad = max(abs(d["adoption_gap_pp"])) * 0.25
    a1.set_xlim(min(d["adoption_gap_pp"]) - pad, max(d["adoption_gap_pp"]) + pad)

    # Deliberately NOT the blue/orange of the left panel: there those two hues
    # encode the SIGN of the gap, and reusing them here would make one colour
    # mean two different things across the same figure.
    h = 0.34
    a2.barh(y - 0.5 * (h + 0.02), d["others_median_stay"], height=h,
            color=MUTED, label="everyone else")
    a2.barh(y + 0.5 * (h + 0.02), d["group_median_stay"], height=h,
            color=SERIES[2], label="the group named")
    a2.set_yticks(y, [""] * len(d))
    a2.set_xlabel("median days to outcome")
    a2.set_title("…and how long they wait")
    a2.xaxis.grid(True); a2.set_axisbelow(True)
    a2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)
    fig.tight_layout()
    return _save(fig, "10-claims.png")


# ------------------------------------------------------- 11 colour adjusted
def fig_colour(colour):
    if colour is None or colour.empty:
        return None
    d = colour.sort_values("standardised_pct")
    y = np.arange(len(d)); h = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 0.56 * len(d) + 1.9))
    ax.barh(y - 0.5 * (h + 0.02), d["crude_pct"], height=h, color=SERIES[0],
            label="crude")
    ax.barh(y + 0.5 * (h + 0.02), d["standardised_pct"], height=h, color=SERIES[1],
            label="standardised for species, age, intake route")
    ax.set_yticks(y, d["value"])
    ax.set_xlabel("adoption rate (%)")
    ax.set_title("Adoption by colour, before and after controlling for confounders")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_xlim(left=min(d["crude_pct"].min(), d["standardised_pct"].min()) - 4)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)
    return _save(fig, "11-colour-adjusted.png")


# ---------------------------------------------------------------------------
def fig_field_ranking(ranks):
    """Which attributes each method leans on. The disagreement is the point."""
    if ranks is None or ranks.empty:
        return None
    d = ranks.sort_values("tree_share_pct")
    y = np.arange(len(d)); h = 0.34
    fig, ax = plt.subplots(figsize=(8.4, 0.56 * len(d) + 2.0))
    ax.barh(y - 0.5 * (h + 0.02), d["tree_share_pct"], height=h,
            color=SERIES[0], label="decision tree (what is sufficient)")
    ax.barh(y + 0.5 * (h + 0.02), d["apriori_share_pct"], height=h,
            color=SERIES[1], label="Apriori (what is associated)")
    ax.set_yticks(y, d["field"])
    ax.set_xlabel("share of the method's attention (%)")
    ax.set_title("Two methods, the same data, different attributes")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    for i, (t, a) in enumerate(zip(d["tree_share_pct"], d["apriori_share_pct"])):
        if t == 0 and a > 0:
            ax.annotate("tree never splits on this", (a, y[i]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=7,
                        color=INK_2, style="italic")
    ax.set_xlim(right=max(d["tree_share_pct"].max(), d["apriori_share_pct"].max()) * 1.6)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=1)
    return _save(fig, "12-field-ranking.png")


def fig_torch_curves(curves):
    """
    Training curves, and the point the torch model exists to make: weighting the
    loss trades accuracy for balance, and you can see the trade happening.
    Two panels, never two y-axes - loss and balanced accuracy are different
    scales.
    """
    if curves is None or curves.empty:
        return None
    d = curves[curves["protocol"] == "random"]
    if d.empty:
        d = curves
    variants = list(dict.fromkeys(d["variant"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for i, v in enumerate(variants):
        g = d[d["variant"] == v].sort_values("epoch")
        label = "weighted loss" if "unweighted" not in v else "unweighted"
        a1.plot(g["epoch"], g["train_loss"], color=SERIES[i], label=f"{label} (train)")
        a1.plot(g["epoch"], g["val_loss"], color=SERIES[i], linestyle="--",
                label=f"{label} (val)")
        a2.plot(g["epoch"], g["val_balanced_acc"], color=SERIES[i], marker="o",
                markersize=3, label=label)
    a1.set_xlabel("epoch")
    a1.set_ylabel("cross-entropy loss (own objective)")
    # the two variants optimise DIFFERENT objectives, so their absolute heights
    # are not comparable; each curve is only meaningful against its own dashed
    # validation twin
    a1.set_title("Convergence (each against its own objective)")
    a1.grid(True); a1.set_axisbelow(True); a1.legend(loc="best", fontsize=7)
    a2.set_xlabel("epoch"); a2.set_ylabel("validation balanced accuracy")
    a2.set_title("What the class weighting buys")
    a2.grid(True); a2.set_axisbelow(True); a2.legend(loc="best", fontsize=7.5)
    fig.suptitle("PyTorch network with learned categorical embeddings",
                 x=0.01, ha="left", fontsize=11, weight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, "13-torch-training.png")


def build_all(df=None, top=None, scale=None, sweep=None, summary=None,
              per_class=None, atyp=None, head=None, colour=None, ranks=None,
              curves=None):
    apply_style()
    made = []
    if df is not None:
        made.append(fig_outcomes(df))
    made.append(fig_rules(top if top is not None else _csv("rules_top.csv")))
    made.append(fig_rule_scaling(scale if scale is not None else _csv("rule_algorithm_scaling.csv"),
                                 sweep if sweep is not None else _csv("support_sweep.csv")))
    s = summary if summary is not None else _csv("supervised_summary.csv")
    made.append(fig_protocols(s))
    made.append(fig_accuracy_trap(s))
    made.append(fig_per_class(per_class if per_class is not None
                              else _csv("supervised_per_class.csv")))
    made.append(fig_confusion("random"))
    made.append(fig_confusion("temporal"))
    made.append(fig_atypicality(atyp if atyp is not None else _csv("atypicality_test.csv")))
    made.append(fig_dendrogram())
    made.append(fig_claims(head if head is not None else _csv("bias_headline.csv")))
    made.append(fig_colour(colour if colour is not None else _csv("bias_colour.csv")))
    made.append(fig_field_ranking(ranks if ranks is not None
                                  else _csv("field_ranking.csv")))
    made.append(fig_torch_curves(curves if curves is not None
                                 else _csv("torch_training_curves.csv")))
    return [m for m in made if m]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load
    build_all(df=load(verbose=False))
