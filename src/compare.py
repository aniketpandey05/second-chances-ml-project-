"""
Do association rules and decision tree paths find the same structure?

Two methods, taught separately, pointed at the same data:

  Apriori      enumerates frequent attribute combinations, then keeps those
               whose link to an outcome beats a lift threshold. It is
               exhaustive within its support floor and has no notion of order.

  Decision tree  greedily picks one attribute at a time to maximise information
               gain, producing a partition. Each leaf is a rule.

If the structure in the data is real, they should agree. Where they disagree is
more interesting than where they agree, and there are two systematic reasons:

  1. NEGATION. A tree splits on "breed is NOT pit bull" as easily as on "breed
     IS pit bull". Apriori over positive items cannot express that at all. Any
     tree path relying on a negative condition has no Apriori counterpart by
     construction, not by disagreement.

  2. SUPPORT vs PURITY. Apriori will not look below its support floor, so a
     small, very high-lift pocket is invisible to it. A tree will happily carve
     one out if it reduces impurity.

This module quantifies both.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from data import FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")

SHORT = {
    "animal_type": "type", "breed_group": "breed", "is_mix": "mixed",
    "colour_group": "colour", "colour_pattern": "pattern",
    "colour_multi": "multicolour", "age_band": "age", "sex": "sex",
    "sterilised_at_intake": "sterilised", "intake_type": "intake",
    "intake_condition": "condition",
}


def _split_name(onehot_name: str) -> tuple[str, str]:
    """'breed_group_Pit Bull' -> ('breed_group', 'Pit Bull')."""
    for f in sorted(FEATURES, key=len, reverse=True):
        if onehot_name.startswith(f + "_"):
            return f, onehot_name[len(f) + 1:]
    return onehot_name, ""


def _pretty(field: str, value: str, positive=True) -> str:
    op = "=" if positive else "!="
    return f"{SHORT.get(field, field)}{op}{value}"


# ---------------------------------------------------------------------------
# tree paths as rules
# ---------------------------------------------------------------------------
def tree_paths(df: pd.DataFrame, max_depth=6, min_samples_leaf=200,
               seed=42) -> pd.DataFrame:
    """
    Fit a tree and turn every leaf into a rule with support, confidence and
    lift computed the same way Apriori computes them, so the two are comparable.
    """
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    X = enc.fit_transform(df[FEATURES])
    y = df["outcome"].to_numpy()
    names = list(enc.get_feature_names_out(FEATURES))

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 random_state=seed).fit(X, y)
    t = clf.tree_
    classes = clf.classes_
    base = {c: float((y == c).mean()) for c in classes}
    n = len(df)

    rows = []

    def walk(node, pos, neg):
        if t.children_left[node] == -1:          # leaf
            counts = t.value[node][0]
            counts = counts * t.n_node_samples[node] / counts.sum()
            top = int(counts.argmax())
            outcome = classes[top]
            support = t.n_node_samples[node] / n
            confidence = float(counts[top] / counts.sum())
            rows.append({
                "outcome": outcome,
                "positive_items": tuple(sorted(pos)),
                "antecedent": " AND ".join(
                    sorted([_pretty(f, v) for f, v in pos]
                           + [_pretty(f, v, False) for f, v in neg])) or "(root)",
                "n_positive": len(pos),
                "n_negative": len(neg),
                "support": round(support, 6),
                "confidence": round(confidence, 5),
                "lift": round(confidence / base[outcome], 4) if base[outcome] else None,
                "n_animals": int(t.n_node_samples[node]),
            })
            return
        field, value = _split_name(names[t.feature[node]])
        # one-hot split at 0.5: left = does NOT have the value, right = has it
        walk(t.children_left[node], pos, neg + [(field, value)])
        walk(t.children_right[node], pos + [(field, value)], neg)

    walk(0, [], [])
    out = pd.DataFrame(rows).sort_values("lift", ascending=False)
    os.makedirs(TABLES, exist_ok=True)
    out.drop(columns=["positive_items"]).to_csv(
        os.path.join(TABLES, "tree_paths.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------
def _apriori_itemsets(rules: pd.DataFrame) -> dict:
    """outcome -> list of (frozenset of 'short=value' items, lift)."""
    out = {}
    for _, r in rules.iterrows():
        items = frozenset(x.strip() for x in r["antecedent"].split(" AND "))
        out.setdefault(r["outcome"], []).append((items, r["lift"]))
    return out


def agreement(tree_rules: pd.DataFrame, apriori_rules: pd.DataFrame) -> pd.DataFrame:
    """
    For each tree leaf, is there an Apriori rule predicting the same outcome
    whose antecedent is contained in the leaf's positive conditions?

    Containment rather than equality is the right test: a leaf is a conjunction
    of conditions, and an Apriori rule covering a subset of them is the same
    claim, stated more generally.
    """
    by_outcome = _apriori_itemsets(apriori_rules) if not apriori_rules.empty else {}
    rows = []
    for _, r in tree_rules.iterrows():
        pos = frozenset(_pretty(f, v) for f, v in r["positive_items"])
        cands = by_outcome.get(r["outcome"], [])
        matches = [lift for items, lift in cands if items and items <= pos]
        rows.append({
            "outcome": r["outcome"],
            "tree_rule": r["antecedent"],
            "n_positive": r["n_positive"],
            "n_negative": r["n_negative"],
            "tree_lift": r["lift"],
            "tree_support": r["support"],
            "apriori_match": len(matches) > 0,
            "n_apriori_matches": len(matches),
            "best_apriori_lift": round(max(matches), 4) if matches else None,
            "reason_if_unmatched": (
                "" if matches else
                "uses negation" if r["n_negative"] and not r["n_positive"] else
                "below Apriori support floor" if r["support"] < 0.01 else
                "no matching Apriori rule"),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLES, "rules_vs_tree.csv"), index=False)
    return out


def field_ranking(tree_rules: pd.DataFrame, apriori_rules: pd.DataFrame,
                  df: pd.DataFrame) -> pd.DataFrame:
    """
    Which attributes does each method lean on? Tree usage is weighted by the
    animals each path covers; Apriori usage by how many rules a field appears in.
    """
    tree_w = {}
    for _, r in tree_rules.iterrows():
        for f, _v in r["positive_items"]:
            tree_w[SHORT.get(f, f)] = tree_w.get(SHORT.get(f, f), 0) + r["n_animals"]
    total = sum(tree_w.values()) or 1

    ap_w = {}
    if not apriori_rules.empty:
        for _, r in apriori_rules.iterrows():
            for item in r["antecedent"].split(" AND "):
                f = item.split("=")[0].strip()
                ap_w[f] = ap_w.get(f, 0) + 1
    ap_total = sum(ap_w.values()) or 1

    fields = sorted(set(tree_w) | set(ap_w))
    out = pd.DataFrame({
        "field": fields,
        "tree_share_pct": [round(100 * tree_w.get(f, 0) / total, 2) for f in fields],
        "apriori_share_pct": [round(100 * ap_w.get(f, 0) / ap_total, 2) for f in fields],
    })
    out["rank_tree"] = out["tree_share_pct"].rank(ascending=False).astype(int)
    out["rank_apriori"] = out["apriori_share_pct"].rank(ascending=False).astype(int)
    out["rank_gap"] = (out["rank_tree"] - out["rank_apriori"]).abs()
    out = out.sort_values("tree_share_pct", ascending=False)
    out.to_csv(os.path.join(TABLES, "field_ranking.csv"), index=False)
    return out


def run(df: pd.DataFrame, apriori_rules: pd.DataFrame, verbose=True):
    tree = tree_paths(df)
    agree = agreement(tree, apriori_rules)
    ranks = field_ranking(tree, apriori_rules, df)

    matched = int(agree["apriori_match"].sum())
    stats = {
        "tree_leaves": len(agree),
        "matched_by_apriori": matched,
        "matched_pct": round(100 * matched / len(agree), 2) if len(agree) else 0.0,
        "unmatched_negation_only": int((agree["reason_if_unmatched"] == "uses negation").sum()),
        "unmatched_below_support": int(
            (agree["reason_if_unmatched"] == "below Apriori support floor").sum()),
        "apriori_rules": len(apriori_rules),
        "rank_correlation": round(float(
            ranks[["rank_tree", "rank_apriori"]].corr(method="spearman").iloc[0, 1]), 4),
    }
    pd.DataFrame([stats]).to_csv(
        os.path.join(TABLES, "rules_vs_tree_summary.csv"), index=False)

    if verbose:
        pd.set_option("display.width", 220)
        print(f"[cmp] tree leaves: {stats['tree_leaves']}  |  "
              f"with a matching Apriori rule: {matched} ({stats['matched_pct']}%)")
        print(f"[cmp] unmatched because they use negation: "
              f"{stats['unmatched_negation_only']}")
        print(f"[cmp] unmatched because below Apriori's support floor: "
              f"{stats['unmatched_below_support']}")
        print(f"[cmp] field-ranking Spearman correlation: {stats['rank_correlation']}")
        print()
        print(ranks.to_string(index=False))
    return tree, agree, ranks, stats


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load, stratified_sample
    import rules as R

    df = stratified_sample(load(verbose=False), 40_000)
    onehot = R.encode(df, verbose=False)
    ap = R.mine(onehot, len(df), min_support=0.01, min_lift=1.2, verbose=False)
    run(df, ap)
