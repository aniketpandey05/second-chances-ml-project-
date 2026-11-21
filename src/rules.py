"""
Association rule mining over animal attributes.

Each animal is one transaction. Its items are its attribute values written as
"field=value", plus its outcome:

    {animal_type=Cat, breed_group=Domestic Shorthair, colour_group=Black,
     age_band=Juvenile, intake_type=Stray, intake_condition=Normal,
     outcome=Adoption}

Rules are filtered to those with an outcome on the right-hand side, which makes
them readable as "these traits go with this fate".

Two deliberate choices, both worth a paragraph in the report:

1. Rules are ranked by LIFT, not confidence. Adoption is ~52% of records, so a
   rule can reach 0.6 confidence while saying nothing. Lift is what shows the
   antecedent actually moved the odds.

2. Apriori produces the reported rules, because Apriori is the algorithm on the
   syllabus. FP-Growth is used where only itemset counts matter, and the two are
   benchmarked head to head in `algorithm_scaling()`. Measured on this dataset,
   Apriori costs 113s at 162k transactions against FP-Growth's 26s, and the gap
   widens with n - Apriori rescans the database once per candidate level, while
   FP-Growth builds its tree in two passes. The itemsets are identical.
"""

from __future__ import annotations

import os
import time

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from data import FEATURES, OUTCOME_CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")

OUTCOME_ITEMS = {f"outcome={c}" for c in OUTCOME_CLASSES}

# Short, readable field names for printed rules. Without the field name a rule
# reads "Cat AND Yes", which is ambiguous (mixed? sterilised? multicoloured?)
# and makes genuinely different rules look like duplicates.
SHORT = {
    "animal_type": "type",
    "breed_group": "breed",
    "is_mix": "mixed",
    "colour_group": "colour",
    "colour_pattern": "pattern",
    "colour_multi": "multicolour",
    "age_band": "age",
    "sex": "sex",
    "sterilised_at_intake": "sterilised",
    "intake_type": "intake",
    "intake_condition": "condition",
}


def _pretty(item: str) -> str:
    field, _, value = item.partition("=")
    return f"{SHORT.get(field, field)}={value}"


def build_transactions(df: pd.DataFrame, include_outcome=True) -> list[list[str]]:
    cols = list(FEATURES) + (["outcome"] if include_outcome else [])
    sub = df[cols]
    return [[f"{c}={v}" for c, v in zip(cols, row)] for row in sub.itertuples(index=False)]


def encode(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    One-hot boolean frame, DENSE on purpose.

    mlxtend's apriori is markedly slower on a sparse frame at this column count,
    and the dense matrix is only ~17 MB for 162k x 105, so there is nothing to
    gain from sparsity here.
    """
    t0 = time.perf_counter()
    tx = build_transactions(df)
    te = TransactionEncoder()
    onehot = pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)
    if verbose:
        print(f"[rules] encoded {onehot.shape[0]:,} transactions x {onehot.shape[1]} items "
              f"in {time.perf_counter()-t0:.1f}s")
    return onehot


def mine(onehot: pd.DataFrame, n_rows: int, min_support=0.01, max_len=3,
         min_lift=1.2, verbose=True):
    """Apriori, then rules whose consequent is a single outcome item."""
    os.makedirs(TABLES, exist_ok=True)

    t0 = time.perf_counter()
    itemsets = apriori(onehot, min_support=min_support, use_colnames=True, max_len=max_len)
    dt = time.perf_counter() - t0
    if verbose:
        print(f"[rules] apriori: {len(itemsets):,} itemsets at support>={min_support} "
              f"in {dt:.1f}s")
    if itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(itemsets, metric="lift", min_threshold=min_lift)
    keep = rules["consequents"].map(lambda s: len(s) == 1 and next(iter(s)) in OUTCOME_ITEMS)
    rules = rules[keep].copy()
    rules = rules[~rules["antecedents"].map(lambda s: bool(set(s) & OUTCOME_ITEMS))]
    if rules.empty:
        return rules

    rules["antecedent"] = rules["antecedents"].map(
        lambda s: " AND ".join(sorted(_pretty(x) for x in s)))
    rules["n_conditions"] = rules["antecedents"].map(len)
    rules["outcome"] = rules["consequents"].map(lambda s: next(iter(s)).split("=", 1)[1])
    rules["n_animals"] = (rules["support"] * n_rows).round().astype(int)

    rules = rules[["outcome", "antecedent", "n_conditions", "support",
                   "confidence", "lift", "n_animals"]]
    rules = rules.drop_duplicates(subset=["outcome", "antecedent"])
    rules = rules.sort_values(["outcome", "lift"], ascending=[True, False])
    rules.to_csv(os.path.join(TABLES, "rules_by_outcome.csv"), index=False)
    if verbose:
        print(f"[rules] {len(rules):,} outcome rules with lift>={min_lift}")
    return rules


def support_sweep(onehot: pd.DataFrame, supports=(0.05, 0.02, 0.01, 0.005, 0.002),
                  max_len=3, verbose=True) -> pd.DataFrame:
    """
    How the itemset count responds to the support threshold, used to justify the
    chosen threshold rather than picking one arbitrarily.

    Run with FP-Growth: identical itemsets, a fraction of the cost. Apriori's
    own cost is measured separately in algorithm_scaling().
    """
    rows = []
    for s in supports:
        t0 = time.perf_counter()
        sets = fpgrowth(onehot, min_support=s, use_colnames=True, max_len=max_len)
        dt = time.perf_counter() - t0
        rows.append({
            "min_support": s,
            "n_itemsets": len(sets),
            "max_itemset_len": int(sets["itemsets"].map(len).max()) if len(sets) else 0,
            "seconds": round(dt, 2),
        })
        if verbose:
            print(f"[rules] support {s:<7} -> {len(sets):>7,} itemsets  ({dt:.1f}s)")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TABLES, "support_sweep.csv"), index=False)
    return table


def algorithm_scaling(onehot: pd.DataFrame, sizes=(20_000, 50_000, 100_000, None),
                      min_support=0.01, max_len=3, verbose=True) -> pd.DataFrame:
    """
    Apriori vs FP-Growth as the transaction count grows. This is a result, not
    an implementation note: it shows Apriori's repeated database scans costing
    more with every extra row while FP-Growth stays nearly flat.
    """
    rows = []
    for n in sizes:
        sub = onehot if n is None else onehot.iloc[:n]
        label = len(sub)
        rec = {"n_transactions": label}
        for name, fn in (("Apriori", apriori), ("FP-Growth", fpgrowth)):
            t0 = time.perf_counter()
            sets = fn(sub, min_support=min_support, use_colnames=True, max_len=max_len)
            rec[f"{name}_seconds"] = round(time.perf_counter() - t0, 2)
            rec["n_itemsets"] = len(sets)
        rec["speedup"] = round(rec["Apriori_seconds"] / max(rec["FP-Growth_seconds"], 1e-9), 2)
        rows.append(rec)
        if verbose:
            print(f"[rules] n={label:>7,}  apriori {rec['Apriori_seconds']:>7.1f}s  "
                  f"fpgrowth {rec['FP-Growth_seconds']:>6.1f}s  "
                  f"({rec['n_itemsets']:,} itemsets, {rec['speedup']}x)")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TABLES, "rule_algorithm_scaling.csv"), index=False)
    return table


def top_rules(rules: pd.DataFrame, per_outcome=6, min_support=0.002) -> pd.DataFrame:
    """Headline table: strongest rules per outcome, above a floor on support."""
    if rules.empty:
        return rules
    r = rules[rules["support"] >= min_support]
    out = (r.sort_values("lift", ascending=False)
             .groupby("outcome", observed=True)
             .head(per_outcome)
             .sort_values(["outcome", "lift"], ascending=[True, False]))
    out.to_csv(os.path.join(TABLES, "rules_top.csv"), index=False)
    return out


def run(df: pd.DataFrame, min_support=0.01, min_lift=1.2, max_len=3,
        scaling=True, verbose=True):
    onehot = encode(df, verbose=verbose)
    rules = mine(onehot, len(df), min_support=min_support, max_len=max_len,
                 min_lift=min_lift, verbose=verbose)
    top = top_rules(rules)
    sweep = support_sweep(onehot, max_len=max_len, verbose=verbose)
    sizes = (20_000, 50_000, 100_000, None) if scaling else (20_000, None)
    scale = algorithm_scaling(onehot, sizes=sizes, min_support=min_support,
                              max_len=max_len, verbose=verbose)
    return rules, top, sweep, scale


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load, stratified_sample

    df = stratified_sample(load(verbose=False), 20_000)
    rules, top, sweep, scale = run(df, min_support=0.01, scaling=False)
    pd.set_option("display.width", 220)
    print()
    print(top[["outcome", "antecedent", "support", "confidence", "lift", "n_animals"]]
          .to_string(index=False))
