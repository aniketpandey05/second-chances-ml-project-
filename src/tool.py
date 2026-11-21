"""
Outcome-risk tool.

Takes an animal's intake attributes and returns a predicted outcome, the
probability of each outcome, and - the point of the whole thing - the REASON:
the decision path that fired, plus any mined association rules that cover this
animal.

A shelter manager cannot act on a risk score. They can act on "this animal is
in the low-adoption group because it is a senior dog surrendered by its owner",
because that suggests what to change: the photograph, the kennel card, a
transfer request, a foster placement.

Deliberately a CLI rather than a Streamlit app: no web server, no extra
dependency, and it runs anywhere Python does.

    python src/tool.py --type Dog --breed "Pit Bull" --age Adult \
                       --colour Black --intake Stray --condition Normal

    python src/tool.py --list          # show the valid values for every field
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import FEATURES, load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")

ARG_TO_FIELD = {
    "type": "animal_type", "breed": "breed_group", "mixed": "is_mix",
    "colour": "colour_group", "pattern": "colour_pattern",
    "multicolour": "colour_multi", "age": "age_band", "sex": "sex",
    "sterilised": "sterilised_at_intake", "intake": "intake_type",
    "condition": "intake_condition",
}
FIELD_TO_ARG = {v: k for k, v in ARG_TO_FIELD.items()}


class RiskModel:
    def __init__(self, df: pd.DataFrame, max_depth=6, min_samples_leaf=200, seed=42):
        self.df = df
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                 dtype=np.float32)
        X = self.enc.fit_transform(df[FEATURES])
        self.y = df["outcome"].to_numpy()
        self.names = list(self.enc.get_feature_names_out(FEATURES))
        self.clf = DecisionTreeClassifier(
            criterion="entropy", max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, random_state=seed).fit(X, self.y)
        self.base = {c: float((self.y == c).mean()) for c in self.clf.classes_}
        self.defaults = {f: df[f].mode().iloc[0] for f in FEATURES}
        self.valid = {f: sorted(df[f].astype(str).unique()) for f in FEATURES}

    # -- explanation ------------------------------------------------------
    def _path(self, x_row) -> list[str]:
        t = self.clf.tree_
        node, out = 0, []
        while t.children_left[node] != -1:
            feat = t.feature[node]
            name = self.names[feat]
            field, value = self._split(name)
            if x_row[feat] <= 0.5:
                out.append(f"{FIELD_TO_ARG.get(field, field)} is NOT {value}")
                node = t.children_left[node]
            else:
                out.append(f"{FIELD_TO_ARG.get(field, field)} is {value}")
                node = t.children_right[node]
        return out

    @staticmethod
    def _split(onehot_name: str):
        for f in sorted(FEATURES, key=len, reverse=True):
            if onehot_name.startswith(f + "_"):
                return f, onehot_name[len(f) + 1:]
        return onehot_name, ""

    def matching_rules(self, animal: dict, top=5) -> pd.DataFrame:
        path = os.path.join(TABLES, "rules_top.csv")
        if not os.path.exists(path):
            return pd.DataFrame()
        rules = pd.read_csv(path)
        items = {f"{FIELD_TO_ARG.get(f, f)}={v}" for f, v in animal.items()}
        keep = rules["antecedent"].map(
            lambda a: set(x.strip() for x in a.split(" AND ")) <= items)
        return rules[keep].sort_values("lift", ascending=False).head(top)

    # -- prediction -------------------------------------------------------
    def assess(self, animal: dict) -> dict:
        full = {**self.defaults, **animal}
        row = pd.DataFrame([full])[FEATURES]
        X = self.enc.transform(row)
        proba = self.clf.predict_proba(X)[0]
        order = np.argsort(proba)[::-1]
        classes = self.clf.classes_
        return {
            "animal": full,
            "supplied": animal,
            "prediction": classes[order[0]],
            "probabilities": {classes[i]: round(float(proba[i]), 4) for i in order},
            "lift_vs_base": {classes[i]: round(float(proba[i] / self.base[classes[i]]), 2)
                             for i in order if self.base[classes[i]] > 0},
            "path": self._path(X[0]),
            "rules": self.matching_rules(full),
        }


def render(result: dict):
    a, sup = result["animal"], result["supplied"]
    print()
    print("=" * 70)
    desc = ", ".join(f"{FIELD_TO_ARG.get(f, f)}={a[f]}" for f in FEATURES
                     if f in sup) or "(all defaults)"
    print(f"ANIMAL  {desc}")
    unspecified = [FIELD_TO_ARG.get(f, f) for f in FEATURES if f not in sup]
    if unspecified:
        print(f"        (unspecified, filled with the most common value: "
              f"{', '.join(unspecified)})")
    print("=" * 70)

    print(f"\nMOST LIKELY OUTCOME: {result['prediction']}\n")
    print(f"  {'outcome':<18}{'probability':>12}{'vs baseline':>14}")
    for outcome, p in result["probabilities"].items():
        lift = result["lift_vs_base"].get(outcome)
        flag = ""
        if lift and lift >= 1.5:
            flag = "  <-- elevated"
        elif lift and lift <= 0.6:
            flag = "  <-- reduced"
        print(f"  {outcome:<18}{100*p:>11.1f}%{(f'{lift:.2f}x' if lift else '-'):>14}{flag}")

    # plain ASCII on purpose: the Windows console is cp1252 and turns an
    # em-dash into a replacement character
    print("\nWHY - the decision path that fired:")
    print("  (a one-hot tree splits on 'is NOT x' as readily as 'is x', so most")
    print("   steps are exclusions; that asymmetry is why Apriori cannot match")
    print("   every tree path - see results/tables/rules_vs_tree.csv)")
    for i, step in enumerate(result["path"], 1):
        print(f"  {i}. {step}")
    if not result["path"]:
        print("  (the tree assigns this animal at the root)")

    rules = result["rules"]
    if len(rules):
        print("\nMINED ASSOCIATION RULES COVERING THIS ANIMAL:")
        for _, r in rules.iterrows():
            print(f"  {r['antecedent']}")
            print(f"      -> {r['outcome']}   lift {r['lift']:.2f}, "
                  f"confidence {100*r['confidence']:.1f}%, n={r['n_animals']:,}")
    else:
        print("\nNo mined rule covers this exact combination "
              "(run run_all.py first, or the combination is below the support floor).")
    print()


def main():
    ap = argparse.ArgumentParser(description="Shelter outcome risk, with the reason.")
    for arg in ARG_TO_FIELD:
        ap.add_argument(f"--{arg}", default=None)
    ap.add_argument("--list", action="store_true", help="show valid values and exit")
    args = ap.parse_args()

    df = load(verbose=False)
    model = RiskModel(df)

    if args.list:
        print("\nValid values per field:\n")
        for f in FEATURES:
            vals = model.valid[f]
            shown = ", ".join(vals[:14]) + (" ..." if len(vals) > 14 else "")
            print(f"  --{FIELD_TO_ARG.get(f, f):<13} {shown}")
        print()
        return

    animal = {ARG_TO_FIELD[k]: v for k, v in vars(args).items()
              if k in ARG_TO_FIELD and v is not None}
    for f, v in animal.items():
        if v not in model.valid[f]:
            print(f"\n'{v}' is not a valid {FIELD_TO_ARG.get(f, f)}. "
                  f"Try: {', '.join(model.valid[f][:10])} ...\n")
            return
    render(model.assess(animal))


if __name__ == "__main__":
    main()
