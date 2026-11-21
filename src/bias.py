"""
Does colour, breed or age actually change an animal's fate?

The trap here is confounding. Black animals are not distributed evenly across
species, ages and intake routes, so a raw comparison of adoption rates measures
the wrong thing: it partly measures "what kind of animal tends to be black".

The fix used here is DIRECT STANDARDISATION, which is a descriptive adjustment
rather than a model, so it stays inside what the course covers and stays
auditable:

  1. split the population into strata - (animal_type x age_band x intake_type)
  2. compute the adoption rate for each colour WITHIN each stratum
  3. average those rates across strata, weighting each stratum by its share of
     the whole population

The result answers "what would this colour's adoption rate be if its animals
were distributed across species, ages and intake routes like everyone else".
Crude and standardised rates are both reported, because the gap between them is
itself the finding.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")

STRATA = ["animal_type", "age_band", "intake_type"]
MIN_CELL = 30          # ignore strata too small to estimate a rate from


def crude(df: pd.DataFrame, field: str) -> pd.DataFrame:
    g = df.groupby(field, observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "adoption_pct": g["outcome"].apply(lambda s: 100 * (s == "Adoption").mean()),
        "euthanasia_pct": g["outcome"].apply(lambda s: 100 * (s == "Euthanasia").mean()),
        "median_stay_days": g["length_of_stay_days"].median(),
    })
    return out.reset_index().rename(columns={field: "value"}).assign(field=field)


def standardised(df: pd.DataFrame, field: str, target="Adoption",
                 strata=STRATA, min_cell=MIN_CELL) -> pd.DataFrame:
    """
    Direct standardisation of the rate of `target` for each value of `field`,
    using the whole population's stratum distribution as the standard.
    """
    df = df.copy()
    df["_hit"] = (df["outcome"] == target).astype(float)
    df["_stratum"] = df[strata].astype(str).agg("|".join, axis=1)

    weights = df["_stratum"].value_counts(normalize=True)

    cell = (df.groupby(["_stratum", field], observed=True)["_hit"]
              .agg(["mean", "size"])
              .reset_index())
    cell = cell[cell["size"] >= min_cell]

    rows = []
    for value, grp in cell.groupby(field, observed=True):
        w = weights.reindex(grp["_stratum"]).to_numpy()
        if not np.isfinite(w).any() or w.sum() == 0:
            continue
        w = w / w.sum()
        rows.append({
            "field": field,
            "value": value,
            "n_in_strata": int(grp["size"].sum()),
            "n_strata": int(len(grp)),
            "standardised_pct": round(float((grp["mean"].to_numpy() * w).sum() * 100), 3),
        })
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame, field: str, target="Adoption") -> pd.DataFrame:
    """Crude next to standardised, so the confounding is visible."""
    c = crude(df, field)
    s = standardised(df, field, target=target)
    out = c.merge(s[["value", "standardised_pct", "n_strata"]], on="value", how="left")
    col = "adoption_pct" if target == "Adoption" else "euthanasia_pct"
    out["crude_pct"] = out[col]
    out["confounding_shift"] = (out["standardised_pct"] - out["crude_pct"]).round(3)
    out = out[["field", "value", "n", "crude_pct", "standardised_pct",
               "confounding_shift", "median_stay_days", "euthanasia_pct"]]
    return out.sort_values("standardised_pct", ascending=False)


def headline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Direct answers to the three claims people actually make, each as a
    like-for-like comparison against the rest of the same species.
    """
    rows = []

    def pair(label, mask, universe, note):
        a, b = df[mask & universe], df[~mask & universe]
        if len(a) < 50 or len(b) < 50:
            return
        rows.append({
            "claim": label,
            "group_n": len(a),
            "group_adoption_pct": round(100 * (a["outcome"] == "Adoption").mean(), 2),
            "others_adoption_pct": round(100 * (b["outcome"] == "Adoption").mean(), 2),
            "adoption_gap_pp": round(100 * ((a["outcome"] == "Adoption").mean()
                                            - (b["outcome"] == "Adoption").mean()), 2),
            "group_median_stay": round(a["length_of_stay_days"].median(), 2),
            "others_median_stay": round(b["length_of_stay_days"].median(), 2),
            "group_euthanasia_pct": round(100 * (a["outcome"] == "Euthanasia").mean(), 2),
            "others_euthanasia_pct": round(100 * (b["outcome"] == "Euthanasia").mean(), 2),
            "note": note,
        })

    cats = df["animal_type"] == "Cat"
    dogs = df["animal_type"] == "Dog"
    black = df["colour_group"] == "Black"

    pair("Black cats vs other cats", black, cats, "the black cat superstition")
    pair("Black dogs vs other dogs", black, dogs, "'black dog syndrome'")
    pair("Pit bull type vs other dogs", df["breed_group"] == "Pit Bull", dogs,
         "breed-label stigma")
    pair("Senior vs non-senior dogs", df["age_band"] == "Senior", dogs,
         "age bias")
    pair("Senior vs non-senior cats", df["age_band"] == "Senior", cats,
         "age bias")
    pair("Mixed breed vs purebred dogs", df["is_mix"] == "Yes", dogs,
         "mixed-breed stigma")

    return pd.DataFrame(rows)


def run(df: pd.DataFrame, verbose=True):
    os.makedirs(TABLES, exist_ok=True)

    parts = [compare(df, f) for f in
             ["colour_group", "breed_group", "age_band", "is_mix",
              "sterilised_at_intake", "intake_type", "intake_condition"]]
    table = pd.concat(parts, ignore_index=True)
    table.to_csv(os.path.join(TABLES, "bias_adoption.csv"), index=False)

    head = headline(df)
    head.to_csv(os.path.join(TABLES, "bias_headline.csv"), index=False)

    colour = compare(df, "colour_group")
    colour.to_csv(os.path.join(TABLES, "bias_colour.csv"), index=False)

    if verbose:
        pd.set_option("display.width", 220)
        print("[bias] headline claims:")
        print(head[["claim", "group_n", "group_adoption_pct", "others_adoption_pct",
                    "adoption_gap_pp", "group_median_stay", "others_median_stay"]]
              .to_string(index=False))
        print()
        print("[bias] adoption by colour, crude vs standardised:")
        print(colour[["value", "n", "crude_pct", "standardised_pct",
                      "confounding_shift", "median_stay_days"]].to_string(index=False))
    return table, head, colour


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load

    run(load(verbose=False))
