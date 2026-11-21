"""
Austin Animal Center: download, join intakes to outcomes, and normalise.

Two public CSVs, no account needed:
  Outcomes  173,775 x 12
  Intakes   173,812 x 12   (joins on Animal ID at ~99.5% coverage)

Three things here are not obvious and matter:

1. DateTime carries MIXED TIMEZONE OFFSETS in both files. Parsing without
   utc=True raises "can't compare offset-naive and offset-aware datetimes".

2. About 10% of Animal IDs appear more than once in Intakes - animals that come
   back. The join must therefore be per OUTCOME EVENT, matched to the most
   recent intake at or before that outcome, not a naive one-row-per-animal
   merge. pandas.merge_asof(direction="backward") does exactly this.

3. LEAKAGE: "Sex upon Outcome" encodes sterilisation status, and shelters
   routinely sterilise animals *as part of* preparing them for adoption. Using
   it as a feature leaks the outcome. We therefore build features from the
   INTAKE columns only (Sex upon Intake, Age upon Intake, Intake Type, Intake
   Condition), which is also what "predict at intake" honestly means.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
TABLES = os.path.join(ROOT, "results", "tables")

OUTCOMES_URL = "https://data.austintexas.gov/api/views/9t4d-g238/rows.csv?accessType=DOWNLOAD"
INTAKES_URL = "https://data.austintexas.gov/api/views/wter-evkm/rows.csv?accessType=DOWNLOAD"
UA = {"User-Agent": "student-coursework-script/1.0"}

OUTCOME_CLASSES = ["Adoption", "Transfer", "Return to Owner", "Euthanasia", "Other"]
ADVERSE = {"Euthanasia"}

# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
def _fetch(url: str, name: str, verbose=True) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        import requests
        if verbose:
            print(f"[data] downloading {name} ...", end="", flush=True)
        r = requests.get(url, timeout=300, headers=UA)
        r.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(r.content)
        if verbose:
            print(f" {len(r.content)/1e6:.1f} MB")
    return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
# Breed families, applied in order. First match wins, so put specific before
# general (e.g. "australian cattle" before "cattle").
BREED_RULES = [
    # cats
    ("Domestic Shorthair", r"domestic shorthair|american shorthair"),
    ("Domestic Mediumhair", r"domestic medium ?hair"),
    ("Domestic Longhair", r"domestic longhair"),
    ("Siamese", r"siamese|snowshoe|balinese"),
    ("Cat (other pedigree)", r"maine coon|manx|russian blue|himalayan|persian|"
                             r"bengal|ragdoll|bombay|angora|tonkinese|burmese|"
                             r"abyssinian|british shorthair|sphynx|cornish rex|devon rex"),
    # dogs - bully group
    ("Pit Bull", r"pit bull|staffordshire|american bully"),
    ("Boxer", r"boxer"),
    ("Rottweiler", r"rottweiler"),
    ("Bulldog", r"bulldog"),
    # dogs - common
    ("Labrador Retriever", r"labrador"),
    ("Chihuahua", r"chihuahua"),
    ("German Shepherd", r"german shepherd"),
    # "Queensland heeler" and "blue heeler" are the same breed as the
    # Australian cattle dog; the shelter's free-text field uses all three.
    ("Australian Cattle Dog", r"australian cattle|blue heeler|queensland heeler"),
    ("Australian Kelpie", r"kelpie"),
    # Texas working dogs, locally very common and otherwise a large "Other"
    ("Cur / Lacy / Carolina", r"\bcur\b|black mouth cur|lacy|carolina dog"),
    ("Doberman", r"doberman"),
    ("Great Dane", r"great dane"),
    ("Mastiff", r"mastiff"),
    ("Chow Chow", r"chow chow"),
    ("Shar Pei", r"sharpei|shar pei"),
    ("Welsh Corgi", r"corgi"),
    ("Cocker Spaniel", r"cocker spaniel"),
    ("Rhodesian Ridgeback", r"ridgeback"),
    ("Whippet / Greyhound", r"whippet|greyhound"),
    ("Akita", r"akita"),
    ("Basenji", r"basenji"),
    ("Weimaraner", r"weimaraner"),
    ("Dalmatian", r"dalmatian"),
    ("Toy breed", r"maltese|\bpug\b|pomeranian|lhasa apso|pekingese|bichon|"
                  r"papillon|havanese|chinese crested|japanese chin|"
                  r"italian greyhound|shiba inu|toy fox"),
    ("Dachshund", r"dachshund"),
    ("Border Collie", r"border collie"),
    ("Australian Shepherd", r"australian shepherd"),
    ("Poodle", r"poodle"),
    ("Beagle", r"beagle"),
    ("Siberian Husky", r"husky"),
    ("Catahoula", r"catahoula"),
    ("Great Pyrenees", r"pyrenees"),
    ("Shih Tzu", r"shih tzu"),
    ("Yorkshire Terrier", r"yorkshire"),
    ("Jack Russell Terrier", r"jack russell|russell terr"),
    ("Miniature Pinscher", r"miniature pinscher"),
    ("Pointer", r"pointer"),
    ("Schnauzer", r"schnauzer"),
    ("Cairn Terrier", r"cairn terrier"),
    ("Shepherd (other)", r"shepherd|collie|malinois|sheepdog"),
    ("Retriever (other)", r"retriever"),
    ("Spaniel / Setter (other)", r"spaniel|setter|vizsla|brittany"),
    ("Hound", r"hound|coonhound|basset|harrier|dachshund|beagle|"
              r"foxhound|bluetick|plott|redbone"),
    ("Terrier (other)", r"terrier|schnauzer"),
]

COLOUR_PATTERNS = ["tabby", "brindle", "tricolor", "calico", "torbie", "tortie",
                   "point", "merle", "smoke", "tiger", "sable", "agouti", "lynx"]

COLOUR_RULES = [
    ("Black", r"^black$|^black "),
    ("White", r"^white$|^white "),
    ("Brown", r"brown|chocolate|liver|seal"),
    ("Tan/Fawn", r"tan|fawn|buff|apricot|gold|yellow"),
    ("Gray/Blue", r"gray|grey|blue|silver"),
    ("Orange/Red", r"orange|red|flame|ruddy"),
    ("Cream", r"cream|lilac|pink"),
    ("Calico/Tortie", r"calico|tortie|torbie"),
    ("Tricolor", r"tricolor"),
]

AGE_RE = re.compile(r"(-?\d+)\s*(day|week|month|year)", re.I)
UNIT_DAYS = {"day": 1, "week": 7, "month": 30.44, "year": 365.25}


def norm_breed(raw: str) -> tuple[str, bool]:
    """Return (breed_group, is_mix)."""
    if not isinstance(raw, str):
        return "Unknown", False
    s = raw.strip().lower()
    is_mix = ("mix" in s) or ("/" in s)
    s = s.split("/")[0]
    s = re.sub(r"\bmix\b", "", s).strip()
    for group, pattern in BREED_RULES:
        if re.search(pattern, s):
            return group, is_mix
    return "Other", is_mix


def norm_colour(raw: str) -> tuple[str, str, bool]:
    """Return (colour_group, pattern, is_multi)."""
    if not isinstance(raw, str):
        return "Unknown", "none", False
    s = raw.strip().lower()
    is_multi = "/" in s
    primary = s.split("/")[0].strip()
    pattern = "none"
    for p in COLOUR_PATTERNS:
        if p in s:
            pattern = p
            break
    base = primary
    for p in COLOUR_PATTERNS:
        base = base.replace(p, "").strip()
    if not base:
        base = primary
    for group, rule in COLOUR_RULES:
        if re.search(rule, base):
            return group, pattern, is_multi
    return "Other", pattern, is_multi


def age_to_days(raw) -> float:
    if not isinstance(raw, str):
        return np.nan
    m = AGE_RE.search(raw)
    if not m:
        return np.nan
    n, unit = int(m.group(1)), m.group(2).lower()
    days = n * UNIT_DAYS[unit]
    return days if days >= 0 else np.nan


def age_band(days: float) -> str:
    if not np.isfinite(days):
        return "Unknown"
    if days < 60:
        return "Neonate"
    if days < 180:
        return "Juvenile"
    if days < 730:
        return "Young adult"
    if days < 2555:          # ~7 years
        return "Adult"
    return "Senior"


def split_sex(raw) -> tuple[str, str]:
    """'Neutered Male' -> ('Male', 'Yes'). Returns (sex, sterilised)."""
    if not isinstance(raw, str):
        return "Unknown", "Unknown"
    s = raw.strip().lower()
    sex = "Male" if "male" in s and "female" not in s else ("Female" if "female" in s else "Unknown")
    if "neutered" in s or "spayed" in s:
        ster = "Yes"
    elif "intact" in s:
        ster = "No"
    else:
        ster = "Unknown"
    return sex, ster


def norm_outcome(raw) -> str:
    if not isinstance(raw, str):
        return "Other"
    s = raw.strip()
    if s in ("Adoption", "Rto-Adopt"):
        return "Adoption"
    if s == "Transfer":
        return "Transfer"
    if s == "Return to Owner":
        return "Return to Owner"
    if s == "Euthanasia":
        return "Euthanasia"
    return "Other"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build(animals=("Dog", "Cat"), verbose=True) -> pd.DataFrame:
    out = _fetch(OUTCOMES_URL, "outcomes.csv", verbose)
    inn = _fetch(INTAKES_URL, "intakes.csv", verbose)

    out["outcome_dt"] = pd.to_datetime(out["DateTime"], errors="coerce",
                                       format="mixed", utc=True)
    inn["intake_dt"] = pd.to_datetime(inn["DateTime"], errors="coerce",
                                      format="mixed", utc=True)

    out = out.dropna(subset=["outcome_dt"]).sort_values("outcome_dt")
    inn = inn.dropna(subset=["intake_dt"]).sort_values("intake_dt")

    keep_in = ["Animal ID", "intake_dt", "Intake Type", "Intake Condition",
               "Sex upon Intake", "Age upon Intake", "Breed", "Color", "Animal Type"]
    keep_out = ["Animal ID", "outcome_dt", "Outcome Type"]

    # most recent intake at or before each outcome event
    merged = pd.merge_asof(
        out[keep_out], inn[keep_in],
        left_on="outcome_dt", right_on="intake_dt",
        by="Animal ID", direction="backward",
    )
    if verbose:
        cov = merged["intake_dt"].notna().mean()
        print(f"[data] join coverage: {cov:.3%} of {len(merged):,} outcome events")

    df = merged.dropna(subset=["intake_dt"]).copy()
    df = df[df["Animal Type"].isin(animals)].copy()

    df["length_of_stay_days"] = (
        (df["outcome_dt"] - df["intake_dt"]).dt.total_seconds() / 86400.0
    )

    breed = df["Breed"].map(norm_breed)
    df["breed_group"] = [b for b, _ in breed]
    df["is_mix"] = ["Yes" if m else "No" for _, m in breed]

    colour = df["Color"].map(norm_colour)
    df["colour_group"] = [c for c, _, _ in colour]
    df["colour_pattern"] = [p for _, p, _ in colour]
    df["colour_multi"] = ["Yes" if m else "No" for _, _, m in colour]

    df["age_days"] = df["Age upon Intake"].map(age_to_days)
    df["age_band"] = df["age_days"].map(age_band)

    sexes = df["Sex upon Intake"].map(split_sex)
    df["sex"] = [s for s, _ in sexes]
    df["sterilised_at_intake"] = [t for _, t in sexes]

    df["intake_type"] = df["Intake Type"].fillna("Unknown")
    df["intake_condition"] = df["Intake Condition"].fillna("Unknown")
    df["animal_type"] = df["Animal Type"]
    df["outcome"] = df["Outcome Type"].map(norm_outcome)
    df["adverse"] = df["outcome"].isin(ADVERSE)
    df["year"] = df["outcome_dt"].dt.year
    df["month"] = df["outcome_dt"].dt.month

    cols = ["Animal ID", "outcome_dt", "intake_dt", "length_of_stay_days", "year", "month",
            "animal_type", "breed_group", "is_mix", "colour_group", "colour_pattern",
            "colour_multi", "age_band", "age_days", "sex", "sterilised_at_intake",
            "intake_type", "intake_condition", "outcome", "adverse"]
    df = df[cols].reset_index(drop=True)

    if verbose:
        print(f"[data] {len(df):,} rows after scoping to {animals}")
    return df


FEATURES = ["animal_type", "breed_group", "is_mix", "colour_group", "colour_pattern",
            "colour_multi", "age_band", "sex", "sterilised_at_intake",
            "intake_type", "intake_condition"]


def load(animals=("Dog", "Cat"), use_cache=True, verbose=True) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    tag = "_".join(a.lower() for a in animals)
    path = os.path.join(CACHE, f"clean_{tag}.parquet")
    if use_cache and os.path.exists(path):
        if verbose:
            print(f"[data] cache hit {os.path.basename(path)}")
        return pd.read_parquet(path)
    df = build(animals, verbose)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:        # pyarrow may be absent
        if verbose:
            print(f"[data] parquet cache unavailable ({type(exc).__name__}); continuing")
    return df


def write_normalisation_map(df: pd.DataFrame):
    """Auditable record of what the normalisation did, for the report."""
    os.makedirs(TABLES, exist_ok=True)
    rows = []
    for col in ["breed_group", "colour_group", "colour_pattern", "age_band",
                "sex", "sterilised_at_intake", "intake_type", "intake_condition",
                "outcome"]:
        vc = df[col].value_counts()
        for value, n in vc.items():
            rows.append({"field": col, "value": value, "count": int(n),
                         "share_pct": round(100 * n / len(df), 3)})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TABLES, "normalisation_map.csv"), index=False)
    return table


def stratified_sample(df: pd.DataFrame, n: int, seed=42) -> pd.DataFrame:
    """Outcome-stratified sample, used for the clustering tiers."""
    if len(df) <= n:
        return df
    frac = n / len(df)
    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in df.groupby("outcome", observed=True):
        want = max(2, min(len(grp), int(round(len(grp) * frac))))
        parts.append(grp.iloc[rng.choice(len(grp), want, replace=False)])
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = load(use_cache=False)
    print()
    print(df[["animal_type", "breed_group", "colour_group", "age_band",
              "intake_type", "intake_condition", "outcome"]].head(8).to_string(index=False))
    print()
    print("outcome distribution:")
    print(df["outcome"].value_counts().to_string())
    print()
    print("length of stay (days):")
    print(df["length_of_stay_days"].describe().round(2).to_string())
    print()
    print("negative stays (should be 0):", int((df["length_of_stay_days"] < 0).sum()))
    print("breed 'Other' share  :", round(100 * (df.breed_group == "Other").mean(), 1), "%")
    print("colour 'Other' share :", round(100 * (df.colour_group == "Other").mean(), 1), "%")
    print("age 'Unknown' share  :", round(100 * (df.age_band == "Unknown").mean(), 1), "%")
    print()
    print("SANITY: euthanasia rate by intake condition")
    t = df.groupby("intake_condition", observed=True)["outcome"].apply(
        lambda s: (s == "Euthanasia").mean() * 100).sort_values(ascending=False)
    print(t.round(2).head(10).to_string())
    write_normalisation_map(df)
