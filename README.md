# SecondChance

**Mining attribute bias in animal shelter outcomes**
Machine Learning and Data Mining — project pipeline

Which animal attributes actually determine shelter outcome, and can an adverse
outcome be anticipated at intake?

```bash
pip install -r requirements.txt
python run_all.py --quick     # ~5 min, end-to-end check
python run_all.py             # the real run
```

Output: `results/tables/*.csv`, `results/plots/*.png`, `results/SUMMARY.md`.

---

## Data

Two public CSVs from the City of Austin open data portal. **No account, no API
key** — the loader downloads and caches them on first run.

| | |
|---|---|
| Outcomes | 173,775 × 12 |
| Intakes | 173,812 × 12 |
| Join | on `Animal ID`, **99.1% coverage** |
| After scoping to dogs and cats | 162,461 outcome events |
| Date range | Oct 2013 – May 2025 |
| Outcome label | Adoption 83,866 / Transfer 46,668 / Return to Owner 25,466 / Euthanasia 4,704 / Other 1,757 |

### Three things that will bite you

1. **`DateTime` has mixed timezone offsets** in both files. Parse with
   `utc=True` or you get `can't compare offset-naive and offset-aware datetimes`.

2. **10% of Animal IDs recur in Intakes** — animals that come back. The join
   must be per *outcome event*, matched to the most recent intake at or before
   that outcome (`pandas.merge_asof(direction="backward")`), not a naive merge.

3. **`Sex upon Outcome` leaks the label.** It encodes sterilisation status, and
   shelters routinely sterilise animals *as part of preparing them for
   adoption*. Using it as a feature is circular. All features come from the
   **intake** columns instead, which is also what "predict at intake" honestly
   means.

---

## Structure

| File | What it does |
|---|---|
| `src/data.py` | download, intake join, breed/colour/age/sex normalisation |
| `src/rules.py` | Apriori + FP-Growth, lift ranking, support sweep |
| `src/supervised.py` | the five classifiers, random *and* temporal protocols |
| `src/torchnn.py` | PyTorch network: categorical embeddings, class-weighted loss |
| `src/compare.py` | association rules vs decision tree paths |
| `src/tool.py` | outcome-risk CLI, returns the rule that fired |
| `src/clustering.py` | the five clustering algorithms on Hamming dissimilarity |
| `src/bias.py` | crude and standardised adoption rates, the headline claims |
| `src/plots.py` | the twelve report figures |
| `run_all.py` | runs everything, writes `results/SUMMARY.md` |

---

## Design decisions worth defending in the viva

**Hamming, not Euclidean.** Every feature is categorical, so Euclidean distance
is meaningless. Hamming — the fraction of the 11 attributes on which two animals
differ — is the right measure. It takes only 12 distinct values, which makes
`eps` directly interpretable: *eps = 2/11 means "animals differing in at most 2
attributes are neighbours"*.

**K-Means is in the comparison on purpose, to lose.** K-Medoids, Hierarchical,
DBSCAN and OPTICS all accept a precomputed dissimilarity matrix, so all four run
on Hamming. K-Means cannot — it computes means, which needs Euclidean geometry,
and the mean of categorical vectors is not an animal. It runs on one-hot data as
the contrast that shows why medoid-based clustering exists. A medoid is a real
animal you can look up; a centroid is not.

**Two evaluation protocols, and the gap is a result.** A random split lets a
model learn period-specific shelter policy and take credit for it — Austin's
intake collapsed from 19,775 in 2019 to 9,778 in 2020. The temporal holdout
(train ≤ 2021, test ≥ 2022) is the honest number, and it is lower.

**Lift, not confidence.** Adoption is ~52% of records, so a rule can reach 0.6
confidence while saying nothing. Lift is what shows the antecedent moved the odds.

**Direct standardisation, not regression.** Regression is not on the syllabus,
and it isn't needed: splitting into (species × age × intake route) strata,
computing rates within each, then reweighting to the whole population's stratum
mix is transparent, auditable, and answers the same question.

**Clustering subsamples, and says so.** A dissimilarity matrix over 162,461
records would be ~105 GB. The metric-based algorithms run on a stratified 6,000;
hierarchical and OPTICS on 3,000. That constraint is reported as a result, not
hidden — it is the reason some algorithms can see all the data and others cannot.

**PyTorch is there for two things sklearn cannot do.** `MLPClassifier` has no
`class_weight` parameter — check it with `inspect.signature` — which is exactly
why it scores the best accuracy (0.628) and nearly the worst balanced accuracy
(0.373). A weighted cross-entropy loss fixes that, lifting balanced accuracy to
**0.486**, the highest in the study. The second reason is learned embeddings:
one-hot puts Labrador as far from Golden Retriever as from Siamese. The
*unweighted* torch net lands within a few thousandths of the sklearn MLP, and
that agreement is the control proving the implementation is sound. Skip it with
`--no-torch`.

---

## Findings

**The folk belief is wrong, and the thing nobody talks about is huge.**

- **"Black dog syndrome" is not supported here.** Black dogs are adopted *more*
  than other dogs (+2.77 pp). Black cats are adopted slightly less (−2.25 pp)
  and wait about a day longer. After standardising for species, age and intake
  route, black sits mid-table — **white animals fare worst**, not black ones.
- **Age dominates everything.** Senior dogs are adopted 23.7 pp less often than
  non-seniors; senior cats 10.3 pp less.
- **Breed label costs time more than it costs adoptions.** Pit bull type dogs
  are adopted 4.3 pp less — but wait a median of **10.0 days against 5.3**.
- **Mixed-breed stigma is not visible** — mixed dogs are adopted *more*.

**The atypicality hypothesis holds.** Animals that DBSCAN labels as noise —
unusual attribute combinations, identified with no labels at all — are
euthanised at roughly **1.9×** the rate of clustered animals, and adopted less
often. OPTICS independently agrees. Labels are used only to score the finding,
never to produce it.

**Apriori's cost is visible.** On identical itemsets, Apriori takes 113s at
162k transactions against FP-Growth's 26s, and the gap widens with n — Apriori
rescans the database once per candidate level.

**Clustering does not recover outcome classes** (ARI ≈ 0.02–0.04). Attribute
clusters are not outcome clusters. That is a real negative result and belongs in
the report as one, rather than being tuned until it looks better.

---

## A guard worth keeping

`clustering.atypicality()` marks a row `reliable=False` when fewer than 30
animals are labelled noise. Without it, a setting that produces 3 noise points
cheerfully reports a "66.7% euthanasia rate" — which is two animals. Figures and
the summary use only reliable rows.

---

## Splitting the work

1. **Data** — Sambhav Ghildiyal — `src/data.py`: the join, the normalisation map, EDA, descriptive figures
2. **Association rules** — Aniket Pandey — `src/rules.py`: support threshold selection, lift
   interpretation, Apriori vs FP-Growth
3. **Classification** — Aditya Bansal — `src/supervised.py`: the five models, imbalance, the
   two protocols, tree paths vs mined rules
4. **Clustering** — Mudit Maheshwari — `src/clustering.py`: Hamming dissimilarity, the metric
   argument, the atypicality test

Everyone shares `src/bias.py`, the report and the presentation.
