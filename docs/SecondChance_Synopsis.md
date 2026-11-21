# SecondChance

**Mining Attribute Bias in Animal Shelter Outcomes with Association Rules and Density-Based Clustering**

Project Synopsis — Machine Learning and Data Mining
Jaypee Institute of Information Technology, Noida

Submitted to: Dr. Ashish Mishra
Submitted by: Sambhav Ghildiyal (23803005), Aniket Pandey (23803008), Aditya Bansal (23803009), Mudit Maheshwari (23803017)

November 2026

---

## Abstract

Municipal animal shelters decide, under permanent capacity pressure, which animals are promoted for adoption, transferred to partner organisations, returned to owners, or euthanised. Practitioners hold strong beliefs about what determines an animal's fate: that black animals are passed over, that bully-breed labels are disqualifying, that senior animals are effectively unadoptable. These beliefs already shape policy, from shelter photography guidelines to the removal of breed labels from kennel cards. They are rarely quantified at scale.

**SecondChance** is an end-to-end data mining study of **173,775 outcome records** published by Austin Animal Center. The project will (i) mine association rules over animal attributes to surface interpretable outcome patterns; (ii) benchmark five classification algorithms predicting an animal's outcome under a leakage-safe temporal split; (iii) segment the population with five clustering algorithms, using dissimilarity measures appropriate to categorical data rather than defaulting to Euclidean distance; (iv) test whether animals that density-based clustering labels as noise — atypical animals — experience measurably worse outcomes; and (v) quantify the marginal effect of colour, breed group and age on adoption while controlling for the remaining attributes.

Deliverables: a reproducible pipeline, a comparative study of the classical data mining toolkit on a large categorical dataset, and an interpretable outcome-risk tool that returns a prediction together with the rule that produced it.

## Problem Statement and Motivation

Shelter intake exceeds adoption capacity almost everywhere, so triage is unavoidable and its consequences are terminal. Staff decide which animals receive foster placement, promotion or extended holding time, and those decisions are guided largely by experience and folklore rather than by measurement. The widely cited "black dog syndrome" — the claim that dark-coated animals wait longer and are adopted less often — is representative: believed strongly enough to drive operational policy, yet reported findings are mixed and most studies cover a single shelter over a short window.

A public dataset of 173,775 outcomes over more than a decade makes the question tractable. Two questions are open:

1. **Which attributes actually determine an animal's outcome, and by how much**, once the remaining attributes are controlled for?
2. **Can an adverse outcome be anticipated early enough to act on it**, using only information recorded when the animal enters the shelter?

Interpretability is a requirement rather than a preference. A shelter manager cannot act on a risk score; they need the reason behind it in order to change something — a photograph, a kennel card, a transfer request. This makes decision trees and association rules the appropriate instruments, and makes the comparison against less transparent models a genuine trade-off to measure rather than a formality.

## Objectives

1. Build a reproducible pipeline and exploratory analysis: outcome distribution, trends over time, and normalisation of 3,010 free-text breed strings and 663 colour strings into analysable categories.
2. Mine association rules over animal attributes using Apriori, ranking patterns by lift rather than confidence alone, and report the rule sets characterising each outcome class.
3. Benchmark Decision Tree, Naive Bayes, KNN, SVM and a Neural Network under one protocol, reporting macro-averaged F1 because the outcome classes are strongly imbalanced.
4. Segment the population with K-Means, K-Medoids, Hierarchical, DBSCAN and OPTICS using a categorical dissimilarity measure, and compare internal and external cluster validity.
5. Test the **atypicality hypothesis**: are animals labelled as noise by DBSCAN and OPTICS over-represented among adverse outcomes?
6. Quantify the marginal effect of colour, breed group, age and sterilisation status on adoption probability, and deliver an interpretable outcome-risk tool.

## Dataset

| Property | Austin Animal Center Outcomes |
|---|---|
| Records | 173,775 outcome events, 2013 to present, updated continuously |
| Features | 12 columns: animal type, breed (3,010 distinct strings), colour (663), sex and sterilisation status, age at outcome, date of birth, outcome timestamp |
| Label | Outcome type — Adoption 84,598; Transfer 48,689; Return to Owner 25,691; Euthanasia 10,833; remaining categories minor |
| Access | Single CSV from the City of Austin open data portal, 21 MB, no account or API key required |
| Extension | The companion *Intakes* dataset joins on Animal ID, adding intake condition and type, and enables length of stay as a second prediction target |

Small enough to process entirely on a laptop CPU; no GPU required at any stage.

## Proposed Methodology

`Ingest & EDA` → `Rule Mining` → `Classification` → `Clustering` → `Bias & Tool`

### 1. Preprocessing and Feature Engineering

Breed strings are normalised into approximately twenty groups with a separate mixed-breed indicator; colour strings are decomposed into primary colour and pattern; age at outcome is parsed from free text into days; the combined sex field is split into sex and sterilisation status, which are distinct predictors. Outcome month and season are derived, since shelter intake is strongly seasonal. Every normalisation rule is recorded in a mapping table reproduced in the report, so the transformation is auditable rather than implicit.

### 2. Association Rule Mining

Each animal is represented as a transaction over its attributes, e.g. `{dog, pit_bull_mix, black, adult, intact_male}`. Apriori is applied across a range of minimum support thresholds selected from the item frequency distribution rather than chosen arbitrarily. Rules are ranked by **lift**, since confidence alone is misleading when one outcome dominates. The resulting rule sets are compared directly against the paths learned by the decision tree — an independent route to the same kind of structure.

### 3. Classification

Decision Tree, Naive Bayes, KNN, SVM (linear and RBF) and a feedforward neural network trained with backpropagation are evaluated on an identical test set. Categorical attributes are one-hot encoded; class weighting addresses imbalance. Evaluation uses stratified k-fold cross-validation together with a **temporal holdout** — training on earlier years and testing on later ones — because a random split would let the model learn period-specific shelter policy and report an optimistic result.

### 4. Clustering with Categorical Dissimilarity

The attributes are categorical, so Euclidean distance is not meaningful and the choice of dissimilarity is a methodological decision rather than a detail. K-Medoids, Hierarchical, DBSCAN and OPTICS all accept an arbitrary precomputed dissimilarity matrix and are run with Hamming and Jaccard measures. **K-Means cannot**: it computes means, which requires Euclidean geometry, and the mean of categorical vectors is not an animal. K-Means on one-hot encoded data is therefore included deliberately as the contrast demonstrating why medoid-based clustering exists. K-Medoids is implemented directly, since `scikit-learn-extra` is unmaintained and does not build on current Python versions.

### 5. Bias Quantification and Outcome-Risk Tool

The marginal effect of each attribute on adoption probability is estimated while holding the remaining attributes fixed, which separates genuine attribute effects from confounding — black animals are disproportionately common among certain breed groups, so an unadjusted comparison measures the wrong thing. The final deliverable accepts an animal's attributes and returns an outcome risk together with the association rule or decision path responsible for it.

## Evaluation Metrics

**Macro-averaged F1** is the primary classification metric, because accuracy is uninformative when one outcome accounts for roughly half of all records. Per-class precision and recall, balanced accuracy and the Matthews Correlation Coefficient are also reported. Association rules are assessed by support, confidence and **lift**. Clustering is assessed externally by Adjusted Rand Index and Normalised Mutual Information against outcome labels, internally by silhouette on the dissimilarity matrix, and by the number of clusters near-pure in a single outcome class. All results are reported as mean ± standard deviation over five random seeds, alongside the temporal holdout result.

## Tools and Technology Stack

| Layer | Tools |
|---|---|
| Language & environment | Python 3.13, Jupyter, Git / GitHub |
| Data processing | pandas, NumPy |
| Association rules | mlxtend (Apriori, FP-Growth) |
| Machine learning | scikit-learn (trees, Naive Bayes, KNN, SVM, MLP, clustering) |
| Clustering support | SciPy (linkage, dendrograms), custom K-Medoids implementation |
| Visualisation | Matplotlib |
| Application | Streamlit outcome-risk tool |
| Hardware | Laptop CPU throughout; no GPU required |

## Work Plan

| Phase | Activities | Duration |
|---|---|---|
| 1 | Data ingestion, exploratory analysis, breed and colour normalisation, baseline decision tree | Week 1 |
| 2 | Association rule mining, support threshold selection, lift-ranked rule interpretation | Week 2 |
| 3 | Classification benchmark, imbalance handling, temporal holdout evaluation | Week 3 |
| 4 | Clustering suite, categorical dissimilarity, atypicality hypothesis test | Week 4 |
| 5 | Bias quantification, outcome-risk tool, report and presentation | Week 5 |

**Responsibilities.**
*Sambhav Ghildiyal:* data pipeline, breed and colour normalisation, exploratory analysis and all descriptive figures.
*Aniket Pandey:* association rule mining, support and lift analysis, and the comparison between mined rules and decision tree paths.
*Aditya Bansal:* the five classification algorithms, imbalance handling and the temporal evaluation protocol.
*Mudit Maheshwari:* the five clustering algorithms, categorical dissimilarity measures and the atypicality hypothesis.
All members share the bias quantification, the final report and the presentation.

## Expected Outcomes

- A reproducible, open-source pipeline running from the raw public CSV to a trained, interpretable outcome predictor.
- A quantified answer to a question shelters act on but rarely measure: the effect of colour, breed group and age on adoption, with confounders controlled.
- A comparison of mined association rules against learned decision tree paths, establishing whether two independent methods recover the same structure.
- An empirical test of whether atypical animals, identified without labels by density-based clustering, experience worse outcomes.
- An outcome-risk tool returning both a prediction and the rule responsible for it, usable in a shelter triage workflow.

## References

1. City of Austin, "Austin Animal Center Outcomes," City of Austin Open Data Portal. https://data.austintexas.gov/
2. R. Agrawal and R. Srikant, "Fast Algorithms for Mining Association Rules," *Proc. 20th Int. Conf. Very Large Data Bases (VLDB)*, 1994, pp. 487–499.
3. M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise," *Proc. 2nd Int. Conf. KDD*, 1996, pp. 226–231.
4. M. Ankerst, M. M. Breunig, H.-P. Kriegel, and J. Sander, "OPTICS: Ordering Points To Identify the Clustering Structure," *Proc. ACM SIGMOD*, 1999, pp. 49–60.
5. L. Kaufman and P. J. Rousseeuw, *Finding Groups in Data: An Introduction to Cluster Analysis*. New York: Wiley, 1990.
6. Z. Huang, "Extensions to the k-Means Algorithm for Clustering Large Data Sets with Categorical Values," *Data Mining and Knowledge Discovery*, vol. 2, no. 3, pp. 283–304, 1998.
7. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
8. S. Raschka, "MLxtend," *Journal of Open Source Software*, vol. 3, no. 24, p. 638, 2018.

> **Note to the group:** references [2]–[8] are the standard primary sources for the algorithms and tools used. Before submission, add two or three domain references on shelter adoption outcomes and "black dog syndrome" from the animal-welfare literature (*Anthrozoös* and the *Journal of Applied Animal Welfare Science* both publish on this) — and verify every citation yourself.
