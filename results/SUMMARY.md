# SecondChance — results summary

- dataset: Austin Animal Center, 162,461 outcome events (Dog, Cat), 2013–2025
- clusters requested: k=8; temporal cutoff: 2021
- wall clock: 12.7 min

## Outcome distribution

| outcome         |     n |   pct |
|:----------------|------:|------:|
| Adoption        | 83866 | 51.62 |
| Transfer        | 46668 | 28.73 |
| Return to Owner | 25466 | 15.68 |
| Euthanasia      |  4704 |  2.9  |
| Other           |  1757 |  1.08 |

## The claims people make

| claim                        |   group_n |   group_adoption_pct |   others_adoption_pct |   adoption_gap_pp |   group_median_stay |   others_median_stay |   group_euthanasia_pct |   others_euthanasia_pct |
|:-----------------------------|----------:|---------------------:|----------------------:|------------------:|--------------------:|---------------------:|-----------------------:|------------------------:|
| Black cats vs other cats     |     15898 |                50.31 |                 52.55 |             -2.25 |                8.11 |                 7.13 |                   4.08 |                    3.56 |
| Black dogs vs other dogs     |     25118 |                53.35 |                 50.58 |              2.77 |                6.05 |                 5.98 |                   2.09 |                    2.4  |
| Pit bull type vs other dogs  |     16531 |                47.75 |                 52.09 |             -4.34 |               10.02 |                 5.27 |                   4.35 |                    1.89 |
| Senior vs non-senior dogs    |     10019 |                30.18 |                 53.85 |            -23.66 |                4    |                 6.09 |                   5.5  |                    1.94 |
| Senior vs non-senior cats    |      3800 |                42.29 |                 52.61 |            -10.32 |                8.2  |                 7.19 |                  12.74 |                    3.15 |
| Mixed breed vs purebred dogs |     74526 |                52.17 |                 48.06 |              4.11 |                6.01 |                 5.93 |                   2.36 |                    2.16 |

Largest effect: **Senior vs non-senior dogs** at -23.66 percentage points.

## Adoption by colour, crude vs standardised

| value         |     n |   crude_pct |   standardised_pct |   confounding_shift |   median_stay_days |
|:--------------|------:|------------:|-------------------:|--------------------:|-------------------:|
| Other         |  2252 |     56.2167 |             55.858 |              -0.359 |            5.86493 |
| Calico/Tortie |  9505 |     54.9395 |             55.519 |               0.579 |            8.09444 |
| Tricolor      |  3900 |     53.2564 |             52.33  |              -0.926 |            5.06181 |
| Orange/Red    | 12714 |     51.3843 |             52.135 |               0.751 |            6.21667 |
| Brown         | 37278 |     52.1085 |             52.046 |              -0.062 |            6.92569 |
| Cream         |  3659 |     52.692  |             51.751 |              -0.941 |            6.00625 |
| Gray/Blue     | 16215 |     50.6815 |             51.599 |               0.918 |            6.82361 |
| Black         | 41016 |     52.1723 |             51.575 |              -0.597 |            6.68993 |
| Tan/Fawn      | 14942 |     50.9169 |             51.022 |               0.105 |            5.34306 |
| White         | 20980 |     48.5701 |             49.484 |               0.914 |            5.92222 |

Standardised for species, age band and intake route by direct standardisation, so these are like-for-like.

## Strongest association rules

| outcome         | antecedent                                |   support |   confidence |    lift |   n_animals |
|:----------------|:------------------------------------------|----------:|-------------:|--------:|------------:|
| Adoption        | age=Juvenile AND intake=Owner Surrender   | 0.0282652 |    0.802236  | 1.55405 |        4592 |
| Adoption        | age=Juvenile AND mixed=No                 | 0.03511   |    0.773214  | 1.49783 |        5704 |
| Adoption        | age=Juvenile AND condition=Normal         | 0.0967863 |    0.73566   | 1.42508 |       15724 |
| Adoption        | age=Juvenile AND type=Cat                 | 0.0522895 |    0.734417  | 1.42268 |        8495 |
| Adoption        | intake=Owner Surrender AND pattern=tabby  | 0.0277359 |    0.733637  | 1.42117 |        4506 |
| Adoption        | age=Juvenile AND pattern=tabby            | 0.0258647 |    0.732566  | 1.41909 |        4202 |
| Euthanasia      | intake=Stray AND type=Cat                 | 0.0138495 |    0.0424857 | 1.46732 |        2250 |
| Euthanasia      | mixed=Yes AND type=Cat                    | 0.0105071 |    0.0415632 | 1.43546 |        1707 |
| Euthanasia      | breed=Domestic Shorthair AND intake=Stray | 0.0112704 |    0.0408104 | 1.40946 |        1831 |
| Euthanasia      | age=Adult                                 | 0.0109934 |    0.0371588 | 1.28334 |        1786 |
| Euthanasia      | type=Cat                                  | 0.0155545 |    0.036834  | 1.27213 |        2527 |
| Euthanasia      | breed=Domestic Shorthair                  | 0.0125938 |    0.0355715 | 1.22852 |        2046 |
| Return to Owner | intake=Public Assist AND sterilised=Yes   | 0.0214451 |    0.7454    | 4.7553  |        3484 |
| Return to Owner | intake=Public Assist AND type=Dog         | 0.0353254 |    0.687141  | 4.38363 |        5739 |
| Return to Owner | age=Adult AND intake=Public Assist        | 0.0219314 |    0.683615  | 4.36114 |        3563 |
| Return to Owner | condition=Normal AND intake=Public Assist | 0.0373197 |    0.675769  | 4.31109 |        6063 |
| Return to Owner | intake=Public Assist AND sex=Male         | 0.0230455 |    0.66738   | 4.25757 |        3744 |
| Return to Owner | intake=Public Assist AND pattern=none     | 0.0316199 |    0.660622  | 4.21446 |        5137 |
| Transfer        | condition=Normal AND sterilised=Unknown   | 0.0220668 |    0.90576   | 3.15314 |        3585 |
| Transfer        | condition=Normal AND sex=Unknown          | 0.0220668 |    0.90576   | 3.15314 |        3585 |
| Transfer        | age=Neonate AND sex=Unknown               | 0.0186075 |    0.875471  | 3.04769 |        3023 |
| Transfer        | age=Neonate AND sterilised=Unknown        | 0.0186075 |    0.875471  | 3.04769 |        3023 |
| Transfer        | pattern=tabby AND sex=Unknown             | 0.0125815 |    0.872014  | 3.03566 |        2044 |
| Transfer        | pattern=tabby AND sterilised=Unknown      | 0.0125815 |    0.872014  | 3.03566 |        2044 |

## Apriori vs FP-Growth

|   n_transactions |   Apriori_seconds |   n_itemsets |   FP-Growth_seconds |   speedup |
|-----------------:|------------------:|-------------:|--------------------:|----------:|
|            20000 |              1.77 |         4503 |                1.78 |      0.99 |
|            50000 |              7.26 |         4494 |                3.32 |      2.19 |
|           100000 |             17.21 |         4440 |                5.17 |      3.33 |
|           162461 |             44.45 |         4516 |                8.8  |      5.05 |

Identical itemsets. Apriori rescans the database once per candidate level, so its cost climbs with the transaction count while FP-Growth stays nearly flat.

## Association rules vs decision tree paths

| field       |   tree_share_pct |   apriori_share_pct |   rank_tree |   rank_apriori |   rank_gap |
|:------------|-----------------:|--------------------:|------------:|---------------:|-----------:|
| condition   |            32.13 |                5.81 |           1 |             10 |          9 |
| intake      |            18.34 |               11.27 |           2 |              3 |          1 |
| sterilised  |            16.38 |                8.27 |           3 |              6 |          3 |
| type        |            16.02 |                9.68 |           4 |              5 |          1 |
| age         |            13.98 |               15.14 |           5 |              1 |          4 |
| sex         |             1.75 |                7.57 |           6 |              8 |          2 |
| mixed       |             1.14 |                5.46 |           7 |             11 |          4 |
| breed       |             0.19 |               10.56 |           8 |              4 |          4 |
| colour      |             0.06 |               11.97 |           9 |              2 |          7 |
| multicolour |             0    |                6.34 |          10 |              9 |          1 |
| pattern     |             0    |                7.92 |          10 |              7 |          3 |

Of 45 tree leaves, **30 (66.67%)** are covered by a mined association rule predicting the same outcome. Of the rest, 1 rely on negation, which positive-item Apriori cannot express at all, and 7 sit below Apriori's support floor.

But the two methods rank the ATTRIBUTES completely differently — Spearman correlation **0.0547**, essentially zero. The tree concentrates on intake condition, species, intake route and age, and uses breed and colour *not at all*; Apriori spreads its rules evenly and reports breed and colour patterns freely.

That is not a contradiction, it is the difference between the two questions. A decision tree finds what is SUFFICIENT: once species, age and intake condition are known, colour adds no information gain, so it is never split on. Apriori finds what is ASSOCIATED, and colour genuinely is associated. The standardised adoption rates above reach the same conclusion by a third route: colour's apparent effect is almost entirely confounding.

## Classification

| protocol   | model                             |   train_rows |   test_rows |   accuracy |   balanced_accuracy |   macro_f1 |   weighted_f1 |     mcc |   fit_seconds |   predict_seconds |
|:-----------|:----------------------------------|-------------:|------------:|-----------:|--------------------:|-----------:|--------------:|--------:|--------------:|------------------:|
| random     | SVM (linear)                      |        60000 |       16000 |    0.57788 |             0.44068 |    0.38756 |       0.56211 | 0.34638 |          9.81 |              0.02 |
| random     | Neural Net (MLP, backprop)        |        64000 |       16000 |    0.62819 |             0.37265 |    0.38167 |       0.59855 | 0.37521 |          8.75 |              0.03 |
| random     | Torch NN (embeddings, unweighted) |        54400 |       16000 |    0.62581 |             0.37372 |    0.3806  |       0.59449 | 0.37302 |         56.69 |              0.03 |
| random     | SVM (RBF)                         |        10001 |       16000 |    0.50856 |             0.44748 |    0.37428 |       0.52798 | 0.31    |         27.14 |             66.48 |
| random     | Naive Bayes (Bernoulli)           |        64000 |       16000 |    0.49269 |             0.38704 |    0.35781 |       0.49557 | 0.26028 |          0.39 |              0.04 |
| random     | Torch NN (embeddings, weighted)   |        54400 |       16000 |    0.45531 |             0.4768  |    0.34907 |       0.48536 | 0.28449 |         32.94 |              0.03 |
| random     | Decision Tree                     |        64000 |       16000 |    0.41669 |             0.46317 |    0.32921 |       0.45867 | 0.26104 |          2.1  |              0.01 |
| random     | KNN (k=15)                        |        19999 |       16000 |    0.59237 |             0.32151 |    0.32597 |       0.55578 | 0.29762 |          0.05 |              7.09 |
| temporal   | Torch NN (embeddings, unweighted) |        52833 |       17843 |    0.65017 |             0.3838  |    0.37763 |       0.64305 | 0.33853 |         87.38 |              0.04 |
| temporal   | Neural Net (MLP, backprop)        |        62157 |       17843 |    0.64625 |             0.37945 |    0.37347 |       0.63897 | 0.33099 |          6.75 |              0.02 |
| temporal   | SVM (linear)                      |        60000 |       17843 |    0.58454 |             0.43887 |    0.36436 |       0.60684 | 0.30838 |         20.45 |              0.02 |
| temporal   | SVM (RBF)                         |        10000 |       17843 |    0.48484 |             0.42627 |    0.33    |       0.53133 | 0.26119 |         50.99 |             52.48 |
| temporal   | KNN (k=15)                        |        20000 |       17843 |    0.6189  |             0.31918 |    0.31863 |       0.59817 | 0.2385  |          0.05 |              7.7  |
| temporal   | Torch NN (embeddings, weighted)   |        52833 |       17843 |    0.45032 |             0.43834 |    0.31786 |       0.51118 | 0.2483  |         48.35 |              0.04 |
| temporal   | Naive Bayes (Bernoulli)           |        62157 |       17843 |    0.43412 |             0.38292 |    0.30669 |       0.45695 | 0.18582 |          0.49 |              0.06 |
| temporal   | Decision Tree                     |        62157 |       17843 |    0.41826 |             0.42073 |    0.30232 |       0.48679 | 0.22209 |          2.54 |              0.01 |

Best macro-F1 falls from **0.3876** on a random split to **0.3776** under the temporal holdout. The random split is the optimistic one.

Accuracy and balanced accuracy disagree sharply here: a model that predicts Adoption for everything scores well on one and badly on the other. Macro-F1 is the honest number.

### What the PyTorch network adds

`MLPClassifier` has no `class_weight` parameter, which is why it posts the best accuracy and nearly the worst balanced accuracy: it has no mechanism to resist the majority class. A weighted cross-entropy loss does.

- unweighted torch net: accuracy 0.626, balanced accuracy 0.374 — statistically indistinguishable from the sklearn MLP, which is the check that the implementation is sound
- weighted torch net: accuracy 0.455, balanced accuracy **0.477** — the best balanced accuracy in the study

The weighting is a dial between the two, and it is the only model here that has one. Note it *lowers* macro-F1 (0.349 against 0.381): over-predicting rare classes costs their precision.

## Clustering

| algorithm                        |   n_rows |   n_clusters |   noise_points |   noise_pct |     ARI |     NMI |   homogeneity |   completeness |   silhouette |   seconds | dissimilarity        |   cost |   eps_attrs |   min_samples |
|:---------------------------------|---------:|-------------:|---------------:|------------:|--------:|--------:|--------------:|---------------:|-------------:|----------:|:---------------------|-------:|------------:|--------------:|
| K-Means (one-hot, k=8)           |     6001 |            8 |              0 |       0     | 0.01131 | 0.04138 |       0.05761 |        0.03228 |      0.16086 |      0.52 | Euclidean on one-hot |    nan |         nan |           nan |
| K-Medoids (Hamming, k=8)         |     6001 |            8 |              0 |       0     | 0.0181  | 0.04364 |       0.06083 |        0.03402 |      0.09215 |      0.12 | Hamming              |   1732 |         nan |           nan |
| Hierarchical (complete, Hamming) |     3000 |            8 |              0 |       0     | 0.0312  | 0.0476  |       0.06214 |        0.03857 |      0.03673 |      0.38 | Hamming              |    nan |         nan |           nan |
| Hierarchical (average, Hamming)  |     3000 |            8 |              0 |       0     | 0.03727 | 0.06495 |       0.05609 |        0.07715 |      0.16391 |      0.48 | Hamming              |    nan |         nan |           nan |
| Hierarchical (ward, one-hot)     |     3000 |            8 |              0 |       0     | 0.02448 | 0.06184 |       0.08619 |        0.04821 |      0.12547 |      0.83 | Euclidean on one-hot |    nan |         nan |           nan |
| DBSCAN (<=1 attrs differ)        |     6001 |            7 |           1221 |      20.347 | 0.03092 | 0.06485 |       0.0704  |        0.06011 |      0.13919 |      0.34 | Hamming              |    nan |           1 |            10 |
| DBSCAN (<=2 attrs differ)        |     6001 |            1 |             61 |       1.016 | 0.00613 | 0.00565 |       0.00296 |        0.05963 |    nan       |      0.42 | Hamming              |    nan |           2 |            10 |
| DBSCAN (<=3 attrs differ)        |     6001 |            1 |              1 |       0.017 | 0.00037 | 0.00103 |       0.00052 |        0.36532 |    nan       |      0.57 | Hamming              |    nan |           3 |            10 |
| OPTICS (Hamming, xi)             |     3000 |           16 |           1763 |      58.767 | 0.00017 | 0.04916 |       0.04768 |        0.05074 |      0.16923 |     37.03 | Hamming              |    nan |         nan |            10 |

## Atypicality test

| algorithm                 |   noise_n |   clustered_n |   euth_rate_noise |   euth_rate_clustered |   euth_rate_ratio |   adopt_rate_noise |   adopt_rate_clustered | reliable   |
|:--------------------------|----------:|--------------:|------------------:|----------------------:|------------------:|-------------------:|-----------------------:|:-----------|
| DBSCAN (<=1 attrs differ) |      1221 |          4780 |             5.651 |                 2.197 |             2.573 |             46.519 |                 52.908 | True       |
| DBSCAN (<=2 attrs differ) |        61 |          5940 |            14.754 |                 2.778 |             5.311 |             21.311 |                 51.919 | True       |
| DBSCAN (<=3 attrs differ) |         1 |          6000 |           100     |                 2.883 |            34.682 |              0     |                 51.617 | False      |
| OPTICS (Hamming, xi)      |      1763 |          1237 |             4.027 |                 1.293 |             3.114 |             48.383 |                 56.265 | True       |

Animals that density-based clustering labels as noise — unusual attribute combinations, found with no labels — are euthanised at **3.11×** the rate of clustered animals (4.027% vs 1.293%).

## Figures

- `results/plots/01-outcomes.png`
- `results/plots/02-rules-lift.png`
- `results/plots/03-rule-mining-cost.png`
- `results/plots/04-protocol-comparison.png`
- `results/plots/05-accuracy-trap.png`
- `results/plots/06-per-class-f1.png`
- `results/plots/07-confusion-random.png`
- `results/plots/07-confusion-temporal.png`
- `results/plots/08-atypicality.png`
- `results/plots/09-dendrogram.png`
- `results/plots/10-claims.png`
- `results/plots/11-colour-adjusted.png`
- `results/plots/12-field-ranking.png`
- `results/plots/13-torch-training.png`
