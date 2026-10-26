## 2026-10-26 — Notebook 06, Step 0–1: baseline + categorical encoding

### Setup
- Data: processed/part_b/labeled/part_b_labeled_2019_2023.parquet
  - 5,626,423 rows x 91 cols, 1,493,778 NPIs, 0 NaN
  - Labels: cumulative LEIE (notebook 04), 2,652 fraud rows (0.0471%), 930 fraud NPIs
  - Fraud rows by year: 2019 840 | 2020 671 | 2021 529 | 2022 369 | 2023 243
  - NOTE: third label count so far (active-only 2,557; crosswalk v2 3,408; cumulative 2,652)
- Features: 86 numeric (30 aggregated + 56 by-Provider); year and NPI excluded
- Split: 85/15 by NPI, stratified at provider level, seed 42
  - Train 4,782,655 rows / 2,232 fraud; Test 843,768 rows / 420 fraud; no NPI overlap
  - Test NPIs saved: results/nb06_step1/test_npis.parquet
- CV: 10-fold by NPI (provider-level StratifiedKFold) on train; ~478K rows / 206–239 fraud per fold
- Undersampling: training folds only; validation folds at natural ratio
- Models: XGB (max_depth=4, hist), RF (max_depth=16); other params default; n_jobs=4
- Threshold: positive-class prior of training data (paper's rule) -> 0.5 at 1:1
- Encoding: OneHotEncoder fit on training fold, min_frequency=50, handle_unknown=infrequent_if_exist

### Step 0 — sanity check (fold 0)
- Model behaves normally: fraud scores clearly higher than non-fraud; ~25% flagged at thr 0.5
- Earlier TPR~1/TNR~0 on the independent test was a bug in the old pipeline, not a threshold issue
- Correction: the pending fix "use 0.00058 as threshold" was wrong for models trained at 1:1; retire it

### Ratio sweep (fold 0, XGB, base+ptype)
| Ratio | AUC    | AUPRC  | TPR   | TNR   | thr     | time |
|-------|--------|--------|-------|-------|---------|------|
| 1:1   | 0.7660 | 0.0038 | 0.652 | 0.749 | 0.50000 | 6s   |
| 1:5   | 0.7525 | 0.0034 | 0.591 | 0.790 | 0.16667 | 6s   |
| 1:50  | 0.7657 | 0.0035 | 0.526 | 0.801 | 0.01961 | 8s   |
| 1:100 | 0.7635 | 0.0028 | 0.574 | 0.801 | 0.00990 | 10s  |
| none  | 0.7546 | 0.0032 | 0.565 | 0.796 | 0.00047 | 102s |
- AUC flat across ratios (within noise, SE ~0.03). Undersampling is NOT the cause of the gap to the paper.
- 1:1 adopted for iteration (equal performance, ~15x faster than full data).

### Step 1 — 10-fold CV, 1:1, seed 42 (mean, 95% CI)
| Features       | Model | AUC                     | AUPRC  | TPR    | TNR    | G-Mean | FN/fold | FP/fold |
|----------------|-------|-------------------------|--------|--------|--------|--------|---------|---------|
| base           | XGB   | 0.7295 (0.7078, 0.7512) | 0.0032 | 0.5975 | 0.7320 | 0.6601 | 89.9    | 128,110 |
| base           | RF    | 0.7399 (0.7162, 0.7636) | 0.0030 | 0.5827 | 0.7630 | 0.6651 | 93.2    | 113,298 |
| base+ptype     | XGB   | 0.7647 (0.7421, 0.7874) | 0.0033 | 0.6364 | 0.7454 | 0.6875 | 81.4    | 121,687 |
| base+ptype     | RF    | 0.7542 (0.7306, 0.7778) | 0.0036 | 0.6113 | 0.7702 | 0.6852 | 86.8    | 109,851 |
| base+ptype+ent | XGB   | 0.7659 (0.7461, 0.7857) | 0.0034 | 0.6554 | 0.7442 | 0.6975 | 77.1    | 122,286 |
| base+ptype+ent | RF    | 0.7575 (0.7363, 0.7786) | 0.0036 | 0.6057 | 0.7722 | 0.6828 | 88.0    | 108,877 |

Paired fold-by-fold differences (same folds):
- +provider_type: XGB +0.035 AUC (10/10 folds up, sd 0.010); RF +0.014 (10/10 folds up)
- +ent_cd on top: XGB +0.001, RF +0.003, mixed direction -> no effect

### Best configuration so far
- XGB, base + provider_type (one-hot), 1:1 undersampling, thr 0.5
  - AUC 0.765, TPR 0.64, TNR 0.75, G-Mean 0.69
- Highest raw numbers: XGB base+ptype+ent (AUC 0.7659, TPR 0.655, G-Mean 0.698), but the ent_cd gain is within noise; keep provider_type only.
- In practice: per fold catches ~144 of ~225 fraud rows while flagging ~122K non-fraud rows (precision ~0.12%)
- Paper (Table 9/10, Part B AE, XGB): AUC 0.954, TPR 0.85, TNR 0.96, G-Mean 0.90 -> large gap remains

### Per-year AUC (fold 0, XGB base+ptype)
| Year | Fraud rows | AUC    |
|------|------------|--------|
| 2019 | 69         | 0.7866 |
| 2020 | 59         | 0.7648 |
| 2021 | 45         | 0.7211 |
| 2022 | 34         | 0.7479 |
| 2023 | 23         | 0.7568 |
- Roughly flat; at most a slight decline (within noise, +/-0.04-0.06 per year)
- 2019 (least label lag, inside paper's window) is only 0.79 -> period/label lag is not the main cause of the gap

### Not yet run
- Cell 9 (independent 15% test). Do not run until the baseline gap is understood.

### Conclusions
1. Encoding provider_type is a real, consistent gain; ent_cd is not useful.
2. Undersampling ratio does not matter for AUC.
3. The gap to the paper (0.76 vs 0.95) is upstream: pipeline, label rule, or label composition, not the period or undersampling.

### Next
1. Count fraud NPIs by LEIE exclusion type (a1/a2/a3/b4/b7), if the type is in the labeled file.
2. Replicate on 2013–2019 with the identical pipeline:
   ~0.95 -> pipeline OK, 2019–2023 is harder; ~0.76 -> pipeline differs from paper.
3. Hold Steps 2 (peer features) and 7 (class weighting) until the baseline gap is explained.
4. Open for Prof. Dehzangi: which label version to standardize on; meaning of 1:50/1:100; 10-fold vs paper's 5x6.