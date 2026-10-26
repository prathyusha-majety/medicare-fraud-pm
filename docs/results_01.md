# Results Log: Medicare Part B Fraud Detection (2019–2023)

Replication and extension of Johnson & Khoshgoftaar (2023), *Data-Centric AI for Healthcare Fraud Detection*, SN Computer Science 4:389.
Data root: `/scratch/sm3664/medicare-fraud/` · Code: `~/medicare-fraud-pm/notebooks/`

---

## NB01: Profile (Oct 25, 2026)

- Raw data: CMS Part B *by Provider and Service* (28 cols) and *by Provider* (81 cols), 2019–2023. Headers are identical across all years.
- Prov_Svc data is clean in every year:
  - No nulls, no `$`/comma strings, no duplicates on NPI+HCPCS+POS.
  - All NPIs are 10 digits; each NPI has one provider type and one entity code.
- No imputation, string cleaning or dedup needed; the paper's Part D-style fixes don't apply.
- 104 provider types with no spelling variants; the paper's 26 manual fixes aren't needed.
- New provider types after 2019: Adult Congenital Heart Disease, Micrographic Dermatologic Surgery, Opioid Treatment Program (2020); Home Infusion Therapy Services (2021); Dental Anesthesiology (2023).
- CMS drops rows with fewer than 11 beneficiaries, so `Tot_Benes` is never blank and low-volume services are absent (limitation).
- Gender is absent from the 2019–2023 files.

## NB02: Clean + Aggregate (Oct 25, 2026)

- Aggregation key: NPI + year + provider type (entity code carried along). Place of service is dropped, as in the paper.
- 5 numeric fields × 6 statistics (min, max, median, mean, sum, std) = 30 features; std = 0 for single-row providers (paper).
- Spot-check: manual and pipeline statistics match exactly.
- Output: `part_b_agg_2019_2023.parquet`, 5,626,423 × 34.
- Rows per year: 2019 1,093,367 · 2020 1,085,313 · 2021 1,123,589 · 2022 1,148,873 · 2023 1,175,281.
- Note: `Avg_*_sum` features are sums of per-HCPCS averages, not totals, and `Tot_Benes_sum` double-counts patients across codes. Kept to match the paper.

## NB03: Enrich (Oct 25, 2026)

- Inner join with *by Provider* on NPI + year; missing values imputed with 0; race and suppression-flag columns dropped (paper method).
- 56 by-Provider features, versus 47 in the paper. CMS now publishes 25 chronic-condition columns instead of 18, and the paper's own feature list sums to 49.
- The join loses 0 aggregated rows. 379,379 by-Provider NPI-years with no claim lines are excluded, as in the paper.
- Output: `part_b_enriched_2019_2023.parquet`, 5,626,423 × 90, 0 NaN, 1,493,778 unique NPIs (about 3.8 years per provider, so k-fold-by-NPI is required).
- Possible extensions:
  - Suppression flags are fully observed ("blank" = not suppressed) and could serve as a feature.
  - Zero-imputing suppressed demographic columns conflates true zeros with suppressed counts of 1–10, which may create artificial drift.

## NB04: LEIE Labeling (Oct 26, 2026)

**Method (paper):** active LEIE list from each year's January snapshot plus the current LEIE list; keep the 7 fraud rules (Table 6); keep the most recent record per NPI; last exclusion year = exclusion year + minimum period, rounded to the nearest year; label rows with `year < last_excl_year` as fraud; left join so no rows are lost.

**Snapshots used (Wayback `UPDATED.csv`)**

| Snapshot | Capture | Rows | Fraud-rule rows |
|---|---|---|---|
| 2019 | 2019-01-10 | 70,973 | 62,192 |
| 2020 | 2020-03-10 (no Jan capture) | 73,039 | 64,043 |
| 2021 | 2021-03-18 (no Jan capture) | 74,386 | 65,309 |
| 2022 | 2022-03-11 (no Jan capture) | 75,445 | 66,268 |
| 2023 | 2023-01-10 | 77,165 | 67,832 |
| Current | Downloaded Oct 2026; latest EXCLDATE 2026-10-20 | 84,093 | 74,284 |

**Minimum exclusion periods (SSA §1128(c)(3))**
- a1, a2, a3: 5 years.
- c3gi: 10 years; c3gii: permanent. Neither occurs in any snapshot.
- b4, b7: **0 years (assumption)**. There's no statutory minimum and the paper doesn't say what it used.

**Results**
- Cumulative fraud NPIs with a valid NPI: 7,970. By rule: a1 3,767 · b4 2,805 · a3 756 · a2 424 · b7 218.
- Matched in Part B: 960 NPIs. After the year rule: **930 fraud NPIs, 2,652 fraud rows, 0.0471%**. The paper reports 0.0456% for 2013–2019.
- Compared with the earlier active-list-only run (2,557 rows, 900 NPIs): +3.7% rows, +3.3% NPIs. The paper saw +15%.
  - Not like-for-like: the earlier run used an older LEIE download and a different cutoff rule.

**Fraud rate by year**

| Year | Rows | Fraud | Rate |
|---|---|---|---|
| 2019 | 1,093,367 | 840 | 0.0768% |
| 2020 | 1,085,313 | 671 | 0.0618% |
| 2021 | 1,123,589 | 529 | 0.0471% |
| 2022 | 1,148,873 | 369 | 0.0321% |
| 2023 | 1,175,281 | 243 | 0.0207% |

**Findings**
- Using only the January snapshots for 2019–2023 gave just 233 fraud NPIs (402 rows). Excluded providers stop billing, so most labels come from exclusions *after* the claim years. **The current list is essential.**
- **Right-censoring:** the fraud rate falls each year because later fraud hasn't been caught yet. In the cross-regime experiment this will look like model decay unless controlled for.

Output: `processed/part_b/labeled/part_b_labeled_2019_2023.parquet`, 5,626,423 × 91.

## NB05: Baseline, paper replication (Oct 26, 2026, smoke test)

**Setup (paper):** XGB with max_depth 4, RF with max_depth 16, defaults otherwise; k-fold-by-NPI, folds stratified by whether an NPI was ever fraud; threshold = training-fold prior.
- Features: SbP 160, A 134, AE 190. Each includes 104 provider-type one-hot columns. Entity code is not used (paper).
- Folds: about 1.12M rows each, 515–549 fraud rows, 186 fraud NPIs per fold.

**Smoke test: XGB, run 0, fold 0 only**

| Dataset | AUC | G-Mean | Paper AUC (XGB, 2013–2019) |
|---|---|---|---|
| SbP | 0.7502 | 0.6749 | 0.8310 |
| A | 0.7121 | 0.6285 | 0.9392 |
| AE | 0.7312 | 0.6513 | 0.9536 |

AE vs A: TPR 0.5209 (+3.2%), TNR 0.8141 (+4.0%), G-Mean 0.6513 (+3.6%), AUPRC 0.0031 (base rate 0.00047).

**Diagnostic: memorization check (paper Fig. 2), AE + XGB, fold 0**

| Split | Train AUC | Test AUC |
|---|---|---|
| Row-level k-fold | 0.9279 | 0.8348 |
| k-fold-by-NPI | 0.9354 | 0.7312 |

**Interpretation**
- Row-level k-fold beats by-NPI by about 0.10, so memorization is present and the evaluation behaves like the paper's.
- Training AUC of 0.93 is well below the paper's (about 0.99), so the signal in this data is weaker, not just the generalization.
- An earlier, independently built pipeline gave AUC of about 0.67. The gap to the paper is therefore not a bug specific to NB05.
- **Open question:** is the gap a real era effect (2019–2023) or a difference from the paper's method? A one-fold result can't settle the SbP > AE > A ordering.

## Deviations from the paper (for the thesis)

1. Snapshots for 2020–2022 are from March (no January capture on Wayback).
2. Rules 1128c3gi and 1128c3gii are absent from every snapshot.
3. The minimum exclusion period for b4 and b7 is assumed to be 0.
4. No gender column.
5. 56 SbP features instead of 47.
6. The SbP dataset uses the same rows as A and AE, not the full by-Provider population.
7. Label horizon: LEIE downloaded Oct 2026.
8. The NPI recovery crosswalk is not used (the paper doesn't use one).

## Next steps

1. Full NB05 run, XGB only (30 folds), for confidence intervals.
2. Rebuild **2013–2019** with NB02–NB05 (the paper's window). This decides whether the gap is the era or the pipeline.
3. NB06: undersampling experiment (see below).
4. Later: break AUC down by exclusion type and by the gap between claim year and exclusion year. Note that 35% of fraud NPIs are b4 license revocations.

## Planned NB06: Undersampling + CV vs Independent Test

- RF and XGB on AE; ratios 1:1, 1:3, 1:5, 1:50, 1:100, plus natural (no undersampling); 10-fold CV vs a 15% independent test; 5 seeds; FN/FP by year.
- **Rules:**
  - Split by NPI at both levels, the independent test set and the CV folds.
  - Undersample training data only. Validation and test keep the natural ratio (about 1:2,100).
  - Threshold = training prior after sampling, 1/(1+r): 0.5, 0.25, 0.167, 0.0196, 0.0099.
- **Note on the earlier run:** the old pipeline showed TPR ≈ 1.0 and TNR ≈ 0.0 on the independent test.
  - A 0.5 threshold alone doesn't explain TNR ≈ 0 at every ratio, so suspect a bug such as mismatched train/test columns.
  - The planned fix of using the population prior (0.00058) as the threshold is wrong for undersampled models.