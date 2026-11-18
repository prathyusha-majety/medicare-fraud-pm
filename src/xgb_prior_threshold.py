"""XGBoost on Part B 2019-2023: k-fold-by-NPI, prior threshold, numeric-only vs +categoricals."""
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

DATA = os.environ.get("DATA_DIR", "/scratch/sm3664/medicare-fraud")
IN = f"{DATA}/processed/part_b/labeled/part_b_labeled_2019_2023.parquet"
OUT = f"{DATA}/results/part_b/xgb_prior_threshold"
os.makedirs(OUT, exist_ok=True)

LABEL, NPI, YEAR = "fraud", "Rndrng_NPI", "year"          # match Cell 2
CATS = ["Rndrng_Prvdr_Type", "Rndrng_Prvdr_Ent_Cd"]
N_JOBS, SEED, K = int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), 42, 5

df = pd.read_parquet(IN)
y = df[LABEL].astype(int).to_numpy()
groups, years = df[NPI].to_numpy(), df[YEAR].to_numpy()

# drop keys, label, categoricals, any LEIE-derived (leaky) or non-numeric columns
leak = [c for c in df.columns if any(s in c.lower() for s in ("excl", "leie", "reinst"))]
other = [c for c in df.columns if c not in [LABEL, NPI, YEAR] + CATS + leak
         and not pd.api.types.is_numeric_dtype(df[c])]
print("dropped leak cols:", leak, "| dropped non-numeric:", other, flush=True)
num = df.drop(columns=[LABEL, NPI, YEAR] + CATS + leak + other).astype("float32")
cat = pd.get_dummies(df[CATS].astype(str), prefix=CATS, dtype="float32")
configs = {"numeric_only": num, "numeric_plus_cats": pd.concat([num, cat], axis=1)}
del df

# folds assigned per NPI, stratified by "ever fraud"; saved for reuse by every later model
rng = np.random.default_rng(SEED)
npi_fraud = pd.Series(y, index=groups).groupby(level=0).max()
parts = []
for cls in (1, 0):
    ids = npi_fraud.index[npi_fraud.to_numpy() == cls].to_numpy()
    ids = rng.permutation(ids)
    for k, chunk in enumerate(np.array_split(ids, K)):
        parts.append(pd.Series(k, index=chunk, dtype="int8"))
fold_s = pd.concat(parts)
fold_s.rename_axis(NPI).rename("fold").reset_index().to_parquet(f"{OUT}/folds_by_npi.parquet", index=False)
folds = fold_s.reindex(groups).to_numpy()

rows = []
for name, X in configs.items():
    Xv = X.to_numpy()
    print(name, "features:", Xv.shape[1], flush=True)
    for k in range(K):
        tr, te = folds != k, folds == k
        prior = y[tr].mean()
        m = XGBClassifier(max_depth=4, tree_method="hist", n_jobs=N_JOBS, random_state=SEED)
        m.fit(Xv[tr], y[tr])
        p = m.predict_proba(Xv[te])[:, 1]
        yt, pred = y[te], (p >= prior).astype(int)
        tpr, tnr = pred[yt == 1].mean(), 1 - pred[yt == 0].mean()
        r = dict(config=name, fold=k, auc=roc_auc_score(yt, p), auprc=average_precision_score(yt, p),
                 tpr=tpr, tnr=tnr, gmean=np.sqrt(tpr * tnr), prior=prior, test_pos=int(yt.sum()))
        for yr in np.unique(years):
            mk = (years[te] == yr) & (yt == 1)
            r[f"recall_{yr}"] = pred[mk].mean() if mk.any() else np.nan
        rows.append(r)
        print(r, flush=True)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/fold_results.csv", index=False)
print(res.drop(columns="fold").groupby("config").agg(["mean", "std"]).round(4).T)