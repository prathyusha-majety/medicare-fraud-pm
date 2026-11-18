"""Compare ML models on Part B 2019-2023 using the saved k-fold-by-NPI folds.
Usage: python src/model_compare.py <lr|rf|et|lgbm|catboost|xgb_w>"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

MODEL = sys.argv[1]
DATA = os.environ.get("DATA_DIR", "/scratch/sm3664/medicare-fraud")
IN = f"{DATA}/processed/part_b/labeled/part_b_labeled_2019_2023.parquet"
FOLDS = f"{DATA}/results/part_b/xgb_prior_threshold/folds_by_npi.parquet"
OUT = f"{DATA}/results/part_b/model_compare"
os.makedirs(OUT, exist_ok=True)
LABEL, NPI, YEAR = "fraud", "Rndrng_NPI", "year"
CATS = ["Rndrng_Prvdr_Type", "Rndrng_Prvdr_Ent_Cd"]
N_JOBS, SEED, K = int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), 42, 5

df = pd.read_parquet(IN)
y = df[LABEL].astype(int).to_numpy()
years = df[YEAR].to_numpy()
folds = pd.read_parquet(FOLDS).set_index(NPI)["fold"].reindex(df[NPI]).to_numpy()
assert not pd.isna(folds).any(), "some NPIs have no fold"
X = pd.concat([df.drop(columns=[LABEL, NPI, YEAR] + CATS).astype("float32"),
               pd.get_dummies(df[CATS].astype(str), prefix=CATS, dtype="float32")], axis=1)
print(MODEL, "features:", X.shape[1], flush=True)   # should be 192, same as XGBoost run
del df
X = X.to_numpy()


def make_model(name, pos_weight):
    if name == "lr":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=300))
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, max_depth=16, class_weight="balanced",
                                      n_jobs=N_JOBS, random_state=SEED)
    if name == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=100, max_depth=16, class_weight="balanced",
                                    n_jobs=N_JOBS, random_state=SEED)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(class_weight="balanced", n_jobs=N_JOBS, random_state=SEED, verbose=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=500, auto_class_weights="Balanced", thread_count=N_JOBS,
                                  random_seed=SEED, verbose=100)
    if name == "xgb_w":
        from xgboost import XGBClassifier
        return XGBClassifier(max_depth=4, tree_method="hist", scale_pos_weight=pos_weight,
                             n_jobs=N_JOBS, random_state=SEED)
    raise ValueError(name)


rows = []
for k in range(K):
    t0 = time.time()
    tr, te = folds != k, folds == k
    pos_w = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    m = make_model(MODEL, pos_w)
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    yt = y[te]
    pred = (p >= 0.5).astype(int)   # balanced weights: 0.5 here = prior threshold on an unweighted model
    tpr, tnr = pred[yt == 1].mean(), 1 - pred[yt == 0].mean()
    r = dict(model=MODEL, fold=k, auc=roc_auc_score(yt, p), auprc=average_precision_score(yt, p),
             tpr=tpr, tnr=tnr, gmean=np.sqrt(tpr * tnr), minutes=round((time.time() - t0) / 60, 1))
    for yr in np.unique(years):
        mk = (years[te] == yr) & (yt == 1)
        r[f"recall_{yr}"] = pred[mk].mean() if mk.any() else np.nan
    rows.append(r)
    print(r, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/{MODEL}_folds.csv", index=False)   # saved after every fold

res = pd.DataFrame(rows)
print(res.drop(columns=["model", "fold"]).agg(["mean", "std"]).round(4).T)