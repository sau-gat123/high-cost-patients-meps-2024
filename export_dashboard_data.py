"""Runs analysis.py, then saves ONLY aggregate results for the dashboard (no person-level rows)."""
import runpy, json, numpy as np, pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

ns = runpy.run_path("analysis.py")
df, y_test, curves = ns["df"], ns["y_test"], ns["curves"]

# curves (test set), thinned to keep files small
roc, pr = [], []
for name, p in curves.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, 300).astype(int))
    roc.append(pd.DataFrame({"model": name, "fpr": fpr[idx], "tpr": tpr[idx]}))
    prec, rec, _ = precision_recall_curve(y_test, p)
    idx = np.unique(np.linspace(0, len(prec) - 1, 300).astype(int))
    pr.append(pd.DataFrame({"model": name, "recall": rec[idx], "precision": prec[idx]}))
pd.concat(roc).to_csv("data/roc_curve.csv", index=False)
pd.concat(pr).to_csv("data/pr_curve.csv", index=False)

# confusion matrices at 0.5
cm = {}
for name, p in curves.items():
    tn, fp, fn, tp = confusion_matrix(y_test, (p >= 0.5).astype(int)).ravel()
    cm[name] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
json.dump(cm, open("data/confusion.json", "w"))

# high-cost rate by group (with group sizes)
def rate(col, labels=None):
    g = df.groupby(col)["high_cost"].agg(["mean", "size"]).reset_index()
    g.columns = ["group", "share_high_cost", "n"]
    if labels: g["group"] = g["group"].map(labels)
    return g
rate(df["INSCOV24"], {1: "Private", 2: "Public only", 3: "Uninsured"}).to_csv("data/rate_insurance.csv", index=False)
chronic = df["chronic_count"].clip(upper=4).astype(int).astype(str).replace({"4": "4+"})
rate(chronic).to_csv("data/rate_chronic.csv", index=False)
rate(df["phys_health"].astype(int).map({1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"})).to_csv("data/rate_physhealth.csv", index=False)
agebin = pd.cut(df["age"], [17, 29, 44, 59, 74, 100], labels=["18-29", "30-44", "45-59", "60-74", "75+"])
rate(agebin).to_csv("data/rate_age.csv", index=False)

# spending distribution among spenders (histogram counts only)
sp = np.log10(df.loc[df["TOTEXP24"] > 0, "TOTEXP24"])
cnt, edges = np.histogram(sp, bins=40)
pd.DataFrame({"log10_spend_mid": (edges[:-1] + edges[1:]) / 2, "adults": cnt}).to_csv("data/spend_hist.csv", index=False)

# headline facts
facts = {"n_adults": int(len(df)), "n_households": int(df["DUID"].nunique()),
         "cutoff": float(ns["cutoff"]), "share_zero": float((df["TOTEXP24"] == 0).mean()),
         "median": float(df["TOTEXP24"].median()), "mean": float(df["TOTEXP24"].mean()),
         "n_high": int(df["high_cost"].sum()), "test_n": int(len(y_test)), "test_pos": int(y_test.sum()),
         "uninsured_n": int(df["ins_uninsured"].sum()),
         "uninsured_high": int(df.loc[df["ins_uninsured"] == 1, "high_cost"].sum())}
json.dump(facts, open("data/facts.json", "w"), indent=1)

# copy result tables
import shutil
for f, t in [("table5_logit_weighted", "odds_ratios"), ("table7_test_performance", "test_performance"),
             ("table8_cv_auc", "cv_auc"), ("table2_high_cost_vs_other", "high_vs_other")]:
    shutil.copy(f"results/{f}.csv", f"data/{t}.csv")
print("exported", facts)
