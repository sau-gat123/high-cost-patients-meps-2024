"""
Classifying High-Cost Patients: Logistic Regression vs Naive Bayes (+ optional KNN)
Data : MEPS 2024 Full-Year Consolidated file (HC-256), AHRQ  (https://meps.ahrq.gov)

How to run:
    1. Put h256.csv in the same folder as this script (or change DATA_PATH).
    2. pip install pandas numpy statsmodels scikit-learn matplotlib
    3. python analysis.py
Tables are saved in ./results and figures in ./figures.

Steps in this script (the usual data-analysis process):
    PART 1  Load and clean the data
    PART 2  Exploratory data analysis (EDA) and plots
    PART 3  Check the model assumptions
    PART 4  Inference model: logistic regression (tests the hypothesis)
    PART 5  Prediction models: train, tune, and evaluate (Logistic, Naive Bayes, KNN)
    PART 6  Summary
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # draw figures to files, no pop-up windows
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import (GroupShuffleSplit, GroupKFold,
                                     StratifiedGroupKFold, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score, f1_score,
                             roc_curve, precision_recall_curve, confusion_matrix)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = "h256.csv"                # <-- change if your file is somewhere else
INCLUDE_KNN = False

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def header(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


# =====================================================================
# PART 1: LOAD AND CLEAN THE DATA
# =====================================================================
header("PART 1: LOAD AND CLEAN")

CHRONIC_FLAGS = ["HIBPDX", "CHDDX", "STRKDX", "EMPHDX", "CHOLDX",
                 "CANCERDX", "DIABDX_M18", "ARTHDX", "ASTHDX"]

KEEP = (["DUID", "PERWT24F", "TOTEXP24",
         "AGELAST", "SEX", "RACETHX",
         "INSCOV24", "POVCAT24",
         "RTHLTH53", "MNHLTH53"] + CHRONIC_FLAGS)

df = pd.read_csv(DATA_PATH, usecols=KEEP)          # load only the columns we need
print(f"Rows in file: {len(df)}")

# Keep adults (18+) with a positive survey weight
df = df[(df["AGELAST"] >= 18) & (df["PERWT24F"] > 0)].copy()
print(f"Adults with weight > 0: {len(df)}")

# MEPS uses negative numbers (-1, -7, -8, -9) for missing answers -> make them NaN
predictor_cols = ["SEX", "RACETHX", "INSCOV24", "POVCAT24",
                  "RTHLTH53", "MNHLTH53"] + CHRONIC_FLAGS
for col in predictor_cols:
    df.loc[df[col] < 0, col] = np.nan

# Chronic condition count (1 = yes, 2 = no). If any answer is missing, the count is NaN.
df["chronic_count"] = (df[CHRONIC_FLAGS] == 1).sum(axis=1)
df.loc[df[CHRONIC_FLAGS].isna().any(axis=1), "chronic_count"] = np.nan

# Drop rows with any missing predictor
model_cols = ["AGELAST", "SEX", "RACETHX", "INSCOV24", "POVCAT24",
              "RTHLTH53", "MNHLTH53", "chronic_count"]
n_before = len(df)
df = df.dropna(subset=model_cols).copy()
print(f"Dropped for missing predictors: {n_before - len(df)}")
print(f"Final analysis sample: {len(df)} adults in {df['DUID'].nunique()} households")

# Readable variables / dummy columns (reference groups in comments)
df["age"] = df["AGELAST"]
df["female"] = (df["SEX"] == 2).astype(int)                       # reference: male
df["ins_public"] = (df["INSCOV24"] == 2).astype(int)              # reference: any private
df["ins_uninsured"] = (df["INSCOV24"] == 3).astype(int)
df["race_hispanic"] = (df["RACETHX"] == 1).astype(int)            # reference: non-Hispanic White
df["race_black"] = (df["RACETHX"] == 3).astype(int)
df["race_asian"] = (df["RACETHX"] == 4).astype(int)
df["race_other"] = (df["RACETHX"] == 5).astype(int)
df["pov_poor"] = (df["POVCAT24"] == 1).astype(int)                # reference: high income
df["pov_near_poor"] = (df["POVCAT24"] == 2).astype(int)
df["pov_low"] = (df["POVCAT24"] == 3).astype(int)
df["pov_middle"] = (df["POVCAT24"] == 4).astype(int)
df["phys_health"] = df["RTHLTH53"]                                # 1 excellent ... 5 poor
df["mental_health"] = df["MNHLTH53"]

# Target: high-cost patient = top 10% of total spending (unweighted, analysis sample)
cutoff = df["TOTEXP24"].quantile(0.90)
df["high_cost"] = (df["TOTEXP24"] >= cutoff).astype(int)
print(f"90th percentile cutoff (TOTEXP24): ${cutoff:,.0f}")
print(f"Share high-cost: {df['high_cost'].mean():.3f}")
print(f"Share with $0 spending: {(df['TOTEXP24'] == 0).mean():.3f}")

FEATURES = ["age", "female",
            "race_hispanic", "race_black", "race_asian", "race_other",
            "ins_public", "ins_uninsured",
            "pov_poor", "pov_near_poor", "pov_low", "pov_middle",
            "chronic_count", "phys_health", "mental_health"]

X = df[FEATURES].astype(float)
y = df["high_cost"]
groups = pd.factorize(df["DUID"])[0]                  # household id as numbers
weights = df["PERWT24F"] / df["PERWT24F"].mean()      # survey weights scaled to average 1


# =====================================================================
# PART 2: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================
header("PART 2: EXPLORATORY DATA ANALYSIS")

# 2.1 Summary table of all variables
summary = X.describe().T[["mean", "std", "min", "max"]].round(3)
summary.to_csv("results/table1_summary.csv")
print(summary)

# 2.2 High-cost vs other adults (average of each variable)
compare = df.groupby("high_cost")[FEATURES].mean().T.round(3)
compare.columns = ["not_high_cost", "high_cost"]
compare.to_csv("results/table2_high_cost_vs_other.csv")
print("\nAverages by group:")
print(compare)

# 2.3 Figure 1: why spending needs special handling (skewed -> log scale)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(df["TOTEXP24"].clip(upper=df["TOTEXP24"].quantile(0.99)), bins=50)
ax[0].set_title("Total spending (clipped at 99th percentile)")
ax[0].set_xlabel("TOTEXP24 ($)")
pos = df.loc[df["TOTEXP24"] > 0, "TOTEXP24"]
ax[1].hist(np.log10(pos), bins=50)
ax[1].axvline(np.log10(cutoff), color="red", linestyle="--", label="High-cost cutoff")
ax[1].set_title("Spending among spenders (log10 scale)")
ax[1].set_xlabel("log10(TOTEXP24)")
ax[1].legend()
plt.tight_layout()
plt.savefig("figures/fig1_spending_distribution.png", dpi=150)
plt.close()

# 2.4 Figure 2: share of high-cost by insurance group and chronic conditions
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ins_labels = {1: "Private", 2: "Public only", 3: "Uninsured"}
rate_ins = df.groupby("INSCOV24")["high_cost"].mean().rename(index=ins_labels)
ax[0].bar(rate_ins.index, rate_ins.values)
ax[0].set_title("Share high-cost by insurance")
ax[0].set_ylabel("Share high-cost")
chronic_grp = df["chronic_count"].clip(upper=4).astype(int).astype(str).replace({"4": "4+"})
rate_chr = df.groupby(chronic_grp)["high_cost"].mean()
ax[1].bar(rate_chr.index, rate_chr.values)
ax[1].set_title("Share high-cost by number of chronic conditions")
ax[1].set_xlabel("Chronic conditions")
plt.tight_layout()
plt.savefig("figures/fig2_high_cost_by_group.png", dpi=150)
plt.close()

# 2.5 Figure 3: correlation heatmap of predictors
corr = X.corr()
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURES)))
ax.set_yticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES, rotation=90)
ax.set_yticklabels(FEATURES)
plt.colorbar(im)
ax.set_title("Correlation between predictors")
plt.tight_layout()
plt.savefig("figures/fig3_correlation_heatmap.png", dpi=150)
plt.close()
print("\nSaved figures 1-3 in ./figures")


# =====================================================================
# PART 3: CHECK THE MODEL ASSUMPTIONS
# =====================================================================
header("PART 3: ASSUMPTION CHECKS")

# 3.1 Enough events per predictor (rule of thumb: at least 10)
events = int(y.sum())
print(f"3.1 Events (high-cost): {events}, predictors: {len(FEATURES)}, "
      f"events per predictor: {events / len(FEATURES):.0f}")

# 3.2 Multicollinearity (VIF). Rule of thumb: VIF above 5-10 is a concern.
X_const = sm.add_constant(X)
vif = pd.DataFrame({
    "variable": X_const.columns,
    "VIF": [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
}).query("variable != 'const'").round(2)
vif.to_csv("results/table3_vif.csv", index=False)
print("\n3.2 VIF:")
print(vif.to_string(index=False))

# 3.3 Linearity of the logit for numeric predictors (Box-Tidwell idea)
# Add x*log(x) terms; if their p-values are small, the straight-line assumption is doubtful.
# (With ~15,000 rows even tiny departures look 'significant', so also judge by eye.)
bt = X.copy()
bt["age_x_ln_age"] = bt["age"] * np.log(bt["age"])
bt["chronic_x_ln"] = (bt["chronic_count"] + 1) * np.log(bt["chronic_count"] + 1)
bt_fit = sm.Logit(y, sm.add_constant(bt)).fit(disp=0)
print("\n3.3 Box-Tidwell p-values (small = non-linear):")
print(bt_fit.pvalues[["age_x_ln_age", "chronic_x_ln"]].round(4))

# 3.4 Independence: people in the same household are related
hh_size = df.groupby("DUID").size()
print(f"\n3.4 {len(hh_size)} households, average {hh_size.mean():.2f} adults per household "
      f"-> we use household-clustered standard errors.")

# 3.5 Naive Bayes assumes predictors are independent: list the strongest correlations
pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
         .stack().rename("corr").reset_index())
pairs["abs_corr"] = pairs["corr"].abs()
top_pairs = pairs.sort_values("abs_corr", ascending=False).head(6)[["level_0", "level_1", "corr"]]
top_pairs.columns = ["variable_1", "variable_2", "corr"]
top_pairs.round(3).to_csv("results/table4_top_correlations.csv", index=False)
print("\n3.5 Strongest correlations between predictors (Naive Bayes independence check):")
print(top_pairs.round(3).to_string(index=False))


# =====================================================================
# PART 4: INFERENCE MODEL - LOGISTIC REGRESSION (tests the hypothesis)
# =====================================================================
header("PART 4: LOGISTIC REGRESSION (INFERENCE)")
# H0: after controlling for the other variables, being uninsured has no effect
#     on the odds of being high-cost (coefficient = 0).
# H1: uninsured adults have LOWER odds than privately insured adults (coefficient < 0).

X_sm = sm.add_constant(X)

# Main model: survey-weighted, standard errors clustered by household
main = sm.GLM(y, X_sm, family=sm.families.Binomial(), freq_weights=weights).fit(
    cov_type="cluster", cov_kwds={"groups": groups})

# Check model: same thing without survey weights
check = sm.GLM(y, X_sm, family=sm.families.Binomial()).fit(
    cov_type="cluster", cov_kwds={"groups": groups})


def odds_table(fit):
    ci = fit.conf_int()
    out = pd.DataFrame({
        "coef": fit.params,
        "odds_ratio": np.exp(fit.params),
        "ci_low": np.exp(ci[0]),
        "ci_high": np.exp(ci[1]),
        "p_value": fit.pvalues,
    })
    return out.round(4)


tab_main = odds_table(main)
tab_main.to_csv("results/table5_logit_weighted.csv")
odds_table(check).to_csv("results/table6_logit_unweighted.csv")
print("Weighted logistic regression (odds ratios):")
print(tab_main)

# The hypothesis test
b = main.params["ins_uninsured"]
p_two = main.pvalues["ins_uninsured"]
p_one = p_two / 2 if b < 0 else 1 - p_two / 2        # one-sided test for coefficient < 0
print("\nH1 TEST (uninsured vs privately insured):")
print(f"  coefficient = {b:.3f}, odds ratio = {np.exp(b):.3f}")
print(f"  95% CI for odds ratio = ({tab_main.loc['ins_uninsured', 'ci_low']:.3f}, "
      f"{tab_main.loc['ins_uninsured', 'ci_high']:.3f})")
print(f"  two-sided p = {p_two:.4f}, one-sided p (H1: coef < 0) = {p_one:.4f}")
print("  Decision at 0.05:", "reject H0" if p_one < 0.05 else "fail to reject H0")
print(f"  Unweighted check: odds ratio = {np.exp(check.params['ins_uninsured']):.3f}, "
      f"p = {check.pvalues['ins_uninsured']:.4f}")

# Figure 4: odds ratios with confidence intervals
plot_tab = tab_main.drop(index="const").iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 7))
ax.errorbar(plot_tab["odds_ratio"], range(len(plot_tab)),
            xerr=[plot_tab["odds_ratio"] - plot_tab["ci_low"],
                  plot_tab["ci_high"] - plot_tab["odds_ratio"]],
            fmt="o", capsize=3)
ax.axvline(1, color="red", linestyle="--")
ax.set_yticks(range(len(plot_tab)))
ax.set_yticklabels(plot_tab.index)
ax.set_xscale("log")
ax.set_xlabel("Odds ratio (log scale), 95% CI")
ax.set_title("Odds of being a high-cost patient")
plt.tight_layout()
plt.savefig("figures/fig4_odds_ratios.png", dpi=150)
plt.close()


# =====================================================================
# PART 5: PREDICTION MODELS - TRAIN, TUNE, EVALUATE
# =====================================================================
header("PART 5: PREDICTION MODELS")

# 5.1 Split into train (80%) and test (20%) BY HOUSEHOLD, so family members
#     never appear on both sides (this prevents data leakage).
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
g_train = groups[train_idx]
print(f"Train: {len(X_train)} rows ({y_train.mean():.3f} high-cost) | "
      f"Test: {len(X_test)} rows ({y_test.mean():.3f} high-cost)")

# 5.2 Define the models
models = {
    # Logistic regression (scaled; very weak penalty so it behaves like the statsmodels one)
    "Logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=2000)),
    # Naive Bayes (Gaussian). Assumes predictors are independent.
    "NaiveBayes": GaussianNB(),
}

if INCLUDE_KNN:
    # KNN needs scaled features; choose K using grouped cross-validation on training data only
    knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
    grid = GridSearchCV(knn_pipe,
                        {"kneighborsclassifier__n_neighbors": [5, 11, 21, 31, 51, 101, 151, 201]},
                        cv=GroupKFold(n_splits=5), scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train, groups=g_train)
    best_k = grid.best_params_["kneighborsclassifier__n_neighbors"]
    print(f"KNN: best K chosen by cross-validation = {best_k}")
    models["KNN"] = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k))

# 5.3 Train each model and evaluate on the held-out test set
prevalence = y_test.mean()
rows, curves = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]            # predicted chance of being high-cost

    # (a) Standard 0.5 cut-off
    pred_05 = (p >= 0.5).astype(int)
    # (b) Flag the 10% of people with the highest predicted risk (a practical use case)
    top_cut = np.quantile(p, 0.90)
    pred_top = (p >= top_cut).astype(int)

    rows.append({
        "model": name,
        "ROC_AUC": roc_auc_score(y_test, p),
        "PR_AUC": average_precision_score(y_test, p),
        "Brier": brier_score_loss(y_test, p),
        "precision@0.5": precision_score(y_test, pred_05, zero_division=0),
        "recall@0.5": recall_score(y_test, pred_05),
        "F1@0.5": f1_score(y_test, pred_05),
        "precision@top10%": precision_score(y_test, pred_top, zero_division=0),
        "recall@top10%": recall_score(y_test, pred_top),
    })
    curves[name] = p
    print(f"\n{name} confusion matrix at 0.5 (rows = actual, cols = predicted):")
    print(confusion_matrix(y_test, pred_05))

results = pd.DataFrame(rows).set_index("model").round(3)
results.to_csv("results/table7_test_performance.csv")
print("\nTEST-SET PERFORMANCE")
print(results.T)
print(f"\nBaselines: 'always say no' accuracy = {1 - prevalence:.3f}; "
      f"random-guess PR_AUC = {prevalence:.3f}; random-guess ROC_AUC = 0.500")

# 5.4 Cross-validation on ALL data (grouped by household) - a more stable estimate
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, model in models.items():
    aucs = []
    for tr, te in cv.split(X, y, groups):
        model.fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], model.predict_proba(X.iloc[te])[:, 1]))
    cv_rows.append({"model": name, "CV_AUC_mean": np.mean(aucs), "CV_AUC_sd": np.std(aucs)})
cv_results = pd.DataFrame(cv_rows).set_index("model").round(3)
cv_results.to_csv("results/table8_cv_auc.csv")
print("\n5-FOLD GROUPED CROSS-VALIDATION (ROC-AUC)")
print(cv_results)

# 5.5 Figure 5: ROC and precision-recall curves on the test set
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for name, p in curves.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, p):.2f})")
    prec, rec, _ = precision_recall_curve(y_test, p)
    ax[1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y_test, p):.2f})")
ax[0].plot([0, 1], [0, 1], "k--", label="Random")
ax[0].set_xlabel("False positive rate")
ax[0].set_ylabel("True positive rate")
ax[0].set_title("ROC curves (test set)")
ax[0].legend()
ax[1].axhline(prevalence, color="k", linestyle="--", label="Random")
ax[1].set_xlabel("Recall")
ax[1].set_ylabel("Precision")
ax[1].set_title("Precision-recall curves (test set)")
ax[1].legend()
plt.tight_layout()
plt.savefig("figures/fig5_roc_pr_curves.png", dpi=150)
plt.close()


# =====================================================================
# PART 6: SUMMARY
# =====================================================================
header("PART 6: SUMMARY")
print("Tables are in ./results and figures in ./figures.")
print("Write your conclusions from YOUR printed output:")
print("  - H1: is the uninsured odds ratio below 1, and is the p-value below 0.05?")
print("  - Which model has the higher ROC_AUC / PR_AUC / recall, and why?")
print("  - Do the VIF and correlation checks explain Naive Bayes' behaviour?")
