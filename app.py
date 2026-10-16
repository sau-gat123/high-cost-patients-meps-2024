"""High-cost patient dashboard (MEPS 2024). Uses only aggregate results in ./data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

D = Path(__file__).parent / "data"
st.set_page_config(page_title="High-Cost Patients: MEPS 2024", layout="wide")

@st.cache_data
def load():
    r = lambda n: pd.read_csv(D / n)
    return dict(
        facts=json.loads((D / "facts.json").read_text()),
        cm=json.loads((D / "confusion.json").read_text()),
        odds=r("odds_ratios.csv").rename(columns={"Unnamed: 0": "term"}),
        perf=r("test_performance.csv").set_index("model"),
        cv=r("cv_auc.csv").set_index("model"),
        roc=r("roc_curve.csv"), pr=r("pr_curve.csv"),
        ins=r("rate_insurance.csv"), chr=r("rate_chronic.csv"),
        phys=r("rate_physhealth.csv"), age=r("rate_age.csv"),
        hist=r("spend_hist.csv"),
    )
d = load(); f = d["facts"]
LABELS = {"age": "Age (per year)", "female": "Female", "race_hispanic": "Hispanic", "race_black": "Black",
          "race_asian": "Asian", "race_other": "Other race", "ins_public": "Public insurance",
          "ins_uninsured": "Uninsured", "pov_poor": "Poverty: poor", "pov_near_poor": "Poverty: near poor",
          "pov_low": "Poverty: low income", "pov_middle": "Poverty: middle income",
          "chronic_count": "Chronic conditions (each)", "phys_health": "Physical health (per step worse)",
          "mental_health": "Mental health (per step worse)"}
BLUE, ORANGE, GREY = "#2a6fbb", "#d9822b", "#8a8f98"

st.title("Who are the high-cost patients?")
st.caption("MEPS 2024 (HC-256), U.S. adults. High-cost = top 10% of total 2024 spending. "
           "Logistic regression vs. Naive Bayes.")

c = st.columns(4)
c[0].metric(f"Adults analysed ({f['n_households']:,} households)", f"{f['n_adults']:,}")
c[1].metric("High-cost cutoff (top 10%)", f"\\${f['cutoff']:,.0f}")
c[2].metric("Test ROC-AUC (logistic, moderate)", f"{d['perf'].loc['Logistic','ROC_AUC']:.3f}")
c[3].metric("Uninsured vs. private, adjusted OR", f"{d['odds'].set_index('term').loc['ins_uninsured','odds_ratio']:.3f}")

t1, t2, t3, t4, t5 = st.tabs(["Who is high-cost", "Drivers (odds ratios)", "Models compared", "Risk calculator", "Limits & method"])

with t1:
    st.subheader("Spending is very skewed")
    st.write(f"Median spending was \\${f['median']:,.0f} but the mean was \\${f['mean']:,.0f}; "
             f"{f['share_zero']*100:.1f}% of adults spent nothing.")
    h = d["hist"]
    fig = px.bar(h, x="log10_spend_mid", y="adults", labels={"log10_spend_mid": "log10 of spending ($), spenders only", "adults": "Adults"})
    fig.update_traces(marker_color=BLUE)
    fig.add_vline(x=np.log10(f["cutoff"]), line_dash="dash", line_color="red", annotation_text="High-cost cutoff")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Share of adults who are high-cost, by group")
    a, b = st.columns(2)
    for col, key, title in [(a, "chr", "Number of chronic conditions"), (b, "ins", "Insurance")]:
        g = d[key]
        fg = px.bar(g, x="group", y="share_high_cost", text=g["share_high_cost"].map("{:.1%}".format),
                    labels={"group": title, "share_high_cost": "Share high-cost"})
        fg.update_traces(marker_color=BLUE, textposition="outside"); fg.update_yaxes(tickformat=".0%")
        col.plotly_chart(fg, width="stretch")
    a, b = st.columns(2)
    for col, key, title, order in [(a, "phys", "Self-rated physical health", ["Excellent", "Very good", "Good", "Fair", "Poor"]),
                                   (b, "age", "Age group", None)]:
        g = d[key]
        fg = px.bar(g, x="group", y="share_high_cost", text=g["share_high_cost"].map("{:.1%}".format),
                    labels={"group": title, "share_high_cost": "Share high-cost"},
                    category_orders={"group": order} if order else None)
        fg.update_traces(marker_color=BLUE, textposition="outside"); fg.update_yaxes(tickformat=".0%")
        col.plotly_chart(fg, width="stretch")
    st.info(f"Only {f['uninsured_high']} of {f['uninsured_n']:,} uninsured adults were high-cost, "
            "so the uninsured estimate is imprecise. These are raw shares, not adjusted for other factors.")

with t2:
    st.subheader("Adjusted odds of being high-cost")
    st.write("Each bar is an odds ratio from survey-weighted logistic regression with household-clustered "
             "standard errors. Right of 1 = higher odds; left of 1 = lower odds. "
             "Reference groups: private insurance, non-Hispanic White, high income, male.")
    o = d["odds"][d["odds"]["term"] != "const"].copy()
    o["label"] = o["term"].map(LABELS)
    o["clear"] = np.where((o["ci_low"] > 1) | (o["ci_high"] < 1), "CI excludes 1", "CI includes 1")
    o = o.iloc[::-1]
    fig = px.scatter(o, x="odds_ratio", y="label", color="clear", log_x=True,
                     color_discrete_map={"CI excludes 1": BLUE, "CI includes 1": GREY},
                     error_x=o["ci_high"] - o["odds_ratio"], error_x_minus=o["odds_ratio"] - o["ci_low"],
                     labels={"odds_ratio": "Odds ratio (log scale, 95% CI)", "label": "", "clear": ""},
                     hover_data={"odds_ratio": ":.3f", "ci_low": ":.3f", "ci_high": ":.3f", "p_value": ":.4f", "label": False, "clear": False})
    fig.add_vline(x=1, line_dash="dash", line_color="red"); fig.update_layout(height=560)
    st.plotly_chart(fig, width="stretch")
    st.success("Hypothesis test (H1): uninsured adults have lower odds than privately insured adults. "
               f"OR = {o.set_index('term').loc['ins_uninsured','odds_ratio']:.3f}; one-sided p < 0.001, so H0 is rejected. "
               "This is an association, not proof of cause.")
    show = o.iloc[::-1][["label", "odds_ratio", "ci_low", "ci_high", "p_value"]].rename(
        columns={"label": "Predictor", "odds_ratio": "Odds ratio", "ci_low": "CI low", "ci_high": "CI high", "p_value": "p-value"})
    st.dataframe(show, hide_index=True, width="stretch")

with t3:
    st.subheader("Logistic regression vs. Naive Bayes (held-out test set)")
    st.write(f"Test set: {f['test_n']:,} adults, {f['test_pos']} truly high-cost ({f['test_pos']/f['test_n']:.1%}). "
             "Households never appear in both training and test data.")
    a, b = st.columns(2)
    fr = go.Figure()
    for m, col in [("Logistic", BLUE), ("NaiveBayes", ORANGE)]:
        s = d["roc"][d["roc"]["model"] == m]
        fr.add_trace(go.Scatter(x=s["fpr"], y=s["tpr"], mode="lines", name=f"{m} (AUC {d['perf'].loc[m,'ROC_AUC']:.3f})", line=dict(color=col)))
    fr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="black", dash="dash")))
    fr.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    a.plotly_chart(fr, width="stretch")
    fp = go.Figure()
    for m, col in [("Logistic", BLUE), ("NaiveBayes", ORANGE)]:
        s = d["pr"][d["pr"]["model"] == m]
        fp.add_trace(go.Scatter(x=s["recall"], y=s["precision"], mode="lines", name=f"{m} (AP {d['perf'].loc[m,'PR_AUC']:.3f})", line=dict(color=col)))
    fp.add_hline(y=f["test_pos"] / f["test_n"], line_dash="dash", line_color="black", annotation_text="Random")
    fp.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision")
    b.plotly_chart(fp, width="stretch")

    p = d["perf"].T.copy(); p.index.name = "Score"
    st.dataframe(p.rename(columns={"NaiveBayes": "Naive Bayes"}), width="stretch")
    cv = d["cv"]
    st.write("**5-fold grouped cross-validation (ROC-AUC, mean ± SD):** "
             + " | ".join(f"{m}: {cv.loc[m,'CV_AUC_mean']:.3f} ± {cv.loc[m,'CV_AUC_sd']:.3f}" for m in cv.index))
    st.subheader("Confusion matrices at the 0.5 cutoff")
    a, b = st.columns(2)
    for col, m in [(a, "Logistic"), (b, "NaiveBayes")]:
        k = d["cm"][m]
        z = [[k["tn"], k["fp"]], [k["fn"], k["tp"]]]
        fg = px.imshow(z, text_auto=True, x=["Predicted no", "Predicted yes"], y=["Actual no", "Actual yes"],
                       color_continuous_scale="Blues", title=m.replace("NaiveBayes", "Naive Bayes"))
        fg.update_layout(coloraxis_showscale=False); col.plotly_chart(fg, width="stretch")
    st.info("Both models rank high-cost adults well above chance. Logistic regression is slightly better on ROC-AUC and "
            "in cross-validation; PR-AUC is nearly tied. At the 0.5 cutoff logistic flags very few adults (rare outcome), "
            "while Naive Bayes flags many with more false alarms. Always saying 'not high-cost' is about 91% accurate, "
            "so accuracy is not used.")

with t4:
    st.subheader("Illustrative risk calculator")
    st.caption("Uses the coefficients of the survey-weighted logistic regression. For learning only, not for real decisions.")
    co = d["odds"].set_index("term")["coef"]
    a, b = st.columns(2)
    age = a.slider("Age", 18, 85, 50)
    sex = a.radio("Sex", ["Male", "Female"], horizontal=True)
    race = a.selectbox("Race/ethnicity", ["Non-Hispanic White", "Hispanic", "Black", "Asian", "Other"])
    ins = a.selectbox("Insurance", ["Any private", "Public only", "Uninsured"])
    pov = b.selectbox("Poverty level", ["High income", "Middle income", "Low income", "Near poor", "Poor"])
    chron = b.slider("Chronic conditions (0-9)", 0, 9, 1)
    ph = b.select_slider("Physical health", options=[1, 2, 3, 4, 5], value=2, format_func=lambda v: {1: "1 Excellent", 2: "2 Very good", 3: "3 Good", 4: "4 Fair", 5: "5 Poor"}[v])
    mh = b.select_slider("Mental health", options=[1, 2, 3, 4, 5], value=2, format_func=lambda v: {1: "1 Excellent", 2: "2 Very good", 3: "3 Good", 4: "4 Fair", 5: "5 Poor"}[v])
    x = {"age": age, "female": sex == "Female", "race_hispanic": race == "Hispanic", "race_black": race == "Black",
         "race_asian": race == "Asian", "race_other": race == "Other", "ins_public": ins == "Public only",
         "ins_uninsured": ins == "Uninsured", "pov_poor": pov == "Poor", "pov_near_poor": pov == "Near poor",
         "pov_low": pov == "Low income", "pov_middle": pov == "Middle income",
         "chronic_count": chron, "phys_health": ph, "mental_health": mh}
    z = co["const"] + sum(co[k] * float(v) for k, v in x.items())
    prob = 1 / (1 + np.exp(-z))
    st.metric(f"Estimated chance of being high-cost (sample average: {f['n_high']/f['n_adults']:.1%})", f"{prob:.1%}")
    st.progress(float(min(prob, 1.0)))
    st.caption("A model with ROC-AUC ~0.8 ranks people reasonably but cannot pinpoint individuals.")

with t5:
    st.subheader("Method")
    st.markdown(f"""
- **Data:** MEPS Household Component 2024 Full-Year Consolidated file (HC-256), AHRQ. Adults 18+ with a positive person weight; {f['n_adults']:,} adults in {f['n_households']:,} households after dropping 362 with missing predictors.
- **Outcome:** top 10% of total 2024 spending (TOTEXP24), cutoff \\${f['cutoff']:,.0f}.
- **Predictors:** age, sex, race/ethnicity, insurance, poverty level, number of chronic conditions (0-9), self-rated physical and mental health.
- **Inference:** survey-weighted logistic regression, standard errors clustered by household, one-sided test for the uninsured coefficient at 0.05.
- **Prediction:** logistic regression vs. Gaussian Naive Bayes, 80/20 split by household, 5-fold grouped cross-validation, seed 42.
""")
    st.subheader("Limits")
    st.markdown(f"""
- Associations from one year of cross-sectional, self-reported data; no causal claims.
- Only {f['uninsured_high']} high-cost adults were uninsured, so the insurance estimate has a wide confidence interval.
- Insurance may partly reflect people gaining coverage after a costly diagnosis.
- Moderate predictive ability: not suitable for decisions about individuals.
- Dashboard shows aggregate results only. No person-level MEPS records are included.
""")
    st.caption("Data source: Agency for Healthcare Research and Quality, MEPS HC-256. Download from meps.ahrq.gov.")
