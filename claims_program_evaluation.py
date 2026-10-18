"""
Healthcare Claims Program Evaluation & Risk Stratification
------------------------------------------------------------
SYNTHETIC DATA -- no real patient, member, or claims data is used anywhere
in this project. Member IDs, diagnoses, costs, and utilization are
simulated to mimic the statistical structure of real healthcare claims
data for portfolio / skills-demonstration purposes.

What this does:
 1. Generates a member-month claims panel (pre/post a Care Management
    Program launch, treated vs. control members).
 2. Loads it into a SQLite database and runs SQL aggregation queries
    (the "SQL" layer).
 3. Runs a Difference-in-Differences (DiD) causal evaluation of the
    program's effect on ER utilization and allowed cost -- a classic
    quasi-experimental program-evaluation design.
 4. Trains a gradient-boosting risk-stratification model that flags
    members likely to become high-cost in the next period.
 5. Writes out results.json + a README-ready results table.
"""

import json
import sqlite3
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1. GENERATE SYNTHETIC MEMBER-MONTH CLAIMS PANEL
# ----------------------------------------------------------------------
N_MEMBERS = 1500
N_MONTHS = 18          # months 0-8 = pre-launch, 9-17 = post-launch
LAUNCH_MONTH = 9

dx_categories = ["Diabetes", "CHF", "COPD", "Hypertension", "Behavioral Health"]

members = pd.DataFrame({
    "member_id": np.arange(1, N_MEMBERS + 1),
    "age": rng.integers(35, 85, N_MEMBERS),
    "dx_category": rng.choice(dx_categories, N_MEMBERS, p=[0.25, 0.15, 0.15, 0.30, 0.15]),
    "risk_score": rng.gamma(shape=2.0, scale=1.3, size=N_MEMBERS),   # baseline chronic-risk score
})
# Treated = enrolled in the Care Management Program (random assignment
# within risk band, like a real-world quasi-experiment with eligible
# non-participants as controls)
members["treated"] = rng.binomial(1, 0.5, N_MEMBERS)

rows = []
for _, m in members.iterrows():
    base_er_lambda = 0.12 + 0.05 * m.risk_score
    base_cost_mean = 250 + 180 * m.risk_score + 2.0 * (m.age - 35)
    for month in range(N_MONTHS):
        post = int(month >= LAUNCH_MONTH)
        # small common secular trend (flu season / inflation) both groups feel
        seasonal = 0.05 * np.sin(month / 2.0) + 0.01 * month
        er_lambda = base_er_lambda * (1 + seasonal)
        cost_mean = base_cost_mean * (1 + 0.01 * month)

        # TRUE program effect (only treated members, only post-launch)
        if m.treated and post:
            er_lambda *= 0.78          # true 22% relative reduction in ER rate
            cost_mean *= 0.90          # true 10% reduction in monthly cost

        er_visits = rng.poisson(max(er_lambda, 0.01))
        cost = max(rng.normal(cost_mean, cost_mean * 0.35), 0)

        rows.append((m.member_id, month, post, m.treated, m.age,
                     m.dx_category, round(m.risk_score, 3),
                     er_visits, round(cost, 2)))

panel = pd.DataFrame(rows, columns=[
    "member_id", "month", "post", "treated", "age", "dx_category",
    "risk_score", "er_visits", "allowed_cost"
])
panel.to_csv("claims_panel.csv", index=False)

# ----------------------------------------------------------------------
# 2. SQL LAYER -- load into SQLite, run aggregation queries
# ----------------------------------------------------------------------
conn = sqlite3.connect("claims.db")
panel.to_sql("claims_panel", conn, if_exists="replace", index=False)
members.to_sql("members", conn, if_exists="replace", index=False)

sql_summary = pd.read_sql_query("""
    SELECT
        treated,
        post,
        COUNT(DISTINCT member_id)               AS n_members,
        ROUND(AVG(er_visits), 3)                AS avg_er_visits_per_member_month,
        ROUND(AVG(allowed_cost), 2)              AS avg_allowed_cost_per_member_month,
        ROUND(SUM(allowed_cost), 2)              AS total_allowed_cost
    FROM claims_panel
    GROUP BY treated, post
    ORDER BY treated, post
""", conn)
sql_summary.to_csv("sql_group_summary.csv", index=False)

dx_cost = pd.read_sql_query("""
    SELECT m.dx_category,
           ROUND(AVG(c.allowed_cost), 2) AS avg_cost_per_member_month,
           ROUND(AVG(c.er_visits), 3)    AS avg_er_visits_per_member_month
    FROM claims_panel c
    JOIN members m ON m.member_id = c.member_id
    GROUP BY m.dx_category
    ORDER BY avg_cost_per_member_month DESC
""", conn)
dx_cost.to_csv("sql_cost_by_diagnosis.csv", index=False)
conn.close()

# ----------------------------------------------------------------------
# 3. DIFFERENCE-IN-DIFFERENCES CAUSAL EVALUATION
# ----------------------------------------------------------------------
# Cost model (OLS, cluster-robust SE by member)
cost_model = smf.ols(
    "allowed_cost ~ treated * post + risk_score + age",
    data=panel
).fit(cov_type="cluster", cov_kwds={"groups": panel["member_id"]})

did_cost_coef = cost_model.params["treated:post"]
did_cost_pval = cost_model.pvalues["treated:post"]
control_post_mean_cost = panel.loc[(panel.treated == 0) & (panel.post == 1), "allowed_cost"].mean()
did_cost_pct = did_cost_coef / control_post_mean_cost * 100

# ER utilization model (Poisson GLM, cluster-robust SE by member)
er_model = smf.glm(
    "er_visits ~ treated * post + risk_score + age",
    data=panel, family=sm.families.Poisson()
).fit(cov_type="cluster", cov_kwds={"groups": panel["member_id"]})

did_er_coef = er_model.params["treated:post"]
did_er_pval = er_model.pvalues["treated:post"]
did_er_pct = (np.exp(did_er_coef) - 1) * 100   # % change in ER rate

# ----------------------------------------------------------------------
# 4. RISK-STRATIFICATION MODEL (predict next-period high-cost member)
# ----------------------------------------------------------------------
member_level = panel.groupby("member_id").agg(
    age=("age", "first"),
    risk_score=("risk_score", "first"),
    treated=("treated", "first"),
    pre_avg_cost=("allowed_cost", lambda s: s[panel.loc[s.index, "post"] == 0].mean()),
    pre_avg_er=("er_visits", lambda s: s[panel.loc[s.index, "post"] == 0].mean()),
    post_avg_cost=("allowed_cost", lambda s: s[panel.loc[s.index, "post"] == 1].mean()),
).reset_index()

cost_threshold = member_level["post_avg_cost"].quantile(0.80)
member_level["high_cost_next_period"] = (member_level["post_avg_cost"] >= cost_threshold).astype(int)

features = ["age", "risk_score", "pre_avg_cost", "pre_avg_er", "treated"]
X = member_level[features]
y = member_level["high_cost_next_period"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = GradientBoostingClassifier(random_state=42)
clf.fit(X_train, y_train)
auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])

importances = dict(zip(features, clf.feature_importances_.round(3)))

# ----------------------------------------------------------------------
# 5. PLOTS
# ----------------------------------------------------------------------
trend = panel.groupby(["month", "treated"])["er_visits"].mean().unstack()
plt.figure(figsize=(6, 4))
plt.plot(trend.index, trend[0], label="Control")
plt.plot(trend.index, trend[1], label="Treated (Care Mgmt Program)")
plt.axvline(LAUNCH_MONTH, color="gray", linestyle="--", label="Program launch")
plt.xlabel("Month")
plt.ylabel("Avg ER visits / member-month")
plt.title("ER Utilization Trend: Treated vs. Control")
plt.legend()
plt.tight_layout()
plt.savefig("er_trend.png", dpi=150)
plt.close()

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Risk Model ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 6. SAVE RESULTS
# ----------------------------------------------------------------------
results = {
    "n_members": int(N_MEMBERS),
    "n_member_months": int(len(panel)),
    "did_cost_effect_dollars_per_member_month": round(did_cost_coef, 2),
    "did_cost_effect_pct": round(did_cost_pct, 1),
    "did_cost_pvalue": round(did_cost_pval, 4),
    "did_er_effect_pct": round(did_er_pct, 1),
    "did_er_pvalue": round(did_er_pval, 4),
    "risk_model_auc": round(float(auc), 3),
    "risk_model_feature_importance": importances,
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

# ----------------------------------------------------------------------
# 7. MODULE A -- POPULATION & COST CONCENTRATION (descriptive)
# ----------------------------------------------------------------------
pre = panel[panel.post == 0].groupby("member_id").agg(
    avg_cost=("allowed_cost", "mean"), risk_score=("risk_score", "first"),
    dx_category=("dx_category", "first"), age=("age", "first"),
).reset_index()

pre["risk_tier"] = pd.qcut(pre["risk_score"], 3, labels=["Low", "Medium", "High"])
tier_summary = pre.groupby("risk_tier", observed=True).agg(
    n_members=("member_id", "count"),
    avg_monthly_cost=("avg_cost", "mean"),
).reset_index()
tier_summary["total_cost_share_pct"] = (
    (tier_summary["n_members"] * tier_summary["avg_monthly_cost"]) /
    (tier_summary["n_members"] * tier_summary["avg_monthly_cost"]).sum() * 100
).round(1)
tier_summary["n_members_share_pct"] = (tier_summary["n_members"] / tier_summary["n_members"].sum() * 100).round(1)

high_tier_cost_share = float(tier_summary.loc[tier_summary.risk_tier == "High", "total_cost_share_pct"].iloc[0])
high_tier_member_share = float(tier_summary.loc[tier_summary.risk_tier == "High", "n_members_share_pct"].iloc[0])

dx_summary = pre.groupby("dx_category").agg(
    n_members=("member_id", "count"), avg_monthly_cost=("avg_cost", "mean")
).reset_index().sort_values("avg_monthly_cost", ascending=False)

# ----------------------------------------------------------------------
# 8. MODULE B -- COST DRIVER ANALYSIS (cross-sectional OLS, pre-period only)
# ----------------------------------------------------------------------
driver_model = smf.ols("avg_cost ~ age + risk_score + C(dx_category)", data=pre).fit()
driver_coefs = driver_model.params.drop("Intercept").round(2).to_dict()
driver_pvals = driver_model.pvalues.drop("Intercept").round(4).to_dict()
driver_r2 = round(driver_model.rsquared, 3)

# clean up labels for display
def clean_label(k):
    return (k.replace("C(dx_category)[T.", "Diagnosis: ")
             .replace("]", "")
             .replace("risk_score", "Chronic risk score (per 1-pt)")
             .replace("age", "Age (per year)"))

driver_table = sorted(
    [{"factor": clean_label(k), "dollars_per_member_month": v, "pvalue": driver_pvals[k]}
     for k, v in driver_coefs.items()],
    key=lambda d: -abs(d["dollars_per_member_month"])
)

# ----------------------------------------------------------------------
# 9. DASHBOARD DATA EXPORT (everything the HTML needs, pre-computed)
# ----------------------------------------------------------------------
dashboard = {
    "headline": {
        "n_members": int(N_MEMBERS),
        "n_member_months": int(len(panel)),
        "high_tier_member_share_pct": high_tier_member_share,
        "high_tier_cost_share_pct": high_tier_cost_share,
        "did_cost_pct": results["did_cost_effect_pct"],
        "did_er_pct": results["did_er_effect_pct"],
        "risk_auc": results["risk_model_auc"],
    },
    "module_a_population": {
        "risk_tiers": tier_summary.to_dict(orient="records"),
        "diagnosis_cost": dx_summary.to_dict(orient="records"),
    },
    "module_b_cost_drivers": {
        "r_squared": driver_r2,
        "drivers": driver_table,
    },
    "module_c_risk_model": {
        "auc": results["risk_model_auc"],
        "feature_importance": results["risk_model_feature_importance"],
        "roc_fpr": fpr.round(3).tolist()[::max(1, len(fpr)//40)],
        "roc_tpr": tpr.round(3).tolist()[::max(1, len(tpr)//40)],
    },
    "module_d_program_evaluation": {
        "did_cost_dollars": results["did_cost_effect_pct"] and results["did_cost_effect_dollars_per_member_month"],
        "did_cost_pct": results["did_cost_effect_pct"],
        "did_cost_pvalue": results["did_cost_pvalue"],
        "did_er_pct": results["did_er_effect_pct"],
        "did_er_pvalue": results["did_er_pvalue"],
        "trend_months": trend.index.tolist(),
        "trend_control": trend[0].round(3).tolist(),
        "trend_treated": trend[1].round(3).tolist(),
        "launch_month": LAUNCH_MONTH,
    },
}
dashboard["raw_sample"] = (
    panel[panel.member_id.isin(rng.choice(panel.member_id.unique(), 60, replace=False))]
    .sort_values(["member_id", "month"])
    .head(220)
    .assign(group=lambda d: d.treated.map({1: "Treated", 0: "Control"}),
            period=lambda d: d.post.map({1: "Post-launch", 0: "Pre-launch"}))
    [["member_id", "month", "group", "period", "age", "dx_category", "risk_score", "er_visits", "allowed_cost"]]
    .to_dict(orient="records")
)

with open("dashboard_data.json", "w") as f:
    json.dump(dashboard, f, indent=2)

print(json.dumps(results, indent=2))
print("\nSQL group summary:\n", sql_summary)
print("\nCost by diagnosis:\n", dx_cost)
print("\nRisk tier summary:\n", tier_summary)
print("\nCost drivers:\n", driver_table)
