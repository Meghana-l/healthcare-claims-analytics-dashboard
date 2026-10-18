# Healthcare Claims Program Evaluation & Risk Stratification

**Synthetic data project** demonstrating healthcare analytics methods: SQL-based
claims aggregation, a quasi-experimental (difference-in-differences) program
evaluation, and a machine-learning risk-stratification model.

> All member, diagnosis, and claims data are simulated (`numpy`-generated) to
> mimic the statistical structure of real healthcare claims data. No real
> patient or member data is used anywhere in this project.

## Scenario
A health plan launches a **Care Management Program (CMP)** for members with
chronic conditions (diabetes, CHF, COPD, hypertension, behavioral health).
Eligible members were split into a treated group (enrolled in CMP) and a
control group (eligible but not enrolled), observed for 9 months before and
9 months after launch — a member-month panel of 1,500 members / 27,000
observations.

## Methods & Stack
- **Data generation:** Python (`numpy`, `pandas`) — member risk scores,
  diagnosis categories, monthly ER visits (Poisson process) and allowed
  cost (right-skewed), with a known secular trend shared by both groups.
- **SQL layer:** SQLite — claims and member tables, with aggregation
  queries for group-level and diagnosis-level summaries.
- **Causal inference / program evaluation:** Difference-in-differences
  (DiD) via `statsmodels`:
  - OLS for allowed cost (cluster-robust SEs by member)
  - Poisson GLM for ER visit counts (cluster-robust SEs by member)
- **Risk stratification model:** Gradient-boosted classifier
  (`scikit-learn`) predicting which members become high-cost (top quintile)
  in the post-launch period, using pre-period claims history and risk score.

## Results

| Metric | Estimate | p-value |
|---|---|---|
| DiD effect on allowed cost | **-$93.54 / member-month (-10.6%)** | < 0.001 |
| DiD effect on ER utilization | **-23.2% relative rate** | < 0.001 |
| Risk model (held-out AUC) | **0.958** | — |

Members enrolled in the Care Management Program showed a statistically
significant reduction in both monthly allowed cost and ER utilization
relative to the control group, after the program launched — consistent with
the program narrowing the cost/utilization gap over time rather than the
groups simply diverging for unrelated reasons (the pre-period trends for
both groups track closely, which is what makes the DiD design valid here).

The risk model's top predictor of becoming high-cost was the member's
baseline chronic-condition risk score (importance 0.89), followed by
pre-period average cost (0.05) — i.e., recent cost history and overall
risk burden are the strongest early-warning signals, which is consistent
with how risk-stratification tools are used in care management in
practice.

## Live case-study page
`index.html` is a self-contained, deployable page (Chart.js via CDN, data
baked in at build time, no backend) that walks through all four modules in
plain English:
1. **Population & cost concentration** — who the highest-cost members are
2. **Cost driver analysis** — what actually predicts cost (regression)
3. **Risk-stratification model** — who becomes expensive next (ML)
4. **Program impact evaluation** — did the program work (difference-in-differences)

Deploy it by pushing this repo to GitHub and importing it in Vercel (root
directory, no build command needed — it's a static file).

## Files
- `claims_program_evaluation.py` — full pipeline (generate → SQL → descriptive → cost drivers → DiD → ML → dashboard data export)
- `index.html` — the deployable case-study page described above
- `dashboard_data.json` — all pre-computed results the page renders
- `claims_panel.csv` — synthetic member-month claims panel
- `claims.db` — SQLite database (members + claims_panel tables)
- `sql_group_summary.csv`, `sql_cost_by_diagnosis.csv` — SQL query outputs
- `results.json` — all model coefficients / p-values / AUC
- `er_trend.png`, `roc_curve.png` — supporting charts (static copies for the README)

## Why this design
Difference-in-differences is a standard quasi-experimental method for
evaluating programs when random assignment isn't available in the real
world (here, assignment is simplified to random for the synthetic data,
but the same DiD code runs unchanged against observational data with
matched controls). Clustering standard errors at the member level accounts
for repeated monthly observations per member.
