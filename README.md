# Healthcare Claims Analytics Dashboard

**Live:** https://healthcare-claims-analytics-dashboa.vercel.app/

A healthcare claims analysis built on simulated data: where the cost sits, what drives it, who's likely to get expensive next, and whether a care program actually worked.

> ⚠️ All member, diagnosis, and claims figures are synthetically generated for methodology demonstration. No real patient or member data is used anywhere in this project.

---

## What's in the dashboard

### 1. Summary numbers

<img width="1147" height="187" alt="Screenshot 2026-10-18 131709" src="https://github.com/user-attachments/assets/1e1e0d8b-0d78-4298-8362-ab0455173301" />


Four headline numbers at the top:

- **Members analyzed (1,500)** — the size of the simulated population, tracked monthly for 18 months.
- **Cost held by top risk tier (49.5%)** — the highest-risk third of members account for almost half of all spending. A small group drives most of the cost.
- **Program cost impact (-10.6%)** — after a care management program launched, enrolled members' costs dropped 10.6% compared to a similar group that wasn't enrolled.
- **Risk model accuracy, AUC (0.958)** — how well a model can tell in advance which members are about to become expensive. 1.0 would be a perfect model, 0.5 would be a random guess, so 0.958 is very accurate.

Every card has a **View data** button that opens the exact numbers behind it, so nothing here has to be taken on faith.

---

### 2. Cost concentration by risk tier

<img width="1121" height="348" alt="Screenshot 2026-10-18 132622" src="https://github.com/user-attachments/assets/ac30be25-6077-4242-ad28-4047cbdf5794" />


Members are split into three equal-sized groups (Low, Medium, High) based on how chronically ill they are. Each group is exactly a third of the population, but they don't spend the same amount.

The High group is a third of the members but close to half of the total cost. The takeaway: a small, identifiable group of members is responsible for most of the spending, which is exactly who a cost-reduction program should focus on.

---

### 3. Cost by diagnosis category

<img width="1121" height="371" alt="Screenshot 2026-10-18 131728" src="https://github.com/user-attachments/assets/2eeb6604-9fa7-4b3c-9628-194ccb878d78" />


This breaks down average monthly cost by condition (CHF, hypertension, diabetes, behavioral health, COPD). The costs across conditions are fairly close to each other.

The takeaway: the diagnosis a member has doesn't explain cost nearly as well as how chronically at-risk they are overall. Two diabetics can cost very differently depending on their overall health risk, so the diagnosis label alone isn't a reliable way to predict cost.

---

### 4. Cost driver regression

<img width="1119" height="394" alt="Screenshot 2026-10-18 131741" src="https://github.com/user-attachments/assets/7d0d1c8b-9471-47ab-9775-f38bb9258297" />


This answers "what actually drives cost, once you account for everything at once." A statistical model looks at age, risk score, and diagnosis together and measures how much each one independently pushes cost up.

The chronic risk score is by far the biggest driver (around $192 per point, per member, per month). Age matters too, but much less. The individual diagnosis categories (shown in grey) don't add much once risk score is already accounted for. The model explains 92% of the variation in cost (R² = 0.922), which is a strong fit.

---

### 5. Risk-stratification model

<img width="1128" height="375" alt="Screenshot 2026-10-18 131753" src="https://github.com/user-attachments/assets/c32136f6-3595-4f56-975d-67b380945605" />


This is a prediction model: using only data from *before* the program started, can it tell which members are about to become expensive? It's tested only on members it never saw during training, so the result reflects real predictive power, not memorization.

- **ROC curve (left)**: shows how well the model separates future high-cost members from everyone else. The green line sitting well above the diagonal dashed line means the model is doing much better than a random guess.
- **Feature importance (right)**: shows what the model actually relies on to make its predictions. The chronic risk score dominates, meaning the model would still work on a brand new group of members, not just this one, since it isn't depending on something specific to this population.

---

### 6. Program impact (difference-in-differences)

<img width="1118" height="448" alt="Screenshot 2026-10-18 131804" src="https://github.com/user-attachments/assets/76343f10-4915-414f-8fdd-67d03606c2fc" />


This checks whether the care management program actually worked, the right way. You can't just compare a member's cost before vs. after the program, because costs drift for everyone over time regardless (seasons, inflation, etc). Instead, this compares the *change* in enrolled members against the *change* in a similar group that wasn't enrolled, over the same period.

Before the launch (the dashed line), both lines move together, which is what makes this a fair comparison. After launch, the enrolled group's line drops noticeably below the control group's line.

- **Cost effect:** -10.6% (about $94 less per member, per month)
- **ER visit effect:** -23.2% fewer ER visits
- **Significance:** p < 0.001 for both, meaning this gap is very unlikely to be due to chance.

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/statsmodels-4C72B0?style=for-the-badge" alt="statsmodels">
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite">
</p>
<p>
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5">
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3">
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript">
  <img src="https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white" alt="Chart.js">
  <img src="https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white" alt="Vercel">
</p>
