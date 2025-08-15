```markdown
# DS-AutoAdvisor — **Seasoned DS Edition**  
**Production-Ready, Scalable, and Easy-to-Glance Workflow**

> This is the same streamlined pipeline, now re-evaluated with a **seasoned data scientist’s eye**.  
> Every step includes: **what to do**, **why it matters**, and **how to scale it safely in production**.

---

## Operating Principles (read once, apply everywhere)
- **Reproducibility:** Pin package versions, set random seeds, version data/model/config (MLflow/DVC).
- **Idempotency:** All steps safe to re-run; writes are **atomic**; outputs include run hash/timestamp.
- **Observability:** Log inputs, metrics, artifacts; trace lineage (data → code → model).
- **Data Contracts:** Schemas are explicit and validated (Great Expectations/pydantic).
- **Security/Privacy:** PII handling (mask/hash/salt), access control, audit logs.
- **Cost/Latency Budgets:** Decide targets early (p95 latency, cost per 1k predictions, throughput).

---

## Phase 1: Data Profiling

**Goal**  
See the *true* shape of the data before making choices that lock you in.

### What to Do
1. **Dataset Overview**: shape, dtypes, missing %, memory.  
2. **Variable Types**: numerical/categorical/datetime; cardinality of categoricals.  
3. **Summary Stats**: mean/median/std/percentiles; uniques.  
4. **Distributions**: histograms/KDE/boxplots for skew/outliers.  
5. **Associations**:
   - num–num: Pearson (linear), Spearman (monotonic), Kendall (small n).
   - num–cat: ANOVA or Kruskal–Wallis.
   - cat–cat: Cramér’s V, Theil’s U.
6. **Target**: class balance / target skew; obvious leakage checks.

**Why this matters**  
Profiling determines **encoding**, **scaling**, **model family**, and rough **effort vs payoff**.

**Practical Test Selection**  
- Linear & ~normal → **Pearson**; monotonic/outliers → **Spearman**; small n/ties → **Kendall**.  
- n > 5k → prefer **visual** checks; formal normality tests over-trigger.

**ML Context & Honest Suggestions**  
- Trees forgive noise; linear/SVM/NN do not.  
- If time-constrained: check **target balance**, **cardinality**, **obvious leakage** first.

**Scalability & Production**  
- **Large data**: sample stratified 1–5% for EDA; keep sample seed & manifest.  
- **Storage**: use columnar (Parquet/Delta), partition by date/entity.  
- **Validation**: Great Expectations on raw + post-ingest; fail fast in CI.

---

## Phase 2: Data Cleaning & Preprocessing

**Goal**  
Deliver consistent, ML-ready data without hiding business realities.

### 2.1 Outlier Detection
- **IQR** (univariate, robust), **Z-Score** (normal), **Isolation Forest** (high-dim), **LOF** (local density), **Elliptic Envelope** (Gaussian), **Ensemble** (vote).

**Why this matters**  
Outliers bias linear models & metrics; they can also signal data quality defects.

**Practical Test Selection**  
- Small p & n<10k → **IQR/Z**.  
- Complex/high-dim → **Isolation Forest**; clustered density → **LOF**.  
- Well-justified Gaussian → **Elliptic**.  
- Production → **Majority** vote (IQR + IF).

**ML Context & Honest Suggestions**  
- Linear/SVM/NN: cap/transform/flag.  
- GBMs: usually robust; don’t over-clean unless outliers are systemic.

**Scalability & Production**  
- **Streaming**: cap by rolling quantiles (windowed) to avoid lookahead.  
- **Batch**: compute bounds per partition; store thresholds in config registry.  
- **Logging**: record outlier rates per column; alert on spikes.

---

### 2.2 Missing Data Handling
- **Simple** (mean/median/mode/constant), **KNN**, **Iterative (MICE-like)**.

**Why this matters**  
Imputation affects distributions and downstream fairness/performance.

**Practical Test Selection**  
- <5% missing → **simple**.  
- 5–20% + correlated features → **KNN/Iterative**.  
- >40% missing → redesign/drop; collect more data.

**ML Context & Honest Suggestions**  
- Fit imputers on **train only**.  
- LightGBM handles NaNs; keep it simple if using GBMs.

**Scalability & Production**  
- Store imputer **artifacts** (params/statistics) in model registry.  
- For streaming, prefer **constant/median** for speed; re-train imputers on schedule.

---

### 2.3 Scaling & Transformation
| Method            | Best For | Notes |
|-------------------|---------|-------|
| StandardScaler    | Gaussian-like | default for linear/SVM/NN |
| MinMaxScaler      | [0,1] range | sensitive to extremes |
| RobustScaler      | Outliers | median/IQR |
| Yeo–Johnson       | Skewed ± | normalizing |
| Box–Cox           | Skewed + | x>0 |
| log/log1p         | Heavy right skew | log1p handles 0 |

**Why this matters**  
Margin/distance-based models depend on scale; trees do not.

**Practical Test Selection**  
- Linear/SVM/NN → **scale**; heavy skew → **YJ/BC/log1p**.  
- Trees → **skip scaling**.

**Scalability & Production**  
- Persist scaler params; verify **train/serve skew** with periodic checks.  
- Online inference: apply the same scaler artifact; reject on schema/NaN drift.

---

### 2.4 Categorical Encoding

| Method              | Description | Use Case |
|---------------------|-------------|----------|
| One-Hot             | Binary per category | Low cardinality |
| Ordinal             | Ordered integers | True order |
| **Label Encoding**  | Category → id | Trees; quick high-card baselines |
| **Binary Encoding** | Category → bits | Med–High cardinality (10–200) |
| Target Encoding     | Mean target per category (CV-safe) | High cardinality + signal |
| Frequency Encoding  | Category frequency/proportion | High cardinality, simple/robust |

**Why this matters**  
Encoding impacts dimensionality, multicollinearity, and leakage risk.

**Threshold Guidelines**  
- **Cardinality**: <10 → One-Hot; 10–50 → **Binary/Ordinal**; 50–500 → **Target/Binary/Frequency**; >500 → Target/Frequency/Hashing.  
- **VIF after encoding (linear)**: **<5 good**, **5–10 monitor**, **>10 re-encode** (Binary/Frequency).  
- **Model family**: Trees → Label/Binary/Target (CatBoost native); Linear/SVM/NN → avoid Label unless ordered.

**Practical Test Selection**  
1) Measure cardinality. 2) Pick per model family. 3) Target-encode with **CV**. 4) Check **VIF** & retrim.

**Scalability & Production**  
- Maintain encoder **state** (mappings/stats) with version tags.  
- Handle **unseen categories** (fallback bucket).  
- Monitor **feature explosion** (cap p ~300–500 unless justified).

---

### 2.5 Feature Engineering
- **Datetime** (year/month/dow/hour, is_weekend), **interactions** (×, +, ratios), **binning** (quantile/KMeans), **aggregations** (group stats).

**Why this matters**  
Good FE moves the needle more than most HPO tweaks.

**Scalability & Production**  
- Define FE in **declarative config**; same codepath offline/online.  
- Validate **offline/online parity**; unit-test critical features.  
- Cache heavy aggregations (feature store) with TTL.

---

## Phase 3: Statistical Assumption Testing & Advisory

**Goal**  
Only run when using **assumption-sensitive** models; otherwise don’t waste cycles.

| Assumption         | Tests | Matters For | If Violated |
|--------------------|-------|-------------|-------------|
| Normality          | Shapiro (n≤5k), JB/D’Agostino, Anderson | OLS inference | Transform or use robust/trees |
| Homoscedasticity   | Breusch–Pagan, White | OLS | Robust SE, transform, WLS |
| Multicollinearity  | **VIF** (<5 good; >10 act), |corr|>0.9 | Linear/Logistic | Drop/merge, **Ridge/Lasso**, PCA |
| Linearity          | Partial residuals, Harvey–Collier | OLS | Polynomials/splines; switch to trees |
| Independence       | Durbin–Watson ≈2 | Time-related OLS | Lags/diffs; time-series models |
| Class Imbalance    | Majority share/ratio | Classification | Class weights, resampling, PR-AUC |

**Why this matters**  
Prevents false confidence from invalid inference and brittle models.

**Practical Test Selection**  
- Using linear/logistic? Run **VIF**, **BP/White**, **linearity check**.  
- Large n? Prefer **residual plots** over binary normality decisions.

**Scalability & Production**  
- VIF is O(p³): **cap features** (≤50) or sample columns.  
- Log **assumption outcomes**; tie to model cards for audits.

---

## Phase 4: Model Advisory, Training & Optimization

**Goal**  
Pick the right complexity, tune just enough, and keep the path to production clean.

### 4.1 Model Selection Logic
1. Infer target type (≤20 unique ints → likely classification).  
2. Check assumptions **only** if using linear/logistic.  
3. Baseline → **sanity** (Linear/Logistic or small tree).  
4. Escalate:
   - Multicollinearity → **Ridge/Lasso/ElasticNet**.  
   - Non-linearity → **RF → GBMs** (XGB/LGBM/CatBoost).  
   - Imbalance → **class weights/balanced ensembles**.

**Why this matters**  
Prevents premature complexity and supports explainability.

**Scalability & Production**  
- Prefer **LightGBM** for speed/scale; CatBoost for categorical-heavy.  
- Track **fit time**, **memory**, **p95 inference** in MLflow; set SLAs.

---

### 4.2 Hyperparameter Optimization (HPO)
| Method        | Use When | Notes |
|---------------|----------|-------|
| Grid Search   | Tiny spaces | Deterministic; expensive |
| Random Search | Big spaces, low budget | 50–100 trials strong baseline |
| Optuna (TPE)  | Medium–large spaces | Adaptive; ≥100 trials; tune pruning |

**Why this matters**  
Most gains come early; avoid diminishing returns.

**Practical Selection**  
Defaults → small **Random** → **Optuna** if ROI warrants.  
Use **repeated/stratified CV** to reduce noise.

**Scalability & Production**  
- **Parallelize** trials (Ray/Dask/Spark/Optuna study).  
- **Early stopping**; set time/CPU budgets.  
- Persist best params; lock seeds for repeatability.

---

## Phase 5: Evaluation, Interpretation & Validation

**Goal**  
Prove value, explain drivers, ensure stability under drift and load.

### 5.1 Metrics
- **Regression**: MAE (robust), RMSE (penalizes large errors), R²/Adj-R².  
- **Classification**: Accuracy (balanced), **F1**, ROC-AUC, **PR-AUC** (imbalanced), Balanced Accuracy.  
- **Ranking**: MAP@K, NDCG.

**Why this matters**  
Align metrics with business risk (e.g., PR-AUC when positives are rare).

**Scalability & Production**  
- Log **confidence intervals** (bootstrap).  
- Track **calibration**; serve thresholds via config.

---

### 5.2 Validation
- **K-Fold** (default 5), **Stratified** for class imbalance, **TimeSeriesSplit** for temporal order.  
- Learning/validation curves to diagnose bias–variance.

**Why this matters**  
Good CV reduces surprise in production.

**Scalability & Production**  
- CV folds can be **sharded**; cache fold splits & seeds.  
- Time-based splits obey **data-time contracts**; forbid leakage with validators.

---

### 5.3 Interpretability
- **SHAP** (global/local), **Permutation importance**, PDP/ICE.

**Why this matters**  
Trust, compliance, and faster iteration on features.

**Scalability & Production**  
- SHAP is heavy: compute on **samples** or use **TreeSHAP**.  
- Log **top-k drivers** per model version for monitoring dashboards.

---

### 5.4 Stability & Robustness
- **Bootstrap** CIs; **Noise tests** (ε levels); **Feature dropout** sensitivity.

**Why this matters**  
Catches brittle models before they hit real traffic.

**Scalability & Production**  
- Automate periodic **shadow eval** on fresh data.  
- Canary deploy with **guardrails** (latency, error rate, data drift).

---

## Monitoring, Drift & Retraining (Production Loop)

- **Data Drift**: KS/PSI/JSD on features & predictions; warn/critical thresholds.  
- **Concept Drift**: performance drop vs baseline; trigger retrain.  
- **Skew Checks**: **offline vs online** feature distributions (same encoder/scaler).  
- **Ops KPIs**: p95 latency, throughput, cost/1k preds, error rate.  
- **Retraining Policy**: schedule + event-driven (drift, data volume %, performance drop).  
- **Rollouts**: A/B, canary; fallback model/version; blue/green infra.

---

## Quick Thresholds Cheat Sheet

- **VIF**: <5 good, 5–10 monitor, >10 fix.  
- **|corr|**: >0.9 likely redundant for linear models.  
- **Imbalance**: majority >90% → treat as imbalanced; use PR-AUC/F1, class weights.  
- **Outliers (IQR)**: 1.5× (screen), 3.0× (strict).  
- **Encoded p**: p > ~300–500 → consider Binary/Target/Frequency; or feature selection.  
- **Latency SLO**: set p95 target early (e.g., <50ms API); track alongside model metrics.

---

## Minimal Daily-Use Flow (90 seconds)

1. Profile: target balance, dtypes, missing %, cardinality (stratified sample).  
2. Choose encoding (thresholds above) and **scale only if** linear/SVM/NN.  
3. Outliers: cap/flag if linear; skip heavy ops for GBMs.  
4. Baseline → CV → quick Random Search; log everything.  
5. Report: business metric + PR-AUC/F1 (if imbalanced) + top SHAP drivers.  
6. If shipping: register model, attach encoders/scalers, add drift monitors & SLOs.

---

## Tooling Checklist (copy/paste into README)

- **Data**: Parquet/Delta, partitioning, schema registry, GE tests.  
- **Orchestration**: Airflow/Prefect; idempotent tasks; backfill-safe.  
- **Tracking**: MLflow (params, metrics, artifacts, model cards).  
- **Registry**: Model + preprocessor bundle; stage → prod promotion.  
- **Serving**: Batch (Spark/EMR) or Online (FastAPI/Triton); autoscale.  
- **Monitoring**: Prometheus/Grafana + drift service; alerts to Slack/Email.  
- **Security**: Secrets manager; role-based access; PII policies.  
- **Cost**: Budgets & alerts; per-step runtime and $ tracked.

---
```
