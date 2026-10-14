# FinOptiX: Explainable and Risk-Aware AI for Cloud Cost Optimization

## Abstract

FinOptiX is a multi-cloud FinOps decision-support prototype. Its contribution is an explainable, risk-aware what-if framework that evaluates catalog-supported optimization alternatives against observed utilization and cost before implementation. The system reports estimated post-action cost, savings, utilization, constraint violations, heuristic risk, evidence quality, and assumptions. The prototype combines synthetic data, Pandas ETL, cost analytics, forecasting, anomaly detection and measured root-cause evidence, transparent recommendation rules, FastAPI, React, optional PostgreSQL persistence, and source-attributed local document retrieval.

## Problem, motivation, and research question

Organizations need to control cloud expenditure while preserving performance and reliability. Spend dashboards identify cost patterns but do not by themselves establish whether a resizing or scheduling action is safe. The project evaluates whether explicit simulation and risk evidence can help a human compare actions before applying them.

**Research question:** Can a risk-aware what-if optimization framework identify cloud cost-saving opportunities while minimizing potential performance and operational risks?

Cloud dashboards, cost forecasting, anomaly detection, retrieval, and rightsizing are established capabilities. They are not claimed as individual novelties. This project tests the integration of observed workload evidence, alternative simulation, hard performance constraints, and transparent ranking.

## Existing approaches and limitations

Provider tools and FinOps practices offer cost allocation, forecasting, anomaly signals, and resource recommendations. Their outputs depend on provider pricing, discount programs, workload telemetry and service-specific behavior. A student prototype cannot reproduce those provider algorithms. A cost-only ranking also ignores operational risk. FinOptiX therefore makes assumptions and constraints visible and leaves changes to human review.

Authoritative background sources indexed for the knowledge assistant include the [AWS Well-Architected COST06-BP03 guidance](https://docs.aws.amazon.com/wellarchitected/2023-10-03/framework/cost_type_size_number_resources_metrics.html), [Azure Advisor cost recommendations](https://learn.microsoft.com/en-us/azure/advisor/advisor-cost-recommendations), [Google Cloud Well-Architected cost optimization](https://docs.cloud.google.com/architecture/framework/cost-optimization/optimize-resource-usage), and the [FinOps Framework](https://www.finops.org/framework/). They provide design context; FinOptiX does not claim to duplicate these services' algorithms.

## Proposed methodology

1. Generate seeded correlated resource-day observations with planted workload patterns and evaluation-only labels.
2. Validate, deduplicate and transform records into a shared cleaned Pandas representation.
3. Aggregate costs and evaluate lag-feature forecasts on time-ordered data.
4. Detect anomalies using Isolation Forest and compare with rolling MAD. Explain a signal with actual before/recent windows, service/region spend contributions, runtime, resource counts, usage and available bill components.
5. Enumerate catalog-supported actions per resource. Compute expected utilization and proposed component costs from configured capacity and illustrative prices.
6. Reject insufficient-evidence or constraint-violating alternatives. Score feasible alternatives using heuristic workload risk, evidence quality and a configurable risk-aware priority formula.
7. Compare savings-only and risk-aware rankings over the identical feasible action/resource set.

## Architecture and implementation

The data path is CSV or seeded generator → shared Pandas ETL → optional PostgreSQL → analytics/model/simulator functions → FastAPI → React. PostgreSQL schema initialization is repeatable. CSV imports persist resources, observations, and costs when `FINOPTIX_DATABASE_URL` is configured. API-generated anomalies, recommendations, simulations, savings estimates, predictions, and audit records are written to their corresponding tables. In local no-database mode, the same analytics functions run against the CSV-backed DataFrame.

The React application includes overview, cost explorer, forecast, anomaly/root-cause, recommendation/ranking, multi-option simulator, savings and copilot views. Structured copilot requests call backend calculations. Knowledge questions retrieve source-attributed passages from Markdown documents in `knowledge/` using TF-IDF. Responses are extractive by default; optional OpenAI-compatible generation can be configured by environment variables and falls back to retrieval if unavailable. Embedding/vector-database services are not implemented.

## Dataset and pricing assumptions

The default file has 13,440 rows over 365 days for 40 resources. Scenarios include normal, underutilized, idle, over-provisioned, highly utilized, short cost spike, gradual increase, seasonal use, new deployment and occasional peaks. The generator creates costs from service/SKU hourly rate, region factor, runtime, utilization, provisioned storage, fill and network egress. Evaluation labels are excluded from model feature lists and simulator inputs.

`backend/catalog/pricing_catalog.json` contains INR hourly SKU rates, capacities, region factors, transitions and billing assumptions. The rates are illustrative rather than current provider prices. Cost estimates use a 30-day projection; compute includes an assumed utilization multiplier, storage uses provisioned capacity and a small utilization-sensitive surcharge, and egress applies a modeled allowance. No discounts, reservations, credits, taxes or actual time-varying prices are modeled.

For resizing, expected CPU and memory are scaled by old/new vCPU and GiB capacity ratios. Historical p95 checks gate options at configured CPU/memory thresholds. This linear approximation does not predict application latency, queueing, burst behavior or cross-resource contention. Shutdown scheduling uses observed eligible hours subject to daily minimum runtime; daily aggregates cannot confirm that a workload is inactive during the proposed window. Storage optimization uses p99 observed use plus headroom. Each result includes assumptions and limitations.

## Mathematical formulation

### Workload risk

`Risk = 0.50 × performance_pressure + 0.25 × variability + 0.25 × criticality`

Risk is a 0–100 heuristic, not a probability. Performance pressure is capped `100 × max(projected p95 CPU / 85, projected p95 memory / 90)`. Variability is capped `100 × max(CPU coefficient of variation, memory coefficient of variation)`. Criticality uses an explicit normalized value when available or an environment mapping. Configured classes are LOW ≤34.9, MEDIUM ≤64.9 and HIGH otherwise. If minimum observation, coverage, or missingness criteria fail, the system returns an insufficient-evidence warning and no feasible alternatives.

### Evidence confidence

`Confidence = 0.20 × observation_volume + 0.20 × coverage + 0.15 × stability + 0.15 × action_consistency + 0.15 × peak_metrics + 0.15 × price_reliability`

Each factor is normalized to [0,1]. Illustrative catalog price reliability is 0.55. The value is an evidence-quality score, not probability of optimization success.

### Risk-aware ranking

`Priority = 100 × (0.45 × min(monthly_saving/current_cost, 1) + 0.10 × non-overlapping evidence quality − 0.45 × (risk/100)²)`

The evidence quality term is the normalized weighted sum of observation volume, coverage, peak-metric availability, and price reliability. It omits stability and recommendation consistency/action fit because those overlap with variability and performance pressure in the risk score. The quadratic penalty separates high-risk actions more strongly. Savings-only uses gross monthly savings. Both rankings receive identical feasible alternatives; no ground-truth labels enter either ranker. The weights and risk thresholds are configurable in the catalog and should be calibrated with real outcomes before use.

## Experimental setup and results

`backend/app/evaluation.py` uses a deterministic development dataset (seed 2026) and a separately seeded, shifted-scenario held-out dataset (seed 106755). Forecast candidate selection uses a chronological validation tail inside the training period; model comparisons use the later, untouched time holdout. Candidates are a 7-day seasonal naive, a four-week weekday median, and a Random Forest using lags 1/7/14/28, weekday and rolling mean. Forecast results remain weak on this synthetic holdout, including negative R²; this change improves model selection hygiene and visibility, not a claim of high accuracy. Anomaly metrics separately report Isolation Forest, a causal 4× rolling-MAD cost rule, and their union. The union raises recall on this seeded holdout at a higher false-positive rate. Results are in [`evaluation-results.json`](evaluation-results.json) and can be reproduced with `cd backend && python -m app.evaluation`.

| Evaluation | Held-out result | Interpretation |
|---|---:|---|
| Forecast MAE / RMSE / MAPE / R² | ₹2,091.12 / ₹4,813.77 / 7.57% / -0.092 | Random Forest selected by validation; test fit remains weak |
| Four-week weekday median MAE / RMSE / MAPE / R² | ₹1,961.92 / ₹4,892.90 / 6.77% / -0.128 | Lower MAE/MAPE on this test block, slightly worse RMSE and R²; do not use test results to tune selection |
| Seasonal-naive MAE / RMSE / MAPE / R² | ₹2,199.49 / ₹5,077.90 / 7.94% / -0.215 | Simple prior-week comparator |
| Isolation Forest precision / recall / F1 / FPR | 0.333 / 0.167 / 0.222 / 0.003 | It missed most planted anomalies in the small holdout |
| Hybrid detector precision / recall / F1 / FPR | 0.194 / 1.000 / 0.324 / 0.035 | Union with rolling-MAD raises recall and false positives |
| Recommendation precision / false recommendation rate | 0.750 / 0.250 | 6 of 8 selected resources matched synthetic opportunity labels |
| Feasible / rejected options | 15 / 22 | 0.595 rejected-option rate among enumerated alternatives |
| Non-overlapping monthly savings estimate | ₹17,467.92 | One selected action per recommended resource; modeled, not realized |
| Mean selected heuristic risk | 44.04 / 100 | Rule score, not measured incident probability |

The risk-aware ranking changed the top result on the held-out pool: savings-only ranked an ₹8,461.09/month shutdown option first, while risk-aware ranking put a ₹5,420.66/month storage option first due to its stronger evidence and lower risk score. The simulator reports zero post-gate performance violations because it rejects alternatives that breach configured p95 limits; this is a property of the gate, not an independently measured production performance result.

These results expose material weaknesses: all three forecast R² values are negative, validation-selected Random Forest did not win on test MAE, hybrid anomaly false positives are appreciable, and recommendation false positives are nonzero. This is why the page shows the candidate comparison and the model-selection/holdout distinction. All labels are generated from scenarios; results do not prove generalization to real cloud estates or production savings.

## Evaluation metrics

- Forecast: MAE, RMSE, MAPE, R², and comparison with seasonal-naive previous-week cost.
- Anomaly: precision, recall, F1 and false-positive rate for Isolation Forest, rolling-MAD threshold and the deployed hybrid union.
- Optimization: recommendation precision/false recommendation rate versus synthetic labels, feasible/rejected option rate, modeled non-overlapping savings and heuristic risk.
- Ranking: savings-only and risk-aware ordering over the same action/resource set.
- Simulator: p95 constraint rejection; actual predicted-versus-realized cost validation remains future work because no cloud changes have been applied.

## Tests

The backend unit/API suite covers catalog validation, pricing, cost savings, capacity scaling, shutdown limits, evidence, risk, ranking, missing data, seeded generation, API behavior, analytics grounding, retrieval citations, and CSV-upload-to-simulation flow. A PostgreSQL round-trip test runs when `FINOPTIX_DATABASE_URL` is configured; it is skipped otherwise. The React production build was verified with `pnpm build`.

## Limitations and future work

Current rates are illustrative; source billing may be monthly-normalized records rather than invoice line items; actual hourly scheduling evidence is absent; linear CPU/memory scaling omits latency and correlated bottlenecks; no real provider connector, cloud action executor, authentication, tenant boundary, model registry, managed vector index, or embeddings are implemented. Optional external LLM generation was not tested against a live endpoint. The PostgreSQL integration could not be exercised in this environment without a running PostgreSQL service. Docker runtime validation also requires Docker, which was unavailable in the build environment.

Next experiments should calibrate simulator predictions from approved canary changes, add application SLO and burst telemetry, improve held-out scenarios, tune and evaluate anomaly detection without labels leaking into features, use real pricing exports and commitment modeling, compare richer forecasting baselines, and add authenticated multi-tenant controls.

### Recommended follow-up work

1. Validate proposed savings and safety against real billing histories and measured outcomes after approved changes.
2. Improve forecasting and anomaly detection; current held-out results show negative forecast R² and low Isolation Forest recall.
3. Replace illustrative prices with verified provider pricing and compare estimated costs with actual bills.
4. Run PostgreSQL integration and restart-persistence checks in a configured database environment.
5. Add read-only cloud billing connectors, authentication, tenant isolation, access controls, and monitoring before production use.

For a repeatable visual demo, upload `data/demo_usage_alternate.csv` through Cost Explorer. The API retains the last uploaded CSV as `data/uploaded_usage.csv`; removing or moving that file before restarting the backend returns the app to the default generated demo. These datasets are synthetic.

### Placement interview summary

“FinOptiX analyzes cloud cost and usage data, then compares potential optimizations before implementation. Its key idea is to make savings, workload constraints, risk, confidence, and assumptions visible together, so teams can review safer alternatives instead of ranking actions by savings alone. The prototype uses synthetic data, so the next step is validation against real cloud histories and measured outcomes.”

## Conclusion

FinOptiX implements a transparent proposal workflow around the central what-if contribution: enumerate alternatives, calculate catalog-based estimates, enforce data/performance constraints, show evidence and assumptions, and compare savings-only with risk-aware ranking. Current synthetic evaluation is useful for testing and demonstrates the approach, while its weak model scores and synthetic-only scope make clear that production effectiveness has not been established.
