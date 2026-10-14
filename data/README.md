# FinOptiX synthetic dataset

`demo_usage.csv` is a deterministic, seeded multi-cloud workload simulation with one daily observation per active resource. Default generation produces 365 days of history for 40 resources (roughly 14,000 records; newly deployed resources appear only after launch). It is synthetic and must not be represented as actual provider billing data.

## Reproduce

From `backend/` after installing `requirements.txt`:

```bash
python -m app.generate_data
```

The default seed is `2026`. Calling `generate(seed=2026, days=365, resources=40)` twice produces identical ordered records. `days` must be at least 180 and there must be at least ten resources so every required scenario is present. The reporting end date is fixed at 2026-10-07 for reproducible project demos.

## Schema

| Column | Meaning |
|---|---|
| `timestamp` | UTC date of the daily resource observation. |
| `cloud_provider` | AWS, Azure, or GCP. |
| `service` | Provider service family (compute, database, or block storage). |
| `region` | Provider region identifier. |
| `resource_id` | Stable synthetic resource key. |
| `resource_type` | `virtual_machine`, `database`, or `block_storage`. |
| `instance_type` | Provider SKU/size; price is based on the SKU catalog in the generator. |
| `cpu_utilization` | Daily mean CPU utilization, percent. |
| `memory_utilization` | Daily mean memory utilization, percent. |
| `network_usage` | Daily network egress/throughput, synthetic GB. |
| `storage_usage` | Used fraction of allocated storage, percent. |
| `allocated_storage_gb` | Provisioned storage capacity used by the storage cost calculation. |
| `running_hours` | Resource runtime in the observed day, 0–24. |
| `compute_monthly_cost` | Modeled compute component before the small measurement noise and planted incident multiplier. |
| `storage_monthly_cost` | Modeled provisioned storage plus usage-sensitive surcharge. |
| `network_monthly_cost` | Modeled egress charge after the illustrative free allowance. |
| `monthly_cost` | Monthly-normalized INR run rate for that observation; ETL divides by 30 to derive `daily_cost`. Cost is modeled from runtime, service/SKU hourly rate, regional multiplier, usage-sensitive compute component, provisioned/used storage, and billable network egress. |
| `department` | Synthetic ownership grouping. |
| `project` | Synthetic workload/project grouping. |
| `environment` | `production` or `development`. |
| `resource_criticality` | Synthetic normalized criticality from 0 (lower) to 1 (higher), used as an explicit risk factor. |
| `scenario_type` | Scenario metadata used to inspect coverage; never a model feature. |
| `is_anomaly` | Ground truth: whether this resource-day belongs to a planted cost-spike incident. |
| `optimization_opportunity` | Ground truth: whether the resource scenario is intentionally eligible for an optimization rule. |
| `optimization_type` | Ground-truth action: `downsize_instance`, `schedule_shutdown`, `rightsize_storage`, or `none`. |
| `expected_risk` | Ground-truth scenario risk label (`LOW`, `MEDIUM`, `HIGH`, or `NONE`). |
| `expected_saving` | Synthetic expected monthly saving in INR for the planted opportunity; zero for non-opportunity rows. |

## Scenario design

Each resource is assigned one of ten deterministic scenario families, with daily measurements perturbed by small seeded noise. Cost is causally linked to resource attributes and measurements rather than sampled independently.

- **Normal:** mid-range utilization, long-lived service, ordinary weekday/runtime and usage variation.
- **Underutilized:** low CPU and memory; a downsize label and expected saving are supplied.
- **Idle:** very low runtime and utilization; shutdown label and larger expected saving.
- **Over-provisioned:** low utilization with a large provisioned size and storage headroom; downsize label.
- **Highly utilized:** sustained high CPU/memory and usage, with high expected operational risk.
- **Sudden cost spike:** a 3-day incident close to the end of the series; runtime, CPU, memory, network, and cost rise together and `is_anomaly` is true for those rows. The incident also applies a modeled cost factor to represent an unobserved billing-rate/usage event; this is synthetic and should not be interpreted as measured unit-price change.
- **Gradual cost increase:** a small daily demand ramp compounds across the year.
- **Seasonal usage:** annual sinusoidal demand plus a weekday/weekend pattern.
- **Newly deployed:** no rows before launch 75 days before the end; demand ramps during initial warm-up.
- **Occasional peak:** two-day demand bursts every 23 days create temporary CPU/memory/network peaks.

The pricing model is illustrative: provider-specific SKU hourly prices are multiplied by regional factors; compute run rate responds to runtime and utilization; storage charges use allocated capacity and observed fill; network charges apply above a modeled free allowance. The model does not reproduce discount plans, taxes, credits, reservations, or exact provider billing semantics.

## Modeling and leakage controls

`backend/app/core.py` explicitly enumerates features. Labels and scenario metadata are preserved by ETL for evaluation but are not used to fit models.

ETL derives `daily_cost = monthly_cost / 30`, per-resource rolling 7-day sum/average, rolling 30-day sum/average, lagged 30-day cost growth, utilization score, estimated idle-hours proxy, cost-per-usage, resource-efficiency score, weekday and calendar month. Rolling features are computed in timestamp order per resource and use only current/past observations.

| Task | Target / detector | Model input features | Excluded columns |
|---|---|---|---|
| Cost forecasting | Daily sum of derived `daily_cost` | Lag-1 daily spend, lag-7 daily spend, day-of-week, trend index | `is_anomaly`, `optimization_opportunity`, `optimization_type`, `expected_risk`, `expected_saving`, `scenario_type`, identifiers and ownership metadata |
| Anomaly detection | Isolation Forest predicted outlier; evaluate against `is_anomaly` after fitting | Derived prior-30-day percentage changes in cost, CPU, memory, network, and runtime, plus storage utilization | All five evaluation labels and `scenario_type`; labels are only read after predictions for precision/recall/F1. Detector contamination is fixed at 0.2%, independent of labels. |
| Optimization evaluation | Rule outputs compared to ground truth | CPU, memory, idle hours, storage utilization, monthly cost and resource history | Labels are not rule inputs; `optimization_type`, `optimization_opportunity`, `expected_risk`, and `expected_saving` are for offline evaluation only |
| What-if simulation | Estimated post-action cost/utilization/risk | Selected resource's current measured monthly cost, utilization, and explicit proposed action | Ground-truth labels and scenario metadata |

The forecasting and anomaly evaluation use chronological holdouts; forecasting features are lagged and labels are excluded. The evaluation CLI generates a distinct seed and scenario assignment for its held-out dataset. For research reporting, treat results against these labels as **synthetic controlled evaluation**, not real-world model validation. The production-facing anomaly endpoint is refit on the available history; only its evaluation endpoint reports holdout performance.
