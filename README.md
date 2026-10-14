# FinOptiX

**Explainable and Risk-Aware AI for Cloud Cost Optimization** is a B.Tech project prototype. Its research contribution is an explainable framework that compares cloud optimization actions before implementation using cost, expected utilization, evidence quality, and operational risk. Dashboards, forecasting, anomaly detection, RAG, and rightsizing are established techniques and are not claimed as individually novel.

> Pricing is synthetic and illustrative. The application does not execute cloud changes. Do not use its estimates as current provider quotes or financial advice.

## Research problem

Cloud cost tools commonly surface spend trends or optimization candidates, but an action that saves the most can violate workload or availability constraints. FinOptiX asks: **Can a risk-aware what-if optimization framework identify cloud cost-saving opportunities while minimizing potential performance and operational risks?** It is evaluated as a decision-support prototype with deterministic synthetic scenarios; it does not establish production accuracy.

## Implemented system

- Seeded, correlated multi-cloud daily records, 365-day demo dataset, resource scenarios and evaluation-only labels.
- Pandas validation, deduplication, missing-value flags, cost/utilization features, and date-based analytics.
- Configurable AWS, Azure and GCP illustrative catalog with region factors, capacity, hourly rates, billing assumptions, and allowed downsizing transitions.
- Multi-option what-if comparison for resizing, scheduled shutdown, and storage capacity reduction. Infeasible options include rejection reasons.
- Explicit evidence-quality, heuristic risk, and risk-aware ranking alongside savings-only ranking.
- Validation-selected forecast across seasonal-naive, weekday-median and Random Forest candidates with separate chronological holdout metrics; Isolation Forest plus rolling-MAD hybrid detection.
- Measured anomaly window evidence for service/region spend, runtime, resource count, usage, and available bill components.
- FastAPI, React dashboard, CSV upload, source-attributed local knowledge retrieval, and optional PostgreSQL persistence.

## Run locally

Requirements: Python 3.12+, Node.js 20+, and pnpm. If `python3.12` is unavailable on macOS, install Python 3.12 from [python.org](https://www.python.org/downloads/macos/) and use the installed `python3` command; FinOptiX does not require Homebrew.

Backend (from the project root):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
python -m app.generate_data
python -m uvicorn app.main:app --reload
```

In a second terminal:

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm dev
```

Open `http://localhost:5173`; API docs are at `http://localhost:8000/docs`. For a Python executable installed under another name, create the virtual environment with that executable and keep using `.venv/bin/python` thereafter.

### Docker

```bash
docker compose up --build
```

Compose starts PostgreSQL, the API and static React frontend (`http://localhost:5173`). The database receives the demo CSV on the API readiness check and persists normalized resources, observations, costs, computed artifacts, simulations, predictions, savings estimates, and audit events. Set `POSTGRES_PASSWORD` before deployment; the Compose default is for local demonstration only. Docker commands require Docker Desktop or Docker Engine with Compose v2.

### Optional database configuration

Without `FINOPTIX_DATABASE_URL`, the API uses the CSV demo/import path. To enable persistence locally, start PostgreSQL and set:

```bash
export FINOPTIX_DATABASE_URL='postgresql://finoptix:your-password@localhost:5432/finoptix'
export POSTGRES_PASSWORD='your-password'
```

By default, CORS permits only localhost Vite origins. For a LAN demonstration, set `FINOPTIX_CORS_ORIGINS` to the exact dashboard origin; do not use a wildcard. Copy `.env.example` for the available configuration names.

The API applies repeatable schema initialization and seeds an empty database from `data/demo_usage.csv`. In the current single-workspace demo, an upload atomically replaces the active PostgreSQL dataset and its derived artifacts so a restart cannot mix old and new uploads; audit events are retained. Tenant-scoped dataset history is future work.

## Simulator methodology

`backend/catalog/pricing_catalog.json` is the single source of catalog capacity and price assumptions. Every price is illustrative INR, not a current market price. Compute estimates use 30 billing-projection days, observed daily running hours, SKU hourly rate, region factor, and a utilization-sensitive multiplier:

`compute = 30 × observed_hours/day × hourly_rate × region_factor × (0.88 + 0.24 × mean(CPU%, memory%)/200)`

Storage uses provisioned GB-month plus the configured utilization-sensitive I/O surcharge. Network uses 30-day estimated usage, a configured free allowance, and an illustrative per-GB rate. Existing exported component costs are preferred; when absent, components are reconstructed from the catalog and scaled to observed total spend if they exceed it. The current run-rate is the mean of the latest 30 observed monthly-normalized cost rows.

For a resize, the simulator uses capacity ratios: `expected CPU = observed CPU × old vCPU/new vCPU`; memory is scaled by old/new GiB. It checks historical p95 utilization against configurable 85% CPU and 90% memory ceilings. This linear approximation does not represent latency, queueing, burst credits, or correlated bottlenecks. Shutdown savings use observed eligible runtime hours, minimum remaining runtime, and the catalog window; daily aggregates cannot prove off-hours inactivity, so schedule confidence is reduced. Storage targets use p99 observed used capacity plus configured headroom and capacity increments. Each alternative exposes assumptions and limitations.

## Risk, confidence, and ranking

All score weights and thresholds are configurable in the pricing catalog. These are heuristics, not calibrated probabilities.

**Workload risk (0–100):**

`risk = 0.50 × performance_pressure + 0.25 × variability + 0.25 × criticality`

Performance pressure is `100 × max(projected p95 CPU/85, projected p95 memory/90)`, capped at 100. Variability is `100 × max(CPU coefficient of variation, memory coefficient of variation)`, capped at 100. Criticality is an explicit resource value when present, otherwise an environment mapping. LOW ≤34.9, MEDIUM ≤64.9, otherwise HIGH. Missing/insufficient history blocks alternatives and emits a warning; it is never treated as low risk.

**Evidence confidence (0–1):** weighted sum of observation volume (20%), coverage (20%), utilization stability (15%), action consistency (15%), peak metric availability (15%), and pricing reliability (15%). The illustrative price reliability factor is 0.55. It measures evidence quality, not probability of success.

**Risk-aware priority:**

`100 × (0.45 × min(monthly savings/current cost, 1) + 0.10 × non-overlapping evidence quality − 0.45 × (risk/100)²)`

The ranking evidence term uses observation volume, coverage, peak-metric availability and price reliability. It excludes stability and action consistency because these overlap with the risk score's workload variability and performance pressure. Only feasible options with sufficient history enter the ranking. Savings-only sorts the exact same candidate action/resource set by monthly savings. No single score is a probability or financial guarantee.

## Dataset and evaluation

See [`data/README.md`](data/README.md). The default dataset has 40 resources and 365 daily observations per resource where applicable. It covers normal, underutilized, idle, over-provisioned, highly utilized, sudden spike, gradual growth, seasonal, new-resource, and occasional-peak cases. The generator seed is deterministic. Synthetic ground-truth labels are evaluation-only and excluded from the explicit forecasting and anomaly feature lists.

Generate development and separate held-out scenarios and write the evaluation report:

```bash
cd backend
python -m app.evaluation
```

Forecast candidate choice uses an inner chronological validation slice; MAE/RMSE/MAPE/R² are then reported on the later time holdout. Anomaly evaluation reports precision/recall/F1/false-positive rate for Isolation Forest, a 4× trailing rolling-MAD rule, and their union. Optimization metrics and identical-candidate-set rankings are written to `docs/evaluation-results.json`. Re-run this command to produce the report for the current code/data. Synthetic results validate mechanics only; negative R² and anomaly false positives are retained as limitations.

## API

| Method | Path | Use |
|---|---|---|
| POST | `/upload` | Validate and ingest a CSV |
| GET | `/cost-summary`, `/cost-trends` | Cost aggregates and trends |
| GET | `/forecast` | Prediction, holdout metrics and actual/predicted points |
| GET | `/anomalies`, `/anomaly-evaluation` | Detected signals and evaluation metrics |
| GET | `/recommendations`, `/savings` | Explainable candidates and savings potential |
| POST | `/what-if` | Compare feasible options for a `resource_id` |
| GET | `/optimization-ranking` | Savings-only and risk-aware ranking |
| GET | `/pricing-catalog` | Catalog, pricing disclaimer and formulas' configuration |
| POST | `/chat` | Structured analytics or local document retrieval (`mode: analytics/knowledge/auto`) |
| GET | `/knowledge/sources` | Indexed source references |
| POST | `/knowledge/reindex` | Reload local Markdown knowledge sources |

## Copilot and RAG boundary

Analytics answers call deterministic backend functions for spend, forecasts, savings, risk, constraints, and anomaly evidence. General guidance uses a local TF-IDF retrieval index over Markdown documents in `knowledge/` and returns source references. Without credentials it returns retrieved passages directly. Optional OpenAI-compatible generation can be enabled with `FINOPTIX_LLM_API_KEY`, `FINOPTIX_LLM_MODEL`, and optionally `FINOPTIX_LLM_BASE_URL`; failures fall back to the retrieved passages. Analytics values never come from an LLM. Embedding models, FAISS/Chroma, organization policy ingestion, and managed vector storage remain future work.

## Database and system architecture

The schema is in [`sql/schema.sql`](sql/schema.sql). Resources have provider-scoped natural uniqueness; observations reference resources and are unique per resource/time; daily normalized costs reference resources and are idempotent by period/source. JSONB keeps original records and evidence/assumptions. Indexes cover dimensions, resource/time, costs, recommendations, and anomalies. Schema setup is repeatable and includes legacy `raw_record` migration. Upload and demo-seed input share the same ETL and analytics functions.

```mermaid
flowchart LR
  A[CSV exports / demo generator] --> B[FastAPI ingestion + Pandas ETL]
  B --> C[(PostgreSQL resources, usage, cost, decisions)]
  B --> D[Shared cleaned DataFrame interface]
  D --> E[Analytics + forecast + anomaly/RCA]
  D --> F[Pricing catalog + recommendation rules]
  F --> G[What-if action alternatives]
  G --> H[Risk, evidence confidence, ranking]
  E --> I[FastAPI]
  H --> I
  I --> J[React dashboard]
  I --> K[Analytics copilot]
  L[Local FinOps Markdown sources] --> M[TF-IDF retrieval index]
  M --> K
```

Future cloud connectors should use read-only roles and idempotent billing-period imports. Scale by bulk loading, partitioning usage by month, caching aggregates, and moving model work into jobs. Add OIDC/RBAC, tenant isolation, managed secrets, structured logs, tracing, monitoring, and deployment approval gates before production.

## Tests and quality

```bash
cd backend
pytest -q
```

Tests cover catalog validation, dynamic cost and savings, utilization constraints, schedule limits, confidence/risk/ranking, missing history, data validation, seeded scenarios, endpoint responses, analytics grounding, and knowledge citations. PostgreSQL round-trip tests require a configured PostgreSQL service. The React app can be checked with `pnpm build` from `frontend/`.

GitHub Actions in `.github/workflows/ci.yml` runs backend tests against PostgreSQL, builds React, and builds both containers. It does not publish images or deploy to AWS; those steps require repository-specific credentials and environment configuration.

## Research framing and limitations

The method is a constrained proposal workflow: estimate alternatives from historical evidence; reject options that fail p95 or data-sufficiency constraints; score feasible options with cost benefit, evidence confidence, and risk; show assumptions for human approval. The synthetic generator imposes known patterns, but that does not prove external validity. Pricing lacks discounts, taxes, reservations, credits and changing rates; daily utilization cannot determine exact hourly schedules; no application SLO or latency telemetry is available; the forecast is a baseline; anomaly labels are synthetic; and the knowledge retriever is lexical TF-IDF. Realized savings and performance violations require controlled deployments and post-change observations.

## Further work and demo notes

For a visibly different demo, upload [`data/demo_usage_alternate.csv`](data/demo_usage_alternate.csv) from **Cost Explorer → Upload CSV**, wait for the loaded-row confirmation, then use the dashboard refresh button. The API keeps the most recently uploaded CSV as its active dataset (`data/uploaded_usage.csv`) and reloads it after a backend restart. To return to the generated default demo, stop the API, move or remove `data/uploaded_usage.csv`, and restart it. All figures remain synthetic and illustrative.

Before making operational claims, the project needs these follow-up steps:

- Validate recommendations against real billing histories and measured post-change outcomes; current evaluation uses synthetic scenarios only.
- Improve forecasting and anomaly detection. Forecast validation-based selection is implemented, but the held-out forecast still has negative R². The hybrid anomaly rule improves recall on this seeded case while raising false positives; both need validation on real billing histories.
- Replace illustrative catalog assumptions with verified provider prices and test modeled estimates against actual bills.
- Configure a live PostgreSQL instance and run persistence/restart checks; CSV persistence is the current local default.
- Add read-only cloud billing connectors, authentication, access controls, tenant isolation, and production monitoring before organizational deployment.

## Two-minute placement explanation

“FinOptiX analyzes cloud cost and usage data, then compares potential optimizations before implementation. Its key idea is to make savings, workload constraints, risk, confidence, and assumptions visible together, so teams can review safer alternatives instead of ranking actions by savings alone. The prototype uses synthetic data, so the next step is validation against real cloud histories and measured outcomes.”
