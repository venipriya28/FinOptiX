"""FinOptiX demo domain logic: deterministic analytics, transparent rules, and simulation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_recall_fscore_support
from .simulator import simulate_options, SimulationError

REQUIRED = ["timestamp", "cloud_provider", "service", "region", "resource_id", "resource_type",
            "instance_type", "cpu_utilization", "memory_utilization", "network_usage", "storage_usage",
            "running_hours", "monthly_cost", "department", "project", "environment"]
GROUND_TRUTH = ["scenario_type", "is_anomaly", "optimization_opportunity", "optimization_type", "expected_risk", "expected_saving"]
OPTIONAL_DATA = ["allocated_storage_gb", "compute_monthly_cost", "storage_monthly_cost", "network_monthly_cost", "resource_criticality"]
EVALUATION_LABELS = ["is_anomaly", "optimization_opportunity", "optimization_type", "expected_risk", "expected_saving"]
FORECAST_FEATURES = ["lag1", "lag7", "lag14", "lag28", "dow", "trend", "mean7"]
ANOMALY_FEATURES = ["cost_change_pct", "cpu_change_pct", "memory_change_pct", "network_change_pct", "runtime_change_pct", "storage_usage"]

def clean_usage(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate, deduplicate and derive daily resource-level features."""
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("No usage observations were provided")
    # Preserve evaluation labels for metrics/reporting, but model feature lists below are explicit
    # and never include labels or scenario_type.
    keep = REQUIRED + [c for c in OPTIONAL_DATA + GROUND_TRUTH if c in frame.columns]
    df = frame[keep].copy().drop_duplicates()
    categorical_required=["cloud_provider","service","region","resource_id","resource_type","instance_type","department","project","environment"]
    for col in categorical_required:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Required dimension {col} contains missing or blank values")
    df["cpu_utilization_missing"] = df["cpu_utilization"].isna()
    df["memory_utilization_missing"] = df["memory_utilization"].isna()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in ["cpu_utilization", "memory_utilization", "network_usage", "storage_usage", "running_hours", "monthly_cost"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    for col in OPTIONAL_DATA:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") if col != "resource_criticality" else df[col]
    df = df.dropna(subset=["timestamp", "resource_id"])
    if df.empty:
        raise ValueError("No valid timestamped usage observations remain after validation")
    if ((df[["cpu_utilization", "memory_utilization"]] < 0) | (df[["cpu_utilization", "memory_utilization"]] > 100)).any().any():
        raise ValueError("CPU and memory utilization must be between 0 and 100")
    if ((df["running_hours"] < 0) | (df["running_hours"] > 24)).any():
        raise ValueError("Running hours must be between 0 and 24")
    if (df[["network_usage", "storage_usage", "monthly_cost"]] < 0).any().any():
        raise ValueError("Network usage, storage usage, and cost cannot be negative")
    if (df["storage_usage"] > 100).any():
        raise ValueError("Storage utilization must be between 0 and 100")
    if "allocated_storage_gb" in df and (df.allocated_storage_gb.dropna() <= 0).any():
        raise ValueError("Allocated storage capacity must be positive when provided")
    if "resource_criticality" in df:
        df["resource_criticality"] = pd.to_numeric(df["resource_criticality"], errors="coerce")
        if ((df.resource_criticality.dropna() < 0) | (df.resource_criticality.dropna() > 1)).any():
            raise ValueError("Resource criticality must be between 0 and 1")
    if df.duplicated(["cloud_provider", "resource_id", "timestamp"]).any():
        raise ValueError("Conflicting duplicate observations for the same resource and timestamp")
    df = df.sort_values(["resource_id", "timestamp"]).reset_index(drop=True)
    df["daily_cost"] = df["monthly_cost"] / 30.0
    cost_by_resource = df.groupby(["cloud_provider", "resource_id"], sort=False)["daily_cost"]
    df["7_day_avg_cost"] = cost_by_resource.transform(lambda series: series.rolling(7, min_periods=1).mean())
    df["weekly_cost"] = cost_by_resource.transform(lambda series: series.rolling(7, min_periods=1).sum())
    df["30_day_avg_cost"] = cost_by_resource.transform(lambda series: series.rolling(30, min_periods=1).mean())
    df["monthly_cost_30d"] = cost_by_resource.transform(lambda series: series.rolling(30, min_periods=1).sum())
    prior_month = cost_by_resource.transform(lambda series: series.shift(30).rolling(30, min_periods=7).mean())
    df["cost_growth_rate"] = ((df["30_day_avg_cost"] - prior_month) / prior_month.abs().clip(lower=1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["utilization_score"] = (df["cpu_utilization"] + df["memory_utilization"]) / 2
    df["idle_hours"] = (24 - df["running_hours"]) + (24 * (1 - df["utilization_score"] / 100))
    df["cost_per_usage"] = df["monthly_cost"] / (df["running_hours"].clip(lower=1) + df["network_usage"].clip(lower=0) + 1)
    df["resource_efficiency_score"] = (df["utilization_score"] / df["monthly_cost"].clip(lower=1) * 1000).clip(0, 100)
    df["day"] = df["timestamp"].dt.date.astype(str)
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.strftime("%Y-%m")
    return df

def cost_summary(df: pd.DataFrame) -> dict[str, Any]:
    latest = df.sort_values("timestamp").drop_duplicates(["cloud_provider", "resource_id"], keep="last")
    total = float(latest.monthly_cost.sum())
    group = lambda col: latest.groupby(col, dropna=False).monthly_cost.sum().sort_values(ascending=False).round(2).to_dict()
    return {"total_monthly_cost": round(total, 2), "currency": "INR", "resource_count": int(latest[["cloud_provider", "resource_id"]].drop_duplicates().shape[0]),
            "by_provider": group("cloud_provider"), "by_service": group("service"), "by_region": group("region"),
            "by_department": group("department"), "by_project": group("project"),
            "average_efficiency": round(float(latest.resource_efficiency_score.mean()), 2),
            "top_resources": latest.sort_values("monthly_cost", ascending=False).head(8)[["cloud_provider", "resource_id", "service", "region", "monthly_cost"]].to_dict("records")}

def recommendations(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return each resource's best feasible catalog-backed action by the risk-aware score."""
    candidates = optimization_candidates(df)
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidates:
        if row["risk_aware_score"] <= 0: continue
        key=(row["cloud_provider"],row["resource_id"])
        current = best.get(key)
        if current is None or row["risk_aware_score"] > current["risk_aware_score"]:
            best[key] = row
    return sorted(best.values(), key=lambda x: x["risk_aware_score"], reverse=True)

def optimization_candidates(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Materialize the common feasible action/resource pool used by both rankers."""
    out = []
    resources=df[["cloud_provider","resource_id"]].dropna().drop_duplicates()
    for cloud_provider, resource_id in resources.itertuples(index=False,name=None):
        resource_id=str(resource_id); cloud_provider=str(cloud_provider)
        resource_rows=df[(df.cloud_provider.astype(str)==cloud_provider)&(df.resource_id.astype(str)==resource_id)]
        try:
            result = simulate_options(df, resource_id, cloud_provider=cloud_provider)
        except (SimulationError, ValueError, KeyError):
            continue
        for option in result["alternatives"]:
            action = option["action"]
            target = option.get("target_instance")
            display = {"instance_rightsizing": f"Downsize to {target}", "scheduled_shutdown": "Schedule non-business shutdown",
                       "storage_optimization": f"Right-size storage to {target}"}.get(action, action)
            reason = {"instance_rightsizing": "Catalog transition is supported and historical p95 CPU/memory fit configured limits.",
                      "scheduled_shutdown": "Environment is schedule-eligible and observed runtime exceeds the configured daily minimum.",
                      "storage_optimization": "p99 observed storage plus configured headroom fits a smaller capacity tier."}.get(action, "Feasible catalog-backed scenario.")
            candidate = {**option, "resource_id": resource_id, "action_label": display, "action": display, "action_code": action,
                         "service": str(resource_rows.sort_values("timestamp").iloc[-1].service),
                         "cloud_provider": cloud_provider,
                         "reason": reason,
                         "estimated_monthly_saving": option["monthly_saving"],
                         "estimated_annual_saving": option["annualized_saving"],
                         "confidence": option["evidence_confidence"], "priority_score": option["risk_aware_score"],
                         "evidence": {"cpu_pct": option["evidence"]["cpu_mean_pct"],
                                      "cpu_p95_pct": option["evidence"]["cpu_p95_pct"],
                                      "memory_pct": option["evidence"]["memory_mean_pct"],
                                      "memory_p95_pct": option["evidence"]["memory_p95_pct"],
                                      "monthly_cost": option["current_monthly_cost"],
                                      "observations": option["evidence"]["observations"],
                                      "coverage_90d": option["evidence"]["coverage_90d"]}}
            out.append(candidate)
    return out

def ranking_comparison(df: pd.DataFrame) -> dict[str, Any]:
    """Compare both rankers on the exact same candidate action/resource pairs."""
    pool = optimization_candidates(df)
    savings = sorted(pool, key=lambda x: (x["estimated_monthly_saving"], x["cloud_provider"], x["resource_id"], x["option_id"]), reverse=True)
    risk = sorted(pool, key=lambda x: (x["risk_aware_score"], x["cloud_provider"], x["resource_id"], x["option_id"]), reverse=True)
    savings_rank = {x["option_id"]: i + 1 for i, x in enumerate(savings)}
    risk_rank = {x["option_id"]: i + 1 for i, x in enumerate(risk)}
    def trim(rows):
        return [{**{k: v for k, v in row.items() if k not in {"assumptions", "limitations", "workload_risk", "confidence_factors"}},
                 "savings_only_rank": savings_rank[row["option_id"]], "risk_aware_rank": risk_rank[row["option_id"]]} for row in rows]
    return {"candidate_count": len(pool), "same_candidate_set": True,
            "savings_only": trim(savings), "risk_aware": trim(risk),
            "formula": "Risk-aware = 100 × (0.45 × savings/current-cost + 0.10 × non-overlapping evidence quality − 0.45 × (risk-score/100)^2). Evidence quality uses observation volume, coverage, peak-metric availability and price reliability; it excludes stability/action-fit already represented in risk.",
            "note": "Savings-only sorts by gross monthly saving; risk-aware ranks the identical feasible action/resource pairs."}

def simulate(df: pd.DataFrame, resource_id: str, action: str | None = None, target_instance: str | None = None,
             cloud_provider: str | None = None) -> dict[str, Any]:
    """Compatibility entry point; returns every feasible and rejected option for the resource."""
    result = simulate_options(df, resource_id, target_instance, cloud_provider=cloud_provider)
    if action not in (None, "compare", "all"):
        result["requested_action"] = action
        result["requested_action_matches"] = [x for x in result["alternatives"] if x["action"] == action or x["option_id"] == action]
        if not result["requested_action_matches"]:
            result["requested_action_warning"] = "Requested action is not a feasible catalog option. Review rejected_options and valid alternatives."
    return result

def _anomaly_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["resource_id", "timestamp"]).copy()
    resource_keys=["cloud_provider","resource_id"]
    prior = df.groupby(resource_keys)["monthly_cost"].transform(lambda s: s.shift(1).rolling(30, min_periods=7).median())
    df["_prior_cost"] = prior
    df["cost_change_pct"] = ((df.monthly_cost - prior) / prior.clip(lower=1) * 100).replace([np.inf, -np.inf], 0).fillna(0)
    for source, target in [("cpu_utilization", "cpu_change_pct"), ("memory_utilization", "memory_change_pct"),
                           ("network_usage", "network_change_pct"), ("running_hours", "runtime_change_pct")]:
        historical = df.groupby(resource_keys)[source].transform(lambda s: s.shift(1).rolling(30, min_periods=7).median())
        df[target] = ((df[source] - historical) / historical.clip(lower=1) * 100).replace([np.inf, -np.inf], 0).fillna(0).clip(-500, 1000)
    features = df[ANOMALY_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, features

def _anomaly_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df, features = _anomaly_feature_frame(df)
    if len(df) < 5: return df, np.zeros(len(df), dtype=bool)
    # Keep the sparse multivariate detector and add a causal robust cost-change guard.
    # No ground-truth columns are read here; both components use current measurements
    # and prior observations only. This can raise recall at the cost of extra alerts.
    isolation = IsolationForest(contamination=0.002, random_state=42, n_estimators=160).fit_predict(features) == -1
    resource_keys = ["cloud_provider", "resource_id"]
    rolling_mad = df.groupby(resource_keys)["monthly_cost"].transform(
        lambda values: values.shift(1).rolling(30, min_periods=7).apply(
            lambda window: float(np.median(np.abs(window - np.median(window))))))
    prior = df["_prior_cost"]
    robust_cost_change = (prior.notna() & (df.monthly_cost - prior).abs().gt(4.0 * rolling_mad.clip(lower=1))).to_numpy()
    return df, isolation | robust_cost_change

def anomaly_evaluation(df: pd.DataFrame) -> dict[str, Any] | None:
    """Evaluate on a chronological test period; evaluation labels never enter features."""
    if "is_anomaly" not in df.columns: return None
    frame, features = _anomaly_feature_frame(df)
    dates = np.sort(frame.timestamp.unique())
    if len(dates) < 8: return None
    cutoff = dates[max(1, int(len(dates) * .8)) - 1]
    train_mask = frame.timestamp <= cutoff
    test_mask = frame.timestamp > cutoff
    if int(train_mask.sum()) < 5 or not bool(test_mask.any()): return None
    model = IsolationForest(contamination=0.002, random_state=42, n_estimators=160)
    model.fit(features.loc[train_mask])
    predicted = model.predict(features.loc[test_mask]) == -1
    held = frame.loc[test_mask]
    truth = held["is_anomaly"].astype(bool).to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
    prior = held["_prior_cost"]
    residual = (held.monthly_cost - prior).abs()
    # The detector's feature frame already contains history-based cost deltas; calculate the baseline from full precomputed history.
    rolling_mad_all = frame.groupby(["cloud_provider","resource_id"]).monthly_cost.transform(lambda s: s.shift(1).rolling(30, min_periods=7).apply(lambda x: float(np.median(np.abs(x - np.median(x))))))
    rolling_mad = rolling_mad_all.loc[test_mask]
    baseline_predicted = ((prior.notna()) & (residual > (4.0 * rolling_mad.clip(lower=1)))).to_numpy()
    hybrid_predicted = predicted | baseline_predicted
    b_precision, b_recall, b_f1, _ = precision_recall_fscore_support(truth, baseline_predicted, average="binary", zero_division=0)
    h_precision, h_recall, h_f1, _ = precision_recall_fscore_support(truth, hybrid_predicted, average="binary", zero_division=0)
    false_positive_rate = float(((predicted) & (~truth)).sum() / max((~truth).sum(), 1))
    baseline_fpr = float(((baseline_predicted) & (~truth)).sum() / max((~truth).sum(), 1))
    return {"detector": "Isolation Forest OR 4× rolling-MAD cost rule",
            "precision": round(float(h_precision), 3), "recall": round(float(h_recall), 3), "f1": round(float(h_f1), 3),
            "false_positive_rate": round(float((hybrid_predicted & ~truth).sum() / max((~truth).sum(), 1)), 3),
            "isolation_forest": {"precision": round(float(precision), 3), "recall": round(float(recall), 3), "f1": round(float(f1), 3),
                "false_positive_rate": round(false_positive_rate, 3), "predicted_rows": int(predicted.sum())},
            "rolling_mad_baseline": {"precision": round(float(b_precision), 3), "recall": round(float(b_recall), 3), "f1": round(float(b_f1), 3), "false_positive_rate": round(baseline_fpr, 3)},
            "hybrid_iforest_or_rolling_mad": {"precision": round(float(h_precision), 3), "recall": round(float(h_recall), 3), "f1": round(float(h_f1), 3),
                "false_positive_rate": round(float((hybrid_predicted & ~truth).sum() / max((~truth).sum(), 1)), 3),
                "predicted_rows": int(hybrid_predicted.sum()), "rule": "Isolation Forest OR a 4× trailing rolling-MAD cost deviation; both use prior observations only."},
            "ground_truth_positive_rows": int(truth.sum()), "predicted_outlier_rows": int(hybrid_predicted.sum()),
            "holdout_rows": len(held), "holdout_start": held.timestamp.min().isoformat(),
            "label_source": "synthetic is_anomaly evaluation label; excluded from model inputs", "evaluation_limit": "Controlled synthetic labels do not establish production performance."}

def _root_cause_evidence(df: pd.DataFrame, event_time: pd.Timestamp) -> dict[str, Any]:
    """Compare measured aggregate spend and workload in recent and preceding windows."""
    times = pd.to_datetime(df.timestamp, utc=True)
    recent_mask = (times > event_time - pd.Timedelta(days=7)) & (times <= event_time)
    baseline_mask = (times > event_time - pd.Timedelta(days=37)) & (times <= event_time - pd.Timedelta(days=7))
    recent, baseline = df.loc[recent_mask].copy(), df.loc[baseline_mask].copy()
    if recent.empty or baseline.empty:
        return {"status": "insufficient_comparison_window", "recent_days": int(recent.day.nunique()) if "day" in recent else 0,
                "baseline_days": int(baseline.day.nunique()) if "day" in baseline else 0,
                "explanation": "A complete recent seven-day and preceding thirty-day comparison window is unavailable."}
    key_cols = ["cloud_provider", "service", "region"]
    def by_service(frame):
        frame=frame.assign(_resource_key=frame.cloud_provider.astype(str)+":"+frame.resource_id.astype(str))
        grouped=frame.groupby(key_cols, dropna=False).agg(spend=("monthly_cost", "sum"), runtime=("running_hours", "mean"),
            cpu=("cpu_utilization", "mean"), memory=("memory_utilization", "mean"), network=("network_usage", "mean"),
            storage=("storage_usage", "mean"), resources=("_resource_key", "nunique")).reset_index()
        grouped["spend"] /= max(int(frame.day.nunique()),1)
        return grouped
    before, after = by_service(baseline), by_service(recent)
    joined = before.merge(after, on=key_cols, how="outer", suffixes=("_before", "_after")).fillna(0)
    joined["spend_delta"] = joined.spend_after - joined.spend_before
    drivers = []
    for _, part in joined.sort_values("spend_delta", ascending=False).head(5).iterrows():
        change = float(part.spend_delta)
        drivers.append({"cloud_provider": str(part.cloud_provider), "service": str(part.service), "region": str(part.region), "spend_delta": round(change, 2),
            "spend_before": round(float(part.spend_before), 2), "spend_after": round(float(part.spend_after), 2),
            "runtime_hours_before": round(float(part.runtime_before), 2), "runtime_hours_after": round(float(part.runtime_after), 2),
            "resource_count_before": int(part.resources_before), "resource_count_after": int(part.resources_after),
            "cpu_mean_before": round(float(part.cpu_before), 2), "cpu_mean_after": round(float(part.cpu_after), 2),
            "memory_mean_before": round(float(part.memory_before), 2), "memory_mean_after": round(float(part.memory_after), 2),
            "network_mean_before": round(float(part.network_before), 2), "network_mean_after": round(float(part.network_after), 2),
            "storage_mean_before": round(float(part.storage_before), 2), "storage_mean_after": round(float(part.storage_after), 2)})
    total_before = float(baseline.groupby("day").monthly_cost.sum().mean())
    total_after = float(recent.groupby("day").monthly_cost.sum().mean())
    component_changes = {}
    for col in ("compute_monthly_cost", "storage_monthly_cost", "network_monthly_cost"):
        if col in df:
            component_changes[col] = round(float(recent.groupby("day")[col].sum().mean() - baseline.groupby("day")[col].sum().mean()), 2)
    resource_keys_before=set(zip(baseline.cloud_provider.astype(str),baseline.resource_id.astype(str)))
    resource_keys_recent=set(zip(recent.cloud_provider.astype(str),recent.resource_id.astype(str)))
    resource_count_before = len(resource_keys_before); resource_count_after = len(resource_keys_recent)
    resources_added = sorted(f"{provider}/{resource}" for provider,resource in resource_keys_recent-resource_keys_before)
    resources_removed = sorted(f"{provider}/{resource}" for provider,resource in resource_keys_before-resource_keys_recent)
    explanation = (f"Measured portfolio monthly run-rate changed by ₹{total_after-total_before:,.0f} between the preceding 30 days and recent 7 days "
        f"(average daily portfolio run-rate estimates ₹{total_before:,.0f} to ₹{total_after:,.0f}); resource counts were {resource_count_before} and {resource_count_after}. "
        "Largest measured service/region contributions are listed below. This window association does not prove causation.")
    return {"status": "measured_comparison", "recent_window": [str((event_time-pd.Timedelta(days=7)).date()), str(event_time.date())],
        "baseline_window": [str((event_time-pd.Timedelta(days=37)).date()), str((event_time-pd.Timedelta(days=7)).date())],
        "average_portfolio_monthly_run_rate_before": round(total_before, 2), "average_portfolio_monthly_run_rate_recent": round(total_after, 2),
        "spend_delta": round(total_after-total_before, 2), "resource_count_before": resource_count_before,
        "resource_count_recent": resource_count_after, "resources_not_observed_in_baseline": resources_added,
        "resource_presence_note": "These IDs appear in the recent window but not in the baseline window; this dataset does not establish their creation dates.",
        "resources_not_seen_in_recent_window": resources_removed, "baseline_observed_days": int(baseline.day.nunique()),
        "recent_observed_days": int(recent.day.nunique()), "component_spend_delta": component_changes,
        "service_region_contributions": drivers, "explanation": explanation,
        "unit_price_change": "not measured: input dataset contains no time-varying unit-price observations"}

def detect_anomalies(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect daily outliers; ground-truth columns are used only after prediction for evaluation."""
    df, predicted = _anomaly_predictions(df)
    if "is_anomaly" in df.columns:
        eval_metrics = anomaly_evaluation(df)
    else: eval_metrics = None
    df["_predicted_anomaly"] = predicted
    med = df.monthly_cost.median(); mad = max((df.monthly_cost-med).abs().median(), 1)
    output=[]
    for idx, row in df[df._predicted_anomaly].iterrows():
        baseline = float(row._prior_cost) if pd.notna(row._prior_cost) else float(med)
        impact = max(0, float(row.monthly_cost - baseline))
        cause_analysis = _root_cause_evidence(df, pd.Timestamp(row.timestamp))
        output.append({"resource_id": row.resource_id, "cloud_provider": row.cloud_provider, "service": row.service, "region": row.region,
                       "timestamp": row.timestamp.isoformat(), "monthly_cost": float(row.monthly_cost), "cost_impact_vs_baseline": round(impact,2), "cost_impact_vs_median": round(impact,2), "severity": "HIGH" if impact > 3*mad else "MEDIUM",
                       "cause": f"{row.cloud_provider} {row.service} in {row.region} changed {row.cost_change_pct:+.1f}% versus its prior 30-day median (₹{baseline:,.0f} to ₹{row.monthly_cost:,.0f}); CPU {row.cpu_utilization:.1f}%, runtime {row.running_hours:.1f} h/day, network {row.network_usage:.1f} GB/day.",
                       "root_cause_analysis": cause_analysis,
                       "evidence": {"prior_30_day_median": round(baseline,2), "cost_change_pct": round(float(row.cost_change_pct),1), "cpu_pct": float(row.cpu_utilization), "running_hours": float(row.running_hours), "network_gb_day": float(row.network_usage)}})
    output.sort(key=lambda item: item["timestamp"], reverse=True)
    # Keep evaluation visible without injecting labels into detector inputs.
    if output: output[0]["evaluation_metrics"] = eval_metrics
    return output

def _forecast_features(daily: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"lag1": daily.shift(1), "lag7": daily.shift(7), "lag14": daily.shift(14),
        "lag28": daily.shift(28), "dow": pd.to_datetime(daily.index).dayofweek,
        "trend": np.arange(len(daily)), "mean7": daily.shift(1).rolling(7, min_periods=7).mean()}, index=daily.index)


def _forecast_metrics(truth: pd.Series, predicted: np.ndarray) -> dict[str, Any]:
    denom = np.maximum(np.abs(truth.to_numpy(dtype=float)), 1.0)
    return {"mae": round(float(mean_absolute_error(truth, predicted)), 2),
        "rmse": round(float(mean_squared_error(truth, predicted) ** .5), 2),
        "mape_pct": round(float(np.mean(np.abs((truth.to_numpy(dtype=float) - predicted) / denom)) * 100), 2),
        "r2": round(float(r2_score(truth, predicted)), 3) if len(truth) > 1 else None}


def _weekday_median(history: list[float], weekday: int, last_weekday: int) -> float:
    # History is ordered and contains only observations before the prediction date.
    start = max(0, len(history) - 35)
    values = [value for index, value in enumerate(history[start:], start=start)
        if (last_weekday - (len(history) - 1 - index)) % 7 == weekday]
    return float(np.median(values[-4:])) if values else float(np.median(history[-7:]))


def forecast(df: pd.DataFrame) -> dict[str, Any]:
    """One-day rolling-origin backtest; choose strategy on a validation tail inside train."""
    daily = df.groupby("day").daily_cost.sum().sort_index().astype(float)
    if len(daily) < 35:
        base = float(daily.tail(min(7, len(daily))).median()) if len(daily) else 0.0
        return {"model": "short-history median baseline", "next_day": round(base, 2), "next_week": round(base * 7, 2),
            "next_month": round(base * 30, 2), "metrics": None, "baseline_metrics": None, "model_comparison": [],
            "warning": "At least 35 daily observations are required for the lag-28 validation and forecast.", "series": []}
    X = _forecast_features(daily).dropna()[FORECAST_FEATURES]
    y = daily.loc[X.index]
    test_start = max(1, int(len(X) * .8))
    # Select using only a trailing validation slice of the pre-test training period.
    train_end = max(2, int(test_start * .8))
    validation_truth = y.iloc[train_end:test_start]
    validation_preds: dict[str, list[float]] = {"seasonal_naive_7d": [], "weekday_median_4w": [], "random_forest_lags": []}
    validation_history = list(daily.loc[:y.index[train_end - 1]].to_numpy(dtype=float))
    for i in range(train_end, test_start):
        date_i = pd.Timestamp(y.index[i])
        validation_preds["seasonal_naive_7d"].append(float(X.iloc[i]["lag7"]))
        validation_preds["weekday_median_4w"].append(_weekday_median(validation_history, date_i.dayofweek, (date_i - pd.Timedelta(days=1)).dayofweek))
        validation_history.append(float(y.iloc[i]))
    if test_start - train_end >= 2:
        validation_model = RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=3)
        validation_model.fit(X.iloc[:train_end], y.iloc[:train_end])
        validation_preds["random_forest_lags"] = validation_model.predict(X.iloc[train_end:test_start]).tolist()
    else:
        validation_preds["random_forest_lags"] = validation_preds["seasonal_naive_7d"]
    validation_mae = {name: float(mean_absolute_error(validation_truth, preds)) for name, preds in validation_preds.items() if len(preds) == len(validation_truth) and len(preds)}
    selected = min(validation_mae, key=validation_mae.get) if validation_mae else "seasonal_naive_7d"

    truth = y.iloc[test_start:]
    model = RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=3).fit(X.iloc[:test_start], y.iloc[:test_start])
    rf_test = model.predict(X.iloc[test_start:])
    seasonal_test = X.iloc[test_start:]["lag7"].to_numpy(dtype=float)
    test_history = list(daily.loc[:y.index[test_start - 1]].to_numpy(dtype=float))
    median_test = []
    for i in range(test_start, len(y)):
        median_test.append(_weekday_median(test_history, pd.Timestamp(y.index[i]).dayofweek, pd.Timestamp(y.index[i - 1]).dayofweek))
        test_history.append(float(y.iloc[i]))
    comparison = {"seasonal_naive_7d": seasonal_test, "weekday_median_4w": np.asarray(median_test), "random_forest_lags": rf_test}
    comparison_metrics = [{"model": name, **_forecast_metrics(truth, values), "validation_mae": round(validation_mae.get(name, float("nan")), 2)}
        for name, values in comparison.items()]
    metrics = _forecast_metrics(truth, comparison[selected])
    baseline_metrics = {**_forecast_metrics(truth, seasonal_test), "model": "seasonal naive (previous week cost)"}

    history = list(daily.to_numpy(dtype=float)); future: list[float] = []
    full_features = _forecast_features(daily).dropna()[FORECAST_FEATURES]
    full_rf = RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=3).fit(full_features, daily.loc[full_features.index])
    for step in range(30):
        future_date = pd.Timestamp(daily.index[-1]) + pd.Timedelta(days=step + 1)
        if selected == "seasonal_naive_7d":
            value = history[-7]
        elif selected == "weekday_median_4w":
            value = _weekday_median(history, future_date.dayofweek, pd.Timestamp(daily.index[-1]).dayofweek + step if step == 0 else (future_date - pd.Timedelta(days=1)).dayofweek)
        else:
            feature = pd.DataFrame([{"lag1": history[-1], "lag7": history[-7], "lag14": history[-14], "lag28": history[-28],
                "dow": future_date.dayofweek, "trend": len(history), "mean7": float(np.mean(history[-7:]))}])[FORECAST_FEATURES]
            value = float(full_rf.predict(feature)[0])
        value = max(0.0, float(value)); future.append(value); history.append(value)
    test_predictions = comparison[selected]
    series = []
    for i, (d, value) in enumerate(zip(y.index, y.to_numpy(dtype=float))):
        prediction = test_predictions[i - test_start] if i >= test_start else None
        series.append({"date": str(d), "actual": round(float(value), 2), "predicted": round(float(prediction), 2) if prediction is not None else None})
    model_names = {"seasonal_naive_7d": "Seasonal naive (previous week)", "weekday_median_4w": "Four-week weekday median", "random_forest_lags": "Random Forest lag-feature model"}
    return {"model": model_names[selected], "next_day": round(future[0], 2), "next_week": round(sum(future[:7]), 2),
        "next_month": round(sum(future), 2), "metrics": metrics, "baseline_metrics": baseline_metrics,
        "model_comparison": comparison_metrics, "selection": {"method": "lowest MAE on a chronological validation tail within the training window", "selected": model_names[selected],
            "validation_mae": round(validation_mae[selected], 2), "test_window_is_not_used_for_selection": True}, "series": series,
        "evaluation_limit": "Synthetic or uploaded historical data backtests do not guarantee future cloud expenditure."}
