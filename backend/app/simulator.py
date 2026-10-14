"""Catalog-priced what-if alternatives using observed workload history."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .catalog import CatalogError, load_catalog, resolve_resource, validate_catalog

class SimulationError(ValueError):
    """The resource or requested optimization cannot be evaluated safely."""

def _quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.quantile(values, q)) if len(values) else float("nan")

def _usage_multiplier(cpu: float, memory: float, catalog: dict[str, Any]) -> float:
    model = catalog["billing"]["utilization_cost_multiplier"]
    return float(model["base"]) + float(model["per_utilization_fraction"]) * max(0.0, min(1.0, (cpu + memory) / 200.0))

def _component_series(history: pd.DataFrame, cfg: dict[str, Any], region_factor: float,
                      sku: dict[str, Any] | None, catalog: dict[str, Any]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Prefer generator/export cost components; reconstruct only when omitted from an import."""
    if "compute_monthly_cost" in history and history.compute_monthly_cost.notna().all():
        compute = pd.to_numeric(history.compute_monthly_cost, errors="coerce")
    elif sku is not None:
        billing = catalog["billing"]
        compute = (pd.to_numeric(history.running_hours).clip(0, 24) * float(billing["monthly_projection_days"])
                   * float(sku["hourly_inr"]) * region_factor
                   * pd.Series([_usage_multiplier(c, m, catalog) for c, m in zip(history.cpu_utilization, history.memory_utilization)], index=history.index))
    else:
        compute = pd.Series(0.0, index=history.index)

    provider = cfg
    if "storage_monthly_cost" in history and history.storage_monthly_cost.notna().all():
        storage = pd.to_numeric(history.storage_monthly_cost, errors="coerce")
    else:
        allocated = pd.to_numeric(history.get("allocated_storage_gb", pd.Series(np.nan, index=history.index)), errors="coerce")
        fill = pd.to_numeric(history.storage_usage, errors="coerce").clip(0, 100) / 100
        storage = allocated * float(provider["storage_inr_per_gb_month"]) * (1 + float(catalog["billing"]["storage_io_surcharge_per_usage_fraction"]) * fill)
    if "network_monthly_cost" in history and history.network_monthly_cost.notna().all():
        network = pd.to_numeric(history.network_monthly_cost, errors="coerce")
    else:
        network_gb = pd.to_numeric(history.network_usage, errors="coerce").clip(lower=0)
        monthly_gb = network_gb * float(catalog["billing"]["monthly_projection_days"])
        free = float(provider.get("network_free_gb_per_month", catalog["billing"]["network_free_gb_per_month"]))
        network = (monthly_gb - free).clip(lower=0) * float(provider["network_inr_per_gb"])
    return compute.astype(float).fillna(0), storage.astype(float).fillna(0), network.astype(float).fillna(0)

def _workload_evidence(history: pd.DataFrame, catalog: dict[str, Any]) -> dict[str, Any]:
    n = len(history)
    latest = pd.to_datetime(history.timestamp, utc=True).max()
    start_window = latest.normalize() - pd.Timedelta(days=89)
    observed_days = pd.to_datetime(history.timestamp, utc=True).dt.normalize()
    in_window = observed_days[observed_days >= start_window]
    expected_days = min(90, max(1, int((latest.normalize() - observed_days.min()).days) + 1))
    coverage = min(1.0, in_window.nunique() / expected_days)
    cpu = pd.to_numeric(history.cpu_utilization, errors="coerce")
    memory = pd.to_numeric(history.memory_utilization, errors="coerce")
    cpu_mean, memory_mean = float(cpu.mean()), float(memory.mean())
    cpu_p95, memory_p95 = _quantile(cpu, .95), _quantile(memory, .95)
    cpu_cv = min(2.0, float(cpu.std(ddof=0) / max(abs(cpu_mean), 5.0)))
    memory_cv = min(2.0, float(memory.std(ddof=0) / max(abs(memory_mean), 5.0)))
    missing_cpu = int(history.get("cpu_utilization_missing", pd.Series(False, index=history.index)).fillna(False).sum())
    missing_memory = int(history.get("memory_utilization_missing", pd.Series(False, index=history.index)).fillna(False).sum())
    # Assess each signal independently so one mostly-missing metric cannot be diluted
    # by the other complete metric.
    missing_share = max(missing_cpu / max(n, 1), missing_memory / max(n, 1))
    minimum = int(catalog["confidence"]["minimum_observations"])
    min_coverage = float(catalog["confidence"]["minimum_coverage"])
    sufficient = n >= minimum and coverage >= min_coverage and missing_share <= .2 and math.isfinite(cpu_p95) and math.isfinite(memory_p95)
    return {"observations": n, "coverage_90d": round(coverage, 3), "expected_days": expected_days,
            "cpu_mean_pct": round(cpu_mean, 2), "cpu_p95_pct": round(cpu_p95, 2) if math.isfinite(cpu_p95) else None,
            "cpu_max_pct": round(float(cpu.max()), 2), "memory_mean_pct": round(memory_mean, 2),
            "memory_p95_pct": round(memory_p95, 2) if math.isfinite(memory_p95) else None,
            "memory_max_pct": round(float(memory.max()), 2), "cpu_cv": round(cpu_cv, 3), "memory_cv": round(memory_cv, 3),
            "missing_utilization_share": round(missing_share, 3), "sufficient": sufficient,
            "warning": None if sufficient else f"Insufficient evidence: need at least {minimum} observations, {min_coverage:.0%} daily coverage, and usable CPU/memory peak data."}

def _confidence(evidence: dict[str, Any], consistency: float, catalog: dict[str, Any], *, peak_metrics: bool = True) -> tuple[float, dict[str, float]]:
    cfg = catalog["confidence"]
    factors = {
        "observation_volume": min(1.0, evidence["observations"] / float(cfg["target_observations"])),
        "coverage": evidence["coverage_90d"],
        "stability": max(0.0, 1.0 - min(1.0, max(evidence["cpu_cv"], evidence["memory_cv"]))),
        "recommendation_consistency": max(0.0, min(1.0, consistency)),
        "peak_metrics": 1.0 if peak_metrics and evidence["observations"] >= cfg["minimum_observations"] else 0.0,
        "price_reliability": float(cfg["illustrative_price_reliability"]),
    }
    weights = cfg["weights"]
    score = sum(float(weights[key]) * factors[key] for key in weights)
    return round(max(0.0, min(1.0, score)), 3), {key: round(value, 3) for key, value in factors.items()}

def _ranking_evidence_quality(factors: dict[str, float], catalog: dict[str, Any]) -> float:
    """Use data volume/coverage/peak availability/pricing only; omit risk-overlapping stability and fit."""
    included = ("observation_volume", "coverage", "peak_metrics", "price_reliability")
    weights = catalog["confidence"]["weights"]
    total = sum(float(weights[key]) for key in included)
    return round(sum(float(weights[key]) * float(factors[key]) for key in included) / total, 3)

def _risk(post_cpu_p95: float, post_memory_p95: float, evidence: dict[str, Any], environment: str,
          catalog: dict[str, Any], resource_criticality: float | None = None) -> tuple[float, str, list[str]]:
    cfg = catalog["risk"]
    cpu_limit, mem_limit = float(cfg["max_p95_cpu_pct"]), float(cfg["max_p95_memory_pct"])
    perf = min(100.0, max(post_cpu_p95 / cpu_limit, post_memory_p95 / mem_limit) * 100)
    volatility = min(100.0, max(evidence["cpu_cv"], evidence["memory_cv"]) * 100)
    criticality_fraction = float(resource_criticality) if resource_criticality is not None else float(cfg["criticality_by_environment"].get(environment.lower(), cfg["criticality_by_environment"]["unknown"]))
    if not math.isfinite(criticality_fraction) or not 0 <= criticality_fraction <= 1:
        raise SimulationError("resource_criticality must be between 0 and 1")
    criticality = criticality_fraction * 100
    weights = cfg["weights"]
    score = float(weights["performance_constraint"]) * perf + float(weights["workload_variability"]) * volatility + float(weights["resource_criticality"]) * criticality
    violations = []
    if post_cpu_p95 > cpu_limit: violations.append(f"Projected p95 CPU {post_cpu_p95:.1f}% exceeds {cpu_limit:.0f}% limit")
    if post_memory_p95 > mem_limit: violations.append(f"Projected p95 memory {post_memory_p95:.1f}% exceeds {mem_limit:.0f}% limit")
    thresholds = cfg["class_thresholds"]
    label = "LOW" if score <= float(thresholds["low_max"]) else "MEDIUM" if score <= float(thresholds["medium_max"]) else "HIGH"
    return round(score, 1), label, violations

def risk_aware_score(monthly_saving: float, current_cost: float, evidence_confidence: float,
                     risk_score: float, catalog: dict[str, Any] | None = None) -> float:
    """A transparent score: savings fraction and evidence quality offset a risk penalty."""
    catalog = catalog or load_catalog()
    values = [monthly_saving, current_cost, evidence_confidence, risk_score]
    if not all(math.isfinite(float(v)) for v in values): raise SimulationError("Ranking inputs must be finite")
    if current_cost <= 0 or monthly_saving < 0 or not 0 <= evidence_confidence <= 1 or not 0 <= risk_score <= 100:
        raise SimulationError("Invalid ranking input: cost > 0, savings >= 0, confidence in [0,1], risk in [0,100] required")
    weights = catalog["objective"]["weights"]
    saving_fraction = min(1.0, monthly_saving / current_cost)
    normalized_risk = risk_score / 100.0
    return round(100 * (float(weights["savings_fraction"]) * saving_fraction
                        + float(weights["evidence_quality"]) * evidence_confidence
                        - float(weights["risk_penalty"]) * normalized_risk ** 2), 2)

def _candidate(action: str, target: str | None, history: pd.DataFrame, provider_cfg: dict[str, Any],
               region_factor: float, current_sku: dict[str, Any] | None, current_cost: float,
               current_compute: float, current_storage: float, current_network: float,
               evidence: dict[str, Any], catalog: dict[str, Any], consistency: float,
               expected_cpu_p95: float, expected_memory_p95: float,
               expected_cpu_mean: float | None = None, expected_memory_mean: float | None = None,
               proposed_compute: float | None = None, proposed_storage: float | None = None,
               model_confidence_scale: float = 1.0, assumptions: list[str] | None = None,
               action_details: dict[str, Any] | None = None) -> dict[str, Any]:
    new_compute = current_compute if proposed_compute is None else proposed_compute
    new_storage = current_storage if proposed_storage is None else proposed_storage
    proposed = max(0.0, current_cost - current_compute - current_storage - current_network + new_compute + new_storage + current_network)
    saving = current_cost - proposed
    environment = str(history.environment.iloc[-1])
    criticality_value = history.iloc[-1].get("resource_criticality")
    criticality_value = float(criticality_value) if pd.notna(criticality_value) else None
    risk_score, risk, violations = _risk(expected_cpu_p95, expected_memory_p95, evidence, environment, catalog, criticality_value)
    confidence, confidence_parts = _confidence(evidence, consistency, catalog)
    confidence = round(confidence * model_confidence_scale, 3)
    ranking_quality = round(_ranking_evidence_quality(confidence_parts, catalog) * model_confidence_scale, 3)
    if not evidence["sufficient"]:
        violations.append(evidence["warning"] or "Insufficient historical evidence")
    if saving <= .01:
        violations.append("The modeled option does not produce positive savings")
    feasible = not violations
    return {"option_id": f"{history.cloud_provider.iloc[-1]}:{history.resource_id.iloc[-1]}:{action}:{target or 'scheduled'}", "action": action, "target_instance": target,
            "current_instance": str(history.instance_type.iloc[-1]), "current_monthly_cost": round(current_cost, 2),
            "proposed_monthly_cost": round(proposed, 2), "monthly_saving": round(saving, 2),
            "annualized_saving": round(round(saving, 2) * 12, 2), "currency": catalog["currency"],
            "expected_cpu_mean_pct": round(evidence["cpu_mean_pct"] if expected_cpu_mean is None else expected_cpu_mean, 1),
            "expected_cpu_p95_pct": round(expected_cpu_p95, 1),
            "expected_memory_mean_pct": round(evidence["memory_mean_pct"] if expected_memory_mean is None else expected_memory_mean, 1),
            "expected_memory_p95_pct": round(expected_memory_p95, 1),
            "performance_constraint_violations": violations, "feasible": feasible,
            "risk_score": risk_score, "risk": "HIGH" if not evidence["sufficient"] else risk,
            "workload_risk": {"performance_component_pct": round(min(100, max(expected_cpu_p95 / float(catalog['risk']['max_p95_cpu_pct']), expected_memory_p95 / float(catalog['risk']['max_p95_memory_pct'])) * 100), 1),
                              "variability_component_pct": round(min(100, max(evidence["cpu_cv"], evidence["memory_cv"]) * 100), 1),
                              "criticality_component_pct": round((criticality_value if criticality_value is not None else float(catalog["risk"]["criticality_by_environment"].get(environment.lower(), catalog["risk"]["criticality_by_environment"]["unknown"]))) * 100, 1)},
            "evidence_confidence": confidence, "confidence_factors": confidence_parts,
            "ranking_evidence_quality": ranking_quality,
            "confidence_interpretation": "Evidence quality score, not probability that the action will succeed.",
            "evidence": evidence, "assumptions": assumptions or [],
            "limitations": ["Catalog rates are illustrative INR assumptions, not provider quotes.", "Percentile utilization does not model application latency, queueing, or correlated multi-resource bottlenecks."],
            "action_details": action_details or {}}

def simulate_options(df: pd.DataFrame, resource_id: str, target_instance: str | None = None,
                     catalog: dict[str, Any] | None = None, cloud_provider: str | None = None) -> dict[str, Any]:
    """Compare all supported actions for a resource and report rejected actions with reasons."""
    catalog = catalog or load_catalog()
    validate_catalog(catalog)
    rows = df[df.resource_id.astype(str) == str(resource_id)].copy()
    if cloud_provider is not None:
        rows = rows[rows.cloud_provider.astype(str) == str(cloud_provider)]
    elif rows.cloud_provider.astype(str).nunique() > 1:
        raise SimulationError(f"Resource ID {resource_id} is ambiguous across providers; specify cloud_provider")
    rows = rows.sort_values("timestamp")
    if rows.empty: raise KeyError(resource_id)
    latest = pd.to_datetime(rows.timestamp, utc=True).max()
    rows = rows[pd.to_datetime(rows.timestamp, utc=True) >= latest - pd.Timedelta(days=89)].copy()
    # Retain normalized row order for meaningful baseline mean/percentile calculations.
    rows = rows.sort_values("timestamp")
    last = rows.iloc[-1]
    provider_cfg, region_cfg, current_sku = resolve_resource(catalog, str(last.cloud_provider), str(last.region), str(last.service), str(last.instance_type))
    if (pd.to_numeric(rows.monthly_cost, errors="coerce") < 0).any(): raise SimulationError("Negative observed costs are invalid")
    evidence = _workload_evidence(rows, catalog)
    current_cost = float(pd.to_numeric(rows.monthly_cost, errors="coerce").tail(30).mean())
    region_factor = region_cfg["factor"]
    compute_series, storage_series, network_series = _component_series(rows, provider_cfg, region_factor, current_sku, catalog)
    current_compute, current_storage, current_network = float(compute_series.tail(30).mean()), float(storage_series.tail(30).mean()), float(network_series.tail(30).mean())
    # Imported data without a component breakdown can make reconstructed components exceed its observed total.
    component_total = current_compute + current_storage + current_network
    if component_total > current_cost and component_total > 0:
        scale = current_cost / component_total
        current_compute *= scale; current_storage *= scale; current_network *= scale
    rejected: list[dict[str, Any]] = []
    alternatives: list[dict[str, Any]] = []
    assumptions_base = [f"Catalog {catalog['catalog_version']}; {catalog['price_source']} pricing; no market quote implied.",
                        "Current run-rate is the mean of the latest 30 observed daily monthly-normalized cost values.",
                        f"Compute billing is projected over {catalog['billing']['monthly_projection_days']} days using observed running hours."]
    current = {"instance_type": str(last.instance_type), "monthly_cost": round(current_cost, 2), "currency": catalog["currency"],
               "observations": evidence["observations"], "evidence": evidence,
               "cost_components_monthly": {"compute": round(current_compute, 2), "storage": round(current_storage, 2), "network": round(current_network, 2)}}

    if current_sku and str(last.service) in provider_cfg.get("compute_services", []):
        targets = list(provider_cfg["compute_skus"][str(last.instance_type)].get("next_smaller", []))
        if target_instance:
            if target_instance not in targets:
                rejected.append({"action": "instance_rightsizing", "target_instance": target_instance, "reason": f"Unsupported transition from {last.instance_type} to {target_instance}"})
            else: targets = [target_instance]
        for target in targets:
            target_sku = provider_cfg["compute_skus"][target]
            target_cpu = pd.to_numeric(rows.cpu_utilization, errors="coerce") * float(current_sku["vcpu"]) / float(target_sku["vcpu"])
            target_mem = pd.to_numeric(rows.memory_utilization, errors="coerce") * float(current_sku["memory_gib"]) / float(target_sku["memory_gib"])
            cpu_p95, mem_p95 = _quantile(target_cpu, .95), _quantile(target_mem, .95)
            billing = catalog["billing"]
            target_compute_values = (pd.to_numeric(rows.running_hours).clip(0, 24) * float(billing["monthly_projection_days"])
                                     * float(target_sku["hourly_inr"]) * region_factor
                                     * pd.Series([_usage_multiplier(c, m, catalog) for c, m in zip(target_cpu, target_mem)], index=rows.index))
            compute_candidate = float(target_compute_values.tail(30).mean())
            consistency = float(((target_cpu <= float(catalog["risk"]["max_p95_cpu_pct"])) & (target_mem <= float(catalog["risk"]["max_p95_memory_pct"]))).mean())
            option = _candidate("instance_rightsizing", target, rows, provider_cfg, region_factor, current_sku,
                                current_cost, current_compute, current_storage, current_network, evidence, catalog, consistency,
                                cpu_p95, mem_p95, expected_cpu_mean=float(target_cpu.mean()), expected_memory_mean=float(target_mem.mean()),
                                proposed_compute=compute_candidate, assumptions=assumptions_base + [
                                    f"CPU capacity scales by vCPU ratio {current_sku['vcpu']}/{target_sku['vcpu']}; memory scales by capacity ratio {current_sku['memory_gib']}/{target_sku['memory_gib']} GiB.",
                                    "Historical daily utilization observations are scaled by target capacity; p95 is used for feasibility, not just the average."],
                                action_details={"target_vcpu": target_sku["vcpu"], "target_memory_gib": target_sku["memory_gib"]})
            if option["feasible"]: alternatives.append(option)
            else: rejected.append(option)

    schedule = catalog["billing"]["shutdown"]
    environment = str(last.environment).lower()
    if environment in [str(x).lower() for x in schedule["eligible_environments"]] and current_sku is not None:
        runtime = pd.to_numeric(rows.running_hours, errors="coerce").clip(0, 24)
        minimum_runtime = float(schedule["minimum_runtime_hours_per_day"])
        eligible = float(schedule["eligible_hours_per_day"])
        possible_by_runtime = (runtime - minimum_runtime).clip(lower=0)
        shutdown_each_day = np.minimum(possible_by_runtime.to_numpy(dtype=float), eligible)
        avg_shutdown = float(np.mean(shutdown_each_day))
        projected_runtime = float(runtime.mean()) - avg_shutdown
        if avg_shutdown > 0 and projected_runtime >= minimum_runtime:
            # Runtime reduction follows observed eligible hours. Compute utilization increases only during remaining runtime.
            active_ratio = float(runtime.mean()) / max(projected_runtime, 1)
            cpu = pd.to_numeric(rows.cpu_utilization, errors="coerce") * active_ratio
            mem = pd.to_numeric(rows.memory_utilization, errors="coerce") * active_ratio
            cpu_p95, mem_p95 = _quantile(cpu, .95), _quantile(mem, .95)
            billing = catalog["billing"]
            active_util = pd.Series([_usage_multiplier(c, m, catalog) for c, m in zip(cpu, mem)], index=rows.index)
            compute_candidate = float((pd.Series(projected_runtime, index=rows.index)
                * float(billing["monthly_projection_days"]) * float(current_sku["hourly_inr"]) * region_factor * active_util).tail(30).mean())
            consistency = float((possible_by_runtime > 0).mean())
            option = _candidate("scheduled_shutdown", None, rows, provider_cfg, region_factor, current_sku,
                                current_cost, current_compute, current_storage, current_network, evidence, catalog, consistency,
                                cpu_p95, mem_p95, expected_cpu_mean=float(cpu.mean()), expected_memory_mean=float(mem.mean()), proposed_compute=compute_candidate,
                                model_confidence_scale=.75 if not schedule.get("requires_hourly_schedule_evidence") else 1.0,
                                assumptions=assumptions_base + [f"Up to {avg_shutdown:.1f} eligible hours/day are removed after retaining {projected_runtime:.1f} average runtime hours/day; minimum is {minimum_runtime:.1f} hours.",
                                    f"Schedule window {schedule['window']}; daily records cannot establish that workload is actually idle in that window."],
                                action_details={"eligible_hours_per_day": round(avg_shutdown, 2), "projected_runtime_hours_per_day": round(projected_runtime, 2),
                                                "non_shutdown_hours_per_day": round(24 - avg_shutdown, 2), "minimum_runtime_hours_per_day": minimum_runtime,
                                                "window_evidence_reliability": schedule["daily_aggregate_evidence_reliability"]})
            if option["feasible"]: alternatives.append(option)
            else: rejected.append(option)
        else:
            rejected.append({"action": "scheduled_shutdown", "reason": "Insufficient runtime above the configured minimum to schedule shutdown hours."})

    allocated = pd.to_numeric(rows.get("allocated_storage_gb", pd.Series(np.nan, index=rows.index)), errors="coerce")
    if allocated.notna().all() and float(allocated.iloc[-1]) > 0:
        observed_gb = allocated * (pd.to_numeric(rows.storage_usage, errors="coerce").clip(0, 100) / 100)
        p99_used = _quantile(observed_gb, .99)
        storage_cfg = catalog["billing"]["storage"]
        headroom = float(storage_cfg["headroom_fraction"])
        increment = float(storage_cfg["capacity_increment_gb"])
        target_capacity = max(float(storage_cfg["minimum_capacity_gb"]), math.ceil((p99_used * (1 + headroom)) / increment) * increment)
        current_capacity = float(allocated.iloc[-1])
        if target_capacity < current_capacity:
            fill = float(pd.to_numeric(rows.storage_usage, errors="coerce").tail(30).mean()) / 100
            io = float(catalog["billing"]["storage_io_surcharge_per_usage_fraction"])
            proposed_storage = target_capacity * float(provider_cfg["storage_inr_per_gb_month"]) * (1 + io * fill)
            consistency = float((observed_gb <= target_capacity / (1 + headroom)).mean())
            option = _candidate("storage_optimization", f"{int(target_capacity)}GB", rows, provider_cfg, region_factor, current_sku,
                                current_cost, current_compute, current_storage, current_network, evidence, catalog, consistency,
                                evidence["cpu_p95_pct"] or 0, evidence["memory_p95_pct"] or 0, proposed_storage=proposed_storage,
                                assumptions=assumptions_base + [f"Target capacity is p99 observed used storage plus {headroom:.0%} headroom, rounded to {increment:.0f} GB increments.",
                                                               "Storage unit rate is provisioned GB-month; storage utilization also affects the configured I/O surcharge."],
                                action_details={"current_capacity_gb": current_capacity, "target_capacity_gb": target_capacity,
                                                "p99_used_gb": round(p99_used, 2), "headroom_fraction": headroom})
            if option["feasible"]: alternatives.append(option)
            else: rejected.append(option)
        else:
            rejected.append({"action": "storage_optimization", "reason": f"p99 use plus headroom requires {target_capacity:.0f} GB; current allocation is {current_capacity:.0f} GB."})
    else:
        rejected.append({"action": "storage_optimization", "reason": "Allocated storage capacity is missing or invalid."})

    for option in alternatives:
        # Objective: savings share (benefit) + evidence quality - expected risk penalty.
        objective = catalog["objective"]["weights"]
        save_share = option["monthly_saving"] / max(current_cost, .01)
        option["savings_only_score"] = round(option["monthly_saving"], 2)
        option["risk_aware_score"] = risk_aware_score(option["monthly_saving"], current_cost,
                                                       option["ranking_evidence_quality"], option["risk_score"], catalog)
        option["decision_explanation"] = (f"{objective['savings_fraction']:.0%} weight on savings share, {objective['evidence_quality']:.0%} on evidence quality, "
            f"and {objective['risk_penalty']:.0%} on squared normalized risk. The evidence term excludes stability and action-fit factors already represented in risk. Savings share={save_share:.3f}; risk={option['risk_score']:.1f}/100.")
        option.pop("_action_details", None)
    alternatives.sort(key=lambda item: item["risk_aware_score"], reverse=True)
    return {"resource_id": str(resource_id), "cloud_provider": str(last.cloud_provider), "catalog_version": catalog["catalog_version"], "pricing_source": catalog["price_source"],
            "current": current, "data_sufficient": evidence["sufficient"], "warning": evidence["warning"],
            "alternatives": alternatives, "rejected_options": rejected,
            "objective": catalog["objective"]["weights"],
            "risk_formula": "0.50 × p95 performance pressure + 0.25 × utilization variability + 0.25 × environment criticality; 100-point heuristic, not probability.",
            "risk_thresholds": catalog["risk"]["class_thresholds"]}
