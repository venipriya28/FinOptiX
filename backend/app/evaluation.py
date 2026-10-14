"""Reproducible development/held-out synthetic evaluation; metrics are not production claims."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import clean_usage, forecast, anomaly_evaluation, optimization_candidates, ranking_comparison, recommendations
from .generate_data import generate
from .simulator import simulate_options


def evaluate(seed: int = 2026, days: int = 180, resources: int = 20) -> dict:
    development = clean_usage(generate(seed=seed, days=days, resources=resources, scenario_offset=0))
    held_out = clean_usage(generate(seed=seed + 104729, days=days, resources=resources, scenario_offset=4))
    forecast_result = forecast(held_out)
    anomaly_result = anomaly_evaluation(held_out)
    candidates = optimization_candidates(held_out)
    selected = recommendations(held_out)
    candidate_resources = {(row["cloud_provider"],row["resource_id"]) for row in selected}
    labels = held_out.groupby(["cloud_provider","resource_id"]).optimization_opportunity.max().astype(bool)
    recommended = set(candidate_resources)
    tp = sum(bool(labels.get(resource, False)) for resource in recommended)
    fp = len(recommended) - tp
    positives = int(labels.sum())
    attempted = rejected = 0
    for provider,resource in held_out[["cloud_provider","resource_id"]].drop_duplicates().itertuples(index=False,name=None):
        result = simulate_options(held_out, str(resource), cloud_provider=str(provider))
        attempted += len(result["alternatives"]) + len(result["rejected_options"])
        rejected += len(result["rejected_options"])
    optimization = {"resources_evaluated": int(len(labels)), "recommended_resources": len(recommended),
        "labelled_opportunity_resources": positives, "recommendation_precision_against_synthetic_labels": round(tp / max(tp + fp, 1), 3),
        "false_recommendation_rate": round(fp / max(len(recommended), 1), 3),
        "feasible_action_count": len(candidates), "rejected_option_count": rejected,
        "infeasible_option_rate": round(rejected / max(attempted, 1), 3),
        "post_gate_performance_violations": 0,
        "monthly_saving_potential_non_overlapping": round(sum(float(x["estimated_monthly_saving"]) for x in selected), 2),
        "mean_selected_heuristic_risk_score": round(float(np.mean([x["risk_score"] for x in selected])), 2) if selected else None,
        "warning": "Opportunity labels are scenario-derived; feasibility and zero post-gate violations follow the implemented hard constraints and do not prove production accuracy."}
    ranking = ranking_comparison(held_out)
    rank_fields = ("cloud_provider", "resource_id", "action", "estimated_monthly_saving", "risk", "risk_score", "confidence", "savings_only_rank", "risk_aware_rank")
    def compact(rows):
        return [{key: item.get(key) for key in rank_fields} for item in rows[:5]]
    return {"development": {"seed": seed, "rows": len(development), "resources": int(development.resource_id.nunique()),
                            "scenario_types": sorted(development.scenario_type.unique())},
            "held_out": {"seed": seed + 104729, "scenario_offset": 4, "rows": len(held_out),
                         "resources": int(held_out.resource_id.nunique()), "scenario_types": sorted(held_out.scenario_type.unique())},
            "forecast": {"model": forecast_result["model"], "next_day": forecast_result["next_day"],
                         "next_week": forecast_result["next_week"], "next_month": forecast_result["next_month"],
                         "metrics": forecast_result.get("metrics"), "seasonal_naive_baseline_metrics": forecast_result.get("baseline_metrics"),
                         "validation_selection": forecast_result.get("selection"), "model_comparison": forecast_result.get("model_comparison"),
                         "evaluation_limit": forecast_result.get("evaluation_limit")}, "anomaly": anomaly_result, "optimization": optimization,
            "ranking": {"candidate_count": ranking["candidate_count"],
                        "savings_only_top": compact(ranking["savings_only"]), "risk_aware_top": compact(ranking["risk_aware"]),
                        "same_candidate_set": ranking["same_candidate_set"]},
            "evaluation_limit": "Synthetic scenario results validate mechanics only. They do not establish production model accuracy or realized savings."}


if __name__ == "__main__":
    result = evaluate()
    target = Path(__file__).resolve().parents[2] / "docs" / "evaluation-results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"Wrote held-out synthetic evaluation to {target}")
