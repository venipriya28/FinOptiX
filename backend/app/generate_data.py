"""Generate seeded, correlated resource-day billing data with scenario evaluation labels."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from .catalog import load_catalog

OUT = Path(__file__).resolve().parents[2] / "data" / "demo_usage.csv"
AS_OF = pd.Timestamp("2026-10-07", tz="UTC")

SCENARIOS = [
    "normal", "underutilized", "idle", "over_provisioned", "highly_utilized",
    "sudden_cost_spike", "gradual_cost_increase", "seasonal_usage",
    "newly_deployed", "occasional_peak",
]

PROVIDERS = {
    "AWS": {"regions": ["ap-south-1", "us-east-1", "eu-west-1"], "services": ["EC2", "RDS", "EBS"],
            "skus": ["t3.medium", "m5.xlarge", "m5.2xlarge"], "hourly": {"t3.medium": 4.5, "m5.xlarge": 14.2, "m5.2xlarge": 28.4}},
    "Azure": {"regions": ["centralindia", "eastus", "westeurope"], "services": ["Azure VM", "Azure SQL", "Managed Disk"],
              "skus": ["Standard_D2s_v5", "Standard_D4s_v5", "Standard_D8s_v5"], "hourly": {"Standard_D2s_v5": 10.1, "Standard_D4s_v5": 20.2, "Standard_D8s_v5": 40.4}},
    "GCP": {"regions": ["asia-south1", "us-central1", "europe-west1"], "services": ["Compute Engine", "Cloud SQL", "Persistent Disk"],
            "skus": ["e2-standard-2", "n2-standard-4", "n2-standard-8"], "hourly": {"e2-standard-2": 8.3, "n2-standard-4": 19.6, "n2-standard-8": 39.2}},
}
REGION_FACTOR = {"ap-south-1": 1.00, "centralindia": 0.96, "asia-south1": 0.93, "us-east-1": 1.08,
                 "eastus": 1.04, "us-central1": 1.00, "eu-west-1": 1.10, "westeurope": 1.08, "europe-west1": 1.05}
RISK = {"underutilized": "LOW", "idle": "LOW", "over_provisioned": "LOW", "storage_inefficient": "LOW",
        "normal": "NONE", "highly_utilized": "HIGH", "sudden_cost_spike": "NONE", "gradual_cost_increase": "NONE",
        "seasonal_usage": "NONE", "newly_deployed": "NONE", "occasional_peak": "MEDIUM"}
SAVING_RATE = {"underutilized": .42, "idle": .70, "over_provisioned": .32, "storage_inefficient": .18}

def generate(seed: int = 2026, days: int = 365, resources: int = 40, scenario_offset: int = 0) -> pd.DataFrame:
    """Return at least one year of daily records. Same parameters always reproduce the same rows.

    Resource-day cost is a monthly-normalized run rate, not a daily invoice line. ETL derives
    daily_cost as monthly_cost / 30. This makes resource recommendations directly comparable
    while preserving daily time series for forecasting and anomaly evaluation.
    """
    if days < 180:
        raise ValueError("Use at least 180 days of history; the default provides 12 months")
    if resources < len(SCENARIOS):
        raise ValueError(f"Use at least {len(SCENARIOS)} resources to cover every scenario")
    rng = np.random.default_rng(seed)
    rate_catalog = load_catalog()
    start = AS_OF - pd.Timedelta(days=days - 1)
    dates = pd.date_range(start=start.normalize(), periods=days, freq="D")
    rows: list[dict] = []

    for n in range(resources):
        scenario = SCENARIOS[(n + scenario_offset) % len(SCENARIOS)]
        provider = list(PROVIDERS)[n % len(PROVIDERS)]
        scenario_cfg = PROVIDERS[provider]
        service = scenario_cfg["services"][(n // len(PROVIDERS)) % len(scenario_cfg["services"])]
        sku = scenario_cfg["skus"][(n // 2) % len(scenario_cfg["skus"])]
        region = scenario_cfg["regions"][(n // 3) % len(scenario_cfg["regions"])]
        resource_id = f"{service.split()[0].upper().replace('SQL','DB')}-{100+n:03d}"
        resource_type = "virtual_machine" if service in {"EC2", "Azure VM", "Compute Engine"} else "database" if "SQL" in service or service in {"RDS", "Cloud SQL"} else "block_storage"
        # Scenarios are assigned deterministically to resources; scenario measurements vary daily.
        base_cpu = {"normal": 42, "underutilized": 11, "idle": 2, "over_provisioned": 18,
                    "highly_utilized": 79, "sudden_cost_spike": 34, "gradual_cost_increase": 30,
                    "seasonal_usage": 36, "newly_deployed": 32, "occasional_peak": 28}[scenario]
        base_mem = {"normal": 48, "underutilized": 18, "idle": 3, "over_provisioned": 23,
                    "highly_utilized": 82, "sudden_cost_spike": 43, "gradual_cost_increase": 37,
                    "seasonal_usage": 40, "newly_deployed": 38, "occasional_peak": 34}[scenario]
        base_runtime = {"idle": 1.0, "newly_deployed": 17, "normal": 21, "underutilized": 23,
                        "over_provisioned": 24, "highly_utilized": 24}.get(scenario, 20)
        provider_prices = rate_catalog["providers"][provider]
        sku_prices = provider_prices["compute_skus"][sku]
        allocated_storage = float(rng.choice([100, 250, 500, 1000]))
        storage_rate = float(provider_prices["storage_inr_per_gb_month"])
        storage_fill = {"underutilized": .18, "idle": .12, "over_provisioned": .28}.get(scenario, .58)
        if scenario == "storage_inefficient": storage_fill = .12
        network_price = float(provider_prices["network_inr_per_gb"])
        deployed_day = days - 75 if scenario == "newly_deployed" else 0
        prev_growth = 1.0

        for day, dt in enumerate(dates):
            if day < deployed_day:
                continue
            annual_season = 1 + .16 * np.sin(2 * np.pi * (dt.dayofyear - 30) / 365.25)
            weekday = .90 if dt.dayofweek >= 5 else 1.03
            seasonal = annual_season * weekday if scenario == "seasonal_usage" else 1.0
            demand = seasonal
            if scenario == "gradual_cost_increase":
                prev_growth = 1 + .0018 * day  # approximately +66% by day 365
                demand *= prev_growth
            if scenario == "newly_deployed":
                demand *= min(1.0, .55 + .45 * (day - deployed_day) / 35)
            if scenario == "occasional_peak" and day % 23 in (0, 1):
                demand *= 1.7

            runtime = float(np.clip(rng.normal(base_runtime, 1.4), 0, 24))
            if scenario == "idle":
                runtime = float(np.clip(rng.normal(1.2, .5), .1, 3.0))
            cpu = float(np.clip(rng.normal(base_cpu * demand, 5), 0.5, 99))
            mem = float(np.clip(rng.normal(base_mem * demand, 5), 0.5, 99))
            if scenario == "sudden_cost_spike" and days - 5 <= day < days - 2:
                # Short, labeled incident: runtime, network, utilization, and cost move together.
                runtime = min(24, runtime + 3)
                cpu = min(99, cpu + 45)
                mem = min(99, mem + 28)
            if scenario == "occasional_peak" and day % 23 in (0, 1):
                cpu = min(99, cpu + 20); mem = min(99, mem + 12)

            utilization = (cpu + mem) / 200
            storage_usage = float(np.clip(rng.normal(storage_fill * demand, .035), .03, .99))
            # Network throughput co-moves with workload demand and has occasional independent noise.
            network_gb = float(max(.1, rng.lognormal(np.log(max(2, 25 + 125 * cpu / 100) * demand), .22)))
            if scenario == "sudden_cost_spike" and days - 5 <= day < days - 2:
                network_gb *= 3.5

            compute_multiplier = rate_catalog["billing"]["utilization_cost_multiplier"]
            compute_month = (30 * runtime * float(sku_prices["hourly_inr"]) * float(provider_prices["regions"][region])
                             * (float(compute_multiplier["base"]) + float(compute_multiplier["per_utilization_fraction"]) * utilization))
            # A small usage-sensitive service component models I/O, licensing, and metered operations.
            storage_month = allocated_storage * storage_rate * (1 + float(rate_catalog["billing"]["storage_io_surcharge_per_usage_fraction"]) * storage_usage)
            billable_network_month = max(0, network_gb * 30 - float(rate_catalog["billing"]["network_free_gb_per_month"])) * network_price
            cost = compute_month + storage_month + billable_network_month
            if scenario == "sudden_cost_spike" and days - 5 <= day < days - 2:
                cost *= 2.4
            # Measurement jitter is low and multiplicative; all dominant cost changes have causal drivers.
            cost *= float(rng.normal(1.0, .018))
            is_anomaly = bool(scenario == "sudden_cost_spike" and days - 5 <= day < days - 2)

            opt_type = {"underutilized": "downsize_instance", "idle": "schedule_shutdown",
                        "over_provisioned": "downsize_instance", "storage_inefficient": "rightsize_storage"}.get(scenario, "none")
            saving_rate = SAVING_RATE.get(scenario, 0)
            rows.append({
                "timestamp": dt.isoformat(), "cloud_provider": provider, "service": service, "region": region,
                "resource_id": resource_id, "resource_type": resource_type, "instance_type": sku,
                "cpu_utilization": round(cpu, 2), "memory_utilization": round(mem, 2),
                "network_usage": round(network_gb, 2), "storage_usage": round(storage_usage * 100, 2),
                "allocated_storage_gb": allocated_storage, "running_hours": round(runtime, 2),
                "compute_monthly_cost": round(compute_month, 2), "storage_monthly_cost": round(storage_month, 2),
                "network_monthly_cost": round(billable_network_month, 2),
                "monthly_cost": round(cost, 2), "department": ["Engineering", "Analytics", "Payments", "Platform"][n % 4],
                "project": ["Atlas", "Mercury", "Checkout", "DataLake"][n % 4],
                "environment": "production" if n % 4 else "development",
                "resource_criticality": 1.0 if n % 4 == 2 else 0.8 if n % 4 else 0.35,
                "scenario_type": scenario,
                # Evaluation-only labels. Excluded from REQUIRED, model feature lists, and simulator inputs.
                "is_anomaly": is_anomaly, "optimization_opportunity": opt_type != "none",
                "optimization_type": opt_type, "expected_risk": RISK.get(scenario, "NONE"),
                "expected_saving": round(cost * saving_rate, 2),
            })
    return pd.DataFrame(rows).sort_values(["timestamp", "resource_id"]).reset_index(drop=True)

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame = generate()
    frame.to_csv(OUT, index=False)
    print(f"Wrote {len(frame):,} resource-day rows across {frame.resource_id.nunique()} resources to {OUT}")
    print(f"History: {frame.timestamp.min()} to {frame.timestamp.max()}; scenarios: {', '.join(sorted(frame.scenario_type.unique()))}")
