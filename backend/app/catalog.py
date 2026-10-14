"""Validated reader for the illustrative, user-editable pricing and resource catalog."""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "catalog" / "pricing_catalog.json"

class CatalogError(ValueError):
    """Catalog is absent, malformed, or does not support a requested configuration."""

@lru_cache(maxsize=8)
def load_catalog(path: str | None = None) -> dict[str, Any]:
    catalog_path = Path(path or os.getenv("FINOPTIX_CATALOG", DEFAULT_CATALOG))
    try:
        catalog = json.loads(catalog_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Could not load pricing catalog at {catalog_path}: {exc}") from exc
    validate_catalog(catalog)
    return catalog

def validate_catalog(catalog: dict[str, Any]) -> None:
    if catalog.get("currency") != "INR" or catalog.get("market_price_claim") is not False:
        raise CatalogError("Catalog must specify INR and explicitly disclaim current market prices")
    if not catalog.get("providers"):
        raise CatalogError("Catalog must define at least one provider")
    billing = catalog.get("billing", {})
    if float(billing.get("monthly_projection_days", 0)) <= 0 or float(billing.get("hours_per_day", 0)) <= 0:
        raise CatalogError("Billing projection days and hours per day must be positive")
    utilization_cost = billing.get("utilization_cost_multiplier", {})
    if any(not math.isfinite(float(utilization_cost.get(key, float("nan")))) or float(utilization_cost[key]) < 0
           for key in ("base", "per_utilization_fraction")):
        raise CatalogError("Utilization cost multipliers must be finite and non-negative")
    shutdown = billing.get("shutdown", {})
    if not 0 <= float(shutdown.get("eligible_hours_per_day", -1)) <= float(billing["hours_per_day"]):
        raise CatalogError("Shutdown eligible hours must fit within one day")
    if not 0 <= float(shutdown.get("minimum_runtime_hours_per_day", -1)) <= float(billing["hours_per_day"]):
        raise CatalogError("Shutdown minimum runtime must fit within one day")
    storage = billing.get("storage", {})
    if float(storage.get("minimum_capacity_gb", 0)) <= 0 or float(storage.get("capacity_increment_gb", 0)) <= 0:
        raise CatalogError("Storage minimum capacity and increment must be positive")
    if float(storage.get("headroom_fraction", -1)) < 0:
        raise CatalogError("Storage headroom fraction must be non-negative")
    for group in (catalog.get("risk", {}).get("weights", {}), catalog.get("confidence", {}).get("weights", {}), catalog.get("objective", {}).get("weights", {})):
        if not group or any(not math.isfinite(float(value)) or float(value) < 0 for value in group.values()) or not math.isclose(sum(map(float, group.values())), 1.0, abs_tol=1e-8):
            raise CatalogError("Risk, confidence, and objective weights must be non-negative and sum to 1")
    reliability = float(catalog.get("confidence", {}).get("illustrative_price_reliability", -1))
    if not 0 <= reliability <= 1:
        raise CatalogError("Illustrative price reliability must be between 0 and 1")
    for provider, cfg in catalog["providers"].items():
        if not cfg.get("regions") or not cfg.get("services"):
            raise CatalogError(f"Provider {provider} is missing regions or services")
        for region, factor in cfg["regions"].items():
            if not math.isfinite(float(factor)) or float(factor) <= 0:
                raise CatalogError(f"Region multiplier must be positive: {provider}/{region}")
        for sku, spec in cfg.get("compute_skus", {}).items():
            price, vcpu, memory = float(spec["hourly_inr"]), int(spec["vcpu"]), float(spec["memory_gib"])
            if not math.isfinite(price) or price <= 0 or vcpu <= 0 or memory <= 0:
                raise CatalogError(f"Invalid compute SKU price/capacity: {provider}/{sku}")
            for target in spec.get("next_smaller", []):
                if target not in cfg["compute_skus"]:
                    raise CatalogError(f"Unknown transition target {provider}/{sku} -> {target}")
                t = cfg["compute_skus"][target]
                if int(t["vcpu"]) >= vcpu or float(t["memory_gib"]) >= memory:
                    raise CatalogError(f"Transition must reduce both CPU and memory: {provider}/{sku} -> {target}")
        for key in ("storage_inr_per_gb_month", "network_inr_per_gb"):
            if float(cfg[key]) < 0 or not math.isfinite(float(cfg[key])):
                raise CatalogError(f"{provider}.{key} must be finite and non-negative")

def resolve_resource(catalog: dict[str, Any], provider: str, region: str, service: str, sku: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    cfg = catalog["providers"].get(provider)
    if cfg is None: raise CatalogError(f"Unsupported provider: {provider}")
    if region not in cfg["regions"]: raise CatalogError(f"Unsupported region {region} for {provider}")
    if service not in cfg["services"]: raise CatalogError(f"Unsupported service {service} for {provider}")
    sku_spec = cfg.get("compute_skus", {}).get(sku)
    if service in cfg.get("compute_services", []) and sku_spec is None:
        raise CatalogError(f"Unknown compute SKU {sku} for {provider}")
    return cfg, {"factor": float(cfg["regions"][region])}, sku_spec
