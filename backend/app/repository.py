"""Optional PostgreSQL persistence; CSV remains the portable demo/import format."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_CANDIDATES = [
    _ROOT / "sql" / "schema.sql",
    Path(__file__).resolve().parents[1] / "sql" / "schema.sql",
]
SCHEMA = Path(
    os.getenv(
        "FINOPTIX_SCHEMA",
        next((path for path in _SCHEMA_CANDIDATES if path.exists()), _SCHEMA_CANDIDATES[0]),
    )
)


def database_url() -> str | None:
    return os.getenv("FINOPTIX_DATABASE_URL") or None


def _connect():
    if not database_url():
        return None
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "FINOPTIX_DATABASE_URL is set but psycopg is not installed"
        ) from exc
    return psycopg.connect(database_url())


def initialize() -> bool:
    """Apply repeatable CREATE/ALTER statements on app startup."""
    conn = _connect()
    if conn is None:
        return False
    with conn:
        conn.execute(SCHEMA.read_text())
        conn.execute(
            "ALTER TABLE cloud_usage ADD COLUMN IF NOT EXISTS "
            "raw_record JSONB NOT NULL DEFAULT '{}'::jsonb"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_resource_observed "
            "ON cloud_usage(resource_fk, observed_at)"
        )
    conn.close()
    return True


def load_frame() -> pd.DataFrame | None:
    conn = _connect()
    if conn is None:
        return None
    with conn:
        rows = conn.execute(
            """SELECT u.raw_record
               FROM cloud_usage u
               JOIN cloud_resources r ON r.id = u.resource_fk
               ORDER BY u.observed_at, r.resource_id"""
        ).fetchall()
    conn.close()
    if not rows:
        return None
    return pd.DataFrame([row[0] for row in rows])


def persist_frame(
    frame: pd.DataFrame,
    source: str = "csv_upload",
    replace_existing: bool = False,
) -> int:
    """Persist observations; optionally replace the active demo dataset."""
    conn = _connect()
    if conn is None:
        return 0

    from psycopg.types.json import Jsonb

    with conn:
        if replace_existing:
            # This app currently exposes one active dataset, not tenant selection.
            # Remove artifacts tied to the previous dataset but retain audit history.
            for table in (
                "savings",
                "optimization_simulations",
                "recommendations",
                "anomalies",
                "model_predictions",
                "cloud_costs",
                "cloud_usage",
                "cloud_resources",
            ):
                conn.execute(f"DELETE FROM {table}")

        resource_columns = [
            "cloud_provider",
            "resource_id",
            "service",
            "region",
            "resource_type",
            "instance_type",
            "department",
            "project",
            "environment",
        ]
        unique_resources = frame[resource_columns].drop_duplicates(
            ["cloud_provider", "resource_id"]
        )
        conn.cursor().executemany(
            """INSERT INTO cloud_resources
               (provider, resource_id, service, region, resource_type,
                instance_type, department, project, environment)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT(provider, resource_id) DO UPDATE SET
                service=EXCLUDED.service,
                region=EXCLUDED.region,
                resource_type=EXCLUDED.resource_type,
                instance_type=EXCLUDED.instance_type,
                department=EXCLUDED.department,
                project=EXCLUDED.project,
                environment=EXCLUDED.environment""",
            unique_resources.itertuples(index=False, name=None),
        )

        resource_ids = {
            (str(provider), str(resource_id)): int(resource_pk)
            for resource_pk, provider, resource_id in conn.execute(
                "SELECT id, provider, resource_id FROM cloud_resources"
            ).fetchall()
        }
        usage_records = []
        cost_records = []

        for record in frame.to_dict(orient="records"):
            resource_fk = resource_ids[
                (str(record["cloud_provider"]), str(record["resource_id"]))
            ]
            observed = pd.Timestamp(record["timestamp"]).to_pydatetime()
            payload = dict(record)
            payload["timestamp"] = pd.Timestamp(record["timestamp"]).isoformat()
            # Normalize pandas/numpy scalars and non-finite values for JSONB.
            payload = json.loads(
                json.dumps(
                    payload,
                    allow_nan=False,
                    default=lambda value: (
                        None
                        if pd.isna(value)
                        else value.item()
                        if hasattr(value, "item")
                        else str(value)
                    ),
                )
            )
            usage_records.append(
                (
                    resource_fk,
                    observed,
                    record["cpu_utilization"],
                    record["memory_utilization"],
                    record["network_usage"],
                    record["storage_usage"],
                    record["running_hours"],
                    record.get("utilization_score"),
                    record.get("idle_hours"),
                    Jsonb(payload),
                )
            )
            observed_date = pd.Timestamp(record["timestamp"]).date()
            cost_records.append(
                (resource_fk, observed_date, observed_date, record["monthly_cost"], source)
            )

        conn.cursor().executemany(
            """INSERT INTO cloud_usage
               (resource_fk, observed_at, cpu_pct, memory_pct, network_usage,
                storage_usage, running_hours, utilization_score, idle_hours, raw_record)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT(resource_fk, observed_at) DO UPDATE SET
                cpu_pct=EXCLUDED.cpu_pct,
                memory_pct=EXCLUDED.memory_pct,
                network_usage=EXCLUDED.network_usage,
                storage_usage=EXCLUDED.storage_usage,
                running_hours=EXCLUDED.running_hours,
                utilization_score=EXCLUDED.utilization_score,
                idle_hours=EXCLUDED.idle_hours,
                raw_record=EXCLUDED.raw_record""",
            usage_records,
        )
        conn.cursor().executemany(
            """INSERT INTO cloud_costs
               (resource_fk, period_start, period_end, amount, currency, source)
               VALUES (%s,%s,%s,%s,'INR',%s)
               ON CONFLICT(resource_fk, period_start, period_end, source)
               DO UPDATE SET amount=EXCLUDED.amount""",
            cost_records,
        )
        count = len(usage_records)
        conn.execute(
            "INSERT INTO audit_logs(action, entity_type, entity_id, details) "
            "VALUES ('data_ingested', 'dataset', %s, %s)",
            (source, Jsonb({"rows": count, "resources": len(resource_ids)})),
        )
    conn.close()
    return count


def persist_json(
    table: str, values: list[dict[str, Any]], dataset: pd.DataFrame
) -> int:
    """Persist recomputable API artifacts for audit/history."""
    allowed = {"anomalies", "recommendations"}
    if table not in allowed:
        raise ValueError("Unsupported artifact table")

    conn = _connect()
    if conn is None or not values:
        if conn is not None:
            conn.close()
        return 0

    from psycopg.types.json import Jsonb

    resources = dataset[["cloud_provider", "resource_id"]].drop_duplicates()
    lookup: dict[tuple[str, str], int] = {}
    with conn:
        for provider, resource_id in resources.itertuples(index=False, name=None):
            row = conn.execute(
                "SELECT id FROM cloud_resources WHERE provider=%s AND resource_id=%s",
                (provider, resource_id),
            ).fetchone()
            if row:
                lookup[(str(provider), str(resource_id))] = int(row[0])

        if table == "anomalies":
            for item in values:
                conn.execute(
                    """INSERT INTO anomalies
                       (resource_fk, detected_at, detector, severity, cost_impact, cause, evidence)
                       VALUES (%s,%s,'iforest_rolling_mad_hybrid',%s,%s,%s,%s)
                       ON CONFLICT(resource_fk, detected_at, detector) DO UPDATE SET
                        severity=EXCLUDED.severity,
                        cost_impact=EXCLUDED.cost_impact,
                        cause=EXCLUDED.cause,
                        evidence=EXCLUDED.evidence""",
                    (
                        lookup.get(
                            (str(item.get("cloud_provider")), str(item.get("resource_id")))
                        ),
                        item.get("timestamp"),
                        item.get("severity", "MEDIUM"),
                        item.get("cost_impact_vs_baseline", 0),
                        item.get("cause", ""),
                        Jsonb(item),
                    ),
                )
        else:
            for item in values:
                resource_fk = lookup.get(
                    (str(item.get("cloud_provider")), str(item.get("resource_id")))
                )
                if resource_fk is None:
                    continue
                conn.execute(
                    """INSERT INTO recommendations
                       (resource_fk, action, reason, evidence, monthly_saving,
                        confidence, risk, priority_score)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT(resource_fk, action, status) DO UPDATE SET
                        reason=EXCLUDED.reason,
                        evidence=EXCLUDED.evidence,
                        monthly_saving=EXCLUDED.monthly_saving,
                        confidence=EXCLUDED.confidence,
                        risk=EXCLUDED.risk,
                        priority_score=EXCLUDED.priority_score,
                        created_at=now()""",
                    (
                        resource_fk,
                        item.get("action", ""),
                        item.get("reason", ""),
                        Jsonb(item.get("evidence", {})),
                        item.get("estimated_monthly_saving", 0),
                        item.get("confidence", 0),
                        item.get("risk", "HIGH"),
                        item.get("priority_score", 0),
                    ),
                )
        conn.execute(
            "INSERT INTO audit_logs(action, entity_type, details) VALUES (%s,%s,%s)",
            (f"{table}_generated", table, Jsonb({"count": len(values)})),
        )
    conn.close()
    return len(values)


def persist_simulation(result: dict[str, Any]) -> int:
    """Persist what-if alternatives and their predicted savings, when configured."""
    conn = _connect()
    if conn is None:
        return 0

    from psycopg.types.json import Jsonb

    inserted = 0
    with conn:
        row = conn.execute(
            "SELECT id FROM cloud_resources "
            "WHERE provider=%s AND resource_id=%s LIMIT 1",
            (result["cloud_provider"], result["resource_id"]),
        ).fetchone()
        if not row:
            conn.close()
            return 0

        resource_fk = row[0]
        for option in result.get("alternatives", []):
            simulation = conn.execute(
                """INSERT INTO optimization_simulations
                   (resource_fk, action, before_cost, predicted_cost, risk, confidence, assumptions)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)
                   RETURNING id""",
                (
                    resource_fk,
                    option["action"],
                    option["current_monthly_cost"],
                    option["proposed_monthly_cost"],
                    option["risk"],
                    option["evidence_confidence"],
                    Jsonb({"assumptions": option["assumptions"], "option": option}),
                ),
            ).fetchone()
            if simulation is None:
                raise RuntimeError("Failed to retrieve inserted simulation ID")

            conn.execute(
                """INSERT INTO savings
                   (simulation_fk, realized_month, predicted_amount, realized_amount, currency)
                   VALUES (%s, date_trunc('month', CURRENT_DATE)::date, %s, NULL, %s)""",
                (simulation[0], option["monthly_saving"], option["currency"]),
            )
            inserted += 1

        conn.execute(
            """INSERT INTO audit_logs(action, entity_type, entity_id, details)
               VALUES ('what_if_simulated', 'resource', %s, %s)""",
            (result["resource_id"], Jsonb({"alternatives": inserted})),
        )
    conn.close()
    return inserted


def persist_prediction(result: dict[str, Any]) -> int:
    """Persist forecast predictions and metrics when PostgreSQL is configured."""
    conn = _connect()
    if conn is None:
        return 0

    from psycopg.types.json import Jsonb

    series = [row for row in result.get("series", []) if row.get("predicted") is not None]
    with conn:
        for row in series:
            conn.execute(
                """INSERT INTO model_predictions
                   (model_name, target_date, predicted_cost, actual_cost, metrics)
                   VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT(model_name, target_date) DO UPDATE SET
                    predicted_cost=EXCLUDED.predicted_cost,
                    actual_cost=EXCLUDED.actual_cost,
                    metrics=EXCLUDED.metrics,
                    created_at=now()""",
                (
                    result.get("model", "unknown"),
                    row["date"],
                    row["predicted"],
                    row.get("actual"),
                    Jsonb(result.get("metrics") or {}),
                ),
            )
        conn.execute(
            "INSERT INTO audit_logs(action, entity_type, details) "
            "VALUES ('forecast_generated', 'model', %s)",
            (Jsonb({"model": result.get("model"), "predictions": len(series)}),),
        )
    conn.close()
    return len(series)
