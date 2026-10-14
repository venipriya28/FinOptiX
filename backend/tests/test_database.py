"""PostgreSQL round-trip integration test; enabled with FINOPTIX_DATABASE_URL."""
import os
import uuid

import pandas as pd
import pytest

from app import repository
from app.core import clean_usage
from app.simulator import simulate_options


@pytest.mark.skipif(not os.getenv("FINOPTIX_DATABASE_URL"), reason="Set FINOPTIX_DATABASE_URL to run PostgreSQL integration test")
def test_postgres_schema_and_observation_round_trip():
    repository.initialize()
    resource_id=f"DB-TEST-{uuid.uuid4()}"
    observations=[]
    for day in range(40):
        observations.append({"timestamp":(pd.Timestamp('2026-08-01',tz='UTC')+pd.Timedelta(days=day)).isoformat(),
            "cloud_provider":"AWS","service":"EC2","region":"ap-south-1","resource_id":resource_id,
            "resource_type":"virtual_machine","instance_type":"m5.2xlarge","cpu_utilization":10,"memory_utilization":20,
            "network_usage":2,"storage_usage":10,"allocated_storage_gb":1000,"running_hours":24,
            "monthly_cost":25000,"department":"test","project":"integration","environment":"development"})
    frame=clean_usage(pd.DataFrame(observations))
    assert repository.persist_frame(frame, source="integration_test") == 40
    loaded=repository.load_frame()
    assert loaded[loaded.resource_id==resource_id].shape[0]==40
    recommendation={"cloud_provider":"AWS","resource_id":resource_id,"action":"Downsize test","reason":"unit test",
        "evidence":{"observations":40},"estimated_monthly_saving":100,"confidence":.8,"risk":"LOW","priority_score":10}
    anomaly={"cloud_provider":"AWS","resource_id":resource_id,"timestamp":"2026-09-09T00:00:00+00:00","severity":"MEDIUM",
        "cost_impact_vs_baseline":25,"cause":"integration test","evidence":{"cost_change_pct":20}}
    assert repository.persist_json("recommendations",[recommendation],frame)==1
    assert repository.persist_json("anomalies",[anomaly],frame)==1
    simulation=simulate_options(frame,resource_id,cloud_provider="AWS")
    assert repository.persist_simulation(simulation)>0
    prediction={"model":"integration","series":[{"date":"2026-10-01","predicted":100,"actual":105}],"metrics":{"mae":5}}
    assert repository.persist_prediction(prediction)==1
    with __import__('psycopg').connect(os.environ["FINOPTIX_DATABASE_URL"]) as conn:
        assert conn.execute("SELECT count(*) FROM cloud_costs c JOIN cloud_resources r ON r.id=c.resource_fk WHERE r.resource_id=%s",(resource_id,)).fetchone()[0]==40
        assert conn.execute("SELECT count(*) FROM recommendations rec JOIN cloud_resources r ON r.id=rec.resource_fk WHERE r.resource_id=%s",(resource_id,)).fetchone()[0]==1
        assert conn.execute("SELECT count(*) FROM anomalies a JOIN cloud_resources r ON r.id=a.resource_fk WHERE r.resource_id=%s",(resource_id,)).fetchone()[0]==1
        assert conn.execute("SELECT count(*) FROM optimization_simulations sim JOIN cloud_resources r ON r.id=sim.resource_fk WHERE r.resource_id=%s",(resource_id,)).fetchone()[0]>0
        assert conn.execute("SELECT count(*) FROM model_predictions WHERE model_name='integration'").fetchone()[0]==1
        assert conn.execute("SELECT count(*) FROM audit_logs WHERE entity_type IN ('dataset','anomalies','recommendations','model')").fetchone()[0]>=5
    replacement=frame.copy()
    replacement["resource_id"]="DB-REPLACEMENT-"+str(uuid.uuid4())
    replacement["timestamp"]=(pd.to_datetime(replacement.timestamp,utc=True)+pd.Timedelta(days=100)).astype(str)
    assert repository.persist_frame(replacement,source="csv_upload",replace_existing=True)==40
    switched=repository.load_frame()
    assert switched.resource_id.nunique()==1
    assert switched.resource_id.iloc[0].startswith("DB-REPLACEMENT-")
    with __import__('psycopg').connect(os.environ["FINOPTIX_DATABASE_URL"]) as conn:
        assert conn.execute("SELECT count(*) FROM recommendations").fetchone()[0]==0
        assert conn.execute("SELECT count(*) FROM anomalies").fetchone()[0]==0
        assert conn.execute("SELECT count(*) FROM optimization_simulations").fetchone()[0]==0
        assert conn.execute("SELECT count(*) FROM model_predictions").fetchone()[0]==0
