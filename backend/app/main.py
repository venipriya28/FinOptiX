from pathlib import Path
from typing import Literal, Optional
import os
import logging
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .core import clean_usage, cost_summary, recommendations, simulate, detect_anomalies, anomaly_evaluation, forecast, ranking_comparison
from .catalog import load_catalog
from . import repository
from . import knowledge

logger = logging.getLogger("finoptix")

DATA = Path(os.getenv("FINOPTIX_DATA", Path(__file__).resolve().parents[2] / "data" / "demo_usage.csv"))
UPLOAD_DATA = Path(os.getenv("FINOPTIX_UPLOAD_DATA", DATA.with_name("uploaded_usage.csv")))
app = FastAPI(title="FinOptiX API", version="0.1.0", description="Explainable, risk-aware cloud cost optimization demo")
_cors_origins = [origin.strip() for origin in os.getenv("FINOPTIX_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
_dev_origin_regex = os.getenv(
    "FINOPTIX_CORS_ORIGIN_REGEX",
    r"^https?://(?:localhost|127\.0\.0\.1):(?:517\d)$",
)
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_origin_regex=_dev_origin_regex,
                   allow_methods=["*"], allow_headers=["*"])
_df: pd.DataFrame | None = None
_database_initialized = False
_active_source = "not_loaded"
def data() -> pd.DataFrame:
    global _df, _database_initialized, _active_source
    if _df is None:
        if repository.database_url():
            if not _database_initialized:
                repository.initialize()
                _database_initialized = True
            stored = repository.load_frame()
            if stored is not None and not stored.empty:
                _df = clean_usage(stored)
                _active_source = "PostgreSQL"
                return _df
        source = UPLOAD_DATA if UPLOAD_DATA.exists() else DATA
        if not source.exists(): raise HTTPException(503, "Demo data missing. Generate it with python -m app.generate_data")
        _df = clean_usage(pd.read_csv(source))
        _active_source = "Uploaded CSV" if source == UPLOAD_DATA else "Demo CSV"
        if repository.database_url():
            repository.persist_frame(_df, "demo_seed")
    return _df

class SimulationRequest(BaseModel):
    resource_id: str = Field(min_length=1, max_length=200)
    cloud_provider: Optional[str] = Field(default=None, max_length=40)
    action: Literal["compare", "all", "instance_rightsizing", "scheduled_shutdown", "storage_optimization"] = "compare"
    target_instance: Optional[str] = Field(default=None, max_length=100)
class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["auto", "analytics", "knowledge"] = "auto"

@app.get("/")
def root():
    """Point developers to the dashboard and interactive API documentation."""
    return {"name": "FinOptiX API", "dashboard": "http://localhost:5173", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    try:
        current=data()
        return {"status":"ok", "dataset":str(DATA.name), "data_source":_active_source, "rows":len(current),
                "resources":int(current.resource_id.nunique()), "storage":"postgresql" if repository.database_url() else "csv"}
    except Exception as exc:
        logger.exception("FinOptiX readiness check failed")
        raise HTTPException(503, "FinOptiX data store is not ready") from exc
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global _df, _database_initialized, _active_source
    if not (file.filename or "").lower().endswith(".csv"): raise HTTPException(400,"Upload a CSV file")
    size=file.file.seek(0,2); file.file.seek(0)
    if size > 50 * 1024 * 1024: raise HTTPException(413,"CSV upload exceeds the 50 MB demo limit")
    try:
        loaded=clean_usage(pd.read_csv(file.file))
        if loaded.empty: raise ValueError("CSV contains no valid observations")
        if len(loaded)>250_000: raise ValueError("CSV exceeds the 250,000 row demo limit")
    except Exception as e:
        raise HTTPException(422,str(e)) from e
    if repository.database_url():
        try:
            if not _database_initialized:
                repository.initialize(); _database_initialized = True
            repository.persist_frame(loaded, "csv_upload", replace_existing=True)
        except Exception as e:
            logger.exception("Database import persistence failed")
            raise HTTPException(503,"Database persistence failed") from e
    try:
        UPLOAD_DATA.parent.mkdir(parents=True,exist_ok=True); loaded.to_csv(UPLOAD_DATA,index=False)
        _df=loaded
        _active_source="Uploaded CSV"
    except Exception as e:
        logger.exception("CSV upload snapshot could not be written")
        raise HTTPException(500,"Could not store the upload snapshot") from e
    return {"loaded_rows":len(_df),"resources":int(_df.resource_id.nunique())}
@app.get("/cost-summary")
def summary(): return cost_summary(data())
@app.get("/cost-trends")
def trends():
    d=data().groupby("day").daily_cost.sum().sort_index()
    return [{"date":str(k),"cost":round(float(v),2)} for k,v in d.items()]
@app.get("/forecast")
def get_forecast():
    result=forecast(data())
    if repository.database_url(): repository.persist_prediction(result)
    return result
@app.get("/anomalies")
def anomalies():
    df=data(); items=detect_anomalies(df)
    if repository.database_url(): repository.persist_json("anomalies", items, df)
    return items
@app.get("/anomaly-evaluation")
def anomaly_metrics(): return anomaly_evaluation(data()) or {"detail":"No ground-truth anomaly labels in this dataset"}
@app.get("/recommendations")
def get_recommendations():
    df=data(); items=recommendations(df)
    if repository.database_url(): repository.persist_json("recommendations", items, df)
    return items
@app.get("/savings")
def savings():
    recs=recommendations(data()); return {"currency":"INR","monthly_opportunity":round(sum(x["estimated_monthly_saving"] for x in recs),2),"annual_opportunity":round(sum(x["estimated_annual_saving"] for x in recs),2),"count":len(recs)}
@app.post("/what-if")
def what_if(request: SimulationRequest):
    try:
        result=simulate(data(),request.resource_id,request.action,request.target_instance,request.cloud_provider)
        if repository.database_url(): repository.persist_simulation(result)
        return result
    except KeyError:raise HTTPException(404,"Resource not found")
    except ValueError as e:raise HTTPException(422,str(e))
@app.get("/optimization-ranking")
def ranking(): return ranking_comparison(data())
@app.get("/pricing-catalog")
def pricing_catalog():
    catalog=load_catalog()
    return {"catalog_version":catalog["catalog_version"],"currency":catalog["currency"],"price_source":catalog["price_source"],
            "disclaimer":catalog["disclaimer"],"providers":catalog["providers"],"billing":catalog["billing"],"risk":catalog["risk"],
            "confidence":catalog["confidence"],"objective":catalog["objective"]}
@app.post("/chat")
def chat(request: ChatRequest):
    """Route analytics to validated calculations and general guidance to local retrieval."""
    q=request.question.lower()
    analytics_terms=("our ", "my ", "this month", "this dataset", "cloud cost increase", "cost increased", "how much can", "savings", "save", "resource", "anomaly", "forecast", "projected", "lowest risk", "performance constraint", "violat")
    mode=request.mode if request.mode != "auto" else ("analytics" if any(term in q for term in analytics_terms) else "knowledge")
    if mode == "knowledge":
        result=knowledge.retrieve(request.question)
        return {**result, "mode":"knowledge_retrieval", "grounding":f"Retrieved from local indexed source documents. {result.get('method','')}"}
    df=data(); summary=cost_summary(df); recs=recommendations(df); anomalies_=detect_anomalies(df)
    if any(word in q for word in ("increase", "increased", "anomal", "spike", "why")):
        item=next((x for x in anomalies_ if x.get("root_cause_analysis",{}).get("status")=="measured_comparison"),None)
        if item:
            rc=item["root_cause_analysis"]
            text=f"A detected {item['cloud_provider']} {item['service']} cost signal changed {item['evidence']['cost_change_pct']:+.1f}% against its prior resource baseline. Measured portfolio monthly run-rate changed by ₹{rc['spend_delta']:,.0f}; observed resource counts changed from {rc['resource_count_before']} to {rc['resource_count_recent']}. Largest provider/service/region contributions: " + "; ".join(f"{x['cloud_provider']} {x['service']} / {x['region']} ₹{x['spend_delta']:+,.0f}" for x in rc["service_region_contributions"][:3]) + ". The window association does not prove causation."
        else:
            text=f"No anomaly has a complete prior-30-day and recent-7-day comparison window. Current modeled monthly run-rate is ₹{summary['total_monthly_cost']:,.0f}; the dataset has no unit-price history."
    elif "project" in q or "forecast" in q or "next month" in q:
        prediction=forecast(df); text=f"The model projects ₹{prediction['next_month']:,.0f} over the next 30 days ({prediction['model']}). Chronological holdout MAE is {prediction.get('metrics',{}).get('mae','unavailable')}; this is a synthetic-data estimate."
    elif "lowest risk" in q or ("risk" in q and "which" in q):
        if recs:
            low=min(recs,key=lambda x:(x["risk_score"],-x["estimated_monthly_saving"]))
            text=f"Lowest heuristic-risk feasible recommendation is {low['resource_id']}: {low['action']} with risk score {low['risk_score']}/100 ({low['risk']}). Estimated saving ₹{low['estimated_monthly_saving']:,.0f}/month; evidence quality {low['confidence']:.0%}, not probability of success."
        else: text="No feasible recommendations with sufficient history are available."
    elif "violat" in q or "constraint" in q:
        from .simulator import simulate_options, SimulationError
        rejected=[]
        for provider,resource in df[["cloud_provider","resource_id"]].drop_duplicates().itertuples(index=False,name=None):
            try:
                result=simulate_options(df,resource,cloud_provider=str(provider))
                rejected.extend((resource,option) for option in result["rejected_options"] if option.get("performance_constraint_violations"))
            except (ValueError, KeyError, SimulationError):
                continue
        text=(f"{len(rejected)} proposed alternatives are rejected for performance constraints. " + "; ".join(f"{resource}: {'; '.join(item['performance_constraint_violations'])}" for resource,item in rejected[:5])) if rejected else "No catalog alternatives in this dataset currently violate the configured p95 constraints."
    elif any(word in q for word in ("save", "opportunit", "waste", "optim")):
        total=sum(r["estimated_monthly_saving"] for r in recs)
        text=f"There are {len(recs)} feasible rule-based opportunities totaling ₹{total:,.0f}/month on this dataset. " + (f"Highest risk-aware priority: {recs[0]['resource_id']} — {recs[0]['action']} (₹{recs[0]['estimated_monthly_saving']:,.0f}/month, {recs[0]['risk']} risk)." if recs else "No qualifying recommendation was found.")
    else:
        text=f"Current modeled monthly cloud expenditure is ₹{summary['total_monthly_cost']:,.0f} across {summary['resource_count']} resources. Ask about cost changes, forecast, savings, risks, or performance constraints."
    return {"answer":text,"mode":"application_analytics","grounding":"Computed by validated FinOptiX analytics functions against the currently loaded records.","sources":["cost_summary","recommendations","forecast","anomaly_evidence"]}

@app.get("/knowledge/sources")
def knowledge_sources(): return knowledge.sources()

@app.post("/knowledge/reindex")
def knowledge_reindex():
    knowledge._index = None
    return {"indexed_sources": knowledge.sources(), "count": len(knowledge.sources())}
