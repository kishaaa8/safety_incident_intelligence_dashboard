"""
FastAPI Backend for ESG & Safety Incident Intelligence Platform.
Serves ML classification, RAG retrieval, CAPA reasoning, analytics, and static frontend.
"""
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Query as FastAPIQuery
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.data_manager import pipeline_manager

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="ESG & Safety Incident Intelligence Platform",
    description="Greenko Renewable Asset AI/ML Incident Triage, Severity Tagging, Recurring Pattern RAG & CAPA Generation",
    version="2.0.0"
)

# Enable CORS for local cross-origin development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class Query(BaseModel):
    text: str = Field("", description="Incident narrative text")

class IncidentSubmission(BaseModel):
    narrative: str = Field(..., min_length=5, description="Full description of the safety or ESG incident")
    asset_type: Optional[str] = Field("Solar", description="Asset type: Solar, Wind, Hydro, Storage, Substation")
    location: Optional[str] = Field("Greenko Site", description="Plant or facility location")
    shift: Optional[str] = Field("General", description="Shift: Morning, Day, Afternoon, Night")

class BatchAnalyzeRequest(BaseModel):
    narratives: List[str] = Field(..., min_length=1, description="List of incident narratives to analyze")

class DatasetSwitchRequest(BaseModel):
    dataset_name: str = Field(..., description="'dummy' for bundled 6 rows or 'expanded' for 60 rows")

# --- Original Starter Endpoints (Preserved for Evaluation Contract) ---

@app.get("/api/demo")
def demo():
    return {
        "case_study": "ESG / Safety Incident Intelligence Dashboard",
        "status": "Production Pipeline Active",
        "expected_outcome": "A dashboard that classifies incident type, severity, recurring patterns, and recommended preventive actions using NLP and summarization logic.",
        "active_dataset": pipeline_manager.active_dataset_name,
        "indexed_records": len(pipeline_manager.current_df) if pipeline_manager.current_df is not None else 0
    }

@app.post("/api/run")
def run_case(query: Query):
    text = query.text.strip()
    if not text:
        text = "Contractor slipped near wet walkway at solar block, minor injury, housekeeping issue"
    analysis = pipeline_manager.analyze_incident(text)
    return {
        "input": text,
        "result": analysis["reasoning"]["executive_summary"],
        "classification": analysis["prediction"],
        "similar_incidents": analysis["similar_incidents"],
        "reasoning": analysis["reasoning"],
        "evaluation_hint": "Trained with Stratified Cross-Validation on renewable energy operational narratives."
    }

# --- Core Intelligence Endpoints ---

@app.post("/api/analyze")
def analyze_incident(query: Query):
    """
    Triage incident: predicts category and severity, calculates composite risk score,
    retrieves top similar historical precedents via RAG, and produces root cause & CAPA.
    """
    text = query.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Narrative text cannot be empty.")
    try:
        result = pipeline_manager.analyze_incident(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")

@app.post("/api/analyze-batch")
def analyze_batch(req: BatchAnalyzeRequest):
    """Analyze multiple incident narratives in a single batch."""
    results = []
    for item in req.narratives:
        text = str(item).strip()
        if text:
            res = pipeline_manager.analyze_incident(text)
            results.append({
                "narrative": text,
                "category": res["prediction"]["category"],
                "severity": res["prediction"]["severity"],
                "risk_score": res["prediction"]["risk_score"],
                "executive_summary": res["reasoning"]["executive_summary"]
            })
    return {"total_analyzed": len(results), "results": results}

@app.get("/api/incidents")
def get_incidents(
    category: Optional[str] = FastAPIQuery(None, description="Filter by category (e.g. safety, environment, esg_compliance, operational_hazard)"),
    severity: Optional[str] = FastAPIQuery(None, description="Filter by severity (low, medium, high, critical)"),
    search: Optional[str] = FastAPIQuery(None, description="Keyword search in narratives")
):
    """Retrieve indexed historical incident records with optional multi-attribute filters."""
    incidents = pipeline_manager.get_all_incidents(category=category, severity=severity, search=search)
    return {"count": len(incidents), "incidents": incidents}

@app.post("/api/incidents")
def add_incident(incident: IncidentSubmission):
    """Log a new field incident, run triage, and index it into the active knowledge base."""
    triage = pipeline_manager.analyze_incident(incident.narrative)
    pred = triage["prediction"]
    reasoning = triage["reasoning"]

    new_id = f"INC-{len(pipeline_manager.current_df) + 1:03d}" if pipeline_manager.current_df is not None else "INC-NEW"
    new_record = {
        "incident_id": new_id,
        "narrative": incident.narrative,
        "target_category": pred["category"],
        "target_severity": pred["severity"],
        "asset_type": incident.asset_type or "Renewable",
        "location": incident.location or "Site",
        "shift": incident.shift or "Day",
        "root_cause": reasoning["root_cause_hypothesis"],
        "capa_action": reasoning["recommended_capa"],
        "regulatory_framework": reasoning["compliance_framework"]
    }

    # Append to current dataframe and update RAG index
    import pandas as pd
    new_row = pd.DataFrame([new_record])
    pipeline_manager.current_df = pd.concat([pipeline_manager.current_df, new_row], ignore_index=True)
    pipeline_manager.rag_engine.build_index(pipeline_manager.current_df)

    return {
        "status": "success",
        "message": f"Incident {new_id} successfully logged and indexed.",
        "record": new_record,
        "triage": triage
    }

@app.get("/api/similar")
def find_similar(
    query: str = FastAPIQuery(..., min_length=3, description="Incident narrative or hazard description to search"),
    top_k: int = FastAPIQuery(4, ge=1, le=10)
):
    """Semantic vector search for similar past incidents and historical resolutions."""
    matches = pipeline_manager.rag_engine.find_similar_incidents(query, top_k=top_k)
    return {
        "query": query,
        "match_count": len(matches),
        "results": matches
    }

@app.get("/api/analytics")
def get_analytics():
    """Aggregated portfolio analytics, severity metrics, and recurring pattern clusters."""
    return pipeline_manager.get_analytics()

@app.get("/api/evaluation")
def get_evaluation():
    """Retrieve stratified 5-fold cross-validation metrics, confusion matrix, and domain analysis."""
    if pipeline_manager.evaluation_cache:
        return pipeline_manager.evaluation_cache
    return {"error": "Evaluation metrics not ready."}

@app.post("/api/dataset/switch")
def switch_dataset(req: DatasetSwitchRequest):
    """Toggle between the minimum bundled dummy dataset (6 rows) and the enriched benchmark (60 rows)."""
    target = req.dataset_name.strip().lower()
    if target not in ["dummy", "expanded"]:
        raise HTTPException(status_code=400, detail="dataset_name must be 'dummy' or 'expanded'")
    info = pipeline_manager.load_dataset(target)
    return {
        "status": "success",
        "active_dataset": target,
        "info": info
    }

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "ESG & Safety Incident Intelligence Platform",
        "timestamp": time.time(),
        "active_dataset": pipeline_manager.active_dataset_name,
        "classifier_trained": pipeline_manager.classifier.is_trained,
        "rag_indexed": pipeline_manager.rag_engine.is_indexed
    }

# --- Serve Frontend Assets ---

# Serve static frontend files if directory exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "ESG Incident Intelligence Backend Running. Frontend index.html not found."}
