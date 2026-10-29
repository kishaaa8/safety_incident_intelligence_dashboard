"""
Data Manager and Pipeline Orchestrator.
Handles loading datasets (bundled dummy vs expanded renewable benchmark),
orchestrating model training, RAG vector indexing, and analytics aggregations.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from backend.ml.classifier import IncidentClassifierPipeline
from backend.ml.rag_engine import IncidentRAGEngine
from backend.ml.reasoning import IncidentReasoningEngine
from backend.ml.evaluator import run_rigorous_evaluation

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

class DataPipelineManager:
    def __init__(self):
        self.classifier = IncidentClassifierPipeline()
        self.rag_engine = IncidentRAGEngine()
        self.reasoning_engine = IncidentReasoningEngine()
        self.current_df: Optional[pd.DataFrame] = None
        self.active_dataset_name: str = "expanded"
        self.evaluation_cache: Optional[Dict[str, Any]] = None

    def initialize(self, dataset_name: str = "expanded"):
        """Initialize data, train models, index RAG, and cache evaluation."""
        self.load_dataset(dataset_name)

    def load_dataset(self, dataset_name: str = "expanded"):
        """Load specified dataset ('dummy' or 'expanded') and re-train pipelines."""
        if dataset_name == "dummy":
            csv_path = DATA_DIR / "safety_esg_incidents.csv"
            name = "Bundled Minimum Dummy Dataset (6 Incidents)"
        else:
            csv_path = DATA_DIR / "safety_esg_incidents_expanded.csv"
            if not csv_path.exists():
                csv_path = DATA_DIR / "safety_esg_incidents.csv"
                name = "Bundled Minimum Dummy Dataset (6 Incidents)"
            else:
                name = "Enriched Greenko Renewable Benchmark (60 Incidents)"

        df = pd.read_csv(csv_path)
        self.current_df = df
        self.active_dataset_name = dataset_name

        # Train classifier
        self.classifier.train(df)

        # Index RAG Engine
        self.rag_engine.build_index(df)

        # Precompute evaluation metrics
        try:
            self.evaluation_cache = run_rigorous_evaluation(df)
        except Exception as e:
            self.evaluation_cache = {"error": f"Evaluation calculation notice: {str(e)}"}

        return {
            "dataset_loaded": name,
            "record_count": len(df),
            "columns": list(df.columns)
        }

    def analyze_incident(self, narrative: str) -> Dict[str, Any]:
        """
        End-to-end incident intelligence pipeline:
        1. NLP classification (Category & Severity with calibrated probabilities)
        2. RAG vector retrieval (Top-k similar past incidents & lessons learned)
        3. Root cause hypothesis & CAPA generation
        4. Executive summary synthesis
        """
        # 1. Classify
        pred = self.classifier.predict(narrative)

        # 2. Retrieve Similar Precedents
        similar = self.rag_engine.find_similar_incidents(narrative, top_k=3)

        # 3. Generate Expert Reasoning
        reasoning = self.reasoning_engine.generate_expert_analysis(narrative, pred, similar)

        return {
            "prediction": pred,
            "similar_incidents": similar,
            "reasoning": reasoning
        }

    def get_all_incidents(self, category: Optional[str] = None, severity: Optional[str] = None, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return list of historical incidents with optional filtering."""
        if self.current_df is None:
            return []

        df = self.current_df.copy()
        if category and category.lower() != "all":
            df = df[df["target_category"].str.lower() == category.lower()]
        if severity and severity.lower() != "all":
            df = df[df["target_severity"].str.lower() == severity.lower()]
        if search:
            q = search.lower()
            df = df[df["narrative"].str.lower().str.contains(q, na=False)]

        records = []
        for idx, row in df.iterrows():
            records.append({
                "incident_id": str(row.get("incident_id", f"INC-{idx+1:03d}")),
                "narrative": str(row.get("narrative", "")),
                "category": str(row.get("target_category", "")),
                "severity": str(row.get("target_severity", "")),
                "asset_type": str(row.get("asset_type", "Solar/Wind")),
                "location": str(row.get("location", "Greenko Asset")),
                "shift": str(row.get("shift", "General")),
                "root_cause": str(row.get("root_cause", "")),
                "capa_action": str(row.get("capa_action", "")),
                "regulatory_framework": str(row.get("regulatory_framework", ""))
            })
        return records

    def get_analytics(self) -> Dict[str, Any]:
        """Aggregate high-level ESG and Safety metrics and distributions for executive dashboard."""
        if self.current_df is None:
            return {}

        df = self.current_df
        total = len(df)
        cat_counts = df["target_category"].value_counts().to_dict()
        sev_counts = df["target_severity"].value_counts().to_dict()

        # Asset distribution if present
        asset_counts = {}
        if "asset_type" in df.columns:
            asset_counts = df["asset_type"].value_counts().to_dict()

        high_critical_count = sev_counts.get("high", 0) + sev_counts.get("critical", 0)
        high_critical_pct = round((high_critical_count / total * 100), 1) if total > 0 else 0

        # Recurring patterns from RAG engine
        pattern_data = self.rag_engine.detect_recurring_patterns()

        return {
            "total_incidents": total,
            "active_dataset": self.active_dataset_name,
            "high_critical_incidents": high_critical_count,
            "high_critical_percentage": high_critical_pct,
            "esg_compliance_index": round(100 - (high_critical_pct * 0.4), 1),
            "category_distribution": cat_counts,
            "severity_distribution": sev_counts,
            "asset_distribution": asset_counts,
            "recurring_patterns": pattern_data
        }

# Global singleton pipeline manager instance
pipeline_manager = DataPipelineManager()
pipeline_manager.initialize("expanded")
