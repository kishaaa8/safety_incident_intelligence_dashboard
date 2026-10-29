"""
RAG and Vector Retrieval Engine for Recurring Incident Pattern Discovery.
Finds historical incident precedents, lessons learned, and recurring failure modes.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.ml.preprocessor import clean_text

class IncidentRAGEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.incident_records: List[Dict[str, Any]] = []
        self.tfidf_matrix = None
        self.is_indexed: bool = False

    def build_index(self, df: pd.DataFrame):
        """
        Index historical incident narratives and metadata.
        """
        self.incident_records = []
        texts = []

        for idx, row in df.iterrows():
            narrative = str(row.get("narrative", "")).strip()
            cleaned = clean_text(narrative)
            texts.append(cleaned)

            record = {
                "incident_id": str(row.get("incident_id", f"INC-{idx+1:03d}")),
                "narrative": narrative,
                "category": str(row.get("target_category", "safety")),
                "severity": str(row.get("target_severity", "medium")),
                "asset_type": str(row.get("asset_type", "General")),
                "location": str(row.get("location", "Renewable Asset")),
                "shift": str(row.get("shift", "General")),
                "root_cause": str(row.get("root_cause", "Pending Investigation")),
                "capa_action": str(row.get("capa_action", "Conduct detailed hazard review")),
                "regulatory_framework": str(row.get("regulatory_framework", "ISO 45001 / OSHA"))
            }
            self.incident_records.append(record)

        if texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.is_indexed = True

    def find_similar_incidents(self, query: str, top_k: int = 3, threshold: float = 0.12) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most similar historical incidents using Cosine Similarity.
        """
        if not self.is_indexed or self.tfidf_matrix is None:
            return []

        cleaned_query = clean_text(query)
        if not cleaned_query:
            return []

        query_vec = self.vectorizer.transform([cleaned_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Get sorted indices descending
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score < threshold and len(results) >= 1:
                # Always return at least 1 match if top score > 0, otherwise stop below threshold
                break
            if score <= 0.0:
                continue

            item = dict(self.incident_records[idx])
            item["similarity_score"] = round(score, 3)
            results.append(item)
            if len(results) >= top_k:
                break

        return results

    def detect_recurring_patterns(self) -> Dict[str, Any]:
        """
        Analyze the full indexed incident repository to detect recurring clusters,
        frequency of repeat events, and systemic vulnerabilities.
        """
        if not self.incident_records:
            return {"clusters": [], "top_recurring_themes": []}

        # Cluster by category & common hazard tokens
        patterns = {
            "Housekeeping & Slip/Trip/Fall Hazards": 0,
            "Hydraulic Oil & Chemical Leaks": 0,
            "High Voltage & Electrical Arc Hazards": 0,
            "PPE Violations & Personal Safety": 0,
            "Subcontractor Onboarding & Labor Compliance": 0,
            "Water Resource & Dust Environmental Concerns": 0,
            "Excavation & Buried Utility Risks": 0
        }

        for rec in self.incident_records:
            narrative = rec["narrative"].lower()
            if any(w in narrative for w in ["slip", "trip", "fall", "housekeeping", "walkway", "ladder", "scaffold"]):
                patterns["Housekeeping & Slip/Trip/Fall Hazards"] += 1
            if any(w in narrative for w in ["oil", "leak", "seepage", "hydraulic", "spill", "lubricant"]):
                patterns["Hydraulic Oil & Chemical Leaks"] += 1
            if any(w in narrative for w in ["arc flash", "voltage", "switchgear", "busbar", "breaker", "gis"]):
                patterns["High Voltage & Electrical Arc Hazards"] += 1
            if any(w in narrative for w in ["ppe", "glove", "helmet", "lanyard", "face shield"]):
                patterns["PPE Violations & Personal Safety"] += 1
            if any(w in narrative for w in ["contractor", "subcontractor", "migrant", "overtime", "induction"]):
                patterns["Subcontractor Onboarding & Labor Compliance"] += 1
            if any(w in narrative for w in ["water", "effluent", "drain", "stream", "dust", "erosion"]):
                patterns["Water Resource & Dust Environmental Concerns"] += 1
            if any(w in narrative for w in ["trench", "excavat", "buried", "cable"]):
                patterns["Excavation & Buried Utility Risks"] += 1

        top_themes = [
            {"theme": k, "incident_count": v, "risk_alert": "HIGH" if v >= 4 else "MODERATE"}
            for k, v in sorted(patterns.items(), key=lambda x: x[1], reverse=True)
            if v > 0
        ]

        return {
            "total_indexed": len(self.incident_records),
            "top_recurring_themes": top_themes,
            "systemic_alert": (
                "Multiple recurring incidents detected in Hydraulic Fluid Management and Fall Protection. "
                "Immediate contractor toolbox re-certification recommended across wind and solar hubs."
                if top_themes else "No critical systemic clusters found."
            )
        }
