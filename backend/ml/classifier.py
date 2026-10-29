"""
Machine Learning Classifiers for Incident Category and Severity Prediction.
Uses Scikit-Learn TF-IDF N-grams with Calibrated Logistic Regression + Domain Heuristic Blending.
"""
import os
import re
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from backend.ml.preprocessor import clean_text, extract_domain_features

CATEGORY_CLASSES = ["safety", "environment", "esg_compliance", "operational_hazard"]
SEVERITY_CLASSES = ["low", "medium", "high", "critical"]

SEVERITY_WEIGHTS = {
    "low": 2.5,
    "medium": 5.0,
    "high": 8.0,
    "critical": 10.0
}

class IncidentClassifierPipeline:
    def __init__(self):
        self.category_pipeline: Optional[Pipeline] = None
        self.severity_pipeline: Optional[Pipeline] = None
        self.is_trained: bool = False
        self.training_sample_count: int = 0
        self.classes_category: List[str] = CATEGORY_CLASSES
        self.classes_severity: List[str] = SEVERITY_CLASSES

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train both category and severity models on the provided DataFrame.
        Expected columns: 'narrative', 'target_category', 'target_severity'.
        """
        if "narrative" not in df.columns or "target_category" not in df.columns or "target_severity" not in df.columns:
            raise ValueError("DataFrame must contain 'narrative', 'target_category', and 'target_severity'")

        texts = [clean_text(str(t)) for t in df["narrative"].tolist()]
        cat_labels = [str(c).strip().lower() for c in df["target_category"].tolist()]
        sev_labels = [str(s).strip().lower() for s in df["target_severity"].tolist()]

        # 1. Category Classifier Pipeline
        # Sublinear TF scaling & unigram+bigram captures compound terms like "arc flash", "oil spill", "cable trench"
        cat_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        cat_base_clf = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced")
        self.category_pipeline = Pipeline([
            ("tfidf", cat_vec),
            ("clf", cat_base_clf)
        ])
        self.category_pipeline.fit(texts, cat_labels)
        self.classes_category = list(self.category_pipeline.classes_)

        # 2. Severity Classifier Pipeline
        sev_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        sev_base_clf = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced")
        self.severity_pipeline = Pipeline([
            ("tfidf", sev_vec),
            ("clf", sev_base_clf)
        ])
        self.severity_pipeline.fit(texts, sev_labels)
        self.classes_severity = list(self.severity_pipeline.classes_)

        self.is_trained = True
        self.training_sample_count = len(texts)

        return {
            "status": "trained",
            "samples": self.training_sample_count,
            "category_classes": self.classes_category,
            "severity_classes": self.classes_severity
        }

    def predict(self, narrative: str) -> Dict[str, Any]:
        """
        Perform hybrid prediction (ML probabilities + domain feature heuristic blend).
        """
        if not self.is_trained:
            raise RuntimeError("Classifier is not trained yet. Call train() first.")

        cleaned = clean_text(narrative)
        domain_feats = extract_domain_features(narrative)

        # Raw ML Probabilities
        cat_probs_raw = self.category_pipeline.predict_proba([cleaned])[0]
        cat_probs = {cls: float(prob) for cls, prob in zip(self.classes_category, cat_probs_raw)}

        sev_probs_raw = self.severity_pipeline.predict_proba([cleaned])[0]
        sev_probs = {cls: float(prob) for cls, prob in zip(self.classes_severity, sev_probs_raw)}

        # Domain Rule Adjustments (Priors)
        # Check strong signals to refine probabilities for out-of-distribution or short text
        adjusted_cat = dict(cat_probs)
        if any(w in cleaned for w in ["oil", "spill", "leak", "seepage", "effluent", "drain", "groundwater", "biodiversity"]):
            adjusted_cat["environment"] = adjusted_cat.get("environment", 0.0) + 0.35
        if any(w in cleaned for w in ["arc flash", "high voltage", "switchgear", "gis", "scada", "bess temperature", "overheating", "excavator damaged"]):
            adjusted_cat["operational_hazard"] = adjusted_cat.get("operational_hazard", 0.0) + 0.35
        if any(w in cleaned for w in ["whistleblower", "audit", "grievance", "bribery", "segregation deviation", "migrant worker", "slavery"]):
            adjusted_cat["esg_compliance"] = adjusted_cat.get("esg_compliance", 0.0) + 0.35
        if any(w in cleaned for w in ["slip", "trip", "fall", "ppe", "cut", "laceration", "burn", "heat exhaustion", "height"]):
            adjusted_cat["safety"] = adjusted_cat.get("safety", 0.0) + 0.30

        # Normalize adjusted category
        cat_sum = sum(adjusted_cat.values())
        if cat_sum > 0:
            adjusted_cat = {k: round(v / cat_sum, 4) for k, v in adjusted_cat.items()}
        best_category = max(adjusted_cat.items(), key=lambda x: x[1])[0]
        cat_confidence = adjusted_cat[best_category]

        # Severity Domain Adjustments
        adjusted_sev = dict(sev_probs)
        crit_score = domain_feats["criticality_heuristic_score"]
        if crit_score >= 0.9:
            adjusted_sev["critical"] = adjusted_sev.get("critical", 0.0) + 0.50
        elif crit_score >= 0.7:
            adjusted_sev["high"] = adjusted_sev.get("high", 0.0) + 0.40
        elif crit_score <= 0.25:
            adjusted_sev["low"] = adjusted_sev.get("low", 0.0) + 0.35

        # Normalize adjusted severity
        sev_sum = sum(adjusted_sev.values())
        if sev_sum > 0:
            adjusted_sev = {k: round(v / sev_sum, 4) for k, v in adjusted_sev.items()}
        best_severity = max(adjusted_sev.items(), key=lambda x: x[1])[0]
        sev_confidence = adjusted_sev[best_severity]

        # Compute Composite Energy Risk Index (Scale 1.0 to 10.0)
        base_sev_score = SEVERITY_WEIGHTS.get(best_severity, 5.0)
        # Factor in high voltage or critical flags
        risk_multiplier = 1.0
        if domain_feats["has_high_voltage"]:
            risk_multiplier += 0.15
        if "critical" in best_severity:
            risk_multiplier += 0.10
        composite_risk_score = round(min(10.0, max(1.0, base_sev_score * risk_multiplier)), 1)

        return {
            "narrative": narrative,
            "category": best_category,
            "category_confidence": round(float(cat_confidence), 3),
            "category_probabilities": adjusted_cat,
            "severity": best_severity,
            "severity_confidence": round(float(sev_confidence), 3),
            "severity_probabilities": adjusted_sev,
            "risk_score": composite_risk_score,
            "domain_features": domain_feats
        }
