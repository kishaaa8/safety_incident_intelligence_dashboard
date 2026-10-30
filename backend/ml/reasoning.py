"""
Root Cause, CAPA Recommendation, and Executive Summarization Reasoning Engine.
Combines deterministic Greenko domain knowledge-graph logic with optional Gemini/OpenAI GenAI synthesis.
"""
import os
import json
import requests
from typing import Dict, Any, List, Optional
from backend.ml.preprocessor import clean_text

# Energy domain regulatory and CAPA reference knowledge
CAPA_KNOWLEDGE_BASE = {
    "slip": {
        "root_cause": "Housekeeping deficiency, compromised drainage, or lack of slip-resistant walkway grating in high-traffic operational zones.",
        "containment": "Immediately barricade the slick section, place wet-floor warning signage, and mop/absorb standing water or moss.",
        "capa": "Install serrated anti-slip fiberglass grating, clear adjacent stormwater drains, and implement twice-daily 5S walkway sign-offs.",
        "regulation": "OSHA 1910.22 Walking-Working Surfaces & ISO 45001 Clause 8.1"
    },
    "oil": {
        "root_cause": "Hydraulic seal degradation, loose camlock coupling, or over-pressurization during turbine/transformer maintenance.",
        "containment": "Deploy emergency spill kit absorbent pads and polypropylene booms immediately to prevent runoff toward surface drainage.",
        "capa": "Replace hydraulic casing gasket kits with fluorocarbon seals, conduct scheduled ultrasonic torque audits, and transition to eco-friendly ester oils.",
        "regulation": "ISO 14001 Environmental Management & EPA Spill Prevention Control (SPCC)"
    },
    "ppe": {
        "root_cause": "Inadequate supervisory oversight, thermal discomfort/heat stress, or poor availability of job-specific certified PPE.",
        "containment": "Immediately halt the unsafe work task; issue compliant PPE from site stores before permitting resumption.",
        "capa": "Procure ergonomic lightweight ventilated PPE, institute mandatory daily safety toolbox talks, and implement behavioral safety observation cards.",
        "regulation": "ISO 45001 & OSHA 1910.132 General PPE Requirements"
    },
    "arc flash": {
        "root_cause": "Enclosure seal weather degradation causing moisture ingress, vermin entry, or racking circuit breakers under energized fault conditions.",
        "containment": "Electrically isolate upstream feeder, tag out switchgear cubicle, and test busbars for zero energy with certified high-voltage probe.",
        "capa": "Reseal cabinet flanges with RTV silicone, install internal space anti-condensation heaters, and mandate 40 cal/cm² arc-rated flash suits.",
        "regulation": "IEEE 1584 Arc Flash Calculations & NFPA 70E Electrical Workplace Safety"
    },
    "height": {
        "root_cause": "Technician fatigue, non-continuous fall arrest equipment, or lack of dual-lanyard 100% tie-off discipline.",
        "containment": "Instruct worker to cease work immediately, descend safely via certified ladder system, and ground technician for fitness check.",
        "capa": "Install permanent rigid vertical lifeline rail glider systems and mandate annual Global Wind Organisation (GWO) certified working at height refresher.",
        "regulation": "OSHA 1910.140 Personal Fall Protection & GWO Safety Standards"
    },
    "waste": {
        "root_cause": "Subcontractor lack of orientation on segregation bins, lack of designated hazardous scrap storage, or logistical dumping delays.",
        "containment": "Physically re-segregate co-mingled scrap, quarantine contaminated drums, and cover open skips with waterproof tarpaulins.",
        "capa": "Establish fenced and labeled e-waste and chemical storage yards, conduct bilingual subcontractor training, and tie disposal audit to contractor invoices.",
        "regulation": "GRI 306 Waste Management & National Hazardous Waste Rules"
    },
    "water": {
        "root_cause": "Automated valve solenoid failure, mechanical pipe cracking from soil shifting, or unchecked continuous flow during solar module cleaning.",
        "containment": "Manually close the isolation ball valve and switch solar module cleaning to dry microfiber brush mode.",
        "capa": "Install acoustic smart flow meters with SCADA leakage alarms, replace mechanical seals with high-cycle valves, and monitor water intensity per MWp.",
        "regulation": "GRI 303 Water and Effluents & Greenko Sustainable Water Policy"
    },
    "labor": {
        "root_cause": "Subcontractor labor broker bypassing gate verification, lack of digital timecards, or contractor under-resourcing leading to excessive hours.",
        "containment": "Suspend unauthorized contractor access, verify IDs against government databases, and enforce mandatory rest cycles.",
        "capa": "Implement biometric RFID gate access integrated with site EHS induction passports and conduct quarterly third-party social audits.",
        "regulation": "ILO Core Labor Standards & GRI 403 / 401 Employment Practices"
    }
}

class IncidentReasoningEngine:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

    def generate_expert_analysis(self, narrative: str, prediction: Dict[str, Any], similar_incidents: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Synthesize root cause hypothesis, CAPA recommendations, regulatory standards,
        and executive summary using deterministic domain intelligence and optional GenAI.
        """
        cleaned = clean_text(narrative)
        category = prediction.get("category", "safety")
        severity = prediction.get("severity", "medium")
        domain_feats = prediction.get("domain_features", {})
        asset = domain_feats.get("primary_asset", "Renewable Plant")

        # 1. Match best domain template
        matched_key = None
        for key in CAPA_KNOWLEDGE_BASE.keys():
            if key in cleaned:
                matched_key = key
                break

        # Fallback based on category
        if not matched_key:
            if category == "environment":
                matched_key = "oil"
            elif category == "esg_compliance":
                matched_key = "waste"
            elif category == "operational_hazard":
                matched_key = "arc flash"
            else:
                matched_key = "ppe"

        domain_info = CAPA_KNOWLEDGE_BASE[matched_key]

        root_cause = domain_info["root_cause"]
        containment = domain_info["containment"]
        capa = domain_info["capa"]
        regulation = domain_info["regulation"]

        # Incorporate historical lessons if retrieved
        historical_context = ""
        if similar_incidents and len(similar_incidents) > 0:
            top_precedent = similar_incidents[0]
            if top_precedent.get("similarity_score", 0) > 0.25:
                historical_context = (
                    f"Corroborated by historical precedent {top_precedent['incident_id']} at {top_precedent.get('location', 'asset')} "
                    f"where action was: '{top_precedent.get('capa_action', '')}'."
                )

        # 2. Executive Management Summary
        exec_summary = (
            f"[{severity.upper()} PRIORITY] {category.replace('_', ' ').title()} incident identified at {asset} facility. "
            f"Primary hazard involves {domain_feats.get('identified_hazards', ['operations'])[0]}. "
            f"Immediate action required: {containment} "
            f"Systemic remediation requires {capa} "
            f"{historical_context}".strip()
        )

        # 3. Optional GenAI Refinement if GEMINI_API_KEY is available
        llm_enhanced = False
        api_key = os.getenv("GEMINI_API_KEY", self.gemini_api_key)
        if api_key:
            try:
                enhanced_summary = self._call_gemini_api(narrative, category, severity, exec_summary, api_key)
                if enhanced_summary:
                    exec_summary = enhanced_summary
                    llm_enhanced = True
            except Exception as e:
                # Silently fallback to deterministic summary to guarantee reliability
                pass

        return {
            "root_cause_hypothesis": root_cause,
            "immediate_containment": containment,
            "recommended_capa": capa,
            "compliance_framework": regulation,
            "executive_summary": exec_summary,
            "llm_enhanced": llm_enhanced,
            "historical_precedent_note": historical_context if historical_context else "No direct recurring precedent detected above threshold."
        }

    def _call_gemini_api(self, narrative: str, category: str, severity: str, base_summary: str, api_key: Optional[str] = None) -> Optional[str]:
        """Call Google Gemini REST API endpoint for executive summary refinement."""
        key = api_key or os.getenv("GEMINI_API_KEY", self.gemini_api_key)
        if not key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
        prompt = (
            f"You are the Head of EHS and Operational Risk for Greenko Renewable Energy. "
            f"Review this incident narrative: '{narrative}'. "
            f"Category: {category}, Severity: {severity}. "
            f"Provide an ultra-crisp 2-3 sentence executive briefing including root cause, containment, and systemic CAPA. "
            f"Be professional, rigorous, and reference renewable asset safety standards."
        )
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 180}
        }
        res = requests.post(url, headers=headers, json=payload, timeout=4.0)
        if res.status_code == 200:
            data = res.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return None
