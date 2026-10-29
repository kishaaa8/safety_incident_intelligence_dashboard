"""
ESG & Safety Incident NLP Preprocessing and Domain Feature Engineering.
Specialized for Greenko Renewable Energy Assets (Solar, Wind, Hydro, Storage, Substation).
"""
import re
from typing import Dict, List, Any, Tuple

# Domain keyword dictionaries mapped to energy sectors and hazard types
DOMAIN_KEYWORDS = {
    "solar": ["solar", "pv", "module", "string", "inverter", "tracker", "dc busbar", "combiner", "panel", "bhadla", "kurnool", "pavagada", "rewa", "charanka"],
    "wind": ["wind", "turbine", "nacelle", "gearbox", "blade", "yaw", "pitch", "tower", "anantapur", "jaisalmer", "dhar", "koppal"],
    "hydro": ["hydro", "penstock", "cavern", "dam", "intake", "desilting", "silt", "powerhouse", "surge tank", "saundatti", "pinnapuram"],
    "storage": ["bess", "battery", "storage", "cell", "rack", "hvac", "electrolyte", "thermal runaway"],
    "substation": ["substation", "switchyard", "switchgear", "gis", "transformer", "busbar", "breaker", "33kv", "400kv", "11kv", "sf6", "arc flash"],
    "ppe": ["ppe", "helmet", "gloves", "lanyard", "harness", "goggles", "face shield", "boots", "vest"],
    "fall_hazard": ["height", "ladder", "slip", "trip", "fall", "scaffolding", "grating", "walkway", "conduit", "hole"],
    "electrical": ["energized", "voltage", "arc flash", "shock", "loto", "busbar", "short circuit", "breaker", "cable"],
    "environmental": ["oil", "leak", "seepage", "spill", "water", "effluent", "drain", "stream", "herbicide", "chemical", "dust", "sf6", "wildlife", "bird"],
    "governance": ["audit", "overtime", "induction", "contractor", "minutes", "bribery", "whistleblower", "grievance", "compliance", "slavery", "labor"]
}

# Criticality trigger tokens
CRITICALITY_SIGNALS = {
    "critical": ["arc flash", "critical", "lost-time", "lti", "fatality", "death", "outrigger sank", "wildfire", "oil discharge stream", "uncontrolled groundwater", "confined space", "height without", "without lanyard"],
    "high": ["high voltage", "burst", "overheating", "spill", "cable damage", "heat exhaustion", "dehydration", "bess temperature", "42c", "scaffolding tag missing", "dropped handheld", "whistleblower", "unregistered migrant", "sf6 gas pressure drop"],
    "medium": ["slip", "mild burn", "detergent", "dust emissions", "valve leakage", "erosion", "silt runoff", "grievance", "push-button seized", "water usage spike", "laceration"],
    "low": ["housekeeping", "segregation", "minutes", "sds", "snake", "first aid", "bird nesting", "plastic packaging", "expired first aid", "training module"]
}

def clean_text(text: str) -> str:
    """Normalize narrative text for NLP feature extraction and tokenization."""
    if not text or not isinstance(text, str):
        return ""
    # Standardize casing & trim
    t = text.strip().lower()
    # Replace non-alphanumeric punctuation with spaces while keeping hyphens inside words
    t = re.sub(r'[^a-z0-9\s\-_]', ' ', t)
    # Collapse multiple whitespaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_domain_features(narrative: str) -> Dict[str, Any]:
    """
    Extract energy-specific domain cues, hazard entities, and regulatory indicators.
    """
    cleaned = clean_text(narrative)
    
    # Asset detection
    detected_assets = []
    for asset, kws in DOMAIN_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(kw)}\b", cleaned) for kw in kws):
            detected_assets.append(asset)
            
    # Primary asset deduction
    primary_asset = "Renewable Site"
    for cat in ["solar", "wind", "hydro", "storage", "substation"]:
        if cat in detected_assets:
            primary_asset = cat.capitalize()
            break
            
    # Criticality score heuristic (0.0 to 1.0)
    crit_score = 0.3 # baseline medium
    if any(sig in cleaned for sig in CRITICALITY_SIGNALS["critical"]):
        crit_score = 0.95
    elif any(sig in cleaned for sig in CRITICALITY_SIGNALS["high"]):
        crit_score = 0.75
    elif any(sig in cleaned for sig in CRITICALITY_SIGNALS["medium"]):
        crit_score = 0.50
    elif any(sig in cleaned for sig in CRITICALITY_SIGNALS["low"]):
        crit_score = 0.20
        
    # Hazard types
    hazards = []
    if "electrical" in detected_assets or any(k in cleaned for k in ["arc flash", "voltage", "loto", "shock", "busbar"]):
        hazards.append("Electrical / Arc Flash")
    if "fall_hazard" in detected_assets or any(k in cleaned for k in ["slip", "trip", "fall", "height", "ladder"]):
        hazards.append("Slip, Trip & Fall / Work at Height")
    if "environmental" in detected_assets or any(k in cleaned for k in ["oil", "leak", "spill", "water", "sf6", "effluent"]):
        hazards.append("Environmental Release / Spillage")
    if "ppe" in detected_assets or "ppe" in cleaned:
        hazards.append("PPE Violation / Procedural Non-compliance")
    if "governance" in detected_assets or any(k in cleaned for k in ["waste", "labor", "contractor", "audit", "overtime"]):
        hazards.append("ESG Governance & Labor Practices")
        
    if not hazards:
        hazards.append("General Operational Incident")

    # ESG Pillar attribution
    esg_pillar = "Social (Safety & People)"
    if "environmental" in detected_assets or any(k in cleaned for k in ["oil", "spill", "leak", "water", "biodiversity", "bird", "waste", "sf6", "detergent"]):
        esg_pillar = "Environmental (E)"
    elif "governance" in detected_assets or any(k in cleaned for k in ["bribery", "whistleblower", "audit", "minutes", "contract", "training module"]):
        esg_pillar = "Governance (G)"
    else:
        esg_pillar = "Social (Safety - S)"

    return {
        "cleaned_text": cleaned,
        "primary_asset": primary_asset,
        "detected_assets": detected_assets,
        "criticality_heuristic_score": crit_score,
        "identified_hazards": hazards,
        "esg_pillar": esg_pillar,
        "word_count": len(cleaned.split()),
        "has_high_voltage": "high voltage" in cleaned or "33kv" in cleaned or "400kv" in cleaned,
        "has_contractor_mention": "contractor" in cleaned or "subcontractor" in cleaned or "vendor" in cleaned
    }
