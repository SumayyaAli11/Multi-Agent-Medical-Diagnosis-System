from pydantic import BaseModel
from typing import List, Optional, Dict

class SymptomInput(BaseModel):
    text: str  # Natural language description
    structured: Optional[Dict[str, bool]] = None  # Optional structured input

class DiseasePrediction(BaseModel):
    disease: str
    confidence: float
    symptoms_matched: List[str]
    explanation: str
    follow_up_questions: List[str]

class DiagnosisResponse(BaseModel):
    predictions: List[DiseasePrediction]
    report: str