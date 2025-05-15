from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from agents import diagnosis_chain
from models import SymptomInput, DiagnosisResponse
import uvicorn

app = FastAPI(title="Medical Diagnosis Assistant API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(symptoms: SymptomInput):
    try:
        # Initialize state
        initial_state = {
            "input": symptoms.text,
            "structured_symptoms": symptoms.structured or {},
            "retrieved_diseases": [],
            "predictions": [],
            "report": ""
        }
        
        # Execute the workflow
        result = diagnosis_chain.invoke(initial_state)
        
        return DiagnosisResponse(
            predictions=result["predictions"],
            report=result["report"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)