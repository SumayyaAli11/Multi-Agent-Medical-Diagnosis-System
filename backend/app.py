from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from agents import diagnosis_chain
from models import SymptomInput, DiagnosisResponse
import uvicorn
from evaluation import MedicalDiagnosisEvaluator
from fastapi.responses import FileResponse


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
    
    # Evaluation Endpoints
@app.post("/evaluation/run")
async def run_evaluation():
    evaluator = MedicalDiagnosisEvaluator("D:\\MAYNOOTH\\SEM 2\\SUMMER PROJECT\\DATASET\\new_test_set.csv")
    metrics = evaluator.evaluate()
    return {"metrics": metrics}

@app.get("/evaluation/confusion-matrix")
async def get_confusion_matrix():
    evaluator = MedicalDiagnosisEvaluator("D:\\MAYNOOTH\\SEM 2\\SUMMER PROJECT\\DATASET\\new_test_set.csv")
    cm_path = "D:\\MAYNOOTH\\SEM 2\\SUMMER PROJECT\\DATASET\\confusion_matrix.csv"
    evaluator.save_confusion_matrix(cm_path)
    return FileResponse(
        path=cm_path,
        media_type="text/csv",
        filename="confusion_matrix.csv"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)