from langgraph.graph import StateGraph, END
from typing import Dict, List, TypedDict
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from config import settings
from models import SymptomInput, DiseasePrediction, DiagnosisResponse
from IPython.display import Image, display 
import os  


# Initialize LLM
llm = ChatGroq(
    temperature=0.7,
    model_name=settings.llm_model,
    api_key=settings.groq_api_key
)

# 1. Custom Document Creation
def create_medical_documents(csv_path: str) -> List[Document]:
    """Convert each row into a document with symptoms + prognosis"""
    df = pd.read_csv(csv_path)
    documents = []
    
    for _, row in df.iterrows():
        symptoms = [col for col in df.columns[:-1] if row[col] == 1]
        prognosis = row['prognosis']
        
        content = f"""
        Patient presents with: {', '.join(symptoms)}.
        Most likely diagnosis: {prognosis}.
        """
        
        metadata = {
            "symptoms": symptoms,
            "prognosis": prognosis,
            "num_symptoms": len(symptoms)
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    return documents

# 2. Vector Store Setup
def build_medical_retriever(csv_path: str):
    documents = create_medical_documents(csv_path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=settings.embedding_model
    )
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Initialize retriever
medical_retriever = build_medical_retriever("D:\\MAYNOOTH\\SEM 2\\SUMMER PROJECT\\DATASET\\testing.csv")

# 3. Agent Definitions
class AgentState(TypedDict):
    input: str
    structured_symptoms: Dict[str, bool]
    retrieved_diseases: List[Dict]
    predictions: List[DiseasePrediction]
    report: str

# Symptom Extraction Agent
def extract_symptoms(state: AgentState):
    if state.get("structured_symptoms"):
        return state
    
    prompt = ChatPromptTemplate.from_template("""
    Extract medical symptoms from the following patient description.
    Return ONLY a comma-separated list of symptom terms. Be precise and clinical.
    
    Input: {input}
    Symptoms:""")
    
    chain = prompt | llm | StrOutputParser()
    symptoms = chain.invoke({"input": state["input"]})
    symptoms_list = [s.strip().lower() for s in symptoms.split(",")]
    
    # Create structured format matching our dataset
    structured = {symptom: True for symptom in symptoms_list}
    return {"structured_symptoms": structured}

# Disease Retrieval Agent
def retrieve_diseases(state: AgentState):
    # Convert structured symptoms to a description for vector search
    active_symptoms = [k for k, v in state["structured_symptoms"].items() if v]
    symptom_text = ", ".join(active_symptoms)
    
    # Retrieve similar cases
    docs = medical_retriever.invoke(symptom_text)
    
    # Process results
    diseases = []
    for doc in docs:
        diseases.append({
            "disease": doc.metadata["prognosis"],
            "symptoms": doc.metadata["symptoms"],
            "score": doc.metadata.get("score", 1.0),
            "content": doc.page_content
        })
    
    return {"retrieved_diseases": diseases}

# Explanation Agent
def generate_explanations(state: AgentState):
    predictions = []
    
    for disease in state["retrieved_diseases"]:
        prompt = ChatPromptTemplate.from_template("""
        Explain the potential diagnosis of {disease} given these symptoms: {symptoms}.
        Provide:
        1. A clinical explanation connecting symptoms to disease
        2. Typical treatment approaches
        3. Confidence level (0-100%) based on symptom match
        
        Write in clear, patient-friendly language.
        """)
        
        chain = prompt | llm | StrOutputParser()
        explanation = chain.invoke({
            "disease": disease["disease"],
            "symptoms": ", ".join(disease["symptoms"])
        })
        
        # Simple confidence calculation (could be enhanced)
        matched_symptoms = set(disease["symptoms"]) & set(state["structured_symptoms"].keys())
        confidence = min(100, len(matched_symptoms) / len(disease["symptoms"]) * 100)
        
        predictions.append(DiseasePrediction(
            disease=disease["disease"],
            confidence=round(confidence, 1),
            symptoms_matched=list(matched_symptoms),
            explanation=explanation,
            follow_up_questions=[]
        ))
    
    return {"predictions": predictions}

# Confidence & Follow-up Agent
def generate_followups(state: AgentState):
    if not state["predictions"]:
        return state
    
    # Get top prediction
    top_pred = max(state["predictions"], key=lambda x: x.confidence)
    
    prompt = ChatPromptTemplate.from_template("""
    Given the primary diagnosis of {disease} (confidence: {confidence}%),
    suggest 3-5 follow-up questions to clarify or confirm this diagnosis.
    
    Current symptoms: {symptoms}
    
    Return each question on a new line.
    """)
    
    chain = prompt | llm | StrOutputParser()
    questions = chain.invoke({
        "disease": top_pred.disease,
        "confidence": top_pred.confidence,
        "symptoms": ", ".join(state["structured_symptoms"].keys())
    }).split("\n")
    
    # Update predictions with follow-ups
    updated_preds = state["predictions"]
    updated_preds[0].follow_up_questions = [q.strip() for q in questions if q.strip()]
    
    return {"predictions": updated_preds}

# Report Generation Agent
def generate_report(state: AgentState):
    if not state["predictions"]:
        return {"report": "No diagnosis could be determined from the provided symptoms."}
    
    prompt = ChatPromptTemplate.from_template("""
    Create a patient-friendly medical report based on these findings:
    
    {predictions}
    
    Include:
    1. Summary of likely conditions
    2. Key symptoms supporting each
    3. Recommended next steps
    4. When to seek immediate care
    
    Use simple language and bullet points where appropriate.
    """)
    
    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({
        "predictions": "\n\n".join(
            f"Diagnosis: {p.disease}\nConfidence: {p.confidence}%\n{p.explanation}"
            for p in state["predictions"]
        )
    })
    
    return {"report": report}

# Build the workflow
workflow = StateGraph(AgentState)

# Define nodes
workflow.add_node("extract_symptoms", extract_symptoms)
workflow.add_node("retrieve_diseases", retrieve_diseases)
workflow.add_node("generate_explanations", generate_explanations)
workflow.add_node("generate_followups", generate_followups)
workflow.add_node("generate_report", generate_report)

# Define edges
workflow.set_entry_point("extract_symptoms")
workflow.add_edge("extract_symptoms", "retrieve_diseases")
workflow.add_edge("retrieve_diseases", "generate_explanations")
workflow.add_edge("generate_explanations", "generate_followups")
workflow.add_edge("generate_followups", "generate_report")
workflow.add_edge("generate_report", END)

# Compile the graph
diagnosis_chain = workflow.compile()


