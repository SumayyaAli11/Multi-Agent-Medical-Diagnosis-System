# 🧠 Multi-Agent Medical Diagnosis Assistant

This project is the **MSc thesis work** , focused on building a multi-agent LLM-based system that takes patient symptoms (natural language or structured input) and retrieves **probable diseases along with medical explanations**. It leverages the power of **large language models**, **agent-based architecture**, and **workflow orchestration** to support intelligent and patient-friendly diagnosis.

---

## 🎯 Project Goal

To develop a modular, intelligent medical assistant that:
- Accepts symptom input in natural language.
- Uses specialized agents to extract structured features, retrieve potential diseases, and generate natural explanations.
- Provides confidence scoring, asks clarifying questions when needed, and outputs a patient-friendly report.

---

## 🧩 Agents & Workflow

The system is composed of multiple agents, each designed for a specific subtask, orchestrated via **LangGraph**:

### 1. 🧾 Symptom Extraction Agent
- Input: Natural language symptom descriptions (e.g., *"I have rash, fatigue, and nausea."*)
- Output: Structured binary symptom representation
- **Tech**: LLM (ChatGroq), semantic embeddings

---

### 2. 🧬 Disease Retrieval Agent
- Performs vector similarity search using **FAISS** on embedded symptom data.
- Returns top-K likely disease matches from the dataset.
- **Tech**: FAISS, custom medical embeddings, rule-based fallback logic

---

### 3. 🩺 Explanation Agent
- Adds context to each disease retrieved using trusted medical knowledge.
- Example: *"Eczema is commonly associated with dry, itchy skin..."*
- **Tech**: LLM prompt engineering or curated medical corpus

---

### 4. ❓ Confidence & Follow-up Agent
- Assesses ambiguity in symptom match.
- Asks clarifying follow-up questions to improve confidence.
- Example: *"Do you also have a fever?"*

---

### 5. 📄 Report Generation Agent
- Summarizes results in simple, human-readable language.
- May suggest next steps like *"You should consult a dermatologist."*

---

## 🗂 Project Structure

![image](https://github.com/user-attachments/assets/785c30a3-66da-4f27-ba00-f75baf7a1c3d)

---

## 🚀 How to Run

### 1. Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```
###2. Frontend (Streamlit)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Make sure both backend and frontend are running on separate terminals.

🔧 Tech Stack
Python, FastAPI, Streamlit

LangGraph for agent workflow orchestration

ChatGroq (LLaMA 3) for LLM-based agents

FAISS for vector similarity disease retrieval

Docker for containerization and portability

AWS EC2 for deployment

📊 Dataset
Custom medical dataset stored in data/training.csv downloaded from Kaggle:https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning 

Each row represents a disease and associated binary symptom indicators

🧪 Status
🛠 Currently in development
Key agents are being integrated and tested. Initial prototype with symptom extraction and disease retrieval pipeline is functional.

🤝 Contributions & Collaboration
This is a thesis project under development and not open for external contribution at this stage. If you're interested in collaboration or research-based inquiry, feel free to reach out via email or LinkedIn.

📬 Contact
Sumayya Ali
📧 sumayyaali.work@gmail.com
🔗www.linkedin.com/in/sumayyaali | https://github.com/SumayyaAli11   
