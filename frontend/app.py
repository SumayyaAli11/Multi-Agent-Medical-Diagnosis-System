import streamlit as st
import requests
import time
from typing import List, Dict
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Set page config
st.set_page_config(
    page_title="Multi-Agent Medical Diagnosis Assistant",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS
# Add this at the top of your Streamlit app (after imports)
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .diagnosis-card {
        border-left: 5px solid #4e79a7;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.25rem;
    }
    .followup-card {
        border-left: 5px solid #e15759;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.25rem;
    }
    /* Report card */
    .report-card {
        border-left: 5px solid #59a14f;
        padding: 1.5rem;
        margin: 1.5rem 0;
        background-color: #ffffff;  /* Pure white */
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        color: #222;  /* Almost black */
    }
    
</style>
""", unsafe_allow_html=True)

def diagnose_symptoms(symptom_text: str, structured_symptoms: Dict[str, bool] = None):
    """Send symptoms to backend API"""
    payload = {
        "text": symptom_text,
        "structured": structured_symptoms
    }
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/diagnose",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the diagnosis service: {e}")
        return None

def display_diagnosis(results: Dict):
    """Display diagnosis results in Streamlit"""
    if not results or "predictions" not in results:
        st.warning("No diagnosis results to display")
        return
    
    st.subheader("Diagnosis Results")
    
    # Display each prediction
    for pred in results["predictions"]:
        with st.expander(f"**{pred['disease']}** (Confidence: {pred['confidence']}%)"):
            st.markdown(f"**Matched Symptoms:** {', '.join(pred['symptoms_matched'])}")
            st.markdown(f"**Explanation:** {pred['explanation']}")
            
            if pred['follow_up_questions']:
                st.markdown("**Follow-up Questions:**")
                for q in pred['follow_up_questions']:
                    st.markdown(f"- {q}")
    
    # Display full report
    if results.get("report"):
        st.subheader("Patient Report Summary")
        st.markdown(f'<div class="report-card">{results["report"]}</div>', unsafe_allow_html=True)

def main():
    with st.sidebar:

        st.subheader("Configuration")
        
    # Groq API Key Input
        api_key = st.text_input("Groq API Key:", 
                          type="password",
                          value=os.getenv("GROQ_API_KEY", ""))
    
     # Validate API key
        if not api_key:
            st.warning("⚠️ Please enter your GROQ API key to proceed. Don't have? refer : https://console.groq.com/keys ")
        if st.button("Reset Session"):
            st.session_state.clear()
            st.rerun()
        # st.subheader("Workflow Diagram")
        # try:
        #     graph_img = diagnosis_chain.get_graph().draw_mermaid_png()
        #     st.image(graph_img, caption="Workflow Diagram", use_container_width=True)
        # except Exception as e:
        #     st.write("Could not display workflow diagram:", e)
        
        
    st.title("Multi-Agent Medical Diagnosis Assistant 🩺")
    st.markdown("Describe your symptoms or select them from the list below.")
    
    # Initialize session state
    if "diagnosis_history" not in st.session_state:
        st.session_state.diagnosis_history = []
    
    # Two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Enter Symptoms")
        
        # Option 1: Natural language input
        symptom_text = st.text_area(
            "Describe your symptoms in your own words:",
            placeholder="e.g., I have itching, rash, and fatigue..."
        )
        
        # Option 2: Structured symptom selection
        st.markdown("**Or select symptoms from list:**")
        
        # Load symptom options (in a real app, load from your dataset)
        symptom_options = [
            "itching", "skin_rash", "fatigue", "headache", 
            "fever", "cough", "joint_pain", "nausea"
        ]  # Add all your symptoms here
        
        structured_symptoms = {
            symptom: st.checkbox(symptom.replace("_", " ").title())
            for symptom in symptom_options
        }
        
        # Diagnosis button
        if st.button("Get Diagnosis", type="primary"):
            if not symptom_text and not any(structured_symptoms.values()):
                st.warning("Please describe symptoms or select from the list")
            else:
                with st.spinner("Analyzing symptoms..."):
                    start_time = time.time()
                    
                    # Call backend API
                    results = diagnose_symptoms(
                        symptom_text,
                        {k: v for k, v in structured_symptoms.items() if v}
                    )
                    
                    if results:
                        # Store in history
                        st.session_state.diagnosis_history.append({
                            "input": symptom_text or str(structured_symptoms),
                            "results": results,
                            "timestamp": time.time()
                        })
                        
                        # Display results
                        display_diagnosis(results)
                        
                        st.success(f"Diagnosis completed in {time.time() - start_time:.2f} seconds")
    
    with col2:
        st.subheader("Diagnosis History")
        
        if st.session_state.diagnosis_history:
            for i, entry in enumerate(reversed(st.session_state.diagnosis_history)):
                with st.expander(f"Case #{len(st.session_state.diagnosis_history)-i} - {time.ctime(entry['timestamp'])}"):
                    st.markdown(f"**Input:** {entry['input']}")
                    
                    if entry['results']['predictions']:
                        top_pred = max(
                            entry['results']['predictions'],
                            key=lambda x: x['confidence']
                        )
                        st.markdown(f"**Top Diagnosis:** {top_pred['disease']} ({top_pred['confidence']}%)")
                    else:
                        st.markdown("No diagnosis was determined")
        else:
            st.info("No diagnosis history yet")
    

if __name__ == "__main__":
    main()
    
    