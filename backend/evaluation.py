import pandas as pd
from typing import List, Dict
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
#import seaborn as sns
#import matplotlib.pyplot as plt
import os
from agents import diagnosis_chain

class MedicalDiagnosisEvaluator:
    def __init__(self, test_data_path: str= str("D:\\MAYNOOTH\\SEM 2\\SUMMER PROJECT\\DATASET\\new_test_set.csv")):
        self.test_df = pd.read_csv(test_data_path)
        self.symptom_columns = [col for col in self.test_df.columns if col != 'prognosis']
        
    def prepare_test_cases(self) -> List[Dict]:
        test_cases = []
        for _, row in self.test_df.iterrows():
            symptoms = {col: bool(row[col]) for col in self.symptom_columns}
            test_cases.append({
                'symptoms': symptoms,
                'true_diagnosis': row['prognosis']
            })
        return test_cases
    
    def run_diagnosis(self, symptoms: Dict[str, bool]) -> Dict:
        symptom_text = "I have " + ", ".join([s for s, present in symptoms.items() if present])
        initial_state = {
            "input": symptom_text,
            "structured_symptoms": symptoms,
            "retrieved_diseases": [],
            "predictions": [],
            "report": ""
        }
        return diagnosis_chain.invoke(initial_state)
    
    def evaluate(self, sample_size: int = None) -> Dict[str, float]:
        test_cases = self.prepare_test_cases()
        if sample_size:
            test_cases = test_cases[:sample_size]
            
        y_true = []
        y_pred = []
        
        for case in tqdm(test_cases, desc="Evaluating cases"):
            result = self.run_diagnosis(case['symptoms'])
            pred_diagnosis = result['predictions'][0].disease if result['predictions'] else "Unknown"
            y_true.append(case['true_diagnosis'])
            y_pred.append(pred_diagnosis)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def save_confusion_matrix(self, output_path: str = "D:\\MAYNOOTH\\SEM 2\\SUMMER PROJECT\\DATASET\\confusion_matrix.csv"):
        test_cases = self.prepare_test_cases()
        y_true = []
        y_pred = []
        
        for case in tqdm(test_cases, desc="Generating confusion matrix"):
            result = self.run_diagnosis(case['symptoms'])
            pred_diagnosis = result['predictions'][0].disease if result['predictions'] else "Unknown"
            y_true.append(case['true_diagnosis'])
            y_pred.append(pred_diagnosis)
        
        classes = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        cm_df.to_csv(output_path)
        
        # plt.figure(figsize=(12, 10))
        # #sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        # plt.title('Confusion Matrix')
        # plt.ylabel('True Diagnosis')
        # plt.xlabel('Predicted Diagnosis')
        # plt.tight_layout()
        # plt.savefig("D:\\MAYNOOTH\\SEM 2\\SUMMER PROJECT\\DATASET\\confusion_matrix.png")
        # plt.close()
        
        return cm_df