from evaluation import MedicalDiagnosisEvaluator
import pandas as pd

def main():
    # Initialize evaluator
    evaluator = MedicalDiagnosisEvaluator("D:\\MAYNOOTH\\SEM 2\\SUMMER PROJECT\\DATASET\\new_test_set.csv")
    
    # Run evaluation on a sample 
    metrics = evaluator.evaluate()
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall (True Positive Rate): {metrics['recall']:.2%}")
    print(f"F1 Score: {metrics['f1']:.2%}")
    
    # Get confusion matrix
    confusion_matrix = evaluator.save_confusion_matrix()
    
    # Save results
    pd.DataFrame([metrics]).to_csv("D:\\MAYNOOTH\\SEM 2\\SUMMER PROJECT\\DATASET\\evaluation_metrics.csv")
    confusion_matrix.to_csv("D:\\MAYNOOTH\\SEM 2\\SUMMER PROJECT\\DATASET\\confusion_matrix.csv")
    
    print("\nConfusion matrix saved for analysis")


if __name__ == "__main__":
    main()