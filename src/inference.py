import argparse
import os
import pandas as pd
import joblib
from google.cloud import storage
from sklearn.metrics import accuracy_score, classification_report

def parse_args():
    parser = argparse.ArgumentParser(description="Vertex AI Iris Inference Script")
    parser.add_argument('--bucket_name', type=str, required=True,
                        help="The unique name of your GCS bucket")
    parser.add_argument('--artifact_prefix', type=str, required=True,
                        help="The timestamped directory prefix, e.g., artifacts/v1/20260614T103000/")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Standardize prefix formatting
    prefix = args.artifact_prefix.strip("/") + "/"
    
    # 1. Initialize GCS Client
    storage_client = storage.Client()
    bucket = storage_client.bucket(args.bucket_name)
    
    # Local placeholder filenames
    local_model = "downloaded_model.joblib"
    local_eval = "downloaded_eval.csv"
    
    # 2. Fetch Run Artifacts from Cloud Storage
    print(f"Fetching models and matching eval sets from: gs://{args.bucket_name}/{prefix}")
    bucket.blob(f"{prefix}model.joblib").download_to_filename(local_model)
    bucket.blob(f"{prefix}eval_set.csv").download_to_filename(local_eval)
    
    # 3. Load Data & Model
    model = joblib.load(local_model)
    df_eval = pd.read_csv(local_eval)
    
    target_col = 'target' if 'target' in df_eval.columns else 'species'
    if target_col not in df_eval.columns:
        target_col = df_eval.columns[-1]
        
    X_eval = df_eval.drop(columns=[target_col])
    y_true = df_eval[target_col]
    
    # 4. Generate Predictions & Calculate Metrics
    y_pred = model.predict(X_eval)
    eval_acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    print("\n=============================================")
    print(f"🚀 Evaluation Performance for Run [{prefix}]")
    print(f"Accuracy: {eval_acc:.4f}")
    print("Full Classification Report:")
    print(report)
    print("=============================================\n")
    
    # 5. Push Metrics Result Back into the Same Timestamped Run Folder
    result_blob = bucket.blob(f"{prefix}evaluation_results.txt")
    
    output_string = f"Evaluation Accuracy: {eval_acc:.4f}\n\nClassification Report:\n{report}"
    result_blob.upload_from_string(output_string)
    print(f"Saved evaluation performance metrics report to GCS run folder.")

if __name__ == "__main__":
    main()