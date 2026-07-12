import os
import argparse
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from google.cloud import storage

def parse_args():
    parser = argparse.ArgumentParser(description="Vertex AI Iris Training Pipeline")
    parser.add_argument('--bucket_name', type=str, required=True, 
                        help="The unique name of your GCS bucket")
    parser.add_argument('--data_version', type=str, default='v1', choices=['v1', 'v2'],
                        help="The data version folder to train on (v1 or v2)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Generate a unique timestamp for this training run execution
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"Starting training pipeline run at: {timestamp}")
    
    # 1. Initialize GCS Client
    storage_client = storage.Client()
    bucket = storage_client.bucket(args.bucket_name)
    
    # 2. Download specific data version from GCS
    gcs_data_path = f"data/{args.data_version}/data.csv"
    local_data_path = "temp_iris_data.csv"
    
    print(f"Downloading data from: gs://{args.bucket_name}/{gcs_data_path}")
    blob = bucket.blob(gcs_data_path)
    blob.download_to_filename(local_data_path)
    
    # 3. Load and Prepare Data
    df = pd.read_csv(local_data_path)
    
    # Target column handling (assumes class/label column is named 'target' or 'species')
    target_col = 'target' if 'target' in df.columns else 'species'
    if target_col not in df.columns:
        # Fallback to the last column if neither named column is present
        target_col = df.columns[-1]
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split into Train (80%) and Evaluation (20%) sets
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Model
    print(f"Training RandomForest Model using {args.data_version} data...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    print(f"Training Complete. Training Accuracy: {train_acc:.4f}")
    
    # 5. Save Artifacts Locally Temporarily
    local_model = "model.joblib"
    local_log = "run_log.txt"
    local_eval = "eval_set.csv"
    
    joblib.dump(model, local_model)
    
    # Save the split evaluation set so inference script reads identical features/rows
    X_eval.assign(**{target_col: y_eval}).to_csv(local_eval, index=False)
    
    with open(local_log, "w") as f:
        f.write(f"Execution Timestamp: {timestamp}\n")
        f.write(f"Dataset Version Used: {args.data_version}\n")
        f.write(f"Training Accuracy Score: {train_acc:.4f}\n")
        
    # 6. Upload Pipeline Artifacts to Timestamped GCS Path
    gcs_output_prefix = f"artifacts/{args.data_version}/{timestamp}/"
    print(f"Uploading artifacts to: gs://{args.bucket_name}/{gcs_output_prefix}")
    
    bucket.blob(f"{gcs_output_prefix}{local_model}").upload_from_filename(local_model)
    bucket.blob(f"{gcs_output_prefix}{local_log}").upload_from_filename(local_log)
    bucket.blob(f"{gcs_output_prefix}{local_eval}").upload_from_filename(local_eval)
    
    print(f"✅ Pipeline run complete. Use this prefix for inference:\n{gcs_output_prefix}\n")

if __name__ == "__main__":
    main()