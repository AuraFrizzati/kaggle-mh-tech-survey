# gcloud auth application-default login
# python3 -m venv .venv
# source .venv/bin/activate
# pip install kaggle google-cloud-bigquery pandas pyarrow python-dotenv

import os
import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from google.cloud import bigquery

# Load environment variables from the .env file
load_dotenv()

# Fetch the GCP project ID from the loaded environment variables
GCP_PROJECT = os.getenv("GCP_PROJECT")

# --- Configuration ---
KAGGLE_DATASET = "abbas829/mental-health-in-tech-survey-20142025" #https://www.kaggle.com/datasets/abbas829/mental-health-in-tech-survey-20142025
BQ_DATASET = "mh_tech_survey"
BQ_TABLE = "mh_tech_survey_raw"
FILE_NAME = "mental_health_survey.csv" # The specific file inside the Kaggle zip

# Define where you want to keep the local copy
LOCAL_DATA_DIR = "./data"

def main():
    # 1. Authenticate APIs
    print("Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()

    print("Authenticating with BigQuery...")
    bq_client = bigquery.Client(project=GCP_PROJECT)
    table_id = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

    # 2. Create the local directory
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    
    # 3. Download and extract
    print(f"Downloading {KAGGLE_DATASET} to {LOCAL_DATA_DIR}...")
    api.dataset_download_files(KAGGLE_DATASET, path=LOCAL_DATA_DIR, unzip=True)
    
    file_path = os.path.join(LOCAL_DATA_DIR, FILE_NAME)
    
    # 4. Clean column names with pandas
    #print("Cleaning column names (converting to snake_case)...")
    #df = pd.read_csv(file_path)
    
    # Strip leading/trailing whitespaces, replace spaces with underscores, and convert to lowercase
    #df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    # Save the cleaned DataFrame back to the CSV, overwriting the raw one
   # df.to_csv(file_path, index=False)
    
    # 5. Configure BigQuery Load Job
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True, 
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE 
    )

    # 6. Stream file to BigQuery
    print(f"Loading {FILE_NAME} into {table_id}...")
    with open(file_path, "rb") as source_file:
        job = bq_client.load_table_from_file(
            source_file, 
            table_id, 
            job_config=job_config
        )

    job.result() 
    
    table = bq_client.get_table(table_id)
    print(f"Success! Uploaded {table.num_rows} rows to {table_id}.")
    print(f"A cleaned local copy of the dataset has been retained at: {file_path}")

if __name__ == "__main__":
    main()

# python kaggle_to_bq.py