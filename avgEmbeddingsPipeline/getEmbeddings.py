import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from datetime import datetime, timedelta
from db_connection import connect_to_db
import os

# Load FinBERT model and tokenizer
print("Loading FinBERT model and tokenizer...")
finbert_model = AutoModel.from_pretrained('ProsusAI/finbert')
finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
print("FinBERT loaded successfully.")

# Load the layoff events dataset
print("Loading layoff dataset...")
final_df = pd.read_csv("finalDataset.csv")
print("Dataset loaded. Total records:", len(final_df))

# Check if we are resuming and get the last processed date
output_file = "finalDataset_with_embeddings.csv"
if os.path.exists(output_file):
    processed_df = pd.read_csv(output_file)
    last_processed_date = processed_df['Date_layoffs'].max()
    final_df = final_df[final_df['Date_layoffs'] > last_processed_date]
    print(f"Resuming from date: {last_processed_date}")
else:
    print("Starting from the beginning.")

# Function to get embedding for a single summary
def get_finbert_embedding(text):
    inputs = finbert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = finbert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()  # CLS token as embedding

# Fetch summaries within a window around the layoff date
def fetch_summaries_within_window(layoff_timestamp, days_before, table_name="summaries"):
    layoff_date = datetime.fromtimestamp(layoff_timestamp)
    start_date = layoff_date - timedelta(days=days_before)
    print(f"Fetching summaries from {start_date.date()} to {layoff_date.date()}...")

    query = f"SELECT summary FROM {table_name} WHERE published_date BETWEEN %s AND %s;"
    with connect_to_db() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (start_date, layoff_date))
            results = cursor.fetchall()

    summaries = [row['summary'] for row in results]
    print(f"Fetched {len(summaries)} summaries.")
    return summaries

# Calculate average embeddings for each layoff event
def calculate_average_embedding(layoff_timestamp, days=90):
    summaries = fetch_summaries_within_window(layoff_timestamp, days_before=days)
    if not summaries:
        print("No summaries found within the window. Returning zero vector.")
        return np.zeros(768)

    print("Calculating embeddings for fetched summaries...")
    embeddings = np.array([get_finbert_embedding(summary) for summary in summaries])
    avg_embedding = np.mean(embeddings, axis=0)
    print("Average embedding calculated.")
    return avg_embedding

# Processing each layoff event and saving in batches
print("Processing each layoff event to calculate average embeddings...")
batch_data = []
for idx, row in final_df.iterrows():
    layoff_timestamp = row['Date_layoffs']
    avg_embedding = calculate_average_embedding(layoff_timestamp)
    batch_data.append([layoff_timestamp, avg_embedding])

    # Write every 5 records
    if len(batch_data) >= 5:
        # Convert batch data to DataFrame
        batch_df = pd.DataFrame(batch_data, columns=["Date_layoffs", "average_embedding"])
        
        # Save the batch to CSV
        if os.path.exists(output_file):
            batch_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            batch_df.to_csv(output_file, mode='w', header=True, index=False)
        
        print(f"Saved batch ending on date {batch_data[-1][0]}.")
        
        # Clear the batch
        batch_data = []

# Save any remaining data
if batch_data:
    batch_df = pd.DataFrame(batch_data, columns=["Date_layoffs", "average_embedding"])
    batch_df.to_csv(output_file, mode='a', header=False, index=False)
    print(f"Final batch saved ending on date {batch_data[-1][0]}.")
