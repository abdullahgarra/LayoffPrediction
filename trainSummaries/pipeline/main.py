import pandas as pd
from db_connection import create_filtered_tables
from data_fetcher import fetch_summaries_within_window
from heuristic_filter import filter_summaries
from data_inserter import initialize_csv, insert_filtered_summaries
from embedding_model import get_example_embeddings

# Define example sentences for filtering
example_sentences = [
    "The company announced a massive layoff affecting thousands of employees.",
    "Due to financial difficulties, several departments will face job cuts.",
    "The stock price plunged following reports of an earnings miss.",
    "A downturn in the market has severely impacted the company's profits.",
    "The company is undergoing a restructuring that will lead to layoffs.",
    "Quarterly losses have forced the company to reduce its workforce.",
    "A significant drop in revenue is pushing the company to lay off workers.",
    "The board decided to cut jobs to manage the company's declining earnings.",
    "Economic challenges are driving layoffs in multiple sectors.",
    "The company announced a hiring freeze and potential job cuts.",
    "Due to unforeseen financial challenges, layoffs have become necessary.",
    "Market volatility has led to a decline in the company’s stock value.",
    "Facing financial instability, the company is downsizing.",
    "The firm is closing down divisions due to underperformance.",
    "Shares fell sharply after the company issued a profit warning."
]
example_embeddings = get_example_embeddings(example_sentences)

# Process each layoff record
def process_and_insert_summaries(layoff_data):
    windows = {"filtered_30_days": 30, "filtered_15_days": 15, "filtered_7_days": 7, "filtered_90_days": 90}
    
    for idx, row in layoff_data.iterrows():
        layoff_timestamp = row['Date_layoffs']
        layoff_id = row.name  
        
        for table_name, days in windows.items():
            summaries = fetch_summaries_within_window(layoff_timestamp, days)
            if summaries:
                filtered_summaries = filter_summaries(summaries, example_embeddings)
                insert_filtered_summaries(layoff_id, filtered_summaries, table_name)

# Load layoff data
layoff_data = pd.read_csv("finalDataset.csv")

# Run the pipeline
initialize_csv()
create_filtered_tables()
process_and_insert_summaries(layoff_data)
