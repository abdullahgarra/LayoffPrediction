import csv
from db_connection import connect_to_db

# Define CSV file path
CSV_FILE_PATH = "filtered_layoff_summaries.csv"

# Initialize the CSV file with headers
def initialize_csv():
    with open(CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["layoff_id", "table_name", "filtered_articles"])  

# Function to insert filtered summaries into the database and CSV
def insert_filtered_summaries(layoff_id, filtered_summaries, table_name):
    with connect_to_db() as connection:
        with connection.cursor() as cursor:
            query = f"""
            REPLACE INTO {table_name} (layoff_id, filtered_articles)
            VALUES (%s, %s)
            """
            cursor.execute(query, (layoff_id, filtered_summaries))
        connection.commit()
    print(f"Inserted filtered summaries for layoff_id {layoff_id} into {table_name}")

    # Append to the CSV file
    with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([layoff_id, table_name, filtered_summaries])
    print(f"Appended filtered summaries for layoff_id {layoff_id} to CSV.")
