from datetime import datetime, timedelta
from db_connection import connect_to_db

def fetch_summaries_within_window(layoff_timestamp, days_before, table_name="summaries"):
    layoff_date = datetime.fromtimestamp(layoff_timestamp)
    start_date = layoff_date - timedelta(days=days_before)
    query = f"SELECT summary FROM {table_name} WHERE published_date BETWEEN %s AND %s;"
    with connect_to_db() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (start_date, layoff_date))
            results = cursor.fetchall()
    
    return [row['summary'] for row in results]
