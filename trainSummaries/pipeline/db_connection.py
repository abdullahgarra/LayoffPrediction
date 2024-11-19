import pymysql
import os

load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
TABLE_NAME = os.getenv("TABLE_NAME")

def connect_to_db():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def create_filtered_tables():
    table_queries = {
        "filtered_30_days": """
            CREATE TABLE IF NOT EXISTS filtered_30_days (
                layoff_id INT PRIMARY KEY,
                filtered_articles TEXT
            )""",
        "filtered_15_days": """
            CREATE TABLE IF NOT EXISTS filtered_15_days (
                layoff_id INT PRIMARY KEY,
                filtered_articles TEXT
            )""",
        "filtered_7_days": """
            CREATE TABLE IF NOT EXISTS filtered_7_days (
                layoff_id INT PRIMARY KEY,
                filtered_articles TEXT
            )""",
        "filtered_90_days": """
            CREATE TABLE IF NOT EXISTS filtered_90_days (
                layoff_id INT PRIMARY KEY,
                filtered_articles TEXT
            )"""
    }
    
    with connect_to_db() as connection:
        with connection.cursor() as cursor:
            for table_name, query in table_queries.items():
                cursor.execute(query)
                print(f"Table '{table_name}' created or verified.")
        connection.commit()
