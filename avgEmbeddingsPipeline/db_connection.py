import pymysql

DB_HOST = "mysqlsrv1.cs.tau.ac.il"
DB_USER = "markfesenko"
DB_PASSWORD = "3mZhryk&^5yP"
DB_NAME = "markfesenko"

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
