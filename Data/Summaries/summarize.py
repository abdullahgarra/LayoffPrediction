import openai
import csv
import time
import pymysql
import argparse
import json 
from dotenv import load_dotenv
import os

load_dotenv()

ApiKey = os.getenv("API_KEY")
client = openai.OpenAI(api_key=ApiKey)

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
TABLE_NAME = os.getenv("TABLE_NAME")

# Connect to the database
def connect_to_db():
    connection = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection


# Save a summary into the database
def save_summary_to_db(connection, publisher, published_date, summary):
    try:
        with connection.cursor() as cursor:
            insert_query = """
            INSERT INTO summaries (publisher, published_date, summary)
            VALUES (%s, %s, %s);
            """
            cursor.execute(insert_query, (publisher, published_date, summary))
            connection.commit()
    except Exception as e:
        print(f"Error saving summary: {e}")

# Load your CSV dataset
def load_news_articles(csv_file):
    articles = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            articles.append({
                "publisher": row["publisher"],
                "published_date": row["published_date"],
                "plain_text": row["plain_text"]
            })
    return articles

# Function to summarize a batch of news articles using the correct OpenAI API structure
def summarize_articles(articles_batch):
    prompt = "Summarize each of the following financial news articles in up to 4 lines individually :\n"
    
    for article in articles_batch:
        prompt += f"Publisher: {article['publisher']}\nDate: {article['published_date']}\nArticle: {article['plain_text']}\n\n"
    
    if len(prompt) > 3000:
        print("Warning: The batch is too large, consider reducing its size.")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[{"role": "system", "content": "You are a financial news summarizer."},
                      {"role": "user", "content": prompt + "\nReturn the result in JSON list format with keys 'publisher', 'published_date', and 'summary'. Don't wrap the json with three ticks (```)"}],
            max_tokens=1024, 
            temperature=0.5
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None

# Function to process the news data in batches and summarize
def process_news_summaries(articles, batch_size=5):
    connection = connect_to_db()
    num_batches = len(articles) // batch_size + (1 if len(articles) % batch_size != 0 else 0)
    
    for i in range(num_batches):
        batch = articles[i * batch_size:(i + 1) * batch_size]
        print(f"Processing batch {i+1}/{num_batches}")
        summary_json = summarize_articles(batch)
        print(summary_json)
        if summary_json:
            try:
                summaries = json.loads(summary_json)
                for article_summary in summaries:
                    # print(article_summary['publisher'])
                    # print(article_summary['published_date'])
                    # print(article_summary['summary'])
                    save_summary_to_db(connection, 
                                       article_summary['publisher'], 
                                       article_summary['published_date'], 
                                       article_summary['summary'])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        
        time.sleep(1)  
    connection.close()

# Main function to execute the pipeline
def main():
    # Parse command line arguments to get the file name
    parser = argparse.ArgumentParser(description="Process news articles for summarization.")
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing the news articles')
    args = parser.parse_args()

    # Load the news articles dataset from the CSV file provided in the terminal
    articles = load_news_articles(args.csv_file)

    # Process and summarize the news articles in batches
    process_news_summaries(articles, batch_size=3)  

if __name__ == "__main__":
    main()
