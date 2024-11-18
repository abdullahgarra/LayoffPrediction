import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
import csv
from tqdm import tqdm
from datetime import datetime
import time 

# Define the target domain as a string
target_domain = "finance.yahoo.com"

# Function to check if the URL contains the target domain
def is_target_domain(url, target_domain):
    return target_domain in url 

# Function to determine the quarter based on the date
def get_quarter(date_str):
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        month = date.month
        if 1 <= month <= 3:
            return 1
        elif 4 <= month <= 6:
            return 2
        elif 7 <= month <= 9:
            return 3
        elif 10 <= month <= 12:
            return 4
    except ValueError:
        return None  # Return None if the date is not valid

# Sequential processing for a specific year
def process_year(year):
    print(f"Processing data for {year}")
    
    # Determine the starting quarter
    start_quarter = 3 if year == 2020 else 1

    # Create a dictionary of CSV writers for each quarter
    csv_writers = {}
    for quarter in range(start_quarter, 5):  # Process from the start quarter to Q4
        output_file = f'filtered_q{quarter}_{year}_news.csv'
        csvfile = open(output_file, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csvfile)
        # Write the CSV header
        csv_writer.writerow(['publisher', 'published_date', 'plain_text'])
        csv_writers[quarter] = (csv_writer, csvfile)
    
    # Load the CCNews dataset for the specified year
    dataset = load_dataset('stanford-oval/ccnews', name=str(year), split='train', streaming=True)
    
    # Create a DataLoader with prefetching and concurrency
    dataloader = DataLoader(dataset, num_workers=16, prefetch_factor=20, batch_size=None)
    
    start_time = time.time()

    count = 0
    # Iterate over the streamed dataset and save records from the target domain
    for record in tqdm(dataloader, mininterval=20):
        publisher = record['publisher']
        # Check if the publisher contains the target domain
        if is_target_domain(publisher, target_domain):
            published_date = record['published_date']  # Published date

            # Determine the quarter based on the published_date
            quarter = get_quarter(published_date)
            if quarter and quarter >= start_quarter:  # Ensure quarter is within the valid range
                plain_text = record['plain_text']  # Article content (plain text)
                # Write the record to the corresponding quarter's CSV file
                csv_writers[quarter][0].writerow([publisher, published_date, plain_text])
                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count} Articles")

    # Track end time
    end_time = time.time()

    # Close all CSV files
    for _, (csv_writer, csvfile) in csv_writers.items():
        csvfile.close()

    # Calculate and print total time taken
    total_time = end_time - start_time
    print(f"Total time taken for {year}: {total_time:.2f} seconds")


# Main function to iterate over multiple years
def main():
    start_year = 2020
    end_year = 2024

    # Loop through each year sequentially
    for year in range(start_year, end_year + 1):
        process_year(year)
        print(f"Finished processing {year}")

if __name__ == "__main__":
    main()
