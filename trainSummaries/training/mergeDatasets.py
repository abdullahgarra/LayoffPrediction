import pandas as pd

# Load the datasets
final_df = pd.read_csv("finalDataset.csv")  # Contains the Percentage column
filtered_summaries_df = pd.read_csv("filtered_layoff_summaries.csv")  # Contains id, table_name, filtered_articles

# Add an 'id' column based on the row index in final_df
final_df['id'] = final_df.index

# Merge datasets on 'id' to associate Percentage with each filtered article window
merged_df = pd.merge(filtered_summaries_df, final_df[['id', 'Percentage']], left_on='layoff_id', right_on='id', how='inner')

# Drop the extra 'id' column from final_df (optional)
merged_df.drop(columns=['id'], inplace=True)

# Confirm the merge and structure of data
print("Merged dataset sample:")
print(merged_df.head())

# Save merged dataset for inspection or further processing
merged_df.to_csv("filtered_articles_with_percentages.csv", index=False)
print("Merged dataset saved as 'filtered_articles_with_percentages.csv'")
