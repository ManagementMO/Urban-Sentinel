import pandas as pd

# List of files to read
files = [f"SR{i}.csv" for i in range(2014, 2021)]  # From SR2014 to SR2021

# Initialize an empty list to store DataFrames
dfs = []

# Loop through each file and extract the required columns
for file in files:
    try:
        # Read the CSV file and select only the required columns
        df = pd.read_csv(file)
        dfs.append(df)  # Append the DataFrame to the list
    except ValueError:
        print(f"Skipping {file}: Missing one of the required columns")

# Combine all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Optional: Drop rows where 'Intersection Street 1' is missing
combined_df = combined_df.dropna(subset=['Intersection Street 1'])

# Output the combined DataFrame to a new CSV file
combined_df.to_csv('mega_combined_output.csv', index=False)

print("CSV files have been combined successfully!")