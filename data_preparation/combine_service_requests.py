import pandas as pd
import glob
import os
from pathlib import Path

# --- Configuration ---
INPUT_PATTERN = "SR*.csv"  # Pattern to match service request files
OUTPUT_CSV = "SRALL.csv"  # Output combined file

print("--- Combining Service Request CSV Files ---")

# 1. Find all SR*.csv files
print(f"  - Looking for files matching pattern '{INPUT_PATTERN}'...")
sr_files = glob.glob(INPUT_PATTERN)

if not sr_files:
    print(f"  - No files found matching pattern '{INPUT_PATTERN}'")
    print(f"  - Make sure you're in the correct directory with SR2014.csv, SR2015.csv, etc.")
    exit(1)

# 2. Sort files by year (extract year from filename)
def extract_year(filename):
    """Extract year from filename like 'SR2019.csv' -> 2019"""
    try:
        # Remove 'SR' prefix and '.csv' suffix, then convert to int
        year_str = filename.replace('SR', '').replace('.csv', '')
        return int(year_str)
    except ValueError:
        # If can't extract year, put at end
        return 9999

sr_files_sorted = sorted(sr_files, key=extract_year)

print(f"  - Found {len(sr_files_sorted)} files:")
for file in sr_files_sorted:
    file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
    print(f"    * {file} ({file_size:.1f} MB)")

# 3. Read and combine all files
print(f"\n  - Reading and combining files...")
combined_data = []
total_rows = 0
skipped_rows = 0

for i, file in enumerate(sr_files_sorted):
    print(f"    * Reading {file}...")
    try:
        # First, try to read normally
        df = pd.read_csv(file)
        rows_in_file = len(df)
        total_rows += rows_in_file
        
        # Add a source file column for tracking
        df['Source_File'] = file
        
        combined_data.append(df)
        print(f"      - Loaded {rows_in_file:,} rows")
        
    except pd.errors.ParserError as e:
        print(f"      - Parser error detected, trying with error handling...")
        try:
            # Try with error handling for malformed lines
            df = pd.read_csv(file, on_bad_lines='skip', engine='python')
            rows_in_file = len(df)
            total_rows += rows_in_file
            
            # Add a source file column for tracking
            df['Source_File'] = file
            
            combined_data.append(df)
            print(f"      - Loaded {rows_in_file:,} rows (some malformed lines skipped)")
            
        except Exception as e2:
            print(f"      - Still failed with error handling: {e2}")
            print(f"      - Trying manual column inspection...")
            
            try:
                # Read just the header to see what columns we have
                with open(file, 'r', encoding='utf-8') as f:
                    header_line = f.readline().strip()
                    columns = [col.strip('"') for col in header_line.split(',')]
                    print(f"      - File has {len(columns)} columns: {columns}")
                
                # If it has extra columns, try reading with specific column names
                if len(columns) > 9:
                    # Use the standard 9 columns from the working files
                    standard_cols = ["Creation Date", "Status", "First 3 Chars of Postal Code", 
                                   "Intersection Street 1", "Intersection Street 2", "Ward", 
                                   "Service Request Type", "Division", "Section"]
                    
                    df = pd.read_csv(file, usecols=range(9), names=standard_cols, 
                                   skiprows=1, on_bad_lines='skip', engine='python')
                    rows_in_file = len(df)
                    total_rows += rows_in_file
                    
                    # Add a source file column for tracking
                    df['Source_File'] = file
                    
                    combined_data.append(df)
                    print(f"      - Loaded {rows_in_file:,} rows using first 9 columns only")
                else:
                    print(f"      - Could not resolve column mismatch")
                    continue
                    
            except Exception as e3:
                print(f"      - Final attempt failed: {e3}")
                continue
                
    except Exception as e:
        print(f"      - ERROR reading {file}: {e}")
        continue

if not combined_data:
    print(f"  - No valid data found in any files!")
    exit(1)

# 4. Concatenate all dataframes
print(f"\n  - Concatenating {len(combined_data)} dataframes...")
final_df = pd.concat(combined_data, ignore_index=True)

print(f"  - Combined dataset contains {len(final_df):,} total rows")

# 5. Sort by Creation Date to ensure chronological order
print(f"  - Sorting by Creation Date...")
try:
    # Convert Creation Date to datetime for proper sorting
    final_df['Creation Date'] = pd.to_datetime(final_df['Creation Date'], errors='coerce')
    final_df = final_df.sort_values('Creation Date')
    
    # Show date range (excluding NaT values)
    valid_dates = final_df['Creation Date'].dropna()
    if len(valid_dates) > 0:
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        print(f"  - Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
except Exception as e:
    print(f"  - WARNING: Could not sort by date: {e}")
    print(f"  - Files will be combined in filename order")

# 6. Save combined file
print(f"\n  - Saving combined data to '{OUTPUT_CSV}'...")
final_df.to_csv(OUTPUT_CSV, index=False)

# 7. Summary statistics
print(f"\n--- Combination Complete! ---")
print(f"  - Input files: {len(sr_files_sorted)}")
print(f"  - Successfully processed: {len(combined_data)}")
print(f"  - Total rows: {len(final_df):,}")
print(f"  - Output file: {OUTPUT_CSV}")
print(f"  - File size: {os.path.getsize(OUTPUT_CSV) / (1024 * 1024):.1f} MB")

# Show breakdown by year
print(f"\n--- Rows by Source File ---")
source_counts = final_df['Source_File'].value_counts().sort_index()
for file, count in source_counts.items():
    print(f"  - {file}: {count:,} rows") 