import pandas as pd

# Load your CSV
df = pd.read_csv("output_with_coords.csv")  # Replace with your actual filename

# Replace this with the actual column name
service_column = "Service Request Type"  # Check with print(df.columns) if needed

# Get top 10 most frequent service requests
top_requests = (
    df[service_column]
    .dropna()
    .value_counts()
    .head(10)
)

# Print as a Python list literal
print("COPY THIS LIST BELOW:\n")
print("[")
for req in top_requests.index:
    print(f"    {repr(req)},")
print("]")
