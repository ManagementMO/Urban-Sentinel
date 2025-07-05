import pandas as pd

# Load your CSV
df = pd.read_csv("output_with_coords.csv")  # Replace with your actual filename

# Replace this with the actual column name
service_column = "Service Request Type"  # You can print(df.columns) to verify

# Get distinct service requests
distinct_requests = df[service_column].dropna().unique()

# Sort them (optional)
distinct_requests = sorted(distinct_requests)

# Print as a Python list literal
print("COPY THIS LIST BELOW:\n")
print("[")
for req in distinct_requests:
    print(f"    {repr(req)},")
print("]")