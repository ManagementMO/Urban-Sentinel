import pandas as pd
import json
from shapely.geometry import LineString, shape
from shapely.strtree import STRtree
from collections import defaultdict

# Load the intersection data (street geometries)
intersection_df = pd.read_csv("IntersectionLookup.csv")

# Map from LINEAR_NAME_FULL to list of LineStrings
street_geometries = defaultdict(list)

# This data loading section is robust and correct.
for _, row in intersection_df.iterrows():
    street_name = row["LINEAR_NAME_FULL"]
    if pd.isna(row["geometry"]):
        continue
    try:
        geom_data = json.loads(row["geometry"])
        geom = shape(geom_data)
        if geom.geom_type == 'LineString':
            geoms_to_process = [geom]
        elif geom.geom_type == 'MultiLineString':
            geoms_to_process = list(geom.geoms)
        else:
            continue
        for line in geoms_to_process:
            if line and not line.is_empty:
                street_geometries[street_name].append(line)
    except (json.JSONDecodeError, TypeError, AttributeError):
        continue

# Load your output.csv
output_df = pd.read_csv("output.csv")

# Check if required columns exist
required_cols = {'Intersection Street 1', 'Intersection Street 2'}
if not required_cols.issubset(output_df.columns):
    raise ValueError(f"output.csv must contain {required_cols} columns")

# Filter the DataFrame
valid_streets = set(street_geometries.keys())
output_df = output_df[
    output_df['Intersection Street 1'].isin(valid_streets) &
    output_df['Intersection Street 2'].isin(valid_streets)
].copy()

# Initialize new columns
latitudes = []
longitudes = []

# Loop through each row and find intersection coordinates
for _, row in output_df.iterrows():
    s1, s2 = row['Intersection Street 1'], row['Intersection Street 2']
    lines1 = street_geometries[s1]
    lines2 = street_geometries[s2]

    if not lines1 or not lines2:
        latitudes.append(None)
        longitudes.append(None)
        continue

    # Determine which list of lines to build the tree from and which to query with
    if len(lines1) < len(lines2):
        tree = STRtree(lines2)
        query_lines = lines1
        search_lines = lines2 # This list corresponds to the geometries in the tree
    else:
        tree = STRtree(lines1)
        query_lines = lines2
        search_lines = lines1 # This list corresponds to the geometries in the tree

    found_point = None

    for l in query_lines:
        # --- MODIFIED SECTION START ---
        # Query the tree to find the *indices* of candidate geometries
        candidate_indices = tree.query(l)
        
        # If no candidates are found, continue to the next line
        if len(candidate_indices) == 0:
            continue

        # Iterate through the indices to get the actual geometries
        for index in candidate_indices:
            # Get the actual geometry object from the search_lines list
            candidate = search_lines[index]
            
            # Now 'candidate' is a valid LineString, and this check will work
            if l.intersects(candidate):
                inter = l.intersection(candidate)
                if inter.is_empty:
                    continue
                if inter.geom_type == "Point":
                    found_point = inter
                    break
                elif inter.geom_type == "MultiPoint":
                    found_point = list(inter.geoms)[0]
                    break
        # --- MODIFIED SECTION END ---
        if found_point:
            break

    if found_point:
        latitudes.append(found_point.y)
        longitudes.append(found_point.x)
    else:
        latitudes.append(None)
        longitudes.append(None)

# Add the new columns to output_df
output_df["latitude"] = latitudes
output_df["longitude"] = longitudes

# Save the final output
output_df.to_csv("output_with_coords.csv", index=False)

print("Processing complete. Saved to output_with_coords.csv")