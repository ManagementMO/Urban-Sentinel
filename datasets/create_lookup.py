import geopandas as gpd
import pandas as pd
from itertools import combinations
from shapely.wkt import loads

# --- Configuration ---
TCL_GEOJSON_PATH = "centreline.geojson"  # Path to your Toronto Centreline GeoJSON file
OUTPUT_LOOKUP_TABLE = "intersection_lookup.csv"

print("--- Part A: Building the Intersection Lookup Table ---")

# 1. Load the Toronto Centreline GeoJSON file
print(f"  - Loading Toronto Centreline from '{TCL_GEOJSON_PATH}'...")
tcl = gpd.read_file(TCL_GEOJSON_PATH)

# 2. Extract all endpoints from all street segments
# An intersection is an endpoint shared by multiple street segments.
endpoints = {}
print("  - Extracting all street segment endpoints...")

for index, row in tcl.iterrows():
    # Process only LineString and MultiLineString geometries
    if row.geometry.geom_type not in ['LineString', 'MultiLineString']:
        continue

    # Get the street name from LINEAR_NAME_FULL
    street_name = ""
    try:
        if pd.notna(row['LINEAR_NAME_FULL']):
            street_name = str(row['LINEAR_NAME_FULL']).strip().upper()
    except KeyError:
        pass  # Column doesn't exist, keep street_name empty
    
    if not street_name:
        continue

    # Handle both LineString and MultiLineString
    if row.geometry.geom_type == 'LineString':
        lines = [row.geometry]
    else:  # MultiLineString
        lines = list(row.geometry.geoms)

    for line in lines:
        # Get the boundary points (start and end of each line segment)
        boundary = line.boundary
        if boundary.is_empty:
            continue

        # Handle boundary - can be Point or MultiPoint
        points = []
        if boundary.geom_type == 'Point':
            points = [boundary]
        elif boundary.geom_type == 'MultiPoint':
            points = list(boundary.geoms)

        for point in points:
            # Use WKT (Well-Known Text) as a unique key for the point geometry
            point_wkt = point.wkt
            
            if point_wkt not in endpoints:
                endpoints[point_wkt] = set()
            endpoints[point_wkt].add(street_name)

print(f"  - Found {len(endpoints)} unique endpoints.")

# 3. Filter for actual intersections (where 2 or more streets meet)
intersections = []
print("  - Identifying and processing actual intersections...")

for point_wkt, street_set in endpoints.items():
    if len(street_set) >= 2:
        # Create all unique pairs of streets at this intersection
        # sorted alphabetically to ensure consistency
        for street1, street2 in combinations(sorted(list(street_set)), 2):
            # Convert point_wkt back to a shapely Point to get coordinates
            point_geom = loads(point_wkt)
            intersections.append({
                "Intersection Street 1": street1,
                "Intersection Street 2": street2,
                "LATITUDE": point_geom.y,
                "LONGITUDE": point_geom.x
            })

print(f"  - Generated {len(intersections)} intersection pairs.")

# 4. Create and save the final lookup DataFrame
lookup_df = pd.DataFrame(intersections)
lookup_df.drop_duplicates(inplace=True)
lookup_df.to_csv(OUTPUT_LOOKUP_TABLE, index=False)

print(f"\n--- Lookup Table Complete! Saved to '{OUTPUT_LOOKUP_TABLE}' ---")