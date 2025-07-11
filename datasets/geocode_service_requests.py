import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# --- Configuration ---
SERVICE_REQUESTS_CSV = "output.csv"  # Path to your service requests CSV
INTERSECTION_LOOKUP_CSV = "lookup.csv"  # Path to the intersection lookup table

OUTPUT_GEOJSON = "service_requests_with_coords.geojson"  # Output GeoJSON file with coordinates

print("--- Geocoding Service Requests Using Intersection Lookup ---")

# 1. Load the service requests data
print(f"  - Loading service requests from '{SERVICE_REQUESTS_CSV}'...")
service_requests = pd.read_csv(SERVICE_REQUESTS_CSV)
print(f"  - Loaded {len(service_requests)} service requests")

# 2. Load the intersection lookup table
print(f"  - Loading intersection lookup table from '{INTERSECTION_LOOKUP_CSV}'...")
lookup_table = pd.read_csv(INTERSECTION_LOOKUP_CSV)
print(f"  - Loaded {len(lookup_table)} intersection pairs")

# 3. Prepare the lookup table for efficient matching
# Create a dictionary for fast lookup: (street1, street2) -> (lat, lon)
print("  - Preparing lookup dictionary...")
lookup_dict = {}

for _, row in lookup_table.iterrows():
    street1 = str(row['Intersection Street 1']).strip().upper()
    street2 = str(row['Intersection Street 2']).strip().upper()
    lat = row['LATITUDE']
    lon = row['LONGITUDE']
    
    # Store both orientations (street1, street2) and (street2, street1)
    lookup_dict[(street1, street2)] = (lat, lon)
    lookup_dict[(street2, street1)] = (lat, lon)

print(f"  - Created lookup dictionary with {len(lookup_dict)} entries")

# 4. Function to geocode intersections
def geocode_intersection(street1, street2):
    """
    Look up coordinates for an intersection of two streets.
    Returns (latitude, longitude) tuple or (None, None) if not found.
    """
    if pd.isna(street1) or pd.isna(street2):
        return None, None
    
    # Standardize street names
    street1_clean = str(street1).strip().upper()
    street2_clean = str(street2).strip().upper()
    
    # Look up in dictionary
    return lookup_dict.get((street1_clean, street2_clean), (None, None))

# 5. Apply geocoding to service requests
print("  - Geocoding intersections...")
coords = service_requests.apply(
    lambda row: geocode_intersection(row['Intersection Street 1'], row['Intersection Street 2']), 
    axis=1
)

# Split the coordinates into separate columns
service_requests['LATITUDE'] = [coord[0] for coord in coords]
service_requests['LONGITUDE'] = [coord[1] for coord in coords]

# 6. Report results
total_requests = len(service_requests)
geocoded_requests = service_requests['LATITUDE'].notna().sum()
geocoding_rate = (geocoded_requests / total_requests) * 100

print(f"\n--- Geocoding Results ---")
print(f"  - Total service requests: {total_requests}")
print(f"  - Successfully geocoded: {geocoded_requests}")
print(f"  - Geocoding rate: {geocoding_rate:.1f}%")
print(f"  - Failed to geocode: {total_requests - geocoded_requests}")

# 7. Show some examples of failed geocoding (for debugging)
failed_requests = service_requests[service_requests['LATITUDE'].isna()]
if len(failed_requests) > 0:
    print(f"\n--- Sample Failed Geocoding Attempts ---")
    sample_failed = failed_requests[['Intersection Street 1', 'Intersection Street 2']].head(10)
    for _, row in sample_failed.iterrows():
        print(f"  - '{row['Intersection Street 1']}' & '{row['Intersection Street 2']}'")

# 8. Create GeoDataFrame and save as GeoJSON
print(f"\n  - Creating GeoJSON with {geocoded_requests} geocoded points...")

# Filter to only geocoded requests (those with coordinates)
geocoded_data = service_requests[service_requests['LATITUDE'].notna()].copy()

if len(geocoded_data) > 0:
    # Create Point geometries from coordinates
    geometry = [Point(lon, lat) for lat, lon in zip(geocoded_data['LATITUDE'], geocoded_data['LONGITUDE'])]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geocoded_data, geometry=geometry, crs='EPSG:4326')
    
    # Remove the separate LATITUDE/LONGITUDE columns since they're now in geometry
    gdf = gdf.drop(['LATITUDE', 'LONGITUDE'], axis=1)
    
    # Save as GeoJSON
    print(f"  - Saving {len(gdf)} geocoded records to '{OUTPUT_GEOJSON}'...")
    gdf.to_file(OUTPUT_GEOJSON, driver='GeoJSON')
    
    print(f"\n--- Geocoding Complete! Saved to '{OUTPUT_GEOJSON}' ---")
    print(f"  - GeoJSON contains {len(gdf)} features with coordinates")
    print(f"  - {total_requests - len(gdf)} records without coordinates were excluded")
else:
    print(f"\n--- No geocoded records found! ---")
    print("  - Check that your intersection lookup table has matching street names") 