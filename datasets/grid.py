import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from geopy.geocoders import Nominatim # The geocoding service
from tqdm import tqdm # The progress bar library
import time # To add delays if needed
import os
import asyncio

# --- Configuration ---
# Dynamically build file paths. Assumes all files are in the same directory as the script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TORONTO_BOUNDARY_SHP = os.path.join(SCRIPT_DIR, "citygcs_regional_mun_wgs84.shp")
SERVICE_REQUESTS_CSV = os.path.join(SCRIPT_DIR, "SR2020.csv")
OUTPUT_FILENAME = os.path.join(SCRIPT_DIR, "toronto_grid_with_311_counts_geocoded.geojson")
# File to save our geocoding results so we don't have to run it again!
GEOCODED_CACHE_FILE = os.path.join(SCRIPT_DIR, "geocoded_intersections.csv")

# --- Step 1 & 2: Create the Master Grid ---
print("Step 1 & 2: Creating the master 250x250m grid...")
# Try to read the shapefile. If .shx is missing, the environment variable may help.
toronto_boundary: gpd.GeoDataFrame = gpd.read_file(TORONTO_BOUNDARY_SHP)

# Ensure the shapefile was loaded successfully
if toronto_boundary is None or toronto_boundary.empty:
    raise ValueError(f"Failed to load shapefile: {TORONTO_BOUNDARY_SHP}")

# The shapefile may not have CRS information embedded. Set it explicitly.
# Based on the filename, this appears to be WGS84 (standard lat/lon coordinates).
if toronto_boundary.crs is None:
    toronto_boundary = toronto_boundary.set_crs(epsg=4326)
    print("  - Set CRS to WGS84 (EPSG:4326)")

toronto_utm = toronto_boundary.to_crs(epsg=32617)
xmin, ymin, xmax, ymax = toronto_utm.total_bounds
cell_size = 250
grid_cells = []
for x in np.arange(xmin, xmax, cell_size):
    for y in np.arange(ymin, ymax, cell_size):
        grid_cells.append(Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)]))
grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs="EPSG:32617")

# Use gpd.overlay for intersection, as gpd.clip is deprecated.
grid = gpd.overlay(grid, toronto_utm, how='intersection')

# Add a unique ID to each cell after the overlay operation
grid['cell_id'] = range(len(grid))

# Keep only the columns we need
grid = grid[['cell_id', 'geometry']]

print("  - Grid created successfully.")


# --- Step 3: Load 311 Data and Geocode Intersections ---
print("\nStep 3: Loading 311 data and geocoding...")
df_311 = pd.read_csv(SERVICE_REQUESTS_CSV)
df_311.dropna(subset=['Intersection Street 1', 'Intersection Street 2'], inplace=True)
print(f"  - Loaded {len(df_311)} service requests with intersection data.")

# **THE OPTIMIZATION STRATEGY**
# 1. Find the unique intersections to minimize API calls
unique_intersections = df_311[['Intersection Street 1', 'Intersection Street 2']].drop_duplicates()
print(f"  - Found {len(unique_intersections)} unique intersections to geocode.")

# TESTING: Limit to first 20 intersections for testing
#unique_intersections = unique_intersections.head(20)
#print(f"  - Limited to {len(unique_intersections)} intersections for testing.")

# 2. Set up the geocoder
# The user_agent is required by Nominatim's terms of service. Name it after your app.
geolocator = Nominatim(user_agent="urban_sentinel_hackathon")

# 3. Geocode each unique intersection and store results in a dictionary
# tqdm will wrap around our loop to create a progress bar
geocoded_results = {}
tqdm.pandas(desc="Geocoding progress")

def geocode_intersection(row):
    try:
        # Construct the query string
        query = f"{row['Intersection Street 1']} & {row['Intersection Street 2']}, Toronto, ON, Canada"
        
        # geopy can be async. We must check if a coroutine is returned before running it.
        location_coro = geolocator.geocode(query, timeout=10)
        location = None
        if asyncio.iscoroutine(location_coro):
            location = asyncio.run(location_coro)
        else:
            location = location_coro # It was a synchronous result

        # Add a small delay to respect API rate limits
        time.sleep(1) 
        if location:
            return (location.latitude, location.longitude)
        else:
            return (None, None)
    except Exception as e:
        # print(f"Error geocoding {query}: {e}")
        return (None, None)

# Apply the geocoding function to the UNIQUE intersections DataFrame
unique_intersections[['LATITUDE', 'LONGITUDE']] = unique_intersections.progress_apply(geocode_intersection, axis=1, result_type="expand")

# 4. Save the geocoded results to a cache file
print(f"\n  - Saving geocoded results to '{GEOCODED_CACHE_FILE}' to save time later.")
unique_intersections.to_csv(GEOCODED_CACHE_FILE, index=False)

# 5. Merge the coordinates back into the original, full 311 DataFrame
df_311 = pd.merge(df_311, unique_intersections, on=['Intersection Street 1', 'Intersection Street 2'], how='left')

# --- Step 4 & onwards: Proceed exactly as before ---
print("\nStep 4: Converting to GeoDataFrame and performing spatial join...")

# Now we have the LATITUDE and LONGITUDE columns, we can continue as planned
df_311.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True) # Drop rows that failed geocoding
gdf_311 = gpd.GeoDataFrame(
    df_311,
    geometry=gpd.points_from_xy(df_311.LONGITUDE, df_311.LATITUDE),
    crs="EPSG:4326"
)
gdf_311_utm = gdf_311.to_crs(grid.crs)

# The rest of the script is identical...
joined_gdf = gpd.sjoin(gdf_311_utm, grid, how="inner", predicate="within")
complaints_per_cell = joined_gdf.groupby('cell_id').size().reset_index(name='complaint_count_2022')
grid_with_counts = grid.merge(complaints_per_cell, on='cell_id', how='left')
grid_with_counts['complaint_count_2022'].fillna(0, inplace=True)
grid_with_counts['complaint_count_2022'] = grid_with_counts['complaint_count_2022'].astype(int)

# --- Final Step: Save the output ---
print(f"\nSaving final output to '{OUTPUT_FILENAME}'...")
grid_final_web = grid_with_counts.to_crs("EPSG:4326")
grid_final_web.to_file(OUTPUT_FILENAME, driver='GeoJSON')
print("\n--- Process Complete! ---")