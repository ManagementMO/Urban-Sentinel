import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import numpy as np


# --- Constants ---
# The boundary file for the city of Toronto
TORONTO_BOUNDARY_SHP = "citygcs_regional_mun_wgs84.shp"
SERVICE_REQUESTS_CSV = "" # e.g., "311_Service_Requests.csv"
OUTPUT_FILENAME = "toronto_grid_with_311_counts.geojson"


# --- Step 1: Create the Master Grid ---
print("Step 1: Creating the master 250x250m grid...")

# 1. Load the shapefile for Toronto's boundary
toronto_boundary = gpd.read_file(TORONTO_BOUNDARY_SHP)

# 2. Project the boundary to a metric Coordinate Reference System (CRS)
# WGS84 (EPSG:4326) uses degrees, which is bad for measuring meters.
# UTM Zone 17N (EPSG:32617) is a CRS that uses meters and is accurate for Toronto.
# This step is absolutely CRITICAL for creating a grid with meter-based dimensions.
print("  - Projecting boundary to UTM (EPSG:32617)...")
toronto_utm = toronto_boundary.to_crs(epsg=32617)

# 3. Get the bounding box coordinates of the projected city area
xmin, ymin, xmax, ymax = toronto_utm.total_bounds

# 4. Define the grid cell size in meters
cell_size = 250  # 250 meters

# 5. Create a list of square Polygons that form the grid
grid_cells = []
for x in np.arange(xmin, xmax, cell_size):
    for y in np.arange(ymin, ymax, cell_size):
        grid_cells.append(Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)]))

# 6. Create the GeoDataFrame from the list of polygons
print(f"  - Generated {len(grid_cells)} potential grid cells...")
grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs="EPSG:32617")

# 7. Add a unique ID to each cell for easier joins later
grid['cell_id'] = range(len(grid))

# 8. Intersect the grid with the actual Toronto boundary
# This removes unnecessary cells that are outside the city's shape (e.g., in Lake Ontario)
# Note: gpd.clip is deprecated. The modern approach is gpd.overlay with how='intersection'.
grid = gpd.overlay(grid, toronto_utm, how="intersection")

# The overlay operation adds columns from the toronto_utm boundary file.
# To replicate the behavior of the old `clip` function, we only keep the original columns.
grid = grid[['cell_id', 'geometry']]

print(f"  - Final grid created with {len(grid)} cells covering Toronto.")


# --- Step 2: Load and Prepare 311 Data ---
print("\nStep 2: Loading and preparing 311 data...")

# 1. Load the 311 data CSV file using pandas
# This may take a moment and consume a significant amount of memory.
# Note: You must update the SERVICE_REQUESTS_CSV constant at the top of the file.
if not SERVICE_REQUESTS_CSV:
    print("  - WARNING: SERVICE_REQUESTS_CSV is not set. Skipping 311 data processing.")
else:
    df_311 = pd.read_csv(SERVICE_REQUESTS_CSV)
    print(f"  - Loaded {len(df_311)} service requests.")

    # 2. Filter out requests that don't have location data
    df_311.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
    print(f"  - {len(df_311)} requests remain after removing rows with no location.")

    # 3. Convert the pandas DataFrame into a GeoDataFrame
    # This tells geopandas that the 'LATITUDE' and 'LONGITUDE' columns represent geographic points.
    gdf_311 = gpd.GeoDataFrame(
        df_311,
        geometry=gpd.points_from_xy(df_311.LONGITUDE, df_311.LATITUDE),
        crs="EPSG:4326"  # The 311 data is in standard WGS84 lat/lon
    )

    # 4. Project the 311 data to the SAME CRS as our grid
    # The spatial join will fail if the CRSs do not match.
    print("  - Projecting 311 data points to UTM...")
    gdf_311_utm = gdf_311.to_crs(grid.crs)  # grid.crs is "EPSG:32617"

    # --- Step 3: Perform the Spatial Join ---
    print("\nStep 3: Spatially joining 311 complaints to the grid...")

    # This operation is like a VLOOKUP or JOIN in SQL, but based on location instead of text.
    # For each point in gdf_311_utm, it finds the polygon in 'grid' that contains it.
    # 'how="inner"' means we only keep the points that fall within a grid cell.
    # 'op="within"' is the spatial relationship we are checking for.
    joined_gdf = gpd.sjoin(gdf_311_utm, grid, how="inner", op="within")

    print("  - Spatial join complete.")

    # --- Step 4: Aggregate Counts and Merge ---
    print("\nStep 4: Aggregating counts per cell...")

    # 1. Group the joined data by the grid cell's unique ID and count the number of entries
    complaints_per_cell = joined_gdf.groupby('cell_id').size().reset_index(name='complaint_count')
    print(f"  - Found {len(complaints_per_cell)} cells with at least one complaint.")

    # 2. Merge these counts back into our master grid DataFrame
    # We use a 'left' merge to ensure we keep ALL grid cells, even those with zero complaints.
    grid_with_counts = grid.merge(complaints_per_cell, on='cell_id', how='left')

    # 3. Clean up the result: Fill cells with no complaints with 0 instead of NaN (Not a Number)
    grid_with_counts['complaint_count'].fillna(0, inplace=True)
    grid_with_counts['complaint_count'] = grid_with_counts['complaint_count'].astype(int)

    print("  - Counts merged back into the master grid.")

    # --- Step 5: Save the Final GeoJSON File ---
    print(f"\nStep 5: Saving the final output to '{OUTPUT_FILENAME}'...")

    # 1. Project the final grid back to WGS84 (lat/lon)
    # This is the standard format for most web mapping tools like Mapbox.
    grid_final_web = grid_with_counts.to_crs("EPSG:4326")

    # 2. Save to GeoJSON
    grid_final_web.to_file(OUTPUT_FILENAME, driver='GeoJSON')

    print("\n--- Process Complete! ---")
    print(f"You can now find your map file at: {OUTPUT_FILENAME}")

# This `grid` GeoDataFrame is now your master file.
# **What you have now:** A `GeoDataFrame` named `grid` where each row represents a 250x250m square within Toronto's borders.
# **What you can do with it:**
# - Save it to a shapefile for use in other GIS software
# - Use it to spatially join other datasets (e.g., crime data)
# - Analyze urban patterns
# - Visualize the grid
