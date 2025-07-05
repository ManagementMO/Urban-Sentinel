import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import os


# --- Configuration ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_311_CSV = os.path.join(SCRIPT_DIR, "output_with_coords.csv") # The file you have
TORONTO_BOUNDARY_SHP = os.path.join(SCRIPT_DIR, "citygcs_regional_mun_wgs84.shp")
OUTPUT_FILENAME = "toronto_grid_with_temporal_features.geojson"

# --- Analysis Years ---
HISTORY_START_YEAR = 2014
HISTORY_END_YEAR = 2019
TARGET_YEAR = 2020

# --- Blight Definition (OPTIMIZED FOR ML) ---
# Carefully curated list of complaint types that are STRONG indicators of urban blight.
# Based on urban planning research and data science best practices.
# These directly correlate with neighborhood decline and property value deterioration.
BLIGHT_INDICATOR_COMPLAINTS = [
    'Road - Pot hole',
    'Traffic Signal Maintenance',
    'CADAVER WILDLIFE',
    'Missing/Damaged Signs',
    'Road - Cleaning/Debris',
    'Litter / Sidewalk & Blvd / Pick Up Request',
    'INJUR/DIST WILDLIFE',
    'Road - Damaged',
    'PXO Maintenance',
    'Litter / Bin / Overflow or Not Picked Up',
]

# The percentile to define a "blighted" area. 0.85 means the top 15%.
BLIGHT_QUANTILE_THRESHOLD = 0.85


# ==============================================================================
# --- SCRIPT EXECUTION ---
# ==============================================================================

def create_temporal_features(yearly_data, grid):
    # ... (This entire function is the same as in your original script)
    print("  - Creating temporal features...")
    all_years = list(range(HISTORY_START_YEAR, HISTORY_END_YEAR + 1))
    all_cells = grid['cell_id'].unique()
    year_cell_combinations = pd.MultiIndex.from_product([all_cells, all_years], names=['cell_id', 'year']).to_frame(index=False)
    complete_data = year_cell_combinations.merge(yearly_data, on=['cell_id', 'year'], how='left').fillna(0)
    complete_data = complete_data.sort_values(['cell_id', 'year'])
    features_list = []
    for cell_id in all_cells:
        cell_data = complete_data[complete_data['cell_id'] == cell_id].copy().sort_values('year')
        features = {'cell_id': cell_id}
        features['total_complaints_mean'] = cell_data['total_complaints'].mean()
        features['total_complaints_std'] = cell_data['total_complaints'].std()
        features['blight_complaints_mean'] = cell_data['blight_complaints'].mean()
        features['blight_complaints_std'] = cell_data['blight_complaints'].std()
        recent_data = cell_data[cell_data['year'] >= 2018]
        features['blight_complaints_recent_mean'] = recent_data['blight_complaints'].mean()
        if len(cell_data) > 1:
            features['blight_complaints_trend'] = np.polyfit(cell_data['year'], cell_data['blight_complaints'], 1)[0]
        else:
            features['blight_complaints_trend'] = 0
        year_latest_data = cell_data[cell_data['year'] == HISTORY_END_YEAR]
        features[f'blight_complaints_{HISTORY_END_YEAR}'] = year_latest_data['blight_complaints'].mean() if len(year_latest_data) > 0 else 0
        features_list.append(features)
    features_df = pd.DataFrame(features_list).fillna(0)
    print(f"  - Created temporal features for {len(features_df)} cells")
    return features_df

def run_data_processing():
    """Main function to run the entire temporal data processing pipeline."""
    print("--- Step 1: Creating the master 250x250m grid... ---")
    if not os.path.exists(TORONTO_BOUNDARY_SHP):
        print(f"ERROR: Boundary shapefile not found at '{TORONTO_BOUNDARY_SHP}'.")
        return
    toronto_boundary = gpd.read_file(TORONTO_BOUNDARY_SHP)
    if toronto_boundary.crs is None:
        toronto_boundary = toronto_boundary.set_crs(epsg=4326)
    toronto_utm = toronto_boundary.to_crs(epsg=32617)
    xmin, ymin, xmax, ymax = toronto_utm.total_bounds
    cell_size = 250
    grid_cells = [Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)]) for x in np.arange(xmin, xmax, cell_size) for y in np.arange(ymin, ymax, cell_size)]
    grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs="EPSG:32617")
    grid = gpd.clip(grid, toronto_utm)
    grid['cell_id'] = range(len(grid))
    print(f"  - Grid created with {len(grid)} cells.")

    print("\n--- Step 2: Loading and preparing consolidated 311 data... ---")
    if not os.path.exists(RAW_311_CSV):
        print(f"ERROR: 311 data CSV not found at '{RAW_311_CSV}'.")
        return
    df_311 = pd.read_csv(RAW_311_CSV)
    df_311.columns = df_311.columns.str.strip()
    df_311['Creation Date'] = pd.to_datetime(df_311['Creation Date'])
    df_311['year'] = df_311['Creation Date'].dt.year
    valid_coords_mask = (df_311['longitude'].notna() & df_311['latitude'].notna() & (df_311['longitude'].between(-180, 180)) & (df_311['latitude'].between(-90, 90)))
    df_311_clean = df_311[valid_coords_mask].copy()
    gdf_311 = gpd.GeoDataFrame(df_311_clean, geometry=gpd.points_from_xy(df_311_clean.longitude, df_311_clean.latitude), crs="EPSG:4326")
    gdf_311_utm = gdf_311.to_crs(grid.crs)
    print("  - Data loaded and projected.")

    historical_data = gdf_311_utm[(gdf_311_utm['year'] >= HISTORY_START_YEAR) & (gdf_311_utm['year'] <= HISTORY_END_YEAR)].copy()
    
    # --- NEW SECTION START: Calculate Most Common Complaint Features ---
    print("\n--- Step 2.5: Calculating most common complaint features... ---")
    
    # Filter for only blight-related complaints from the historical data
    historical_blight_data = historical_data[historical_data['Service Request Type'].isin(BLIGHT_INDICATOR_COMPLAINTS)].copy()
    
    # Spatially join all historical blight data with the grid
    joined_blight_all_years = gpd.sjoin(historical_blight_data, grid, how="inner", predicate="within")
    
    # Calculate overall most common blight complaint (historical period)
    # The .mode()[0] gets the most frequent item. We handle cases where a cell has no blight complaints.
    overall_common = joined_blight_all_years.groupby('cell_id')['Service Request Type'].apply(lambda x: x.mode()[0] if not x.empty else "None").reset_index(name='overall_most_common_blight')
    
    # Calculate most common blight complaint for the most recent year of historical data
    recent_blight_data = joined_blight_all_years[joined_blight_all_years['year'] == HISTORY_END_YEAR]
    recent_common = recent_blight_data.groupby('cell_id')['Service Request Type'].apply(lambda x: x.mode()[0] if not x.empty else "None").reset_index(name='recent_most_common_blight')
    
    # Merge these new features together
    common_complaint_features = pd.merge(overall_common, recent_common, on='cell_id', how='outer')
    print(f"  - Calculated most common complaint features for {len(common_complaint_features)} cells.")
    # --- NEW SECTION END ---

    print(f"\n--- Step 3: Creating yearly aggregated data ({HISTORY_START_YEAR}-{HISTORY_END_YEAR})... ---")
    yearly_data_list = []
    for year in range(HISTORY_START_YEAR, HISTORY_END_YEAR + 1):
        year_data = historical_data[historical_data['year'] == year].copy()
        if len(year_data) == 0:
            continue
        joined_data = gpd.sjoin(year_data, grid, how="inner", predicate="within")
        total_counts = joined_data.groupby('cell_id').size().reset_index(name='total_complaints')
        blight_data = joined_data[joined_data['Service Request Type'].isin(BLIGHT_INDICATOR_COMPLAINTS)]
        blight_counts = blight_data.groupby('cell_id').size().reset_index(name='blight_complaints')
        year_counts = total_counts.merge(blight_counts, on='cell_id', how='left').fillna(0)
        year_counts['year'] = year
        yearly_data_list.append(year_counts)
    yearly_data = pd.concat(yearly_data_list, ignore_index=True)
    print(f"  - Created yearly aggregated data.")

    print("\n--- Step 4: Creating rich temporal features for ML... ---")
    temporal_features = create_temporal_features(yearly_data, grid)

    print(f"\n--- Step 5: Creating target variable for {TARGET_YEAR} prediction... ---")
    target_data = gdf_311_utm[gdf_311_utm['year'] == TARGET_YEAR].copy()
    if len(target_data) == 0:
        print(f"  - WARNING: No data for target year {TARGET_YEAR}. Cannot create target variable.")
        target_df = pd.DataFrame({'cell_id': grid['cell_id'], 'target_blight_count': 0, 'is_blighted': 0})
    else:
        target_complaints = target_data[target_data['Service Request Type'].isin(BLIGHT_INDICATOR_COMPLAINTS)]
        joined_target = gpd.sjoin(target_complaints, grid, how="inner", predicate="within")
        target_counts = joined_target.groupby('cell_id').size().reset_index(name='target_blight_count')
        target_df = pd.merge(pd.DataFrame({'cell_id': grid['cell_id']}), target_counts, on='cell_id', how='left').fillna(0)
        blight_threshold = target_df['target_blight_count'].quantile(BLIGHT_QUANTILE_THRESHOLD)
        target_df['is_blighted'] = (target_df['target_blight_count'] > blight_threshold).astype(int)
        print(f"  - Blight threshold: > {blight_threshold:.1f} complaints. Cells flagged: {target_df['is_blighted'].sum()}")

    print("\n--- Step 6: Combining features with target and creating final dataset... ---")
    # Merge temporal features with target
    final_data = temporal_features.merge(target_df, on='cell_id', how='left')
    
    # --- MODIFIED SECTION: Merge the new common complaint features ---
    final_data = final_data.merge(common_complaint_features, on='cell_id', how='left')
    # Fill cells that had no blight complaints with "None"
    final_data['overall_most_common_blight'] = final_data['overall_most_common_blight'].fillna("None")
    final_data['recent_most_common_blight'] = final_data['recent_most_common_blight'].fillna("None")
    # --- END MODIFIED SECTION ---
    
    final_grid = grid.merge(final_data, on='cell_id', how='left')
    final_grid['is_blighted'] = final_grid['is_blighted'].fillna(0)
    print(f"  - Final dataset: {len(final_grid)} cells with {len(final_data.columns)-3} features")

    print("\n--- Step 7: Saving the model-ready dataset... ---")
    final_grid_web = final_grid.to_crs("EPSG:4326")
    final_grid_web.to_file(OUTPUT_FILENAME, driver='GeoJSON')
    
    print(f"\n==================== PROCESS COMPLETE ====================")
    print(f"🎯 Model-ready dataset saved: {OUTPUT_FILENAME}")
    print("✅ Now includes 'overall_most_common_blight' and 'recent_most_common_blight' fields.")

if __name__ == '__main__':
    run_data_processing()