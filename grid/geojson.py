import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import os

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# --- File Paths ---
# This makes the script portable and easy to run.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_311_CSV = os.path.join(SCRIPT_DIR, "output_with_coords.csv")
TORONTO_BOUNDARY_SHP = os.path.join(SCRIPT_DIR, "citygcs_regional_mun_wgs84.shp")
OUTPUT_FILENAME = "model_ready_temporal_features.geojson"

# --- Analysis Years (CRITICAL CHANGE) ---
# We use the history from 2014-2019 to generate features.
FEATURE_START_YEAR = 2014
FEATURE_END_YEAR = 2019
# We will predict the outcome for 2020, as this is the last year of data we have.
TARGET_YEAR = 2020

# --- Blight Definition (OPTIMIZED FOR ML) ---
# Carefully curated list of complaint types that are STRONG indicators of urban blight.
# Based on urban planning research and data science best practices.
# These directly correlate with neighborhood decline and property value deterioration.
BLIGHT_INDICATOR_COMPLAINTS = [
    # === PHYSICAL DETERIORATION (Core Blight Indicators) ===
    'Road - Pot hole',                              # Infrastructure decay
    'Road - Damaged',                               # Infrastructure failure
    'Road - Sinking',                               # Severe infrastructure issues
    'Sidewalk - Damaged / Concrete',                # Pedestrian infrastructure decay
    'Sidewalk - Damaged /Brick/Interlock',          # Pedestrian infrastructure decay
    
    # === VISUAL BLIGHT & NEGLECT ===
    'Graffiti',                                     # Visual blight indicator
    'Graffiti - Private Property',                  # Property neglect
    'Graffiti - Public Property',                   # Public space neglect
    'Road - Graffiti Complaint',                    # Infrastructure vandalism
    'Sidewalk - Graffiti Complaint',                # Pedestrian area vandalism
    'Traffic Sign - Graffiti Complaint',            # Public infrastructure vandalism
    
    # === WASTE & SANITATION ISSUES ===
    'Litter / Bin / Overflow or Not Picked Up',     # Sanitation breakdown
    'Litter / Illegal Dumping Cleanup',             # Illegal waste disposal
    'Illegal Dumping',                              # Environmental neglect
    'Illegal Dumping / Discharge',                  # Environmental contamination
    'Garbage Collection - Missed Pick-Up',          # Service breakdown
    'Litter / Sidewalk & Blvd / Pick Up Request',   # Public space cleanliness
    
    # === PROPERTY NEGLECT ===
    'Long Grass and Weeds',                         # Property abandonment indicator
    'Property Standards',                           # Code violations
    'Construction-Unsafe/Untidy Condition',         # Dangerous/neglected construction
    'Complaint / Investigation - Grass and Weeds Enforcement', # Property maintenance violations
    
    # === INFRASTRUCTURE FAILURE ===
    'Catch Basin - Blocked / Flooding',             # Drainage system failure
    'Catch Basin - Damaged Maintenance Requested',  # Infrastructure deterioration
    'Catch basin (Storm) - Damage',                 # Storm system failure
    'Sewer main-Backup',                            # Sanitation system failure
    'Sewer Service Line-Blocked',                   # Individual property sanitation issues
    'Street Light Out',                             # Public safety infrastructure failure
    
    # === ABANDONED/DERELICT CONDITIONS ===
    'Complaint/Investigation -Abandoned Bikes',      # Abandonment indicators
    'Dead Animal On Expressway',                    # Maintenance neglect
    'Dangerous Private Tree Investigation',          # Property safety hazards
    
    # === PUBLIC SAFETY DETERIORATION ===
    'Sink Hole',                                    # Severe infrastructure danger
    'Bridge - Damaged Structure',                   # Critical infrastructure failure
    'Fence - Damaged',                              # Property security breakdown
    'Guardrail - Damaged',                          # Safety infrastructure failure
]

# The percentile to define a "blighted" area. 0.90 means the top 10%.
BLIGHT_QUANTILE_THRESHOLD = 0.90


# ==============================================================================
# --- SCRIPT EXECUTION ---
# ==============================================================================

def create_temporal_features(yearly_data, grid):
    """Creates rich temporal features from yearly complaint data."""
    print("  - Creating temporal features...")
    
    all_years = list(range(FEATURE_START_YEAR, FEATURE_END_YEAR + 1))
    all_cells = grid['cell_id'].unique()
    
    # Create a complete template of all cells for all years
    year_cell_combinations = pd.MultiIndex.from_product(
        [all_cells, all_years], names=['cell_id', 'year']
    ).to_frame(index=False)
    
    complete_data = year_cell_combinations.merge(yearly_data, on=['cell_id', 'year'], how='left').fillna(0)
    complete_data = complete_data.sort_values(['cell_id', 'year'])
    
    features_list = []
    
    # Group by cell_id to perform vectorized operations where possible
    grouped = complete_data.groupby('cell_id')

    # Basic stats
    means = grouped[['total_complaints', 'blight_complaints']].mean().add_suffix('_mean')
    stds = grouped[['total_complaints', 'blight_complaints']].std().add_suffix('_std').fillna(0)
    sums = grouped[['total_complaints', 'blight_complaints']].sum().add_suffix('_sum')
    
    base_features = pd.concat([means, stds, sums], axis=1)

    for cell_id, cell_data in grouped:
        features = {'cell_id': cell_id}
        
        # Trend over the entire period (using polyfit for slope)
        years = cell_data['year'].values
        if len(years) > 1:
            features['total_complaints_trend'] = np.polyfit(years, cell_data['total_complaints'], 1)[0]
            features['blight_complaints_trend'] = np.polyfit(years, cell_data['blight_complaints'], 1)[0]
        else:
            features['total_complaints_trend'] = 0
            features['blight_complaints_trend'] = 0

        # Recent trend (last 3 years of feature data)
        recent_data = cell_data[cell_data['year'] >= FEATURE_END_YEAR - 2]
        if len(recent_data) > 1:
            features['total_complaints_recent_trend'] = np.polyfit(recent_data['year'], recent_data['total_complaints'], 1)[0]
            features['blight_complaints_recent_trend'] = np.polyfit(recent_data['year'], recent_data['blight_complaints'], 1)[0]
        else:
            features['total_complaints_recent_trend'] = 0
            features['blight_complaints_recent_trend'] = 0
            
        # Lagged features (values from the most recent years)
        for year_lag in range(3): # Get last 3 years
            year = FEATURE_END_YEAR - year_lag
            lag_data = cell_data[cell_data['year'] == year]
            features[f'total_complaints_{year}'] = lag_data['total_complaints'].iloc[0] if not lag_data.empty else 0
            features[f'blight_complaints_{year}'] = lag_data['blight_complaints'].iloc[0] if not lag_data.empty else 0
            
        features_list.append(features)
        
    temporal_features = pd.DataFrame(features_list)
    final_features = base_features.merge(temporal_features, on='cell_id', how='left').fillna(0)

    print(f"  - Created {len(final_features.columns) - 1} temporal features for {len(final_features)} cells.")
    return final_features


def run_data_processing():
    """Main function to run the entire temporal data processing pipeline."""

    print("--- Step 1: Creating the master 250x250m grid... ---")
    toronto_boundary = gpd.read_file(TORONTO_BOUNDARY_SHP)
    if toronto_boundary.crs is None:
        toronto_boundary = toronto_boundary.set_crs(epsg=4326)
    toronto_utm = toronto_boundary.to_crs(epsg=32617)
    xmin, ymin, xmax, ymax = toronto_utm.total_bounds
    cell_size = 250
    grid_cells = [Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)])
                  for x in np.arange(xmin, xmax, cell_size) for y in np.arange(ymin, ymax, cell_size)]
    grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs="EPSG:32617").pipe(gpd.clip, toronto_utm)
    grid['cell_id'] = range(len(grid))
    print(f"  - Grid created successfully with {len(grid)} cells.")

    print("\n--- Step 2: Loading and preparing consolidated 311 data... ---")
    df_311 = pd.read_csv(RAW_311_CSV, low_memory=False)
    df_311.columns = df_311.columns.str.strip()
    df_311['Creation Date'] = pd.to_datetime(df_311['Creation Date'])
    df_311['year'] = df_311['Creation Date'].dt.year
    valid_coords_mask = df_311['latitude'].notna() & df_311['longitude'].notna()
    gdf_311 = gpd.GeoDataFrame(df_311[valid_coords_mask], geometry=gpd.points_from_xy(df_311[valid_coords_mask].longitude, df_311[valid_coords_mask].latitude), crs="EPSG:4326")
    gdf_311_utm = gdf_311.to_crs(grid.crs)
    print("  - Data successfully loaded and projected.")

    print(f"\n--- Step 3: Aggregating historical data ({FEATURE_START_YEAR}-{FEATURE_END_YEAR})... ---")
    historical_data = gdf_311_utm[gdf_311_utm['year'].isin(range(FEATURE_START_YEAR, FEATURE_END_YEAR + 1))].copy()
    
    # Perform ONE spatial join for all historical data - much more efficient
    joined_historical = gpd.sjoin(historical_data, grid, how="inner", predicate="within")

    # Aggregate total complaints by cell and year
    total_agg = joined_historical.groupby(['cell_id', 'year']).size().reset_index(name='total_complaints')
    
    # Aggregate blight complaints by cell and year
    blight_agg = joined_historical[joined_historical['Service Request Type'].isin(BLIGHT_INDICATOR_COMPLAINTS)] \
        .groupby(['cell_id', 'year']).size().reset_index(name='blight_complaints')

    # Combine total and blight yearly data
    yearly_data = total_agg.merge(blight_agg, on=['cell_id', 'year'], how='left').fillna(0)
    print("  - Yearly aggregated data created.")

    print("\n--- Step 4: Creating rich temporal features for ML... ---")
    temporal_features_df = create_temporal_features(yearly_data, grid)

    print(f"\n--- Step 5: Creating target variable for {TARGET_YEAR} prediction... ---")
    target_data = gdf_311_utm[gdf_311_utm['year'] == TARGET_YEAR].copy()
    target_complaints = target_data[target_data['Service Request Type'].isin(BLIGHT_INDICATOR_COMPLAINTS)]
    joined_target = gpd.sjoin(target_complaints, grid, how="inner", predicate="within")
    target_counts = joined_target.groupby('cell_id').size().reset_index(name='target_blight_count')
    target_df = grid[['cell_id']].merge(target_counts, on='cell_id', how='left').fillna(0)
    blight_threshold = target_df['target_blight_count'].quantile(BLIGHT_QUANTILE_THRESHOLD)
    target_df['is_blighted'] = (target_df['target_blight_count'] > blight_threshold).astype(int)
    print(f"  - Blight threshold: > {blight_threshold:.1f} complaints per cell.")
    print(f"  - Cells flagged as blighted in {TARGET_YEAR}: {target_df['is_blighted'].sum()} ({target_df['is_blighted'].mean()*100:.1f}%)")

    print("\n--- Step 6: Finalizing and saving the dataset... ---")
    final_data = temporal_features_df.merge(target_df, on='cell_id', how='left').fillna(0)
    final_grid = grid[['cell_id', 'geometry']].merge(final_data, on='cell_id', how='left')
    final_grid_web = final_grid.to_crs("EPSG:4326")
    final_grid_web.to_file(OUTPUT_FILENAME, driver='GeoJSON')
    
    print(f"\n==================== PROCESS COMPLETE ====================")
    print(f"✅ Model-ready temporal dataset saved: {OUTPUT_FILENAME}")
    print(f"📈 Training period: {FEATURE_START_YEAR}-{FEATURE_END_YEAR} | Target year: {TARGET_YEAR}")
    print("✅ Ready for the next step: Model Training!")

if __name__ == '__main__':
    run_data_processing()