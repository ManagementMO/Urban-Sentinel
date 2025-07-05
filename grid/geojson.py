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

# --- Blight Definition (CRITICAL CHANGE) ---
# A curated, high-signal list of complaint types that strongly indicate blight.
# This is MUCH more effective than using all complaint types.
BLIGHT_INDICATOR_COMPLAINTS = [
    'Access/AODA Complaint',
    'All / Hazardous Waste / Pick Up Request',
    'All-Way Stop Sign Controls',
    'Alternate Side Parking',
    'Animals',
    'Appliance (Emergency)',
    'Appliance Doors On',
    'Application Mail Out / Non-Res',
    'Bees/Wasp',
    'Bicycle Signals',
    'Bin Investigation Request',
    'Blocked Access By Parking',
    'Bollard - Damaged',
    'Boulevard - Leaf Pick-up Mechanical',
    'Boulevard - Pick-Up Shopping Carts',
    'Boulevard - Plough Damage',
    'Boulevards - Damaged Asphalt',
    'Boulevards - Snow Piled Too High / Too Much',
    'Boulevards - Weed Removal',
    'Boulevards-Grass Cutting',
    'Bridge - Damaged Structure',
    'Bridge - Debris / Litter',
    'Bridge - Graffiti Complaint',
    'Bridge - Surface Repair',
    'Bridge Falling Debris',
    'Bridge Icy Needs Sand/Salt',
    'Bus Stop Icy Needs Sand/Salt',
    'Bus Stops Snow Clearing Required',
    'Business Complaint',
    'By-Law Contravention Invest',
    'Bylaw Enforcement: Excavation',
    'CADAVER DOMESTIC',
    'CADAVER WILDLIFE',
    'Catch Basin - Blocked / Flooding',
    'Catch Basin - Damaged Maintenance Requested',
    'Catch Basin - Debris / Litter',
    'Catch Basin -Cover Missing / Damaged / Loose',
    'Catch basin (Storm) - Damage',
    'Catch basin (Storm) - Other',
    'Catch basin (Storm) - Overflowing',
    'Catch basin Maintenance and Repair',
    'Catch basin on Expressway damaged',
    'Catch basin on Expressway requires cleaning',
    'Clothing Drop Boxes',
    'Comment / Suggestion',
    'Commercial Dog Walkers',
    'Commercial Enterprises',
    'Commercial Loading Zone',
    'Commercial Tree Maint Req',
    'Commercial Tree Planting',
    'Commercial Tree Pruning',
    'Commercial Tree Removal',
    'Commercial Tree Storm Clean Up',
    'Complaint - Crossing Guard Conduct',
    'Complaint - Staff / Equipment / Attitude / Behaviour',
    'Complaint - Vendor SCG no Show',
    'Complaint / Investigation - Grass and Weeds Enforcement',
    'Complaint / Investigation - Idling Enforcement',
    'Complaint / Investigation - Leaves',
    'Complaint / Investigation - Water Discharge',
    'Complaint / Property Damaged',
    'Complaint regarding Contractor',
    'Complaint-Access',
    'Complaint-Disability',
    'Complaint-Outcome of the Service',
    'Complaint-Process and Procedures',
    'Complaint-Staff Conduct',
    'Complaint-Time Line of the Service',
    'Complaint/Investigation - Encroachment',
    'Complaint/Investigation -Abandoned Bikes',
    'Complaint/Investigation -Illegal Parking',
    'Compliment-Employee/Operation',
    'Conduct',
    'Construction-Unsafe/Untidy Condition',
    'Containers',
    'Contaminated Waste/Preparation',
    'Corner Parking Prohibition',
    'Culverts - Blocked',
    'Culverts-Damaged / Maintenance Requested',
    'Curb - Adjust Height (Too High/Low)',
    'Curb - Damaged',
    'Dangerous Private Tree Investigation',
    'Dead Animal On Expressway',
    'Disabled Loading Zone',
    "Disabled Persons' Parking Space",
    'Dispute SR Status/Bins',
    'Dispute SR Status/Collections Curb Day',
    'Dispute SR Status/Collections FEL',
    'Dispute SR Status/Collections Nights',
    'Dispute SR Status/Litter Operations',
    'District Operations-Attitude and Behavior',
    'District Operations-Compliment',
    'District Operations-Construction Site Unsafe',
    'District Operations-Contractor Related',
    'District Operations-Equipment/Untidy Const Site',
    'District Operations-Process',
    'District Operations-Restoration',
    'District Operations-Timeliness',
    'Ditch Maintenance Requested',
    'Dogs off Leash',
    'Driveway - Damaged / Ponding',
    'Driveway-Blocked By Windrow',
    'EAB Exemption Request',
    'ENF/INVEST AN TO AN',
    'ENF/INVEST AN TO HU',
    'ENF/INVEST ANIM CARE',
    'ENF/INVEST ATTACK TO AN',
    'ENF/INVEST ATTACK TO HU',
    'ENF/INVEST COLLAR',
    'ENF/INVEST DAL HOME',
    'ENF/INVEST DOLA',
    'ENF/INVEST EXCREMENT',
    'ENF/INVEST EXTREME',
    'ENF/INVEST LICENCE',
    'ENF/INVEST MENACE',
    'ENF/INVEST MUZZLE',
    'ENF/INVEST NO LEASH',
    'ENF/INVEST NOISE',
    'ENF/INVEST NUISAN CAT',
    'ENF/INVEST PIGEONS',
    'ENF/INVEST PIT BULL',
    'ENF/INVEST PROH ANIMA',
    'ENF/INVEST SHELTER',
    'ENF/INVEST TETHER',
    'ENF/INVEST TIED EXCES',
    'ENF/INVEST TOO MANY',
    'ENF/INVEST UNSAN COND',
    'ENF/INVEST WALK MANY',
    'Election Signs',
    'Employee Comment',
    'Encroachments',
    'Expressway Fence - Damaged',
    'Expressway Guide Rail Damaged',
    'Expressway requires cleaning.',
    'FEL Non-Res / Garbage / Not Picked Up',
    'Fence',
    'Fence - Damaged',
    'Fireworks',
    'Flashing Beacon Maintenance',
    'Free-floating Car Share - Parking Complaint - Communauto Flex',
    'Games',
    'Garbage / Park / Bin Damaged',
    'Garbage / Park / Bin Graffiti on Bin',
    'Garbage / Park / Bin Installation',
    'Garbage / Park / Bin Missing',
    'Garbage / Park / Bin Overflow',
    'Garbage / Park / Bin Removal / Relocate',
    'General Parking Regulations',
    'General Pruning',
    'General Tree Maintenance',
    'Graffiti',
    'Graffiti on Hydro Asset',
    'Guardrail - Damaged',
    'Gypsy Moth Control Insp',
    'Heavy Trucks',
    'Hydrant-After Usage Test',
    'Hydrant-Damage',
    'Hydrant-Leaking',
    'Hydro - Brush Pick-up',
    'INJUR/DIST DOMESTIC',
    'INJUR/DIST WILDLIFE',
    'IPM Inspection',
    'Ice and Snow Complaint',
    'Illegal Dumping',
    'Illegal Dumping / Discharge',
    'Illegal Off-Street Parking',
    'Illegal On-Street Parking',
    'Intersection Safety Review',
    'Investigate Pavement Markings',
    'Investigate Regulatory Signs',
    'Investigate Temporary Condition Markings',
    'Investigate Temporary Condition Signs',
    'Investigate Vehicles Leaving Roadway',
    'Investigate Warning Signs',
    'Lane Designation',
    'Laneway - Salting / Sanding / Salt',
    'Laneway - Snow Not Ploughed',
    'Laneway - Surface Damage',
    'Left/Right Turn Signal Priority Features',
    'Litter / Bike Removal Inquiry',
    'Litter / Bin / Graffiti on Bin',
    'Litter / Bin / Litter Bin Installation',
    'Litter / Bin / Overflow or Not Picked Up',
    'Litter / Bin / Reinstall|| Replace Missing',
    'Litter / Bin / Relocate',
    'Litter / Bin / Removal',
    'Litter / Illegal Dumping Cleanup',
    'Litter / Laneway / Clean Up',
    'Litter / Sidewalk & Blvd / Pick Up Request',
    'Litter / Special Event / Pick Up Request',
    'Litter/Needle Cleanup',
    'Locate-Emergency',
    'Long Grass and Weeds',
    'MLS Hazard Tree Invst',
    'Maintenance Hole - Overflowing',
    'Maintenance Hole - Sunken / High',
    'Maintenance Hole-Damage',
    'Maintenance Hole-Missing Cover',
    'Maintenance Holes -Damage / Repair',
    'Maintenance Holes Lid Loose/Missing',
    'Missed Leaf Collection',
    'Missing/Damaged Flexible Bollards',
    'Missing/Damaged School Flashing Beacons',
    'Missing/Damaged Signs',
    'Missing/Damaged Watch Your Speed Boards',
    'Missing/Faded Pavement Markings',
    'Motor Coach Parking',
    'Mulching',
    'Multiple SRs/Collections Curb Day',
    'Multiple SRs/Collections Nights',
    'Multiple SRs/Litter Operations',
    'New Pedestrian Crossover',
    'New Traffic Control Signal Request',
    'Noise',
    'Noise Complaint',
    'Non-Res Garbage Bag / Not Picked Up',
    'Non-Res Garbage Bag Nite / Not Picked Up',
    'Non-Res Organic Bin Nite / Not Picked Up',
    'Non-Res Recycle Bin Nite / Not Picked Up',
    'Non-Res Yard Waste / Not Picked Up',
    'One-way Streets',
    'Operational Comment or Complaint',
    'Operator / Operations Complaint',
    'Operator / Operations Compliment',
    'PARKS COM PERMIT',
    'PARKS EXCREMENT',
    'PARKS LEASH',
    'PARKS MORE 3 DOG',
    'PXO Maintenance',
    'Park Use',
    'Parking',
    'Parking in a Public Lane',
    'Parks Ravine Safety Mtc FNEM',
    'Pedestrian Crossing Protection',
    'Pedestrian Crossover Operation',
    'Pedestrian Issues/Timing/Delays',
    'Pedestrian Refuge Island',
    'Permit Inspection',
    'Pit Cover/Paver Maintenance',
    'Planting 11 Plus Trees FNEM',
    'Pollution Spill Response',
    'Postering City Property/Structures',
    'Postering Kiosks',
    'Pot hole on Expressway',
    'Power Outage',
    'Private Tree Inspection',
    'Prohibited Acts/Pollicking',
    'Prohibited Waste',
    'Property Damaged/Collections Curb Day',
    'Property Damaged/Collections FEL',
    'Property Damaged/Collections Nights',
    'Property Damaged/Litter Operations',
    'Property Standards',
    'Public Spaces Complaint',
    'RESCU Maintenance',
    'Ravine Inspection',
    'Recycle / Park / Bin Damaged',
    'Recycle / Park / Bin Installation',
    'Recycle / Park / Bin Missing',
    'Recycle / Park / Bin Overflow',
    'Recycle / Park / Bin Relocate / Removal',
    'Registration - Toronto Water',
    'Res / Garbage / Multiple Addresses Not Picked Up',
    'Res / Garbage / Not Picked Up',
    'Res / Org&Garbage Multiple Addresses / Not Picked Up',
    'Res / Org&Recycle Multiple Addresses / Not Picked Up',
    'Res / Organic Bin / New Account',
    'Res / Organic Bin / Wrong Delivery',
    'Res / Organic Green Bin / Multiple Addresses / Not Picked Up',
    'Res / Organic Green Bin / Not Picked Up',
    'Res / Organic Green Bin / Retrieval',
    'Res / Recycle / Multiple Addresses / Not Picked Up',
    'Res / Recycle / Not Picked Up',
    'Res / Recycle Front&Side / Not Picked Up',
    'Res / Yard Waste Multiple Addresses / Not Picked Up',
    'Res Above Comm / Nite Garbage / Not Picked Up',
    'Res Above Comm / Nite Recycle / Not Picked Up',
    'Reserved Lane',
    'Residential / XMAS Tree / Not Picked Up',
    'Residential / Yard Waste / Not Picked Up',
    'Residential Furniture / Not Picked Up',
    'Residential: Bin: Repair or Replace Body/Handle',
    'Residential: Bin: Wrong Delivery',
    'Residential: Garbage Bin: Missing',
    'Residential: Garbage Bin: New Account Medium',
    'Residential: Recycle Bin: New Account Large',
    'Respond to Locates Request',
    'Restoration Related',
    'Retaining Wall - Damage / Repair',
    'Road - Cleaning/Debris',
    'Road - Damaged',
    'Road - Graffiti Complaint',
    'Road - Gravel Roads/Construction',
    'Road - Plough Damage',
    'Road - Pot hole',
    'Road - Sanding / Salting Required',
    'Road - Sinking',
    'Road Allowance',
    'Road Design',
    'Road Ploughing Required',
    'Road Water Ponding',
    'Road damaged on Expressway',
    'Road-Illegal Dumping',
    'Road-Winter Request/ Complaint',
    'Roadside - Plough Damage',
    'Roadside Utility Cut - Settlement',
    'Roadway Utility Cut - Settlement',
    'Rooming House',
    'SERVICES PROT CUST',
    'STRAY AT LARGE',
    'STRAY CONFINED',
    'Salting-Winter (WSL/HYDT/VALVE/Watermain Break Locations etc.)',
    'School Bus Loading Zone',
    'School Crossing Guard - No Show',
    'School Zone Safety Review',
    'School-Related Warning Signs',
    'Sewer Odour',
    'Sewer Service Line-Blocked',
    'Sewer Service Line-Cleanout Repair',
    'Sewer main-Backup',
    'Shoulder - Maintenance',
    'Shoulder on Expressway Damaged',
    'Sidewalk - Cleaning',
    'Sidewalk - Damaged / Concrete',
    'Sidewalk - Damaged /Brick/Interlock',
    'Sidewalk - Graffiti Complaint',
    'Sidewalk - Paraplegic Ramps',
    'Sidewalk - Seniors Snow Clearing',
    'Sidewalk - Snow Clearing',
    'Sidewalk Icy|| Needs Sand/Salt',
    'Sidewalk-Water Ponding',
    'Sight Line Obstruction',
    'Sign Maintenance',
    'Signal Timing Review/Vehicle Delays',
    'Signs',
    'Sink Hole',
    'Snow Removal - General',
    'Snow Removal - School Zone',
    'Snow Removal - Sightline Problem',
    'Speed Bumps in Laneway',
    'Speed Watch Programme',
    'Speeding',
    'Spills/Cleanup/Collections Curb Day',
    'Spills/Cleanup/Collections FEL',
    'Spills/Cleanup/Collections Nights',
    'Spills/Cleanup/Litter Operations',
    'Spills/Cleanup/PRM',
    'Staff Conduct / Collections / Parks',
    'Staff Conduct/Collections Curb Day',
    'Staff Conduct/Collections FEL',
    'Staff Conduct/Collections Nights',
    'Staff Conduct/Litter Operations',
    'Staff Conduct/Non-Collections',
    'Staff Service Complaint',
    'Staff Service Compliment',
    'Stemming',
    'Stoop N Scoop',
    'Storm Clean Up',
    'Street Light Out',
    'Street furniture damaged',
    'Streetcar Platforms',
    'Student Crossing Issues',
    'Student Pick-up/Drop-off Area',
    'Stumping',
    'TAS_COYOT RESP AN-ATTACK',
    'TAS_COYOT RESP AN-BITE',
    'TAS_COYOT RESP HU-ATTACK',
    'TAS_COYOT RESP HU-BITE',
    'TAS_COYOT RESP PUB-SAFETY',
    'TAS_STRAY ATTACK',
    'TAS_STRAY MENACE',
    'Taxicab Stand',
    'Taxi|| Limo Complaint',
    'Temporary Signs',
    'Time Limit or Excessive Duration Parking',
    'Tow Truck Complaint',
    'Trades Complaint',
    'Traffic Calming Measures',
    'Traffic Infiltration',
    'Traffic Island - Damaged',
    'Traffic Island-Grass Needs Cutting',
    'Traffic Sign - Graffiti Complaint',
    'Traffic Signal - Graffiti Complaint',
    'Traffic Signal Maintenance',
    'Transfer Station Comment',
    'Transfer Station Odour Complaint',
    'Tree Planting',
    'Trees and Plants',
    'Unknown - TSU-SCG03',
    'Vehicles',
    'Walkway - Snow Clearing/ Salting required',
    'Walkway - damaged',
    'Walkway-Weeds Need Cutting',
    'Waste',
    'Waste Storage',
    'Water Meter-Broken/Frozen',
    'Water Service Line-Check Water Service Box',
    'Water Service Line-Damaged Water Service Box',
    'Water Service Line-Leaking',
    'Water Service Line-Locate / Adjust service box',
    'Water Service Line-Low Pressure|| Low Flow Insp',
    'Water Service Line-No Water',
    'Water Service Line-Turn Off',
    'Water Service Line-Turn Off/Burst',
    'Water Service Line-Turn On',
    'Water Service Test for High Lead Content',
    'Water Valve-Leaking',
    'Watercourse Investigation',
    'Watercourses-Blocked/Flooding',
    'Watercourses-Erosion/Washout',
    'Watercourses-Outfalls/Inlets',
    'Watermain Valve - Turn Off',
    'Watermain Valve - Turn On',
    'Watermain-Possible Break',
    'West Nile Virus - Standing Water / Roadway',
    'Wrong Location/Time/Day',
    'Zoning',
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