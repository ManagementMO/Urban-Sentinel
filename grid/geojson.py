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
# We use the entire history from 2014-2020 to understand each cell's behavior
HISTORY_START_YEAR = 2014
HISTORY_END_YEAR = 2020
TARGET_YEAR = 2021  # The year we want to predict

# --- Blight Definition ---
# List of complaint types that strongly indicate blight or neglect.
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

# The percentile to define a "blighted" area. 0.90 means the top 10% of cells
# with the most blight complaints will be flagged as blighted.
BLIGHT_QUANTILE_THRESHOLD = 0.90


# ==============================================================================
# --- SCRIPT EXECUTION ---
# You generally do not need to edit below this line.
# ==============================================================================

def create_temporal_features(yearly_data, grid):
    """
    Create rich temporal features from yearly complaint data.
    
    Args:
        yearly_data: DataFrame with columns ['cell_id', 'year', 'total_complaints', 'blight_complaints']
        grid: GeoDataFrame with the grid cells
    
    Returns:
        DataFrame with temporal features for each cell
    """
    print("  - Creating temporal features...")
    
    # Ensure we have data for all years and all cells (fill missing with 0)
    all_years = list(range(HISTORY_START_YEAR, HISTORY_END_YEAR + 1))
    all_cells = grid['cell_id'].unique()
    
    # Create a complete year-cell combination
    year_cell_combinations = pd.MultiIndex.from_product(
        [all_cells, all_years], 
        names=['cell_id', 'year']
    ).to_frame(index=False)
    
    # Merge with actual data and fill missing values with 0
    complete_data = year_cell_combinations.merge(yearly_data, on=['cell_id', 'year'], how='left')
    complete_data['total_complaints'] = complete_data['total_complaints'].fillna(0)
    complete_data['blight_complaints'] = complete_data['blight_complaints'].fillna(0)
    
    # Sort by cell_id and year for temporal calculations
    complete_data = complete_data.sort_values(['cell_id', 'year'])
    
    # Create temporal features for each cell
    features_list = []
    
    for cell_id in all_cells:
        cell_data = complete_data[complete_data['cell_id'] == cell_id].copy()
        cell_data = cell_data.sort_values('year')
        
        features = {'cell_id': cell_id}
        
        # === BASIC STATISTICS (2014-2020) ===
        features['total_complaints_mean'] = cell_data['total_complaints'].mean()
        features['total_complaints_std'] = cell_data['total_complaints'].std()
        features['total_complaints_sum'] = cell_data['total_complaints'].sum()
        features['total_complaints_max'] = cell_data['total_complaints'].max()
        features['total_complaints_min'] = cell_data['total_complaints'].min()
        
        features['blight_complaints_mean'] = cell_data['blight_complaints'].mean()
        features['blight_complaints_std'] = cell_data['blight_complaints'].std()
        features['blight_complaints_sum'] = cell_data['blight_complaints'].sum()
        features['blight_complaints_max'] = cell_data['blight_complaints'].max()
        features['blight_complaints_min'] = cell_data['blight_complaints'].min()
        
        # === RECENT HISTORY (LAST 3 YEARS: 2018-2020) ===
        recent_data = cell_data[cell_data['year'] >= 2018]
        features['total_complaints_recent_mean'] = recent_data['total_complaints'].mean()
        features['blight_complaints_recent_mean'] = recent_data['blight_complaints'].mean()
        features['total_complaints_recent_sum'] = recent_data['total_complaints'].sum()
        features['blight_complaints_recent_sum'] = recent_data['blight_complaints'].sum()
        
        # === TREND ANALYSIS ===
        years = cell_data['year'].values
        total_complaints = cell_data['total_complaints'].values
        blight_complaints = cell_data['blight_complaints'].values
        
        # Linear trend (slope) over the entire period
        if len(years) > 1:
            total_trend = np.polyfit(years, total_complaints, 1)[0]
            blight_trend = np.polyfit(years, blight_complaints, 1)[0]
        else:
            total_trend = 0
            blight_trend = 0
            
        features['total_complaints_trend'] = total_trend
        features['blight_complaints_trend'] = blight_trend
        
        # Recent trend (last 3 years)
        if len(recent_data) > 1:
            recent_years = recent_data['year'].values
            recent_total = recent_data['total_complaints'].values
            recent_blight = recent_data['blight_complaints'].values
            recent_total_trend = np.polyfit(recent_years, recent_total, 1)[0]
            recent_blight_trend = np.polyfit(recent_years, recent_blight, 1)[0]
        else:
            recent_total_trend = 0
            recent_blight_trend = 0
            
        features['total_complaints_recent_trend'] = recent_total_trend
        features['blight_complaints_recent_trend'] = recent_blight_trend
        
        # === MOMENTUM AND ACCELERATION ===
        # Year-over-year changes
        cell_data['total_yoy_change'] = cell_data['total_complaints'].diff()
        cell_data['blight_yoy_change'] = cell_data['blight_complaints'].diff()
        
        features['total_yoy_change_mean'] = cell_data['total_yoy_change'].mean()
        features['blight_yoy_change_mean'] = cell_data['blight_yoy_change'].mean()
        features['total_yoy_change_std'] = cell_data['total_yoy_change'].std()
        features['blight_yoy_change_std'] = cell_data['blight_yoy_change'].std()
        
        # Acceleration (change in the rate of change)
        cell_data['total_acceleration'] = cell_data['total_yoy_change'].diff()
        cell_data['blight_acceleration'] = cell_data['blight_yoy_change'].diff()
        
        features['total_acceleration_mean'] = cell_data['total_acceleration'].mean()
        features['blight_acceleration_mean'] = cell_data['blight_acceleration'].mean()
        
        # === ROLLING AVERAGES ===
        # 3-year rolling averages (centered)
        cell_data['total_rolling_3yr'] = cell_data['total_complaints'].rolling(window=3, center=True).mean()
        cell_data['blight_rolling_3yr'] = cell_data['blight_complaints'].rolling(window=3, center=True).mean()
        
        features['total_rolling_3yr_mean'] = cell_data['total_rolling_3yr'].mean()
        features['blight_rolling_3yr_mean'] = cell_data['blight_rolling_3yr'].mean()
        
        # === VOLATILITY ===
        # Coefficient of variation (std/mean) - measures consistency
        features['total_volatility'] = features['total_complaints_std'] / (features['total_complaints_mean'] + 1e-6)
        features['blight_volatility'] = features['blight_complaints_std'] / (features['blight_complaints_mean'] + 1e-6)
        
        # === TEMPORAL PATTERNS ===
        # Peak year and minimum year
        features['total_peak_year'] = cell_data.loc[cell_data['total_complaints'].idxmax(), 'year']
        features['blight_peak_year'] = cell_data.loc[cell_data['blight_complaints'].idxmax(), 'year']
        features['total_min_year'] = cell_data.loc[cell_data['total_complaints'].idxmin(), 'year']
        features['blight_min_year'] = cell_data.loc[cell_data['blight_complaints'].idxmin(), 'year']
        
        # Time since peak/minimum
        features['years_since_total_peak'] = HISTORY_END_YEAR - features['total_peak_year']
        features['years_since_blight_peak'] = HISTORY_END_YEAR - features['blight_peak_year']
        
        # === LAGGED FEATURES ===
        # Values from specific recent years
        features['total_complaints_2020'] = cell_data[cell_data['year'] == 2020]['total_complaints'].iloc[0]
        features['total_complaints_2019'] = cell_data[cell_data['year'] == 2019]['total_complaints'].iloc[0]
        features['total_complaints_2018'] = cell_data[cell_data['year'] == 2018]['total_complaints'].iloc[0]
        
        features['blight_complaints_2020'] = cell_data[cell_data['year'] == 2020]['blight_complaints'].iloc[0]
        features['blight_complaints_2019'] = cell_data[cell_data['year'] == 2019]['blight_complaints'].iloc[0]
        features['blight_complaints_2018'] = cell_data[cell_data['year'] == 2018]['blight_complaints'].iloc[0]
        
        # === EARLY VS LATE PERIOD COMPARISON ===
        early_period = cell_data[cell_data['year'] <= 2017]  # 2014-2017
        late_period = cell_data[cell_data['year'] >= 2018]   # 2018-2020
        
        features['total_early_mean'] = early_period['total_complaints'].mean()
        features['total_late_mean'] = late_period['total_complaints'].mean()
        features['blight_early_mean'] = early_period['blight_complaints'].mean()
        features['blight_late_mean'] = late_period['blight_complaints'].mean()
        
        # Change from early to late period
        features['total_early_to_late_change'] = features['total_late_mean'] - features['total_early_mean']
        features['blight_early_to_late_change'] = features['blight_late_mean'] - features['blight_early_mean']
        
        # === CONSISTENCY PATTERNS ===
        # How many years had above-median complaints
        total_median = cell_data['total_complaints'].median()
        blight_median = cell_data['blight_complaints'].median()
        
        features['years_above_median_total'] = (cell_data['total_complaints'] > total_median).sum()
        features['years_above_median_blight'] = (cell_data['blight_complaints'] > blight_median).sum()
        
        # === RATIOS ===
        # Blight as percentage of total complaints
        features['blight_ratio_mean'] = (cell_data['blight_complaints'] / (cell_data['total_complaints'] + 1e-6)).mean()
        features['blight_ratio_recent'] = (recent_data['blight_complaints'] / (recent_data['total_complaints'] + 1e-6)).mean()
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Fill any remaining NaN values with 0
    features_df = features_df.fillna(0)
    
    print(f"  - Created {len(features_df.columns) - 1} temporal features for {len(features_df)} cells")
    return features_df


def run_data_processing():
    """Main function to run the entire temporal data processing pipeline."""

    # --- Step 1: Create the Master Grid ---
    print("--- Step 1: Creating the master 250x250m grid... ---")
    if not os.path.exists(TORONTO_BOUNDARY_SHP):
        print(f"ERROR: Boundary shapefile not found at '{TORONTO_BOUNDARY_SHP}'. Please check the path.")
        return
    
    toronto_boundary = gpd.read_file(TORONTO_BOUNDARY_SHP)
    
    # Set CRS if not defined (assuming WGS84 for Toronto boundary)
    if toronto_boundary.crs is None:
        print("  - Setting CRS to WGS84 (EPSG:4326)")
        toronto_boundary = toronto_boundary.set_crs(epsg=4326)
    
    toronto_utm = toronto_boundary.to_crs(epsg=32617)
    xmin, ymin, xmax, ymax = toronto_utm.total_bounds
    cell_size = 250
    grid_cells = []
    for x in np.arange(xmin, xmax, cell_size):
        for y in np.arange(ymin, ymax, cell_size):
            grid_cells.append(Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)]))
    
    grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs="EPSG:32617")
    grid = gpd.clip(grid, toronto_utm)
    grid['cell_id'] = range(len(grid))
    print(f"  - Grid created successfully with {len(grid)} cells.")

    # --- Step 2: Load and Prepare Consolidated 311 Data ---
    print("\n--- Step 2: Loading and preparing consolidated 311 data... ---")
    if not os.path.exists(RAW_311_CSV):
        print(f"ERROR: 311 data CSV not found at '{RAW_311_CSV}'. Please check the path.")
        return
        
    df_311 = pd.read_csv(RAW_311_CSV)

    # Clean up column names by stripping whitespace
    df_311.columns = df_311.columns.str.strip()

    # Convert date column to datetime objects
    df_311['Creation Date'] = pd.to_datetime(df_311['Creation Date'])
    df_311['year'] = df_311['Creation Date'].dt.year
    print(f"  - Loaded {len(df_311)} records from years {df_311['year'].min()} to {df_311['year'].max()}.")

    # Filter out records with invalid coordinates (NaN or outside reasonable bounds)
    valid_coords_mask = (
        df_311['longitude'].notna() & 
        df_311['latitude'].notna() & 
        (df_311['longitude'] >= -180) & 
        (df_311['longitude'] <= 180) & 
        (df_311['latitude'] >= -90) & 
        (df_311['latitude'] <= 90)
    )
    
    df_311_clean = df_311[valid_coords_mask].copy()
    print(f"  - Filtered to {len(df_311_clean)} records with valid coordinates.")

    # Convert to a GeoDataFrame
    gdf_311 = gpd.GeoDataFrame(
        df_311_clean,
        geometry=gpd.points_from_xy(df_311_clean.longitude, df_311_clean.latitude),
        crs="EPSG:4326"
    )

    # Project to the same CRS as the grid
    gdf_311_utm = gdf_311.to_crs(grid.crs)
    print("  - Data successfully loaded and projected.")

    # --- Step 3: Create Yearly Aggregated Data for Temporal Analysis ---
    print(f"\n--- Step 3: Creating yearly aggregated data ({HISTORY_START_YEAR}-{HISTORY_END_YEAR})... ---")
    
    # Filter to our historical period
    historical_data = gdf_311_utm[
        (gdf_311_utm['year'] >= HISTORY_START_YEAR) & 
        (gdf_311_utm['year'] <= HISTORY_END_YEAR)
    ].copy()
    
    print(f"  - Using {len(historical_data)} records from {HISTORY_START_YEAR}-{HISTORY_END_YEAR}")
    
    # Create yearly data for each cell
    yearly_data_list = []
    
    for year in range(HISTORY_START_YEAR, HISTORY_END_YEAR + 1):
        print(f"  - Processing year {year}...")
        year_data = historical_data[historical_data['year'] == year].copy()
        
        if len(year_data) == 0:
            print(f"    - No data for {year}, filling with zeros")
            # Create empty data for this year
            for cell_id in grid['cell_id']:
                yearly_data_list.append({
                    'cell_id': cell_id,
                    'year': year,
                    'total_complaints': 0,
                    'blight_complaints': 0
                })
            continue
        
        # Spatial join with grid
        joined_data = gpd.sjoin(year_data, grid, how="inner", predicate="within")
        
        # Calculate total complaints per cell
        total_counts = joined_data.groupby('cell_id').size().reset_index(name='total_complaints')
        
        # Calculate blight indicator complaints per cell
        blight_data = joined_data[joined_data['Service Request Type'].isin(BLIGHT_INDICATOR_COMPLAINTS)]
        blight_counts = blight_data.groupby('cell_id').size().reset_index(name='blight_complaints')
        
        # Merge counts
        year_counts = total_counts.merge(blight_counts, on='cell_id', how='left')
        year_counts['blight_complaints'] = year_counts['blight_complaints'].fillna(0)
        year_counts['year'] = year
        
        # Add cells with no complaints (they still exist in the grid)
        all_cells = pd.DataFrame({'cell_id': grid['cell_id']})
        year_counts = all_cells.merge(year_counts, on='cell_id', how='left')
        year_counts['total_complaints'] = year_counts['total_complaints'].fillna(0)
        year_counts['blight_complaints'] = year_counts['blight_complaints'].fillna(0)
        year_counts['year'] = year_counts['year'].fillna(year)
        
        yearly_data_list.extend(year_counts.to_dict('records'))
        print(f"    - {year}: {len(year_counts)} cells processed")
    
    yearly_data = pd.DataFrame(yearly_data_list)
    print(f"  - Created yearly aggregated data: {len(yearly_data)} cell-year combinations")

    # --- Step 4: Create Rich Temporal Features ---
    print("\n--- Step 4: Creating rich temporal features for ML... ---")
    temporal_features = create_temporal_features(yearly_data, grid)

    # --- Step 5: Create Target Variable (Predict TARGET_YEAR) ---
    print(f"\n--- Step 5: Creating target variable for {TARGET_YEAR} prediction... ---")
    
    # Check if target year data exists
    target_data = gdf_311_utm[gdf_311_utm['year'] == TARGET_YEAR].copy()
    
    if len(target_data) == 0:
        print(f"  - WARNING: No data available for target year {TARGET_YEAR}")
        print(f"  - Creating synthetic target based on trend extrapolation")
        
        # For demonstration, we'll create a synthetic target based on the trend
        # In practice, you'd either have this data or predict it differently
        synthetic_target = []
        for _, row in temporal_features.iterrows():
            # Use trend to extrapolate 2021 blight level
            predicted_blight = max(0, row['blight_complaints_2020'] + row['blight_complaints_trend'])
            synthetic_target.append({
                'cell_id': row['cell_id'],
                'target_blight_count': predicted_blight
            })
        
        target_df = pd.DataFrame(synthetic_target)
        print(f"  - Created synthetic target using trend extrapolation")
        
    else:
        # Use actual target year data
        print(f"  - Using actual {TARGET_YEAR} data for target variable")
        target_complaints = target_data[target_data['Service Request Type'].isin(BLIGHT_INDICATOR_COMPLAINTS)]
        joined_target = gpd.sjoin(target_complaints, grid, how="inner", predicate="within")
        target_counts = joined_target.groupby('cell_id').size().reset_index(name='target_blight_count')
        
        # Ensure all cells are represented
        all_cells = pd.DataFrame({'cell_id': grid['cell_id']})
        target_df = all_cells.merge(target_counts, on='cell_id', how='left')
        target_df['target_blight_count'] = target_df['target_blight_count'].fillna(0)
    
    # Define blight threshold and binary target
    blight_threshold = target_df['target_blight_count'].quantile(BLIGHT_QUANTILE_THRESHOLD)
    target_df['is_blighted'] = (target_df['target_blight_count'] > blight_threshold).astype(int)
    
    print(f"  - Blight threshold: > {blight_threshold:.1f} complaints per cell")
    print(f"  - Cells flagged as blighted: {target_df['is_blighted'].sum()} ({target_df['is_blighted'].mean()*100:.1f}%)")

    # --- Step 6: Combine Features with Target and Finalize ---
    print("\n--- Step 6: Combining features with target and creating final dataset... ---")
    
    # Merge temporal features with target
    final_data = temporal_features.merge(target_df, on='cell_id', how='left')
    
    # Merge with grid geometry
    final_grid = grid.merge(final_data, on='cell_id', how='left')
    
    # Fill any missing targets with 0 (shouldn't happen, but just in case)
    final_grid['target_blight_count'] = final_grid['target_blight_count'].fillna(0)
    final_grid['is_blighted'] = final_grid['is_blighted'].fillna(0)
    
    print(f"  - Final dataset: {len(final_grid)} cells with {len(final_data.columns)-1} features")
    
    # --- Step 7: Save the Model-Ready Dataset ---
    print("\n--- Step 7: Saving the model-ready dataset... ---")
    
    # Project back to WGS84 for web mapping
    final_grid_web = final_grid.to_crs("EPSG:4326")
    
    # Save the file
    final_grid_web.to_file(OUTPUT_FILENAME, driver='GeoJSON')
    
    print(f"\n==================== PROCESS COMPLETE ====================")
    print(f"🎯 Model-ready temporal dataset saved: {OUTPUT_FILENAME}")
    print(f"📊 Features: {len(final_data.columns)-3} temporal features per cell")  # -3 for cell_id, target_blight_count, is_blighted
    print(f"🎯 Target: Predict blight in {TARGET_YEAR} based on {HISTORY_START_YEAR}-{HISTORY_END_YEAR} patterns")
    print(f"📈 Training period: {HISTORY_START_YEAR}-{HISTORY_END_YEAR} ({HISTORY_END_YEAR - HISTORY_START_YEAR + 1} years)")
    print(f"🏙️  Grid cells: {len(final_grid)} cells (250m x 250m)")
    print("✅ Ready for machine learning!")


if __name__ == '__main__':
    run_data_processing()