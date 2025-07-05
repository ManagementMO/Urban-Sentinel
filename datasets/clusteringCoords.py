import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# --- Helper Function for Distance Calculation ---
def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # Radius of earth in kilometers
    return km

# --- 1. Load and Prepare the Data ---
try:
    df = pd.read_csv("output_with_coords.csv")
except FileNotFoundError:
    print("output_with_coords.csv not found. Using a sample DataFrame for demonstration.")
    # Sample data with more points to make cluster stats meaningful
    # (This is just for demonstration if the file is missing)
    np.random.seed(42)
    center1 = [43.65, -79.38]
    center2 = [43.77, -79.25]
    points1 = np.random.randn(100, 2) * 0.005 + center1
    points2 = np.random.randn(150, 2) * 0.008 + center2
    sample_coords = np.vstack([points1, points2])
    df = pd.DataFrame(sample_coords, columns=['latitude', 'longitude'])
    df['Service Request Type'] = np.random.choice(['Graffiti', 'Pothole', 'Illegal Dumping'], size=len(df))

df.dropna(subset=['latitude', 'longitude'], inplace=True)
coords = df[['latitude', 'longitude']].values

# --- 2. Run DBSCAN Model ---
kms_per_radian = 6371.0
epsilon_in_km = 0.4  # Increased eps slightly for better sample clustering
eps = epsilon_in_km / kms_per_radian
min_samples = 10

print(f"Running DBSCAN...")
db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
df['cluster'] = db.labels_

# --- 3. Analyze Clusters and Calculate Advanced Metrics ---
hotspot_df = df[df['cluster'] != -1].copy()
grouped = hotspot_df.groupby('cluster')

hotspot_summary = []
for cluster_id, cluster_data in grouped:
    center_lat = cluster_data['latitude'].mean()
    center_lon = cluster_data['longitude'].mean()
    
    # --- NEW: Calculate Distances and Radii ---
    # Calculate the distance of every point in the cluster to its center
    distances_km = haversine_distance(cluster_data['longitude'], cluster_data['latitude'], center_lon, center_lat)
    
    avg_radius_km = distances_km.mean()
    max_radius_km = distances_km.max()

    # --- NEW: Calculate Density Gradient/Fall-off ---
    # Define concentric rings (e.g., in 100-meter increments up to the max radius)
    num_rings = 5 # We'll create 5 rings for the gradient
    ring_width = max_radius_km / num_rings
    
    ring_counts = []
    for i in range(num_rings):
        outer_boundary = (i + 1) * ring_width
        inner_boundary = i * ring_width
        # Count points within this ring
        count = np.sum((distances_km > inner_boundary) & (distances_km <= outer_boundary))
        ring_counts.append(count)
    
    # Format the gradient as a string for the CSV
    density_falloff = " -> ".join(map(str, ring_counts))

    hotspot_summary.append({
        'hotspot_id': cluster_id,
        'center_latitude': center_lat,
        'center_longitude': center_lon,
        'num_complaints': len(cluster_data),
        'major_complaint': cluster_data['Service Request Type'].mode()[0],
        'avg_radius_km': avg_radius_km,
        'max_radius_km': max_radius_km,
        'density_falloff': density_falloff
    })

# --- 4. Finalize and Save Output ---
if not hotspot_summary:
    print("\nNo hotspots were found with the current settings.")
else:
    final_hotspots_df = pd.DataFrame(hotspot_summary)
    
    # Create a simple risk score for sorting
    min_complaints = final_hotspots_df['num_complaints'].min()
    max_complaints = final_hotspots_df['num_complaints'].max()
    if max_complaints > min_complaints:
        final_hotspots_df['risk_score'] = ((final_hotspots_df['num_complaints'] - min_complaints) / (max_complaints - min_complaints)) * 100
    else:
        final_hotspots_df['risk_score'] = 100.0

    final_hotspots_df.sort_values('risk_score', ascending=False, inplace=True)
    
    # Save the results
    final_hotspots_df.to_csv("urban_decay_hotspots.csv", index=False)

    print("\nAdvanced hotspot analysis complete!")
    print("Results saved to 'complaint_hotspots_advanced.csv'.")
    print("\nTop 5 Hotspots:")
    print(final_hotspots_df.head().round(3))