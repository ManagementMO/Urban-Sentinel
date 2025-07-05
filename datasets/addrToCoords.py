import pandas as pd
from geopy.geocoders import Nominatim
import time

# Initialize Geolocator
geolocator = Nominatim(user_agent="myGeocoder")

# Function to get latitude and longitude from address
def get_coordinates(address):
    try:
        location = geolocator.geocode(address)
        if location:
            
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        print(f"Error geocoding address: {address}. Error: {e}")
        return None, None

# Read the combined CSV file
combined_df = pd.read_csv('output.csv')

# Initialize lists for latitudes and longitudes
latitudes = []
longitudes = []

# Iterate through each address and get coordinates
for address in combined_df['Intersection Street 1']:
    lat, lon = get_coordinates(address)
    latitudes.append(lat)
    longitudes.append(lon)

    # Add a delay to prevent hitting geocoding rate limits
    time.sleep(1)  # You can adjust the sleep time based on the API usage limits

# Add new columns for latitude and longitude
combined_df['Latitude'] = latitudes
combined_df['Longitude'] = longitudes

# Save the updated DataFrame with coordinates
combined_df.to_csv('blight_coords.csv', index=False)

print("Coordinates added and CSV saved successfully!")