import geopandas as gpd
import pandas as pd
import os

# Define all facility types and their file paths
geojson_files = {
    "highway": "parameters/highway.geojson",
    "hospital": "parameters/hospital.geojson",
    "park": "parameters/park.geojson",
    "police": "parameters/police.geojson",
    "school": "parameters/school.geojson",
    "station": "parameters/station.geojson",
    "supermarket": "parameters/supermarket.geojson"
}

# Store all facility data
all_data = []

# Process each `geojson` file
for category, file_path in geojson_files.items():
    if os.path.exists(file_path):  # Ensure the file exists
        print(f"📌 Processing {category} data...")

        # Read `geojson`
        gdf = gpd.read_file(file_path)

        # Extract `node_id`
        if "@id" in gdf.columns:
            gdf["node_id"] = gdf["@id"]
        else:
            print(f"⚠️ {category} does not have '@id', please check the JSON structure!")
            continue

        # Extract `lat, lon`
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y

        # Keep only `node_id`, `lat`, `lon`, `category`
        gdf_cleaned = gdf[["node_id", "lat", "lon"]].dropna()
        gdf_cleaned["category"] = category  # Add category

        # Store in the list
        all_data.append(gdf_cleaned)

# Merge all data
final_df = pd.concat(all_data, ignore_index=True)

# Save to CSV
final_df.to_csv("data/all_facilities_data.csv", index=False)

print("✅ All `geojson` data extraction completed and saved to `data/all_facilities_data.csv`!")
