import numpy as np
import pandas as pd
import json
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from functools import reduce
from shapely.geometry import Point, Polygon
import ast

def cal_dis(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])  # Convert to radians

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c  # Return distance in km


# ---------- Main Program --------------

df = pd.read_csv("data/real-estate-data.csv")

# Set the number of clusters for region classification
num_clusters = 25  # You can adjust this number to control the number of regions

# Use only latitude and longitude for clustering
coords = df[['lt', 'lg']].dropna()  # Remove NaN values to prevent errors

# Perform KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df.loc[coords.index, 'region'] = kmeans.fit_predict(coords)

# Convert the region column to integer type
df['region'] = df['region'].astype(int)

# Save the updated dataset
df.to_csv("data/estate_with_region.csv", index=False)

####################################################

# Train a nearest neighbors model for later searches
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(coords)

# Function to get the nearest region information for a given facility type
def get_nearest_region_info(file_name):
    house_info = {}

    # Open and read the specified GeoJSON file
    with open(f"./parameters/{file_name}.geojson", "r") as highway_f:
        file_info = json.load(highway_f)['features']

        for info in file_info:
            target_lon = info['geometry']['coordinates'][0]  # Longitude
            target_lat = info['geometry']['coordinates'][1]  # Latitude

            # Iterate over real estate properties to calculate the distance
            for house in df.itertuples():
                if pd.isna(house.lt) or pd.isna(house.lg):
                    continue  # Skip entries with missing coordinates

                # Calculate the distance to the facility
                distance = cal_dis(house.lt, house.lg, target_lat, target_lon)
                distance = round(distance, 2)

                # Store the minimum distance for each house
                if house.id_ not in house_info:
                    house_info[house.id_] = distance
                house_info[house.id_] = min(house_info[house.id_], distance)

    return house_info


# highway
highway_temp = get_nearest_region_info("highway")
highway_info = pd.DataFrame({
    'id_': list(highway_temp.keys()), 
    'minHighwayDis': [v for v in highway_temp.values()]
})

hospital_temp = get_nearest_region_info("hospital")
hospital_info = pd.DataFrame({
    'id_': list(hospital_temp.keys()), 
    'minHospitalDis': [v for v in hospital_temp.values()]
})

mall_temp = get_nearest_region_info("mall+supermarket+marketplace")
mall_info = pd.DataFrame({
    'id_': list(mall_temp.keys()),
    'minMallDis': [v for v in mall_temp.values()]
})

park_temp = get_nearest_region_info("park")
park_info = pd.DataFrame({
    'id_': list(park_temp.keys()), 
    'minParkDis': [v for v in park_temp.values()]
})

police_temp = get_nearest_region_info("police")
police_info = pd.DataFrame({
    'id_': list(police_temp.keys()), 
    'minPoliceDis': [v for v in police_temp.values()]
})

school_temp = get_nearest_region_info("school")
school_info = pd.DataFrame({
    'id_': list(school_temp.keys()), 
    'minSchoolDis': [v for v in school_temp.values()]
})

station_temp = get_nearest_region_info("station")
station_info = pd.DataFrame({
    'id_': list(station_temp.keys()), 
    'minStationDis': [v for v in station_temp.values()]
})

# 犯罪率
crime_data = pd.read_csv("data/crime_data.csv")
crime_info = {
    'id_': [],
    'crime_rate_per_100000_people': []
}
for row in crime_data.itertuples():
    poly_coords = ast.literal_eval(row.latitude_longitude)
    poly = Polygon(poly_coords)
    for house in df.itertuples():
        if pd.isna(house.lt) or pd.isna(house.lg):
            continue
        point = Point(house.lg, house.lt)
        if poly.contains(point):
            crime_info['id_'].append(house.id_)
            crime_info['crime_rate_per_100000_people'].append(int(row.crime_rate_per_100000_people))
crime_data = pd.DataFrame(crime_info)

dfs = [df, highway_info, hospital_info, mall_info, park_info, police_info, school_info, station_info, crime_data] 

result_df = reduce(lambda left, right: pd.merge(left, right, on='id_', how='outer'), dfs)

# result_df["region"] = result_df["region"].astype(int)
# result_df = result_df.sort_values(by="region")

result_df.to_csv("data/new_info.csv", index=False)

