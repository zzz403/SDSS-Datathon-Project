import json
import pandas as pd

# Load the JSON file
json_file_path = "parameters/neighbourhood-crime-rates - 4326.geojson"
with open(json_file_path, "r") as f:
    crime_data = json.load(f)

# Extract relevant crime statistics
crime_records = []

# crime_rate for 2024
for feature in crime_data["features"]:
    properties = feature["properties"]
    geometry = feature["geometry"]

    crime_info = {
        "crime_rate_per_100000_people": ((properties["ASSAULT_2024"] + properties["AUTOTHEFT_2024"] +
                                            properties["BIKETHEFT_2024"] + properties["BREAKENTER_2024"] +
                                            properties["HOMICIDE_2024"] + properties["ROBBERY_2024"] +
                        properties["SHOOTING_2024"] + properties["THEFTFROMMV_2024"] + properties["THEFTOVER_2024"])
                       / properties["POPULATION_2024"]) * 100000,
        "violent_crimes(homicide+shooting)": properties["HOMICIDE_2024"] + properties["SHOOTING_2024"],
        "latitude_longitude": geometry["coordinates"][0][0],
    }

    crime_records.append(crime_info)

# Convert to DataFrame
df_crime = pd.DataFrame(crime_records)
df_crime.to_csv("data/crime_data.csv", index=False)
# Display first few rows
print(df_crime)
