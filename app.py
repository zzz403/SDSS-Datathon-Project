import streamlit as st
import numpy as np
import pandas as pd
import folium
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point

# --------------- 1. Load data ----------------
st.title("🏡 Toronto Real Estate Analysis Dashboard")

# Load CSV data
df = pd.read_csv("data/new_info.csv")

# ---------------- 1.1 Some data preprocessing ----------------
# Ensure `region` is in string format
df["region"] = df["region"].astype(str)

# Load facilities data
facilities_df = pd.read_csv("data/all_facilities_data.csv")
facilities_df["node_id"] = facilities_df["node_id"].astype(str)

# Count the number of facilities in each region
facility_counts = facilities_df.groupby("category")["node_id"].count().reset_index()
facility_counts.columns = ["category", "count"]

# Calculate the average distance to the nearest facility for each region
facility_distance = facilities_df.groupby("category")["lat"].mean().reset_index()
facility_distance.columns = ["category", "avg_lat"]

# Merge facility data into the main dataframe
df = df.merge(facility_counts, left_on="region", right_on="category", how="left")
df = df.merge(facility_distance, left_on="region", right_on="category", how="left")

# Handle NaN values (some `region`s may lack certain facilities)
df.fillna(0, inplace=True)


# Handle NaN and inf values:
df["price"] = pd.to_numeric(df["price"], errors='coerce')  # Ensure the column is numeric, converting non-numeric values to NaN
df["price"] = df["price"].replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
df["price"] = df["price"].fillna(df["price"].median()).astype(int)  # Fill NaN with the median value and convert to integer

# Apply the same processing to other columns:
for col in ["building_age", "beds", "baths"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric type
    df[col] = df[col].fillna(df[col].median()).astype(int)  # Fill NaN with the median value

# Calculate price range
price_min, price_max = int(df["price"].min()), int(df["price"].max())

# --------------- 2. Sidebar Filters ----------------
st.sidebar.header("🔍 Filter Options")

# Region selection
selected_region = st.sidebar.multiselect("🗺️ Select Regions", df["region"].unique(), default=df["region"].unique())

# Price range selection
price_range = st.sidebar.slider("💰 Select Price Range (C$)", price_min, price_max, (price_min, price_max))

# Number of bedrooms selection
bedrooms = st.sidebar.multiselect("🛏️ Number of Bedrooms", df["beds"].unique(), default=df["beds"].unique())

# Building age range selection
age_min, age_max = int(df["building_age"].min()), int(df["building_age"].max())
building_age_range = st.sidebar.slider("🏗️ Select Building Age", age_min, age_max, (age_min, age_max))

# Apply filters
df_filtered = df[
    (df["region"].isin(selected_region)) &
    (df["price"] >= price_range[0]) & (df["price"] <= price_range[1]) &
    (df["beds"].isin(bedrooms)) &
    (df["building_age"] >= building_age_range[0]) & (df["building_age"] <= building_age_range[1])
]

# Display the filtered results
st.write(f"📊 {len(df_filtered)} properties match the selected filters.")

# --------------- 3. Map Visualization ----------------
st.subheader("📍 Real Estate Location Map")

m = folium.Map(location=[df["lt"].mean(), df["lg"].mean()], zoom_start=12)

# Set the price color
low_price = df_filtered["price"].quantile(0.33)
high_price = df_filtered["price"].quantile(0.66)

for _, row in df_filtered.iterrows():
    if row["price"] <= low_price:
        color = "green"
    elif row["price"] <= high_price:
        color = "blue"
    else:
        color = "red"

    folium.Marker(
        location=[row["lt"], row["lg"]],
        popup=f"🏡 Price: ${row['price']:,.0f}\n🛏️ {row['beds']} beds 🛁 {row['baths']} baths\n🏗️ {row['building_age']} years",
        icon=folium.Icon(color=color),
    ).add_to(m)

folium_static(m)

# Load `data/region_info.csv`
region_info = pd.read_csv("data/region_info.csv")

# List of facility types
facility_types = ["highway", "hospital", "mall", "park", "police", "school", "station"]

# Parse the number of facilities in each `region`
region_counts = []
for _, row in region_info.iterrows():
    region = str(row["region"])  # Ensure `region` is in string format
    facility_count = {"region": region}
    
    for facility in facility_types:
        if pd.notna(row[facility + "_ids"]):  # Ensure `facility_ids` is not empty
            facility_count[facility] = len(row[facility + "_ids"].split(", "))  # Count the number of facilities
        else:
            facility_count[facility] = 0  # Fill missing values with 0
    
    region_counts.append(facility_count)

# Convert to DataFrame
region_facility_df = pd.DataFrame(region_counts)

# Calculate the total number of facilities per `region`
region_facility_df["total_facilities"] = region_facility_df[facility_types].sum(axis=1)

st.subheader("🏢 Facility Count by Region")

# Filter the selected `region`
filtered_df = region_facility_df[region_facility_df["region"].isin(selected_region)]

# Plot a stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))
filtered_df.set_index("region")[facility_types].plot(kind="bar", stacked=True, ax=ax, cmap="coolwarm")
ax.set_xlabel("Region")
ax.set_ylabel("Facility Count")
ax.set_title("Facility Count by Type in Each Region")
plt.xticks(rotation=45)
st.pyplot(fig)

# Facility count bar chart
st.subheader("📍 Facility Locations on Map")

m = folium.Map(location=[df["lt"].mean(), df["lg"].mean()], zoom_start=12)

# Add facility points to the map
for _, row in facilities_df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color="blue",
        fill=True,
        fill_opacity=0.7,
        popup=row["category"],
    ).add_to(m)

# Render the updated map in Streamlit
folium_static(m)

# --------------- 4. Data Analysis Charts ----------------

# Price distribution histogram
st.subheader("📊 Price Distribution")
fig, ax = plt.subplots()
sns.histplot(df_filtered["price"], bins=30, kde=True, ax=ax)
ax.set_xlabel("Price (C$)")
ax.set_ylabel("Count")
st.pyplot(fig)

# Price vs. Building Age
st.subheader("📈 Price vs. Building Age")
fig, ax = plt.subplots()
sns.scatterplot(data=df_filtered, x="building_age", y="price", hue="beds", size="baths", palette="coolwarm", ax=ax)
ax.set_xlabel("Building Age (Years)")
ax.set_ylabel("Price (C$)")
st.pyplot(fig)

# Average Price by Region
st.subheader("🏙️ Average Price by Region")
fig, ax = plt.subplots()
sns.barplot(data=df_filtered, x="region", y="price", ci=None, palette="Blues", ax=ax)
ax.set_xlabel("Region")
ax.set_ylabel("Average Price (C$)")
st.pyplot(fig)

# Price Comparison by Number of Bedrooms
st.subheader("🛏️ Price Comparison by Number of Bedrooms")
fig, ax = plt.subplots()
sns.boxplot(data=df_filtered, x="beds", y="price", palette="Set2", ax=ax)
ax.set_xlabel("Number of Bedrooms")
ax.set_ylabel("Price (C$)")
st.pyplot(fig)

# Exposure vs. Price
st.subheader("🌞 Exposure vs. Price")
fig, ax = plt.subplots()
sns.boxplot(data=df_filtered, x="exposure", y="price", palette="coolwarm", ax=ax)
ax.set_xlabel("Exposure")
ax.set_ylabel("Price (C$)")
st.pyplot(fig)

# --------------- 5. Key Financial Metrics ----------------
st.subheader("📊 Key Financial Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Market Value", f"${df_filtered['price'].sum():,.0f}")
col2.metric("🏡 Avg Price per Listing", f"${df_filtered['price'].mean():,.0f}")
col3.metric("📈 Median Price", f"${df_filtered['price'].median():,.0f}")
