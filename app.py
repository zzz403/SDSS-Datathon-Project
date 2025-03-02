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

# --------------- 1. 加载数据 ----------------
st.title("🏡 Toronto Real Estate Analysis Dashboard")


# 读取CSV数据
df = pd.read_csv("new_info.csv")

# ---------------- 1.1 some data adding ----------------
# 将区域列转换为整数类型
df['region'] = pd.to_numeric(df["region"], errors='coerce')

# 确保 `region` 是字符串格式
df["region"] = df["region"].astype(str)

# 读取设施数据
facilities_df = pd.read_csv("all_facilities_data.csv")

# 统计每个 region 的设施数量
facility_counts = facilities_df.groupby("category")["node_id"].count().reset_index()
facility_counts.columns = ["category", "count"]

# 计算每个 region 到最近设施的平均距离
facility_distance = facilities_df.groupby("category")["lat"].mean().reset_index()
facility_distance.columns = ["category", "avg_lat"]

# 设施数据加入 df
df = df.merge(facility_counts, left_on="region", right_on="category", how="left")
df = df.merge(facility_distance, left_on="region", right_on="category", how="left")

# 处理 NaN（有些 `region` 可能缺少某些设施）
df.fillna(0, inplace=True)


# 处理NaN和inf：
df["price"] = pd.to_numeric(df["price"], errors='coerce')  # 确保列是数值型，非数值转换为 NaN
df["price"] = df["price"].replace([np.inf, -np.inf], np.nan)  # 将无穷值替换为 NaN
df["price"] = df["price"].fillna(df["price"].median()).astype(int)  # 用中位数填充 NaN，然后转换为整数

# 其他列同样处理：
for col in ["building_age", "beds", "baths"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # 转换为数值型
    df[col] = df[col].fillna(df[col].median()).astype(int)  # 用中位数填充 NaN

# 计算房价范围
price_min, price_max = int(df["price"].min()), int(df["price"].max())

# --------------- 2. 侧边栏筛选器 ----------------
st.sidebar.header("🔍 Filter Options")

# region
selected_region = st.sidebar.multiselect("🗺️ Select Regions", df["region"].unique(), default=df["region"].unique())

# 价格范围
price_range = st.sidebar.slider("💰 Select Price Range (C$)", price_min, price_max, (price_min, price_max))

# 卧室数
bedrooms = st.sidebar.multiselect("🛏️ Number of Bedrooms", df["beds"].unique(), default=df["beds"].unique())

# 建筑年龄范围
age_min, age_max = int(df["building_age"].min()), int(df["building_age"].max())
building_age_range = st.sidebar.slider("🏗️ Select Building Age", age_min, age_max, (age_min, age_max))

# 应用筛选器
df_filtered = df[
    (df["region"].isin(selected_region)) &
    (df["price"] >= price_range[0]) & (df["price"] <= price_range[1]) &
    (df["beds"].isin(bedrooms)) &
    (df["building_age"] >= building_age_range[0]) & (df["building_age"] <= building_age_range[1])
]

# 显示筛选后的数据
st.write(f"📊 {len(df_filtered)} properties match the selected filters.")

# 设施数量条形图
st.subheader("🏢 Facilities Count by Category")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=facility_counts, x="category", y="count", palette="coolwarm", ax=ax)
ax.set_xlabel("Facility Type")
ax.set_ylabel("Count")
st.pyplot(fig)

st.subheader("📍 Facility Locations on Map")

m = folium.Map(location=[df["lt"].mean(), df["lg"].mean()], zoom_start=12)

# 添加设施点
for _, row in facilities_df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color="blue",
        fill=True,
        fill_opacity=0.7,
        popup=row["category"],
    ).add_to(m)

folium_static(m)


# --------------- 3. 地图可视化 ----------------
st.subheader("📍 Real Estate Location Map")

m = folium.Map(location=[df["lt"].mean(), df["lg"].mean()], zoom_start=12)

# 设定价格颜色编码
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

# --------------- 4. 数据分析图表 ----------------

# 价格分布直方图
st.subheader("📊 Price Distribution")
fig, ax = plt.subplots()
sns.histplot(df_filtered["price"], bins=30, kde=True, ax=ax)
ax.set_xlabel("Price (C$)")
ax.set_ylabel("Count")
st.pyplot(fig)

# 房价 vs. 建筑年龄
st.subheader("📈 Price vs. Building Age")
fig, ax = plt.subplots()
sns.scatterplot(data=df_filtered, x="building_age", y="price", hue="beds", size="baths", palette="coolwarm", ax=ax)
ax.set_xlabel("Building Age (Years)")
ax.set_ylabel("Price (C$)")
st.pyplot(fig)

# 不同选区（region）房价均值
st.subheader("🏙️ Average Price by Region")
fig, ax = plt.subplots()
sns.barplot(data=df_filtered, x="region", y="price", ci=None, palette="Blues", ax=ax)
ax.set_xlabel("region")
ax.set_ylabel("Average Price (C$)")
st.pyplot(fig)

# 不同房型（卧室数）价格对比
st.subheader("🛏️ Price Comparison by Number of Bedrooms")
fig, ax = plt.subplots()
sns.boxplot(data=df_filtered, x="beds", y="price", palette="Set2", ax=ax)
ax.set_xlabel("Number of Bedrooms")
ax.set_ylabel("Price (C$)")
st.pyplot(fig)

# 曝光方向 vs. 房价
st.subheader("🌞 Exposure vs. Price")
fig, ax = plt.subplots()
sns.boxplot(data=df_filtered, x="exposure", y="price", palette="coolwarm", ax=ax)
ax.set_xlabel("Exposure")
ax.set_ylabel("Price (C$)")
st.pyplot(fig)

# --------------- 5. 关键财务指标 ----------------
st.subheader("📊 Key Financial Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Market Value", f"${df_filtered['price'].sum():,.0f}")
col2.metric("🏡 Avg Price per Listing", f"${df_filtered['price'].mean():,.0f}")
col3.metric("📈 Median Price", f"${df_filtered['price'].median():,.0f}")

