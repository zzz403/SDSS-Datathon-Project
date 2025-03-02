import geopandas as gpd
import pandas as pd
import os

# 定义所有设施类型及其文件路径
geojson_files = {
    "highway": "parameters/highway.geojson",
    "hospital": "parameters/hospital.geojson",
    "park": "parameters/park.geojson",
    "police": "parameters/police.geojson",
    "school": "parameters/school.geojson",
    "station": "parameters/station.geojson",
    "supermarket": "parameters/supermarket.geojson"
}

# 存储所有设施数据
all_data = []

# 处理每个 `geojson` 文件
for category, file_path in geojson_files.items():
    if os.path.exists(file_path):  # 确保文件存在
        print(f"📌 处理 {category} 数据...")
        
        # 读取 `geojson`
        gdf = gpd.read_file(file_path)

        # 提取 `node_id`
        if "@id" in gdf.columns:
            gdf["node_id"] = gdf["@id"]
        else:
            print(f"⚠️ {category} 没有 '@id'，请检查 JSON 结构！")
            continue

        # 提取 `lat, lon`
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y

        # 仅保留 `node_id`, `lat`, `lon`, `category`
        gdf_cleaned = gdf[["node_id", "lat", "lon"]].dropna()
        gdf_cleaned["category"] = category  # 添加类别

        # 存入列表
        all_data.append(gdf_cleaned)

# 合并所有数据
final_df = pd.concat(all_data, ignore_index=True)

# 保存到 CSV
final_df.to_csv("all_facilities_data.csv", index=False)

print("✅ 所有 `geojson` 数据提取完成，已保存至 `all_facilities_data.csv`！")
