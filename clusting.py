import numpy as np
import pandas as pd
import json
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from functools import reduce

def cal_dis(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半径，单位：km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])  # 转换为弧度

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c  # 返回单位为 km


# ---------- main --------------

df = pd.read_csv("real-estate-data.csv")

# ---------------- 1.1 some data adding ----------------

# 选择分类区域的数量
num_clusters = 25  # 你可以调整这个数字来控制区域的数量

# 只使用经纬度进行聚类
coords = df[['lt', 'lg']].dropna()  # 移除 NaN 值，防止错误

# 进行 KMeans 聚类
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df.loc[coords.index, 'region'] = kmeans.fit_predict(coords)

# 将区域列转换为整数类型
df['region'] = df['region'].astype(int)

df.to_csv("estate_with_region.csv", index=False)

####################################################

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(coords)


# hospital
def get_nearest_region_info(file_name):
    
    region_info = {}
    with open(f"./parameters/{file_name}.geojson", "r") as highway_f:
        highway_info = json.load(highway_f)['features']

        for highway in highway_info:
            target_lon = highway['geometry']['coordinates'][0]  # 经度
            target_lat = highway['geometry']['coordinates'][1]  # 纬度

            # 查询最近的房产点（修正格式）
            target_point = np.array([[target_lat, target_lon]])  # 修正传入数据格式
            distance, idx = nbrs.kneighbors(target_point)
        
            # 找到最近的 region
            nearest_region = df.iloc[idx[0][0]]['region']
            if nearest_region not in region_info:
                region_info[nearest_region] = []
            region_info[nearest_region].append(highway.get("id", "unknown"))  # 防止 id 不存在
    return region_info

# highway
highway_temp = get_nearest_region_info("highway")
highway_info = pd.DataFrame({
    'region': list(highway_temp.keys()), 
    'highway_ids': [", ".join(map(str, v)) for v in highway_temp.values()]
})

hospital_temp = get_nearest_region_info("hospital")
hospital_info = pd.DataFrame({
    'region': list(hospital_temp.keys()), 
    'hospital_ids': [", ".join(map(str, v)) for v in hospital_temp.values()]
})

mall_temp = get_nearest_region_info("mall+supermarket+marketplace")
mall_info = pd.DataFrame({
    'region': list(mall_temp.keys()), 
    'mall_ids': [", ".join(map(str, v)) for v in mall_temp.values()]
})

park_temp = get_nearest_region_info("park")
park_info = pd.DataFrame({
    'region': list(park_temp.keys()), 
    'park_ids': [", ".join(map(str, v)) for v in park_temp.values()]
})

police_temp = get_nearest_region_info("police")
police_info = pd.DataFrame({
    'region': list(police_temp.keys()), 
    'police_ids': [", ".join(map(str, v)) for v in police_temp.values()]
})

school_temp = get_nearest_region_info("school")
school_info = pd.DataFrame({
    'region': list(school_temp.keys()), 
    'school_ids': [", ".join(map(str, v)) for v in school_temp.values()]
})

station_temp = get_nearest_region_info("station")
station_info = pd.DataFrame({
    'region': list(station_temp.keys()), 
    'station_ids': [", ".join(map(str, v)) for v in station_temp.values()]
})

dfs = [highway_info, hospital_info, mall_info, park_info, police_info, school_info, station_info] 

result_df = reduce(lambda left, right: pd.merge(left, right, on='region', how='outer'), dfs)

result_df["region"] = result_df["region"].astype(int)
result_df = result_df.sort_values(by="region")

result_df.to_csv("region_info.csv", index=False)

