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
    
    house_info = {}
    with open(f"./parameters/{file_name}.geojson", "r") as highway_f:
        file_info = json.load(highway_f)['features']

        for info in file_info:
            target_lon = info['geometry']['coordinates'][0]  # 经度
            target_lat = info['geometry']['coordinates'][1]  # 纬度

            for house in df.itertuples():
                if pd.isna(house.lt) or pd.isna(house.lg):
                    continue
                distance = cal_dis(house.lt, house.lg, target_lat, target_lon)
                distance = round(distance, 2)
                if house.id_ not in house_info:
                    house_info[house.id_] = distance
                house_info[house.id_] = min(house_info[house.id_], distance)
                
    return house_info

# highway
highway_temp = get_nearest_region_info("highway")
highway_info = pd.DataFrame({
    'id_': list(highway_temp.keys()), 
    'hasHighway': [v for v in highway_temp.values()]
})

hospital_temp = get_nearest_region_info("hospital")
hospital_info = pd.DataFrame({
    'id_': list(hospital_temp.keys()), 
    'hasHospital': [v for v in hospital_temp.values()]
})

mall_temp = get_nearest_region_info("mall+supermarket+marketplace")
mall_info = pd.DataFrame({
    'id_': list(mall_temp.keys()),
    'hasMall': [v for v in mall_temp.values()]
})

park_temp = get_nearest_region_info("park")
park_info = pd.DataFrame({
    'id_': list(park_temp.keys()), 
    'hasPark': [v for v in park_temp.values()]
})

police_temp = get_nearest_region_info("police")
police_info = pd.DataFrame({
    'id_': list(police_temp.keys()), 
    'hasPolice': [v for v in police_temp.values()]
})

school_temp = get_nearest_region_info("school")
school_info = pd.DataFrame({
    'id_': list(school_temp.keys()), 
    'hasSchool': [v for v in school_temp.values()]
})

station_temp = get_nearest_region_info("station")
station_info = pd.DataFrame({
    'id_': list(station_temp.keys()), 
    'hasStation': [v for v in station_temp.values()]
})

dfs = [df, highway_info, hospital_info, mall_info, park_info, police_info, school_info, station_info] 

result_df = reduce(lambda left, right: pd.merge(left, right, on='id_', how='outer'), dfs)

# result_df["region"] = result_df["region"].astype(int)
# result_df = result_df.sort_values(by="region")

result_df.to_csv("new_info.csv", index=False)

