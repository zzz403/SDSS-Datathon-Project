import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 读取数据
import pandas as pd

# 读取数据
df = pd.read_csv("estate_with_region.csv")
# 选择特征和目标变量
# DEN,size,parking,exposure

df["DEN"] = df["DEN"].map({"No": 0, "Yes": 1})
df["parking"] = df["parking"].map({"N": 0, "Yes": 1})
df["size"] = df["size"].str.extract(r'(\d{3,4})').astype(float)  # 提取数值部分
exposure_mapping = {
    "N": 0,  # 北
    "E": 90,  # 东
    "S": 180,  # 南
    "We": 270,  # 西边
    "No": -1  # 代表无朝向
}
df["exposure"] = df["exposure"].map(exposure_mapping).fillna(-1)  # 用 -1 代表无朝向


# 处理数值列的缺失值（填充中位数）
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 处理分类列的缺失值（填充众数）
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])


features = ["DEN","size","parking","beds", "baths", "D_mkt", "building_age", "maint"]
target = "price"

# 1️⃣ 先训练一个 KNN 分类器，预测新房产属于哪个 region
X_knn = df[features]
y_knn = df["region"]

# 拆分训练集和测试集
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

# how many each regions in the test set

# 训练 KNN 分类器
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_knn, y_train_knn)

# 预测区域（region）
y_pred_knn = knn.predict(X_test_knn)

# 2️⃣ 在每个 region 内训练 Decision Tree 回归模型
region_models = {}
mae_scores = []
r2_scores = []

trained_regions = []

for region in df["region"].unique():
    df_region = df[df["region"] == region]

    if len(df_region) < 10:  # 数据量太少的区域跳过
        continue
    X = df_region[features]
    y = df_region[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练 Decision Tree 回归模型
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)

    # 预测
    y_pred = dt.predict(X_test)

    print(f"Region: {region}, y_pred: {y_pred}")

    # 计算 MAE 和 R²
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mae_scores.append(mae)
    r2_scores.append(r2)

    # 存储模型
    region_models[region] = dt

    trained_regions.append(region)

# 计算整体 MAE 和 R²
avg_mae = np.mean(mae_scores)
avg_r2 = np.mean(r2_scores)

# 展示结
result_df = pd.DataFrame({"Region": list(region_models.keys()), "MAE": mae_scores, "R² Score": r2_scores})
print("Model Performance by Region")
print(result_df)

# 输出整体评估
avg_mae, avg_r2
