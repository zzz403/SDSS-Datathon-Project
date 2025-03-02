import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor  # 导入 XGBoost
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

# 读取数据
df = pd.read_csv("estate_with_region.csv")

# 处理分类变量（转换为数值）
df["DEN"] = df["DEN"].map({"No": 0, "Yes": 1})
df["parking"] = df["parking"].map({"N": 0, "Yes": 1})
df["size"] = df["size"].str.extract(r'(\d{3,4})').astype(float)  # 提取数值部分

exposure_mapping = {"N": 0, "E": 90, "S": 180, "We": 270, "No": -1}
df["exposure"] = df["exposure"].map(exposure_mapping).fillna(-1)  # 用 -1 代表无朝向

# 处理数值列的缺失值（填充中位数）
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 选择特征和目标变量
features = ["DEN", "size", "parking", "beds", "baths", "D_mkt", "building_age", "maint","exposure","lt","lg"]
target = "price"

# 训练 KNN 分类器，预测新房产属于哪个 region
X_knn = df[features]
y_knn = df["region"]

X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_knn, y_train_knn)

y_pred_knn = knn.predict(X_test_knn)  # 预测区域（region）

# 训练 XGBoost 回归模型（按区域划分）
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
    
    # 转换数据格式（XGBoost DMatrix）
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # XGBoost 参数
    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mae"
    }
    
    # 训练模型
    evals = [(dtrain, "train"), (dtest, "eval")]
    xgb_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False
    )

    y_pred = xgb_model.predict(dtest)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    mae_scores.append(mae)
    r2_scores.append(r2)
    region_models[region] = xgb_model
    trained_regions.append(region)

# 计算整体 MAE 和 R²
avg_mae = np.mean(mae_scores)
avg_r2 = np.mean(r2_scores)

# 展示结果
result_df = pd.DataFrame({"Region": trained_regions, "MAE": mae_scores, "R² Score": r2_scores})
print("Model Performance by Region")
print(result_df)

# 输出整体评估
print("Average MAE:", avg_mae)
print("Average R² Score:", avg_r2)
