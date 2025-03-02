import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 读取数据
df = pd.read_csv("estate_with_region.csv")

# 处理分类变量（转换为数值）
df["DEN"] = df["DEN"].map({"No": 0, "Yes": 1})
df["parking"] = df["parking"].map({"N": 0, "Yes": 1})
df["ward"] = df["ward"].map({"W10": 0,"W11": 1,"W12":2})
df["size"] = df["size"].str.extract(r'(\d{3,4})').astype(float)  # 提取数值部分

exposure_mapping = {"N": 0, "E": 90, "S": 180, "We": 270, "No": -1}
df["exposure"] = df["exposure"].map(exposure_mapping).fillna(-1)  # 用 -1 代表无朝向

# 处理数值列的缺失值（填充中位数）
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 处理 `region` 变量（使用 Label Encoding）
le = LabelEncoder()
df["region"] = le.fit_transform(df["region"])

# 选择特征和目标变量
# ["DEN", "size", "parking", "beds", "baths", "D_mkt", "building_age", "maint", "exposure", "lt", "lg"]
features = ["size", "parking", "beds", "D_mkt", "building_age", "maint", "exposure", "lt", "lg"]
target = "price"

X = df[features]
y = df[target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换数据格式（XGBoost DMatrix）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost 参数
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.08,
    "max_depth": 8,
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

# 进行预测
y_pred = xgb_model.predict(dtest)

# 计算误差
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"R² Score: {r2}")
