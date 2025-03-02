import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/new_info.csv")

# Shuffle the dataset if needed
df = df.sample(frac=1).reset_index(drop=True)

# Process categorical variables (convert to numerical values)
df["DEN"] = df["DEN"].map({"No": 0, "Yes": 1})
df["parking"] = df["parking"].map({"N": 0, "Yes": 1})
df["ward"] = df["ward"].map({"W10": 0, "W11": 1, "W12": 2})
df["size"] = df["size"].str.extract(r'(\d{3,4})').astype(float)  # Extract numerical part

# Map exposure directions to numerical values
exposure_mapping = {"N": 0, "E": 90, "S": 180, "We": 270, "No": -1}
df["exposure"] = df["exposure"].map(exposure_mapping).fillna(-1)  # Use -1 for missing exposure values

# Fill missing values in numerical columns with median values
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Encode `region` using Label Encoding
le = LabelEncoder()
df["region"] = le.fit_transform(df["region"])

# Select features and target variable
# ["DEN", "size", "parking", "beds", "baths", "D_mkt", "building_age", "maint", "exposure", "lt", "lg"]
# minHighwayDis, minHospitalDis, minMallDis, minParkDis, minPoliceDis, minSchoolDis, minStationDis

# Compute mean distance for hospital and categorize properties accordingly
mean_hospital_distance = df["minHospitalDis"].mean()
df["hasHospital"] = df["minHospitalDis"].apply(lambda x: 1 if x < mean_hospital_distance else 0)

# Compute mean distance for mall and categorize properties accordingly
mean_mall_distance = df["minMallDis"].mean()
df["hasMall"] = df["minMallDis"].apply(lambda x: 1 if x < mean_mall_distance else 0)

# Compute mean distance for park and categorize properties accordingly
mean_park_distance = df["minParkDis"].mean()
df["hasPark"] = df["minParkDis"].apply(lambda x: 1 if x < mean_park_distance else 0)

# Compute mean distance for police station and categorize properties accordingly
mean_police_distance = df["minPoliceDis"].mean()
df["hasPolice"] = df["minPoliceDis"].apply(lambda x: 1 if x < mean_police_distance else 0)

# Define feature columns
features = ["DEN", "size", "parking", "beds",
            "D_mkt", "building_age", "maint",
            "exposure", "lt", "lg", "minHighwayDis",
            "minMallDis", "minParkDis",
            "minPoliceDis", "minSchoolDis", "minStationDis",
            "hasHospital", "hasMall", "hasPark", "hasPolice", "crime_rate_per_100000_people"]

# Define target variable
target = "price"

X = df[features]
y = df[target]
regions = df["region"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, region_train, region_test = train_test_split(
    X, y, regions, test_size=0.2, random_state=42
)

# Convert data format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost parameters
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.08,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mae"
}

# Train the XGBoost model
evals = [(dtrain, "train"), (dtest, "eval")]
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=False
)

# Make predictions
y_pred = xgb_model.predict(dtest)

# Print original vs predicted price
print("Origin Price vs Predicted Price")
print(pd.DataFrame({"Origin Price": y_test, "Predicted Price": y_pred}))

# Compute overall error
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Overall MAE: {mae}")
print(f"Overall R² Score: {r2}")

# Compute R² score for each region
region_r2_scores = []
unique_regions = np.unique(region_test)

for region in unique_regions:
    mask = (region_test == region)
    if sum(mask) > 5:  # Ensure there is sufficient data in the test set for the region
        r2_region = r2_score(y_test[mask], y_pred[mask])
        region_r2_scores.append((region, r2_region))

# Convert results into a DataFrame
region_r2_df = pd.DataFrame(region_r2_scores, columns=["Region", "R² Score"])
print("R² Score by Region")
print(region_r2_df)
