# Datathon Project - HomeRadar

**Toronto Datathon 2024**  
**Date:** 3/2/2025  
**Team Name:** ByteBuilders  
**Team Members:**  
- Zhongze (August) Zheng [[GitHub](https://github.com/zzz403)]
- Zhengyu (Joey) Wang [[GitHub](https://github.com/wzy403)]  
- Zixiang (Terry) Huang [[GitHub](https://github.com/trrrrrrry)]  
- Yuqing (Janice) Liu [[GitHub](https://github.com/LiuYuqing14)]



## Project Overview

- **Problem Statement:** The Toronto real estate market is highly dynamic, influenced by factors such as neighborhood amenities, public transit proximity, economic conditions, and market trends. However, predicting real estate prices accurately remains a significant challenge due to market complexity, missing data, and potential biases. The goal of this project is to develop a machine learning model that can accurately predict real estate prices across Toronto's diverse neighborhoods by analyzing various influencing factors.

- **Solution Summary:** To address this challenge, our project utilizes data science and machine learning techniques to develop a real estate price prediction model. Our solution involves:

    - Comprehensive Data Analysis – Identifying key factors affecting property prices, such as location, building age, and amenities.
    - Feature Engineering & Model Development – Implementing a robust machine learning model (e.g., XGBoost, Random Forest) to generate accurate price predictions.
    - Visualization & Insights – Presenting findings through interactive data visualizations to help homebuyers, investors, and real estate agents make informed decisions.
    - Performance Evaluation & Optimization – Ensuring high accuracy through hyperparameter tuning and handling missing data effectively.

- **Datathon Theme Alignment:** This project aligns with the Toronto Datathon themes by:
    -  Providing actionable insights into the real estate market – Helping users understand how market trends, location, and property attributes affect housing prices.
    - Leveraging data science and AI – Using machine learning models to analyze housing data, predict future prices, and uncover market patterns.
    -  Supporting informed decision-making – Offering insights for homebuyers, investors, and urban planners to optimize their strategies.
- **Key Features:** 
    - Data-Driven Price Prediction – A machine learning model trained on real estate data to forecast property prices accurately.
    - Location-Based Analysis – Evaluating the impact of neighborhood amenities, public transport access, and crime rates on real estate prices.
    - Market Trend Insights – Identifying key trends in price fluctuations, the influence of economic conditions, and property type comparisons.
    - Interactive Visualizations – Dynamic dashboards displaying real estate trends, pricing distribution, and geographic insights for better decision-making.
    - Scalable & Reproducible Model – Well-documented machine learning pipeline that can be expanded for future real estate analyses.



## Dataset & Data Processing

- **Dataset Source:** The dataset used in this project includes real estate listings and market data for Toronto. The data comes from:

    - Competition-provided dataset – Official Datathon dataset containing property details, pricing, and neighborhood features.
    - Publicly available datasets – Supplementary data sources such as:

- **Data Cleaning & Preprocessing:** To ensure a high-quality dataset, we performed comprehensive data preprocessing, including:
    - Handling Missing Values – Imputed missing values using median imputation for numerical features (e.g., building age, lot size) and mode imputation for categorical variables.
    - Feature Engineering – Created new variables such as distance to nearest subway station, crime rate per neighborhood, and property price per square foot.
    - Outlier Detection & Removal – Identified extreme property prices using IQR-based filtering to remove unrealistic listings.


## Installation & Setup

### 1️. Clone the repository
```sh
git clone https://github.com/zzz403/SDSS-Datathon-Project.git
cd datathon-project
```

### 2️. Install dependencies
```sh
pip install -r requirements.txt
```

### 3️. Run the project
```sh
streamlit run app.py
```



## Model & Algorithm

- **Model Used:** We experimented with multiple models and selected the most effective one based on performance evaluation:

    - XGBoost (Extreme Gradient Boosting) - Final model

    - K-Nearest Neighbors (KNN) - Used for region classification

    - Linear Regression (Baseline Model) - For performance comparison

    - Random Forest (Initial Experimentation) - Tried but not selected
- **Hyperparameter Tuning:** To optimize our model, we performed **Grid Search and Cross-Validation** on key hyperparameters:

| Hyperparameter          | Tuned Value | Purpose |
|-------------------------|------------|------------------------------------------------|
| `learning_rate`        | 0.08       | Controls step size for each boosting iteration |
| `max_depth`           | 8          | Limits tree depth to prevent overfitting |
| `subsample`           | 0.8        | Uses a fraction of the training data per boosting round |
| `colsample_bytree`    | 0.8        | Selects a subset of features per tree |

- **Performance Metrics:** 

To evaluate model effectiveness, we used the following key metrics:

| Model                | R² Score |
|---------------------|-----------------------------|
| **XGBoost**        | **0.90**                    |
| Random Forest      | 0.85                         |
| KNN               | 0.75                         |
| Linear Regression  | 0.68                         |

*For a comprehensive report, please see our official [report-file-name](./report-file-name).*


## Evaluation Criteria Breakdown

| Criterion    | Score (Out of 20%) | Explanation |
|-------------|-------------------|-------------|
| **Relevance** | ✅ | Clearly aligned with [Datathon theme] |
| **Feasibility** | ✅ | Solution is practical and implementable |
| **Creativity** | ✅ | Unique approach and visualization |
| **Code Quality** | ✅ | Code is well-structured and documented |
| **Executability** | ✅ | The project runs without major issues |


## **🛠 Challenges & Future Improvements**

✔ **Challenges:**  
- How to deal with a bunch of raw data in file to find the correlation between each variable and make a future prediction
on the targeted data is the big problem we are facing. Therefor, the first thing we did was data cleaning and feature 
engineering.
- Which model is best fit? That's need a lot of accuracy comparison and evaluation on each model. We tried to use 
different types of models，like supervised-learning and clustering model, and evaluate by different types of graph.

✔ **Future Improvements:**  
- To enhance model diversity and robustness, we propose using the Random Subspace Method for future adjustments, which 
could mitigate imbalanced feature of our model and improve generalization.
- The sample size of our model is small. To generate a more accurate prediction, work on higher size of sample can achieve 
our goal! 


## License

This project is licensed under the GNU License. For more details, please refer to the [LICENSE](./LICENSE) file.

Information regarding third-party licenses can be found in the [THIRD-PARTY](./THIRD-PARTY) file.
