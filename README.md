
# 🚖 NYC Taxi Fare Prediction Dashboard  

A Machine Learning project that predicts **taxi fares in NYC** based on trip details such as pickup/dropoff locations, datetime, and passenger count. The project also includes an **interactive Streamlit dashboard** for visualization and real-time predictions.  

---

## 📌 Features  
- **Data Preprocessing & Cleaning** – Handling missing values, outliers, and feature engineering.  
- **Exploratory Data Analysis (EDA)** – Geospatial insights, time-based analysis, and trip patterns.  
- **Model Training & Evaluation** – Regression models including Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting.  
- **Model Comparison** – Evaluated using R², MSE, RMSE, and MAE.  
- **Interactive Dashboard (Streamlit)** –  
  - User inputs pickup/dropoff location & time.  
  - Model predicts estimated fare.  
  - Warns user if past datetime is selected.  

---

## 📂 Project Structure  
```
📦 URBAN_TAXI_FARE/
│── app/
│   └── streamlit_app.py
│
│── data/
│   ├── taxi_fare.csv
│   ├── tripfare_cleaned.csv
│   └── tripfare_feature_engg.csv
│
│── Models/
│   └── best_taxi_fare_model.pkl
│
│── notebooks/
│   ├── EDA.ipynb
│   ├── feature_engg.ipynb
│   └── models.ipynb
│
│── requirements.txt
│── README.md
│── LICENSE (optional, e.g. MIT)
```

---

## 📊 Model Evaluation Results  

| Model              | R²      | MSE      | RMSE   | MAE   |
|--------------------|---------|----------|--------|-------|
| Linear Regression  | 1.0000  | ~0       | ~0     | ~0    |
| Ridge              | 1.0000  | 5.1e-06  | 0.0022 | 0.0001|
| Random Forest      | 0.9993  | 0.104    | 0.323  | 0.021 |
| Gradient Boosting  | 0.9992  | 0.129    | 0.360  | 0.173 |
| Lasso              | 0.9944  | 0.917    | 0.958  | 0.504 |

---

## 📜 License  
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  

---
