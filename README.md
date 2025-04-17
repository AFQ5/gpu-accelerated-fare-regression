# 🚖 NYC Taxi Fare Prediction with GPU Acceleration

This project demonstrates the power of GPU-accelerated machine learning using the RAPIDS ecosystem and tree-based models like XGBoost and LightGBM. It showcases an end-to-end workflow for predicting taxi fares in New York City.

---

## 🧠 Project Description

The notebook utilizes the **NYC Taxi Fare Prediction** dataset and implements regression models using **cuDF**, **XGBoost (GPU)**, and **LightGBM** to predict taxi fares based on pickup and dropoff locations, time, and passenger count.

---

## 🛠️ Features

- ✅ Data preprocessing using `cuDF` for fast, GPU-based manipulation  
- 📅 Feature engineering from datetime and geolocation  
- 🚀 Model training using GPU-accelerated **XGBoost** and **LightGBM**  
- 📉 Evaluation using **RMSE** and **R²**  
- 📊 EDA visualizations to understand fare distribution and outliers  

---

## 🧩 Libraries Used

| Library   | Description                                       |
|-----------|---------------------------------------------------|
| `cuDF`    | GPU dataframe library (pandas-like)               |
| `cuML`    | GPU-accelerated ML algorithms                     |
| `XGBoost` | Tree boosting model, used in GPU mode             |
| `LightGBM`| Fast gradient boosting library                    |
| `Dask`    | Scalable parallel computing for big data          |

---

## 📊 Datasets Overview

### 1. 🟨 NYC Taxi Fare Prediction
- **Task:** Regression  
- **Source:** [Kaggle - NYC Taxi Fare Prediction](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction)  
- **Size:** 55M+ rows (Kaggle version)  
- **Description:** Predict taxi fare based on pickup/dropoff location, time, and passenger count.  

*Note: Dataset was removed from this repository to comply with data sharing policies. Please download it directly from the Kaggle competition page.*

---

## 📈 Model Summary

| Model            | Notes                     |
|------------------|---------------------------|
| XGBoost (GPU)    | Trained using CuPy arrays |
| LightGBM         | Trained on CPU for comparison |

---

## 📂 Project Structure

```bash
.
├── model/
│   ├── winner_xgb_model.json      # Trained XGBoost model
│   ├── cuml_rf_model.joblib       # Trained cuML Random Forest model
│   └── catboost_model.cbm         # Trained CatBoost model
├── NYC_Taxi_Fare_Prediction_GPU.ipynb   # Main notebook
├── README.md                            # Project documentation
└── LICENSE
```
---

## 🧪 Getting Started

To run this notebook:

1. Clone the repository  
2. Set up a GPU-enabled environment with RAPIDS  
3. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction)  
4. Open the notebook and run all cells  

---

## 📬 Contact

Made by **Abdullah Alqahtani** – Feel free to reach out for collaboration or feedback!

---

## 📘 License

This project is open source and available under the [MIT License](LICENSE).
