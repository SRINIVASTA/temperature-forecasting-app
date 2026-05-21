# 🌡️ Temperature Forecasting App

A Streamlit web application that provides historical climate summaries and uses Machine Learning models to forecast average daily temperatures. It comes pre-loaded with local default weather data and supports custom user dataset uploads.

---

## 🚀 Features

*   **Dual Data Input System**: 
    *   Automatically detects and loads a default local dataset (`weather_data.csv`).
    *   Allows users to dynamically upload custom files to override the default system.
*   **Interactive Sidebar Filters**: Sort and view metrics by specific years or calendar months.
*   **Aggregated Summaries**: Instantly computes average temperature parameters and total precipitation amounts across Overall, Yearly, or Monthly timelines.
*   **Machine Learning Engines**: Select between **Random Forest**, **Gradient Boosting**, or **XGBoost** regressor models.
*   **Visual Charts**: Dynamic line charts for temporal patterns, actual vs. predicted timeline plots, and model feature importance tables.

---

## 📂 Project Repository Structure

```text
├── streamlit_app.py      # Main Streamlit application source script
├── weather_data.csv      # Default fallback local dataset file
├── requirements.txt      # Python system dependencies manifest
└── README.md             # Project documentation (This file)
```

---

## 📊 CSV Data Schema Format

Any dataset provided or uploaded must contain commas separating values with these exact case-sensitive column headers:

```csv
Date,Rain,Temp Max,Temp Min
2026-05-01,0.0,32.5,22.1
2026-05-02,4.5,29.0,21.5
```

> ⚠️ **Note**: Your dataset must contain **at least 10 valid chronological rows** after applying sidebar filtering to train the machine learning architectures successfully.

---

## 🛠️ Local Installation & Execution

Follow these steps to run the application on your local laptop environment:

1. **Clone or download** this repository to your local directory.
2. **Install all required dependencies** via pip matching the environment specifications:
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute the execution command** inside your project folder:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 🌐 Streamlit Cloud Deployment

When deploying this project directly to the **Streamlit Community Cloud Platform**:
1. Commit `streamlit_app.py`, `requirements.txt`, and your baseline default `weather_data.csv` to a public GitHub repository.
2. Link the repository to your Streamlit Cloud account dashboard.
3. Keep the Main File Path designated explicitly as `streamlit_app.py`.
