import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt

# Feature Engineering
def feature_engineering(df):
    df['Temp Avg'] = (df['Temp Max'] + df['Temp Min']) / 2
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    return df

# Train Model
def train_model(X_train, y_train, model_name):
    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    st.markdown("<h1 style='text-align: center;'>🌡️ Temperature Forecasting with Different Models</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload a CSV file (Date, Rain, Temp Max, Temp Min)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Parse date
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Clean up Rain column
        df['Rain'] = pd.to_numeric(df['Rain'], errors='coerce').fillna(0)

        # Drop rows with missing values
        df = df.dropna(subset=['Date', 'Temp Max', 'Temp Min', 'Rain'])

        # Feature engineering
        df = feature_engineering(df)
        df = df.sort_values("Date")
        df = df.dropna(subset=["Temp Avg"])

        # Sidebar Filters
        st.sidebar.header("🔎 Filter Options")

        years = sorted(df['year'].unique())
        selected_year = st.sidebar.selectbox("Select Year", options=["All"] + years, index=0)

        months = list(range(1, 13))
        month_names = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June",
                       7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"}
        # Display month names, internally filter by month number
        month_labels = ["All"] + [month_names[m] for m in months]
        selected_month_label = st.sidebar.selectbox("Select Month (Optional)", options=month_labels, index=0)
        selected_month = None if selected_month_label == "All" else {v:k for k,v in month_names.items()}[selected_month_label]

        summary_type = st.sidebar.radio("Summary Type", ["Overall", "Yearly", "Monthly"])

        # Filter data based on selection
        filtered_df = df.copy()
        if selected_year != "All":
            filtered_df = filtered_df[filtered_df['year'] == selected_year]
        if selected_month is not None:
            filtered_df = filtered_df[filtered_df['month'] == selected_month]

        # Summary Display
        if summary_type == "Yearly":
            summary = filtered_df.groupby("year").agg({
                "Temp Avg": "mean",
                "Rain": "sum"
            }).reset_index()
            st.write("📅 **Yearly Summary**")
            st.dataframe(summary)
            st.line_chart(summary.set_index("year")["Temp Avg"])

        elif summary_type == "Monthly":
            summary = filtered_df.groupby("month").agg({
                "Temp Avg": "mean",
                "Rain": "sum"
            }).reset_index()
            # Sort by month (calendar order)
            summary = summary.sort_values("month")
            summary["Month"] = summary["month"].map(month_names)
            st.write("📅 **Monthly Summary**")
            st.dataframe(summary[["Month", "Temp Avg", "Rain"]])
            # Make Month categorical with correct order for line chart
            summary['Month'] = pd.Categorical(summary['Month'], categories=[month_names[m] for m in months], ordered=True)
            st.line_chart(summary.sort_values('Month').set_index("Month")["Temp Avg"])

        else:
            temp_avg = filtered_df["Temp Avg"].mean()
            rain_total = filtered_df["Rain"].sum()
            st.write("📊 **Overall Summary**")
            st.metric("Average Temperature", f"{temp_avg:.2f}°C")
            st.metric("Total Rainfall", f"{rain_total:.2f} mm")

        # Model Training and Forecasting
        st.divider()
        st.subheader("📈 Forecasting Model")

        if len(filtered_df) < 10:
            st.warning("Not enough data for model training. Try a different filter or upload more data.")
            return

        features = ['year', 'month', 'day', 'Rain']
        target = 'Temp Avg'

        split_index = int(0.8 * len(filtered_df))
        train_df = filtered_df.iloc[:split_index]
        test_df = filtered_df.iloc[split_index:]

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        model_name = st.selectbox("🤖 Choose a model", ["Random Forest", "Gradient Boosting", "XGBoost"])
        model = train_model(X_train, y_train, model_name)

        predictions = model.predict(X_test)

        # Check for NaNs
        if np.isnan(predictions).any() or np.isnan(y_test.values).any():
            st.error("❌ Error: NaNs found in predicted or actual values. Please clean your dataset.")
            return

        mae = mean_absolute_error(y_test, predictions)
        st.success(f"📉 Mean Absolute Error (MAE): {mae:.2f}")

        # Plot actual vs predicted with x-axis labels rotated 45 degrees
        st.write("### 📊 Actual vs Predicted Temperatures")
        fig, ax = plt.subplots()
        ax.plot(test_df['Date'], y_test.values, label="Actual", marker='o')
        ax.plot(test_df['Date'], predictions, label="Predicted", marker='x')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Feature Importance
        st.write("### 📌 Feature Importance")
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({'feature': features, 'importance': importance})
            st.bar_chart(importance_df.set_index('feature'))
        else:
            st.info("Feature importance not available for this model.")

if __name__ == "__main__":
    main()
