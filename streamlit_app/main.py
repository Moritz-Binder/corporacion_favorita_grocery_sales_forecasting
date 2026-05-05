import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from darts import TimeSeries
from darts.models import Prophet
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import os
import sys
from pathlib import Path
import streamlit as st
import datetime

# Add the project root to the python path
sys.path.append(os.path.abspath(".."))
from src.processing import DateFeatureTransformer, TimeSeriesWrangler, LagFeatureTransformer, WindowFeatureTransformer

BASE_DIR = Path(__file__).resolve().parent
PROJECT_PATH = BASE_DIR.parent

@st.cache_data
def data_prep(path, date, lags_var, windows_var, target, max_forecast_horizon):
    #-----------------
    # Define Functions
    #-----------------
    def forecasting_pipeline(model, y_train, x_dates, num_days, test, test_dates, lag_list, rolling_list, target, selected_features):
        """
        Upgraded recursive forecast with dynamic lag and rolling window generation.
        """

        # Create history starting with the end of training data
        history = y_train.copy().set_index(x_dates)
        xogen = test.copy().set_index(test_dates)
        last_date = history.index[-1]
        future = []

        for i in range(num_days):
            next_date = last_date + pd.Timedelta(days=1)
            feature_dict = {}

            # --- Dynamic Lags ---
            for lag in lag_list:
                feature_dict[f'{target}_lag_{lag}'] = [history[target].iloc[-lag]]

            # --- Dynamic Rolling Stats ---
            for window in rolling_list:
                feature_dict[f'{target}_rolling_{window}_mean'] = [history[target].iloc[-window:].mean()]
                feature_dict[f'{target}_rolling_{window}_std'] = [history[target].iloc[-window:].std()]

            # Create row DataFrame
            row = pd.DataFrame(data=feature_dict, index=[next_date])
            
            # Merge with exogenous variables (holidays, prices, etc.)
            xogen = xogen.loc[:, ~xogen.columns.str.contains('lag|rolling', case=False)]
            rows = row.merge(xogen, how='left', left_index=True, right_index=True)
            
            # CRITICAL: Ensure column order matches exactly what the model was trained on
            #list_columns = [c for c in rows.columns if c != target]
            rows = rows[selected_features]

            # This assumes the test dataframe columns are in the correct order
            if rows.isnull().any().any():
                rows = rows.fillna(0)

            # Predict and update history
            y_hat = model.predict(rows)[0]
            future.append(y_hat)
            
            history = pd.concat([history, pd.DataFrame(data={'unit_sales': y_hat}, index=[next_date])])
            last_date = next_date

        return future
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    def get_model_metadata_by_tags(tag_dict: dict):
        """
        Finds a model based on metadata tags and returns its name 
        and the parameters used during its training run.
        """
        
        # 1. Build the filter string (Same as your original logic)
        filters = [f"tags.{k} = '{v}'" for k, v in tag_dict.items()]
        filter_string = " AND ".join(filters)
        
        # 2. Search the registry
        results = client.search_registered_models(filter_string=filter_string)
        
        if not results:
            raise ValueError(f"No model found matching: {tag_dict}")
        
        # 3. Target the first match and get the latest version
        target_name = results[0].name
        # We fetch the latest version to find the specific Run ID associated with it
        latest_versions = client.get_latest_versions(target_name, stages=["None", "Production"])
        
        if not latest_versions:
            raise ValueError(f"No versions found for model: {target_name}")
            
        latest_v_info = latest_versions[0]
        run_id = latest_v_info.run_id
        version_num = latest_v_info.version

        # 4. Fetch the Run data to get parameters
        # This is the "Lineage" jump: Registry -> Tracking Server
        run_data = client.get_run(run_id)
        params = run_data.data.params

        print(f"Metadata retrieved for: {target_name} (v{version_num}) from Run: {run_id}")
        
        return {
            "model_name": target_name,
            "version": version_num,
            "run_id": run_id,
            "parameters": params
        }

    def build_fit_model(dict, data, target, date, max_forecast_horizon):
        int_params = [
                'n_estimators', 'max_depth', 'min_samples_split', 'max_iter', 'min_samples_leaf', 'n_jobs', 'random_state'
            ]
        float_params = ['max_features', 'changepoint_range', 'seasonality_prior_scale', 'holidays_prior_scale', 'changepoint_prior_scale']
        clean_params = {}
        for k, v in dict["parameters"].items():
            # Standard integer casting
            if k in int_params:
                clean_params[k] = int(float(v))
            elif k in float_params:
                clean_params[k] = float(v)
            else:
                clean_params[k] = v
        
        if "RandomForest" in dict["model_name"]:
            model = RandomForestRegressor(**clean_params)
        elif "XGBoost" in dict["model_name"]:
            model = xgb.XGBRegressor(**clean_params)
        elif "Prophet" in dict["model_name"]:
            model = Prophet(**clean_params)
        else:
            raise ValueError(f"Unsupported model type in metadata: {dict['model_name']}")
        
        if "Prophet" not in dict["model_name"]:
            model.fit(data.drop(columns=[target, date])[:-max_forecast_horizon], data[target][:-max_forecast_horizon])
        else:
            series = TimeSeries.from_dataframe(data[:-max_forecast_horizon], date, target)
            future_covariance = TimeSeries.from_dataframe(data.drop(columns=[target])[:-max_forecast_horizon], date,data.loc[:, ~data.columns.str.contains('lag|rolling|index', case=False)].drop(columns=[target, date]).columns.to_list())
            model.fit(series, future_covariance)

        if "Prophet" not in dict["model_name"]:
            selected_columns = data.drop(columns=[target, date]).columns.to_list()
        else:
            selected_columns = data.loc[:, ~data.columns.str.contains('lag|rolling|index', case=False)].drop(columns=[target, date]).columns.to_list()

        return (model, selected_columns)
        

    #---------------------
    # Get best models dict
    #---------------------

    # Random Forest Model
    my_tags = {
        "forecast": "weekly",
        "active": "true"
    }

    weekly_model = get_model_metadata_by_tags(my_tags)

    # XGBoost Model
    my_tags = {
        "forecast": "monthly",
        "active": "true"
    }

    monthly_model = get_model_metadata_by_tags(my_tags)

    # Prophet Model
    my_tags = {
        "forecast": "quarterly",
        "active": "true"
    }

    quarterly_model = get_model_metadata_by_tags(my_tags)

    #----------
    # get data
    #----------
    # oil data
    oil_df = pd.read_csv(str(path) + "/data/raw/" + "oil.csv")

    # Initialize the wrangler
    wrangler = TimeSeriesWrangler(
        date_col='date', 
        fill_col='dcoilwtico', 
        freq='D', 
        fill_method='ffill'
    )

    # Run the cleaning logic
    oil = wrangler.clean(oil_df)

    # timeseries data
    timeseries_df = pd.read_csv(str(path) + "/data/raw/" + "timeseries.csv")

    # Initialize the wrangler
    wrangler = TimeSeriesWrangler(
        date_col='date', 
        fill_col='unit_sales', 
        freq='D', 
        fill_method='zeros'
    )

    # Run the cleaning logic
    timeseries = wrangler.clean(timeseries_df)

    #------------
    # Fit models
    #------------
    date_pipeline = Pipeline([
        ('date_features', DateFeatureTransformer(column_name=date,features=['is_weekend', 'is_payday', 'is_holiday', 'month', 'year', 'day_of_week'], payday_val=15, country='EC', drop_date_col=False)),
        ('lag_features', LagFeatureTransformer({target: lags_var}, fill_method='bfill')),
        ('window_features', WindowFeatureTransformer({target: windows_var}, fill_method='bfill'))
    ])

    timeseries_features = date_pipeline.fit_transform(timeseries.reset_index())

    # Join in the oil data as an exogenous variable
    timeseries_oil = timeseries_features.merge(oil, on=date, how='left')

    # Set the date as the index first
    df = timeseries_oil.set_index(date)

    # after errors raised fixing oil data via forward fill
    if 'dcoilwtico' in timeseries_oil.columns:
        timeseries_oil['dcoilwtico'] = timeseries_oil['dcoilwtico'].ffill().fillna(0)

    # Identify where the input breaks (NaNs) and report them in a user-friendly way
    nan_report = timeseries_oil.isna().sum()
    problematic_cols = nan_report[nan_report > 0]

    if not problematic_cols.empty:
        # Build a detailed error message
        error_msg = "\n" + "-"*30 + "\nDATA INTEGRITY BREAKPOINT\n" + "-"*30
        for col, count in problematic_cols.items():
            error_msg += f"\n❌ Column '{col}': {count} missing values ({100*count/len(timeseries_oil):.2f}%)"
        
        # Logic for your oil data: Oil usually lacks weekend data.
        if 'dcoilwtico' in problematic_cols:
            error_msg += "\n\n💡 Pro-tip: Oil prices are often NaN on weekends. Consider forwardfilling."
        
        # Hard stop (The "Breakpoint")
        raise ValueError(error_msg)

    weekly_model_fitted = build_fit_model(weekly_model, timeseries_oil, target, date, max_forecast_horizon)
    monthly_model_fitted = build_fit_model(monthly_model, timeseries_oil, target, date, max_forecast_horizon)
    quarterly_model_fitted = build_fit_model(quarterly_model, timeseries_oil, target, date, max_forecast_horizon)

    #-----------------
    # build forecasts
    #-----------------

    actual_sales = timeseries.copy()
    actual_sales['date'] = pd.to_datetime(actual_sales['date'])
    actual_sales = actual_sales.set_index('date')

    forecast_weekly = forecasting_pipeline(weekly_model_fitted[0], timeseries_oil[[target,date]].copy()[:-90], date, 7, timeseries_oil, date, lags_var, windows_var, target, weekly_model_fitted[1])
    forecast_weekly = pd.DataFrame(forecast_weekly, columns=['y_hat'])
    forecast_weekly = forecast_weekly.set_index(timeseries_oil[date].copy()[-max_forecast_horizon:-max_forecast_horizon+7])

    forecast_monthly = forecasting_pipeline(monthly_model_fitted[0], timeseries_oil[[target,date]].copy()[:-90], date, 30, timeseries_oil, date, lags_var, windows_var, target, monthly_model_fitted[1])
    forecast_monthly = pd.DataFrame(forecast_monthly, columns=['y_hat'])
    forecast_monthly = forecast_monthly.set_index(timeseries_oil[date].copy()[-max_forecast_horizon:-max_forecast_horizon+30])

    forecast_quarterly = quarterly_model_fitted[0].predict(n=90,future_covariates=TimeSeries.from_dataframe(timeseries_oil[-max_forecast_horizon:], time_col=date, value_cols=quarterly_model_fitted[1]))
    forecast_quarterly = forecast_quarterly.to_dataframe()

    return (actual_sales, forecast_weekly, forecast_monthly, forecast_quarterly, timeseries_oil)

path = "data/"
date_col = 'date'
lags_var=[1,2,3,4,5,6,7]
windows_var=[7,14,21]
target = 'unit_sales'
max_forecast_horizon=90 
actual_sales, forecast_weekly, forecast_monthly, forecast_quarterly, timeseries_oil = data_prep(PROJECT_PATH, date_col, lags_var, windows_var, target, max_forecast_horizon)

#------------------------
# Building Streamlit App
#------------------------
# configs
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    /* Remove padding from the top of the main container */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.title('Sales Forecasting Corporacion Favorita')
st.markdown("Explore historical sales and future predictions across different time horizons.")

# 1. Forecast Selection (Buttons)
forecast_option = st.radio(
    "Select Forecast Horizon:",
    ["Weekly Forecast", "Monthly Forecast", "Quarterly Forecast"],
    horizontal=True
)

actual_target_col = 'unit_sales' 
# Map selection to the correct dataframe
if forecast_option == "Weekly Forecast":
    df_forecast = forecast_weekly
    forecast_target_col = 'y_hat' 
elif forecast_option == "Monthly Forecast":
    df_forecast = forecast_monthly
    forecast_target_col = 'y_hat'
else:
    df_forecast = forecast_quarterly
    forecast_target_col = 'unit_sales'

# 2. Sidebar - Date Slicer
st.sidebar.header("Filter Settings")

# Determine global min and max dates for the slicer
min_date = actual_sales.index.min().date()
# The max date should account for the future forecast horizon
max_date = pd.to_datetime("2014-01-01").date()

# Default start date (e.g., 1 year back from the last actual sales date to avoid clutter)
default_start = max_date - pd.DateOffset(days=120)
if default_start < pd.Timestamp(min_date):
    default_start = pd.Timestamp(min_date)

slider_min = datetime.datetime.combine(min_date, datetime.time.min)
slider_max = datetime.datetime.combine(max_date, datetime.time.max)
slider_default = datetime.datetime.combine(default_start.date(), datetime.time.min)

date_range = st.sidebar.slider(
    "Select Date Range for Plotting:",
    min_value=slider_min,
    max_value=slider_max,
    value=(slider_default, slider_max),
    format="DD MMM YYYY",
    help="Move the left handle to change how far back the plot goes. Keep the right handle at the end."
)

# Holiday toggle
st.sidebar.subheader("Event Overlays")
show_holidays = st.sidebar.toggle("Highlight Holidays", value=False)
show_paydays = st.sidebar.toggle("Highlight Paydays", value=False)

# 3. Filter Data Based on Date Slicer
if len(date_range) == 2:
    start_date, end_date = date_range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter actuals
    mask_actual = (actual_sales.index >= start_date) & (actual_sales.index <= end_date)
    plot_actual = actual_sales.loc[mask_actual]
else:
    # Fallback if the user hasn't selected a full range yet
    plot_actual = actual_sales.copy()

# Filter the oil/events dataframe for the same period
plot_events = timeseries_oil[(timeseries_oil[date_col] >= start_date) & (timeseries_oil[date_col] <= pd.to_datetime(df_forecast.index.max().date()))]

# 4. Plotting (Using matplotlib/seaborn as imported)
st.subheader(f"{forecast_option} vs Actual Sales")

fig, ax = plt.subplots(figsize=(14, 6))

# Highlight Holidays (Light Red Spans)
if show_holidays:
    holiday_dates = plot_events[plot_events['date_is_holiday'] == 1][date_col]
    for i, hole_date in enumerate(holiday_dates):
        ax.axvspan(hole_date, hole_date + pd.Timedelta(days=1), 
                   color='red', alpha=0.15, label='Holiday' if i == 0 else "")

# Highlight Paydays (Light Green Spans)
if show_paydays:
    payday_dates = plot_events[plot_events['date_is_payday'] == 1][date_col]
    for i, pay_date in enumerate(payday_dates):
        ax.axvspan(pay_date, pay_date + pd.Timedelta(days=1), 
                   color='green', alpha=0.15, label='Payday' if i == 0 else "")
        
# Plot Actuals
sns.lineplot(
    data=plot_actual, 
    x=date_col, 
    y=actual_target_col, 
    label='Actual Sales', 
    color='blue', 
    ax=ax
)

# Plot Forecast
sns.lineplot(
    data=df_forecast, 
    x=date_col, 
    y=forecast_target_col, 
    label=forecast_option, 
    color='orange', 
    linestyle='--', 
    ax=ax
)

ax.set_title("Historical and Forecasted Sales")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Render plot in Streamlit
st.pyplot(fig)

# 5. Wide Format Data Table
st.subheader("Data View")

# Round forecast
df_forecast[forecast_target_col] = df_forecast[forecast_target_col].astype('int')

# Standardize column names for clean merging
temp_actual = plot_actual[[actual_target_col]].copy().rename(columns={actual_target_col: 'Actual Sales'})
temp_forecast = df_forecast[[forecast_target_col]].copy().rename(columns={forecast_target_col: 'Predicted Sales'})

# Merge on date to align actuals and predictions
merged_df = pd.merge(temp_actual, temp_forecast, right_index=True, left_index=True, how='outer').sort_values(date_col)

# Convert dates back to string format for cleaner display in the table
merged_df[date_col] = merged_df.index.strftime('%Y-%m-%d')

# Transpose the dataframe to create a "Wide Format" (Dates as columns, Metrics as rows)
wide_format_df = merged_df.set_index(date_col).T

# Display using Streamlit dataframe with horizontal scrolling enabled
st.dataframe(wide_format_df, use_container_width=True)


