import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
# import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from data_processing import add_holidays_count
from Data_selection import get_holidays_list
from data_processing import get_lin_regression_trend

import streamlit as st

import plotly.express as px


def train_sarimax_model(df_product, value_to_plot, params):
    """
    Trains a SARIMAX model and updates the Streamlit UI with status messages.
    
    Parameters:
        df_product (pd.DataFrame): The dataset containing the time series and exogenous variables.
        value_to_plot (str): The target variable to model.
        params (dict): Dictionary containing 'order' and 'seasonal_order' for SARIMAX.
    """
    train_status_placeholder = st.empty()
    
    with train_status_placeholder.status("Training model...", expanded=True) as train_status:
        train_warning_msg = st.warning("It will take some time to train the model and may lead to app crash because of limited resources on Cloud.")
        
        model_fit = SARIMAX(
            df_product[value_to_plot],
            exog=df_product['holidays_count'],
            enforce_stationarity=False,
            enforce_invertibility=False,
            order=params["order"],
            seasonal_order=params["seasonal_order"]
        ).fit()
        
        train_warning_msg.empty()
        train_status.update(label="Model trained successfully!", state="complete")
        train_status_placeholder.empty()
        
    return model_fit

def prepare_forecasting_df(df_product, forecast_horizon, freq, timestamp_col, resolution):
    """
    Prepares the forecasting DataFrame by generating future timestamps and adding holiday counts.
    
    Parameters:
        df_product (pd.DataFrame): The dataset containing the time series.
        forecast_horizon (int): Number of future periods to forecast.
        freq (str): Frequency of the time series data.
        timestamp_col (str): The name of the timestamp column.
        resolution (str): Resolution for holiday adjustments.
    
    Returns:
        pd.DataFrame: The forecasting DataFrame with holidays count.
    """
    forecast_dates = pd.date_range(start=df_product.index[-1], periods=forecast_horizon + 1, freq=freq)[1:]
    forecasting_df = pd.DataFrame({timestamp_col: forecast_dates})
    
    holidays_list = get_holidays_list(forecast_dates[0], forecast_dates[-1])
    forecasting_df = add_holidays_count(sales_df=forecasting_df, resolution=resolution, holidays_list=holidays_list)
    
    return forecasting_df

def generate_forecast(model_fit, forecasting_df, forecast_horizon, confidence_level):
    """
    Generates a forecast using a trained SARIMAX model and updates the Streamlit UI with status messages.
    
    Parameters:
        model_fit: The trained SARIMAX model.
        forecasting_df (pd.DataFrame): The forecasting DataFrame with holidays count.
        forecast_horizon (int): Number of future periods to forecast.
        confidence_level (float): Confidence level for prediction intervals.
    
    Returns:
        pd.DataFrame: The forecast DataFrame with mean forecast and confidence intervals.
    """
    fit_status_placeholder = st.empty()
    
    with fit_status_placeholder.status("Forecast loading...", expanded=True) as fit_status:
        fit_warning_msg = st.warning("It will take some time to forecast and may lead to app crash because of limited resources on Cloud.")
        
        forecast_result = model_fit.get_forecast(
            steps=forecast_horizon,
            exog=forecasting_df['holidays_count'],
            dynamic=True,
        )
        
        fit_warning_msg.empty()
        fit_status.update(label="Forecasting loaded successfully!", state="complete")
        fit_status_placeholder.empty()
        
    forecast_df = forecast_result.summary_frame(alpha=1 - confidence_level)
    forecast_df = forecast_df[["mean", "mean_ci_lower", "mean_ci_upper"]]
    forecast_df.columns = ["forecast", "lower_bound", "upper_bound"]
    
    return forecast_df

def generate_weekly_forecast(df_product, value_to_plot, params, forecast_horizon, confidence_level, timestamp_col="date"):
    """
    Generates a weekly forecast by first predicting daily and monthly values, then resampling to weekly using both forecasts.
    
    Parameters:
        df_product (pd.DataFrame): The dataset containing the time series.
        value_to_plot (str): The target variable to model.
        params (dict): Dictionary containing 'order' and 'seasonal_order' for SARIMAX.
        forecast_horizon (int): Number of future periods to forecast.
        confidence_level (float): Confidence level for prediction intervals.
        timestamp_col (str): The name of the timestamp column.
    
    Returns:
        pd.DataFrame: The final weekly forecast DataFrame.
    """
    # Train model on daily data
    daily_model = train_sarimax_model(df_product, value_to_plot, params)
    daily_forecasting_df = prepare_forecasting_df(df_product, forecast_horizon * 7, "D", timestamp_col, "daily")
    daily_forecast = generate_forecast(daily_model, daily_forecasting_df, forecast_horizon * 7, confidence_level)
    daily_forecast[timestamp_col] = daily_forecasting_df[timestamp_col].values
    
    # Train model on monthly data
    monthly_model = train_sarimax_model(df_product.resample("M").sum(), value_to_plot, params)
    monthly_forecasting_df = prepare_forecasting_df(df_product, forecast_horizon // 4, "M", timestamp_col, "monthly")
    monthly_forecast = generate_forecast(monthly_model, monthly_forecasting_df, forecast_horizon // 4, confidence_level)
    monthly_forecast[timestamp_col] = monthly_forecasting_df[timestamp_col].values
    
    # Resample daily forecast to weekly (starting from Monday)
    weekly_forecast_from_daily = daily_forecast.set_index(timestamp_col).resample("W-MON").sum().reset_index()
    
    # Resample monthly forecast to weekly (starting from Monday)
    weekly_forecast_from_monthly = monthly_forecast.set_index(timestamp_col).resample("W-MON").interpolate().reset_index()
    
    # Combine both forecasts (averaging them for better accuracy)
    final_weekly_forecast = weekly_forecast_from_daily.merge(
        weekly_forecast_from_monthly, on=timestamp_col, suffixes=("_daily", "_monthly")
    )
    final_weekly_forecast["forecast"] = (
        final_weekly_forecast["forecast_daily"] + final_weekly_forecast["forecast_monthly"]
    ) / 2
    final_weekly_forecast["lower_bound"] = (
        final_weekly_forecast["lower_bound_daily"] + final_weekly_forecast["lower_bound_monthly"]
    ) / 2
    final_weekly_forecast["upper_bound"] = (
        final_weekly_forecast["upper_bound_daily"] + final_weekly_forecast["upper_bound_monthly"]
    ) / 2
    
    return final_weekly_forecast[[timestamp_col, "forecast", "lower_bound", "upper_bound"]]


def forecast_sales(sales_df, 
                    product_or_product_group, 
                    product_to_forecast,
                    value_to_plot, 
                    resolution,
                    model="sarima", 
                    forecast_horizon=None, 
                    model_params=None, 
                    confidence_level=0.95,
                    validation_forecast=False):
    """
    Forecasts sales for different products or product groups with confidence intervals.

    Parameters:
    - sales_df: pd.DataFrame -> Input DataFrame with columns [product_or_product_group, 
                                                        timestamp_col, value_to_plot]
    - product_or_product_group: str -> Column name for product or product group
    - timestamp_col: str -> Column name for timestamp
    - value_to_plot: str -> Column name for sales figures
    - model: str -> Forecasting model to use ('sarima' or 'random_forest')
    - forecast_horizon: int -> Number of periods ahead to predict
    - model_params: dict -> Parameters for the chosen model (optional)
    - resolution: str -> Data resolution ('day', 'week', or 'month')
    - confidence_level: float -> Confidence level for prediction intervals

    Returns:
    - pd.DataFrame -> Forecasted sales with confidence intervals
    """
    timestamp_col = f'Timestamp_{resolution}'
    freq = "D" if resolution == "day" else ("W-MON" if resolution == "week" else "MS")

    sales_df[timestamp_col] = pd.to_datetime(sales_df[timestamp_col])
    sales_df = sales_df.sort_values(by=[product_or_product_group, timestamp_col])
    
    df_product = sales_df[sales_df[product_or_product_group] == product_to_forecast].copy()
    df_product = df_product.set_index(timestamp_col)

    df_product[value_to_plot].fillna(0, inplace=True)

    model_file_name = f"{model}_model_{resolution}_{value_to_plot}_{product_to_forecast}.pkl"
    models_folder = Path("./saved_models/")
    
    forecast_horizon = forecast_horizon or {"day": 30, "week": 8, "month": 4}[resolution]
    if model == "sarima":
        min_data_points = len(df_product)
        seasonal_period = {"day": 7, "week": min(52, max(4, min_data_points // 2)), "month": min(12, max(3, min_data_points // 2))}[resolution]
        
        if min_data_points < 10:
            print(f"Warning: Too few observations ({min_data_points}) for SARIMA. Switching to non-seasonal ARIMA.")
            seasonal_period = 0  # Disable seasonality
        params = model_params or {"order": (0, 1, 1), "seasonal_order": (0, 1, 1, seasonal_period)}
        if model_file_name in os.listdir(models_folder) and not validation_forecast:
            model_fit = joblib.load(models_folder/model_file_name)
        else:
            model_fit = train_sarimax_model(df_product, value_to_plot, params)
            # if not validation_forecast:
            #     joblib.dump(model_fit,models_folder/model_file_name , compress=3)
        forecasting_df = prepare_forecasting_df(df_product, forecast_horizon, freq, timestamp_col, resolution)
        
        forecast_df = generate_forecast(model_fit, forecasting_df, forecast_horizon, confidence_level)
    elif model == "random_forest":
        df_product["time_index"] = range(len(df_product))
        df_product["day_of_week"] = df_product.index.dayofweek  # Add day of week feature
        X = df_product[["time_index", "day_of_week"]]
        y = df_product[value_to_plot]
        
        X_train, y_train = X[:-forecast_horizon], y[:-forecast_horizon]
        X_future = pd.DataFrame({
            "time_index": range(X.time_index.max() + 1, X.time_index.max() + 1 + forecast_horizon),
            "day_of_week": [(df_product.index[-1] + pd.Timedelta(days=i)).dayofweek for i in range(1, forecast_horizon + 1)]
        })
        
        params = model_params if model_params else {"n_estimators": 100, "max_depth": 5}
        rf_model = RandomForestRegressor(**params)
        rf_model.fit(X_train, y_train)
        forecast = rf_model.predict(X_future)
        
        residuals = y_train - rf_model.predict(X_train)
        std_dev = np.std(residuals)
        margin_of_error = 1.96 * std_dev  # Approximate 95% CI
        lower_bound = forecast - margin_of_error
        upper_bound = forecast + margin_of_error
        
        forecast_df = pd.DataFrame({"forecast": forecast, "lower_bound": lower_bound, "upper_bound": upper_bound})
    else:
        raise ValueError("Unsupported model. Choose 'sarima' or 'random_forest'.")

    forecast_dates = pd.date_range(start=df_product.index[-1], periods=forecast_horizon + 1, freq=freq)[1:]

    forecast_df[timestamp_col] = forecast_dates
    forecast_df[product_or_product_group] = product_to_forecast
    desired_order = [timestamp_col,product_or_product_group ] + [col for col in forecast_df.columns if col not in [timestamp_col, product_or_product_group]]
    return forecast_df[desired_order].reset_index(drop =True)



def plot_forecast(sales_df, 
                  forecast_dfs,
                  products_to_plot, 
                  product_or_product_group, 
                  resolution,
                  value_to_plot,
                  start_date,
                  end_date,
                  validation_plot=False):
    """
    Plots past sales data along with forecasted sales and confidence intervals using Plotly.
    
    Parameters:
    - sales_df: pd.DataFrame -> Historical sales data
    - forecast_dfs: dict of pd.DataFrame -> List of forecasted sales DataFrames with confidence intervals
    - products_to_plot: list -> List of product names to plot
    - product_or_product_group: str -> Column name for product or product group
    - resolution: str -> Data resolution ('day', 'week', or 'month')
    - value_to_plot: str -> Column name for sales figures
    
    Returns:
    - fig: Plotly figure object
    """
    fig = go.Figure()
    timestamp_col = f'Timestamp_{resolution}'
    colors = px.colors.qualitative.Set1  # A set of distinct colors for different products
    color_map = {product: colors[i % len(colors)] for i, product in enumerate(products_to_plot 
                                                                              + [ f'{product}_validation' for product in products_to_plot])}
 
    
    for product in products_to_plot:
        df_product = sales_df[sales_df[product_or_product_group] == product].copy()
        cutoff_date = start_date
        forecast_df = forecast_dfs.get((resolution, 
                                        product_or_product_group, 
                                        product, 
                                        value_to_plot), None)
        if forecast_df is None:
            raise ValueError(f"No forecast found for product: {product_or_product_group}, element: {product}, value: {value_to_plot}")
        
        color = color_map[product]
        if not validation_plot:
            # Interpolate actual sales to connect smoothly with forecast
            last_actual_timestamp = df_product[timestamp_col].max()
            first_forecast_timestamp = forecast_df[timestamp_col].min()
            last_actual_value = df_product[df_product[timestamp_col] == last_actual_timestamp][value_to_plot].values[0]
            first_forecast_value = forecast_df[forecast_df[timestamp_col] == first_forecast_timestamp]["forecast"].values[0]
            
            interpolated_x = [last_actual_timestamp, first_forecast_timestamp]
            interpolated_y = [last_actual_value, first_forecast_value]

            regression_window = cutoff_date , sales_df[timestamp_col].max()
   

            df_reg,_ = get_lin_regression_trend(sales_df,
                            resolution,
                            product,
                            product_or_product_group,
                            value_to_plot,
                            regression_window)
        
            # Add Regression Line
            fig.add_trace(go.Scatter(
                x=df_reg[timestamp_col], 
                y=df_reg[f"{value_to_plot}_trend"], 
                mode='lines', 
                showlegend=False,
                line=dict(color=color, dash='dot')
            ))
        
        fig.add_trace(go.Scatter(x=df_product[timestamp_col], y=df_product[value_to_plot], 
                                 mode='lines', name=f'Sales - {product}', 
                                 line=dict(color=color)))
        if not validation_plot:
            fig.add_trace(go.Scatter(x=interpolated_x, y=interpolated_y, 
                                 mode='lines', line=dict(color=color, dash='dot'), showlegend=False))
        
            # Add confidence interval shading (without legend entry)
            fig.add_trace(go.Scatter(x=forecast_df[timestamp_col].tolist() + forecast_df[timestamp_col].tolist()[::-1],
                                    y=forecast_df["upper_bound"].tolist() + forecast_df["lower_bound"].tolist()[::-1],
                                    fill='toself', fillcolor=color.replace("rgb", "rgba").replace(")", ", 0.2)"),
                                    line=dict(color='rgba(255,255,255,0)'),
                                    showlegend=False))
            
            # Ensure forecast curve is hoverable by adding it after confidence interval (without legend entry)
        if validation_plot:
            show_legend = True
            #color = color_map[f'{product}_validation']
            #dash = None
        else:
            show_legend = False
            #dash='dash'

        fig.add_trace(go.Scatter(x=forecast_df[timestamp_col], y=forecast_df["forecast"], 
                                 mode='lines', line=dict(color=color, dash='dash'), 
                                 hoverinfo='x+y',name=f'Retro Forecast - {product}', showlegend=show_legend))


    
    # Layout settings
    from_date = forecast_df[timestamp_col].min().strftime('%Y-%m-%d')
    to_date = forecast_df[timestamp_col].max().strftime('%Y-%m-%d')
    fig.update_layout(title=f"Forecasted {value_to_plot} for {product_or_product_group} | {from_date} - {to_date}", 
                      hovermode = "x unified",
                      yaxis_title=value_to_plot, 
                      legend=dict(x=1, y=0.5), 
                      xaxis=dict(range=[cutoff_date, forecast_df[timestamp_col].max()]))
    
    return fig

def evaluate_model(y_true, y_pred):
        return {
            "R2 Score": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
        }

def plot_validate_forecast(sales_df,
                            validate_forecasting_df, 
                            categories_to_plot,
                            product_or_product_group,
                            resolution,
                              value_to_plot):
    
    timestamp_col = f'Timestamp_{resolution}'

    
    
    # Create a dataframe for Plotly
    df_plot = y_true.copy().set_index(timestamp_col)
    df_plot.rename(columns = {value_to_plot : "Actual"}, inplace = True)

    df_plot["Forecast"] = y_pred["forecast"].copy()
    # lower_bound, upper_bound = conf_int
    df_plot["Lower Bound"] = y_pred["lower_bound"].copy()
    df_plot["Upper Bound"] = y_pred["upper_bound"].copy()
    # Calculate evaluation metrics
    evaluate_model_dict = evaluate_model(y_true=df_plot["Actual"], y_pred=df_plot["Forecast"])

    # Create the figure
    fig = go.Figure()

    # Add actual values
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["Actual"], mode="lines",
        name="Actual", line=dict(color="blue")
    ))

    # Add forecast values
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["Forecast"], mode="lines",
        name="Forecast", line=dict(color="red")
    ))

    # Add confidence interval (shaded region)
    fig.add_trace(go.Scatter(
        x=df_plot.index.tolist() + df_plot.index[::-1].tolist(),
        y=df_plot["Upper Bound"].tolist() + df_plot["Lower Bound"][::-1].tolist(),
        fill="toself",
        fillcolor="rgba(255, 0, 0, 0.2)",  # Light pink
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval"
    ))
        
    # Set title and subtitle with evaluation metrics
    fig.update_layout(
        title=f"Validation Forecast (per {freq_name})",
        # xaxis_title="Time",
        yaxis_title=value_to_plot,
        legend_title="Legend",
        # template="plotly_white",
        annotations=[
            dict(
                text=f"R2 Score: {evaluate_model_dict['R2 Score']:.2f} | "
                     f"MAE: {evaluate_model_dict['MAE']:.2f} | "
                     f"RMSE: {evaluate_model_dict['RMSE']:.2f}",
                x=0.5, y=-0.2, xref="paper", yref="paper",
                showarrow=False, font=dict(size=12, color="gray")
            )
        ]
    )

    return fig  # Return the Plotly figure object


