import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from datetime import datetime, timedelta

# Suppress specific warnings from statsmodels, pandas
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning) # Suppress any deprecation warnings

# --- Configuration ---
st.set_page_config(layout="wide", page_title="EV Sales Predictor (SARIMA Model)")

# --- Load Data and Models ---
@st.cache_data # Cache data loading for performance
def load_data():
    """Loads and preprocesses the historical EV sales data."""
    try:
        df = pd.read_csv("IEA-EV-dataEV salesHistoricalCars.csv")
        df.columns = df.columns.str.strip() # Remove extra spaces from headers
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['year', 'value']) # Drop rows with NaN in year or value
        df['year'] = df['year'].astype(int) # Ensure year is integer
        return df
    except FileNotFoundError:
        st.error("Error: 'IEA-EV-dataEV salesHistoricalCars.csv' not found. Please ensure the file is in the same directory.")
        st.stop() # Stop the app if file is not found
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop() # Stop the app on other data loading errors

@st.cache_resource # Cache model loading for performance
def load_models():
    """Loads the pre-trained SARIMA models for BEV and PHEV."""
    bev_model = None
    phev_model = None
    try:
        bev_model = joblib.load("ev_model_bev.joblib")
        st.sidebar.success("BEV model loaded.")
    except FileNotFoundError:
        st.error("Error: 'ev_model_bev.joblib' not found. Please run 'train_model.py' first.")
    except Exception as e:
        st.error(f"Error loading BEV model: {e}")

    try:
        phev_model = joblib.load("ev_model_phev.joblib")
        st.sidebar.success("PHEV model loaded.")
    except FileNotFoundError:
        st.error("Error: 'ev_model_phev.joblib' not found. Please run 'train_model.py' first.")
    except Exception as e:
        st.error(f"Error loading PHEV model: {e}")

    if bev_model is None and phev_model is None:
        st.stop() # Stop if no models could be loaded
    return bev_model, phev_model

df_raw = load_data()
bev_sarima_results, phev_sarima_results = load_models()


# --- Data Preprocessing for Display and Prediction ---
# Filter for EV sales (BEV and PHEV Cars, Historical, Vehicles) for display
df_filtered_display = df_raw[
    (df_raw['parameter'] == 'EV sales') &
    (df_raw['mode'] == 'Cars') &
    (df_raw['unit'] == 'Vehicles') &
    (df_raw['category'] == 'Historical') &
    (df_raw['powertrain'].isin(['BEV', 'PHEV']))
].copy() # Use .copy() to avoid SettingWithCopyWarning

# Handle potential missing 'region' column gracefully
if 'region' not in df_filtered_display.columns:
    st.warning("The 'region' column was not found in the dataset. Region-wise filtering and display will not be available.")
    df_filtered_display['region'] = 'Overall' # Add a dummy region for overall view
    all_regions = ['Overall']
else:
    all_regions = sorted(df_filtered_display['region'].unique().tolist())


# --- Create monthly time series for historical data, separated by powertrain ---
# This DataFrame will be used for plotting historical BEV and PHEV lines
monthly_historical_powertrain_data_list = []
for (year, powertrain), group_df in df_filtered_display.groupby(['year', 'powertrain']):
    total_sales_for_year_powertrain = group_df['value'].sum()
    monthly_sales_value = total_sales_for_year_powertrain / 12

    for month in range(1, 13):
        date_str = f"{year}-{month:02d}-01"
        monthly_historical_powertrain_data_list.append({
            "date": pd.Timestamp(date_str),
            "powertrain": powertrain,
            "sales": monthly_sales_value
        })

# Create a DataFrame from the list of dictionaries. The 'date' column will already be datetime.
historical_monthly_powertrain_df = pd.DataFrame(monthly_historical_powertrain_data_list)

# Pivot the DataFrame to have powertrains as columns
historical_monthly_powertrain_df = historical_monthly_powertrain_df.pivot_table(
    index='date', columns='powertrain', values='sales', fill_value=0
)
historical_monthly_powertrain_df.columns.name = None # Remove column name 'powertrain'
historical_monthly_powertrain_df = historical_monthly_powertrain_df.sort_index() # Sort by date index


# --- Create monthly time series for individual powertrain historical data (for prediction base) ---
# These DataFrames are used as the input for their respective SARIMA prediction models
historical_monthly_bev_df = historical_monthly_powertrain_df[['BEV']].rename(columns={'BEV': 'sales'})
historical_monthly_phev_df = historical_monthly_powertrain_df[['PHEV']].rename(columns={'PHEV': 'sales'})


# --- Streamlit UI ---
st.title("Electric Car Sales Analysis and Prediction")

# st.markdown("""
# This application provides insights into historical Electric Vehicle (EV) sales and forecasts future trends using **SARIMA** time series models.
# You can explore sales data year-wise and region-wise, and see the predicted monthly sales for future years.
# """)

# --- Sidebar Filters ---
st.sidebar.header("Filter Options for Historical Data")

# Year filter for historical data
min_year_data = int(df_filtered_display['year'].min())
max_year_data = int(df_filtered_display['year'].max())
selected_years = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year_data,
    max_value=max_year_data,
    value=(min_year_data, max_year_data) # Default to full range
)

# Region filter for historical data - DROPDOWN
if 'region' in df_raw.columns:
    selected_region = st.sidebar.selectbox(
        "Select Region",
        options=all_regions,
        index=all_regions.index('Overall') if 'Overall' in all_regions else 0 # Default to 'Overall' or first region
    )
else:
    selected_region = 'Overall' # Default if no region column
    st.sidebar.info("Region filter unavailable: 'region' column not found.")


# Filter historical data based on sidebar selections
# Adjusted filtering for single selected region
df_display = df_filtered_display[
    (df_filtered_display['year'] >= selected_years[0]) &
    (df_filtered_display['year'] <= selected_years[1]) &
    (df_filtered_display['region'] == selected_region) # Changed from .isin() to ==
]

# --- Main Content ---

# Tabbed interface for different views
tab1, tab2, tab3 = st.tabs(["Overall Sales & Prediction", "Year-wise Sales", "Region-wise Sales"])

with tab1:
    # st.header("Overall EV Sales (Historical & Predicted by Powertrain up to 2040)") # Updated header

    # --- Prediction Logic for 2040 ---
    # Find the maximum year in the dataset used for training
    max_data_year = historical_monthly_powertrain_df.index.year.max()

    # Determine the number of steps to forecast to reach the end of 2040
    num_forecast_steps = (2040 - max_data_year) * 12 # Changed target year to 2040

    forecast_bev_series = pd.Series([], dtype='float64')
    forecast_phev_series = pd.Series([], dtype='float64')
    forecast_index = pd.DatetimeIndex([])

    if num_forecast_steps > 0:
        last_historical_date_bev = historical_monthly_bev_df.index.max()
        last_historical_date_phev = historical_monthly_phev_df.index.max()

        # Use the latest of the two last historical dates for forecast start
        last_historical_date_for_forecast = max(last_historical_date_bev, last_historical_date_phev)

        forecast_index = pd.date_range(
            start=last_historical_date_for_forecast + pd.DateOffset(months=1),
            periods=num_forecast_steps,
            freq='MS' # Month Start frequency
        )

        try:
            if bev_sarima_results:
                forecast_bev = bev_sarima_results.predict(
                    start=len(historical_monthly_bev_df),
                    end=len(historical_monthly_bev_df) + num_forecast_steps - 1
                )
                forecast_bev_series = pd.Series(forecast_bev.values, index=forecast_index, name='BEV')
                # Filter to include all years from (max_data_year + 1) up to 2040
                forecast_bev_series = forecast_bev_series[forecast_bev_series.index.year >= (max_data_year + 1)]
        except Exception as e:
            st.error(f"Error generating BEV forecast: {e}")

        try:
            if phev_sarima_results:
                forecast_phev = phev_sarima_results.predict(
                    start=len(historical_monthly_phev_df),
                    end=len(historical_monthly_phev_df) + num_forecast_steps - 1
                )
                forecast_phev_series = pd.Series(forecast_phev.values, index=forecast_index, name='PHEV')
                # Filter to include all years from (max_data_year + 1) up to 2040
                forecast_phev_series = forecast_phev_series[forecast_phev_series.index.year >= (max_data_year + 1)]
        except Exception as e:
            st.error(f"Error generating PHEV forecast: {e}")

        # st.write(f"Predicting sales up to 2040 (based on data up to {max_data_year}).") # Updated text

    else:
        st.info(f"Historical data already includes or goes beyond 2040 (max year in data: {max_data_year}). No further predictions are needed.") # Updated text


    # --- Combine historical and predicted data for plotting ---
    # Prepare historical data for plotting
    plot_df_historical_melted = historical_monthly_powertrain_df.reset_index().melt(
        id_vars=['date'], var_name='Category', value_name='Sales'
    )
    plot_df_historical_melted['Category'] = plot_df_historical_melted['Category'] + ' (Historical)'

    # Prepare predicted data for plotting
    plot_df_predicted_list = []
    if not forecast_bev_series.empty:
        plot_df_predicted_list.append(pd.DataFrame({
            'date': forecast_bev_series.index,
            'Sales': forecast_bev_series.values,
            'Category': 'BEV (Predicted)'
        }))
    if not forecast_phev_series.empty:
        plot_df_predicted_list.append(pd.DataFrame({
            'date': forecast_phev_series.index,
            'Sales': forecast_phev_series.values,
            'Category': 'PHEV (Predicted)'
        }))

    plot_df_predicted = pd.concat(plot_df_predicted_list, ignore_index=True) if plot_df_predicted_list else pd.DataFrame()


    # Concatenate historical and predicted dataframes
    if not plot_df_predicted.empty:
        combined_plot_df = pd.concat([plot_df_historical_melted, plot_df_predicted], ignore_index=True)
    else:
        combined_plot_df = plot_df_historical_melted.copy()

    # Sort by date for proper line plotting
    combined_plot_df = combined_plot_df.sort_values(by='date')

    # Plot overall sales with prediction
    fig_overall = px.line(
        combined_plot_df,
        x='date',
        y='Sales',
        color='Category', # Color by the new 'Category' column
        title='Overall EV Sales (Historical and Predicted by Powertrain up to 2040)', # Updated title
        labels={'Sales': 'Number of Vehicles', 'date': 'Date'},
        color_discrete_map={
            'BEV (Historical)': 'blue',
            'PHEV (Historical)': 'green',
            'BEV (Predicted)': 'red',    # New color for predicted BEV
            'PHEV (Predicted)': 'orange' # New color for predicted PHEV
        },
        height=600 # Increased chart height
    )
    fig_overall.update_layout(hovermode="x unified")
    st.plotly_chart(fig_overall, use_container_width=True)

    # Checkbox to show raw data
    show_raw_data = st.checkbox("Show Predicted Monthly Sales Data")

    if show_raw_data:
        st.subheader("Predicted Monthly Sales up to 2040") # Updated subheader
        if not forecast_bev_series.empty or not forecast_phev_series.empty:
            # Create a DataFrame for predicted sales, including all predicted years
            predicted_future_df = pd.DataFrame(index=forecast_index[forecast_index.year >= (max_data_year + 1)])
            if not forecast_bev_series.empty:
                predicted_future_df['BEV Predicted Sales'] = forecast_bev_series
            if not forecast_phev_series.empty:
                predicted_future_df['PHEV Predicted Sales'] = forecast_phev_series

            st.dataframe(predicted_future_df.reset_index().rename(columns={'index': 'Date'}), use_container_width=True, hide_index=True)
        else:
            st.info("No predictions for future years to display.") # Updated text


with tab2:
    st.header("Year-wise EV Sales (Historical)")
    st.write("Use the 'Select Year Range' and 'Select Region' filters in the sidebar to adjust the view.")

    if not df_display.empty:
        # Group by year and powertrain for year-wise breakdown
        yearly_powertrain_sales = df_display.groupby(['year', 'powertrain'])['value'].sum().reset_index()
        fig_year_powertrain = px.bar(
            yearly_powertrain_sales,
            x='year',
            y='value',
            color='powertrain',
            title=f'EV Sales by Powertrain ({selected_years[0]}-{selected_years[1]}) for {selected_region}',
            labels={'value': 'Number of Vehicles', 'year': 'Year'},
            barmode='group', # Group bars for BEV and PHEV
            height=400 # Increased chart height
        )
        st.plotly_chart(fig_year_powertrain, use_container_width=True)

        # Group by year and region for year-wise regional breakdown
        st.subheader(f"Total EV Sales for {selected_region} ({selected_years[0]}-{selected_years[1]})")
        total_sales_for_region = df_display['value'].sum()
        st.metric(label=f"Total Sales in {selected_region}", value=f"{total_sales_for_region:,.0f} Vehicles")

    else:
        st.info("No historical data available for the selected year range and region. Adjust filters in the sidebar.")


with tab3:
    st.header("Region-wise EV Sales (Historical)")
    st.write("The data in this tab is filtered by the 'Select Year Range' in the sidebar. The 'Select Region' filter applies to other tabs.")
    st.write("This tab shows sales across *all* regions (if 'region' column is available) within the selected year range.")

    if 'region' in df_raw.columns: # Check if region column exists in raw data
        # For this tab, we want to show all regions within the selected year range
        df_all_regions_in_range = df_filtered_display[
            (df_filtered_display['year'] >= selected_years[0]) &
            (df_filtered_display['year'] <= selected_years[1])
        ]

        if not df_all_regions_in_range.empty:
            # Group by year and region for region-wise time series
            region_time_series = df_all_regions_in_range.groupby(['year', 'region'])['value'].sum().reset_index()

            fig_region_trend = px.line(
                region_time_series,
                x='year',
                y='value',
                color='region',
                title='Historical EV Sales Trend by Region (All Regions in Selected Year Range)',
                labels={'value': 'Number of Vehicles', 'year': 'Year'},
                hover_data={'value': True, 'year': True, 'region': True},
                height=400 # Increased chart height
            )
            fig_region_trend.update_layout(hovermode="x unified")
            st.plotly_chart(fig_region_trend, use_container_width=True)

            st.subheader("Total Regional Sales Breakdown (All Regions in Selected Year Range)")
            # Pivot table for a clearer view of regional sales per year
            regional_pivot = df_all_regions_in_range.groupby('region')['value'].sum().reset_index()
            fig_regional_bar = px.bar(
                regional_pivot,
                x='region',
                y='value',
                title='Total EV Sales by Region (Selected Year Range)',
                labels={'value': 'Total Vehicles', 'region': 'Region'},
                height=400 # Increased chart height
            )
            st.plotly_chart(fig_regional_bar, use_container_width=True)

        else:
            st.info("No historical data available for the selected year range across any regions. Adjust year filter in the sidebar.")
    else:
        st.info("Region-wise analysis is not available as the 'region' column was not found in the dataset.")

