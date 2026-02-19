import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import warnings

# Suppress specific warnings from statsmodels
warnings.filterwarnings("ignore", module="statsmodels")

print("Starting model training script...")

# --- 1. Load Data ---
try:
    df = pd.read_csv("IEA-EV-dataEV salesHistoricalCars.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'IEA-EV-dataEV salesHistoricalCars.csv' not found. Please ensure the file is in the same directory.")
    exit()

# --- 2. Data Cleaning and Initial Filtering ---
df.columns = df.columns.str.strip()  # Remove extra spaces from headers

# Convert 'year' and 'value' to numeric, coercing errors to NaN
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['value'] = pd.to_numeric(df['value'], errors='coerce')

# Drop rows where 'year' or 'value' are NaN after conversion
df = df.dropna(subset=['year', 'value'])

# Convert 'year' to integer type
df['year'] = df['year'].astype(int)

print(f"Original data shape: {df.shape}")

# --- 3. Filter Data for EV Sales (BEV and PHEV Cars) ---
filtered_df = df[
    (df['parameter'] == 'EV sales') &
    (df['mode'] == 'Cars') &
    (df['unit'] == 'Vehicles') &
    (df['category'] == 'Historical') &
    (df['powertrain'].isin(['BEV', 'PHEV']))
].copy() # Use .copy() to avoid SettingWithCopyWarning

print(f"Filtered data shape for model training: {filtered_df.shape}")

# Check if filtered_df is empty
if filtered_df.empty:
    print("Error: Filtered DataFrame is empty. No data to train the model.")
    exit()

# --- 4. Prepare Data for BEV and PHEV Models ---
powertrains = ['BEV', 'PHEV']
models_to_train = {}

for pt in powertrains:
    print(f"\n--- Preparing data for {pt} model ---")
    pt_df = filtered_df[filtered_df['powertrain'] == pt].copy()

    if pt_df.empty:
        print(f"Warning: No data found for {pt}. Skipping model training for {pt}.")
        continue

    # Aggregate Yearly Sales for Time Series Model for specific powertrain
    yearly_sales_pt = pt_df.groupby('year')['value'].sum().reset_index()
    yearly_sales_pt['year'] = yearly_sales_pt['year'].astype(int)

    # Convert Yearly Data to Monthly Data for specific powertrain
    monthly_data_list_pt = []
    for _, row in yearly_sales_pt.iterrows():
        year = int(row['year'])
        total_sales_for_year = row['value']
        monthly_sales_value = total_sales_for_year / 12

        for month in range(1, 13):
            date_str = f"{year}-{month:02d}-01"
            monthly_data_list_pt.append({
                "date": pd.Timestamp(date_str),
                "sales": monthly_sales_value
            })

    monthly_df_pt = pd.DataFrame(monthly_data_list_pt).set_index("date")
    monthly_df_pt = monthly_df_pt.sort_index()

    print(f"Monthly data head for {pt} SARIMA training:\n{monthly_df_pt.head()}")
    print(f"Monthly data tail for {pt} SARIMA training:\n{monthly_df_pt.tail()}")

    # --- 5. Train SARIMA Model for specific powertrain ---
    print(f"Training SARIMA model for {pt}...")
    try:
        model_pt = SARIMAX(monthly_df_pt["sales"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results_pt = model_pt.fit(disp=False, maxiter=200)
        models_to_train[pt] = results_pt
        print(f"SARIMA model training complete for {pt}.")
    except Exception as e:
        print(f"Error during SARIMA model training for {pt}: {e}")
        print("This might happen if the time series is too short or has insufficient variation for the chosen SARIMA parameters.")

# --- 6. Save Models ---
for pt, model_results in models_to_train.items():
    model_filename = f"ev_model_{pt.lower()}.joblib"
    joblib.dump(model_results, model_filename)
    print(f"Model for {pt} trained and saved to {model_filename}")

print("\nModel training script finished.")
