import pandas as pd
from datetime import datetime
import sys


#%% Parameters

power_factor = 0.9

#%% Data Validation Functions

def validate_raw_tc(raw_tc_data):
    required_headers = {'MTR_ID', 'XFMR_ID', 'XFRM_SIZE'}
    
    # Check headers
    if set(raw_tc_data.columns) != required_headers:
        sys.exit(f"Error: RAW_TC.xlsx must contain headers: {required_headers}. Found: {set(raw_tc_data.columns)}. Fix and re-upload.")
    
    # Check for missing values in required columns
    missing_values = raw_tc_data.isnull().sum()
    missing_cols = missing_values[missing_values > 0].index.tolist()
    if missing_cols:
        missing_rows = raw_tc_data[missing_cols].isnull().any(axis=1)
        sys.exit(f"Error: Missing values found in columns {missing_cols} at rows {list(missing_rows[missing_rows].index)}. Fix and re-upload.")

    # Convert MTR_ID and XFMR_ID to string for consistency
    raw_tc_data['MTR_ID'] = raw_tc_data['MTR_ID'].astype(str)
    raw_tc_data['XFMR_ID'] = raw_tc_data['XFMR_ID'].astype(str)

    # Check transformer size format
    def valid_transformer_size(value):
        if isinstance(value, (int, float)) and value > 0:
            return True
        if isinstance(value, str) and any(c.isdigit() for c in value) and ('PL' in value or 'PD' in value):
            return True
        return False
    
    invalid_xfrm_rows = raw_tc_data[~raw_tc_data['XFRM_SIZE'].apply(valid_transformer_size)].index.tolist()
    if invalid_xfrm_rows:
        sys.exit(f"Error: Invalid XFRM_SIZE values found at rows {invalid_xfrm_rows}. Must be a positive number or in format xx_1PL, xx_xx_2PL, etc. Fix and re-upload.")
    
    return set(raw_tc_data['MTR_ID'])

def validate_raw_ami(raw_ami_data, valid_mtr_ids):
    
    # Check headers
    if raw_ami_data.columns[0] != "Time":
        sys.exit(f"Error: The first column in RAW_AMI.xlsx must be 'Time'. Found: '{raw_ami_data.columns[0]}'. Fix and re-upload.")
    
    # Check if all meter IDs appear in RAW_TC.xlsx
    missing_mtr_ids = set(raw_ami_data.columns[1:]) - valid_mtr_ids
    if missing_mtr_ids:
        sys.exit(f"Error: The following meter IDs in RAW_AMI.xlsx are missing from RAW_TC.xlsx: {missing_mtr_ids}. Fix and re-upload.")
    
    # Check for missing values
    missing_values = raw_ami_data.isnull().sum()
    missing_cols = missing_values[missing_values > 0].index.tolist()
    if missing_cols:
        missing_rows = raw_ami_data[missing_cols].isnull().any(axis=1)
        sys.exit(f"Error: Missing values found in columns {missing_cols} at rows {list(missing_rows[missing_rows].index)}. Fix and re-upload.")

    # Check time format and hourly resolution
    raw_ami_data['Time'] = pd.to_datetime(raw_ami_data['Time'], errors='coerce')
    if raw_ami_data['Time'].isna().any():
        invalid_rows = raw_ami_data[raw_ami_data['Time'].isna()].index.tolist()
        sys.exit(f"Error: Invalid time format in RAW_AMI.xlsx at rows {invalid_rows}. Ensure the 'Time' column follows a consistent datetime format.")
    
    # Check 8760 rows constraint
    if len(raw_ami_data) != 8760:
        sys.exit(f"Error: RAW_AMI.xlsx must contain 8760 rows of data (excluding header). Found: {len(raw_ami_data)} rows. Fix and re-upload.")
    
    return raw_ami_data

#%% Transformer data Processing

def parse_transformer_size(size):
    if not isinstance(size, str):
        return 0  # Return 0 if the size is not a string
    
    parts = size.split('_')
    numeric_components = []  

    for part in parts:
        if part.replace('.', '', 1).isdigit():  
            numeric_components.append(float(part))
        elif 'PD' in part or 'PL' in part:
            break

    return sum(numeric_components)

def transform_raw_tc_with_index_and_check(raw_tc_data, raw_ami_data, power_factor=0.9):
    customer_indices = {str(meter_id): idx + 1 for idx, meter_id in enumerate(raw_ami_data.columns[1:])}
    raw_tc_data['Customer Index'] = raw_tc_data['MTR_ID'].map(customer_indices)
    raw_tc_data['Transformer Rating (kVA)'] = raw_tc_data['XFRM_SIZE'].apply(parse_transformer_size)
    raw_tc_data['Transformer Rating (kW)'] = raw_tc_data['Transformer Rating (kVA)'] * power_factor
    grouped = raw_tc_data.groupby('XFMR_ID').agg({
        'Transformer Rating (kVA)': 'first',
        'Transformer Rating (kW)': 'first',
        'Customer Index': lambda x: ', '.join(map(str, sorted(x.dropna())))
    }).reset_index()

    grouped.rename(columns={
        'XFMR_ID': 'Transformer',
        'Transformer Rating (kVA)': 'Transformer Rating (kVA)',
        'Transformer Rating (kW)': 'Transformer Rating (kW)',
        'Customer Index': 'Customer Indexes'
    }, inplace=True)

    grouped = grouped[['Transformer', 'Transformer Rating (kVA)', 'Customer Indexes', 'Transformer Rating (kW)']]
    
    return grouped

def transform_raw_to_ami(raw_data):
    # Extract date and hour from the timestamp
    raw_data = raw_data.rename(columns={raw_data.columns[0]: 'Timestamp'})
    raw_data['Date'] = pd.to_datetime(raw_data['Timestamp']).dt.date
    raw_data['Hour'] = pd.to_datetime(raw_data['Timestamp']).dt.hour

    # Determine day type (weekday/weekend)
    raw_data['Day Type'] = raw_data['Date'].apply(lambda x: 'Weekend' if datetime.strptime(str(x), "%Y-%m-%d").weekday() >= 5 else 'Weekday')

    # Determine season
    def get_season(date):
        month = date.month
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Winter'
    raw_data['Season'] = raw_data['Date'].apply(lambda x: get_season(datetime.strptime(str(x), "%Y-%m-%d")))

    # Map customer meter IDs to Customer 1, Customer 2, ..., Customer n
    customer_columns = raw_data.columns[1:-4]  # Exclude Timestamp, Date, Hour, Day Type, Season
    customer_mapping = {col: f"Customer {i+1}" for i, col in enumerate(customer_columns)}
    raw_data = raw_data.rename(columns=customer_mapping)

    # Select and reorder columns for output
    desired_columns = ['Date', 'Hour', 'Day Type', 'Season'] + list(customer_mapping.values())
    transformed_data = raw_data[desired_columns]
    
    return transformed_data

#%% Main Execution

if __name__ == "__main__":
    raw_tc_path = "RAW_TC.xlsx"
    raw_ami_path = "RAW_AMI.xlsx"

    raw_tc_data = pd.read_excel(raw_tc_path)
    raw_data = pd.read_excel(raw_ami_path)

    valid_mtr_ids = validate_raw_tc(raw_tc_data)
    raw_data = validate_raw_ami(raw_data, valid_mtr_ids)

    transformed_tc_data = transform_raw_tc_with_index_and_check(raw_tc_data, raw_data, power_factor)
    transformed_ami_data = transform_raw_to_ami(raw_data)

    transformed_tc_data.to_excel("Transformer_Customer_Info.xlsx", index=False)
    transformed_ami_data.to_excel("AMI_Data.xlsx", index=False)

    print("Transformation complete. Files saved as 'Transformer_Customer_Info.xlsx' and 'AMI_Data.xlsx'.")

