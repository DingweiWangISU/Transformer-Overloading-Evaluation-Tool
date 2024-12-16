import pandas as pd

#%% Transformer data Processing


def parse_transformer_size(size):
    """
    Parse transformer size from coded format to actual kVA value.
    - Single component before PD/PL: Use that number.
    - Multiple components before PD/PL: Sum all components.
    """
    if not isinstance(size, str):
        return 0  # Return 0 if the size is not a string
    
    parts = size.split('_')
    numeric_components = []  # Collect numeric components

    for part in parts:
        if part.replace('.', '', 1).isdigit():  # Check if it's a numeric component (integer or float)
            numeric_components.append(float(part))
        elif 'PD' in part or 'PL' in part:
            # Stop processing at the first PD/PL (phase information or other metadata)
            break

    # Sum up all numeric components
    total_kva = sum(numeric_components)

    return total_kva



def transform_raw_tc_with_index_and_check(raw_tc_data, raw_ami_data, power_factor=0.9):
    """Transform RAW_TC data to match the desired format with proper customer indexing."""
    # Map meter IDs to customer indices
    customer_indices = {meter_id: idx + 1 for idx, meter_id in enumerate(raw_ami_data.columns[1:])}

    # Update RAW_TC to use customer indices
    raw_tc_data['Customer Index'] = raw_tc_data['MTR_ID'].map(customer_indices)

    # Check and transform transformer size
    raw_tc_data['Transformer Rating (kVA)'] = raw_tc_data['XFRM_SIZE'].apply(parse_transformer_size)

    # Calculate transformer rating in kW
    raw_tc_data['Transformer Rating (kW)'] = raw_tc_data['Transformer Rating (kVA)'] * power_factor

    # Group by transformer ID
    grouped = raw_tc_data.groupby('XFMR_ID').agg({
        'Transformer Rating (kVA)': 'first',
        'Transformer Rating (kW)': 'first',
        'Customer Index': lambda x: ', '.join(map(str, sorted(x.dropna())))
    }).reset_index()

    # Rename columns to match the desired format
    grouped.rename(columns={
        'XFMR_ID': 'Transformer',
        'Transformer Rating (kVA)': 'Transformer Rating (kVA)',
        'Transformer Rating (kW)': 'Transformer Rating (kW)',
        'Customer Index': 'Customer Indexes'
    }, inplace=True)

    # Reorder columns to match the specified format
    grouped = grouped[['Transformer', 'Transformer Rating (kVA)', 'Customer Indexes', 'Transformer Rating (kW)']]
    
    return grouped


if __name__ == "__main__":
    # Example usage
    raw_tc_path = "RAW_TC.xlsx"  # Update to the actual path
    raw_ami_path = "RAW_AMI.xlsx"  # Update to the actual path
    
    # Load the data
    raw_tc_data = pd.read_excel(raw_tc_path)
    raw_ami_data = pd.read_excel(raw_ami_path)
    
    # Transform the data
    power_factor = 0.9  # Define power factor
    
    transformed_data = transform_raw_tc_with_index_and_check(raw_tc_data, raw_ami_data, power_factor)
    
    # Save the transformed data
    output_path = "Transformer_Customer_Info.xlsx"
    transformed_data.to_excel(output_path, index=False)
    print(f"Transformation complete. File saved as {output_path}.")



#%% AMI data Processing


import pandas as pd
from datetime import datetime

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


if __name__ == "__main__":
    # Read the raw AMI data
    raw_data = raw_ami_data

    # Transform the data
    transformed_data = transform_raw_to_ami(raw_data)

    # Save the transformed data to a new Excel file
    transformed_data.to_excel("AMI_Data.xlsx", index=False)
    print("Transformation complete. File saved as 'AMI_Data.xlsx'.")
