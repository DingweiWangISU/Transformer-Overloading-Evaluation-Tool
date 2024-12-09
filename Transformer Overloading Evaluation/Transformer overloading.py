from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd
import sklearn
import random

# Define Penetration Level (number of customers to apply EV and HP profiles to)
Pen_Level_EV = 15  # Adjust this value as needed
Pen_Level_HP = 5


# Load both files
transformer_file_path = 'transformer_customer_info.xlsx'
ami_file_path = f"output\Final Aggregated Data_EVPenLevel_{Pen_Level_EV} and HPPenLevel_{Pen_Level_HP}.xlsx"

# Read the transformer and customer file
transformer_data = pd.read_excel(transformer_file_path)
ami_data = pd.read_excel(ami_file_path)

# Split the customer indexes in the transformer data
transformer_data['Customer Indexes'] = transformer_data['Customer Indexes'].apply(lambda x: [int(i.strip()) for i in x.split(',')])

# Create an empty DataFrame to store the summed loads for each transformer
total_load_per_transformer = pd.DataFrame()

# Loop through each transformer and sum the corresponding customer loads from the AMI data
for idx, row in transformer_data.iterrows():
    transformer_name = row['Transformer']  # Use actual transformer name
    customer_indexes = row['Customer Indexes']
    
    # Sum the load for the specified customers (shift indexes by -1 for 0-based indexing)
    total_load_per_transformer[transformer_name] = ami_data[[f"Customer {i}" for i in customer_indexes]].sum(axis=1)

# Bring back the first four columns (Date, Hour, Day Type, Season) from the AMI data and add them to the total_load_per_transformer
metadata_columns = ['Date', 'Hour', 'Day Type', 'Season']
final_result = pd.concat([ami_data[metadata_columns], total_load_per_transformer], axis=1)


# Re-add the 'Month' column based on the 'Date' column
final_result['Month'] = pd.to_datetime(final_result['Date']).dt.month

# Step 1: Find the maximum load and corresponding date (month, day, and hour) for each transformer
transformer_capacity = transformer_data.set_index('Transformer')['Transformer Rating (kW)']
max_load_info = []
for transformer in transformer_capacity.index:
    max_load = final_result[transformer].max()
    max_load_row = final_result[final_result[transformer] == max_load].iloc[0]
    max_load_date = f"{max_load_row['Date']} Hour: {max_load_row['Hour']}"
    
    # Append the results as a dictionary
    max_load_info.append({
        "Transformer": transformer,
        "Max Load (kW)": max_load,
        "Date of Max Load": max_load_date
    })

# Convert to DataFrame for ease of visualization
max_load_df = pd.DataFrame(max_load_info)

# Step 2: Count annual overload occurrences for each transformer
overload_info = []
for transformer, capacity in transformer_capacity.items():
    # Count how many times the load exceeds capacity
    annual_overloads = (final_result[transformer] > capacity).sum()
    
    # Append the overload count
    overload_info.append({
        "Transformer": transformer,
        "Annual Overloads": annual_overloads
    })

# Step 3: Breakdown by month - Calculate monthly overload occurrences and monthly maximum load
monthly_overload_info = []
for transformer, capacity in transformer_capacity.items():
    for month in range(1, 13):  # Loop over months 1 to 12
        month_data = final_result[final_result['Month'] == month]
        
        # Count monthly overloads
        monthly_overloads = (month_data[transformer] > capacity).sum()
        
        # Find the max load for this month and its date
        max_monthly_load = month_data[transformer].max()
        max_monthly_load_row = month_data[month_data[transformer] == max_monthly_load].iloc[0]
        max_monthly_load_date = f"{max_monthly_load_row['Date']} Hour: {max_monthly_load_row['Hour']}"
        
        # Append the results
        monthly_overload_info.append({
            "Transformer": transformer,
            "Month": month,
            "Monthly Overloads": monthly_overloads,
            "Max Monthly Load (kW)": max_monthly_load,
            "Date of Max Monthly Load": max_monthly_load_date
        })

# Convert the overload info to DataFrames
annual_overload_df = pd.DataFrame(overload_info)
monthly_overload_df = pd.DataFrame(monthly_overload_info)

# Save all data into one Excel file with multiple sheets
merged_output_file = f'output/Transformer_Load_Analysis_Results_pen_level_{Pen_Level_EV} and {Pen_Level_HP}.xlsx'

with pd.ExcelWriter(merged_output_file) as writer:
    max_load_df.to_excel(writer, sheet_name='Max Load per Transformer', index=False)
    annual_overload_df.to_excel(writer, sheet_name='Annual Overloads', index=False)
    monthly_overload_df.to_excel(writer, sheet_name='Monthly Overloads Breakdown', index=False)
    
    
import matplotlib.pyplot as plt

# Group the monthly overloading data by transformer and month for visualization
monthly_overload_pivot = monthly_overload_df.pivot(index='Month', columns='Transformer', values='Monthly Overloads')

# Plot the data
plt.figure(figsize=(25, 20))

# Plot each transformer's monthly overloading
monthly_overload_pivot.plot(kind='line', marker='o', figsize=(25, 20))

plt.title(f'Monthly Overloading Count for Each Transformer with pen level of {Pen_Level_EV} and {Pen_Level_HP}')
plt.xlabel('Month')
plt.ylabel('Overloading Count')
plt.grid(True)
plt.legend(title="Transformer", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show the plot
plt.show()
    

    
    
    
    
    
