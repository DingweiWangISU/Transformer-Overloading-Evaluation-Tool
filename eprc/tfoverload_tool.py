#!/usr/bin/env python3
#
# TFOverload_Tool Class contains all the logic of the
# tool from the original main.py script.
#
# This should be refactored to reduce code duplication
# and improve readability, but that is currently a 
# lower priority than getting this working for 
# CLI and WEB usage. --20250221, MVA.
#

import sys, random, uuid, re, warnings
from datetime import datetime, timedelta
from IPython import get_ipython
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class TFOverload_Tool:
    def __init__(self, amifile, tcfile, evpen, hppen, outfolder):
        self.uuid         = uuid.uuid4()
        self.amifile      = amifile
        self.tcfile       = tcfile
        self.evpen        = evpen
        self.hppen        = hppen
        self.outfolder    = outfolder
        self.power_factor = 0.9

    def whoami(self):
        return str(self.uuid)

    def run(self):
        #get_ipython().magic('reset -sf')
        warnings.filterwarnings("ignore")
        
        #%% Parameters
        power_factor = self.power_factor
        
        # Define Penetration Levels as percentages
        if self.evpen < 0 or self.evpen > 100:
            raise Exception(f"EV Penetration percentage (--evpen) must be between 0 and 100. Passed value {self.evpen} was not.")
        Pen_Level_EV_percentage = self.evpen  # Percentage of customers for EV

        if self.hppen < 0 or self.hppen > 100:
            raise Exception(f"HP Penetration percentage (--hppen) must be between 0 and 100. Passed value {self.hppen} was not.")
        Pen_Level_HP_percentage = self.hppen  # Percentage of customers for HP

        
        #%% Data Validation Functions
        def validate_raw_tc(raw_tc_data):
            required_headers = {'MTR_ID', 'XFMR_ID', 'XFRM_SIZE'}
            
            # Check headers
            if set(raw_tc_data.columns) != required_headers:
                raise Exception(f"RAW_TC.xlsx must contain headers: {required_headers}. Found: {set(raw_tc_data.columns)}. Fix and re-upload.")
            
            # Check for missing values in required columns
            missing_values = raw_tc_data.isnull().sum()
            missing_cols = missing_values[missing_values > 0].index.tolist()
            if missing_cols:
                missing_rows = raw_tc_data[missing_cols].isnull().any(axis=1)
                raise Exception(f"Missing values found in columns {missing_cols} at rows {list(missing_rows[missing_rows].index)}. Fix and re-upload.")
        
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
                raise Exception(f"Invalid XFRM_SIZE values found at rows {invalid_xfrm_rows}. Must be a positive number or in format xx_1PL, xx_xx_2PL, etc. Fix and re-upload.")
            
            return set(raw_tc_data['MTR_ID'])
        
        def validate_raw_ami(raw_ami_data, valid_mtr_ids):
            
            # Check headers
            if raw_ami_data.columns[0] != "Time":
                raise Exception(f"The first column in RAW_AMI.xlsx must be 'Time'. Found: '{raw_ami_data.columns[0]}'. Fix and re-upload.")
            
            # Check if all meter IDs appear in RAW_TC.xlsx
            missing_mtr_ids = set(raw_ami_data.columns[1:]) - valid_mtr_ids
            if missing_mtr_ids:
                raise Exception(f"The following meter IDs in RAW_AMI.xlsx are missing from RAW_TC.xlsx: {missing_mtr_ids}. Fix and re-upload.")
            
            # Check for missing values
            missing_values = raw_ami_data.isnull().sum()
            missing_cols = missing_values[missing_values > 0].index.tolist()
            if missing_cols:
                missing_rows = raw_ami_data[missing_cols].isnull().any(axis=1)
                raise Exception(f"Missing values found in columns {missing_cols} at rows {list(missing_rows[missing_rows].index)}. Fix and re-upload.")
        
            # Check time format and hourly resolution
            raw_ami_data['Time'] = pd.to_datetime(raw_ami_data['Time'], errors='coerce')
            if raw_ami_data['Time'].isna().any():
                invalid_rows = raw_ami_data[raw_ami_data['Time'].isna()].index.tolist()
                raise Exception(f"Invalid time format in RAW_AMI.xlsx at rows {invalid_rows}. Ensure the 'Time' column follows a consistent datetime format.")
            
            # Check 8760 rows constraint
            if len(raw_ami_data) != 8760:
                raise Exception(f"RAW_AMI.xlsx must contain 8760 rows of data (excluding header). Found: {len(raw_ami_data)} rows. Fix and re-upload.")
            
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
        
        def transform_raw_tc_with_index_and_check(raw_tc_data, raw_ami_data, power_factor=power_factor):
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
        
        #if __name__ == "__main__":
        #raw_tc_path = "RAW_TC.xlsx"
        #raw_ami_path = "RAW_AMI.xlsx"
        raw_tc_path  = self.tcfile
        raw_ami_path = self.amifile
        
        raw_tc_data = pd.read_excel(raw_tc_path)
        raw_data = pd.read_excel(raw_ami_path)
        # Identify the year from the first timestamp
        original_first_timestamp = pd.to_datetime(raw_data.iloc[0, 0])  # First value in the time column
        year = original_first_timestamp.year  # Extract the year
        
        # Generate a universal time range for the identified year (8760 hourly timestamps)
        new_time_range = pd.date_range(start=f"{year}-01-01 00:00:00", end=f"{year}-12-31 23:00:00", freq='H')
        
        # Replace the first column with the universal timestamp
        raw_data.iloc[:, 0] = new_time_range
        detected_year = year
        valid_mtr_ids = validate_raw_tc(raw_tc_data)
        raw_data = validate_raw_ami(raw_data, valid_mtr_ids)
        # Extract meter IDs (assuming the first column is "Time", so meter IDs start from column index 1)
        meter_id_mapping = {f"Customer {i+1}": meter_id for i, meter_id in enumerate(raw_data.columns[1:])}
        
        transformed_tc_data = transform_raw_tc_with_index_and_check(raw_tc_data, raw_data, power_factor)
        transformed_ami_data = transform_raw_to_ami(raw_data)
        
        # transformed_tc_data.to_excel("Transformer_Customer_Info.xlsx", index=False)
        # transformed_ami_data.to_excel("AMI_Data.xlsx", index=False)
        
        # print("Transformation complete. Files saved as 'Transformer_Customer_Info.xlsx' and 'AMI_Data.xlsx'.")
            
        print("Data Transformation complete.")
        
        
        
        #%% Add EV and Heat Pump Profiles
        
        # Read the AMI data
        # AMI_Hourly = pd.read_excel("AMI_Data.xlsx")
        AMI_Hourly = transformed_ami_data
        
        HP_Fall= pd.read_excel("data/hp/fall_centroid.xlsx")
        HP_Winter= pd.read_excel("data/hp/winter_centroid.xlsx")
        HP_Summer= pd.read_excel("data/hp/summer_centroid.xlsx")
        HP_Spring= pd.read_excel("data/hp/spring_centroid.xlsx")
        
        
        EV_Winter_Weekend1= pd.read_excel("data/ev/weekends/Weekends_Winter_cluster_1.xlsx")
        EV_Winter_Weekend2= pd.read_excel("data/ev/weekends/Weekends_Winter_cluster_2.xlsx")
        
        EV_Summer_Weekend1=pd.read_excel("data/ev/weekends/Weekends_Summer_cluster_1.xlsx")
        EV_Summer_Weekend2=pd.read_excel("data/ev/weekends/Weekends_Summer_cluster_2.xlsx")
        EV_Summer_Weekend3=pd.read_excel("data/ev/weekends/Weekends_Summer_cluster_3.xlsx")
        
        EV_Spring_Weekend1=pd.read_excel("data/ev/weekends/Weekends_Spring_cluster_1.xlsx")
        EV_Spring_Weekend2=pd.read_excel("data/ev/weekends/Weekends_Spring_cluster_2.xlsx")
        
        EV_Fall_Weekend1=pd.read_excel("data/ev/weekends/Weekends_Fall_cluster_1.xlsx")
        EV_Fall_Weekend2=pd.read_excel("data/ev/weekends/Weekends_Fall_cluster_2.xlsx")
        
        
        EV_Winter_Weekdays1=pd.read_excel("data/ev/weekdays/Weekdays_Winter_cluster_1.xlsx")
        EV_Winter_Weekdays2=pd.read_excel("data/ev/weekdays/Weekdays_Winter_cluster_2.xlsx")
        EV_Winter_Weekdays3=pd.read_excel("data/ev/weekdays/Weekdays_Winter_cluster_3.xlsx")
        EV_Winter_Weekdays4=pd.read_excel("data/ev/weekdays/Weekdays_Winter_cluster_4.xlsx")
        
        EV_Summer_Weekdays1=pd.read_excel("data/ev/weekdays/Weekdays_Summer_cluster_1.xlsx")
        EV_Summer_Weekdays2=pd.read_excel("data/ev/weekdays/Weekdays_Summer_cluster_2.xlsx")
        EV_Summer_Weekdays3=pd.read_excel("data/ev/weekdays/Weekdays_Summer_cluster_3.xlsx")
        
        EV_Spring_Weekdays1=pd.read_excel("data/ev/weekdays/Weekdays_Spring_cluster_1.xlsx")
        EV_Spring_Weekdays2=pd.read_excel("data/ev/weekdays/Weekdays_Spring_cluster_2.xlsx")
        EV_Spring_Weekdays3=pd.read_excel("data/ev/weekdays/Weekdays_Spring_cluster_3.xlsx")
        
        EV_Fall_Weekdays1=pd.read_excel("data/ev/weekdays/Weekdays_Fall_cluster_1.xlsx")
        EV_Fall_Weekdays2=pd.read_excel("data/ev/weekdays/Weekdays_Fall_cluster_2.xlsx")
        
        
        
        # Identify columns corresponding to customers 
        customer_columns = [col for col in AMI_Hourly.columns if col.lower().startswith('customer')]
        
        # Calculate the total number of customers
        total_customers = len(customer_columns)
        
        # Display the total customers
        print(f"The total customer number is {total_customers}")
        
        
        
        # Calculate the number of customers for EV and HP based on percentages
        Pen_Level_EV = max(int(total_customers * Pen_Level_EV_percentage / 100), 1)
        Pen_Level_HP = max(int(total_customers * Pen_Level_HP_percentage / 100), 1)
        
        # Display the calculated numbers
        
        print(f"Number of customers selected for EV profiles: {Pen_Level_EV}")
        print(f"Number of customers selected for HP profiles: {Pen_Level_HP}")
        
        
        # DataFrame to store the final results (copy of the original to preserve unchanged data)
        AMI_Total = AMI_Hourly.copy()
        
        # Randomly select customers based on the penetration level
        selected_customers_EV = random.sample(list(AMI_Hourly.columns[4:]), Pen_Level_EV)
        selected_customers_HP = random.sample(list(AMI_Hourly.columns[4:]), Pen_Level_HP)
        
        
        # Function to extract numeric part for sorting
        def extract_customer_number(customer_name):
            return int(re.search(r'\d+', customer_name).group())  # Extract numeric digits and convert to int
        
        # Sort the selected customers numerically based on the numeric part
        selected_customers_EV_sorted = sorted(selected_customers_EV, key=extract_customer_number)
        selected_customers_HP_sorted = sorted(selected_customers_HP, key=extract_customer_number)
        
        # Convert Customer Indexes to Original Meter IDs
        selected_customers_EV_meters = [meter_id_mapping[customer] for customer in selected_customers_EV_sorted]
        selected_customers_HP_meters = [meter_id_mapping[customer] for customer in selected_customers_HP_sorted]
        
        # Display the selected customers with original meter IDs
        print(f"The following customers are selected as EV users: {selected_customers_EV_meters}")
        print(f"The following customers are selected as HP users: {selected_customers_HP_meters}")
        
        
        
        #%% AMI PLUS EV
        
        
        for customer in selected_customers_EV:
            
            # Extract the AMI profile for the selected customer
            customer_ami_profile = AMI_Total[customer] #Generate AMI Profile 
            
            excluded_columns = AMI_Hourly.iloc[:, :4]
            
            Customer_Profile_AMI = pd.concat([excluded_columns, customer_ami_profile], axis=1)
            
            #Generate HP Profile for seasons to add to AMI Profile
            
            # Define a dictionary to map seasons to their corresponding DataFrames
            season_datasets = {
                'Fall': HP_Fall,
                'Winter': HP_Winter,
                'Spring': HP_Spring,
                'Summer': HP_Summer
            }
            
            # Define probabilities for each season
            probabilities = {
                'Fall': [0.9288, 0.0712],
                'Winter': [0.242, 0.085, 0.644, 0.029],
                'Spring': [0.7834, 0.2166],
                'Summer': [0.8, 0.2]
            }
            
            # Initialize a dictionary to store the final DataFrames for each season
            seasonal_dataframes = {}
            
            # Loop through each season and its corresponding DataFrame
            for season, dataset in season_datasets.items():
                # Clean the DataFrame to focus on relevant columns (ignoring the first unnamed column)
                df_cleaned = dataset.iloc[:, 1:]
            
                # Select a row based on the defined probabilities for the current season
                selected_row = df_cleaned.sample(n=1, weights=probabilities[season], random_state=1)
            
                # Keep the time index and convert it to a datetime format
                selected_row.reset_index(drop=True, inplace=True)
            
                # Melt the DataFrame to have time and value in two columns
                melted_row = selected_row.melt(var_name='Time', value_name='Value')
            
                # Convert the 'Time' column to datetime for easier manipulation
                melted_row['Time'] = pd.to_datetime(melted_row['Time'], format='%H:%M')
            
                # Create a new column for the hour in military time
                melted_row['Hour'] = melted_row['Time'].dt.strftime('%H')
            
                # Add a 'Season' column with the current season
                melted_row['Season'] = season
            
                # Group by hour and average the values
                hourly_avg = melted_row.groupby(['Hour', 'Season'])['Value'].mean().reset_index()
            
                # Rename the columns for clarity
                hourly_avg.columns = ['Hour', 'Season', 'Average Value']
            
                # Store the seasonal DataFrame in the dictionary
                seasonal_dataframes[season] = hourly_avg
            
            # Save each DataFrame as a separate variable
            Customer_HP_Fall = seasonal_dataframes['Fall']
            Customer_HP_Winter = seasonal_dataframes['Winter']
            Customer_HP_Spring = seasonal_dataframes['Spring']
            Customer_HP_Summer = seasonal_dataframes['Summer']
            
            Customer_HP_Fall.iloc[:, -1] = 0
            Customer_HP_Winter.iloc[:, -1] = 0    
            Customer_HP_Spring.iloc[:, -1] = 0    
            Customer_HP_Summer.iloc[:, -1] = 0    
            
        
            
            # Function to select a sheet based on manually entered probabilities and print the selected sheet
            def select_and_sample_with_named_dataframe(prob_1FE, prob_2FE):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Fall_Weekend1', 'EV_Fall_Weekend2'], p=[prob_1FE, prob_2FE])
                
                # # Print the selected spreadsheet
                # print(f"Selected Spreadsheet: {sheet_choice}")
                
                # Select the appropriate dataframe
                selected_df = EV_Fall_Weekend1 if sheet_choice == 'EV_Fall_Weekend1' else EV_Fall_Weekend2
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Fall'
                day_type = 'weekend'
                
                # Create a DataFrame
                Customer_Ev_Fall_Weekend = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Fall_Weekend
            
        
            prob_1FE = 0.8824  # Probability for selecting EV_Fall_Weekend1
            prob_2FE = 0.1176  # Probability for selecting EV_Fall_Weekend2
            Customer_Ev_Fall_Weekend = select_and_sample_with_named_dataframe(prob_1FE, prob_2FE)
            
            
            # Function to select a sheet based on manually entered probabilities and print the selected sheet
            def select_and_sample_with_named_dataframe(prob_1SPE, prob_2SPE):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Spring_Weekend1', 'EV_Spring_Weekend2'], p=[prob_1SPE, prob_2SPE])
                
                # # Print the selected spreadsheet
                # print(f"Selected Spreadsheet: {sheet_choice}")
                
                # Select the appropriate dataframe
                selected_df = EV_Spring_Weekend1 if sheet_choice == 'EV_Spring_Weekend1' else EV_Spring_Weekend2
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Spring'
                day_type = 'weekend'
                
                # Create a DataFrame
                Customer_Ev_Spring_Weekend = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Spring_Weekend
            
        
            prob_1SPE = 0.5987  # Probability for selecting EV_Spring_Weekend1
            prob_2SPE = 0.4013  # Probability for selecting EV_Spring_Weekend2
            Customer_Ev_Spring_Weekend = select_and_sample_with_named_dataframe(prob_1SPE, prob_2SPE)
            
            
            
            # Function to select a sheet based on manually entered probabilities and print the selected sheet
            def select_and_sample_with_named_dataframe(prob_1WE, prob_2WE):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Winter_Weekend1', 'EV_Winter_Weekend2'], p=[prob_1WE, prob_2WE])
                
                # # Print the selected spreadsheet
                # print(f"Selected Spreadsheet: {sheet_choice}")
                
                # Select the appropriate dataframe
                selected_df = EV_Winter_Weekend1 if sheet_choice == 'EV_Winter_Weekend1' else EV_Winter_Weekend2
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Winter'
                day_type = 'weekend'
                
                # Create a DataFrame
                Customer_Ev_Winter_Weekend = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Winter_Weekend
            
            prob_1WE =  0.6688  # Probability for selecting EV_Fall_Weekend1
            prob_2WE = 0.3312  # Probability for selecting EV_Fall_Weekend2
            Customer_Ev_Winter_Weekend = select_and_sample_with_named_dataframe(prob_1WE, prob_2WE)
            
            
            
            def select_and_sample_with_named_dataframe(prob_1SUE, prob_2SUE, prob_3SUE):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Summer_Weekend1', 'EV_Summer_Weekend2', 'EV_Summer_Weekend3'], 
                                                p=[prob_1SUE, prob_2SUE, prob_3SUE])
                
                # # Print the selected spreadsheet
                # print(f"Selected Spreadsheet: {sheet_choice}")
                
                # Select the appropriate dataframe
                if sheet_choice == 'EV_Summer_Weekend1':
                    selected_df = EV_Summer_Weekend1
                elif sheet_choice == 'EV_Summer_Weekend2':
                    selected_df = EV_Summer_Weekend2
                else:
                    selected_df = EV_Summer_Weekend3
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Summer'
                day_type = 'weekend'
                
                # Create a DataFrame
                Customer_Ev_Summer_Weekend = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Summer_Weekend
            
        
            prob_1SUE = 0.3211  # Probability for selecting EV_Summer_Weekend1
            prob_2SUE = 0.4495  # Probability for selecting EV_Summer_Weekend2
            prob_3SUE = 0.2294  # Probability for selecting EV_Summer_Weekend3
            Customer_Ev_Summer_Weekend = select_and_sample_with_named_dataframe(prob_1SUE, prob_2SUE, prob_3SUE)
            
            
            
            def select_and_sample_with_named_dataframe(prob_1SUD, prob_2SUD, prob_3SUD):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Summer_Weekdays1', 'EV_Summer_Weekdays2', 'EV_Summer_Weekdays3'], 
                                                p=[prob_1SUD, prob_2SUD, prob_3SUD])
                
                
                # Select the appropriate dataframe
                if sheet_choice == 'EV_Summer_Weekdays1':
                    selected_df = EV_Summer_Weekdays1
                elif sheet_choice == 'EV_Summer_Weekdays2':
                    selected_df = EV_Summer_Weekdays2
                else:
                    selected_df = EV_Summer_Weekdays3
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Summer'
                day_type = 'weekday'
                
                # Create a DataFrame
                Customer_Ev_Summer_Weekday = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Summer_Weekday
            
            prob_1SUD = 0.2698 
            prob_2SUD= 0.2734  
            prob_3SUD = 0.4568 
            Customer_Ev_Summer_Weekday = select_and_sample_with_named_dataframe(prob_1SUD, prob_2SUD, prob_3SUD)
            
            
            
            def select_and_sample_with_named_dataframe(prob_1SPD, prob_2SPD, prob_3SPD):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Spring_Weekdays1', 'EV_Spring_Weekdays2', 'EV_Spring_Weekdays3'], 
                                                p=[prob_1SPD, prob_2SPD, prob_3SPD])
                
                # # Print the selected spreadsheet
                # print(f"Selected Spreadsheet: {sheet_choice}")
                
                # Select the appropriate dataframe
                if sheet_choice == 'EV_Spring_Weekdays1':
                    selected_df = EV_Spring_Weekdays1
                elif sheet_choice == 'EV_Spring_Weekdays2':
                    selected_df = EV_Spring_Weekdays2
                else:
                    selected_df = EV_Spring_Weekdays3
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Spring'
                day_type = 'weekday'
                
                # Create a DataFrame
                Customer_Ev_Spring_Weekday = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Spring_Weekday
            
        
            prob_1SPD = 0.2793  
            prob_2SPD = 0.4441  
            prob_3SPD = 0.2766  
            Customer_Ev_Spring_Weekday = select_and_sample_with_named_dataframe(prob_1SPD, prob_2SPD, prob_3SPD)
            
            
            
            def select_and_sample_with_named_dataframe(prob_1FD, prob_2FD):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Fall_Weekdays1', 'EV_Fall_Weekdays2'], 
                                                p=[prob_1FD, prob_2FD])
                
                # # Print the selected spreadsheet
                # print(f"Selected Spreadsheet: {sheet_choice}")
                
                # Select the appropriate dataframe
                if sheet_choice == 'EV_Fall_Weekdays1':
                    selected_df = EV_Fall_Weekdays1
                else:
                    selected_df = EV_Fall_Weekdays2
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Fall'
                day_type = 'weekday'
                
                # Create a DataFrame
                Customer_Ev_Fall_Weekday = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Fall_Weekday
            
        
            prob_1FD = 0.6284  
            prob_2FD = 0.3716 
            Customer_Ev_Fall_Weekday = select_and_sample_with_named_dataframe(prob_1FD, prob_2FD)
            
            
            
            def select_and_sample_with_named_dataframe(prob_1WD, prob_2WD, prob_3WD, prob_4WD):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Winter_Weekdays1', 'EV_Winter_Weekdays2', 'EV_Winter_Weekdays3', 'EV_Winter_Weekdays4'], 
                                                p=[prob_1WD, prob_2WD, prob_3WD, prob_4WD])
                
                # # Print the selected spreadsheet
                # print(f"Selected Spreadsheet: {sheet_choice}")
                
                # Select the appropriate dataframe
                if sheet_choice == 'EV_Winter_Weekdays1':
                    selected_df = EV_Winter_Weekdays1
                elif sheet_choice == 'EV_Winter_Weekdays2':
                    selected_df = EV_Winter_Weekdays2
                elif sheet_choice == 'EV_Winter_Weekdays3':
                    selected_df = EV_Winter_Weekdays3
                else:
                    selected_df = EV_Winter_Weekdays4
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Winter'
                day_type = 'weekday'
                
                # Create a DataFrame
                Customer_Ev_Winter_Weekday = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Winter_Weekday
        
            prob_1WD = 0.3887  
            prob_2WD = 0.1841  
            prob_3WD = 0.1765  
            prob_4WD = 0.2507  
            Customer_Ev_Winter_Weekday = select_and_sample_with_named_dataframe(prob_1WD, prob_2WD, prob_3WD, prob_4WD)
            
            
            
            
            # Assuming Customer_Profile_AMI is already defined in the script
            # Slice the DataFrame into the specified seasonal ranges
            AMI_Winter1 = Customer_Profile_AMI.iloc[0:1416].copy()  # Rows 0 to 1415 (Winter 1)
            AMI_Spring = Customer_Profile_AMI.iloc[1416:3624].copy()  # Rows 1416 to 3623 (Spring)
            AMI_Summer = Customer_Profile_AMI.iloc[3624:5832].copy()  # Rows 3624 to 5831 (Summer)
            AMI_Fall = Customer_Profile_AMI.iloc[5832:8016].copy()  # Rows 5832 to 8015 (Fall)
            AMI_Winter2 = Customer_Profile_AMI.iloc[8016:8760].copy()  # Rows 8016 to 8759 (Winter 2)
            
        
        
            
            # Create a new DataFrame by repeating Customer_HP_Winter 59 times
            def create_repeated_customer_hp_winter(customer_hp_df, repeat_times):
                # Repeat the Customer_HP_Winter data 'repeat_times' times
                repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
                
                return repeated_customer_hp
            
            # Create a new DataFrame where Customer_HP_Winter is repeated 59 times
            Customer_W1 = create_repeated_customer_hp_winter(Customer_HP_Winter, 59)
            
            
            # Function to add column 5 of AMI_Winter1 to column 3 of Customer_W1
            def add_columns(ami_df, customer_df):
                # Extract column 5 from AMI_Winter1 (0-indexed, so it's column at index 4)
                ami_column = ami_df.iloc[:, 4]
                
                # Extract column 3 from Customer_W1 (0-indexed, so it's column at index 2)
                customer_column = customer_df.iloc[:, 2]
                
                # Add the two columns together
                new_column = ami_column + customer_column
                
                # Create a new DataFrame with the added column
                new_df = customer_df.copy()
                new_df['HP_Plus_AMI'] = new_column
                
                return new_df
            
            # Assuming AMI_Winter1 and Customer_W1 are already defined in the environment
            HP_Plus_AMI_W1 = add_columns(AMI_Winter1, Customer_W1)
                
            
            def create_repeated_customer_hp_spring(customer_hp_df, repeat_times):
                # Repeat the Customer_HP_Spring data 'repeat_times' times
                repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
                return repeated_customer_hp
            
            # Assuming Customer_HP_Spring is already defined
            Customer_Spring = create_repeated_customer_hp_spring(Customer_HP_Spring, 92)
            
        
            # Function to add column 5 of AMI_Spring to column 3 of Customer_Spring
            def add_columns_spring(ami_df, customer_df):
                # Reset indices to ensure proper alignment
                ami_df = ami_df.reset_index(drop=True)
                customer_df = customer_df.reset_index(drop=True)
                
                # Extract relevant columns and ensure they're numeric
                ami_column = pd.to_numeric(ami_df.iloc[:, 4], errors='coerce')  # Column 5 in AMI
                customer_column = pd.to_numeric(customer_df.iloc[:, 2], errors='coerce')  # Column 3 in Customer_HP
                
                # Add the columns together
                new_column = ami_column + customer_column
                
                # Create a new DataFrame with the added column
                new_df = customer_df.copy()
                new_df['HP_Plus_AMI_Spring'] = new_column
                
                return new_df
            
            # Assuming AMI_Spring and Customer_Spring are already defined in your environment
            HP_Plus_AMI_Spring = add_columns_spring(AMI_Spring, Customer_Spring)
            
        
            
            
            # Create a new DataFrame by repeating Customer_HP_Summer 92 times
            def create_repeated_customer_hp_summer(customer_hp_df, repeat_times):
                # Repeat the Customer_HP_Summer data 'repeat_times' times
                repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
                return repeated_customer_hp
            
            # Assuming Customer_HP_Summer is already defined
            Customer_Summer = create_repeated_customer_hp_summer(Customer_HP_Summer, 92)
            
            # Display the shape and first few rows of the new DataFrame to verify the operation
            # print(f'Customer_Summer shape: {Customer_Summer.shape}')
            # print(Customer_Summer.head())
            
            def add_columns_summer(ami_df, customer_df):
                # Reset indices to ensure proper alignment
                ami_df = ami_df.reset_index(drop=True)
                customer_df = customer_df.reset_index(drop=True)
                
                # Extract relevant columns and ensure they're numeric
                ami_column = pd.to_numeric(ami_df.iloc[:, 4], errors='coerce')  # Column 5 in AMI
                customer_column = pd.to_numeric(customer_df.iloc[:, 2], errors='coerce')  # Column 3 in Customer_HP
                
                # Add the columns together
                new_column = ami_column + customer_column
                
                # Create a new DataFrame with the added column
                new_df = customer_df.copy()
                new_df['HP_Plus_AMI_Summer'] = new_column
                
                return new_df
            
            HP_Plus_AMI_Summer = add_columns_summer(AMI_Summer, Customer_Summer)
            
        
            
            # Create a new DataFrame by repeating Customer_HP_Fall 91 times
            def create_repeated_customer_hp_fall(customer_hp_df, repeat_times):
                # Repeat the Customer_HP_Fall data 'repeat_times' times
                repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
                return repeated_customer_hp
            
            # Assuming Customer_HP_Fall is already defined
            Customer_Fall = create_repeated_customer_hp_fall(Customer_HP_Fall, 91)
            
            
            # Function to add column 5 of AMI_Fall to column 3 of Customer_Fall
            def add_columns_fall(ami_df, customer_df):
                # Reset indices to ensure proper alignment
                ami_df = ami_df.reset_index(drop=True)
                customer_df = customer_df.reset_index(drop=True)
                
                # Extract relevant columns and ensure they're numeric
                ami_column = pd.to_numeric(ami_df.iloc[:, 4], errors='coerce')  # Column 5 in AMI
                customer_column = pd.to_numeric(customer_df.iloc[:, 2], errors='coerce')  # Column 3 in Customer_HP
                
                # Add the columns together
                new_column = ami_column + customer_column
                
                # Create a new DataFrame with the added column
                new_df = customer_df.copy()
                new_df['HP_Plus_AMI_Fall'] = new_column
                
                return new_df
            
            # Assuming AMI_Fall and Customer_Fall are already defined
            HP_Plus_AMI_Fall = add_columns_fall(AMI_Fall, Customer_Fall)
                
            
            
            # Create a new DataFrame by repeating Customer_HP_Winter 31 times
            def create_repeated_customer_hp_winter(customer_hp_df, repeat_times):
                # Repeat the Customer_HP_Winter data 'repeat_times' times
                repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
                return repeated_customer_hp
            
            # Assuming Customer_HP_Winter is already defined
            Customer_W2 = create_repeated_customer_hp_winter(Customer_HP_Winter, 31)
        
            
            # Function to add column 5 of AMI_Winter2 to column 3 of Customer_W2
            def add_columns_winter2(ami_df, customer_df):
                # Reset indices to ensure proper alignment
                ami_df = ami_df.reset_index(drop=True)
                customer_df = customer_df.reset_index(drop=True)
                
                # Extract column 5 from AMI_Winter2 (0-indexed, so it's column at index 4) and ensure it's numeric
                ami_column = pd.to_numeric(ami_df.iloc[:, 4], errors='coerce')
                
                # Extract column 3 from Customer_W2 (0-indexed, so it's column at index 2) and ensure it's numeric
                customer_column = pd.to_numeric(customer_df.iloc[:, 2], errors='coerce')
                
                # Add the two columns together
                new_column = ami_column + customer_column
                
                # Create a new DataFrame with the added column
                new_df = customer_df.copy()
                new_df['HP_Plus_AMI_W2'] = new_column
                
                return new_df
            
            # Assuming AMI_Winter2 and Customer_W2 are already defined
            HP_Plus_AMI_W2 = add_columns_winter2(AMI_Winter2, Customer_W2)
            
            
            
            # Function to add a datetime column to represent each hour starting from January 1st, as the first column
            def add_datetime_column_first(dataframe, start_date=None):
                if start_date is None:  
                    start_date = f"{detected_year}-01-01"  # Dynamically set the year
                date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
                
                # Insert this date range as the first column in the DataFrame
                dataframe.insert(0, 'Datetime', date_range)
                
                return dataframe
            
            # Add a datetime column to HP_Plus_AMI_W1 starting from January 1st, 2023, as the first column
            HP_Plus_AMI_W1 = add_datetime_column_first(HP_Plus_AMI_W1)
            
            # Display the first few rows of the DataFrame to verify the operation
            # print(HP_Plus_AMI_W1.head())
            
            def add_datetime_column_first_spring(dataframe, start_date=None):
                if start_date is None:  
                    start_date = f"{detected_year}-03-01"  # Dynamically set the year
                date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
                
                # Insert this date range as the first column in the DataFrame
                dataframe.insert(0, 'Datetime', date_range)
                
                return dataframe
            
            # Add a datetime column to HP_Plus_AMI_Spring starting from March 1st, 2023, as the first column
            HP_Plus_AMI_Spring = add_datetime_column_first_spring(HP_Plus_AMI_Spring)
            
        
            
            # Function to add a datetime column to represent each hour starting from June 1st, 2023, as the first column
            def add_datetime_column_first_summer(dataframe, start_date=None):
                if start_date is None:  
                    start_date = f"{detected_year}-06-01"  # Dynamically set the year
                date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
                
                # Insert this date range as the first column in the DataFrame
                dataframe.insert(0, 'Datetime', date_range)
                
                return dataframe
            
            # Add a datetime column to HP_Plus_AMI_Summer starting from June 1st, 2023, as the first column
            HP_Plus_AMI_Summer = add_datetime_column_first_summer(HP_Plus_AMI_Summer)
            
            
            # Function to add a datetime column to represent each hour starting from August 1st, 2023, as the first column
            def add_datetime_column_first_fall(dataframe, start_date=None):
                if start_date is None:  
                    start_date = f"{detected_year}-08-01"  # Dynamically set the year
                date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
                
                # Insert this date range as the first column in the DataFrame
                dataframe.insert(0, 'Datetime', date_range)
                
                return dataframe
            
            # Add a datetime column to HP_Plus_AMI_Fall starting from August 1st, 2023, as the first column
            HP_Plus_AMI_Fall = add_datetime_column_first_fall(HP_Plus_AMI_Fall)
            
            
            
            # Function to add a datetime column to represent each hour starting from December 1st, 2023, as the first column
            def add_datetime_column_first_winter2(dataframe, start_date=None):
                if start_date is None:  
                    start_date = f"{detected_year}-12-01"  # Dynamically set the year
                date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
                
                # Insert this date range as the first column in the DataFrame
                dataframe.insert(0, 'Datetime', date_range)
                
                return dataframe
            
            # Add a datetime column to HP_Plus_AMI_W2 starting from December 1st, 2023, as the first column
            HP_Plus_AMI_W2 = add_datetime_column_first_winter2(HP_Plus_AMI_W2)
            
        
            
            # Function to add a weekday/weekend column to a DataFrame, making it the second column
            def add_weekday_weekend_column(dataframe, datetime_col='Datetime'):
                # Determine if each date in the 'Datetime' column is a weekend or a weekday
                weekday_or_weekend = dataframe[datetime_col].dt.weekday.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
                
                # Insert the new column as the second column in the DataFrame
                dataframe.insert(1, 'Weekday_Weekend', weekday_or_weekend)
                
                return dataframe
            
            # Add the weekday/weekend column to each season and W2
            HP_Plus_AMI_W1 = add_weekday_weekend_column(HP_Plus_AMI_W1)
            HP_Plus_AMI_Spring = add_weekday_weekend_column(HP_Plus_AMI_Spring)
            HP_Plus_AMI_Summer = add_weekday_weekend_column(HP_Plus_AMI_Summer)
            HP_Plus_AMI_Fall = add_weekday_weekend_column(HP_Plus_AMI_Fall)
            HP_Plus_AMI_W2 = add_weekday_weekend_column(HP_Plus_AMI_W2)
            
        
            
            # Function to add the columns together only for weekdays in HP_Plus_AMI_W1
            def add_ev_hp_columns_weekdays(customer_ev_df, hp_plus_ami_df):
                # Repeat the Customer_Ev_Winter_Weekday to match the number of days in HP_Plus_AMI_W1 (59 days)
                repeated_customer_ev = pd.concat([customer_ev_df] * 59, ignore_index=True)
                    # Reset indices for alignment
                repeated_customer_ev = repeated_customer_ev.reset_index(drop=True)
                hp_plus_ami_df = hp_plus_ami_df.reset_index(drop=True)
                
                # Extract the relevant columns (0-indexed: column 4 is index 3 for Customer_Ev, column 6 is index 5 for HP_Plus_AMI)
                customer_ev_column = pd.to_numeric(repeated_customer_ev.iloc[:, 3], errors='coerce')
                hp_plus_ami_column = pd.to_numeric(hp_plus_ami_df.iloc[:, 5], errors='coerce')
                
                # Add the columns together only for rows in HP_Plus_AMI_W1 that are weekdays
                combined_column = hp_plus_ami_column.copy()
                combined_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekday'] += customer_ev_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekday']
                
                # Create a new DataFrame with the result
                new_df = hp_plus_ami_df.copy()
                new_df['EV_HP_Combined'] = combined_column
                
                return new_df
            
            # Assuming Customer_Ev_Winter_Weekday and HP_Plus_AMI_W1 are already defined
            HP_Plus_AMI_W1_with_EV = add_ev_hp_columns_weekdays(Customer_Ev_Winter_Weekday, HP_Plus_AMI_W1)
            
            
            # Function to add the columns together only for weekends in HP_Plus_AMI_W1
            def add_ev_hp_columns_weekends(customer_ev_df, hp_plus_ami_df):
                # Repeat the Customer_Ev_Winter_Weekday to match the number of days in HP_Plus_AMI_W1 (59 days)
                repeated_customer_ev = pd.concat([customer_ev_df] * 59, ignore_index=True)
                    # Reset indices for alignment
                repeated_customer_ev = repeated_customer_ev.reset_index(drop=True)
                hp_plus_ami_df = hp_plus_ami_df.reset_index(drop=True)
                
                # Extract the relevant columns (0-indexed: column 4 is index 3 for Customer_Ev, column 6 is index 5 for HP_Plus_AMI)
                customer_ev_column = pd.to_numeric(repeated_customer_ev.iloc[:, 3], errors='coerce')
                hp_plus_ami_column = pd.to_numeric(hp_plus_ami_df.iloc[:, 5], errors='coerce')
                
                # Add the columns together only for rows in HP_Plus_AMI_W1 that are weekends
                combined_column = hp_plus_ami_column.copy()
                combined_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekend'] += customer_ev_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekend']
                
                # Create a new DataFrame with the result
                new_df = hp_plus_ami_df.copy()
                new_df['EV_HP_Combined'] = combined_column
                
                return new_df
            
            # Assuming Customer_Ev_Winter_Weekday and HP_Plus_AMI_W1 are already defined
            HP_Plus_AMI_W1_with_EV_weekends = add_ev_hp_columns_weekends(Customer_Ev_Winter_Weekday, HP_Plus_AMI_W1)
            
            
            
            # Function to add the columns together for weekdays in a given season
            def add_ev_hp_columns_weekdays(customer_ev_df, hp_plus_ami_df, repeat_days):
                # Repeat the Customer_EV data to match the number of days in the seasonal DataFrame
                repeated_customer_ev = pd.concat([customer_ev_df] * repeat_days, ignore_index=True)
                    # Reset indices for alignment
                repeated_customer_ev = repeated_customer_ev.reset_index(drop=True)
                hp_plus_ami_df = hp_plus_ami_df.reset_index(drop=True)
                
                # Extract the relevant columns (0-indexed: column 4 is index 3 for Customer_EV, column 6 is index 5 for HP_Plus_AMI)
                customer_ev_column = pd.to_numeric(repeated_customer_ev.iloc[:, 3], errors='coerce')
                hp_plus_ami_column = pd.to_numeric(hp_plus_ami_df.iloc[:, 5], errors='coerce')
                
                # Add the columns together only for rows in HP_Plus_AMI that are weekdays
                combined_column = hp_plus_ami_column.copy()
                combined_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekday'] += customer_ev_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekday']
                
                # Create a new DataFrame with the result
                new_df = hp_plus_ami_df.copy()
                new_df['EV_HP_Combined'] = combined_column
                
                return new_df
            
            # Function to add the columns together for weekends in a given season
            def add_ev_hp_columns_weekends(customer_ev_df, hp_plus_ami_df, repeat_days):
                # Repeat the Customer_EV data to match the number of days in the seasonal DataFrame
                repeated_customer_ev = pd.concat([customer_ev_df] * repeat_days, ignore_index=True)
                    # Reset indices for alignment
                repeated_customer_ev = repeated_customer_ev.reset_index(drop=True)
                hp_plus_ami_df = hp_plus_ami_df.reset_index(drop=True)
                
                # Extract the relevant columns (0-indexed: column 4 is index 3 for Customer_EV, column 6 is index 5 for HP_Plus_AMI)
                customer_ev_column = pd.to_numeric(repeated_customer_ev.iloc[:, 3], errors='coerce')
                hp_plus_ami_column = pd.to_numeric(hp_plus_ami_df.iloc[:, 5], errors='coerce')
                
                # Add the columns together only for rows in HP_Plus_AMI that are weekends
                combined_column = hp_plus_ami_column.copy()
                combined_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekend'] += customer_ev_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekend']
                
                # Create a new DataFrame with the result
                new_df = hp_plus_ami_df.copy()
                new_df['EV_HP_Combined'] = combined_column
                
                return new_df
            
            # Assuming seasonal Customer_EV DataFrames and HP_Plus_AMI DataFrames are defined
            
            # Spring (92 days)
            HP_Plus_AMI_Spring_with_EV_weekdays = add_ev_hp_columns_weekdays(Customer_Ev_Spring_Weekday, HP_Plus_AMI_Spring, 92)
            HP_Plus_AMI_Spring_with_EV_weekends = add_ev_hp_columns_weekends(Customer_Ev_Spring_Weekend, HP_Plus_AMI_Spring, 92)
            
            # Summer (92 days)
            HP_Plus_AMI_Summer_with_EV_weekdays = add_ev_hp_columns_weekdays(Customer_Ev_Summer_Weekday, HP_Plus_AMI_Summer, 92)
            HP_Plus_AMI_Summer_with_EV_weekends = add_ev_hp_columns_weekends(Customer_Ev_Summer_Weekend, HP_Plus_AMI_Summer, 92)
            
            # Fall (91 days)
            HP_Plus_AMI_Fall_with_EV_weekdays = add_ev_hp_columns_weekdays(Customer_Ev_Fall_Weekday, HP_Plus_AMI_Fall, 91)
            HP_Plus_AMI_Fall_with_EV_weekends = add_ev_hp_columns_weekends(Customer_Ev_Fall_Weekend, HP_Plus_AMI_Fall, 91)
            
            # Winter 2 (31 days)
            HP_Plus_AMI_W2_with_EV_weekdays = add_ev_hp_columns_weekdays(Customer_Ev_Winter_Weekday, HP_Plus_AMI_W2, 31)
            HP_Plus_AMI_W2_with_EV_weekends = add_ev_hp_columns_weekends(Customer_Ev_Winter_Weekend, HP_Plus_AMI_W2, 31)
            
            
            
            # Function to filter for weekdays or weekends and select specific columns
            def filter_and_select_columns(df, filter_condition, columns_to_keep):
                # Filter the DataFrame for weekdays or weekends
                filtered_df = df[df['Weekday_Weekend'] == filter_condition]
                
                # Select specific columns (0-indexed: columns 1, 2, 3, 4, and 7 correspond to indices 0, 1, 2, 3, and 6)
                filtered_df = filtered_df.iloc[:, columns_to_keep]
                
                return filtered_df
            
            # Indices of the columns to keep: [0, 1, 2, 3, 6] corresponding to columns 1, 2, 3, 4, and 7
            columns_to_keep = [0, 1, 2, 3, 6]
            
            # Filter and keep only weekends for EV_weekends DataFrames
            winter1_weekends = filter_and_select_columns(HP_Plus_AMI_W1_with_EV_weekends, 'Weekend', columns_to_keep)
            spring_weekends = filter_and_select_columns(HP_Plus_AMI_Spring_with_EV_weekends, 'Weekend', columns_to_keep)
            summer_weekends = filter_and_select_columns(HP_Plus_AMI_Summer_with_EV_weekends, 'Weekend', columns_to_keep)
            fall_weekends = filter_and_select_columns(HP_Plus_AMI_Fall_with_EV_weekends, 'Weekend', columns_to_keep)
            winter2_weekends = filter_and_select_columns(HP_Plus_AMI_W2_with_EV_weekends, 'Weekend', columns_to_keep)
            
            # Filter and keep only weekdays for EV_weekdays DataFrames
            winter1_weekdays = filter_and_select_columns(HP_Plus_AMI_W1_with_EV, 'Weekday', columns_to_keep)
            spring_weekdays = filter_and_select_columns(HP_Plus_AMI_Spring_with_EV_weekdays, 'Weekday', columns_to_keep)
            summer_weekdays = filter_and_select_columns(HP_Plus_AMI_Summer_with_EV_weekdays, 'Weekday', columns_to_keep)
            fall_weekdays = filter_and_select_columns(HP_Plus_AMI_Fall_with_EV_weekdays, 'Weekday', columns_to_keep)
            winter2_weekdays = filter_and_select_columns(HP_Plus_AMI_W2_with_EV_weekdays, 'Weekday', columns_to_keep)
            
            # Combine all the filtered DataFrames into a single DataFrame
            Customer_Total_Usage = pd.concat([winter1_weekends, spring_weekends, summer_weekends, fall_weekends, winter2_weekends,
                                              winter1_weekdays, spring_weekdays, summer_weekdays, fall_weekdays, winter2_weekdays], 
                                              ignore_index=True)
            
        
            # Rename the 5th column in Customer_Total_Usage with the name of the customer from the 5th column of Customer_Profile_AMI
            def rename_customer_column_from_profile(usage_df, profile_df):
                # Get the name of the 5th column from Customer_Profile_AMI
                customer_name = profile_df.columns[4]
                
                # Rename the 5th column (0-indexed: index 4) in Customer_Total_Usage
                usage_df.rename(columns={usage_df.columns[4]: customer_name}, inplace=True)
                
                return usage_df
            
            # Assuming Customer_Profile_AMI and Customer_Total_Usage are already defined
            Customer_Total_Usage = rename_customer_column_from_profile(Customer_Total_Usage, Customer_Profile_AMI)
            
        
            
            # Extract the customer name from the 5th column of Customer_Profile_AMI
            customer_name = Customer_Profile_AMI.columns[4]
            
            def replace_customer_profile(ami_df, total_usage_df, customer_name):
                # Check if the customer column exists in both DataFrames
                if customer_name in ami_df.columns and customer_name in total_usage_df.columns:
                    # Replace the original customer profile in AMI_Hourly with the new aggregated profile
                    ami_df[customer_name] = total_usage_df[customer_name].values
                else:
                    print(f"Customer '{customer_name}' not found in one of the DataFrames.")
                
                return ami_df
            
            # Replace the original customer profile with the new aggregated profile
            AMI_PLUS_EV = replace_customer_profile(AMI_Hourly, Customer_Total_Usage, customer_name)
            
        
        
        #%% AMI PLUS HP
        
        
        for customer in selected_customers_HP:
            
            # Extract the AMI profile for the selected customer
            customer_ami_profile = AMI_Total[customer] #Generate AMI Profile 
            
            excluded_columns = AMI_Hourly.iloc[:, :4]
            
            Customer_Profile_AMI = pd.concat([excluded_columns, customer_ami_profile], axis=1)
        
            
            #Generate HP Profile for seasons to add to AMI Profile
            
            # Define a dictionary to map seasons to their corresponding DataFrames
            season_datasets = {
                'Fall': HP_Fall,
                'Winter': HP_Winter,
                'Spring': HP_Spring,
                'Summer': HP_Summer
            }
            
            # Define probabilities for each season
            probabilities = {
                'Fall': [0.9288, 0.0712],
                'Winter': [0.242, 0.085, 0.644, 0.029],
                'Spring': [0.7834, 0.2166],
                'Summer': [0.8, 0.2]
            }
            
            
            # Initialize a dictionary to store the final DataFrames for each season
            seasonal_dataframes = {}
            
            # Loop through each season and its corresponding DataFrame
            for season, dataset in season_datasets.items():
                # Clean the DataFrame to focus on relevant columns (ignoring the first unnamed column)
                df_cleaned = dataset.iloc[:, 1:]
            
                # Select a row based on the defined probabilities for the current season
                selected_row = df_cleaned.sample(n=1, weights=probabilities[season], random_state=1)
            
                # Keep the time index and convert it to a datetime format
                selected_row.reset_index(drop=True, inplace=True)
            
                # Melt the DataFrame to have time and value in two columns
                melted_row = selected_row.melt(var_name='Time', value_name='Value')
            
                # Convert the 'Time' column to datetime for easier manipulation
                melted_row['Time'] = pd.to_datetime(melted_row['Time'], format='%H:%M')
            
                # Create a new column for the hour in military time
                melted_row['Hour'] = melted_row['Time'].dt.strftime('%H')
            
                # Add a 'Season' column with the current season
                melted_row['Season'] = season
            
                # Group by hour and average the values
                hourly_avg = melted_row.groupby(['Hour', 'Season'])['Value'].mean().reset_index()
            
                # Rename the columns for clarity
                hourly_avg.columns = ['Hour', 'Season', 'Average Value']
            
                # Store the seasonal DataFrame in the dictionary
                seasonal_dataframes[season] = hourly_avg
            
            # Save each DataFrame as a separate variable
            Customer_HP_Fall = seasonal_dataframes['Fall']
            Customer_HP_Winter = seasonal_dataframes['Winter']
            Customer_HP_Spring = seasonal_dataframes['Spring']
            Customer_HP_Summer = seasonal_dataframes['Summer']
                
            
            
            # Function to select a sheet based on manually entered probabilities and print the selected sheet
            def select_and_sample_with_named_dataframe(prob_1FE, prob_2FE):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Fall_Weekend1', 'EV_Fall_Weekend2'], p=[prob_1FE, prob_2FE])
                
                # Print the selected spreadsheet
                # print(f"Selected Spreadsheet: {sheet_choice}")
                
                # Select the appropriate dataframe
                selected_df = EV_Fall_Weekend1 if sheet_choice == 'EV_Fall_Weekend1' else EV_Fall_Weekend2
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Fall'
                day_type = 'weekend'
                
                # Create a DataFrame
                Customer_Ev_Fall_Weekend = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Fall_Weekend
            
            prob_1FE = 0.6  # Probability for selecting EV_Fall_Weekend1
            prob_2FE = 0.4  # Probability for selecting EV_Fall_Weekend2
            Customer_Ev_Fall_Weekend = select_and_sample_with_named_dataframe(prob_1FE, prob_2FE)
            Customer_Ev_Fall_Weekend.iloc[:, -1] = 0
            
            # Function to select a sheet based on manually entered probabilities and print the selected sheet
            def select_and_sample_with_named_dataframe(prob_1SPE, prob_2SPE):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Spring_Weekend1', 'EV_Spring_Weekend2'], p=[prob_1SPE, prob_2SPE])
                
        
                
                # Select the appropriate dataframe
                selected_df = EV_Spring_Weekend1 if sheet_choice == 'EV_Spring_Weekend1' else EV_Spring_Weekend2
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Spring'
                day_type = 'weekend'
                
                # Create a DataFrame
                Customer_Ev_Spring_Weekend = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Spring_Weekend
            
        
            prob_1SPE = 0.5987  # Probability for selecting EV_Spring_Weekend1
            prob_2SPE = 0.4013 # Probability for selecting EV_Spring_Weekend2
            Customer_Ev_Spring_Weekend = select_and_sample_with_named_dataframe(prob_1SPE, prob_2SPE)
            Customer_Ev_Spring_Weekend.iloc[:, -1] = 0
            
            
            # Function to select a sheet based on manually entered probabilities and print the selected sheet
            def select_and_sample_with_named_dataframe(prob_1WE, prob_2WE):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Winter_Weekend1', 'EV_Winter_Weekend2'], p=[prob_1WE, prob_2WE])
        
                
                # Select the appropriate dataframe
                selected_df = EV_Winter_Weekend1 if sheet_choice == 'EV_Winter_Weekend1' else EV_Winter_Weekend2
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Winter'
                day_type = 'weekend'
                
                # Create a DataFrame
                Customer_Ev_Winter_Weekend = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Winter_Weekend
            
        
            prob_1WE = 0.6688  # Probability for selecting EV_Winter_Weekend1
            prob_2WE = 0.3312  # Probability for selecting EV_Winter_Weekend2
            Customer_Ev_Winter_Weekend = select_and_sample_with_named_dataframe(prob_1WE, prob_2WE)
            Customer_Ev_Winter_Weekend.iloc[:, -1] = 0
            
            
            def select_and_sample_with_named_dataframe(prob_1SUE, prob_2SUE, prob_3SUE):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Summer_Weekend1', 'EV_Summer_Weekend2', 'EV_Summer_Weekend3'], 
                                                p=[prob_1SUE, prob_2SUE, prob_3SUE])
                
               
                # Select the appropriate dataframe
                if sheet_choice == 'EV_Summer_Weekend1':
                    selected_df = EV_Summer_Weekend1
                elif sheet_choice == 'EV_Summer_Weekend2':
                    selected_df = EV_Summer_Weekend2
                else:
                    selected_df = EV_Summer_Weekend3
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Summer'
                day_type = 'weekend'
                
                # Create a DataFrame
                Customer_Ev_Summer_Weekend = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Summer_Weekend
            
            prob_1SUE = 0.4  # Probability for selecting EV_Summer_Weekend1
            prob_2SUE = 0.3  # Probability for selecting EV_Summer_Weekend2
            prob_3SUE = 0.3  # Probability for selecting EV_Summer_Weekend3
            Customer_Ev_Summer_Weekend = select_and_sample_with_named_dataframe(prob_1SUE, prob_2SUE, prob_3SUE)
            Customer_Ev_Summer_Weekend.iloc[:, -1] = 0
            
            
            def select_and_sample_with_named_dataframe(prob_1SUD, prob_2SUD, prob_3SUD):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Summer_Weekdays1', 'EV_Summer_Weekdays2', 'EV_Summer_Weekdays3'], 
                                                p=[prob_1SUD, prob_2SUD, prob_3SUD])
                
                
                # Select the appropriate dataframe
                if sheet_choice == 'EV_Summer_Weekdays1':
                    selected_df = EV_Summer_Weekdays1
                elif sheet_choice == 'EV_Summer_Weekdays2':
                    selected_df = EV_Summer_Weekdays2
                else:
                    selected_df = EV_Summer_Weekdays3
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Summer'
                day_type = 'weekday'
                
                # Create a DataFrame
                Customer_Ev_Summer_Weekday = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Summer_Weekday
            
            # Example usage with probabilities for each sheet
            prob_1SUD = 0.2698  
            prob_2SUD= 0.2734  
            prob_3SUD = 0.4568 
            Customer_Ev_Summer_Weekday = select_and_sample_with_named_dataframe(prob_1SUD, prob_2SUD, prob_3SUD)
            Customer_Ev_Summer_Weekday.iloc[:, -1] = 0
            
            
            def select_and_sample_with_named_dataframe(prob_1SPD, prob_2SPD, prob_3SPD):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Spring_Weekdays1', 'EV_Spring_Weekdays2', 'EV_Spring_Weekdays3'], 
                                                p=[prob_1SPD, prob_2SPD, prob_3SPD])
                
                
                # Select the appropriate dataframe
                if sheet_choice == 'EV_Spring_Weekdays1':
                    selected_df = EV_Spring_Weekdays1
                elif sheet_choice == 'EV_Spring_Weekdays2':
                    selected_df = EV_Spring_Weekdays2
                else:
                    selected_df = EV_Spring_Weekdays3
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Spring'
                day_type = 'weekday'
                
                # Create a DataFrame
                Customer_Ev_Spring_Weekday = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Spring_Weekday
            
            prob_1SPD = 0.2793  
            prob_2SPD = 0.4441  
            prob_3SPD = 0.2766  
            Customer_Ev_Spring_Weekday = select_and_sample_with_named_dataframe(prob_1SPD, prob_2SPD, prob_3SPD)
            Customer_Ev_Spring_Weekday.iloc[:, -1] = 0
            
            
            def select_and_sample_with_named_dataframe(prob_1FD, prob_2FD):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Fall_Weekdays1', 'EV_Fall_Weekdays2'], 
                                                p=[prob_1FD, prob_2FD])
                
                # Print the selected spreadsheet
                # print(f"Selected Spreadsheet: {sheet_choice}")
                
                # Select the appropriate dataframe
                if sheet_choice == 'EV_Fall_Weekdays1':
                    selected_df = EV_Fall_Weekdays1
                else:
                    selected_df = EV_Fall_Weekdays2
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Fall'
                day_type = 'weekday'
                
                # Create a DataFrame
                Customer_Ev_Fall_Weekday = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Fall_Weekday
            
            # Example usage with probabilities for each sheet
            prob_1FD = 0.6284  
            prob_2FD = 0.3716  
            Customer_Ev_Fall_Weekday = select_and_sample_with_named_dataframe(prob_1FD, prob_2FD)
            Customer_Ev_Fall_Weekday.iloc[:, -1] = 0
            
            
            def select_and_sample_with_named_dataframe(prob_1WD, prob_2WD, prob_3WD, prob_4WD):
                # Select a sheet based on the probabilities
                sheet_choice = np.random.choice(['EV_Winter_Weekdays1', 'EV_Winter_Weekdays2', 'EV_Winter_Weekdays3', 'EV_Winter_Weekdays4'], 
                                                p=[prob_1WD, prob_2WD, prob_3WD, prob_4WD])
        
                
                # Select the appropriate dataframe
                if sheet_choice == 'EV_Winter_Weekdays1':
                    selected_df = EV_Winter_Weekdays1
                elif sheet_choice == 'EV_Winter_Weekdays2':
                    selected_df = EV_Winter_Weekdays2
                elif sheet_choice == 'EV_Winter_Weekdays3':
                    selected_df = EV_Winter_Weekdays3
                else:
                    selected_df = EV_Winter_Weekdays4
                
                # Randomly select a row, excluding the first row
                random_row = selected_df.iloc[np.random.randint(1, selected_df.shape[0])]
                
                # Create a new DataFrame with the desired format
                hours = list(range(24))
                season = 'Winter'
                day_type = 'weekday'
                
                # Create a DataFrame
                Customer_Ev_Winter_Weekday = pd.DataFrame({
                    'Hour': hours,
                    'Day Type': [day_type] * 24,
                    'Season': [season] * 24,
                    'Data': random_row.values
                })
                
                return Customer_Ev_Winter_Weekday
            
            prob_1WD = 0.3887  
            prob_2WD = 0.1841 
            prob_3WD = 0.1765  
            prob_4WD = 0.2507 
            Customer_Ev_Winter_Weekday = select_and_sample_with_named_dataframe(prob_1WD, prob_2WD, prob_3WD, prob_4WD)
            Customer_Ev_Winter_Weekday.iloc[:, -1] = 0
            
            
            # Slice the DataFrame into the specified seasonal ranges
            AMI_Winter1 = Customer_Profile_AMI.iloc[0:1416].copy()  # Rows 0 to 1415 (Winter 1)
            AMI_Spring = Customer_Profile_AMI.iloc[1416:3624].copy()  # Rows 1416 to 3623 (Spring)
            AMI_Summer = Customer_Profile_AMI.iloc[3624:5832].copy()  # Rows 3624 to 5831 (Summer)
            AMI_Fall = Customer_Profile_AMI.iloc[5832:8016].copy()  # Rows 5832 to 8015 (Fall)
            AMI_Winter2 = Customer_Profile_AMI.iloc[8016:8760].copy()  # Rows 8016 to 8759 (Winter 2)
            
            
            # Create a new DataFrame by repeating Customer_HP_Winter 59 times
            def create_repeated_customer_hp_winter(customer_hp_df, repeat_times):
                # Repeat the Customer_HP_Winter data 'repeat_times' times
                repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
                
                return repeated_customer_hp
            
            # Create a new DataFrame where Customer_HP_Winter is repeated 59 times
            Customer_W1 = create_repeated_customer_hp_winter(Customer_HP_Winter, 59)
            
            
            # Function to add column 5 of AMI_Winter1 to column 3 of Customer_W1
            def add_columns(ami_df, customer_df):
                # Extract column 5 from AMI_Winter1 (0-indexed, so it's column at index 4)
                ami_column = ami_df.iloc[:, 4]
                
                # Extract column 3 from Customer_W1 (0-indexed, so it's column at index 2)
                customer_column = customer_df.iloc[:, 2]
                
                # Add the two columns together
                new_column = ami_column + customer_column
                
                # Create a new DataFrame with the added column
                new_df = customer_df.copy()
                new_df['HP_Plus_AMI'] = new_column
                
                return new_df
            
            # Assuming AMI_Winter1 and Customer_W1 are already defined in the environment
            HP_Plus_AMI_W1 = add_columns(AMI_Winter1, Customer_W1)
        
            
            def create_repeated_customer_hp_spring(customer_hp_df, repeat_times):
                # Repeat the Customer_HP_Spring data 'repeat_times' times
                repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
                return repeated_customer_hp
            
            # Assuming Customer_HP_Spring is already defined
            Customer_Spring = create_repeated_customer_hp_spring(Customer_HP_Spring, 92)
        
            
            # Function to add column 5 of AMI_Spring to column 3 of Customer_Spring
            def add_columns_spring(ami_df, customer_df):
                # Reset indices to ensure proper alignment
                ami_df = ami_df.reset_index(drop=True)
                customer_df = customer_df.reset_index(drop=True)
                
                # Extract relevant columns and ensure they're numeric
                ami_column = pd.to_numeric(ami_df.iloc[:, 4], errors='coerce')  # Column 5 in AMI
                customer_column = pd.to_numeric(customer_df.iloc[:, 2], errors='coerce')  # Column 3 in Customer_HP
                
                # Add the columns together
                new_column = ami_column + customer_column
                
                # Create a new DataFrame with the added column
                new_df = customer_df.copy()
                new_df['HP_Plus_AMI_Spring'] = new_column
                
                return new_df
            
            # Assuming AMI_Spring and Customer_Spring are already defined in your environment
            HP_Plus_AMI_Spring = add_columns_spring(AMI_Spring, Customer_Spring)
            
            
            # Create a new DataFrame by repeating Customer_HP_Summer 92 times
            def create_repeated_customer_hp_summer(customer_hp_df, repeat_times):
                # Repeat the Customer_HP_Summer data 'repeat_times' times
                repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
                return repeated_customer_hp
            
            # Assuming Customer_HP_Summer is already defined
            Customer_Summer = create_repeated_customer_hp_summer(Customer_HP_Summer, 92)
            
            
            def add_columns_summer(ami_df, customer_df):
                # Reset indices to ensure proper alignment
                ami_df = ami_df.reset_index(drop=True)
                customer_df = customer_df.reset_index(drop=True)
                
                # Extract relevant columns and ensure they're numeric
                ami_column = pd.to_numeric(ami_df.iloc[:, 4], errors='coerce')  # Column 5 in AMI
                customer_column = pd.to_numeric(customer_df.iloc[:, 2], errors='coerce')  # Column 3 in Customer_HP
                
                # Add the columns together
                new_column = ami_column + customer_column
                
                # Create a new DataFrame with the added column
                new_df = customer_df.copy()
                new_df['HP_Plus_AMI_Summer'] = new_column
                
                return new_df
            
            HP_Plus_AMI_Summer = add_columns_summer(AMI_Summer, Customer_Summer)
            
        
            
            
            # Create a new DataFrame by repeating Customer_HP_Fall 91 times
            def create_repeated_customer_hp_fall(customer_hp_df, repeat_times):
                # Repeat the Customer_HP_Fall data 'repeat_times' times
                repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
                return repeated_customer_hp
            
        
            Customer_Fall = create_repeated_customer_hp_fall(Customer_HP_Fall, 91)
            
        
            
            # Function to add column 5 of AMI_Fall to column 3 of Customer_Fall
            def add_columns_fall(ami_df, customer_df):
                # Reset indices to ensure proper alignment
                ami_df = ami_df.reset_index(drop=True)
                customer_df = customer_df.reset_index(drop=True)
                
                # Extract relevant columns and ensure they're numeric
                ami_column = pd.to_numeric(ami_df.iloc[:, 4], errors='coerce')  # Column 5 in AMI
                customer_column = pd.to_numeric(customer_df.iloc[:, 2], errors='coerce')  # Column 3 in Customer_HP
                
                # Add the columns together
                new_column = ami_column + customer_column
                
                # Create a new DataFrame with the added column
                new_df = customer_df.copy()
                new_df['HP_Plus_AMI_Fall'] = new_column
                
                return new_df
            
            HP_Plus_AMI_Fall = add_columns_fall(AMI_Fall, Customer_Fall)
            
        
            # Create a new DataFrame by repeating Customer_HP_Winter 31 times
            def create_repeated_customer_hp_winter(customer_hp_df, repeat_times):
                # Repeat the Customer_HP_Winter data 'repeat_times' times
                repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
                return repeated_customer_hp
            
            # Assuming Customer_HP_Winter is already defined
            Customer_W2 = create_repeated_customer_hp_winter(Customer_HP_Winter, 31)
            
        
            
            # Function to add column 5 of AMI_Winter2 to column 3 of Customer_W2
            def add_columns_winter2(ami_df, customer_df):
                # Reset indices to ensure proper alignment
                ami_df = ami_df.reset_index(drop=True)
                customer_df = customer_df.reset_index(drop=True)
                
                # Extract column 5 from AMI_Winter2 (0-indexed, so it's column at index 4) and ensure it's numeric
                ami_column = pd.to_numeric(ami_df.iloc[:, 4], errors='coerce')
                
                # Extract column 3 from Customer_W2 (0-indexed, so it's column at index 2) and ensure it's numeric
                customer_column = pd.to_numeric(customer_df.iloc[:, 2], errors='coerce')
                
                # Add the two columns together
                new_column = ami_column + customer_column
                
                # Create a new DataFrame with the added column
                new_df = customer_df.copy()
                new_df['HP_Plus_AMI_W2'] = new_column
                
                return new_df
            
            # Assuming AMI_Winter2 and Customer_W2 are already defined
            HP_Plus_AMI_W2 = add_columns_winter2(AMI_Winter2, Customer_W2)
            
        
            
            # Function to add a datetime column to represent each hour starting from January 1st, as the first column
            def add_datetime_column_first(dataframe, start_date=None):
                if start_date is None:  
                    start_date = f"{detected_year}-01-01"  # Dynamically set the year
                date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
                
                # Insert this date range as the first column in the DataFrame
                dataframe.insert(0, 'Datetime', date_range)
                
                return dataframe
            
            # Add a datetime column to HP_Plus_AMI_W1 starting from January 1st, 2023, as the first column
            HP_Plus_AMI_W1 = add_datetime_column_first(HP_Plus_AMI_W1)
            
            
            def add_datetime_column_first_spring(dataframe, start_date=None):
                if start_date is None:  
                    start_date = f"{detected_year}-03-01"  # Dynamically set the year
                date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
                
                # Insert this date range as the first column in the DataFrame
                dataframe.insert(0, 'Datetime', date_range)
                
                return dataframe
            
            # Add a datetime column to HP_Plus_AMI_Spring starting from March 1st, 2023, as the first column
            HP_Plus_AMI_Spring = add_datetime_column_first_spring(HP_Plus_AMI_Spring)
            
        
            
            # Function to add a datetime column to represent each hour starting from June 1st, 2023, as the first column
            def add_datetime_column_first_summer(dataframe, start_date=None):
                if start_date is None:  
                    start_date = f"{detected_year}-06-01"  # Dynamically set the year
                date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
                
                # Insert this date range as the first column in the DataFrame
                dataframe.insert(0, 'Datetime', date_range)
                
                return dataframe
            
            # Add a datetime column to HP_Plus_AMI_Summer starting from June 1st, 2023, as the first column
            HP_Plus_AMI_Summer = add_datetime_column_first_summer(HP_Plus_AMI_Summer)
            
            
            # Function to add a datetime column to represent each hour starting from August 1st, 2023, as the first column
            def add_datetime_column_first_fall(dataframe, start_date=None):
                if start_date is None:  
                    start_date = f"{detected_year}-08-01"  # Dynamically set the year
                date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
                
                # Insert this date range as the first column in the DataFrame
                dataframe.insert(0, 'Datetime', date_range)
                
                return dataframe
            
            # Add a datetime column to HP_Plus_AMI_Fall starting from August 1st, 2023, as the first column
            HP_Plus_AMI_Fall = add_datetime_column_first_fall(HP_Plus_AMI_Fall)
            
            
            
            # Function to add a datetime column to represent each hour starting from December 1st, 2023, as the first column
            def add_datetime_column_first_winter2(dataframe, start_date=None):
                if start_date is None:  
                    start_date = f"{detected_year}-12-01"  # Dynamically set the year
                date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
                
                # Insert this date range as the first column in the DataFrame
                dataframe.insert(0, 'Datetime', date_range)
                
                return dataframe
            
            # Add a datetime column to HP_Plus_AMI_W2 starting from December 1st, as the first column
            HP_Plus_AMI_W2 = add_datetime_column_first_winter2(HP_Plus_AMI_W2)
        
            
            # Function to add a weekday/weekend column to a DataFrame, making it the second column
            def add_weekday_weekend_column(dataframe, datetime_col='Datetime'):
                # Determine if each date in the 'Datetime' column is a weekend or a weekday
                weekday_or_weekend = dataframe[datetime_col].dt.weekday.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
                
                # Insert the new column as the second column in the DataFrame
                dataframe.insert(1, 'Weekday_Weekend', weekday_or_weekend)
                
                return dataframe
            
            # Add the weekday/weekend column to each season and W2
            HP_Plus_AMI_W1 = add_weekday_weekend_column(HP_Plus_AMI_W1)
            HP_Plus_AMI_Spring = add_weekday_weekend_column(HP_Plus_AMI_Spring)
            HP_Plus_AMI_Summer = add_weekday_weekend_column(HP_Plus_AMI_Summer)
            HP_Plus_AMI_Fall = add_weekday_weekend_column(HP_Plus_AMI_Fall)
            HP_Plus_AMI_W2 = add_weekday_weekend_column(HP_Plus_AMI_W2)
                  
            
            
            # Function to add the columns together only for weekdays in HP_Plus_AMI_W1
            def add_ev_hp_columns_weekdays(customer_ev_df, hp_plus_ami_df):
                # Repeat the Customer_Ev_Winter_Weekday to match the number of days in HP_Plus_AMI_W1 (59 days)
                repeated_customer_ev = pd.concat([customer_ev_df] * 59, ignore_index=True)
                    # Reset indices for alignment
                repeated_customer_ev = repeated_customer_ev.reset_index(drop=True)
                hp_plus_ami_df = hp_plus_ami_df.reset_index(drop=True)
                
                # Extract the relevant columns (0-indexed: column 4 is index 3 for Customer_Ev, column 6 is index 5 for HP_Plus_AMI)
                customer_ev_column = pd.to_numeric(repeated_customer_ev.iloc[:, 3], errors='coerce')
                hp_plus_ami_column = pd.to_numeric(hp_plus_ami_df.iloc[:, 5], errors='coerce')
                
                # Add the columns together only for rows in HP_Plus_AMI_W1 that are weekdays
                combined_column = hp_plus_ami_column.copy()
                combined_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekday'] += customer_ev_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekday']
                
                # Create a new DataFrame with the result
                new_df = hp_plus_ami_df.copy()
                new_df['EV_HP_Combined'] = combined_column
                
                return new_df
            
            # Assuming Customer_Ev_Winter_Weekday and HP_Plus_AMI_W1 are already defined
            HP_Plus_AMI_W1_with_EV = add_ev_hp_columns_weekdays(Customer_Ev_Winter_Weekday, HP_Plus_AMI_W1)
            
        
            
            # Function to add the columns together only for weekends in HP_Plus_AMI_W1
            def add_ev_hp_columns_weekends(customer_ev_df, hp_plus_ami_df):
                # Repeat the Customer_Ev_Winter_Weekday to match the number of days in HP_Plus_AMI_W1 (59 days)
                repeated_customer_ev = pd.concat([customer_ev_df] * 59, ignore_index=True)
                    # Reset indices for alignment
                repeated_customer_ev = repeated_customer_ev.reset_index(drop=True)
                hp_plus_ami_df = hp_plus_ami_df.reset_index(drop=True)
                
                # Extract the relevant columns (0-indexed: column 4 is index 3 for Customer_Ev, column 6 is index 5 for HP_Plus_AMI)
                customer_ev_column = pd.to_numeric(repeated_customer_ev.iloc[:, 3], errors='coerce')
                hp_plus_ami_column = pd.to_numeric(hp_plus_ami_df.iloc[:, 5], errors='coerce')
                
                # Add the columns together only for rows in HP_Plus_AMI_W1 that are weekends
                combined_column = hp_plus_ami_column.copy()
                combined_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekend'] += customer_ev_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekend']
                
                # Create a new DataFrame with the result
                new_df = hp_plus_ami_df.copy()
                new_df['EV_HP_Combined'] = combined_column
                
                return new_df
            
            # Assuming Customer_Ev_Winter_Weekday and HP_Plus_AMI_W1 are already defined
            HP_Plus_AMI_W1_with_EV_weekends = add_ev_hp_columns_weekends(Customer_Ev_Winter_Weekday, HP_Plus_AMI_W1)
            
            
            
            # Function to add the columns together for weekdays in a given season
            def add_ev_hp_columns_weekdays(customer_ev_df, hp_plus_ami_df, repeat_days):
                # Repeat the Customer_EV data to match the number of days in the seasonal DataFrame
                repeated_customer_ev = pd.concat([customer_ev_df] * repeat_days, ignore_index=True)
                    # Reset indices for alignment
                repeated_customer_ev = repeated_customer_ev.reset_index(drop=True)
                hp_plus_ami_df = hp_plus_ami_df.reset_index(drop=True)
                
                # Extract the relevant columns (0-indexed: column 4 is index 3 for Customer_EV, column 6 is index 5 for HP_Plus_AMI)
                customer_ev_column = pd.to_numeric(repeated_customer_ev.iloc[:, 3], errors='coerce')
                hp_plus_ami_column = pd.to_numeric(hp_plus_ami_df.iloc[:, 5], errors='coerce')
                
                # Add the columns together only for rows in HP_Plus_AMI that are weekdays
                combined_column = hp_plus_ami_column.copy()
                combined_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekday'] += customer_ev_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekday']
                
                # Create a new DataFrame with the result
                new_df = hp_plus_ami_df.copy()
                new_df['EV_HP_Combined'] = combined_column
                
                return new_df
            
            # Function to add the columns together for weekends in a given season
            def add_ev_hp_columns_weekends(customer_ev_df, hp_plus_ami_df, repeat_days):
                # Repeat the Customer_EV data to match the number of days in the seasonal DataFrame
                repeated_customer_ev = pd.concat([customer_ev_df] * repeat_days, ignore_index=True)
                    # Reset indices for alignment
                repeated_customer_ev = repeated_customer_ev.reset_index(drop=True)
                hp_plus_ami_df = hp_plus_ami_df.reset_index(drop=True)
                
                # Extract the relevant columns (0-indexed: column 4 is index 3 for Customer_EV, column 6 is index 5 for HP_Plus_AMI)
                customer_ev_column = pd.to_numeric(repeated_customer_ev.iloc[:, 3], errors='coerce')
                hp_plus_ami_column = pd.to_numeric(hp_plus_ami_df.iloc[:, 5], errors='coerce')
                
                # Add the columns together only for rows in HP_Plus_AMI that are weekends
                combined_column = hp_plus_ami_column.copy()
                combined_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekend'] += customer_ev_column[hp_plus_ami_df['Weekday_Weekend'] == 'Weekend']
                
                # Create a new DataFrame with the result
                new_df = hp_plus_ami_df.copy()
                new_df['EV_HP_Combined'] = combined_column
                
                return new_df
            
            # Assuming seasonal Customer_EV DataFrames and HP_Plus_AMI DataFrames are defined
            
            # Spring (92 days)
            HP_Plus_AMI_Spring_with_EV_weekdays = add_ev_hp_columns_weekdays(Customer_Ev_Spring_Weekday, HP_Plus_AMI_Spring, 92)
            HP_Plus_AMI_Spring_with_EV_weekends = add_ev_hp_columns_weekends(Customer_Ev_Spring_Weekend, HP_Plus_AMI_Spring, 92)
            
            # Summer (92 days)
            HP_Plus_AMI_Summer_with_EV_weekdays = add_ev_hp_columns_weekdays(Customer_Ev_Summer_Weekday, HP_Plus_AMI_Summer, 92)
            HP_Plus_AMI_Summer_with_EV_weekends = add_ev_hp_columns_weekends(Customer_Ev_Summer_Weekend, HP_Plus_AMI_Summer, 92)
            
            # Fall (91 days)
            HP_Plus_AMI_Fall_with_EV_weekdays = add_ev_hp_columns_weekdays(Customer_Ev_Fall_Weekday, HP_Plus_AMI_Fall, 91)
            HP_Plus_AMI_Fall_with_EV_weekends = add_ev_hp_columns_weekends(Customer_Ev_Fall_Weekend, HP_Plus_AMI_Fall, 91)
            
            # Winter 2 (31 days)
            HP_Plus_AMI_W2_with_EV_weekdays = add_ev_hp_columns_weekdays(Customer_Ev_Winter_Weekday, HP_Plus_AMI_W2, 31)
            HP_Plus_AMI_W2_with_EV_weekends = add_ev_hp_columns_weekends(Customer_Ev_Winter_Weekend, HP_Plus_AMI_W2, 31)
            
            
            
            # Function to filter for weekdays or weekends and select specific columns
            def filter_and_select_columns(df, filter_condition, columns_to_keep):
                # Filter the DataFrame for weekdays or weekends
                filtered_df = df[df['Weekday_Weekend'] == filter_condition]
                
                # Select specific columns (0-indexed: columns 1, 2, 3, 4, and 7 correspond to indices 0, 1, 2, 3, and 6)
                filtered_df = filtered_df.iloc[:, columns_to_keep]
                
                return filtered_df
            
            # Indices of the columns to keep: [0, 1, 2, 3, 6] corresponding to columns 1, 2, 3, 4, and 7
            columns_to_keep = [0, 1, 2, 3, 6]
            
            # Filter and keep only weekends for EV_weekends DataFrames
            winter1_weekends = filter_and_select_columns(HP_Plus_AMI_W1_with_EV_weekends, 'Weekend', columns_to_keep)
            spring_weekends = filter_and_select_columns(HP_Plus_AMI_Spring_with_EV_weekends, 'Weekend', columns_to_keep)
            summer_weekends = filter_and_select_columns(HP_Plus_AMI_Summer_with_EV_weekends, 'Weekend', columns_to_keep)
            fall_weekends = filter_and_select_columns(HP_Plus_AMI_Fall_with_EV_weekends, 'Weekend', columns_to_keep)
            winter2_weekends = filter_and_select_columns(HP_Plus_AMI_W2_with_EV_weekends, 'Weekend', columns_to_keep)
            
            # Filter and keep only weekdays for EV_weekdays DataFrames
            winter1_weekdays = filter_and_select_columns(HP_Plus_AMI_W1_with_EV, 'Weekday', columns_to_keep)
            spring_weekdays = filter_and_select_columns(HP_Plus_AMI_Spring_with_EV_weekdays, 'Weekday', columns_to_keep)
            summer_weekdays = filter_and_select_columns(HP_Plus_AMI_Summer_with_EV_weekdays, 'Weekday', columns_to_keep)
            fall_weekdays = filter_and_select_columns(HP_Plus_AMI_Fall_with_EV_weekdays, 'Weekday', columns_to_keep)
            winter2_weekdays = filter_and_select_columns(HP_Plus_AMI_W2_with_EV_weekdays, 'Weekday', columns_to_keep)
            
            # Combine all the filtered DataFrames into a single DataFrame
            Customer_Total_Usage = pd.concat([winter1_weekends, spring_weekends, summer_weekends, fall_weekends, winter2_weekends,
                                              winter1_weekdays, spring_weekdays, summer_weekdays, fall_weekdays, winter2_weekdays], 
                                              ignore_index=True)
                
            
            # Rename the 5th column in Customer_Total_Usage with the name of the customer from the 5th column of Customer_Profile_AMI
            def rename_customer_column_from_profile(usage_df, profile_df):
                # Get the name of the 5th column from Customer_Profile_AMI
                customer_name = profile_df.columns[4]
                
                # Rename the 5th column (0-indexed: index 4) in Customer_Total_Usage
                usage_df.rename(columns={usage_df.columns[4]: customer_name}, inplace=True)
                
                return usage_df
            
            # Assuming Customer_Profile_AMI and Customer_Total_Usage are already defined
            Customer_Total_Usage = rename_customer_column_from_profile(Customer_Total_Usage, Customer_Profile_AMI)
            
            
            # Extract the customer name from the 5th column of Customer_Profile_AMI
            customer_name = Customer_Profile_AMI.columns[4]
            
            def replace_customer_profile(ami_df, total_usage_df, customer_name):
                # Check if the customer column exists in both DataFrames
                if customer_name in ami_df.columns and customer_name in total_usage_df.columns:
                    # Replace the original customer profile in AMI_Hourly with the new aggregated profile
                    ami_df[customer_name] = total_usage_df[customer_name].values
                else:
                    print(f"Customer '{customer_name}' not found in one of the DataFrames.")
                
                return ami_df
            
            # Replace the original customer profile with the new aggregated profile
            Final_Aggregated_Data = replace_customer_profile(AMI_Hourly, Customer_Total_Usage, customer_name)
            
            
        print(f"Aggregated Data with EVPenLevel {Pen_Level_EV_percentage} and HPPenLevel {Pen_Level_HP_percentage} Generated")
        
        #%% Define the output file path with the penetration level in the file name
        output_file_path = f"{self.outfolder}/Final Aggregated Data_EVPenLevel_{Pen_Level_EV_percentage} and HPPenLevel_{Pen_Level_HP_percentage}.xlsx"
        
        # Use pandas to export the DataFrame to an Excel file
        Final_Aggregated_Data.to_excel(output_file_path, index=False)
        
        print(f"Data exported successfully to {output_file_path}")

        #%% Overloading evaluation
        
        # Load both files
        # transformer_file_path = 'transformer_customer_info.xlsx'
        # ami_file_path = f"output\Final Aggregated Data_EVPenLevel_{Pen_Level_EV} and HPPenLevel_{Pen_Level_HP}.xlsx"
        
        # Read the transformer and customer file
        transformer_data = transformed_tc_data
        # ami_data = pd.read_excel(ami_file_path)
        ami_data = Final_Aggregated_Data
        
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
        
        
        
        
        # Function to delay timestamp by 1 hour, handling extra time formatting
        def delay_by_one_hour(date_str):
            try:
                # Extract date and hour, ensuring proper format
                date_part, hour_part = date_str.split(" Hour: ")
        
        
                date_cleaned = date_part.split()[0]  
        
        
                hour_cleaned = ''.join(filter(str.isdigit, hour_part))
        
                # Convert to datetime object
                datetime_obj = datetime.strptime(f"{date_cleaned} {hour_cleaned}", "%Y-%m-%d %H")
        
                # Add one hour
                adjusted_datetime = datetime_obj + timedelta(hours=1)
        
                # Return the new timestamp in the original format (YYYY-MM-DD Hour: HH)
                return f"{adjusted_datetime.strftime('%Y-%m-%d')} Hour: {adjusted_datetime.hour}"
        
            except Exception as e:
                print(f"Error processing timestamp: {date_str}. Exception: {e}")
                return date_str  # Return original string if there's an error
        
        
        # Step 1: Find the maximum load and corresponding date (month, day, and hour) for each transformer
        transformer_capacity = transformer_data.set_index('Transformer')['Transformer Rating (kW)']
        max_load_info = []
        for transformer in transformer_capacity.index:
            max_load = final_result[transformer].max()
            max_load_row = final_result[final_result[transformer] == max_load].iloc[0]
            max_load_date = delay_by_one_hour(f"{max_load_row['Date']} Hour: {max_load_row['Hour']}")  # Apply hour shift
        
            # Append the results as a dictionary
            max_load_info.append({
                "Transformer": transformer,
                "Transformer Capacity (kW)": transformer_capacity[transformer], 
                "Max Load (kW)": max_load,
                "Date of Max Load": max_load_date
            })
        
        # Convert to DataFrame
        max_load_df = pd.DataFrame(max_load_info)
        
        
        # Step 2: Count annual overload occurrences for each transformer at different thresholds
        overload_info = []
        for transformer, capacity in transformer_capacity.items():
            # Overload counts for different thresholds
            overload_100 = (final_result[transformer] > capacity).sum()  # > 100% capacity
            overload_120 = (final_result[transformer] > 1.2 * capacity).sum()  # > 120% capacity
            overload_140 = (final_result[transformer] > 1.4 * capacity).sum()  # > 140% capacity
            
            # Append the overload counts to the dictionary
            overload_info.append({
                "Transformer": transformer,
                "Transformer Capacity (kW)": capacity, 
                "Annual Overloads > 100% of capacity": overload_100,
                "Annual Overloads > 120% of capacity": overload_120,
                "Annual Overloads > 140% of capacity": overload_140
            })
        
        # Convert to DataFrame
        annual_overload_df = pd.DataFrame(overload_info)
        
        
        
        # Step 3: Breakdown by month - Calculate monthly overload occurrences at different thresholds
        monthly_overload_info = []
        for transformer, capacity in transformer_capacity.items():
            for month in range(1, 13):  # Loop over months 1 to 12
                month_data = final_result[final_result['Month'] == month]
                
                # Count monthly overloads for different thresholds
                monthly_overloads_100 = (month_data[transformer] > capacity).sum()  # > 100% capacity
                monthly_overloads_120 = (month_data[transformer] > 1.2 * capacity).sum()  # > 120% capacity
                monthly_overloads_140 = (month_data[transformer] > 1.4 * capacity).sum()  # > 140% capacity
                
                # Find the max load for this month and its date
                max_monthly_load = month_data[transformer].max()
                max_monthly_load_row = month_data[month_data[transformer] == max_monthly_load].iloc[0]
                max_monthly_load_date = delay_by_one_hour(f"{max_monthly_load_row['Date']} Hour: {max_monthly_load_row['Hour']}")  # Apply hour shift
        
                # Append the results
                monthly_overload_info.append({
                    "Transformer": transformer,
                    "Transformer Capacity (kW)": capacity, 
                    "Month": month,
                    "Monthly Overloads > 100% of capacity": monthly_overloads_100,
                    "Monthly Overloads > 120% of capacity": monthly_overloads_120,
                    "Monthly Overloads > 140% of capacity": monthly_overloads_140,
                    "Max Monthly Load (kW)": max_monthly_load,
                    "Date of Max Monthly Load": max_monthly_load_date
                })
        
        # Convert to DataFrame
        monthly_overload_df = pd.DataFrame(monthly_overload_info)
        
        
        
        # Save all data into one Excel file with multiple sheets
        merged_output_file = f'{self.outfolder}/Transformer_Load_Analysis_Results_pen_level_{Pen_Level_EV_percentage} and {Pen_Level_HP_percentage}.xlsx'
        
        with pd.ExcelWriter(merged_output_file) as writer:
            max_load_df.to_excel(writer, sheet_name='Max Load per Transformer', index=False)
            annual_overload_df.to_excel(writer, sheet_name='Annual Overloads', index=False)
            monthly_overload_df.to_excel(writer, sheet_name='Monthly Overloads Breakdown', index=False)
            
            
        #import matplotlib.pyplot as plt
        
        # Group the monthly overloading data by transformer and month for visualization
        monthly_overload_pivot_100 = monthly_overload_df.pivot(index='Month', columns='Transformer', values='Monthly Overloads > 100% of capacity')
        monthly_overload_pivot_120 = monthly_overload_df.pivot(index='Month', columns='Transformer', values='Monthly Overloads > 120% of capacity')
        monthly_overload_pivot_140 = monthly_overload_df.pivot(index='Month', columns='Transformer', values='Monthly Overloads > 140% of capacity')
        
        
        # Plot the data
        plt.figure(figsize=(25, 20))
        
        ## Plot for Monthly Overloads > 100%
        plt.figure(figsize=(12, 6))
        monthly_overload_pivot_100.plot(kind='line', marker='o')
        plt.title('Monthly Overloading Count > 100% for Each Transformer')
        plt.xlabel('Month')
        plt.ylabel('Overloading Count')
        plt.grid(True)
        plt.legend(title="Transformer", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        
        # Plot for Monthly Overloads > 120%
        plt.figure(figsize=(12, 6))
        monthly_overload_pivot_120.plot(kind='line', marker='o')
        plt.title('Monthly Overloading Count > 120% for Each Transformer')
        plt.xlabel('Month')
        plt.ylabel('Overloading Count')
        plt.grid(True)
        plt.legend(title="Transformer", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        
        # Plot for Monthly Overloads > 140%
        plt.figure(figsize=(12, 6))
        monthly_overload_pivot_140.plot(kind='line', marker='o')
        plt.title('Monthly Overloading Count > 140% for Each Transformer')
        plt.xlabel('Month')
        plt.ylabel('Overloading Count')
        plt.grid(True)
        plt.legend(title="Transformer", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        
        
        # Show the plot
        plt.show()
            
        print(f"Transformer Loading Results exported successfully to {merged_output_file}")
            
            
#
# EOF
