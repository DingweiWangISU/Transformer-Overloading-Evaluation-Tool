from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd
import sklearn
import random


AMI_Hourly= pd.read_excel("AMI_Data.xlsx")

HP_Fall= pd.read_excel("Heat pump data/fall_centroid.xlsx")
HP_Winter= pd.read_excel("Heat pump data/winter_centroid.xlsx")
HP_Summer= pd.read_excel("Heat pump data/summer_centroid.xlsx")
HP_Spring= pd.read_excel("Heat pump data/spring_centroid.xlsx")


EV_Winter_Weekend1= pd.read_excel("EV data for Weekends/Weekends_Winter_cluster_1.xlsx")
EV_Winter_Weekend2= pd.read_excel("EV data for Weekends/Weekends_Winter_cluster_2.xlsx")

EV_Summer_Weekend1=pd.read_excel("EV data for Weekends/Weekends_Summer_cluster_1.xlsx")
EV_Summer_Weekend2=pd.read_excel("EV data for Weekends/Weekends_Summer_cluster_2.xlsx")
EV_Summer_Weekend3=pd.read_excel("EV data for Weekends/Weekends_Summer_cluster_3.xlsx")

EV_Spring_Weekend1=pd.read_excel("EV data for Weekends/Weekends_Spring_cluster_1.xlsx")
EV_Spring_Weekend2=pd.read_excel("EV data for Weekends/Weekends_Spring_cluster_2.xlsx")

EV_Fall_Weekend1=pd.read_excel("EV data for Weekends/Weekends_Fall_cluster_1.xlsx")
EV_Fall_Weekend2=pd.read_excel("EV data for Weekends/Weekends_Fall_cluster_2.xlsx")


EV_Winter_Weekdays1=pd.read_excel("EV data for Weekdays/Weekdays_Winter_cluster_1.xlsx")
EV_Winter_Weekdays2=pd.read_excel("EV data for Weekdays/Weekdays_Winter_cluster_2.xlsx")
EV_Winter_Weekdays3=pd.read_excel("EV data for Weekdays/Weekdays_Winter_cluster_3.xlsx")
EV_Winter_Weekdays4=pd.read_excel("EV data for Weekdays/Weekdays_Winter_cluster_4.xlsx")

EV_Summer_Weekdays1=pd.read_excel("EV data for Weekdays/Weekdays_Summer_cluster_1.xlsx")
EV_Summer_Weekdays2=pd.read_excel("EV data for Weekdays/Weekdays_Summer_cluster_2.xlsx")
EV_Summer_Weekdays3=pd.read_excel("EV data for Weekdays/Weekdays_Summer_cluster_3.xlsx")

EV_Spring_Weekdays1=pd.read_excel("EV data for Weekdays/Weekdays_Spring_cluster_1.xlsx")
EV_Spring_Weekdays2=pd.read_excel("EV data for Weekdays/Weekdays_Spring_cluster_2.xlsx")
EV_Spring_Weekdays3=pd.read_excel("EV data for Weekdays/Weekdays_Spring_cluster_3.xlsx")

EV_Fall_Weekdays1=pd.read_excel("EV data for Weekdays/Weekdays_Fall_cluster_1.xlsx")
EV_Fall_Weekdays2=pd.read_excel("EV data for Weekdays/Weekdays_Fall_cluster_2.xlsx")



# Define Penetration Level (number of customers to apply EV and HP profiles to)
Pen_Level_EV = 15  # Adjust this value as needed
Pen_Level_HP = 5


# DataFrame to store the final results (copy of the original to preserve unchanged data)
AMI_Total = AMI_Hourly.copy()

# Randomly select customers based on the penetration level
selected_customers_EV = random.sample(list(AMI_Hourly.columns[4:]), Pen_Level_EV)
selected_customers_HP = random.sample(list(AMI_Hourly.columns[4:]), Pen_Level_HP)







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
        'Fall': [0.5, 0.5],
        'Winter': [0.25, 0.25, 0.25, 0.25],
        'Spring': [0.4, 0.6],
        'Summer': [0.7, 0.3]
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
    
    
    print("Customer_HP_Fall:")
    print(Customer_HP_Fall, "\n")
    
    print("Customer_HP_Winter:")
    print(Customer_HP_Winter, "\n")
    
    print("Customer_HP_Spring:")
    print(Customer_HP_Spring, "\n")
    
    print("Customer_HP_Summer:")
    print(Customer_HP_Summer, "\n")
    
    
    
    # Function to select a sheet based on manually entered probabilities and print the selected sheet
    def select_and_sample_with_named_dataframe(prob_1FE, prob_2FE):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Fall_Weekend1', 'EV_Fall_Weekend2'], p=[prob_1FE, prob_2FE])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    
    # Example usage with probabilities for each sheet
    prob_1FE = 0.6  # Probability for selecting EV_Fall_Weekend1
    prob_2FE = 0.4  # Probability for selecting EV_Fall_Weekend2
    Customer_Ev_Fall_Weekend = select_and_sample_with_named_dataframe(prob_1FE, prob_2FE)
    
    
    # Function to select a sheet based on manually entered probabilities and print the selected sheet
    def select_and_sample_with_named_dataframe(prob_1SPE, prob_2SPE):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Spring_Weekend1', 'EV_Spring_Weekend2'], p=[prob_1SPE, prob_2SPE])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    
    # Example usage with probabilities for each sheet
    prob_1SPE = 0.6  # Probability for selecting EV_Fall_Weekend1
    prob_2SPE = 0.4  # Probability for selecting EV_Fall_Weekend2
    Customer_Ev_Spring_Weekend = select_and_sample_with_named_dataframe(prob_1SPE, prob_2SPE)
    
    
    
    # Function to select a sheet based on manually entered probabilities and print the selected sheet
    def select_and_sample_with_named_dataframe(prob_1WE, prob_2WE):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Winter_Weekend1', 'EV_Winter_Weekend2'], p=[prob_1WE, prob_2WE])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    
    # Example usage with probabilities for each sheet
    prob_1WE = 0.6  # Probability for selecting EV_Fall_Weekend1
    prob_2WE = 0.4  # Probability for selecting EV_Fall_Weekend2
    Customer_Ev_Winter_Weekend = select_and_sample_with_named_dataframe(prob_1WE, prob_2WE)
    
    
    
    def select_and_sample_with_named_dataframe(prob_1SUE, prob_2SUE, prob_3SUE):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Summer_Weekend1', 'EV_Summer_Weekend2', 'EV_Summer_Weekend3'], 
                                        p=[prob_1SUE, prob_2SUE, prob_3SUE])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    
    # Example usage with probabilities for each sheet
    prob_1SUE = 0.4  # Probability for selecting EV_Summer_Weekend1
    prob_2SUE = 0.3  # Probability for selecting EV_Summer_Weekend2
    prob_3SUE = 0.3  # Probability for selecting EV_Summer_Weekend3
    Customer_Ev_Summer_Weekend = select_and_sample_with_named_dataframe(prob_1SUE, prob_2SUE, prob_3SUE)
    
    
    
    def select_and_sample_with_named_dataframe(prob_1SUD, prob_2SUD, prob_3SUD):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Summer_Weekdays1', 'EV_Summer_Weekdays2', 'EV_Summer_Weekdays3'], 
                                        p=[prob_1SUD, prob_2SUD, prob_3SUD])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    prob_1SUD = 0.4  
    prob_2SUD= 0.3  
    prob_3SUD = 0.3  
    Customer_Ev_Summer_Weekday = select_and_sample_with_named_dataframe(prob_1SUD, prob_2SUD, prob_3SUD)
    
    
    
    def select_and_sample_with_named_dataframe(prob_1SPD, prob_2SPD, prob_3SPD):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Spring_Weekdays1', 'EV_Spring_Weekdays2', 'EV_Spring_Weekdays3'], 
                                        p=[prob_1SPD, prob_2SPD, prob_3SPD])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    
    # Example usage with probabilities for each sheet
    prob_1SPD = 0.4  
    prob_2SPD = 0.3  
    prob_3SPD = 0.3  
    Customer_Ev_Spring_Weekday = select_and_sample_with_named_dataframe(prob_1SPD, prob_2SPD, prob_3SPD)
    
    
    
    def select_and_sample_with_named_dataframe(prob_1FD, prob_2FD):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Fall_Weekdays1', 'EV_Fall_Weekdays2'], 
                                        p=[prob_1FD, prob_2FD])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    prob_1FD = 0.5  
    prob_2FD = 0.5  
    Customer_Ev_Fall_Weekday = select_and_sample_with_named_dataframe(prob_1FD, prob_2FD)
    
    
    
    def select_and_sample_with_named_dataframe(prob_1WD, prob_2WD, prob_3WD, prob_4WD):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Winter_Weekdays1', 'EV_Winter_Weekdays2', 'EV_Winter_Weekdays3', 'EV_Winter_Weekdays4'], 
                                        p=[prob_1WD, prob_2WD, prob_3WD, prob_4WD])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    
    # Example usage with probabilities for each sheet
    prob_1WD = 0.25  
    prob_2WD = 0.25  
    prob_3WD = 0.25  
    prob_4WD = 0.25  
    Customer_Ev_Winter_Weekday = select_and_sample_with_named_dataframe(prob_1WD, prob_2WD, prob_3WD, prob_4WD)
    
    
    
    
    # Assuming Customer_Profile_AMI is already defined in the script
    # Slice the DataFrame into the specified seasonal ranges
    AMI_Winter1 = Customer_Profile_AMI.iloc[0:1416].copy()  # Rows 0 to 1415 (Winter 1)
    AMI_Spring = Customer_Profile_AMI.iloc[1416:3624].copy()  # Rows 1416 to 3623 (Spring)
    AMI_Summer = Customer_Profile_AMI.iloc[3624:5832].copy()  # Rows 3624 to 5831 (Summer)
    AMI_Fall = Customer_Profile_AMI.iloc[5832:8016].copy()  # Rows 5832 to 8015 (Fall)
    AMI_Winter2 = Customer_Profile_AMI.iloc[8016:8760].copy()  # Rows 8016 to 8759 (Winter 2)
    
    # Display the shapes of the newly created DataFrames to verify
    print(f'AMI_Winter1 shape: {AMI_Winter1.shape}')
    print(f'AMI_Spring shape: {AMI_Spring.shape}')
    print(f'AMI_Summer shape: {AMI_Summer.shape}')
    print(f'AMI_Fall shape: {AMI_Fall.shape}')
    print(f'AMI_Winter2 shape: {AMI_Winter2.shape}')
    
    

    
    # Create a new DataFrame by repeating Customer_HP_Winter 59 times
    def create_repeated_customer_hp_winter(customer_hp_df, repeat_times):
        # Repeat the Customer_HP_Winter data 'repeat_times' times
        repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
        
        return repeated_customer_hp
    
    # Create a new DataFrame where Customer_HP_Winter is repeated 59 times
    Customer_W1 = create_repeated_customer_hp_winter(Customer_HP_Winter, 59)
    
    # Display the shape of the new DataFrame to verify the operation
    print(f'Customer_W1 shape: {Customer_W1.shape}')
    
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
    
    # Display the first few rows of the new DataFrame to verify the operation
    print(HP_Plus_AMI_W1.head())
    
    
    def create_repeated_customer_hp_spring(customer_hp_df, repeat_times):
        # Repeat the Customer_HP_Spring data 'repeat_times' times
        repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
        return repeated_customer_hp
    
    # Assuming Customer_HP_Spring is already defined
    Customer_Spring = create_repeated_customer_hp_spring(Customer_HP_Spring, 92)
    
    # Display the shape and first few rows of the new DataFrame to verify the operation
    print(f'Customer_Spring shape: {Customer_Spring.shape}')
    print(Customer_Spring.head())
    
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
    
    # Display the first few rows of the new DataFrame to verify the operation
    print(f'HP_Plus_AMI_Spring shape: {HP_Plus_AMI_Spring.shape}')
    print(HP_Plus_AMI_Spring.head())
    
    
    
    # Create a new DataFrame by repeating Customer_HP_Summer 92 times
    def create_repeated_customer_hp_summer(customer_hp_df, repeat_times):
        # Repeat the Customer_HP_Summer data 'repeat_times' times
        repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
        return repeated_customer_hp
    
    # Assuming Customer_HP_Summer is already defined
    Customer_Summer = create_repeated_customer_hp_summer(Customer_HP_Summer, 92)
    
    # Display the shape and first few rows of the new DataFrame to verify the operation
    print(f'Customer_Summer shape: {Customer_Summer.shape}')
    print(Customer_Summer.head())
    
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
    
    # Display the shape and first few rows of the new DataFrame to verify the operation
    print(f'HP_Plus_AMI_Summer shape: {HP_Plus_AMI_Summer.shape}')
    print(HP_Plus_AMI_Summer.head())
    
    
    # Create a new DataFrame by repeating Customer_HP_Fall 91 times
    def create_repeated_customer_hp_fall(customer_hp_df, repeat_times):
        # Repeat the Customer_HP_Fall data 'repeat_times' times
        repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
        return repeated_customer_hp
    
    # Assuming Customer_HP_Fall is already defined
    Customer_Fall = create_repeated_customer_hp_fall(Customer_HP_Fall, 91)
    
    # Display the shape and first few rows of the new DataFrame to verify the operation
    print(f'Customer_Fall shape: {Customer_Fall.shape}')
    print(Customer_Fall.head())
    
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
    
    # Display the shape and first few rows of the new DataFrame to verify the operation
    print(f'HP_Plus_AMI_Fall shape: {HP_Plus_AMI_Fall.shape}')
    print(HP_Plus_AMI_Fall.head())
    
    
    
    # Create a new DataFrame by repeating Customer_HP_Winter 31 times
    def create_repeated_customer_hp_winter(customer_hp_df, repeat_times):
        # Repeat the Customer_HP_Winter data 'repeat_times' times
        repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
        return repeated_customer_hp
    
    # Assuming Customer_HP_Winter is already defined
    Customer_W2 = create_repeated_customer_hp_winter(Customer_HP_Winter, 31)
    
    # Display the shape and first few rows of the new DataFrame to verify the operation
    print(f'Customer_W2 shape: {Customer_W2.shape}')
    print(Customer_W2.head())
    
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
    
    # Display the shape and first few rows of the new DataFrame to verify the operation
    print(f'HP_Plus_AMI_W2 shape: {HP_Plus_AMI_W2.shape}')
    print(HP_Plus_AMI_W2.head())
    
    
    # Function to add a datetime column to represent each hour starting from January 1st, as the first column
    def add_datetime_column_first(dataframe, start_date='2023-01-01'):
        # Generate a date range starting from 'start_date', with hourly frequency, matching the length of the DataFrame
        date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
        
        # Insert this date range as the first column in the DataFrame
        dataframe.insert(0, 'Datetime', date_range)
        
        return dataframe
    
    # Add a datetime column to HP_Plus_AMI_W1 starting from January 1st, 2023, as the first column
    HP_Plus_AMI_W1 = add_datetime_column_first(HP_Plus_AMI_W1)
    
    # Display the first few rows of the DataFrame to verify the operation
    print(HP_Plus_AMI_W1.head())
    
    def add_datetime_column_first_spring(dataframe, start_date='2023-03-01'):
        # Generate a date range starting from 'start_date', with hourly frequency, matching the length of the DataFrame
        date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
        
        # Insert this date range as the first column in the DataFrame
        dataframe.insert(0, 'Datetime', date_range)
        
        return dataframe
    
    # Add a datetime column to HP_Plus_AMI_Spring starting from March 1st, 2023, as the first column
    HP_Plus_AMI_Spring = add_datetime_column_first_spring(HP_Plus_AMI_Spring)
    
    # Display the first few rows of the DataFrame to verify the operation
    print(HP_Plus_AMI_Spring.head())
    
    # Function to add a datetime column to represent each hour starting from June 1st, 2023, as the first column
    def add_datetime_column_first_summer(dataframe, start_date='2023-06-01'):
        # Generate a date range starting from 'start_date', with hourly frequency, matching the length of the DataFrame
        date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
        
        # Insert this date range as the first column in the DataFrame
        dataframe.insert(0, 'Datetime', date_range)
        
        return dataframe
    
    # Add a datetime column to HP_Plus_AMI_Summer starting from June 1st, 2023, as the first column
    HP_Plus_AMI_Summer = add_datetime_column_first_summer(HP_Plus_AMI_Summer)
    
    # Display the first few rows of the DataFrame to verify the operation
    print(HP_Plus_AMI_Summer.head())
    
    
    # Function to add a datetime column to represent each hour starting from August 1st, 2023, as the first column
    def add_datetime_column_first_fall(dataframe, start_date='2023-08-01'):
        # Generate a date range starting from 'start_date', with hourly frequency, matching the length of the DataFrame
        date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
        
        # Insert this date range as the first column in the DataFrame
        dataframe.insert(0, 'Datetime', date_range)
        
        return dataframe
    
    # Add a datetime column to HP_Plus_AMI_Fall starting from August 1st, 2023, as the first column
    HP_Plus_AMI_Fall = add_datetime_column_first_fall(HP_Plus_AMI_Fall)
    
    # Display the first few rows of the DataFrame to verify the operation
    print(HP_Plus_AMI_Fall.head())
    
    
    # Function to add a datetime column to represent each hour starting from December 1st, 2023, as the first column
    def add_datetime_column_first_winter2(dataframe, start_date='2023-12-01'):
        # Generate a date range starting from 'start_date', with hourly frequency, matching the length of the DataFrame
        date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
        
        # Insert this date range as the first column in the DataFrame
        dataframe.insert(0, 'Datetime', date_range)
        
        return dataframe
    
    # Add a datetime column to HP_Plus_AMI_W2 starting from December 1st, 2023, as the first column
    HP_Plus_AMI_W2 = add_datetime_column_first_winter2(HP_Plus_AMI_W2)
    
    # Display the first few rows of the DataFrame to verify the operation
    print(HP_Plus_AMI_W2.head())
    
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
    
    # Display the first few rows of each DataFrame to verify the operation
    print("Winter 1:")
    print(HP_Plus_AMI_W1.head())
    print("\nSpring:")
    print(HP_Plus_AMI_Spring.head())
    print("\nSummer:")
    print(HP_Plus_AMI_Summer.head())
    print("\nFall:")
    print(HP_Plus_AMI_Fall.head())
    print("\nWinter 2:")
    print(HP_Plus_AMI_W2.head())
    
    
    
    
    
    
    
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
    
    # Display the first few rows of the new DataFrame to verify the operation
    print(HP_Plus_AMI_W1_with_EV.head())
    
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
    
    # Display the first few rows of the new DataFrame to verify the operation
    print(HP_Plus_AMI_W1_with_EV_weekends.head())
    
    
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
    
    # Display the first few rows of the new DataFrames to verify the operation
    print("Spring - Weekdays:")
    print(HP_Plus_AMI_Spring_with_EV_weekdays.head())
    print("\nSpring - Weekends:")
    print(HP_Plus_AMI_Spring_with_EV_weekends.head())
    
    print("\nSummer - Weekdays:")
    print(HP_Plus_AMI_Summer_with_EV_weekdays.head())
    print("\nSummer - Weekends:")
    print(HP_Plus_AMI_Summer_with_EV_weekends.head())
    
    print("\nFall - Weekdays:")
    print(HP_Plus_AMI_Fall_with_EV_weekdays.head())
    print("\nFall - Weekends:")
    print(HP_Plus_AMI_Fall_with_EV_weekends.head())
    
    print("\nWinter 2 - Weekdays:")
    print(HP_Plus_AMI_W2_with_EV_weekdays.head())
    print("\nWinter 2 - Weekends:")
    print(HP_Plus_AMI_W2_with_EV_weekends.head())
    
    
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
    
    # Display the first few rows of the new DataFrame to verify the operation
    print(f'Customer_Total_Usage shape: {Customer_Total_Usage.shape}')
    print(Customer_Total_Usage.head())
    
    
    # Rename the 5th column in Customer_Total_Usage with the name of the customer from the 5th column of Customer_Profile_AMI
    def rename_customer_column_from_profile(usage_df, profile_df):
        # Get the name of the 5th column from Customer_Profile_AMI
        customer_name = profile_df.columns[4]
        
        # Rename the 5th column (0-indexed: index 4) in Customer_Total_Usage
        usage_df.rename(columns={usage_df.columns[4]: customer_name}, inplace=True)
        
        return usage_df
    
    # Assuming Customer_Profile_AMI and Customer_Total_Usage are already defined
    Customer_Total_Usage = rename_customer_column_from_profile(Customer_Total_Usage, Customer_Profile_AMI)
    
    # Display the first few rows of the modified DataFrame to verify the operation
    print(Customer_Total_Usage.head())
    
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
    
    # Display the updated AMI_Hourly to verify the changes
    print(AMI_PLUS_EV.head())
    


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
        'Fall': [0.5, 0.5],
        'Winter': [0.25, 0.25, 0.25, 0.25],
        'Spring': [0.4, 0.6],
        'Summer': [0.7, 0.3]
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
        
    
    print("Customer_HP_Fall:")
    print(Customer_HP_Fall, "\n")
    
    print("Customer_HP_Winter:")
    print(Customer_HP_Winter, "\n")
    
    print("Customer_HP_Spring:")
    print(Customer_HP_Spring, "\n")
    
    print("Customer_HP_Summer:")
    print(Customer_HP_Summer, "\n")
    
    
    
    # Function to select a sheet based on manually entered probabilities and print the selected sheet
    def select_and_sample_with_named_dataframe(prob_1FE, prob_2FE):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Fall_Weekend1', 'EV_Fall_Weekend2'], p=[prob_1FE, prob_2FE])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    
    # Example usage with probabilities for each sheet
    prob_1FE = 0.6  # Probability for selecting EV_Fall_Weekend1
    prob_2FE = 0.4  # Probability for selecting EV_Fall_Weekend2
    Customer_Ev_Fall_Weekend = select_and_sample_with_named_dataframe(prob_1FE, prob_2FE)
    Customer_Ev_Fall_Weekend.iloc[:, -1] = 0
    
    # Function to select a sheet based on manually entered probabilities and print the selected sheet
    def select_and_sample_with_named_dataframe(prob_1SPE, prob_2SPE):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Spring_Weekend1', 'EV_Spring_Weekend2'], p=[prob_1SPE, prob_2SPE])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    
    # Example usage with probabilities for each sheet
    prob_1SPE = 0.6  # Probability for selecting EV_Fall_Weekend1
    prob_2SPE = 0.4  # Probability for selecting EV_Fall_Weekend2
    Customer_Ev_Spring_Weekend = select_and_sample_with_named_dataframe(prob_1SPE, prob_2SPE)
    Customer_Ev_Spring_Weekend.iloc[:, -1] = 0
    
    
    # Function to select a sheet based on manually entered probabilities and print the selected sheet
    def select_and_sample_with_named_dataframe(prob_1WE, prob_2WE):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Winter_Weekend1', 'EV_Winter_Weekend2'], p=[prob_1WE, prob_2WE])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    
    # Example usage with probabilities for each sheet
    prob_1WE = 0.6  # Probability for selecting EV_Fall_Weekend1
    prob_2WE = 0.4  # Probability for selecting EV_Fall_Weekend2
    Customer_Ev_Winter_Weekend = select_and_sample_with_named_dataframe(prob_1WE, prob_2WE)
    Customer_Ev_Winter_Weekend.iloc[:, -1] = 0
    
    
    def select_and_sample_with_named_dataframe(prob_1SUE, prob_2SUE, prob_3SUE):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Summer_Weekend1', 'EV_Summer_Weekend2', 'EV_Summer_Weekend3'], 
                                        p=[prob_1SUE, prob_2SUE, prob_3SUE])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    
    # Example usage with probabilities for each sheet
    prob_1SUE = 0.4  # Probability for selecting EV_Summer_Weekend1
    prob_2SUE = 0.3  # Probability for selecting EV_Summer_Weekend2
    prob_3SUE = 0.3  # Probability for selecting EV_Summer_Weekend3
    Customer_Ev_Summer_Weekend = select_and_sample_with_named_dataframe(prob_1SUE, prob_2SUE, prob_3SUE)
    Customer_Ev_Summer_Weekend.iloc[:, -1] = 0
    
    
    def select_and_sample_with_named_dataframe(prob_1SUD, prob_2SUD, prob_3SUD):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Summer_Weekdays1', 'EV_Summer_Weekdays2', 'EV_Summer_Weekdays3'], 
                                        p=[prob_1SUD, prob_2SUD, prob_3SUD])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    prob_1SUD = 0.4  
    prob_2SUD= 0.3  
    prob_3SUD = 0.3  
    Customer_Ev_Summer_Weekday = select_and_sample_with_named_dataframe(prob_1SUD, prob_2SUD, prob_3SUD)
    Customer_Ev_Summer_Weekday.iloc[:, -1] = 0
    
    
    def select_and_sample_with_named_dataframe(prob_1SPD, prob_2SPD, prob_3SPD):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Spring_Weekdays1', 'EV_Spring_Weekdays2', 'EV_Spring_Weekdays3'], 
                                        p=[prob_1SPD, prob_2SPD, prob_3SPD])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    
    # Example usage with probabilities for each sheet
    prob_1SPD = 0.4  
    prob_2SPD = 0.3  
    prob_3SPD = 0.3  
    Customer_Ev_Spring_Weekday = select_and_sample_with_named_dataframe(prob_1SPD, prob_2SPD, prob_3SPD)
    Customer_Ev_Spring_Weekday.iloc[:, -1] = 0
    
    
    def select_and_sample_with_named_dataframe(prob_1FD, prob_2FD):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Fall_Weekdays1', 'EV_Fall_Weekdays2'], 
                                        p=[prob_1FD, prob_2FD])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    prob_1FD = 0.5  
    prob_2FD = 0.5  
    Customer_Ev_Fall_Weekday = select_and_sample_with_named_dataframe(prob_1FD, prob_2FD)
    Customer_Ev_Fall_Weekday.iloc[:, -1] = 0
    
    
    def select_and_sample_with_named_dataframe(prob_1WD, prob_2WD, prob_3WD, prob_4WD):
        # Select a sheet based on the probabilities
        sheet_choice = np.random.choice(['EV_Winter_Weekdays1', 'EV_Winter_Weekdays2', 'EV_Winter_Weekdays3', 'EV_Winter_Weekdays4'], 
                                        p=[prob_1WD, prob_2WD, prob_3WD, prob_4WD])
        
        # Print the selected spreadsheet
        print(f"Selected Spreadsheet: {sheet_choice}")
        
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
    
    # Example usage with probabilities for each sheet
    prob_1WD = 0.25  
    prob_2WD = 0.25  
    prob_3WD = 0.25  
    prob_4WD = 0.25  
    Customer_Ev_Winter_Weekday = select_and_sample_with_named_dataframe(prob_1WD, prob_2WD, prob_3WD, prob_4WD)
    Customer_Ev_Winter_Weekday.iloc[:, -1] = 0
    
    
    
    # Assuming Customer_Profile_AMI is already defined in the script
    # Slice the DataFrame into the specified seasonal ranges
    AMI_Winter1 = Customer_Profile_AMI.iloc[0:1416].copy()  # Rows 0 to 1415 (Winter 1)
    AMI_Spring = Customer_Profile_AMI.iloc[1416:3624].copy()  # Rows 1416 to 3623 (Spring)
    AMI_Summer = Customer_Profile_AMI.iloc[3624:5832].copy()  # Rows 3624 to 5831 (Summer)
    AMI_Fall = Customer_Profile_AMI.iloc[5832:8016].copy()  # Rows 5832 to 8015 (Fall)
    AMI_Winter2 = Customer_Profile_AMI.iloc[8016:8760].copy()  # Rows 8016 to 8759 (Winter 2)
    
    # Display the shapes of the newly created DataFrames to verify
    print(f'AMI_Winter1 shape: {AMI_Winter1.shape}')
    print(f'AMI_Spring shape: {AMI_Spring.shape}')
    print(f'AMI_Summer shape: {AMI_Summer.shape}')
    print(f'AMI_Fall shape: {AMI_Fall.shape}')
    print(f'AMI_Winter2 shape: {AMI_Winter2.shape}')
    
    

    
    # Create a new DataFrame by repeating Customer_HP_Winter 59 times
    def create_repeated_customer_hp_winter(customer_hp_df, repeat_times):
        # Repeat the Customer_HP_Winter data 'repeat_times' times
        repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
        
        return repeated_customer_hp
    
    # Create a new DataFrame where Customer_HP_Winter is repeated 59 times
    Customer_W1 = create_repeated_customer_hp_winter(Customer_HP_Winter, 59)
    
    # Display the shape of the new DataFrame to verify the operation
    print(f'Customer_W1 shape: {Customer_W1.shape}')
    
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
    
    # Display the first few rows of the new DataFrame to verify the operation
    print(HP_Plus_AMI_W1.head())
    
    
    def create_repeated_customer_hp_spring(customer_hp_df, repeat_times):
        # Repeat the Customer_HP_Spring data 'repeat_times' times
        repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
        return repeated_customer_hp
    
    # Assuming Customer_HP_Spring is already defined
    Customer_Spring = create_repeated_customer_hp_spring(Customer_HP_Spring, 92)
    
    # Display the shape and first few rows of the new DataFrame to verify the operation
    print(f'Customer_Spring shape: {Customer_Spring.shape}')
    print(Customer_Spring.head())
    
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
    
    # Display the first few rows of the new DataFrame to verify the operation
    print(f'HP_Plus_AMI_Spring shape: {HP_Plus_AMI_Spring.shape}')
    print(HP_Plus_AMI_Spring.head())
    
    
    
    # Create a new DataFrame by repeating Customer_HP_Summer 92 times
    def create_repeated_customer_hp_summer(customer_hp_df, repeat_times):
        # Repeat the Customer_HP_Summer data 'repeat_times' times
        repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
        return repeated_customer_hp
    
    # Assuming Customer_HP_Summer is already defined
    Customer_Summer = create_repeated_customer_hp_summer(Customer_HP_Summer, 92)
    
    # Display the shape and first few rows of the new DataFrame to verify the operation
    print(f'Customer_Summer shape: {Customer_Summer.shape}')
    print(Customer_Summer.head())
    
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
    
    # Display the shape and first few rows of the new DataFrame to verify the operation
    print(f'HP_Plus_AMI_Summer shape: {HP_Plus_AMI_Summer.shape}')
    print(HP_Plus_AMI_Summer.head())
    
    
    # Create a new DataFrame by repeating Customer_HP_Fall 91 times
    def create_repeated_customer_hp_fall(customer_hp_df, repeat_times):
        # Repeat the Customer_HP_Fall data 'repeat_times' times
        repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
        return repeated_customer_hp
    
    # Assuming Customer_HP_Fall is already defined
    Customer_Fall = create_repeated_customer_hp_fall(Customer_HP_Fall, 91)
    
    # Display the shape and first few rows of the new DataFrame to verify the operation
    print(f'Customer_Fall shape: {Customer_Fall.shape}')
    print(Customer_Fall.head())
    
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
    
    # Display the shape and first few rows of the new DataFrame to verify the operation
    print(f'HP_Plus_AMI_Fall shape: {HP_Plus_AMI_Fall.shape}')
    print(HP_Plus_AMI_Fall.head())
    
    
    
    # Create a new DataFrame by repeating Customer_HP_Winter 31 times
    def create_repeated_customer_hp_winter(customer_hp_df, repeat_times):
        # Repeat the Customer_HP_Winter data 'repeat_times' times
        repeated_customer_hp = pd.concat([customer_hp_df] * repeat_times, ignore_index=True)
        return repeated_customer_hp
    
    # Assuming Customer_HP_Winter is already defined
    Customer_W2 = create_repeated_customer_hp_winter(Customer_HP_Winter, 31)
    
    # Display the shape and first few rows of the new DataFrame to verify the operation
    print(f'Customer_W2 shape: {Customer_W2.shape}')
    print(Customer_W2.head())
    
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
    
    # Display the shape and first few rows of the new DataFrame to verify the operation
    print(f'HP_Plus_AMI_W2 shape: {HP_Plus_AMI_W2.shape}')
    print(HP_Plus_AMI_W2.head())
    
    
    # Function to add a datetime column to represent each hour starting from January 1st, as the first column
    def add_datetime_column_first(dataframe, start_date='2023-01-01'):
        # Generate a date range starting from 'start_date', with hourly frequency, matching the length of the DataFrame
        date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
        
        # Insert this date range as the first column in the DataFrame
        dataframe.insert(0, 'Datetime', date_range)
        
        return dataframe
    
    # Add a datetime column to HP_Plus_AMI_W1 starting from January 1st, 2023, as the first column
    HP_Plus_AMI_W1 = add_datetime_column_first(HP_Plus_AMI_W1)
    
    # Display the first few rows of the DataFrame to verify the operation
    print(HP_Plus_AMI_W1.head())
    
    def add_datetime_column_first_spring(dataframe, start_date='2023-03-01'):
        # Generate a date range starting from 'start_date', with hourly frequency, matching the length of the DataFrame
        date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
        
        # Insert this date range as the first column in the DataFrame
        dataframe.insert(0, 'Datetime', date_range)
        
        return dataframe
    
    # Add a datetime column to HP_Plus_AMI_Spring starting from March 1st, 2023, as the first column
    HP_Plus_AMI_Spring = add_datetime_column_first_spring(HP_Plus_AMI_Spring)
    
    # Display the first few rows of the DataFrame to verify the operation
    print(HP_Plus_AMI_Spring.head())
    
    # Function to add a datetime column to represent each hour starting from June 1st, 2023, as the first column
    def add_datetime_column_first_summer(dataframe, start_date='2023-06-01'):
        # Generate a date range starting from 'start_date', with hourly frequency, matching the length of the DataFrame
        date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
        
        # Insert this date range as the first column in the DataFrame
        dataframe.insert(0, 'Datetime', date_range)
        
        return dataframe
    
    # Add a datetime column to HP_Plus_AMI_Summer starting from June 1st, 2023, as the first column
    HP_Plus_AMI_Summer = add_datetime_column_first_summer(HP_Plus_AMI_Summer)
    
    # Display the first few rows of the DataFrame to verify the operation
    print(HP_Plus_AMI_Summer.head())
    
    
    # Function to add a datetime column to represent each hour starting from August 1st, 2023, as the first column
    def add_datetime_column_first_fall(dataframe, start_date='2023-08-01'):
        # Generate a date range starting from 'start_date', with hourly frequency, matching the length of the DataFrame
        date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
        
        # Insert this date range as the first column in the DataFrame
        dataframe.insert(0, 'Datetime', date_range)
        
        return dataframe
    
    # Add a datetime column to HP_Plus_AMI_Fall starting from August 1st, 2023, as the first column
    HP_Plus_AMI_Fall = add_datetime_column_first_fall(HP_Plus_AMI_Fall)
    
    # Display the first few rows of the DataFrame to verify the operation
    print(HP_Plus_AMI_Fall.head())
    
    
    # Function to add a datetime column to represent each hour starting from December 1st, 2023, as the first column
    def add_datetime_column_first_winter2(dataframe, start_date='2023-12-01'):
        # Generate a date range starting from 'start_date', with hourly frequency, matching the length of the DataFrame
        date_range = pd.date_range(start=start_date, periods=len(dataframe), freq='H')
        
        # Insert this date range as the first column in the DataFrame
        dataframe.insert(0, 'Datetime', date_range)
        
        return dataframe
    
    # Add a datetime column to HP_Plus_AMI_W2 starting from December 1st, 2023, as the first column
    HP_Plus_AMI_W2 = add_datetime_column_first_winter2(HP_Plus_AMI_W2)
    
    # Display the first few rows of the DataFrame to verify the operation
    print(HP_Plus_AMI_W2.head())
    
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
    
    # Display the first few rows of each DataFrame to verify the operation
    print("Winter 1:")
    print(HP_Plus_AMI_W1.head())
    print("\nSpring:")
    print(HP_Plus_AMI_Spring.head())
    print("\nSummer:")
    print(HP_Plus_AMI_Summer.head())
    print("\nFall:")
    print(HP_Plus_AMI_Fall.head())
    print("\nWinter 2:")
    print(HP_Plus_AMI_W2.head())
    
    
    
    
    
    
    
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
    
    # Display the first few rows of the new DataFrame to verify the operation
    print(HP_Plus_AMI_W1_with_EV.head())
    
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
    
    # Display the first few rows of the new DataFrame to verify the operation
    print(HP_Plus_AMI_W1_with_EV_weekends.head())
    
    
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
    
    # Display the first few rows of the new DataFrames to verify the operation
    print("Spring - Weekdays:")
    print(HP_Plus_AMI_Spring_with_EV_weekdays.head())
    print("\nSpring - Weekends:")
    print(HP_Plus_AMI_Spring_with_EV_weekends.head())
    
    print("\nSummer - Weekdays:")
    print(HP_Plus_AMI_Summer_with_EV_weekdays.head())
    print("\nSummer - Weekends:")
    print(HP_Plus_AMI_Summer_with_EV_weekends.head())
    
    print("\nFall - Weekdays:")
    print(HP_Plus_AMI_Fall_with_EV_weekdays.head())
    print("\nFall - Weekends:")
    print(HP_Plus_AMI_Fall_with_EV_weekends.head())
    
    print("\nWinter 2 - Weekdays:")
    print(HP_Plus_AMI_W2_with_EV_weekdays.head())
    print("\nWinter 2 - Weekends:")
    print(HP_Plus_AMI_W2_with_EV_weekends.head())
    
    
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
    
    # Display the first few rows of the new DataFrame to verify the operation
    print(f'Customer_Total_Usage shape: {Customer_Total_Usage.shape}')
    print(Customer_Total_Usage.head())
    
    
    # Rename the 5th column in Customer_Total_Usage with the name of the customer from the 5th column of Customer_Profile_AMI
    def rename_customer_column_from_profile(usage_df, profile_df):
        # Get the name of the 5th column from Customer_Profile_AMI
        customer_name = profile_df.columns[4]
        
        # Rename the 5th column (0-indexed: index 4) in Customer_Total_Usage
        usage_df.rename(columns={usage_df.columns[4]: customer_name}, inplace=True)
        
        return usage_df
    
    # Assuming Customer_Profile_AMI and Customer_Total_Usage are already defined
    Customer_Total_Usage = rename_customer_column_from_profile(Customer_Total_Usage, Customer_Profile_AMI)
    
    # Display the first few rows of the modified DataFrame to verify the operation
    print(Customer_Total_Usage.head())
    
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
    
    # Display the updated AMI_Hourly to verify the changes
    print(Final_Aggregated_Data.head())
    


#%% Define the output file path with the penetration level in the file name
output_file_path = f"output\Final Aggregated Data_EVPenLevel_{Pen_Level_EV} and HPPenLevel_{Pen_Level_HP}.xlsx"

# Use pandas to export the DataFrame to an Excel file
Final_Aggregated_Data.to_excel(output_file_path, index=False)

print(f"Data exported successfully to {output_file_path}")

    
    