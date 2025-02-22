# Transformer Overloading Evaluation Tool

The Transformer Overloading Evaluation Tool provides guidance for identifying transformers that would overload based on existing transformer and customer information given specified heat pump (HP) and electric vehicle (EV) pentration levels.


## Two Excel files in the “.xlsx” extension are required for usage:
    1.	AMI_Data.xlsx
    2.	transformer_customer_info.xlsx

### AMI_Data.xlsx
This is the file to store one-year hourly AMI data for all the customers in a feeder. The format of the file is as follows: 
 
**Headers**: Date, Hour, Day Type, Season, Customer X.

#### Data Format 
- Date: YYYY-MM-DD or MM/DD/YYYY
- Hour: Hour of the day from the data (0 to 23)
- Day type: Specify if the date is a weekday or weekend
- Season: season based on current date.
  - Spring – Mar, Apr, May;
  - Summer – June, July, Aug;
  - Fall – Sep, Oct, Nov;
  - Winter – Dec, Jan, Feb.
- Customer X: customer AMI data, from customer 1 to customer n.

### transformer_customer_info.xlsx
This is the file to store transformer specifications and transformer-customer connectivity in a feeder. The format of the file is as follows: 
 
**Headers**: Transformer, Transformer Rating (kVA), Customer Indexes, Transformer Rating (kW)

#### Data Format
- Transformer: transformer labels.
- Transformer Rating (kVA): KVA rating of the transformer.
- Customer Indexes: specify which customers are connected to this transformer.
- Transformer Rating (kW): transformer kVA rating converted to kW rating, you can assume a power factor.

## Algorithm Usage

First, open Customer profile Gen_EV_HP.py for aggregated load profile generation. You need to specify the penetration level of the EV and heat pump (They can be zero).
 
Note that this is the number of customers, not the percentage.
Run the rest of the code, and you will get a file in the output folder called `Final Aggregated Data_EVPenLevel_{Pen_Level_EV} and HPPenLevel_{Pen_Level_HP}.xlsx`. The value of {Pen_Level_EV} and {Pen_Level_HP} will be the same as the one you set. You may change the penetration level and generate multiple profiles.

Next, open Transformer overloading.py. Change the Penetration Level to the file that you generated in the previous step.
 
Run the rest of the code, you will get an Excel file in the output folder called `Transformer_Load_Analysis_Results_pen_level_{Pen_Level_EV} and {Pen_Level_HP}.xlsx` with your specified penetration levels. This file contains three sheets. The first one is the maximum load per transformer over a year. The second one is the count of overloads per transformer over a year. The third one is the monthly overload breakdown for each transformer.

## Installation / Deployment

This repository contains a few helper scripts for installation/deployment:
* `mkvenv.sh` will create a python virtual environment with the necessary packages.
* `python3 cli.py` can be used to run the application via the command line (no web interface needed).
* `run.sh` will start a gunicorn web server on port 8080. Once running, you can go to [http://localhost:8080/](http://localhost:8080/) to access the web interface for the application.

## Additional Documentation

* Refer to the [docs](docs) subfolder for additional documentation.
* Refer to [AUTHORS.md](AUTHORS.md) for citation information and developer contact details.

