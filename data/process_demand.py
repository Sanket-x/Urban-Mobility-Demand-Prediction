import pandas as pd
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Bengaluru Ola.csv')

# 1. Load your dataset
df = pd.read_csv(csv_path)

# 2. Keep only the necessary columns to save memory and process faster
cols_to_keep = ['Date', 'Time', 'Pickup Location', 'Booking Status']
df = df[cols_to_keep]

# 3. Combine 'Date' and 'Time' into a single Datetime column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)

# 4. Extract 'hour' and 'day' features from the new Datetime column
df['Hour'] = df['Datetime'].dt.hour
df['Day_of_Week'] = df['Datetime'].dt.dayofweek
df['Date_Only'] = df['Datetime'].dt.date

# 6. Create the demand dataset by counting rides per hour per location
demand_df = (
    df.groupby(['Date_Only', 'Hour', 'Day_of_Week', 'Pickup Location'])
    .size()  # Count the number of rows in each group
    .reset_index(name='Demand_Count') # Rename the resulting count column
)

# 7. Sort the dataset chronologically and by location for better readability
demand_df = demand_df.sort_values(by=['Date_Only', 'Hour', 'Pickup Location']).reset_index(drop=True)

# Let's see the first few rows of our new demand dataset
print("First few rows of the demand dataset:")
print(demand_df.head(10))

# Optional: Save the aggregated demand dataset to a new CSV file
output_path = os.path.join(script_dir, 'hourly_demand_dataset.csv')
demand_df.to_csv(output_path, index=False)
print(f"\nSaved demand dataset to: {output_path}")
