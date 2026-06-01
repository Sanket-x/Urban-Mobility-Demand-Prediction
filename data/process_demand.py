import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Bengaluru Ola.csv')

df = pd.read_csv(csv_path)

cols_to_keep = ['Date', 'Time', 'Pickup Location', 'Booking Status']
df = df[cols_to_keep]

df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)

df['Hour'] = df['Datetime'].dt.hour
df['Day_of_Week'] = df['Datetime'].dt.dayofweek
df['Date_Only'] = df['Datetime'].dt.date

demand_df = (
    df.groupby(['Date_Only', 'Hour', 'Day_of_Week', 'Pickup Location'])
    .size()                                          
    .reset_index(name='Demand_Count')                                    
)

demand_df = demand_df.sort_values(by=['Date_Only', 'Hour', 'Pickup Location']).reset_index(drop=True)

print("First few rows of the demand dataset:")
print(demand_df.head(10))

output_path = os.path.join(script_dir, 'hourly_demand_dataset.csv')
demand_df.to_csv(output_path, index=False)
print(f"\nSaved demand dataset to: {output_path}")
