import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'hourly_demand_dataset.csv')
artifacts_dir = r"C:\Users\Sanket Thakore\.gemini\antigravity\brain\9f338159-df60-4b3b-a27e-ba54642f6cf6"

# Load Dataset
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)

# Set up visual style
sns.set_theme(style="whitegrid")

# 1. Plot demand vs hour (line chart)
plt.figure(figsize=(10, 6))
hourly_demand = df.groupby('Hour')['Demand_Count'].mean().reset_index()
sns.lineplot(data=hourly_demand, x='Hour', y='Demand_Count', marker='o', color='royalblue', linewidth=2)
plt.title('Average Demand vs Hour of Day', fontsize=16)
plt.xlabel('Hour of Day (0-23)', fontsize=12)
plt.ylabel('Average Hourly Demand', fontsize=12)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, 'demand_vs_hour.png'), dpi=150)
plt.close()

# 2. Identify peak hours
peak_hours = hourly_demand.sort_values(by='Demand_Count', ascending=False).head(3)
print("\n--- PEAK HOURS ---")
print(peak_hours.to_string(index=False))

# 3. Plot top 10 high demand locations
plt.figure(figsize=(12, 6))
top_locations = df.groupby('Pickup Location')['Demand_Count'].sum().reset_index()
top_locations = top_locations.sort_values(by='Demand_Count', ascending=False).head(10)
sns.barplot(data=top_locations, x='Pickup Location', y='Demand_Count', hue='Pickup Location', palette='viridis', legend=False)
plt.title('Top 10 Highest Demand Locations', fontsize=16)
plt.xlabel('Pickup Location', fontsize=12)
plt.ylabel('Total Demand (Over entire period)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, 'top_10_locations.png'), dpi=150)
plt.close()

print("\n--- TOP 5 LOCATIONS ---")
print(top_locations.head(5).to_string(index=False))

# 4. Show distribution of demand
plt.figure(figsize=(10, 6))
sns.histplot(df['Demand_Count'], bins=30, kde=True, color='mediumpurple')
plt.title('Distribution of Hourly Demand Extent per Location', fontsize=16)
plt.xlabel('Number of Requests in an Hour (Demand Magnitude)', fontsize=12)
plt.ylabel('Frequency (Log scale for visibility)', fontsize=12)
plt.yscale('log') # Use log scale because lower count frequencies are usually huge
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, 'demand_distribution.png'), dpi=150)
plt.close()

print(f"\nEDA Complete! Saved 3 plots to artifacts folder.")
