import pandas as pd
import matplotlib.pyplot as plt

# Load climate data
climate_df = pd.read_excel('Data/ClimateTimeSeries.xlsx', sheet_name='weather_climate', engine='openpyxl')
climate_df['date'] = pd.to_datetime(climate_df['date'])

# Load crop measurements data
crop_df = pd.read_excel('Data/CropMeasurements.xlsx', sheet_name='All data', engine='openpyxl')
crop_df['date'] = pd.to_datetime(crop_df['date'])

# Merge datasets on date (and time if available and relevant)
merged_df = pd.merge(crop_df, climate_df, on='date', how='inner')

# Example: Scatter plot to visualize relationship between t_air and crop growth (e.g., height)
plt.figure(figsize=(8, 6))
plt.scatter(merged_df['t_air'], merged_df['Plantheigth'])  # Replace 'crop_height' with actual column name
plt.xlabel('Air Temperature (t_air)')
plt.ylabel('Plantheigth')
plt.title('Relationship between Air Temperature and Crop Height')
plt.grid(True)
plt.show()

# Example: Correlation matrix to see relationships between variables
print(merged_df[['t_air', 'co2', 'rh', 'Plantheigth']].corr())  # Add/remove columns as needed

# You can repeat similar plots for CO2, RH, or other variables vs crop growth metrics