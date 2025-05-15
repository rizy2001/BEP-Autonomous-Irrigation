import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Specify the path to your Excel file
file_path = 'Data\ClimateTimeSeries.xlsx'

# Read the Excel file into a DataFrame using the openpyxl engine
df = pd.read_excel(file_path, sheet_name='weather_climate', engine='openpyxl')

# Ensure the 'date' column is in datetime format and normalize the 'date' column to only include the date part
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.date


# Filter the DataFrame for the specific date (2023/09/05)
filtered_df = df[df['date'] == datetime(2023, 9, 5).date()]

# print the column t_air of the filtered DataFrame
print(filtered_df['t_air'])

# Plot the 't_air' values for the specific date
plt.figure(figsize=(10, 6))
plt.plot(filtered_df['time'], filtered_df['t_air'], marker='o', label='t_air')# Customize the plot
plt.title('t_air on 2023/09/05')
plt.xlabel('Time')
plt.ylabel('t_air')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()