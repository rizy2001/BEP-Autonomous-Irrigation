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

# Plot t_air, rh, co2, t_rail in one window with separate subplots
fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

axs[0].plot(filtered_df['time'], filtered_df['t_air'], marker='o', label='t_air')
axs[0].set_title('t_air on 2023/09/05')
axs[0].set_ylabel('t_air')
axs[0].grid(True)

axs[1].plot(filtered_df['time'], filtered_df['rh'], marker='x', color='orange', label='rh')
axs[1].set_title('rh on 2023/09/05')
axs[1].set_ylabel('rh')
axs[1].grid(True)

axs[2].plot(filtered_df['time'], filtered_df['co2'], marker='s', color='green', label='co2')
axs[2].set_title('co2 on 2023/09/05')
axs[2].set_ylabel('co2')
axs[2].grid(True)

axs[3].plot(filtered_df['time'], filtered_df['t_rail'], marker='^', color='red', label='t_rail')
axs[3].set_title('t_rail on 2023/09/05')
axs[3].set_xlabel('Time')
axs[3].set_ylabel('t_rail')
axs[3].grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()