import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams.update({'font.size': 14})

# Load your Excel or CSV file
df = pd.read_excel('Master Data - met sensoren_nieuwe code.xlsx')

# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Define dry weights and max water contents for each LoadCell

dry_weights = {
    'LoadCell_1': 75,
    'LoadCell_2': 60,
    'LoadCell_3': 85,
    'LoadCell_4': 80,
    'LoadCell_5': 85,
    'LoadCell_6': 75,
    'LoadCell_7': 85,
    'LoadCell_8': 80
}

max_water_contents = {
    'LoadCell_1': 585,
    'LoadCell_2': 585,
    'LoadCell_3': 585,
    'LoadCell_4': 585,
    'LoadCell_5': 585,
    'LoadCell_6': 585,
    'LoadCell_7': 585,
    'LoadCell_8': 585
}

"""
# --- IQR Filtering ---
loadcell_cols = [f'LoadCell_{i}' for i in range(1, 9)]
Q1 = df[loadcell_cols].quantile(0.25)
Q3 = df[loadcell_cols].quantile(0.75)
IQR = Q3 - Q1

# Keep rows where all LoadCell values are within [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
condition = ~((df[loadcell_cols] < (Q1 - 1.5 * IQR)) | (df[loadcell_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
df = df[condition]
"""

valid_range = (400, 800)
for col in [f'LoadCell_{i}' for i in range(1, 9)]:
    df = df[(df[col] >= valid_range[0]) & (df[col] <= valid_range[1])]

# Calculate moisture content for each plant
for lc in dry_weights.keys():
    df[f'{lc}_moisture'] = (df[lc] - dry_weights[lc]) / max_water_contents[lc] * 100

# Plot moisture content over time
plt.figure(figsize=(12, 6))
for lc in dry_weights.keys():
    plt.plot(df['Timestamp'], df[f'{lc}_moisture'], label=lc)

# Format x-axis to show Hour Day-Month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M %d-%b'))
plt.gcf().autofmt_xdate()

plt.xlabel('Time [h D-M]')
plt.ylabel('Moisture Content [%]')
plt.title('Moisture Content of 8 Plants over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
