import pandas as pd

# Specify the path to your Excel file
file_path = 'Data\Measurements From BEP Group 2021.xlsx'

# Read the Excel file into a DataFrame using the openpyxl engine
df = pd.read_excel(file_path, sheet_name='27-05-21 and onwards', engine='openpyxl')

# Display the first few rows of the DataFrame
print(df.head(50))