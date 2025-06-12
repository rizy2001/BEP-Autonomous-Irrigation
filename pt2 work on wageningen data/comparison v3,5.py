import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import math

# === CONFIGURATION ===
FILE_PATH = r"Data\Wageningen 2nd greenhouse challenge automatoes data.xlsx"
SHEET_NAME = "March"

def exp_decay(t, a, b, c):
    return a * np.exp(-b * t) + c

def van_genuchten(h, theta_r, theta_s, alpha, n):
    m = 1 - 1 / n
    return theta_r + (theta_s - theta_r) / ((1 + (alpha * h)**n) ** m)

def safe_standardize(series):
    std = series.std(ddof=0)
    if std != 0:
        return (series - series.mean()) / std
    else:
        return series - series.mean()

# === LOAD & CLEAN DATA ===
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df = df[['%time', 'WC_slab1', 'water_sup', 'Tot_PAR', 'Tair']].copy()
df.columns = ['timestamp', 'WC_slab1', 'water_sup', 'Tot_PAR', 'Tair']

for col in ['WC_slab1', 'water_sup', 'Tot_PAR', 'Tair']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
df.dropna(inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('timestamp', inplace=True)
df['date'] = df['timestamp'].dt.date

unique_dates = sorted(df['date'].unique())
split_index = int(len(unique_dates) * 0.8)
train_dates = unique_dates[:split_index]
test_dates = unique_dates[split_index:]

train_segments = []
test_segments = []

# === SEGMENT DRYDOWN EVENTS ACROSS DAYS (IMPROVED LOGIC) ===
df['water_sup_diff'] = df['water_sup'].diff().fillna(0)
unique_dates = sorted(df['date'].unique())

train_segments = []
test_segments = []

for i in range(len(unique_dates) - 1):
    day1 = unique_dates[i]
    day2 = unique_dates[i + 1]

    df_day1 = df[df['date'] == day1]
    df_day2 = df[df['date'] == day2]

    if df_day1.empty or df_day2.empty:
        continue

    last_water_time_day1 = df_day1[df_day1['water_sup_diff'] > 0]['timestamp'].max()
    first_water_time_day2 = df_day2[df_day2['water_sup_diff'] > 0]['timestamp'].min()

    if pd.isna(last_water_time_day1) or pd.isna(first_water_time_day2):
        continue

    segment_mask = (df['timestamp'] > last_water_time_day1) & (df['timestamp'] < first_water_time_day2)
    segment = df[segment_mask].copy()

    if segment.empty or len(segment) < 6:
        continue

    # Normalize & prepare
    segment['time'] = (segment['timestamp'] - segment['timestamp'].min()).dt.total_seconds() / 3600
    theta_min = segment['WC_slab1'].min()
    theta_max = segment['WC_slab1'].max()
    if theta_max == theta_min:
        continue

    segment['theta'] = (segment['WC_slab1'] - theta_min) / (theta_max - theta_min)
    segment['h_sim'] = np.exp(segment['time'] / segment['time'].max() * 5)
    segment['Tot_PAR'] = safe_standardize(segment['Tot_PAR'])
    segment['Tair'] = safe_standardize(segment['Tair'])

    # Use first date for train/test split
    if day1 in train_dates and day2 in train_dates:
        train_segments.append(segment)
    elif day1 in test_dates or day2 in test_dates:
        test_segments.append((day1, segment))

# Handle last day
last_day = unique_dates[-1]
df_last_day = df[df['date'] == last_day]
last_water_time_last_day = df_last_day[df_last_day['water_sup_diff'] > 0]['timestamp'].max()

if pd.notna(last_water_time_last_day):
    segment_mask = (df['timestamp'] > last_water_time_last_day)
    segment = df[segment_mask].copy()
    if not segment.empty and len(segment) >= 6:
        segment['time'] = (segment['timestamp'] - segment['timestamp'].min()).dt.total_seconds() / 3600
        theta_min = segment['WC_slab1'].min()
        theta_max = segment['WC_slab1'].max()
        if theta_max > theta_min:
            segment['theta'] = (segment['WC_slab1'] - theta_min) / (theta_max - theta_min)
            segment['h_sim'] = np.exp(segment['time'] / segment['time'].max() * 5)
            segment['Tot_PAR'] = safe_standardize(segment['Tot_PAR'])
            segment['Tair'] = safe_standardize(segment['Tair'])
            if last_day in train_dates:
                train_segments.append(segment)
            else:
                test_segments.append((last_day, segment))


# === TRAIN MODELS ON TRAINING SET ===
train_data = pd.concat(train_segments)
t_train = train_data['time'].values
theta_train = train_data['theta'].values
par_train = train_data['Tot_PAR'].values
tair_train = train_data['Tair'].values
h_train = train_data['h_sim'].values

# Train water balance model
phys_params = None
try:
    def et_rate(t, k1, k2, k3):
        return k1 * par_train + k2 * tair_train + k3

    def theta_phys_model(t, k1, k2, k3):
        et = et_rate(t, k1, k2, k3)
        cum_et = cumulative_trapezoid(et, t, initial=0)
        return theta_train[0] - cum_et

    phys_params, _ = curve_fit(theta_phys_model, t_train, theta_train, p0=[0.01, 0.01, 0.01])
except:
    pass

# Train Van Genuchten
vg_params = None
try:
    vg_params, _ = curve_fit(van_genuchten, h_train, theta_train,
                             p0=[0.05, 1.0, 0.05, 2.0],
                             bounds=([0, 0.8, 0.0001, 1.1], [0.5, 1.2, 1.0, 5.0]),
                             maxfev=10000)
except:
    pass

# Train grey-box (residual of VG + linear regression)
linreg_grey = LinearRegression()
try:
    vg_train_pred = van_genuchten(h_train, *vg_params)
    residuals = theta_train - vg_train_pred
    linreg_grey.fit(train_data[['time']], residuals)
except:
    linreg_grey = None

# Train exponential decay
exp_params = None
try:
    exp_params, _ = curve_fit(exp_decay, t_train, theta_train, p0=[1, 0.1, 0])
except:
    pass

# Train linear regression
linreg = LinearRegression()
try:
    linreg.fit(train_data[['time']], theta_train)
except:
    linreg = None

# === APPLY MODELS TO TEST SEGMENTS ===
segments_by_id = {}
results = []
event_id = 0

for date, segment in test_segments:
    t = segment['time'].values
    theta = segment['theta'].values
    segment = segment.copy()

    try:
        if phys_params is not None:
            et = phys_params[0] * segment['Tot_PAR'].values + phys_params[1] * segment['Tair'].values + phys_params[2]
            cum_et = cumulative_trapezoid(et, t, initial=0)
            segment.loc[:, 'water_balance_pred'] = theta[0] - cum_et
            r2_wb = r2_score(theta, segment['water_balance_pred'])
        else:
            r2_wb = None
    except:
        r2_wb = None

    try:
        if vg_params is not None:
            segment.loc[:, 'vg_pred'] = van_genuchten(segment['h_sim'], *vg_params)
            r2_vg = r2_score(theta, segment['vg_pred'])
            if linreg_grey is not None:
                segment.loc[:, 'greybox_pred'] = segment['vg_pred'] + linreg_grey.predict(segment[['time']])
                r2_grey = r2_score(theta, segment['greybox_pred'])
            else:
                r2_grey = None
        else:
            r2_vg = None
            r2_grey = None
    except:
        r2_vg = None
        r2_grey = None

    try:
        if exp_params is not None:
            segment.loc[:, 'exp_pred'] = exp_decay(t, *exp_params)
            r2_exp = r2_score(theta, segment['exp_pred'])
        else:
            r2_exp = None
    except:
        r2_exp = None

    try:
        if linreg is not None:
            segment.loc[:, 'lin_pred'] = linreg.predict(segment[['time']])
            r2_lin = r2_score(theta, segment['lin_pred'])
        else:
            r2_lin = None
    except:
        r2_lin = None

    segments_by_id[event_id] = segment.copy()
    results.append({
        'drydown_event_id': event_id,
        'Van Genuchten R2': r2_vg,
        'Water Balance R2': r2_wb,
        'Exponential Decay R2': r2_exp,
        'Linear Regression R2': r2_lin,
        'Grey-box Model R2': r2_grey,
        'date': date
    })
    event_id += 1

# === PLOT RESULTS ===
last_3_dates = test_dates[-3:]
segments_to_plot = [i for i, r in enumerate(results) if r['date'] in last_3_dates]
fig, axes = plt.subplots(len(segments_to_plot), 1, figsize=(10, 4 * len(segments_to_plot)))

if len(segments_to_plot) == 1:
    axes = [axes]

for ax, event_id in zip(axes, segments_to_plot):
    segment = segments_by_id[event_id]
    ax.plot(segment['time'], segment['theta'], 'o', label='Observed')
    for model in ['vg_pred', 'water_balance_pred', 'exp_pred', 'lin_pred', 'greybox_pred']:
        if model in segment:
            ax.plot(segment['time'], segment[model], label=model.replace('_', ' ').title())
    ax.set_title(f'Drydown Event {event_id} ({segment["timestamp"].dt.date.iloc[0]})')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Normalized WC')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# === SUMMARY ===
comparison_df = pd.DataFrame(results)
model_cols = ['Van Genuchten R2', 'Water Balance R2', 'Exponential Decay R2', 'Linear Regression R2', 'Grey-box Model R2']
comparison_df['Best Model'] = comparison_df[model_cols].idxmax(axis=1)
best_model_counts = comparison_df['Best Model'].value_counts()

print("Model Comparison Summary:")
print(comparison_df.sort_values('drydown_event_id').to_string(index=False))
print("\nBest Model Frequency Summary:")
print(best_model_counts.to_string())




'''PLOT THE SEGMENTS
# Combine all segments
all_segments = train_segments + [seg for _, seg in test_segments]
n_segments = len(all_segments)
segments_per_fig = 3
n_figs = math.ceil(n_segments / segments_per_fig)

for fig_num in range(n_figs):
    fig, axes = plt.subplots(segments_per_fig, 1, figsize=(12, 4 * segments_per_fig), sharex=False)

    if segments_per_fig == 1:
        axes = [axes]

    for i in range(segments_per_fig):
        seg_idx = fig_num * segments_per_fig + i
        if seg_idx >= n_segments:
            break

        segment = all_segments[seg_idx]
        ax = axes[i]

        # Main axis: theta
        ax.plot(segment['time'], segment['theta'], 'o-', label='θ (Normalized WC)', color='blue')
        ax.set_ylabel('θ')
        ax.set_xlabel('Time (h)')
        ax.set_title(f"Drydown Event {seg_idx} - {segment['timestamp'].dt.date.iloc[0]}")
        ax.grid(True)

        # Secondary axis: PAR and Tair (standardized)
        ax2 = ax.twinx()
        ax2.plot(segment['time'], segment['Tot_PAR'], '--', label='Tot_PAR (standardized)', color='green', alpha=0.5)
        ax2.plot(segment['time'], segment['Tair'], ':', label='Tair (standardized)', color='red', alpha=0.5)
        ax2.set_ylabel('Standardized PAR / Tair')

        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()
'''


# === Filter to last 20% of days ===
test_dates_set = set(test_dates)
full_data_test = df[df['date'].isin(test_dates_set)].copy()
full_data_test['time_hours'] = (full_data_test['timestamp'] - full_data_test['timestamp'].min()).dt.total_seconds() / 3600

# === Plot full test data with overlayed drydown segments ===
fig, ax1 = plt.subplots(figsize=(14, 6))

# Main y-axis: water content
ax1.plot(full_data_test['time_hours'], full_data_test['WC_slab1'], '-', label='WC_slab1 (Full)', color='skyblue')
ax1.set_ylabel('Water Content (WC_slab1)', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')

# Overlay drydown segments (only from test set)
colors = plt.cm.tab10.colors
for i, (date, segment) in enumerate(test_segments):
    seg = segment.copy()
    seg_time = (seg['timestamp'] - full_data_test['timestamp'].min()).dt.total_seconds() / 3600
    ax1.plot(seg_time, seg['WC_slab1'], 'o-', color=colors[i % len(colors)], label=f'Drydown {i}', alpha=0.9)

# Second y-axis: water supplied
ax2 = ax1.twinx()
ax2.plot(full_data_test['time_hours'], full_data_test['water_sup'], '-', label='Water Supplied', color='gray', alpha=0.5)
ax2.set_ylabel('Cumulative Water Supplied', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

# Legend & labels
ax1.set_title("Test Drydown Segment Selection (Last 20% of Days)")
ax1.set_xlabel("Time (hours since start of test period)")
fig.tight_layout()

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small')

plt.grid(True)
plt.show()
