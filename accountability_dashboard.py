# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calplot
import plotly.express as px
import plotly.graph_objects as go
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from plotly.subplots import make_subplots

# ========== Page config ==========
st.set_page_config(layout="wide", page_title="Accountability Tracker")

# ========== Google Sheets auth (keeps your existing approach) ==========
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = st.secrets["google"]
credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(credentials)

sheet = client.open("Accountability Tracker").sheet1
data = sheet.get_all_records()
df = pd.DataFrame(data)

# Ensure Date column is datetime and sorted
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df = df.reset_index(drop=True)

# ========== Backfill optional columns if missing ==========
# Phase column (user should edit in Google Sheet to set actual phases)
if "Phase" not in df.columns:
    df["Phase"] = "Cut"

# Target Calories optional column (editable in sheet)
if "Target Calories" not in df.columns:
    df["Target Calories"] = np.nan

# ========== Core computed columns ==========
# NOTE: keep your existing maintenance baseline (you used 1930 before)
maintenance_base = 1930

# Deficit (existing)
df['Deficit'] = df['Calories from Exercise'] + maintenance_base - df['Calories Consumed']
df['Goal_Deficit'] = df['Deficit'] > 0
df['All_Goals_Met'] = df['Goal_Deficit'] & df['Protein > 130']

# Rolling averages
df['7Day_Rolling_Weight'] = df['Weight'].rolling(window=7, min_periods=1).mean()
df['7Day_Rolling_BF'] = df['BF%'].rolling(window=7, min_periods=1).mean()
df['7Day_Rolling_Deficit'] = df['Deficit'].rolling(window=7, min_periods=1).mean()
df['7Day_Rolling_Steps'] = df['Steps'].rolling(window=7, min_periods=1).mean()
df['7Day_Rolling_Activity_Calories'] = df['Calories from Exercise'].rolling(window=7, min_periods=1).mean()
df['7Day_Rolling_Consumed_Calories'] = df['Calories Consumed'].rolling(window=7, min_periods=1).mean()

# Rate of change (7-day rolling weight diff)
df['7Day_Rolling_Weight_Change'] = df['7Day_Rolling_Weight'].diff()

# Cumulative deficit & estimated weight lost from deficit
df['Cumulative_Deficit'] = df['Deficit'].cumsum()
df['Weight_Lost_From_Deficit'] = df['Cumulative_Deficit'] / 3500.0

# Surplus calculations (useful for maintenance/bulk tracking)
# Surplus = consumed - (maintenance_base + exercise)  => positive values are true surplus above maintenance+exercise
df['Surplus'] = df['Calories Consumed'] - (maintenance_base + df['Calories from Exercise'])
df['True_Surplus'] = df['Surplus'].clip(lower=0)
df['7Day_Rolling_Surplus'] = df['True_Surplus'].rolling(window=7, min_periods=1).mean()

# Projected / rate metrics
# weekly projected gain from true surplus (7-day sum / 3500)
df['7Day_Weekly_Projected_Gain_lbs'] = df['True_Surplus'].rolling(window=7, min_periods=1).sum() / 3500.0

# lightweight "avg weight lost per week" (historical)
# avoid dividing by zero
days = (df['Date'].max() - df['Date'].min()).days or 1
df['Avg_Weight_Lost_Per_Week'] = (df['Weight'].iloc[0] - df['7Day_Rolling_Weight'].iloc[-1]) / (days / 7.0)

# ========== Sidebar filters ==========
st.sidebar.title("📅 Filters")
start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())
df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))].copy()
if df_filtered.empty:
    st.warning("No data after applying date filters. Please widen the date range.")
    st.stop()

# ========== Planning parameters (you can adjust) ==========
weeks_bulking = st.sidebar.number_input("Weeks to bulk (projection)", min_value=1, max_value=52, value=20)
expected_gain_per_week = st.sidebar.number_input("Expected gain per week (lbs)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

# ========== Metrics section (adaptive rows) ==========
st.title("🏋️ Accountability Tracker")

# computed summary values (based on filtered window)
goal_days = df_filtered['All_Goals_Met'].sum()
total_days = len(df_filtered)
percent = (goal_days / total_days) * 100 if total_days else 0

starting_weight = df_filtered['Weight'].iloc[0]
latest_avg_weight = df_filtered['7Day_Rolling_Weight'].iloc[-1]
weight_lost_observed = starting_weight - latest_avg_weight

latest_estimated_weight_loss_from_deficit = df_filtered['Weight_Lost_From_Deficit'].iloc[-1]
latest_weekly_proj_gain = df_filtered['7Day_Weekly_Projected_Gain_lbs'].iloc[-1]

avg_calories = df_filtered['7Day_Rolling_Consumed_Calories'].iloc[-1]
avg_deficit = df_filtered['7Day_Rolling_Deficit'].iloc[-1]
avg_steps = df_filtered['7Day_Rolling_Steps'].iloc[-1]
avg_exercise_cal = df_filtered['7Day_Rolling_Activity_Calories'].iloc[-1]

# Compute streaks
def compute_streaks(series):
    max_streak = curr_streak = 0
    last_date = None
    for date, met in zip(series.index, series):
        if met:
            if last_date is None or (date - last_date).days == 1:
                curr_streak += 1
            else:
                curr_streak = 1
            max_streak = max(max_streak, curr_streak)
        else:
            curr_streak = 0
        last_date = date
    return max_streak

df_streak = df_filtered.copy().set_index('Date')
longest_streak = compute_streaks(df_streak['All_Goals_Met'])
current_streak = 0
for met in df_streak.sort_index(ascending=False)['All_Goals_Met']:
    if met:
        current_streak += 1
    else:
        break

# Projected bulk end weight
start_weight_for_projection = latest_avg_weight
projected_total_gain = weeks_bulking * expected_gain_per_week
projected_bulk_end_weight = start_weight_for_projection + projected_total_gain

# Weeks until Bulk: find first date with Phase == 'Bulk' in the full df (not filtered)
bulk_dates = df[df['Phase'].str.lower() == 'bulk']
if not bulk_dates.empty:
    bulk_start_date = bulk_dates['Date'].min()
    days_until_bulk = (bulk_start_date - df_filtered['Date'].max()).days
    weeks_until_bulk = max(days_until_bulk / 7.0, 0)
else:
    # If no 'Bulk' labeled yet, show 0 (or you might rely on your planned timeline)
    weeks_until_bulk = 0.0

# Pack metrics (adaptive rows of 4)
metrics = [
    ("✅ Days All Goals Met", f"{goal_days}/{total_days} ({percent:.1f}%)"),
    ("⚖️ Weight (Observed Loss)", f"{weight_lost_observed:.1f} lbs"),
    ("⚖️ Weight Lost (Est from Deficit)", f"{latest_estimated_weight_loss_from_deficit:.1f} lbs"),
    ("📅 RL7 Weight Lost/Week", f"{latest_weekly_proj_gain:.2f} lbs"),
    ("💪 Body Fat % Lost", f"{(df_filtered['BF%'].iloc[0] - df_filtered['7Day_Rolling_BF'].iloc[-1]):.1f}%"),
    ("🔥 RL7 Calories Consumed", f"{avg_calories:.0f} kcal"),
    ("📉 RL7 Daily Deficit", f"{avg_deficit:.0f} kcal"),
    ("👣 RL7 Daily Steps", f"{avg_steps:.0f}"),
    ("🏃 RL7 Exercise Calories", f"{avg_exercise_cal:.0f} kcal"),
    ("⏳ Weeks Until Bulk", f"{weeks_until_bulk:.1f}"),
    ("🔮 Projected Weight After Bulk", f"{projected_bulk_end_weight:.1f} lbs"),
    ("🔥 Longest Streak", f"{longest_streak} days"),
    ("🔥 Current Streak", f"{current_streak} days")
]

cols_per_row = 4
for i in range(0, len(metrics), cols_per_row):
    cols = st.columns(min(cols_per_row, len(metrics) - i))
    for col, (title, value) in zip(cols, metrics[i:i+cols_per_row]):
        col.metric(title, value)

# ========== Plots: paired plots + weight change + cumulative deficit ==========
st.subheader("📅 Metrics over time")

# Plot color palettes (dark for raw, light for rolling)
dark_colors = [
    "#1f77b4",  # weight
    "#ff7f0e",  # bf
    "#2ca02c",  # steps
    "#d62728",  # calories consumed
    "#9467bd",  # exercise cal
    "#8c564b",  # deficit
]
light_colors = [
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"
]
# extra colors
bar_color = "#e377c2"
cum_color = "#7f7f7f"

# Subplot titles (paired)
plot_titles = [
    "Weight", "BF%", "Steps",
    "Calories Consumed", "Exercise Calories", "Deficit",
    "7-Day Rolling Weight Change", "Cumulative Deficit"
]

fig = make_subplots(rows=4, cols=2, subplot_titles=plot_titles, vertical_spacing=0.08, horizontal_spacing=0.08)

paired_plots = [
    ('Weight', '7Day_Rolling_Weight'),
    ('BF%', '7Day_Rolling_BF'),
    ('Steps', '7Day_Rolling_Steps'),
    ('Calories Consumed', '7Day_Rolling_Consumed_Calories'),
    ('Calories from Exercise', '7Day_Rolling_Activity_Calories'),
    ('Deficit', '7Day_Rolling_Deficit'),
]

# Add paired traces (dark for raw, light for rolling)
for i, (raw, rolling) in enumerate(paired_plots[:6]):
    row = (i // 2) + 1
    col = (i % 2) + 1
    fig.add_trace(
        go.Scatter(
            x=df_filtered['Date'], y=df_filtered[raw],
            mode='lines', line=dict(color=dark_colors[i], width=2),
            name=raw, showlegend=False
        ),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=df_filtered['Date'], y=df_filtered[rolling],
            mode='lines', line=dict(color=light_colors[i], width=3, dash='dash'),
            name=f"{rolling} (7D)", showlegend=False
        ),
        row=row, col=col
    )

# 7-Day Rolling Weight Change as bar (dedicated row 4 col1)
fig.add_trace(
    go.Bar(
        x=df_filtered['Date'],
        y=df_filtered['7Day_Rolling_Weight_Change'],
        marker_color=bar_color,
        showlegend=False
    ),
    row=4, col=1
)
# y=0 line for the bar plot
fig.add_hline(y=0, line_dash="dash", line_color="black", row=4, col=1)

# Cumulative Deficit (row 4 col2)
fig.add_trace(
    go.Scatter(
        x=df_filtered['Date'], y=df_filtered['Cumulative_Deficit'],
        mode='lines', line=dict(color=cum_color, width=2),
        showlegend=False
    ),
    row=4, col=2
)

# Add projected bulk weight line to Weight subplot (row=1,col=1)
# Build projection dates starting the day after last filtered date
last_date = df_filtered['Date'].max()
proj_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=weeks_bulking + 1, freq='W')  # weekly points
proj_weights = [start_weight_for_projection + expected_gain_per_week * i for i in range(len(proj_dates))]

fig.add_trace(
    go.Scatter(
        x=proj_dates, y=proj_weights,
        mode='lines', line=dict(color='black', width=2, dash='dot'),
        showlegend=False
    ),
    row=1, col=1
)

# ========== Phase background shading (based on df_filtered Phase values) ==========
phase_colors = {
    "cut": "rgba(255,150,150,0.12)",
    "maintenance": "rgba(200,200,200,0.12)",
    "bulk": "rgba(150,255,150,0.12)"
}

# group consecutive dates by phase to add vrects
if 'Phase' in df_filtered.columns:
    df_phase = df_filtered[['Date', 'Phase']].copy()
    # Normalize phase text
    df_phase['Phase_norm'] = df_phase['Phase'].str.lower().fillna('cut')
    # find contiguous segments
    df_phase['segment'] = (df_phase['Phase_norm'] != df_phase['Phase_norm'].shift(1)).cumsum()
    segments = df_phase.groupby('segment').agg(start_date=('Date','min'), end_date=('Date','max'), phase=('Phase_norm','first'))
    for _, seg in segments.iterrows():
        color = phase_colors.get(seg['phase'], "rgba(200,200,200,0.08)")
        fig.add_vrect(
            x0=seg['start_date'],
            x1=seg['end_date'],
            fillcolor=color,
            opacity=0.25,
            line_width=0,
            layer="below"
        )

# ========== Layout tweaks ==========
fig.update_layout(
    height=1400,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black', size=12),
    showlegend=False,
    margin=dict(l=20, r=20, t=40, b=20)
)

# show y-axis ticks and make them black
fig.update_yaxes(showticklabels=True, tickfont=dict(color='black'), zeroline=False)
fig.update_xaxes(showticklabels=False, zeroline=False)

# remove gridlines
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

# Render chart
st.plotly_chart(fig, use_container_width=True)

# ========== Optionally show raw data preview and instructions ==========
with st.expander("Debug / Raw data (filtered)"):
    st.dataframe(df_filtered.tail(50))

st.markdown(
    """
    **How to prepare for the bulk**  
    1. Add `Phase` and `Target Calories` columns to your Google Sheet for specific dates.  
    2. During maintenance, keep `Calories Consumed` near your target calories and set the Phase to "Maintenance".  
    3. During bulk, set Phase to "Bulk" and aim for the desired weekly surplus (this app shows `7Day_Weekly_Projected_Gain_lbs`).  
    4. Use the sidebar inputs to tweak `weeks to bulk` and `expected gain per week` to update projections.
    """
)
