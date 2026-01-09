import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import calplot
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from plotly.subplots import make_subplots


# Define scope for Google Sheets and Google Drive access
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Load credentials from Streamlit secrets
creds_dict = st.secrets["google"]

# Authenticate
credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(credentials)

# Open sheet
sheet = client.open("Accountability Tracker").sheet1
data = sheet.get_all_records()
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])


# Load and preprocess
# df = pd.read_csv("Accountability.csv", parse_dates=['Date'])
df.sort_values('Date', inplace=True)

# Compute goal columns
df['Deficit'] = df['Calories from Exercise'] - df['Calories Consumed']
df['Goal_Deficit'] = df['Deficit'] > 0
# df['Goal_Steps'] = df['Steps'] > 10000
df['All_Goals_Met'] = df['Goal_Deficit'] & df['Protein > 130']

# Rolling averages
df['7Day_Rolling_Weight'] = df['Weight'].rolling(window=7, min_periods=1).mean()
df['7Day_Rolling_BF'] = df['BF%'].rolling(window=7, min_periods=1).mean()
df['7Day_Rolling_Deficit'] = df['Deficit'].rolling(window = 7, min_periods=1).mean()
df['7Day_Rolling_Steps'] = df['Steps'].rolling(window = 7, min_periods = 1).mean()
df['7Day_Rolling_Activity_Calories'] = df['Calories from Exercise'].rolling(window = 7, min_periods=1).mean()
df['7Day_Rolling_Consumed_Calories'] = df['Calories Consumed'].rolling(window = 7, min_periods=1).mean()
df['7Day_Rolling_Muscle'] = df['Muscle Mass'].rolling(window=7, min_periods=1).mean()


# Rate of change (day-to-day difference) of 7-day rolling average weight
df['7Day_Rolling_Weight_Change'] = df['7Day_Rolling_Weight'].diff()

# Cumulative deficit over time
df['Cumulative_Deficit'] = df['Deficit'].cumsum()

# weight lost from deficit
df['Weight_Lost_From_Deficit'] = df['Cumulative_Deficit'] / 3500

# average weight lost per week
df['Avg_Weight_Lost_Per_Week'] = df['Weight_Lost_From_Deficit'] / (len(df) / 7)

# 7 day rolling average of weight lost per week
df['7Day_Rolling_Avg_Weight_Lost_Per_Week'] = df['Deficit'].rolling(window=7, min_periods=1).sum() / 3500 / 1


# Sidebar filters
st.sidebar.title("📅 Filters")
start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())
df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

st.title("🏋️ Accountability Tracker")
# ====================
# ✅ Goal Completion + Summary Stats + Streaks Adaptive Layout
# ====================

# Compute values
goal_days = df_filtered['All_Goals_Met'].sum()
total_days = len(df_filtered)
percent = (goal_days / total_days) * 100 if total_days else 0

starting_weight = df_filtered['Weight'].iloc[0]
latest_avg_weight = df_filtered['7Day_Rolling_Weight'].iloc[-1]
weight_lost = starting_weight - latest_avg_weight

starting_bf = df_filtered['BF%'].iloc[0]
latest_avg_bf = df_filtered['7Day_Rolling_BF'].iloc[-1]
bf_lost = starting_bf - latest_avg_bf
# -------------------------------
# Target weight line: -1 lb/week
# -------------------------------
days_from_start = (df_filtered['Date'] - df_filtered['Date'].iloc[0]).dt.days
projected_weight_1lb_per_week = df_filtered['Weight'].iloc[0] - (days_from_start / 9)


# last time my weight was the lowest
df_filtered_sorted = df_filtered.sort_values(['Weight', 'Date'], ascending=[True, False])
lowest_weight_date = df_filtered_sorted['Date'].iloc[0]
lowest_weight = df_filtered_sorted['Weight'].iloc[0]

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

# Adaptive layout: define metrics in rows of 4 for smaller screens
metrics = [
    ("✅ Days All Goals Met", f"{goal_days}/{total_days} ({percent:.1f}%)"),
    ("⚖️ Weight Lost (Observed)", f"{weight_lost:.1f} lbs"),
    ("⚖️ Weight Lost (Deficit)", f"{df_filtered['Weight_Lost_From_Deficit'].iloc[-1]:.1f} lbs"),
    ("📅 RL7 Weight Lost/Week", f"{df_filtered['7Day_Rolling_Avg_Weight_Lost_Per_Week'].iloc[-1]:.2f} lbs"),
    ("💪 Body Fat % Lost", f"{bf_lost:.1f}%"),
    ("⚖️ RL7 Weight", f"{latest_avg_weight:.1f} lbs"),
    ("💪 RL7 Body Fat %", f"{latest_avg_bf:.1f}%"),
    ("💪 RL7 Muscle Mass", f"{df_filtered['7Day_Rolling_Muscle'].iloc[-1]:.1f} lbs"),
    ("🔥 RL7 Calories Consumed", f"{avg_calories:.0f} kcal"),
    ("📉 RL7 Daily Deficit", f"{avg_deficit:.0f} kcal"),
    ("👣 RL7 Daily Steps", f"{avg_steps:.0f}"),
    ("🏃 RL7 Exercise Calories", f"{avg_exercise_cal:.0f} kcal"),
    ("🔥 Longest Streak", f"{longest_streak} days"),
    ("🔥 Current Streak", f"{current_streak} days"),
    ("⚖️ Lowest Weight", f"{lowest_weight:.1f} lbs"),
    ("📅 Date of Lowest Weight", f"{lowest_weight_date.date()}"),
    (("Days Since Lowest Weight"), f"{(df_filtered['Date'].iloc[-1] - lowest_weight_date).days} days")
]

# Display metrics in rows of 4
cols_per_row = 4
for i in range(0, len(metrics), cols_per_row):
    cols = st.columns(min(cols_per_row, len(metrics) - i))
    for col, (title, value) in zip(cols, metrics[i:i+cols_per_row]):
        col.metric(title, value)

# ============================
# 📊 UPDATED PLOTTING SECTION
# ============================

st.subheader("📅 Metrics over time")

# Colors
dark_colors = [
    "#1f77b4",  # Weight
    "#ff7f0e",  # BF%
    "#2ca02c",  # Steps
    "#d62728",  # Consumed
    "#9467bd",  # Exercise
    "#8c564b",  # Deficit
    "#17becf"   # Muscle Mass (NEW)
]

light_colors = [
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#9edae5"   # Muscle Mass (NEW)
]

plot_titles = [
    "Weight", 
    "BF%",
    "Muscle Mass",
    "Steps", 
    "Calories Consumed",
    "Exercise Calories", 
    "Deficit",
    "7-Day Rolling Weight Change",
    "Cumulative Deficit",
    "Deficit vs. 7-Day Rolling Weight Change"
]

# Create 5 rows × 2 columns (10 total plots)
fig = make_subplots(rows=5, cols=2, subplot_titles=plot_titles)

# Paired plots (raw + 7-day rolling)
paired_plots = [
    ('Weight', '7Day_Rolling_Weight'),
    ('BF%', '7Day_Rolling_BF'),
    ('Muscle Mass', '7Day_Rolling_Muscle'),
    ('Steps', '7Day_Rolling_Steps'),
    ('Calories Consumed', '7Day_Rolling_Consumed_Calories'),
    ('Calories from Exercise', '7Day_Rolling_Activity_Calories'),
    ('Deficit', '7Day_Rolling_Deficit')
]

# Add paired line plots (rows 1–4)
for i, (raw, rolling) in enumerate(paired_plots):
    row = (i // 2) + 1
    col = (i % 2) + 1
    fig.add_trace(
        go.Scatter(
            x=df_filtered['Date'], y=df_filtered[raw],
            mode='lines', line=dict(color=light_colors[i], width=2),
            showlegend=False
        ),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=df_filtered['Date'], y=df_filtered[rolling],
            mode='lines', line=dict(color=dark_colors[i], width=3),
            showlegend=False
        ),
        row=row, col=col
    )

    # --- Target 1 lb/week loss line (ONLY on Weight plot) ---
    if raw == 'Weight':
        fig.add_trace(
            go.Scatter(
                x=df_filtered['Date'],
                y=projected_weight_1lb_per_week,
                mode='lines',
                line=dict(color='black', width=3, dash='dot'),
                showlegend=False
            ),
            row=row, col=col
        )


# 7-Day Rolling Weight Change (row 5, col 1)
fig.add_trace(
    go.Bar(
        x=df_filtered['Date'],
        y=df_filtered['7Day_Rolling_Weight_Change'],
        marker_color="#e377c2",
        showlegend=False
    ),
    row=4, col=2
)
fig.add_hline(y=0, line_dash="dash", line_color="black", row=5, col=1)

# Cumulative Deficit (row 5, col 1) — AREA CHART
fig.add_trace(
    go.Scatter(
        x=df_filtered['Date'],
        y=df_filtered['Cumulative_Deficit'],
        mode='lines',
        line=dict(color="#7f7f7f", width=2),
        fill='tozeroy',  # <-- makes it an area chart
        fillcolor="rgba(127,127,127,0.3)",  # light transparent fill
        showlegend=False
    ),
    row=5, col=1
)

# Scatter plot inside subplot row 5, col 2
fig.add_trace(
    go.Scatter(
        x=df_filtered['Deficit'],
        y=df_filtered['7Day_Rolling_Weight_Change'],
        mode='markers',
        marker=dict(size=6, color="#1f77b4", opacity=0.7),
        showlegend=False
    ),
    row=5, col=2
)
# Horizontal line at y = 0
fig.add_hline(
    y=0,
    line_dash="dash",
    line_color="black",
    row=5, col=2
)

# Vertical line at x = 0
fig.add_vline(
    x=0,
    line_dash="dash",
    line_color="black",
    row=5, col=2
)

fig.update_xaxes(title_text="Daily Deficit (kcal)", row=5, col=2)
fig.update_yaxes(title_text="RL7 Weight Change (lbs/day)", row=5, col=2)


# Global styling
fig.update_layout(
    height=2000,
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black')
)

# Axes cleanup
fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
fig.update_yaxes(showgrid=False, showticklabels=True, zeroline=False, tickfont=dict(color='black'))

st.plotly_chart(fig, use_container_width=True)