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

# st.set_page_config(layout="wide")


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
df['Deficit'] = df['Calories from Exercise'] + 1930 - df['Calories Consumed']
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

# Rate of change (day-to-day difference) of 7-day rolling average weight
df['7Day_Rolling_Weight_Change'] = df['7Day_Rolling_Weight'].diff()

# Cumulative deficit over time
df['Cumulative_Deficit'] = df['Deficit'].cumsum()

# weight lost from deficit
df['Weight_Lost_From_Deficit'] = df['Cumulative_Deficit'] / 3500

# average weight lost per week
df['Avg_Weight_Lost_Per_Week'] = df['Weight_Lost_From_Deficit'] / (len(df) / 7)


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
    ("⚖️ Weight  (Observed)", f"{weight_lost:.1f} lbs"),
    ("⚖️ Weight Lost (Deficit)", f"{df_filtered['Weight_Lost_From_Deficit'].iloc[-1]:.1f} lbs"),
    ("📅 Avg Weight Lost/Week", f"{df_filtered['Avg_Weight_Lost_Per_Week'].iloc[-1]:.2f} lbs"),
    ("💪 Body Fat % Lost", f"{bf_lost:.1f}%"),
    ("🔥 RL7 Calories Consumed", f"{avg_calories:.0f} kcal"),
    ("📉 RL7 Daily Deficit", f"{avg_deficit:.0f} kcal"),
    ("👣 RL7 Daily Steps", f"{avg_steps:.0f}"),
    ("🏃 RL7 Exercise Calories", f"{avg_exercise_cal:.0f} kcal"),
    ("🔥 Longest Streak", f"{longest_streak} days"),
    ("🔥 Current Streak", f"{current_streak} days")
]

# Display metrics in rows of 4
cols_per_row = 4
for i in range(0, len(metrics), cols_per_row):
    cols = st.columns(min(cols_per_row, len(metrics) - i))
    for col, (title, value) in zip(cols, metrics[i:i+cols_per_row]):
        col.metric(title, value)

# ====================
# 📅 Calendar View of Goal Completion (Current Month Only)
# ====================
st.subheader("📅 Metrics over time")

# Convert Date to datetime (just in case)
df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])

# # Get current month and year
# today = pd.Timestamp.today()
# current_month_data = df_filtered[
#     (df_filtered['Date'].dt.month == today.month) &
#     (df_filtered['Date'].dt.year == today.year)
# ]

# # Use 1 for goal met, 0 for not met
# calendar_data = current_month_data.set_index('Date')['All_Goals_Met'].astype(int)

# # Generate calendar heatmap
# fig_cal, ax_cal = calplot.calplot(
#     calendar_data,
#     how='sum',
#     cmap='OrRd',
#     figsize=(12, 3.5),
#     colorbar=True,
#     suptitle='Days All Goals Met – This Month',
# )

# st.pyplot(fig_cal)


# # === PLOTTING SECTION ===
# # ⚖️ Estimated vs Actual Weight Lost
# st.subheader("⚖️ Estimated vs Actual Weight Lost")
# estimated_weight_lost = df_filtered['Deficit'].sum() / 3500 if len(df_filtered) > 0 else 0
# actual_weight_lost = df_filtered['Weight'].iloc[0] - df_filtered['7Day_Rolling_Weight'].iloc[-1] if len(df_filtered) > 0 else 0

# fig_weight_compare = go.Figure()
# fig_weight_compare.add_trace(go.Bar(
#     x=['Estimated', 'Actual'],
#     y=[estimated_weight_lost, actual_weight_lost],
#     marker_color=['#636EFA', '#EF553B'],
#     text=[f"{estimated_weight_lost:.1f}", f"{actual_weight_lost:.1f}"],
#     textposition='auto'
# ))
# fig_weight_compare.update_layout(
#     xaxis=dict(showticklabels=True, showgrid=False, zeroline=False),
#     yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
#     plot_bgcolor='white', paper_bgcolor='white',
#     margin=dict(l=10, r=10, t=25, b=10),
#     showlegend=False
# )
# st.plotly_chart(fig_weight_compare, use_container_width=True)
# Colors for each subplot pair
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
]

# ====================
# Short titles for each subplot
# Short titles for each subplot
plot_titles = [
    "Weight", "BF%", "Steps",
    "Calories Consumed", "Exercise Calories", "Deficit",
    "7-Day Rolling Weight Change", "Cumulative Deficit"
]

# Dark colors for raw metrics
dark_colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]

# Lighter versions for 7-day rolling metrics
light_colors = [
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
    "#c49c94",  # light brown
]

# Create subplot grid: 4 rows x 2 cols → 8 plots
fig = make_subplots(rows=4, cols=2, subplot_titles=plot_titles)

# Add paired line plots
paired_plots = [
    ('Weight', '7Day_Rolling_Weight'),
    ('BF%', '7Day_Rolling_BF'),
    ('Steps', '7Day_Rolling_Steps'),
    ('Calories Consumed', '7Day_Rolling_Consumed_Calories'),
    ('Calories from Exercise', '7Day_Rolling_Activity_Calories'),
    ('Deficit', '7Day_Rolling_Deficit'),
]

# Add the first 6 paired plots to first 3 rows
for i, (raw, rolling) in enumerate(paired_plots[:6]):
    row = (i // 2) + 1
    col = (i % 2) + 1
    # Raw metric → light color
    fig.add_trace(
        go.Scatter(
            x=df_filtered['Date'], y=df_filtered[raw],
            mode='lines', line=dict(color=light_colors[i], width=2),
            name=raw, showlegend=False
        ),
        row=row, col=col
    )
    # Rolling metric → dark color
    fig.add_trace(
        go.Scatter(
            x=df_filtered['Date'], y=df_filtered[rolling],
            mode='lines', line=dict(color=dark_colors[i], width=3, dash='dash'),
            name=f"{rolling} (7D)", showlegend=False
        ),
        row=row, col=col
    )
# 7-Day Rolling Weight Change (full row)
fig.add_trace(
    go.Bar(
        x=df_filtered['Date'], y=df_filtered['7Day_Rolling_Weight_Change'],
        marker_color=colors[6], name='7D Rolling Weight Change', showlegend=False
    ),
    row=4, col=1
)
fig.add_hline(y=0, line_dash="dash", line_color="black", row=4, col=1)

# Cumulative Deficit (last subplot)
fig.add_trace(
    go.Scatter(
        x=df_filtered['Date'], y=df_filtered['Cumulative_Deficit'],
        mode='lines', line=dict(color=colors[7], width=2),
        name='Cumulative Deficit', showlegend=False
    ),
    row=4, col=2
)

# Global layout
fig.update_layout(
    height=1600,
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black')
)

# Make Y-axis values visible and black
fig.update_yaxes(showticklabels=True, zeroline=False, tickfont=dict(color='black'))
fig.update_xaxes(showticklabels=False, zeroline=False)

# Render in Streamlit
st.plotly_chart(fig, use_container_width=True)
