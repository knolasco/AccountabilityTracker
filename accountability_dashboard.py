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

st.set_page_config(layout="wide")


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


# Sidebar filters
st.sidebar.title("📅 Filters")
start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())
df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

st.title("🏋️ Accountability Tracker")

# ====================
# ✅ Goal Completion
# ====================
goal_days = df_filtered['All_Goals_Met'].sum()
total_days = len(df_filtered)
percent = (goal_days / total_days) * 100 if total_days else 0
st.metric("✅ Days All Goals Met", f"{goal_days} / {total_days} ({percent:.1f}%)")

# ====================
# 🔥 Summary Stats (Updated with Weight/BF% Lost)
# ====================

starting_weight = df_filtered['Weight'].iloc[0]
latest_avg_weight = df_filtered['7Day_Rolling_Weight'].iloc[-1]
weight_lost = starting_weight - latest_avg_weight

starting_bf = df_filtered['BF%'].iloc[0]
latest_avg_bf = df_filtered['7Day_Rolling_BF'].iloc[-1]
bf_lost = starting_bf - latest_avg_bf

# Display weight/BF% lost first
col0, col00 = st.columns(2)
with col0:
    st.metric("⚖️ Weight Lost", f"{weight_lost:.1f} lbs")
with col00:
    st.metric("💪 Body Fat % Lost", f"{bf_lost:.1f}%")

# Existing stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🔥 Avg Calories Consumed", f"{df_filtered['Calories Consumed'].mean():.0f} kcal")
with col2:
    st.metric("📉 Avg Daily Deficit", f"{df_filtered['Deficit'].mean():.0f} kcal")
with col3:
    st.metric("👣 Avg Daily Steps", f"{df_filtered['Steps'].mean():.0f}")

col4, col5 = st.columns(2)
with col4:
    st.metric("🏃 Avg Exercise Calories", f"{df_filtered['Calories from Exercise'].mean():.0f} kcal")
with col5:
    st.metric("⚖️ Rolling 7 Day Weight", f"{latest_avg_weight:.0f} lbs")

# ====================
# 🔁 Goal Streak Counters
# ====================

st.subheader("🔁 Goal Streaks")

# Helper to compute streaks
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

# Use 'Date' as index to compute streaks cleanly
df_streak = df_filtered.copy()
df_streak = df_streak.set_index('Date')
longest_streak = compute_streaks(df_streak['All_Goals_Met'])

# Compute current streak
today_or_latest = df_streak.index.max()
reversed_days = df_streak.sort_index(ascending=False)
current_streak = 0
for met in reversed_days['All_Goals_Met']:
    if met:
        current_streak += 1
    else:
        break

col_streak1, col_streak2 = st.columns(2)
with col_streak1:
    st.metric("🔥 Longest Streak (All Goals Met)", f"{longest_streak} days")
with col_streak2:
    st.metric("🔥 Current Streak", f"{current_streak} days")

# ====================
# 📅 Calendar View of Goal Completion (Current Month Only)
# ====================
st.subheader("📅 Goal Completion Calendar (Current Month)")

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


# === PLOTTING SECTION ===
# ⚖️ Estimated vs Actual Weight Lost
st.subheader("⚖️ Estimated vs Actual Weight Lost")
estimated_weight_lost = df_filtered['Deficit'].sum() / 3500 if len(df_filtered) > 0 else 0
actual_weight_lost = df_filtered['Weight'].iloc[0] - df_filtered['7Day_Rolling_Weight'].iloc[-1] if len(df_filtered) > 0 else 0

fig_weight_compare = go.Figure()
fig_weight_compare.add_trace(go.Bar(
    x=['Estimated', 'Actual'],
    y=[estimated_weight_lost, actual_weight_lost],
    marker_color=['#636EFA', '#EF553B'],
    text=[f"{estimated_weight_lost:.1f}", f"{actual_weight_lost:.1f}"],
    textposition='auto'
))
fig_weight_compare.update_layout(
    xaxis=dict(showticklabels=True, showgrid=False, zeroline=False),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    plot_bgcolor='white', paper_bgcolor='white',
    margin=dict(l=10, r=10, t=25, b=10),
    showlegend=False
)
st.plotly_chart(fig_weight_compare, use_container_width=True)

# Short titles for each plot
plot_titles = [
    "Weight",
    "7D Rolling Weight",
    "7D Rolling Weight Change",
    "Body Fat %",
    "7D Rolling BF%",
    "Steps",
    "7D Rolling Steps",
    "Calories Consumed",
    "7D Rolling Consumed Calories",
    "Exercise Calories",
    "7D Rolling Exercise Calories",
    "Deficit",
    "7D Rolling Deficit",
    "Cumulative Deficit"
]

# Create subplot grid: 12 plots total → 2 rows x 4 cols
fig = make_subplots(rows=4, cols=4, subplot_titles = plot_titles)



# Define a color palette with enough distinct colors using the requested hex codes
colors = [
    "#ac3a44", "#dbb13b", "#536437",
    "#ac3a44", "#dbb13b", "#536437",
    "#ac3a44", "#dbb13b", "#536437",
    "#ac3a44", "#dbb13b", "#536437",
    "#ac3a44", "#dbb13b", "#536437"
]

# Define traces
plots = [
    go.Scatter(x=df_filtered['Date'], y=df_filtered['Weight'], mode='lines', line=dict(color=colors[0], width=2), name=plot_titles[0]),
    go.Scatter(x=df_filtered['Date'], y=df_filtered['7Day_Rolling_Weight'], mode='lines', line=dict(color=colors[1], width=2), name=plot_titles[1]),
    go.Scatter(x=df_filtered['Date'], y=df_filtered['7Day_Rolling_Weight_Change'], mode='lines', line=dict(color=colors[2], width=2), name=plot_titles[2]),
    go.Scatter(x=df_filtered['Date'], y=df_filtered['BF%'], mode='lines', line=dict(color=colors[3], width=2), name=plot_titles[3]),
    go.Scatter(x=df_filtered['Date'], y=df_filtered['7Day_Rolling_BF'], mode='lines', line=dict(color=colors[4], width=2), name=plot_titles[4]),
    go.Scatter(x=df_filtered['Date'], y=df_filtered['Steps'], mode='lines', line=dict(color=colors[5], width=2), name=plot_titles[5]),
    go.Scatter(x=df_filtered['Date'], y=df_filtered['7Day_Rolling_Steps'], mode='lines', line=dict(color=colors[6], width=2), name=plot_titles[6]),
    go.Scatter(x=df_filtered['Date'], y=df_filtered['Calories Consumed'], mode='lines', line=dict(color=colors[7], width=2), name=plot_titles[7]),
    go.Scatter(x=df_filtered['Date'], y=df_filtered['7Day_Rolling_Consumed_Calories'], mode='lines', line=dict(color=colors[8], width=2), name=plot_titles[8]),
    go.Scatter(x=df_filtered['Date'], y=df_filtered['Calories from Exercise'], mode='lines', line=dict(color=colors[9], width=2), name=plot_titles[9]),
    go.Scatter(x=df_filtered['Date'], y=df_filtered['7Day_Rolling_Activity_Calories'], mode='lines', line=dict(color=colors[10], width=2), name=plot_titles[10]),
    go.Scatter(x=df_filtered['Date'], y=df_filtered['Deficit'], mode='lines', line=dict(color=colors[11], width=2), name=plot_titles[11]),
    go.Scatter(x=df_filtered['Date'], y=df_filtered['7Day_Rolling_Deficit'], mode='lines', line=dict(color=colors[12], width=2), name=plot_titles[12]),
    go.Scatter(x=df_filtered['Date'], y=df_filtered['Cumulative_Deficit'], mode='lines', line=dict(color=colors[13], width=2), name=plot_titles[13]),
]

# Add traces to subplot grid
for i, trace in enumerate(plots):
    row = (i // 4) + 1
    col = (i % 4) + 1
    fig.add_trace(trace, row=row, col=col)

# Global layout for compact sparklines
fig.update_layout(
    showlegend=False,
    height=600,  # adjust depending on number of rows
    margin=dict(l=5, r=5, t=10, b=5),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

# Remove ticks, labels, and grids for compactness
fig.update_xaxes(showticklabels=False, zeroline=False)
fig.update_yaxes(showticklabels=False, zeroline=False)

# Render in Streamlit
st.plotly_chart(fig)
