import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ============================
# 🔐 Google Sheets Auth
# ============================

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds_dict = st.secrets["google"]
credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(credentials)

# ============================
# 📊 Phase Selection
# ============================

st.sidebar.title("📊 Phase Selection")

phase = st.sidebar.selectbox(
    "Select Phase",
    ["Cut", "Maintenance", "Lean Bulk"]
)

SHEET_MAP = {
    "Cut": "Accountability Tracker - Cut",
    "Maintenance": "Accountability Tracker - Maintenance",
    "Lean Bulk": "Accountability Tracker - Lean Bulk"
}

# ============================
# 📥 Load Sheet
# ============================

def load_sheet(sheet_name):
    sheet = client.open(sheet_name).sheet1
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    return df

df = load_sheet(SHEET_MAP[phase])

# ============================
# ⚙️ Preprocessing
# ============================

def preprocess_df(df, phase):
    df = df.copy()

    # Energy balance (positive = deficit)
    df['Energy_Balance'] = df['Calories from Exercise'] - df['Calories Consumed']

    if phase == "Cut":
        df['Goal_Energy'] = df['Energy_Balance'] > 0
    elif phase == "Maintenance":
        df['Goal_Energy'] = df['Energy_Balance'].between(-150, 150)
    else:  # Lean Bulk
        df['Goal_Energy'] = df['Energy_Balance'] < -150

    df['All_Goals_Met'] = df['Goal_Energy'] & df['Protein > 130']

    # Rolling metrics
    df['7Day_Rolling_Weight'] = df['Weight'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_BF'] = df['BF%'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Energy'] = df['Energy_Balance'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Steps'] = df['Steps'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Muscle'] = df['Muscle Mass'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Consumed'] = df['Calories Consumed'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Exercise'] = df['Calories from Exercise'].rolling(7, min_periods=1).mean()

    # Cumulative energy
    df['Cumulative_Energy'] = df['Energy_Balance'].cumsum()
    df['Weight_Change_From_Energy'] = df['Cumulative_Energy'] / 3500

    return df

df = preprocess_df(df, phase)

# ============================
# 📅 Date Filters
# ============================

st.sidebar.title("📅 Filters")

start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())

df_filtered = df[
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
]

# ============================
# 🧮 Summary Metrics
# ============================

st.title(f"🏋️ Accountability Tracker — {phase}")

goal_days = df_filtered['All_Goals_Met'].sum()
total_days = len(df_filtered)
percent = (goal_days / total_days * 100) if total_days else 0

starting_weight = df_filtered['Weight'].iloc[0]
latest_avg_weight = df_filtered['7Day_Rolling_Weight'].iloc[-1]
weight_change = latest_avg_weight - starting_weight

starting_bf = df_filtered['BF%'].iloc[0]
latest_avg_bf = df_filtered['7Day_Rolling_BF'].iloc[-1]
bf_change = latest_avg_bf - starting_bf

avg_energy = df_filtered['7Day_Rolling_Energy'].iloc[-1]
avg_steps = df_filtered['7Day_Rolling_Steps'].iloc[-1]
avg_exercise = df_filtered['7Day_Rolling_Exercise'].iloc[-1]
avg_consumed = df_filtered['7Day_Rolling_Consumed'].iloc[-1]

label = {
    "Cut": "Daily Deficit",
    "Maintenance": "Energy Balance",
    "Lean Bulk": "Daily Surplus"
}[phase]

metrics = [
    ("✅ Days Goals Met", f"{goal_days}/{total_days} ({percent:.1f}%)"),
    ("⚖️ Weight Change", f"{weight_change:+.1f} lbs"),
    ("💪 BF% Change", f"{bf_change:+.1f}%"),
    ("⚖️ RL7 Weight", f"{latest_avg_weight:.1f} lbs"),
    ("💪 RL7 Muscle", f"{df_filtered['7Day_Rolling_Muscle'].iloc[-1]:.1f} lbs"),
    ("🔥 RL7 Calories", f"{avg_consumed:.0f} kcal"),
    (f"🔥 RL7 {label}", f"{avg_energy:.0f} kcal"),
    ("👣 RL7 Steps", f"{avg_steps:.0f}"),
    ("🏃 RL7 Exercise Cals", f"{avg_exercise:.0f}")
]

cols = st.columns(3)
for i, (title, value) in enumerate(metrics):
    cols[i % 3].metric(title, value)

# ============================
# 📈 Projection Line
# ============================

days_from_start = (df_filtered['Date'] - df_filtered['Date'].iloc[0]).dt.days

weekly_rate = {
    "Cut": -1.0,
    "Maintenance": 0.0,
    "Lean Bulk": 0.25
}[phase]

projected_weight = (
    df_filtered['Weight'].iloc[0]
    + (days_from_start / 7) * weekly_rate
)

# ============================
# 📊 Plots
# ============================

fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=[
        "Weight", "BF%",
        "Muscle Mass", "Steps",
        "Calories Consumed", "Exercise Calories",
        "Energy Balance", "Cumulative Energy"
    ]
)

pairs = [
    ("Weight", "7Day_Rolling_Weight"),
    ("BF%", "7Day_Rolling_BF"),
    ("Muscle Mass", "7Day_Rolling_Muscle"),
    ("Steps", "7Day_Rolling_Steps"),
    ("Calories Consumed", "7Day_Rolling_Consumed"),
    ("Calories from Exercise", "7Day_Rolling_Exercise"),
    ("Energy_Balance", "7Day_Rolling_Energy")
]

for i, (raw, rolling) in enumerate(pairs):
    row = (i // 2) + 1
    col = (i % 2) + 1

    fig.add_trace(
        go.Scatter(x=df_filtered['Date'], y=df_filtered[raw],
                   line=dict(color="lightgray"), showlegend=False),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=df_filtered['Date'], y=df_filtered[rolling],
                   line=dict(color="black", width=3), showlegend=False),
        row=row, col=col
    )

    if raw == "Weight":
        fig.add_trace(
            go.Scatter(
                x=df_filtered['Date'],
                y=projected_weight,
                line=dict(color="black", dash="dot"),
                showlegend=False
            ),
            row=row, col=col
        )

fig.add_trace(
    go.Scatter(
        x=df_filtered['Date'],
        y=df_filtered['Cumulative_Energy'],
        fill='tozeroy',
        line=dict(color="gray"),
        showlegend=False
    ),
    row=4, col=2
)

fig.update_layout(
    height=1600,
    plot_bgcolor="white",
    paper_bgcolor="white"
)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

st.plotly_chart(fig, use_container_width=True)