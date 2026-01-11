import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# =========================
# 🔐 GOOGLE SHEETS SETUP
# =========================
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds_dict = st.secrets["google"]
credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(credentials)

SPREADSHEET_NAME = "Accountability Tracker"

PHASE_TABS = {
    "Cut": "Cut",
    "Maintenance": "Maintenance",
    "Lean Bulk": "Lean Bulk"
}

# =========================
# 📥 LOAD DATA FUNCTION
# =========================
@st.cache_data(ttl=300)
def load_phase_data(tab_name):
    sheet = client.open(SPREADSHEET_NAME).worksheet(tab_name)
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # -------------------------
    # Metrics & Derived Columns
    # -------------------------
    df['Deficit'] = df['Calories from Exercise'] - df['Calories Consumed']
    df['Goal_Deficit'] = df['Deficit'] > 0
    df['All_Goals_Met'] = df['Goal_Deficit'] & df['Protein > 130']

    # Rolling averages
    df['7Day_Rolling_Weight'] = df['Weight'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_BF'] = df['BF%'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Deficit'] = df['Deficit'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Steps'] = df['Steps'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Consumed_Calories'] = df['Calories Consumed'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Activity_Calories'] = df['Calories from Exercise'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Muscle'] = df['Muscle Mass'].rolling(7, min_periods=1).mean()

    # Weight change & deficit math
    df['7Day_Rolling_Weight_Change'] = df['7Day_Rolling_Weight'].diff()
    df['Cumulative_Deficit'] = df['Deficit'].cumsum()
    df['Weight_Lost_From_Deficit'] = df['Cumulative_Deficit'] / 3500
    df['7Day_Rolling_Avg_Weight_Lost_Per_Week'] = (
        df['Deficit'].rolling(7, min_periods=1).sum() / 3500
    )

    return df


# =========================
# 🎛️ SIDEBAR
# =========================
st.sidebar.title("⚙️ Controls")

selected_phase = st.sidebar.selectbox(
    "Phase",
    list(PHASE_TABS.keys())
)

df = load_phase_data(PHASE_TABS[selected_phase])

start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())

df = df[(df['Date'] >= pd.to_datetime(start_date)) &
        (df['Date'] <= pd.to_datetime(end_date))]

# =========================
# 🏋️ HEADER
# =========================
st.title(f"🏋️ Accountability Tracker — {selected_phase}")

# =========================
# 📊 SUMMARY METRICS
# =========================
goal_days = df['All_Goals_Met'].sum()
total_days = len(df)
percent = (goal_days / total_days) * 100 if total_days else 0

starting_weight = df['Weight'].iloc[0]
latest_weight = df['7Day_Rolling_Weight'].iloc[-1]
weight_lost = starting_weight - latest_weight

cols = st.columns(4)
cols[0].metric("✅ Goal Days", f"{goal_days}/{total_days}")
cols[1].metric("📈 Success Rate", f"{percent:.1f}%")
cols[2].metric("⚖️ Weight Change", f"{weight_lost:.1f} lbs")
cols[3].metric("🔥 RL7 Deficit", f"{df['7Day_Rolling_Deficit'].iloc[-1]:.0f} kcal")

# =========================
# 📈 PLOTS
# =========================
fig = make_subplots(
    rows=4,
    cols=2,
    subplot_titles=[
        "Weight",
        "Body Fat %",
        "Muscle Mass",
        "Steps",
        "Calories Consumed",
        "Exercise Calories",
        "Daily Deficit",
        "Cumulative Deficit"
    ]
)

pairs = [
    ('Weight', '7Day_Rolling_Weight'),
    ('BF%', '7Day_Rolling_BF'),
    ('Muscle Mass', '7Day_Rolling_Muscle'),
    ('Steps', '7Day_Rolling_Steps'),
    ('Calories Consumed', '7Day_Rolling_Consumed_Calories'),
    ('Calories from Exercise', '7Day_Rolling_Activity_Calories'),
    ('Deficit', '7Day_Rolling_Deficit'),
]

for i, (raw, rolling) in enumerate(pairs):
    r = (i // 2) + 1
    c = (i % 2) + 1

    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df[raw],
            line=dict(width=2),
            opacity=0.3,
            showlegend=False
        ),
        row=r, col=c
    )

    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df[rolling],
            line=dict(width=3),
            showlegend=False
        ),
        row=r, col=c
    )

# Cumulative deficit (area)
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['Cumulative_Deficit'],
        fill='tozeroy',
        showlegend=False
    ),
    row=4, col=2
)

fig.update_layout(
    height=1600,
    plot_bgcolor="white",
    paper_bgcolor="white",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)
