import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# =========================
# 🔐 GOOGLE SHEETS AUTH
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
# 📥 LOAD DATA
# =========================
@st.cache_data(ttl=300)
def load_data(tab_name):
    sheet = client.open(SPREADSHEET_NAME).worksheet(tab_name)
    df = pd.DataFrame(sheet.get_all_records())

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # Core metrics
    df['Deficit'] = df['Calories from Exercise'] - df['Calories Consumed']
    df['Goal_Deficit'] = df['Deficit'] > 0
    df['All_Goals_Met'] = df['Goal_Deficit'] & df['Protein > 130']

    # Rolling averages
    df['7Day_Rolling_Weight'] = df['Weight'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_BF'] = df['BF%'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Deficit'] = df['Deficit'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Steps'] = df['Steps'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Activity_Calories'] = df['Calories from Exercise'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Consumed_Calories'] = df['Calories Consumed'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Muscle'] = df['Muscle Mass'].rolling(7, min_periods=1).mean()

    # Change & deficit math
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

phase = st.sidebar.selectbox("Phase", list(PHASE_TABS.keys()))
df = load_data(PHASE_TABS[phase])

start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())

df_filtered = df[
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
]

# =========================
# 🏋️ TITLE
# =========================
st.title(f"🏋️ Accountability Tracker — {phase}")

# =========================
# 📊 PLOTTING (UNCHANGED)
# =========================

dark_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b", "#17becf"
]

light_colors = [
    "#aec7e8", "#ffbb78", "#98df8a",
    "#ff9896", "#c5b0d5", "#c49c94", "#9edae5"
]

plot_titles = [
    "Weight", "BF%", "Muscle Mass", "Steps",
    "Calories Consumed", "Exercise Calories",
    "Deficit", "7-Day Rolling Weight Change",
    "Cumulative Deficit", "Deficit vs RL7 Weight Change"
]

fig = make_subplots(rows=5, cols=2, subplot_titles=plot_titles)

paired_plots = [
    ('Weight', '7Day_Rolling_Weight'),
    ('BF%', '7Day_Rolling_BF'),
    ('Muscle Mass', '7Day_Rolling_Muscle'),
    ('Steps', '7Day_Rolling_Steps'),
    ('Calories Consumed', '7Day_Rolling_Consumed_Calories'),
    ('Calories from Exercise', '7Day_Rolling_Activity_Calories'),
    ('Deficit', '7Day_Rolling_Deficit')
]

for i, (raw, rolling) in enumerate(paired_plots):
    row = (i // 2) + 1
    col = (i % 2) + 1

    fig.add_trace(
        go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered[raw],
            mode='lines',
            line=dict(color=light_colors[i], width=2),
            showlegend=False
        ),
        row=row, col=col
    )

    fig.add_trace(
        go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered[rolling],
            mode='lines',
            line=dict(color=dark_colors[i], width=3),
            showlegend=False
        ),
        row=row, col=col
    )

# 7-day rolling weight change (BAR)
fig.add_trace(
    go.Bar(
        x=df_filtered['Date'],
        y=df_filtered['7Day_Rolling_Weight_Change'],
        marker_color="#e377c2",
        showlegend=False
    ),
    row=4, col=2
)

# Cumulative deficit (AREA)
fig.add_trace(
    go.Scatter(
        x=df_filtered['Date'],
        y=df_filtered['Cumulative_Deficit'],
        mode='lines',
        fill='tozeroy',
        fillcolor="rgba(127,127,127,0.3)",
        line=dict(color="#7f7f7f", width=2),
        showlegend=False
    ),
    row=5, col=1
)

# Scatter: deficit vs weight change
fig.add_trace(
    go.Scatter(
        x=df_filtered['Deficit'],
        y=df_filtered['7Day_Rolling_Weight_Change'],
        mode='markers',
        marker=dict(size=6, opacity=0.7),
        showlegend=False
    ),
    row=5, col=2
)

fig.add_hline(y=0, row=5, col=2, line_dash="dash", line_color="black")
fig.add_vline(x=0, row=5, col=2, line_dash="dash", line_color="black")

fig.update_layout(
    height=2000,
    plot_bgcolor="white",
    paper_bgcolor="white",
    showlegend=False,
    font=dict(color="black")
)

fig.update_xaxes(showgrid=False, showticklabels=False)
fig.update_yaxes(showgrid=False, zeroline=False)

st.plotly_chart(fig, use_container_width=True)
