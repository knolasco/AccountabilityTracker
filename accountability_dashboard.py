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
    BMR = 1950  # your stated BMR
    BASE_LIGHT_TDEE = 2400  # your best current "real-world" light-day maintenance baseline
    MEDIUM_BUMP = 200
    HARD_BUMP = 400

    # Helper: robust activity scoring using multiple signals
    def classify_activity(row):
        steps = row.get("Steps", 0) or 0
        mins = row.get("Exercise Minutes", 0) or 0
        g = row.get("Garmin_Total_Cals", 0) or 0

        # 1) Easy: low training load OR low Garmin total, even if steps are high
        # This matches your 2/4 (60 min, 12k steps, 2456 total) being "easy"
        if (mins < 70 and g < 2550):
            return "Light"

        # 2) Hard: very high training load OR Garmin clearly high
        if (mins >= 160) or (g >= 2950):
            return "Hard"

        # 3) Medium: everything else, with a small boost for high-step days
        # If you want, you can treat very high steps as Medium even with moderate minutes.
        if (mins >= 70) or (g >= 2650) or (steps >= 12000 and g >= 2550):
            return "Medium"

        return "Light"

    df["Activity_Level"] = df.apply(classify_activity, axis=1)

    # Map activity level to estimated TDEE
    level_to_bump = {"Light": 0, "Medium": MEDIUM_BUMP, "Hard": HARD_BUMP}
    df["Estimated_TDEE"] = df["Activity_Level"].map(level_to_bump).fillna(0) + BASE_LIGHT_TDEE

    # NEW deficit logic: estimated expenditure minus intake
    df["Deficit"] = df["Estimated_TDEE"] - df["Calories Consumed"]
    df["Goal_Deficit"] = df["Deficit"] > 0
    df["All_Goals_Met"] = df["Goal_Deficit"] & df["Protein > 130"]

    # Rolling averages
    df['7Day_Rolling_Weight'] = df['Weight'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_BF'] = df['BF%'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Deficit'] = df['Deficit'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Steps'] = df['Steps'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Activity_Calories'] = df['Estimated_TDEE'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Consumed_Calories'] = df['Calories Consumed'].rolling(7, min_periods=1).mean()
    df['7Day_Rolling_Muscle'] = df['Muscle Mass'].rolling(7, min_periods=1).mean()

    # Change & deficit math
    df['7Day_Rolling_Weight_Change'] = df['7Day_Rolling_Weight'].diff()
    df['Cumulative_Deficit'] = df['Deficit'].cumsum()
    df['Weight_Lost_From_Deficit'] = df['Cumulative_Deficit'] / 3500
    df['7Day_Rolling_Avg_Weight_Lost_Per_Week'] = (
        df['Deficit'].rolling(7, min_periods=1).sum() / 3500
    )
    # Weekly rate of change (7-day rolling avg vs 7 days ago)
    df['RL7_Weekly_Weight_Diff'] = df['7Day_Rolling_Weight'].diff(7)


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
# 📊 PHASE-SPECIFIC METRICS (WITH DELTAS)
# =========================

def weekly_delta(series):
    if len(series) < 8:
        return None
    return series.iloc[-1] - series.iloc[-8]

latest = df_filtered.iloc[-1]

# ---- Shared rolling metrics ----
rl7_weight = df_filtered['7Day_Rolling_Weight']
rl7_calories = df_filtered['7Day_Rolling_Consumed_Calories']
rl7_steps = df_filtered['7Day_Rolling_Steps']
rl7_active = df_filtered['7Day_Rolling_Activity_Calories']
rl7_deficit = df_filtered['7Day_Rolling_Deficit']
rl7_weight_change = df_filtered['7Day_Rolling_Weight_Change']

starting_weight = df_filtered['Weight'].iloc[0]

# ---- CUT PHASE ----
if phase == "Cut":

    goal_days = df_filtered['All_Goals_Met'].sum()
    total_days = len(df_filtered)

    weight_lost = starting_weight - rl7_weight.iloc[-1]
    weight_lost_week = df_filtered['7Day_Rolling_Avg_Weight_Lost_Per_Week'].iloc[-1]

    # Lowest weight
    df_sorted = df_filtered.sort_values(['Weight', 'Date'], ascending=[True, False])
    lowest_weight = df_sorted['Weight'].iloc[0]
    lowest_weight_date = df_sorted['Date'].iloc[0]
    days_since_lowest = (df_filtered['Date'].iloc[-1] - lowest_weight_date).days

    # Streaks
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

    df_streak = df_filtered.set_index('Date')
    longest_streak = compute_streaks(df_streak['All_Goals_Met'])

    current_streak = 0
    for met in df_streak.sort_index(ascending=False)['All_Goals_Met']:
        if met:
            current_streak += 1
        else:
            break

    metrics = [
        ("✅ Days All Goals Met", f"{goal_days}/{total_days}", None),
        ("⚖️ Weight Lost", f"{weight_lost:.1f} lbs", weekly_delta(rl7_weight) * -1),
        ("📉 RL7 Weight Lost / Week", f"{weight_lost_week:.2f} lbs", None),
        ("⚖️ RL7 Weight", f"{rl7_weight.iloc[-1]:.1f} lbs", weekly_delta(rl7_weight)),
        ("🔥 RL7 Calories", f"{rl7_calories.iloc[-1]:.0f}", weekly_delta(rl7_calories)),
        ("👣 RL7 Steps", f"{rl7_steps.iloc[-1]:.0f}", weekly_delta(rl7_steps)),
        ("📉 RL7 Deficit", f"{rl7_deficit.iloc[-1]:.0f}", weekly_delta(rl7_deficit)),
        ("🏃 RL7 Active Cal", f"{rl7_active.iloc[-1]:.0f}", weekly_delta(rl7_active)),
        ("⚖️ Lowest Weight", f"{lowest_weight:.1f} lbs", None),
        ("📅 Days Since Lowest", f"{days_since_lowest} days", None),
        ("🔥 Longest Streak", f"{longest_streak} days", None),
        ("🔥 Current Streak", f"{current_streak} days", None),
    ]

# ---- MAINTENANCE PHASE ----
elif phase == "Maintenance":

    metrics = [
        ("⚖️ Starting Weight", f"{starting_weight:.1f} lbs", None),
        ("⚖️ RL7 Weight", f"{rl7_weight.iloc[-1]:.1f} lbs", weekly_delta(rl7_weight)),
        ("🔥 RL7 Calories", f"{rl7_calories.iloc[-1]:.0f}", weekly_delta(rl7_calories)),
        ("👣 RL7 Steps", f"{rl7_steps.iloc[-1]:.0f}", weekly_delta(rl7_steps)),
        ("🏃 RL7 Active Cal", f"{rl7_active.iloc[-1]:.0f}", weekly_delta(rl7_active)),
    ]

# ---- LEAN BULK PHASE ----
elif phase == "Lean Bulk":

    rl7_surplus = rl7_calories - rl7_active
    rl7_weight_gain_week = rl7_weight.diff(7).iloc[-1]

    metrics = [
        ("⚖️ Starting Weight", f"{starting_weight:.1f} lbs", None),
        ("⚖️ RL7 Weight", f"{rl7_weight.iloc[-1]:.1f} lbs", weekly_delta(rl7_weight)),
        ("🔥 RL7 Calories", f"{rl7_calories.iloc[-1]:.0f}", weekly_delta(rl7_calories)),
        ("👣 RL7 Steps", f"{rl7_steps.iloc[-1]:.0f}", weekly_delta(rl7_steps)),
        ("📈 RL7 Surplus", f"{rl7_surplus.iloc[-1]:.0f}", weekly_delta(rl7_surplus)),
        ("🏃 RL7 Active Cal", f"{rl7_active.iloc[-1]:.0f}", weekly_delta(rl7_active)),
        ("📈 RL7 Weight Gain / Week", f"{rl7_weight_gain_week:.2f} lbs", None),
    ]

# ---- DISPLAY ----
st.subheader("📌 Key Metrics")

cols_per_row = 4
for i in range(0, len(metrics), cols_per_row):
    cols = st.columns(min(cols_per_row, len(metrics) - i))
    for col, (title, value, delta) in zip(cols, metrics[i:i + cols_per_row]):
        if delta is not None:
            col.metric(title, value, f"{delta:+.1f}")
        else:
            col.metric(title, value)


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
    "Cumulative Deficit", "RL7 Weekly Weight Change (lbs/week)"
]

fig = make_subplots(rows=5, cols=2, subplot_titles=plot_titles)

paired_plots = [
    ('Weight', '7Day_Rolling_Weight'),
    ('BF%', '7Day_Rolling_BF'),
    ('Muscle Mass', '7Day_Rolling_Muscle'),
    ('Steps', '7Day_Rolling_Steps'),
    ('Calories Consumed', '7Day_Rolling_Consumed_Calories'),
    ('Estimated_TDEE', '7Day_Rolling_Activity_Calories'),
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
fig.update_yaxes(range=[1700, 3200], row=3, col=2)


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

# RL7 weekly weight change (BAR)
fig.add_trace(
    go.Bar(
        x=df_filtered['Date'],
        y=df_filtered['RL7_Weekly_Weight_Diff'],
        marker_color="#1f77b4",
        showlegend=False
    ),
    row=5, col=2
)

# Reference lines: ±1 lb/week
fig.add_hline(
    y=-1,
    row=5,
    col=2,
    line_dash="dash",
    line_color="black"
)

fig.add_hline(
    y=0,
    row=5,
    col=2,
    line_dash="dot",
    line_color="gray"
)

fig.add_hline(
    y=1,
    row=5,
    col=2,
    line_dash="dash",
    line_color="black"
)

fig.update_yaxes(
    title_text="lbs / week",
    row=5,
    col=2
)


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
