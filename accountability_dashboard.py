import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from plotly.subplots import make_subplots


# ==========================
# Google Sheets Connection
# ==========================

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds_dict = st.secrets["google"]

credentials = ServiceAccountCredentials.from_json_keyfile_dict(
    creds_dict,
    scope
)

client = gspread.authorize(credentials)

sheet = client.open("Accountability Tracker").sheet1

data = sheet.get_all_records()

df = pd.DataFrame(data)


# ==========================
# Data Cleaning
# ==========================

df['Date'] = pd.to_datetime(df['Date'])

df.sort_values(
    'Date',
    inplace=True
)
# Convert numeric columns from strings to numbers
numeric_cols = [
    'Calories from Exercise',
    'Calories Consumed',
    'Weight',
    'Steps',
    'Muscle Mass',
    'Protein > 130'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


# ==========================
# Calculations
# ==========================

# Daily deficit
df['Deficit'] = (
    df['Calories from Exercise']
    - df['Calories Consumed']
)


# Goals
df['Goal_Deficit'] = df['Deficit'] > 0

df['All_Goals_Met'] = (
    df['Goal_Deficit']
    &
    df['Protein > 130']
)


# Rolling averages

df['7Day_Rolling_Weight'] = (
    df['Weight']
    .rolling(
        window=7,
        min_periods=1
    )
    .mean()
)


df['7Day_Rolling_Deficit'] = (
    df['Deficit']
    .rolling(
        window=7,
        min_periods=1
    )
    .mean()
)


df['7Day_Rolling_Steps'] = (
    df['Steps']
    .rolling(
        window=7,
        min_periods=1
    )
    .mean()
)


df['7Day_Rolling_Activity_Calories'] = (
    df['Calories from Exercise']
    .rolling(
        window=7,
        min_periods=1
    )
    .mean()
)


df['7Day_Rolling_Consumed_Calories'] = (
    df['Calories Consumed']
    .rolling(
        window=7,
        min_periods=1
    )
    .mean()
)



# Weight change

df['7Day_Rolling_Weight_Change'] = (
    df['7Day_Rolling_Weight']
    .diff()
)



# Cumulative deficit

df['Cumulative_Deficit'] = (
    df['Deficit']
    .cumsum()
)



# Estimated weight loss

df['Weight_Lost_From_Deficit'] = (
    df['Cumulative_Deficit']
    /
    3500
)



df['Avg_Weight_Lost_Per_Week'] = (
    df['Weight_Lost_From_Deficit']
    /
    (len(df) / 7)
)



df['7Day_Rolling_Avg_Weight_Lost_Per_Week'] = (
    df['Deficit']
    .rolling(
        window=7,
        min_periods=1
    )
    .sum()
    /
    3500
)



# ==========================
# Sidebar Filters
# ==========================

st.sidebar.title("📅 Filters")

start_date = st.sidebar.date_input(
    "Start Date",
    df['Date'].min()
)

end_date = st.sidebar.date_input(
    "End Date",
    df['Date'].max()
)


df_filtered = df[
    (df['Date'] >= pd.to_datetime(start_date))
    &
    (df['Date'] <= pd.to_datetime(end_date))
]


st.title("🏋️ Accountability Tracker")



# ==========================
# Summary Calculations
# ==========================

goal_days = (
    df_filtered['All_Goals_Met']
    .sum()
)

total_days = len(df_filtered)


percent = (
    goal_days / total_days * 100
    if total_days > 0
    else 0
)



starting_weight = (
    df_filtered['Weight']
    .iloc[0]
)


latest_avg_weight = (
    df_filtered['7Day_Rolling_Weight']
    .iloc[-1]
)


weight_lost = (
    starting_weight
    -
    latest_avg_weight
)



avg_calories = (
    df_filtered['7Day_Rolling_Consumed_Calories']
    .iloc[-1]
)


avg_deficit = (
    df_filtered['7Day_Rolling_Deficit']
    .iloc[-1]
)


avg_steps = (
    df_filtered['7Day_Rolling_Steps']
    .iloc[-1]
)


avg_exercise_cal = (
    df_filtered['7Day_Rolling_Activity_Calories']
    .iloc[-1]
)



# Lowest weight

lowest_weight_row = (
    df_filtered
    .sort_values(
        ['Weight','Date'],
        ascending=[True,False]
    )
    .iloc[0]
)


lowest_weight = lowest_weight_row['Weight']

lowest_weight_date = lowest_weight_row['Date']



# ==========================
# Streak Calculations
# ==========================

def compute_streaks(series):

    max_streak = 0
    curr_streak = 0
    last_date = None

    for date, met in zip(series.index, series):

        if met:

            if (
                last_date is None
                or
                (date-last_date).days == 1
            ):
                curr_streak += 1

            else:
                curr_streak = 1


            max_streak = max(
                max_streak,
                curr_streak
            )

        else:

            curr_streak = 0


        last_date = date


    return max_streak



df_streak = (
    df_filtered
    .copy()
    .set_index('Date')
)


longest_streak = compute_streaks(
    df_streak['All_Goals_Met']
)


current_streak = 0


for met in (
    df_streak
    .sort_index(ascending=False)
    ['All_Goals_Met']
):

    if met:
        current_streak += 1

    else:
        break



# ==========================
# Metrics Display
# ==========================

metrics = [

    (
        "✅ Days All Goals Met",
        f"{goal_days}/{total_days} ({percent:.1f}%)"
    ),

    (
        "⚖️ Weight Lost",
        f"{weight_lost:.1f} lbs"
    ),

    (
        "⚖️ Weight Lost (Deficit)",
        f"{df_filtered['Weight_Lost_From_Deficit'].iloc[-1]:.1f} lbs"
    ),

    (
        "📅 RL7 Weight Lost/Week",
        f"{df_filtered['7Day_Rolling_Avg_Weight_Lost_Per_Week'].iloc[-1]:.2f} lbs"
    ),

    (
        "⚖️ RL7 Weight",
        f"{latest_avg_weight:.1f} lbs"
    ),

    (
        "🔥 RL7 Calories",
        f"{avg_calories:.0f} kcal"
    ),

    (
        "📉 RL7 Deficit",
        f"{avg_deficit:.0f} kcal"
    ),

    (
        "👣 RL7 Steps",
        f"{avg_steps:.0f}"
    ),

    (
        "🏃 RL7 Exercise Calories",
        f"{avg_exercise_cal:.0f} kcal"
    ),

    (
        "🔥 Longest Streak",
        f"{longest_streak} days"
    ),

    (
        "🔥 Current Streak",
        f"{current_streak} days"
    ),

    (
        "⚖️ Lowest Weight",
        f"{lowest_weight:.1f} lbs"
    ),

    (
        "📅 Lowest Weight Date",
        f"{lowest_weight_date.date()}"
    ),

    (
        "Days Since Lowest Weight",
        f"{(df_filtered['Date'].iloc[-1]-lowest_weight_date).days}"
    )
]


cols_per_row = 4


for i in range(
    0,
    len(metrics),
    cols_per_row
):

    cols = st.columns(
        min(
            cols_per_row,
            len(metrics)-i
        )
    )

    for col, (title,value) in zip(
        cols,
        metrics[i:i+cols_per_row]
    ):

        col.metric(
            title,
            value
        )

# ==========================
# 📊 Metrics Over Time
# ==========================

st.subheader("📅 Metrics Over Time")


# ==========================
# Colors
# ==========================

dark_colors = [

    "#1f77b4",  # Weight
    "#2ca02c",  # Steps
    "#d62728",  # Calories Consumed
    "#9467bd",  # Exercise Calories
    "#8c564b"   # Deficit

]


light_colors = [

    "#aec7e8",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94"

]



# ==========================
# Plot Titles
# ==========================

plot_titles = [

    "Weight",

    "Steps",

    "Calories Consumed",

    "Exercise Calories",

    "Deficit",

    "7-Day Rolling Weight Change",

    "Cumulative Deficit",

    "Deficit vs Weight Change"

]



# ==========================
# Create subplot layout
# ==========================

fig = make_subplots(

    rows=4,

    cols=2,

    subplot_titles=plot_titles

)



# ==========================
# Paired Metrics
# ==========================

paired_plots = [

    (
        "Weight",
        "7Day_Rolling_Weight"
    ),

    (
        "Steps",
        "7Day_Rolling_Steps"
    ),

    (
        "Calories Consumed",
        "7Day_Rolling_Consumed_Calories"
    ),

    (
        "Calories from Exercise",
        "7Day_Rolling_Activity_Calories"
    ),

    (
        "Deficit",
        "7Day_Rolling_Deficit"
    )

]



for i, (raw, rolling) in enumerate(paired_plots):

    row = (i // 2) + 1

    col = (i % 2) + 1



    # Raw metric

    fig.add_trace(

        go.Scatter(

            x=df_filtered['Date'],

            y=df_filtered[raw],

            mode="lines",

            line=dict(

                color=light_colors[i],

                width=2

            ),

            showlegend=False

        ),

        row=row,

        col=col

    )



    # Rolling metric

    fig.add_trace(

        go.Scatter(

            x=df_filtered['Date'],

            y=df_filtered[rolling],

            mode="lines",

            line=dict(

                color=dark_colors[i],

                width=3,

                dash="dash"

            ),

            showlegend=False

        ),

        row=row,

        col=col

    )





# ==========================
# RL7 Weight Change
# ==========================

fig.add_trace(

    go.Bar(

        x=df_filtered['Date'],

        y=df_filtered['7Day_Rolling_Weight_Change'],

        marker_color="#e377c2",

        showlegend=False

    ),

    row=3,

    col=2

)



# Zero line

fig.add_hline(

    y=0,

    line_dash="dash",

    line_color="black",

    row=3,

    col=2

)





# ==========================
# Cumulative Deficit Area Chart
# ==========================

fig.add_trace(

    go.Scatter(

        x=df_filtered['Date'],

        y=df_filtered['Cumulative_Deficit'],

        mode="lines",

        fill="tozeroy",

        line=dict(

            color="#7f7f7f",

            width=2

        ),

        showlegend=False

    ),

    row=4,

    col=1

)





# ==========================
# Deficit vs Weight Change Scatter
# ==========================

fig.add_trace(

    go.Scatter(

        x=df_filtered['Deficit'],

        y=df_filtered['7Day_Rolling_Weight_Change'],

        mode="markers",

        marker=dict(

            size=7,

            color="#1f77b4",

            opacity=0.7

        ),

        showlegend=False

    ),

    row=4,

    col=2

)



# Horizontal zero line

fig.add_hline(

    y=0,

    line_dash="dash",

    line_color="black",

    row=4,

    col=2

)



# Vertical zero line

fig.add_vline(

    x=0,

    line_dash="dash",

    line_color="black",

    row=4,

    col=2

)



fig.update_xaxes(

    title_text="Daily Deficit (kcal)",

    row=4,

    col=2

)


fig.update_yaxes(

    title_text="RL7 Weight Change",

    row=4,

    col=2

)





# ==========================
# Add 1 lb/week Target Weight Line
# ==========================

starting_weight = df_filtered['Weight'].iloc[0]

days_elapsed = (

    df_filtered['Date']

    -

    df_filtered['Date'].iloc[0]

).dt.days



df_filtered['Target_1lb_week_weight'] = (

    starting_weight

    -

    days_elapsed / 7

)



fig.add_trace(

    go.Scatter(

        x=df_filtered['Date'],

        y=df_filtered['Target_1lb_week_weight'],

        mode="lines",

        line=dict(

            color="black",

            width=2,

            dash="dot"

        ),

        showlegend=False

    ),

    row=1,

    col=1

)





# ==========================
# Formatting
# ==========================

fig.update_layout(

    height=1700,

    showlegend=False,

    plot_bgcolor="white",

    paper_bgcolor="white",

    font=dict(

        color="black"

    )

)



fig.update_xaxes(

    showgrid=False,

    showticklabels=False,

    zeroline=False

)


fig.update_yaxes(

    showgrid=False,

    showticklabels=True,

    tickfont=dict(

        color="black"

    ),

    zeroline=False

)



# Render

st.plotly_chart(

    fig,

    use_container_width=True

)