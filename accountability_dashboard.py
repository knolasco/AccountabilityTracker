import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import calplot
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials

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

# ⚖️ Body Weight Trend
st.subheader("⚖️ Body Weight Trend")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Weight'],
                          mode='lines+markers', name='Daily Weight', line=dict(color='gray')))
fig1.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['7Day_Rolling_Weight'],
                          mode='lines', name='7-Day Avg', line=dict(color='blue', width=3)))
fig1.update_layout(
    xaxis_title='Date',
    yaxis_title='Weight (lbs)',
    xaxis_tickangle=-45,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    xaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    yaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    legend=dict(font=dict(color='black'))  # <-- Add this line
)
st.plotly_chart(fig1, use_container_width=True)

# 📈 Rate of Change of 7-Day Rolling Weight
st.subheader("📈 Rate of Change – 7-Day Rolling Avg Weight")

fig_change = go.Figure()
fig_change.add_trace(go.Bar(
    x=df_filtered['Date'],
    y=df_filtered['7Day_Rolling_Weight_Change'],
    marker_color=df_filtered['7Day_Rolling_Weight_Change'].apply(lambda x: 'red' if x > 0 else 'green'),
    name="Daily Change"
))
fig_change.add_hline(y=0, line_dash="dash", line_color="black")
fig_change.update_layout(
    xaxis_title='Date',
    yaxis_title='Change in 7-Day Avg Weight (lbs)',
    xaxis_tickangle=-45,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    xaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    yaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black'))
)
st.plotly_chart(fig_change, use_container_width=True)


# 💪 Body Fat % Trend
st.subheader("💪 Body Fat % Trend")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['BF%'],
                          mode='lines+markers', name='Daily BF%', line=dict(color='violet')))
fig2.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['7Day_Rolling_BF'],
                          mode='lines', name='7-Day Avg', line=dict(color='purple', width=3)))
fig2.update_layout(
    xaxis_title='Date',
    yaxis_title='Body Fat %',
    xaxis_tickangle=-45,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    xaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    yaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    legend=dict(font=dict(color='black'))  # <-- Add this line
)
st.plotly_chart(fig2, use_container_width=True)

# ====================
# ⚖️ Estimated vs Actual Weight Lost
# ====================
st.subheader("⚖️ Estimated vs Actual Weight Lost")

# Calculate estimated weight lost from deficits
estimated_weight_lost = df_filtered['Deficit'].sum() / 3500 if len(df_filtered) > 0 else 0
actual_weight_lost = df_filtered['Weight'].iloc[0] - df_filtered['7Day_Rolling_Weight'].iloc[-1] if len(df_filtered) > 0 else 0

fig_weight_compare = go.Figure()
fig_weight_compare.add_trace(go.Bar(
    x=['Estimated (from Deficit)', 'Actual (Scale)'],
    y=[estimated_weight_lost, actual_weight_lost],
    marker_color=['#636EFA', '#EF553B'],
    text=[f"{estimated_weight_lost:.2f} lbs", f"{actual_weight_lost:.2f} lbs"],
    textposition='auto'
))
fig_weight_compare.update_layout(
    yaxis_title='Weight Lost (lbs)',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    xaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    yaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    showlegend=False,
    title="Estimated vs Actual Weight Lost"
)
st.plotly_chart(fig_weight_compare, use_container_width=True)


# 👣 Daily Steps
st.subheader("👣 Daily Steps")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Steps'],
                          mode='lines+markers', name='Daily Steps', line=dict(color='#4e79a7')))
fig3.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['7Day_Rolling_Steps'],
                          mode='lines', name='7-Day Avg', line=dict(color='#f28e2c', width=3)))
fig3.add_hline(y=10000, line_dash="dash", line_color="orange", annotation_text="Goal: 10,000", annotation_position="top left")
fig3.update_layout(
    xaxis_title='Date',
    yaxis_title='Steps',
    xaxis_tickangle=-45,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    xaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    yaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    legend=dict(font=dict(color='black'))  

)
st.plotly_chart(fig3, use_container_width=True)

# 🔥 Daily Calories Consumed
st.subheader("🔥 Daily Calories Consumed")
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Calories Consumed'],
                          mode='lines+markers', name='Daily Consumed Calories', line=dict(color='#76b7b2')))
fig4.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['7Day_Rolling_Consumed_Calories'],
                          mode='lines', name='7-Day Avg', line=dict(color='#e15759', width=3)))
fig4.update_layout(
    xaxis_title='Date',
    yaxis_title='Calories',
    xaxis_tickangle=-45,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    xaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    yaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    legend=dict(font=dict(color='black')) 
)
st.plotly_chart(fig4, use_container_width=True)

# 🏃 Daily Calories from Exercise
st.subheader("🏃 Daily Calories from Exercise")
fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Calories from Exercise'],
                          mode='lines+markers', name='Daily Burned Calories', line=dict(color='#59a14f')))
fig5.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['7Day_Rolling_Activity_Calories'],
                          mode='lines', name='7-Day Avg', line=dict(color='#edc949', width=3)))
fig5.update_layout(
    xaxis_title='Date',
    yaxis_title='Calories Burned',
    xaxis_tickangle=-45,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    xaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    yaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    legend=dict(font=dict(color='black')) 
)
st.plotly_chart(fig5, use_container_width=True)

# 📉 Daily Caloric Deficit
st.subheader("📉 Daily Caloric Deficit")
fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Deficit'],
                          mode='lines+markers', name='Deficit', line=dict(color='#af7aa1')))
fig6.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['7Day_Rolling_Deficit'],
                          mode='lines+markers', name='7-Day Avg', line=dict(color='#ff9da7', width=3)))
fig6.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="No Deficit", annotation_position="top left")
fig6.update_layout(
    xaxis_title='Date',
    yaxis_title='Deficit (kcal)',
    xaxis_tickangle=-45,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    xaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    yaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black')),
    legend=dict(font=dict(color='black')) 
)
st.plotly_chart(fig6, use_container_width=True)

# 📊 Cumulative Caloric Deficit (with Estimated Weight Loss)
st.subheader("📊 Cumulative Caloric Deficit & Estimated Weight Loss")

# Compute cumulative deficit & weight loss
df['Cumulative_Deficit'] = df['Deficit'].cumsum()
df['Cumulative_Weight_Lost'] = df['Cumulative_Deficit'] / 3500

fig_cum_deficit = go.Figure()

# Primary y-axis trace (Cumulative Deficit in kcal)
fig_cum_deficit.add_trace(go.Scatter(
    x=df_filtered['Date'],
    y=df_filtered['Cumulative_Deficit'],
    mode='lines',
    name='Cumulative Deficit (kcal)',
    line=dict(color='blue', width=3),
    yaxis="y1"
))

# Secondary y-axis trace (Cumulative Weight Lost in lbs)
fig_cum_deficit.add_trace(go.Scatter(
    x=df_filtered['Date'],
    y=df_filtered['Cumulative_Weight_Lost'],
    mode='lines',
    name='Estimated Weight Lost (lbs)',
    line=dict(color='green', width=3, dash="dot"),
    yaxis="y2"
))

# Add reference line for no deficit
fig_cum_deficit.add_hline(
    y=0,
    line_dash="dash",
    line_color="red",
    annotation_text="No Net Deficit",
    annotation_position="top left"
)

# Layout with dual y-axes
fig_cum_deficit.update_layout(
    xaxis_title='Date',
    yaxis=dict(
        title='Cumulative Deficit (kcal)',
        color='blue'
    ),
    yaxis2=dict(
        title='Estimated Weight Lost (lbs)',
        overlaying='y',
        side='right',
        color='green'
    ),
    xaxis_tickangle=-45,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    xaxis=dict(color='black', title_font=dict(color='black'), tickfont=dict(color='black'))
)

st.plotly_chart(fig_cum_deficit, use_container_width=True)




# # ====================
# # 📈 Correlation Matrix
# # ====================
# import seaborn as sns

# st.subheader("📊 Correlation Between Metrics")

# # Select numeric features for correlation
# corr_cols = [
#     'Calories Consumed',
#     'Calories from Exercise',
#     'Deficit',
#     'Weight',
#     'BF%',
#     'Protein > 130',
#     'Exercise Minutes',
#     'Steps'
# ]

# # Clean and compute correlations
# corr_data = df_filtered[corr_cols].copy()
# corr_data['Protein > 130'] = corr_data['Protein > 130'].astype(int)
# corr_matrix = corr_data.corr()

# # Plot heatmap
# fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", ax=ax_corr)
# ax_corr.set_title("Correlation Matrix")
# fig_corr.tight_layout()
# st.pyplot(fig_corr)
