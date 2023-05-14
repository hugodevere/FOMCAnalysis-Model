import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from FOMCAnalysis import FOMCAnalysis
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Set the background color and add a black line
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Set the start date and last date
start_date = '2023-01-01'
last_date = '2023-03-22'
ActualDay = datetime.today()

# Load data using cache
@st.cache(allow_output_mutation=True)
def load_data():
    analysis = FOMCAnalysis(start_date=start_date, restart_data=False)
    FOMC_participants_list = analysis.offi
    return analysis.extract_analysis(), FOMC_participants_list

data, offi = load_data()

# Set plot style and font family
plt.style.use('default')
plt.rcParams["font.family"] = "Times New Roman"

# Set page title
st.title("FOMC Committee Monitor by De Vere Research")

st.markdown(
    """
    <hr style="border: 1px solid black">
    """,
    unsafe_allow_html=True
)
# Plot Hawk vs Dove score and Sentiment index
st.subheader('FOMC Hawk vs Dove score, Sentiment index, market reaction using SOFR futures')
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), facecolor='w', sharex=True)
ax1s, ax2s = ax1.twinx(), ax2.twinx()
all_ranked = data['All_headlines'].sort_index(ascending=True)
all_ranked[['Fedspeak_hawk_dove_score']].rolling(window=pd.Timedelta('90 days')).sum().plot(ax=ax1, color='black')
all_ranked[['Fedspeak_sentiment_score']].rolling(window=pd.Timedelta('90 days')).sum().plot(ax=ax2, color='black')   
market_sentiment = pd.DataFrame((all_ranked['Price'] / all_ranked['Volume']).resample('1D').sum().ffill().rolling(pd.Timedelta('90 days')).sum()).dropna()
market_sentiment.columns = ['Market reaction']
market_sentiment.plot(ax=ax1s) 
market_sentiment.plot(ax=ax2s)
for ax in [ax1, ax2, ax1s, ax2s]:
    ax.spines['top'].set_visible(False)
for ax in [ax1, ax2]:
    ax.legend(loc='upper left')
    ax.set_ylabel('Cumulative 3-month impulse')
ax1.set_title('Fedspeak Hawk vs Dove Score (Tadle 2022) vs Market sentiment')
ax2.set_title('Fedspeak Sentiment (VADER Sentiment Analysis) vs Market sentiment')
plt.text(0.5, 0, 'De Vere Research. Data source from Refinitiv. Believed to be accurate but does not guarantee its accuracy.',
         fontsize=10, ha='center', transform=fig.transFigure)

st.pyplot(plt)

# Display top 20 most traded Fedspeak headlines
st.subheader(f'Top 20 most traded Fedspeak headlines since {start_date}')
st.dataframe(data['Recent_headlines'])
# Select a date for comparison in the scatter plot

st.markdown(
    """
    <hr style="border: 1px solid black">
    """,
    unsafe_allow_html=True
)
comparison_date = st.date_input("Select a date for comparison", datetime.strptime(last_date, "%Y-%m-%d"))


# Plot Market vs Importance
st.subheader(f'Market vs Importance (black: {comparison_date}, blue: {comparison_date.strftime("%b %d")})')
fig, ax2 = plt.subplots(figsize=(7, 7), facecolor='w')
last_impulses = []
for n in offi:
    dt = data['Filtered_df'][n]
    dt = dt.sort_index(ascending=True).resample('1D').sum().ffill()
    dt['Impulse'] = (dt['Price'] / dt['Volume']).rolling(pd.Timedelta('90 days')).sum()
    if not dt.empty:
        last_impulses.append((n, dt['Impulse'].iloc[-1]))
last_impulses.sort(key=lambda x: x[1], reverse=True)
for i, (n, last_impulse) in enumerate(last_impulses):
    dt = data['Filtered_df'][n]
    dt = dt.sort_index(ascending=True).resample('1D').sum().ffill()
    dt['Impulse'] = (dt['Price'] / dt['Volume']).rolling(pd.Timedelta('90 days')).sum()
    dt['Importance'] = abs((dt['Price'] / dt['Volume'])).rolling(pd.Timedelta('90 days')).mean()
    dts = dt[dt.index > comparison_date.strftime("%Y-%m-%d")]
    try:
        if n != 'Powell':
            ax2.scatter(x=dts['Importance'].tail(1).values[0], y=dts['Impulse'].tail(1).values[0], color='black', s=0)
            ax2.scatter(x=dts['Importance'].head(1).values[0], y=dts['Impulse'].head(1).values[0], color='tab:blue', s=0)
            txt = n
            ax2.annotate(txt, (dts['Importance'].tail(1).values[0], dts['Impulse'].tail(1).values[0]), color='black')
            ax2.annotate(txt, (dts['Importance'].head(1).values[0], dts['Impulse'].head(1).values[0]), color='tab:blue')
    except:
        pass

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.axhline(y=0, color='black', linestyle='--', label='+ve dovish and -ve hawkish')
ax2.set_ylabel('Market reaction')
ax2.set_xlabel('Importance (Absolute 3m cumulative market moves on headlines)')
ax2.legend()
plt.text(0.5, 0, 'De Vere Research. Data source from Refinitiv. Believed to be accurate but does not guarantee its accuracy.',
         fontsize=10, ha='center', transform=fig.transFigure)

plt.tight_layout()
plt.show()
st.pyplot(plt)

# Select an official to display their data
selected_official = st.selectbox('Select an official:', offi)
if selected_official in data['Filtered_df']:
    selected_dataframe = data['Filtered_df'][selected_official]
    selected_dataframe = selected_dataframe.sort_index(ascending=True)
    st.dataframe(selected_dataframe.tail(20))
    # Plot Hawk vs Dove score and Sentiment index for the selected Fedspeaker
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), facecolor='w', sharex=True)
    ax1s, ax2s = ax1.twinx(), ax2.twinx()
    selected_dataframe[['Fedspeak_hawk_dove_score']].rolling(window=pd.Timedelta('90 days')).sum().plot(ax=ax1, color='black')
    selected_dataframe[['Fedspeak_sentiment_score']].rolling(window=pd.Timedelta('90 days')).sum().plot(ax=ax2, color='black')
    market_sentiment = pd.DataFrame((selected_dataframe['Price'] / selected_dataframe['Volume']).resample('1D').sum().ffill().rolling(pd.Timedelta('90 days')).sum()).dropna()
    market_sentiment.columns = ['Market reaction']
    market_sentiment.plot(ax=ax1s)
    market_sentiment.plot(ax=ax2s)
    for ax in [ax1, ax2, ax1s, ax2s]:
        ax.spines['top'].set_visible(False)
    for ax in [ax1, ax2]:
        ax.legend(loc='upper left')
        ax.set_ylabel('Cumulative 3-month impulse')
    ax1.set_title(f'{selected_official}: Hawk vs Dove Score (Tadle 2022) vs Market sentiment')
    ax2.set_title(f'{selected_official}: Sentiment (VADER Sentiment Analysis) vs Market sentiment')
    plt.text(0.5, -0, 'De Vere Research. Data source from Refinitiv. Believed to be accurate but does not guarantee its accuracy.',
         fontsize=10, ha='center', transform=fig.transFigure)

    st.pyplot(plt)

# Display data source and credits
st.markdown('''
De Vere Research. Data source from Refinitiv. Believed to be accurate but does not guarantee its accuracy. 
Market reaction to headlines is computed as follows: for each Fedspeak-related headline, calculate the total in volume from t to t+2 and price change from t-3 minute vs. t+2 minute, 
where t is the interval minute in which the headline was released. For headlines with the same timeframe of release, only the first headline released was taken. 
Tadle, R. C. (2022). FOMC minutes sentiments and their impact on financial markets. Journal of Economics and Business.
''')