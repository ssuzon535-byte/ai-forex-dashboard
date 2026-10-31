import streamlit as st

st.set_page_config(
    page_title="AI Forex Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Forex Intelligence")
st.caption("Exness MT5 | AI Market Analysis")

st.sidebar.header("Market Settings")

market = st.sidebar.selectbox(
    "Select Market",
    ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY",
     "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"]
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1H", "15M", "10M", "5M"]
)

st.subheader(f"{market} Market Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Live Price", "Waiting")
col2.metric("Trend", "Pending")
col3.metric("RSI", "--")
col4.metric("ADX", "--")

st.divider()

st.subheader("Multi-Timeframe Analysis")

st.table({
    "Timeframe": ["1H", "15M", "10M", "5M"],
    "Trend": ["Waiting", "Waiting", "Waiting", "Waiting"]
})

st.divider()

st.subheader("AI Trading Signal")

st.info("Market data connection is not added yet.")

st.write("Signal: Waiting")
st.write("Entry: --")
st.write("Stop Loss: --")
st.write("Take Profit: --")

st.divider()

st.subheader("Risk Management")
st.write("Lot Size: --")
st.write("Risk Reward: --")

st.caption("Version 0.1 | Dashboard Prototype")
