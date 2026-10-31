import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import datetime

# =====================================
# PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="AI Forex Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Forex Intelligence")
st.caption("Exness MT5 | Forex & Gold Market Analysis")

# =====================================
# API KEY
# =====================================

try:
    API_KEY = st.secrets["TWELVE_DATA_API_KEY"]
except Exception:
    API_KEY = ""

# =====================================
# MARKET SETTINGS
# =====================================

MARKETS = {
    "XAUUSD Gold": "XAU/USD",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "USDCHF": "USD/CHF",
    "AUDUSD": "AUD/USD",
    "NZDUSD": "NZD/USD",
    "USDCAD": "USD/CAD",
    "EURJPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY",
    "EURGBP": "EUR/GBP",
    "AUDJPY": "AUD/JPY"
}

TIMEFRAMES = {
    "1H": "1h",
    "15M": "15min",
    "10M": "10min",
    "5M": "5min"
}

st.sidebar.header("⚙️ Market Settings")

market_name = st.sidebar.selectbox(
    "Select Market",
    list(MARKETS.keys())
)

symbol = MARKETS[market_name]

timeframe_name = st.sidebar.selectbox(
    "Select Timeframe",
    list(TIMEFRAMES.keys()),
    index=3
)

interval = TIMEFRAMES[timeframe_name]

st.sidebar.divider()

# =====================================
# RISK SETTINGS
# =====================================

st.sidebar.header("Risk Management")

balance = st.sidebar.number_input(
    "Account Balance (USD)",
    min_value=10.0,
    value=1000.0,
    step=100.0
)

risk_percent = st.sidebar.slider(
    "Risk Per Trade (%)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5
)

risk_reward = st.sidebar.selectbox(
    "Risk Reward Ratio",
    [1.0, 1.5, 2.0, 2.5, 3.0],
    index=2
)

# =====================================
# DATA FETCHING
# =====================================

@st.cache_data(ttl=60)
def get_market_data(symbol, interval, api_key):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 200,
        "apikey": api_key
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    result = response.json()

    if "values" not in result:
        raise ValueError(
            result.get("message", "Market data unavailable")
        )

    df = pd.DataFrame(result["values"])

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.dropna()

    df = df.sort_values("datetime").reset_index(drop=True)

    if len(df) < 60:
        raise ValueError("Not enough candle data")

    return df


# =====================================
# TECHNICAL INDICATORS
# =====================================

def calculate_indicators(df):

    df = df.copy()

    # EMA
    df["EMA20"] = df["close"].ewm(
        span=20, adjust=False
    ).mean()

    df["EMA50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    df["EMA200"] = df["close"].ewm(
        span=200, adjust=False
    ).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    # ADX
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0),
        0
    )

    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0),
        0
    )

    atr_safe = df["ATR"].replace(0, np.nan)

    plus_di = 100 * (
        plus_dm.ewm(alpha=1/14, adjust=False).mean()
        / atr_safe
    )

    minus_di = 100 * (
        minus_dm.ewm(alpha=1/14, adjust=False).mean()
        / atr_safe
    )

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    df["ADX"] = dx.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    return df


# =====================================
# TREND ANALYSIS
# =====================================

def get_trend(row):

    if row["EMA20"] > row["EMA50"] > row["EMA200"]:
        return "Bullish"

    if row["EMA20"] < row["EMA50"] < row["EMA200"]:
        return "Bearish"

    return "Sideways"


# =====================================
# SIGNAL ENGINE
# =====================================

def get_signal(df):

    row = df.iloc[-2]

    previous = df.iloc[-3]

    bullish = (
        row["EMA20"] > row["EMA50"]
        and row["close"] > row["EMA20"]
        and row["RSI"] > 50
        and row["RSI"] < 70
        and row["ADX"] >= 20
        and row["close"] > row["open"]
    )

    bearish = (
        row["EMA20"] < row["EMA50"]
        and row["close"] < row["EMA20"]
        and row["RSI"] < 50
        and row["RSI"] > 30
        and row["ADX"] >= 20
        and row["close"] < row["open"]
    )

    if bullish:
        return "BUY"

    if bearish:
        return "SELL"

    return "WAIT"


# =====================================
# LOAD MARKET
# =====================================

if not API_KEY:

    st.error(
        "Twelve Data API Key not found. "
        "Check Streamlit App Settings > Secrets."
    )

    st.stop()

try:

    with st.spinner("Loading market data..."):

        raw_df = get_market_data(
            symbol,
            interval,
            API_KEY
        )

        df = calculate_indicators(raw_df)

except Exception as error:

    st.error(f"Market data error: {error}")

    st.info(
        "Check your API Key, market symbol, "
        "timeframe and API plan."
    )

    st.stop()


# Use last completed candle for signal calculations
row = df.iloc[-2]

price = float(row["close"])

trend = get_trend(row)

signal = get_signal(df)

rsi = float(row["RSI"])

adx = float(row["ADX"])

atr = float(row["ATR"])

# =====================================
# ENTRY / SL / TP
# =====================================

entry = price

stop_loss = None
tp1 = None
tp2 = None
tp3 = None

if signal == "BUY":

    stop_loss = entry - (atr * 1.5)

    risk_distance = entry - stop_loss

    tp1 = entry + risk_distance

    tp2 = entry + (risk_distance * 2)

    tp3 = entry + (risk_distance * 3)

elif signal == "SELL":

    stop_loss = entry + (atr * 1.5)

    risk_distance = stop_loss - entry

    tp1 = entry - risk_distance

    tp2 = entry - (risk_distance * 2)

    tp3 = entry - (risk_distance * 3)


# =====================================
# MARKET OVERVIEW
# =====================================

st.subheader(f"{market_name} Market Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Last Completed Price",
    f"{price:.5f}"
)

col2.metric(
    "Trend",
    trend
)

col3.metric(
    "RSI",
    f"{rsi:.2f}"
)

col4.metric(
    "ADX",
    f"{adx:.2f}"
)

st.caption(
    f"Last completed candle: {row['datetime']}"
)

st.divider()

# =====================================
# MULTI TIMEFRAME ANALYSIS
# =====================================

st.subheader("Multi-Timeframe Analysis")

st.caption(
    "Enable this to request separate data for all four timeframes. "
    "This uses additional API credits."
)

enable_mtf = st.checkbox(
    "Load 1H, 15M, 10M and 5M analysis",
    value=False
)

if enable_mtf:

    mtf_results = []

    with st.spinner("Checking multiple timeframes..."):

        for name, tf in TIMEFRAMES.items():

            try:

                mtf_raw = get_market_data(
                    symbol,
                    tf,
                    API_KEY
                )

                mtf_df = calculate_indicators(mtf_raw)

                mtf_row = mtf_df.iloc[-2]

                mtf_results.append({
                    "Timeframe": name,
                    "Trend": get_trend(mtf_row),
                    "RSI": round(float(mtf_row["RSI"]), 2),
                    "ADX": round(float(mtf_row["ADX"]), 2)
                })

            except Exception:

                mtf_results.append({
                    "Timeframe": name,
                    "Trend": "Unavailable",
                    "RSI": "--",
                    "ADX": "--"
                })

    st.dataframe(
        pd.DataFrame(mtf_results),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Multi-timeframe analysis is currently switched off "
        "to save API credits."
    )

st.divider()

# =====================================
# SUPPORT & RESISTANCE
# =====================================

st.subheader("Support & Resistance")

support = float(
    df["low"].iloc[-51:-1].min()
)

resistance = float(
    df["high"].iloc[-51:-1].max()
)

c1, c2 = st.columns(2)

c1.metric("Support", f"{support:.5f}")

c2.metric("Resistance", f"{resistance:.5f}")

st.divider()

# =====================================
# TRADING SIGNAL
# =====================================

st.subheader("AI Trading Signal")

if signal == "BUY":

    st.success("🟢 BUY SIGNAL")

elif signal == "SELL":

    st.error("🔴 SELL SIGNAL")

else:

    st.warning("⏳ WAIT - No clear signal")

st.write(f"**Entry Price:** {entry:.5f}")

if stop_loss is not None:

    st.write(f"**Stop Loss:** {stop_loss:.5f}")

    st.write(f"**Take Profit 1:** {tp1:.5f}")

    st.write(f"**Take Profit 2:** {tp2:.5f}")

    st.write(f"**Take Profit 3:** {tp3:.5f}")

else:

    st.write("Stop Loss: --")

    st.write("Take Profit 1: --")

    st.write("Take Profit 2: --")

    st.write("Take Profit 3: --")

st.caption(
    "Signals use technical rules and completed candles. "
    "They are not guaranteed predictions."
)

st.divider()

# =====================================
# PRICE CHART
# =====================================

st.subheader("Market Chart")

chart = go.Figure()

chart.add_trace(
    go.Candlestick(
        x=df["datetime"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price"
    )
)

chart.add_trace(
    go.Scatter(
        x=df["datetime"],
        y=df["EMA20"],
        name="EMA 20"
    )
)

chart.add_trace(
    go.Scatter(
        x=df["datetime"],
        y=df["EMA50"],
        name="EMA 50"
    )
)

if signal == "BUY":

    chart.add_annotation(
        x=row["datetime"],
        y=price,
        text="BUY ↑",
        showarrow=True,
        arrowhead=2
    )

elif signal == "SELL":

    chart.add_annotation(
        x=row["datetime"],
        y=price,
        text="SELL ↓",
        showarrow=True,
        arrowhead=2
    )

chart.update_layout(
    height=550,
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)

st.plotly_chart(
    chart,
    use_container_width=True
)

st.divider()

# =====================================
# RISK MANAGEMENT
# =====================================

st.subheader("Risk Management")

risk_amount = balance * (risk_percent / 100)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Account Balance",
    f"${balance:.2f}"
)

c2.metric(
    "Risk Amount",
    f"${risk_amount:.2f}"
)

c3.metric(
    "Risk Per Trade",
    f"{risk_percent:.1f}%"
)

if stop_loss is not None:

    stop_distance = abs(entry - stop_loss)

    st.write(
        f"**Price Risk Distance:** {stop_distance:.5f}"
    )

    st.write(
        "**Lot Size:** Broker contract specifications "
        "are required for an accurate calculation."
    )

    st.write(
        f"**Selected Risk Reward:** 1:{risk_reward}"
    )

else:

    st.info(
        "Lot size and trade risk calculation will appear "
        "when a valid signal is available."
    )

st.divider()

# =====================================
# MARKET STATUS
# =====================================

st.subheader("Market Analysis Summary")

st.write(f"Market: {market_name}")

st.write(f"Timeframe: {timeframe_name}")

st.write(f"Trend: {trend}")

st.write(f"Signal: {signal}")

st.write(
    f"RSI Status: "
    f"{'Overbought' if rsi >= 70 else 'Oversold' if rsi <= 30 else 'Neutral'}"
)

st.write(
    f"ADX Status: "
    f"{'Strong Trend' if adx >= 25 else 'Weak Trend'}"
)

st.caption(
    "Version 1.0 | AI Forex Intelligence | "
    "Market data and technical analysis"
)

st.caption(
    "Educational analysis only. Always verify prices "
    "and trade risk on Exness MT5 before trading."
)
