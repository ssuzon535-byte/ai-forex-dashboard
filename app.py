
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(
    page_title="AI Forex Market Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Forex Market Analysis Dashboard")
st.caption("XAUUSD & Major Forex Pairs | Technical Analysis")

try:
    API_KEY = st.secrets["TWELVE_DATA_API_KEY"]
except Exception:
    API_KEY = ""

SYMBOLS = [
    "XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY",
    "USD/CHF", "AUD/USD", "NZD/USD", "USD/CAD",
    "EUR/JPY", "GBP/JPY", "EUR/GBP", "AUD/JPY"
]

TIMEFRAMES = {
    "1H": "1h",
    "15M": "15min",
    "10M": "10min",
    "5M": "5min"
}


@st.cache_data(ttl=60)
def get_data(symbol, interval, api_key):
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 500,
        "apikey": api_key,
        "timezone": "UTC"
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    result = response.json()

    if "values" not in result:
        raise ValueError(
            result.get("message", "No market data returned")
        )

    df = pd.DataFrame(result["values"])

    df["datetime"] = pd.to_datetime(
        df["datetime"], utc=True
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def indicators(df):
    d = df.copy()

    close = d["close"]

    d["EMA20"] = close.ewm(span=20).mean()
    d["EMA50"] = close.ewm(span=50).mean()
    d["EMA200"] = close.ewm(span=200).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    d["RSI"] = (
        100 - (100 / (1 + rs))
    ).fillna(50)

    previous = close.shift(1)

    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - previous).abs(),
        (d["low"] - previous).abs()
    ], axis=1).max(axis=1)

    d["ATR"] = tr.ewm(
        alpha=1/14, adjust=False
    ).mean()

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()

    d["MACD"] = ema12 - ema26
    d["MACD_SIGNAL"] = d["MACD"].ewm(span=9).mean()

    d["MACD_HIST"] = (
        d["MACD"] - d["MACD_SIGNAL"]
    )

    d["BB_MID"] = close.rolling(20).mean()
    std = close.rolling(20).std()

    d["BB_UPPER"] = d["BB_MID"] + 2 * std
    d["BB_LOWER"] = d["BB_MID"] - 2 * std

    d["ATR_PERCENT"] = (
        d["ATR"] / close
    ) * 100

    up = d["high"].diff()
    down = -d["low"].diff()

    plus_dm = up.where(
        (up > down) & (up > 0), 0
    )

    minus_dm = down.where(
        (down > up) & (down > 0), 0
    )

    plus_di = (
        100 * plus_dm.ewm(alpha=1/14).mean()
        / d["ATR"].replace(0, np.nan)
    )

    minus_di = (
        100 * minus_dm.ewm(alpha=1/14).mean()
        / d["ATR"].replace(0, np.nan)
    )

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    d["ADX"] = dx.ewm(alpha=1/14).mean().fillna(0)

    return d


def candle_pattern(d):
    if len(d) < 2:
        return "Insufficient data"

    a = d.iloc[-1]
    p = d.iloc[-2]

    body = abs(a["close"] - a["open"])
    total = max(a["high"] - a["low"], 1e-10)

    upper = a["high"] - max(a["open"], a["close"])
    lower = min(a["open"], a["close"]) - a["low"]

    if body / total < 0.10:
        return "Doji"

    if (
        lower > body * 2
        and a["close"] > a["open"]
    ):
        return "Bullish Pin Bar"

    if (
        upper > body * 2
        and a["close"] < a["open"]
    ):
        return "Bearish Pin Bar"

    if (
        a["close"] > a["open"]
        and p["close"] < p["open"]
        and a["close"] > p["open"]
    ):
        return "Bullish Engulfing"

    if (
        a["close"] < a["open"]
        and p["close"] > p["open"]
        and a["close"] < p["open"]
    ):
        return "Bearish Engulfing"

    return "No clear pattern"


def volatility(d):
    values = d["ATR_PERCENT"].dropna()

    if len(values) < 30:
        return "Insufficient data"

    current = values.iloc[-1]
    average = values.tail(100).median()

    if average == 0:
        return "Unavailable"

    ratio = current / average

    if ratio >= 1.7:
        return "Very High"
    elif ratio >= 1.25:
        return "High"
    elif ratio <= 0.65:
        return "Low"

    return "Normal"


def session_name():
    hour = datetime.now(timezone.utc).hour

    sessions = []

    if 0 <= hour < 9:
        sessions.append("Asian")

    if 7 <= hour < 16:
        sessions.append("London")

    if 13 <= hour < 22:
        sessions.append("New York")

    return ", ".join(sessions) if sessions else "Between sessions"


def market_analysis(d, rr):
    last = d.iloc[-1]

    bullish = (
        last["EMA20"] > last["EMA50"]
        > last["EMA200"]
    )

    bearish = (
        last["EMA20"] < last["EMA50"]
        < last["EMA200"]
    )

    buy_score = 0
    sell_score = 0

    if bullish:
        buy_score += 2

    if bearish:
        sell_score += 2

    if last["close"] > last["EMA20"]:
        buy_score += 1

    if last["close"] < last["EMA20"]:
        sell_score += 1

    if last["RSI"] > 52:
        buy_score += 1

    if last["RSI"] < 48:
        sell_score += 1

    if last["MACD_HIST"] > 0:
        buy_score += 1

    if last["MACD_HIST"] < 0:
        sell_score += 1

    if last["ADX"] >= 20:
        if last["EMA20"] > last["EMA50"]:
            buy_score += 1
        elif last["EMA20"] < last["EMA50"]:
            sell_score += 1

    signal = "WAIT"
    entry = None
    sl = None
    targets = []
    score = max(buy_score, sell_score) * 10

    if (
        buy_score >= 5
        and buy_score >= sell_score + 2
        and last["close"] > last["EMA50"]
    ):
        signal = "BUY"
        entry = float(last["close"])
        sl = entry - 1.5 * float(last["ATR"])
        risk = entry - sl

        targets = [
            entry + risk * rr,
            entry + risk * rr * 2,
            entry + risk * rr * 3
        ]

        score = min(95, 50 + (buy_score - sell_score) * 8)

    elif (
        sell_score >= 5
        and sell_score >= buy_score + 2
        and last["close"] < last["EMA50"]
    ):
        signal = "SELL"
        entry = float(last["close"])
        sl = entry + 1.5 * float(last["ATR"])
        risk = sl - entry

        targets = [
            entry - risk * rr,
            entry - risk * rr * 2,
            entry - risk * rr * 3
        ]

        score = min(95, 50 + (sell_score - buy_score) * 8)

    trend = (
        "Bullish" if bullish
        else "Bearish" if bearish
        else "Sideways / Mixed"
    )

    return signal, trend, entry, sl, targets, score


def backtest(d):
    x = d.copy()

    future = x["close"].shift(-5)

    buy = (
        (x["EMA20"] > x["EMA50"])
        & (x["EMA50"] > x["EMA200"])
        & (x["RSI"] > 52)
        & (x["MACD_HIST"] > 0)
    )

    sell = (
        (x["EMA20"] < x["EMA50"])
        & (x["EMA50"] < x["EMA200"])
        & (x["RSI"] < 48)
        & (x["MACD_HIST"] < 0)
    )

    valid = future.notna()
    signals = (buy | sell) & valid

    selected = x.loc[signals]

    if selected.empty:
        return 0, 0, 0

    directions = np.where(buy.loc[signals], 1, -1)

    moves = (
        future.loc[signals].values
        - selected["close"].values
    ) * directions

    wins = int((moves > 0).sum())

    return len(selected), wins, wins / len(selected) * 100


with st.sidebar:
    st.header("Market Settings")

    symbol = st.selectbox("Market", SYMBOLS)

    timeframe = st.selectbox(
        "Timeframe", list(TIMEFRAMES.keys()),
        index=3
    )

    rr = st.selectbox(
        "Risk Reward", [1, 1.5, 2, 2.5, 3],
        index=2
    )

    risk_amount = st.number_input(
        "Risk Amount", min_value=0.0,
        value=10.0
    )

    mtf = st.checkbox(
        "Multi-Timeframe Analysis",
        value=False
    )

    st.caption(
        "MTF uses extra API credits. Keep OFF when credits are low."
    )

if not API_KEY:
    st.error(
        "TWELVE_DATA_API_KEY missing from Streamlit Secrets."
    )
    st.stop()

try:
    with st.spinner("Loading market data..."):
        raw = get_data(
            symbol, TIMEFRAMES[timeframe], API_KEY
        )

except Exception as e:
    st.error(f"Market data error: {e}")
    st.stop()

if len(raw) < 210:
    st.warning(
        "Less than 210 candles received. EMA200 analysis may be limited."
    )

df = indicators(raw)
last = df.iloc[-1]

signal, trend, entry, sl, targets, score = market_analysis(
    df, rr
)

pattern = candle_pattern(df)
vol = volatility(df)

st.subheader(f"{symbol} | {timeframe} Market Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Price", f"{last['close']:.5f}")
c2.metric("Trend", trend)
c3.metric("RSI", f"{last['RSI']:.2f}")
c4.metric("ADX", f"{last['ADX']:.2f}")

st.caption(
    "Latest candle UTC: "
    + last["datetime"].strftime("%Y-%m-%d %H:%M")
)

st.divider()
st.subheader("Trade Signal")

s1, s2, s3 = st.columns(3)

s1.metric("Signal", signal)
s2.metric("Signal Quality", f"{score}/100")
s3.metric("Volatility", vol)

if signal != "WAIT":
    a, b, c, e, f = st.columns(5)

    a.metric("Entry", f"{entry:.5f}")
    b.metric("Stop Loss", f"{sl:.5f}")
    c.metric("TP1", f"{targets[0]:.5f}")
    e.metric("TP2", f"{targets[1]:.5f}")
    f.metric("TP3", f"{targets[2]:.5f}")

    st.caption(
        f"Risk amount entered: {risk_amount:.2f}. "
        "Lot size is not calculated without verified broker contract data."
    )
else:
    st.info("WAIT: No qualifying setup. Avoid forcing a trade.")

st.divider()

left, middle, right = st.columns(3)

with left:
    st.subheader("Indicators")
    st.write(f"EMA20: {last['EMA20']:.5f}")
    st.write(f"EMA50: {last['EMA50']:.5f}")
    st.write(f"EMA200: {last['EMA200']:.5f}")
    st.write(f"ATR: {last['ATR']:.5f}")
    st.write(f"MACD: {last['MACD']:.5f}")
    st.write(f"Bollinger Upper: {last['BB_UPPER']:.5f}")
    st.write(f"Bollinger Lower: {last['BB_LOWER']:.5f}")

with middle:
    st.subheader("Candle & Structure")
    st.write(f"Pattern: {pattern}")

    recent = df.tail(50)

    support = recent["low"].min()
    resistance = recent["high"].max()

    st.write(f"Support: {support:.5f}")
    st.write(f"Resistance: {resistance:.5f}")

    if last["close"] > resistance:
        st.write("Possible bullish breakout")
    elif last["close"] < support:
        st.write("Possible bearish breakout")
    else:
        st.write("No clear 50-candle breakout")

with right:
    st.subheader("Trading Session")
    st.write(session_name())
    st.write(f"Volatility: {vol}")

    if last["ADX"] < 15:
        st.warning("Weak trend / choppy market")

    if last["RSI"] >= 75 or last["RSI"] <= 25:
        st.warning("RSI extreme: reversal risk")

    if vol == "Very High":
        st.warning("Very high volatility")

# Reversal warning
st.subheader("Trend Reversal Warning")

previous = df.iloc[-2]

if (
    last["EMA20"] > last["EMA50"]
    and last["MACD_HIST"] < 0
):
    st.warning(
        "Bullish EMA alignment but negative MACD momentum."
    )

elif (
    last["EMA20"] < last["EMA50"]
    and last["MACD_HIST"] > 0
):
    st.warning(
        "Bearish EMA alignment but positive MACD momentum."
    )

elif (
    previous["RSI"] > 70
    and last["RSI"] < previous["RSI"]
):
    st.warning("RSI turning down from an extreme.")

else:
    st.success("No reversal warning from current rules.")

# No trade zone
st.subheader("No Trade Zone")

reasons = []

if last["ADX"] < 15:
    reasons.append("Weak trend")

if vol == "Very High":
    reasons.append("Very high volatility")

if pattern == "Doji":
    reasons.append("Doji candle")

if signal == "WAIT":
    reasons.append("No valid signal")

if reasons:
    st.warning(", ".join(reasons))
else:
    st.info("No current no-trade filter triggered.")

# Chart
st.divider()
st.subheader("Price Chart")

view = df.tail(180)

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=view["datetime"],
    open=view["open"],
    high=view["high"],
    low=view["low"],
    close=view["close"],
    name="Price"
))

for column, name in [
    ("EMA20", "EMA 20"),
    ("EMA50", "EMA 50"),
    ("EMA200", "EMA 200")
]:
    fig.add_trace(go.Scatter(
        x=view["datetime"],
        y=view[column],
        name=name
    ))

if signal != "WAIT":
    fig.add_hline(
        y=entry, line_dash="dot",
        annotation_text="Entry"
    )

    fig.add_hline(
        y=sl, line_dash="dash",
        annotation_text="SL"
    )

    for i, target in enumerate(targets, 1):
        fig.add_hline(
            y=target, line_dash="dash",
            annotation_text=f"TP{i}"
        )

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# MTF
st.divider()
st.subheader("Multi-Timeframe Analysis")

if mtf:
    st.warning(
        "This will make additional API calls and consume credits."
    )

    rows = []

    try:
        for label, interval in TIMEFRAMES.items():

            if label == timeframe:
                mdf = df
            else:
                mraw = get_data(symbol, interval, API_KEY)
                mdf = indicators(mraw)

            m = mdf.iloc[-1]

            mt = (
                "Bullish"
                if m["EMA20"] > m["EMA50"] > m["EMA200"]
                else "Bearish"
                if m["EMA20"] < m["EMA50"] < m["EMA200"]
                else "Sideways"
            )

            rows.append({
                "Timeframe": label,
                "Trend": mt,
                "RSI": round(m["RSI"], 2)
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"MTF unavailable: {e}")

else:
    st.caption("MTF disabled to save API credits.")

# Backtesting
st.divider()
st.subheader("Historical Directional Check")

st.caption(
    "This simplified test checks direction after five candles. "
    "It is not a complete profit backtest and excludes spread, "
    "commission and slippage."
)

trades, wins, rate = backtest(df)

b1, b2, b3 = st.columns(3)

b1.metric("Signals", trades)
b2.metric("Correct Direction", wins)
b3.metric("Directional Hit Rate", f"{rate:.1f}%")

# Trade journal
st.divider()
st.subheader("Trade Journal")

if "journal" not in st.session_state:
    st.session_state.journal = []

with st.form("trade_form"):

    j_market = st.selectbox(
        "Market", SYMBOLS, key="journal_market"
    )

    j_direction = st.selectbox(
        "Direction", ["BUY", "SELL"]
    )

    j_result = st.selectbox(
        "Result", ["Win", "Loss", "Break-even", "Open"]
    )

    j_entry = st.number_input(
        "Entry Price", min_value=0.0,
        value=float(last["close"])
    )

    j_note = st.text_input("Trade Note")

    save = st.form_submit_button("Save Trade")

    if save:
        st.session_state.journal.append({
            "Time UTC": datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%d %H:%M"),
            "Market": j_market,
            "Direction": j_direction,
            "Result": j_result,
            "Entry": j_entry,
            "Note": j_note
        })

        st.success("Trade saved in this session.")

if st.session_state.journal:
    journal_df = pd.DataFrame(
        st.session_state.journal
    )

    st.dataframe(
        journal_df,
        use_container_width=True
    )

    st.download_button(
        "Download Journal CSV",
        journal_df.to_csv(index=False),
        "trade_journal.csv",
        "text/csv"
    )

# Data connection status
st.divider()
st.subheader("Data Connection Status")

d1, d2, d3 = st.columns(3)

d1.metric("Market Data", "Connected")
d2.metric("Latest Candle", last["datetime"].strftime("%H:%M UTC"))
d3.metric("API Credit Balance", "Unavailable")

st.info(
    "Exact API credits must be checked in your Twelve Data account. "
    "The candle response does not provide a reliable remaining-credit balance."
)

st.warning(
    "Live news, live broker spread, push alerts and MT5 execution "
    "are not connected. No unavailable live data is fabricated."
)

with st.expander("Dashboard Limitations"):
    st.write("""
    - Signals use technical rules, not a trained AI prediction model.
    - Entry uses the latest returned candle close.
    - SL uses 1.5 ATR.
    - Support, resistance and structure are simplified approximations.
    - Session times use approximate UTC windows.
    - No signal guarantees profit or accuracy.
    - Confirm spread, news and execution on your broker.
    """)

st.caption(
    "Educational analysis only. Forex and CFD trading involves substantial risk."
)
