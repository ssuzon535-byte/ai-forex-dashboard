import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(
    page_title="AI Forex Market Dashboard",
    page_icon="📈",
    layout="wide"
)

SYMBOLS = {
    "Gold (XAU/USD)": "XAU/USD",
    "Silver (XAG/USD)": "XAG/USD",
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "USD/JPY": "USD/JPY",
    "USD/CHF": "USD/CHF",
    "AUD/USD": "AUD/USD",
    "NZD/USD": "NZD/USD",
    "USD/CAD": "USD/CAD",
    "EUR/GBP": "EUR/GBP",
    "EUR/JPY": "EUR/JPY",
    "GBP/JPY": "GBP/JPY",
    "AUD/JPY": "AUD/JPY",
    "EUR/AUD": "EUR/AUD",
    "GBP/AUD": "GBP/AUD",
    "USD/BDT": "USD/BDT",
}

INTERVALS = {
    "1H": "1h",
    "15M": "15min",
    "10M": "10min",
    "5M": "5min",
}

st.title("AI Forex Market Dashboard")
st.caption(
    "Technical analysis dashboard. Signals are estimates, "
    "not guaranteed trading recommendations."
)

with st.sidebar:
    st.header("Market Settings")

    market_name = st.selectbox(
        "Market",
        list(SYMBOLS.keys())
    )
    symbol = SYMBOLS[market_name]

    timeframe_label = st.selectbox(
        "Timeframe",
        list(INTERVALS.keys()),
        index=3
    )
    interval = INTERVALS[timeframe_label]

    candle_count = st.selectbox(
        "Candles to load",
        [100, 150, 200, 300],
        index=2
    )

    st.divider()
    st.subheader("Risk Settings")

    balance = st.number_input(
        "Account Balance (USD)",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    risk_percent = st.number_input(
        "Risk per Trade (%)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )

    rr1 = st.number_input(
        "TP1 Risk:Reward",
        min_value=0.5,
        max_value=10.0,
        value=1.0
    )

    rr2 = st.number_input(
        "TP2 Risk:Reward",
        min_value=0.5,
        max_value=10.0,
        value=2.0
    )

    rr3 = st.number_input(
        "TP3 Risk:Reward",
        min_value=0.5,
        max_value=10.0,
        value=3.0
    )

    mtf_enabled = st.checkbox(
        "Multi-Timeframe Analysis",
        value=False
    )

    st.caption(
        "MTF uses extra API credits. Keep it off "
        "if your credits are limited."
    )


api_key = st.secrets.get("TWELVE_DATA_API_KEY", "")

if not api_key:
    st.error(
        "API key missing. Add TWELVE_DATA_API_KEY "
        "in Streamlit Settings > Secrets."
    )
    st.stop()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles(api_key, symbol, interval, count):
    response = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": count,
            "apikey": api_key,
            "format": "JSON",
        },
        timeout=20,
    )

    response.raise_for_status()
    data = response.json()

    if "values" not in data:
        raise ValueError(
            data.get("message", "No candle data returned.")
        )

    df = pd.DataFrame(data["values"])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    df = (
        df.dropna(
            subset=[
                "datetime",
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return df, data.get("meta", {})


def add_indicators(df):
    d = df.copy()

    for period in [20, 50, 200]:
        d[f"EMA{period}"] = d["close"].ewm(
            span=period,
            adjust=False
        ).mean()

    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + rs)

    previous_close = d["close"].shift(1)

    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - previous_close).abs(),
            (d["low"] - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    d["ATR"] = tr.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    up_move = d["high"].diff()
    down_move = -d["low"].diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0),
        0.0
    )

    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0),
        0.0
    )

    plus_smoothed = plus_dm.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    minus_smoothed = minus_dm.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    atr_safe = d["ATR"].replace(0, np.nan)

    plus_di = 100 * plus_smoothed / atr_safe
    minus_di = 100 * minus_smoothed / atr_safe

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    d["ADX"] = dx.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    d["MACD"] = (
        d["close"].ewm(span=12, adjust=False).mean()
        - d["close"].ewm(span=26, adjust=False).mean()
    )

    d["MACD_Signal"] = d["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    d["BB_Mid"] = d["close"].rolling(20).mean()
    deviation = d["close"].rolling(20).std()

    d["BB_Upper"] = d["BB_Mid"] + 2 * deviation
    d["BB_Lower"] = d["BB_Mid"] - 2 * deviation

    return d


def get_signal(row):
    required = [
        "close",
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI",
        "MACD",
        "MACD_Signal",
        "ADX",
    ]

    if any(pd.isna(row[x]) for x in required):
        return "WAIT", 0, "Insufficient indicator data"

    buy_checks = [
        row["close"] > row["EMA20"],
        row["EMA20"] > row["EMA50"],
        row["EMA50"] > row["EMA200"],
        row["RSI"] > 52,
        row["MACD"] > row["MACD_Signal"],
        row["ADX"] > 18,
    ]

    sell_checks = [
        row["close"] < row["EMA20"],
        row["EMA20"] < row["EMA50"],
        row["EMA50"] < row["EMA200"],
        row["RSI"] < 48,
        row["MACD"] < row["MACD_Signal"],
        row["ADX"] > 18,
    ]

    buy_score = sum(buy_checks)
    sell_score = sum(sell_checks)

    if buy_score >= 5 and buy_score > sell_score:
        return (
            "BUY",
            round(buy_score / 6 * 100),
            "Bullish technical conditions"
        )

    if sell_score >= 5 and sell_score > buy_score:
        return (
            "SELL",
            round(sell_score / 6 * 100),
            "Bearish technical conditions"
        )

    score = max(buy_score, sell_score)

    return (
        "WAIT",
        round(score / 6 * 100),
        "Mixed or weak conditions"
    )


def trade_levels(row, signal, rewards):
    entry = float(row["close"])
    atr = row["ATR"]

    if (
        signal not in ["BUY", "SELL"]
        or pd.isna(atr)
        or atr <= 0
    ):
        return entry, np.nan, [np.nan, np.nan, np.nan]

    distance = float(atr) * 1.5

    if signal == "BUY":
        sl = entry - distance
        targets = [
            entry + distance * reward
            for reward in rewards
        ]
    else:
        sl = entry + distance
        targets = [
            entry - distance * reward
            for reward in rewards
        ]

    return entry, sl, targets


def detect_patterns(df):
    if len(df) < 2:
        return ["Not enough candles"]

    current = df.iloc[-1]
    previous = df.iloc[-2]

    candle_range = max(
        current["high"] - current["low"],
        1e-12
    )

    body = abs(current["close"] - current["open"])

    upper_wick = (
        current["high"]
        - max(current["open"], current["close"])
    )

    lower_wick = (
        min(current["open"], current["close"])
        - current["low"]
    )

    patterns = []

    if body / candle_range < 0.1:
        patterns.append("Doji-like candle")

    if lower_wick > body * 2:
        patterns.append("Possible bullish pin bar")

    if upper_wick > body * 2:
        patterns.append("Possible bearish pin bar")

    if (
        current["close"] > current["open"]
        and previous["close"] < previous["open"]
        and current["close"] >= previous["open"]
        and current["open"] <= previous["close"]
    ):
        patterns.append("Bullish engulfing")

    if (
        current["close"] < current["open"]
        and previous["close"] > previous["open"]
        and current["close"] <= previous["open"]
        and current["open"] >= previous["close"]
    ):
        patterns.append("Bearish engulfing")

    return patterns or ["No common pattern detected"]


def run_backtest(df, reward=2.0):
    trades = []
    position = None

    for i in range(205, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if position is not None:
            if position["side"] == "BUY":
                hit_sl = next_row["low"] <= position["sl"]
                hit_tp = next_row["high"] >= position["tp"]
            else:
                hit_sl = next_row["high"] >= position["sl"]
                hit_tp = next_row["low"] <= position["tp"]

            if hit_sl or hit_tp:
                result = -1.0 if hit_sl else reward

                trades.append({
                    "Entry Time": position["time"],
                    "Side": position["side"],
                    "Entry": position["entry"],
                    "Exit": (
                        position["sl"]
                        if hit_sl
                        else position["tp"]
                    ),
                    "R Result": result,
                })

                position = None

            continue

        signal, _, _ = get_signal(row)

        if (
            signal not in ["BUY", "SELL"]
            or pd.isna(row["ATR"])
            or row["ATR"] <= 0
        ):
            continue

        entry = float(next_row["open"])
        distance = float(row["ATR"]) * 1.5

        if signal == "BUY":
            position = {
                "side": "BUY",
                "entry": entry,
                "sl": entry - distance,
                "tp": entry + distance * reward,
                "time": next_row["datetime"],
            }
        else:
            position = {
                "side": "SELL",
                "entry": entry,
                "sl": entry + distance,
                "tp": entry - distance * reward,
                "time": next_row["datetime"],
            }

    return pd.DataFrame(trades)


try:
    with st.spinner("Loading market data..."):
        raw_df, metadata = fetch_candles(
            api_key,
            symbol,
            interval,
            candle_count
        )

    df = add_indicators(raw_df)

except Exception as error:
    st.error(f"Market data error: {error}")
    st.info(
        "Check the selected pair, timeframe and remaining "
        "Twelve Data API credits."
    )
    st.stop()


if df.empty:
    st.error("No valid candle data returned.")
    st.stop()

last = df.iloc[-1]
signal, quality, reason = get_signal(last)

entry, sl, targets = trade_levels(
    last,
    signal,
    [rr1, rr2, rr3]
)

support = float(df["low"].tail(50).min())
resistance = float(df["high"].tail(50).max())
risk_cash = balance * risk_percent / 100

if pd.isna(last["ATR"]):
    volatility = "Unavailable"
else:
    median_atr = df["ATR"].tail(100).median()

    if pd.isna(median_atr) or median_atr <= 0:
        volatility = "Unavailable"
    elif last["ATR"] > median_atr * 1.5:
        volatility = "High"
    elif last["ATR"] < median_atr * 0.7:
        volatility = "Low"
    else:
        volatility = "Normal"

if last["EMA20"] > last["EMA50"] > last["EMA200"]:
    trend = "Bullish"
elif last["EMA20"] < last["EMA50"] < last["EMA200"]:
    trend = "Bearish"
else:
    trend = "Sideways / Mixed"

st.caption(
    f"Market: {symbol} | Timeframe: {timeframe_label} | "
    f"Latest candle: {last['datetime']}"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Latest Price", f"{last['close']:.5f}")
c2.metric("Signal", signal, f"Quality: {quality}/100")
c3.metric("Trend", trend)
c4.metric("Volatility", volatility)

st.subheader("Trade Setup")

p1, p2, p3, p4, p5 = st.columns(5)

p1.metric("Entry Reference", f"{entry:.5f}")
p2.metric("Stop Loss", f"{sl:.5f}" if pd.notna(sl) else "N/A")
p3.metric("TP1", f"{targets[0]:.5f}" if pd.notna(targets[0]) else "N/A")
p4.metric("TP2", f"{targets[1]:.5f}" if pd.notna(targets[1]) else "N/A")
p5.metric("TP3", f"{targets[2]:.5f}" if pd.notna(targets[2]) else "N/A")

st.write(f"**Reason:** {reason}")
st.write(f"**Support estimate:** {support:.5f}")
st.write(f"**Resistance estimate:** {resistance:.5f}")

st.caption(
    "Entry is the latest candle close for reference only, "
    "not a live execution price. SL/TP are ATR-based estimates."
)

tabs = st.tabs([
    "Chart & Indicators",
    "Price Action",
    "Risk Calculator",
    "Backtest",
    "Journal & Status",
])

with tabs[0]:
    chart = go.Figure(data=[
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
        )
    ])

    for column in ["EMA20", "EMA50", "EMA200"]:
        chart.add_trace(go.Scatter(
            x=df["datetime"],
            y=df[column],
            name=column,
        ))

    chart.add_trace(go.Scatter(
        x=df["datetime"],
        y=df["BB_Upper"],
        name="Bollinger Upper",
    ))

    chart.add_trace(go.Scatter(
        x=df["datetime"],
        y=df["BB_Lower"],
        name="Bollinger Lower",
    ))

    chart.update_layout(
        title=f"{symbol} - {timeframe_label}",
        height=600,
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(chart, use_container_width=True)

    a, b, c, d = st.columns(4)

    a.metric(
        "RSI",
        f"{last['RSI']:.2f}" if pd.notna(last["RSI"]) else "N/A"
    )
    b.metric(
        "ADX",
        f"{last['ADX']:.2f}" if pd.notna(last["ADX"]) else "N/A"
    )
    c.metric(
        "MACD",
        f"{last['MACD']:.5f}" if pd.notna(last["MACD"]) else "N/A"
    )
    d.metric(
        "ATR",
        f"{last['ATR']:.5f}" if pd.notna(last["ATR"]) else "N/A"
    )


with tabs[1]:
    st.subheader("Candle Patterns")

    for pattern in detect_patterns(df):
        st.write(f"- {pattern}")

    recent_high = (
        df["high"].iloc[-21:-1].max()
        if len(df) >= 22
        else df["high"].max()
    )

    recent_low = (
        df["low"].iloc[-21:-1].min()
        if len(df) >= 22
        else df["low"].min()
    )

    if last["close"] > recent_high:
        st.success("Possible bullish range breakout.")
    elif last["close"] < recent_low:
        st.warning("Possible bearish range breakout.")
    else:
        st.info("No recent range breakout detected.")

    st.write(f"Recent range high: {recent_high:.5f}")
    st.write(f"Recent range low: {recent_low:.5f}")

    if signal == "WAIT" or (
        pd.notna(last["ADX"]) and last["ADX"] < 15
    ):
        st.warning(
            "Possible no-trade condition: mixed or weak trend."
        )
    else:
        st.info(
            "The basic filter did not flag a weak-trend condition."
        )

    st.caption(
        "Breakout is a simple price-range check. This version does "
        "not identify real order flow or confirmed institutional "
        "order blocks."
    )


with tabs[2]:
    st.subheader("Risk Calculator")

    r1, r2, r3 = st.columns(3)

    r1.metric("Risk Amount", f"${risk_cash:.2f}")
    r2.metric("Risk Percentage", f"{risk_percent:.2f}%")
    r3.metric("TP2 Reward/Risk", f"1:{rr2:g}")

    st.warning(
        "Automatic lot size is not calculated because contract size, "
        "tick value and broker specifications must be verified."
    )

    st.write(
        "For Exness, check the exact symbol specifications in MT5 "
        "before calculating lot size."
    )


with tabs[3]:
    st.subheader("Historical Backtest")

    reward = st.selectbox(
        "Take Profit Risk/Reward",
        [1.0, 1.5, 2.0, 3.0],
        index=2,
    )

    bt = run_backtest(df, reward)

    if bt.empty:
        st.info(
            "No closed trades found in the loaded candles. "
            "Try loading more candles or another timeframe."
        )
    else:
        wins = int((bt["R Result"] > 0).sum())

        gross_profit = bt.loc[
            bt["R Result"] > 0,
            "R Result"
        ].sum()

        gross_loss = abs(
            bt.loc[
                bt["R Result"] < 0,
                "R Result"
            ].sum()
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else np.nan
        )

        equity = bt["R Result"].cumsum()
        drawdown = (equity.cummax() - equity).max()

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Closed Trades", str(len(bt)))
        m2.metric("Win Rate", f"{wins / len(bt) * 100:.1f}%")
        m3.metric(
            "Profit Factor",
            f"{profit_factor:.2f}" if pd.notna(profit_factor) else "N/A"
        )
        m4.metric("Max Drawdown", f"{drawdown:.2f}R")

        st.metric("Net Result", f"{bt['R Result'].sum():.2f}R")

        st.dataframe(bt.tail(100), use_container_width=True)

        st.download_button(
            "Download Backtest CSV",
            bt.to_csv(index=False).encode("utf-8"),
            file_name="forex_backtest.csv",
            mime="text/csv",
        )

    st.caption(
        "This simplified backtest excludes spread, slippage and "
        "commission. If SL and TP are touched in one candle, SL "
        "is counted first. Results are not guaranteed to match "
        "live trading."
    )


with tabs[4]:
    st.subheader("Trade Journal")

    st.caption(
        "Journal entries are stored in the current Streamlit session, "
        "not in a permanent database."
    )

    with st.form("journal_form"):
        j_market = st.text_input("Market", value=symbol)
        j_side = st.selectbox("Side", ["BUY", "SELL"])

        j_entry = st.number_input(
            "Entry Price",
            min_value=0.0,
            value=float(last["close"])
        )

        j_exit = st.number_input(
            "Exit Price",
            min_value=0.0,
            value=float(last["close"])
        )

        j_pl = st.number_input(
            "Profit/Loss USD",
            value=0.0
        )

        j_notes = st.text_input("Notes")

        submitted = st.form_submit_button("Add Journal Entry")

    if "journal_rows" not in st.session_state:
        st.session_state.journal_rows = []

    if submitted:
        st.session_state.journal_rows.append({
            "Time UTC": datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "Market": j_market,
            "Side": j_side,
            "Entry": j_entry,
            "Exit": j_exit,
            "P/L USD": j_pl,
            "Notes": j_notes,
        })

    if st.session_state.journal_rows:
        journal = pd.DataFrame(
            st.session_state.journal_rows
        )

        st.dataframe(
            journal,
            use_container_width=True
        )

        st.metric(
            "Journal Net P/L",
            f"${journal['P/L USD'].sum():.2f}"
        )

        st.download_button(
            "Download Journal CSV",
            journal.to_csv(index=False).encode("utf-8"),
            file_name="forex
