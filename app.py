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
    "EUR/AUD": "EUR/AUD",
    "EUR/CHF": "EUR/CHF",
    "GBP/JPY": "GBP/JPY",
    "GBP/AUD": "GBP/AUD",
    "GBP/CHF": "GBP/CHF",
    "AUD/JPY": "AUD/JPY",
    "AUD/CAD": "AUD/CAD",
    "NZD/JPY": "NZD/JPY",
    "CAD/JPY": "CAD/JPY",
    "CHF/JPY": "CHF/JPY",
    "USD/BDT (check provider support)": "USD/BDT",
}

INTERVALS = {
    "1H": "1h",
    "15M": "15min",
    "10M": "10min",
    "5M": "5min",
}

st.title("AI Forex Market Dashboard")
st.caption(
    "Technical analysis dashboard | Signals are estimates, "
    "not guaranteed trade recommendations."
)

with st.sidebar:
    st.header("Market Settings")

    selected_name = st.selectbox(
        "Currency Pair",
        list(SYMBOLS.keys()),
        index=0
    )
    symbol = SYMBOLS[selected_name]

    timeframe_label = st.selectbox(
        "Timeframe",
        list(INTERVALS.keys()),
        index=3
    )
    interval = INTERVALS[timeframe_label]

    outputsize = st.select_slider(
        "Candles to load",
        options=[100, 150, 200, 300],
        value=200
    )

    st.divider()
    st.subheader("Risk Settings")

    account_balance = st.number_input(
        "Account Balance (USD)",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    risk_pct = st.number_input(
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
        value=1.0,
        step=0.5
    )

    rr2 = st.number_input(
        "TP2 Risk:Reward",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5
    )

    rr3 = st.number_input(
        "TP3 Risk:Reward",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.5
    )

    st.divider()

    mtf_enabled = st.checkbox(
        "Multi-Timeframe Confirmation (uses extra API credits)",
        value=False
    )

    st.caption(
        "Keep MTF off if your data API credits are limited."
    )


api_key = st.secrets.get("TWELVE_DATA_API_KEY", "")

if not api_key:
    st.error(
        "Missing TWELVE_DATA_API_KEY. Add it in Streamlit "
        "Settings → Secrets. Never place your key in public code."
    )
    st.stop()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles(api_key, symbol, interval, outputsize):
    url = "https://api.twelvedata.com/time_series"

    response = requests.get(
        url,
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": api_key,
            "format": "JSON",
        },
        timeout=20,
    )

    response.raise_for_status()
    data = response.json()

    if data.get("status") == "error" or "values" not in data:
        message = data.get(
            "message",
            "No candle data returned by the provider."
        )
        raise ValueError(message)

    df = pd.DataFrame(data["values"])

    required = [
        "datetime",
        "open",
        "high",
        "low",
        "close"
    ]

    if not all(col in df.columns for col in required):
        raise ValueError(
            "Provider response is missing OHLC candle fields."
        )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    df = (
        df.dropna(subset=required)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return df, data.get("meta", {})


def add_indicators(df):
    d = df.copy()

    d["EMA20"] = d["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    d["EMA50"] = d["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    d["EMA200"] = d["close"].ewm(
        span=200,
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

    d["RSI"] = 100 - (100 / (1 + rs))

    prev_close = d["close"].shift(1)

    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    d["ATR"] = tr.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    # Directional movement and ADX
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

    atr_safe = d["ATR"].replace(0, np.nan)

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean()
        / atr_safe
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean()
        / atr_safe
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
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

    d["MACD_signal"] = d["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    d["BB_mid"] = d["close"].rolling(20).mean()

    bb_std = d["close"].rolling(20).std()

    d["BB_upper"] = d["BB_mid"] + 2 * bb_std
    d["BB_lower"] = d["BB_mid"] - 2 * bb_std

    if "volume" in d.columns:
        d["volume_avg"] = d["volume"].rolling(20).mean()
    else:
        d["volume_avg"] = np.nan

    return d


def get_signal(row):
    values = [
        row["close"],
        row["EMA20"],
        row["EMA50"],
        row["EMA200"],
        row["RSI"],
        row["MACD"],
        row["MACD_signal"],
        row["ADX"],
    ]

    if not all(pd.notna(value) for value in values):
        return (
            "WAIT",
            0,
            "Not enough valid indicator data"
        )

    buy_checks = [
        row["close"] > row["EMA20"],
        row["EMA20"] > row["EMA50"],
        row["EMA50"] > row["EMA200"],
        row["RSI"] > 52,
        row["MACD"] > row["MACD_signal"],
        row["ADX"] > 18,
    ]

    sell_checks = [
        row["close"] < row["EMA20"],
        row["EMA20"] < row["EMA50"],
        row["EMA50"] < row["EMA200"],
        row["RSI"] < 48,
        row["MACD"] < row["MACD_signal"],
        row["ADX"] > 18,
    ]

    buy_score = sum(buy_checks)
    sell_score = sum(sell_checks)

    if buy_score >= 5 and buy_score > sell_score:
        return (
            "BUY",
            int(round(buy_score / 6 * 100)),
            "Bullish technical conditions"
        )

    if sell_score >= 5 and sell_score > buy_score:
        return (
            "SELL",
            int(round(sell_score / 6 * 100)),
            "Bearish technical conditions"
        )

    return (
        "WAIT",
        int(round(max(buy_score, sell_score) / 6 * 100)),
        "Conditions are mixed or trend strength is limited"
    )


def make_trade_levels(row, signal, rr_values):
    entry = float(row["close"])

    atr = (
        float(row["ATR"])
        if pd.notna(row["ATR"]) and row["ATR"] > 0
        else np.nan
    )

    if signal == "BUY" and pd.notna(atr):
        sl = entry - 1.5 * atr
        risk = entry - sl

        targets = [
            entry + risk * reward
            for reward in rr_values
        ]

        return entry, sl, targets

    if signal == "SELL" and pd.notna(atr):
        sl = entry + 1.5 * atr
        risk = sl - entry

        targets = [
            entry - risk * reward
            for reward in rr_values
        ]

        return entry, sl, targets

    return entry, np.nan, [np.nan, np.nan, np.nan]


def pattern_labels(df):
    if len(df) < 2:
        return ["Not enough candles"]

    c = df.iloc[-1]
    p = df.iloc[-2]

    candle_range = max(
        c["high"] - c["low"],
        1e-12
    )

    body = abs(c["close"] - c["open"])

    upper_wick = (
        c["high"] - max(c["open"], c["close"])
    )

    lower_wick = (
        min(c["open"], c["close"]) - c["low"]
    )

    found = []

    if body / candle_range < 0.1:
        found.append("Doji-like candle")

    if (
        lower_wick > body * 2
        and upper_wick < body * 1.5
    ):
        found.append("Possible bullish pin bar")

    if (
        upper_wick > body * 2
        and lower_wick < body * 1.5
    ):
        found.append("Possible bearish pin bar")

    if (
        c["close"] > c["open"]
        and p["close"] < p["open"]
        and c["close"] >= p["open"]
        and c["open"] <= p["close"]
    ):
        found.append("Bullish engulfing")

    if (
        c["close"] < c["open"]
        and p["close"] > p["open"]
        and c["close"] <= p["open"]
        and c["open"] >= p["close"]
    ):
        found.append("Bearish engulfing")

    return found or ["No common pattern detected"]


def run_backtest(df, rr=2.0, sl_atr=1.5):
    trades = []
    position = None

    for i in range(205, len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]

        if position is not None:
            if position["side"] == "BUY":
                hit_sl = nxt["low"] <= position["sl"]
                hit_tp = nxt["high"] >= position["tp"]
            else:
                hit_sl = nxt["high"] >= position["sl"]
                hit_tp = nxt["low"] <= position["tp"]

            if hit_sl or hit_tp:
                # If both levels are touched, count SL first.
                outcome = -1.0 if hit_sl else rr

                exit_price = (
                    position["sl"]
                    if hit_sl
                    else position["tp"]
                )

                trades.append({
                    "Entry Time": position["time"],
                    "Side": position["side"],
                    "Entry": position["entry"],
                    "Exit": exit_price,
                    "R Result": outcome,
                })

                position = None

            continue

        signal, _, _ = get_signal(row)

        if (
            signal not in ("BUY", "SELL")
            or pd.isna(row["ATR"])
            or row["ATR"] <= 0
        ):
            continue

        entry = float(nxt["open"])
        risk = float(row["ATR"] * sl_atr)

        if signal == "BUY":
            position = {
                "side": "BUY",
                "entry": entry,
                "sl": entry - risk,
                "tp": entry + risk * rr,
                "time": nxt["datetime"],
            }

        else:
            position = {
                "side": "SELL",
                "entry": entry,
                "sl": entry + risk,
                "tp": entry - risk * rr,
                "time": nxt["datetime"],
            }

    result = pd.DataFrame(trades)

    if position is not None:
        result.attrs["open_trade_note"] = (
            "An open simulated trade remained at the end "
            "and is excluded from closed-trade statistics."
        )

    return result


try:
    with st.spinner(
        f"Loading {symbol} {timeframe_label} candles..."
    ):
        raw_df, metadata = fetch_candles(
            api_key,
            symbol,
            interval,
            outputsize
        )

    df = add_indicators(raw_df)

except Exception as exc:
    st.error(f"Could not load {symbol} data: {exc}")

    st.info(
        "This pair or timeframe may not be supported by your "
        "Twelve Data plan, or API credits may be exhausted. "
        "Check the provider dashboard before enabling "
        "multi-timeframe requests."
    )

    st.stop()


if len(df) < 30:
    st.warning(
        "The provider returned fewer than 30 candles. "
        "Some indicators and signals may be unreliable."
    )


last = df.iloc[-1]

signal, quality, reason = get_signal(last)

entry, sl, targets = make_trade_levels(
    last,
    signal,
    [rr1, rr2, rr3]
)

risk_cash = account_balance * risk_pct / 100

support = float(df["low"].tail(50).min())
resistance = float(df["high"].tail(50).max())

atr_median = df["ATR"].tail(100).median()

if (
    pd.notna(last["ATR"])
    and pd.notna(atr_median)
    and atr_median > 0
):
    if last["ATR"] > atr_median * 1.5:
        volatility = "High"
    elif last["ATR"] < atr_median * 0.7:
        volatility = "Low"
    else:
        volatility = "Normal"
else:
    volatility = "Unavailable"


if (
    last["EMA20"] > last["EMA50"] > last["EMA200"]
):
    trend = "Bullish"
elif (
    last["EMA20"] < last["EMA50"] < last["EMA200"]
):
    trend = "Bearish"
else:
    trend = "Sideways / Mixed"


st.caption(
    f"Selected market: **{symbol}** · "
    f"Timeframe: **{timeframe_label}** · "
    f"Latest returned candle: {last['datetime']}"
)

a, b, c, d = st.columns(4)

a.metric("Latest Price", f"{last['close']:.5f}")
b.metric("Signal", signal, f"Quality score: {quality}/100")
c.metric("Trend", trend)
d.metric("Volatility", volatility)


st.subheader("Trade Setup")

t1, t2, t3, t4, t5 = st.columns(5)

t1.metric("Entry Reference", f"{entry:.5f}")

t2.metric(
    "Stop Loss",
    f"{sl:.5f}" if pd.notna(sl) else "N/A"
)

t3.metric(
    "TP1",
    f"{targets[0]:.5f}" if pd.notna(targets[0]) else "N/A"
)

t4.metric(
    "TP2",
    f"{targets[1]:.5f}" if pd.notna(targets[1]) else "N/A"
)

t5.metric(
    "TP3",
    f"{targets[2]:.5f}" if pd.notna(targets[2]) else "N/A"
)

st.write(
    f"**Signal reasoning:** {reason}. "
    f"**Support proxy:** {support:.5f} · "
    f"**Resistance proxy:** {resistance:.5f}"
)

st.caption(
    "Entry is the latest returned candle close for reference only, "
    "not a live execution price. SL/TP are ATR-based estimates."
)


tabs = st.tabs([
    "Chart & Indicators",
    "Price Action",
    "Risk Calculator",
    "Backtest",
    "Journal & Status"
])


with tabs[0]:
    chart = go.Figure(
        data=[
            go.Candlestick(
                x=df["datetime"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=symbol
            )
        ]
    )

    chart.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["EMA20"],
            name="EMA 20",
            line={"width": 1}
        )
    )

    chart.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["EMA50"],
            name="EMA 50",
            line={"width": 1}
        )
    )

    chart.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["EMA200"],
            name="EMA 200",
            line={"width": 1}
        )
    )

    chart.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["BB_upper"],
            name="BB Upper",
            line={"width": 1, "dash": "dot"}
        )
    )

    chart.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["BB_lower"],
            name="BB Lower",
            line={"width": 1, "dash": "dot"}
        )
    )

    chart.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        title=f"{symbol} · {timeframe_label}",
        margin={"l": 10, "r": 10, "t": 45, "b": 10}
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )

    i1, i2, i3, i4 = st.columns(4)

    i1.metric(
        "RSI (14)",
        f"{last['RSI']:.2f}" if pd.notna(last["RSI"]) else "N/A"
    )

    i2.metric(
        "ADX (14)",
        f"{last['ADX']:.2f}" if pd.notna(last["ADX"]) else "N/A"
    )

    i3.metric(
        "MACD",
        f"{last['MACD']:.5f}" if pd.notna(last["MACD"]) else "N/A"
    )

    i4.metric(
        "ATR (14)",
        f"{last['ATR']:.5f}" if pd.notna(last["ATR"]) else "N/A"
    )

    if pd.notna(last["BB_mid"]):
        st.write(
            f"**Bollinger Bands:** "
            f"Lower {last['BB_lower']:.5f} · "
            f"Middle {last['BB_mid']:.5f} · "
            f"Upper {last['BB_upper']:.5f}"
        )
    else:
        st.write("Bollinger Bands are warming up.")


with tabs[1]:
    st.markdown("### Candle Patterns")

    for item in pattern_labels(df):
        st.write(f"• {item}")

    st.markdown("### Market Structure Proxies")

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

    current = df.iloc[-1]

    if current["close"] > recent_high:
        st.success(
            "Possible bullish break of recent range (BOS proxy)."
        )

    elif current["close"] < recent_low:
        st.warning(
            "Possible bearish break of recent range (BOS proxy)."
        )

    else:
        st.info(
            "No close beyond the recent 20-candle range. "
            "No BOS proxy detected."
        )

    if (
        current["high"] > recent_high
        and current["close"] < recent_high
    ):
        st.warning("Possible buy-side liquidity sweep proxy.")

    if (
        current["low"] < recent_low
        and current["close"] > recent_low
    ):
        st.warning("Possible sell-side liquidity sweep proxy.")

    st.write(
        f"**Recent range low:** {recent_low:.5f} · "
        f"**Recent range high:** {recent_high:.5f}"
    )

    st.caption(
        "BOS, liquidity sweep, order block and FVG require rule "
        "definitions. This version only shows basic range-break "
        "and sweep proxies. It does not claim institutional "
        "order-flow data."
    )

    st.markdown("### No-Trade Filter")

    if (
        signal == "WAIT"
        or (
            pd.notna(last["ADX"])
            and last["ADX"] < 15
        )
    ):
        st.warning(
            "Conditions are mixed or trend strength is weak. "
            "Consider waiting."
        )
    else:
        st.info(
            "The rule-based filter did not flag a weak-trend "
            "no-trade condition. This is not a guarantee."
        )


with tabs[2]:
    st.markdown("### Risk Budget")

    r1, r2, r3 = st.columns(3)

    r1.metric("Risk Amount", f"${risk_cash:.2f}")
    r2.metric("Risk Percentage", f"{risk_pct:.2f}%")
    r3.metric("Risk-to-Reward", f"1:{rr2:g} (TP2)")

    st.markdown("### Lot Size")

    st.warning(
        "Lot size is not auto-calculated because contract size, "
        "tick value, account currency conversion and broker "
        "symbol specifications must be verified for each instrument."
    )

    st.write(
        "For a reliable lot-size calculator, use the exact "
        "contract and tick specifications shown in your Exness "
        "MT5 symbol details.
