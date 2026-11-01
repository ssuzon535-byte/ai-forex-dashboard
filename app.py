import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================================================
# PAGE
# =========================================================
st.set_page_config(
    page_title="AI Forex Market Analysis V2",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Forex Market Analysis V2")
st.caption(
    "Live Forex/XAUUSD technical analysis • MTF • Market Regime • "
    "Accuracy Backtest • Smart SL/TP"
)

# =========================================================
# SETTINGS
# =========================================================
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
    "CAD/JPY": "CAD/JPY",
    "CHF/JPY": "CHF/JPY",
    "USD/BDT": "USD/BDT"
}

INTERVALS = {
    "1H": "1h",
    "15M": "15min",
    "10M": "10min",
    "5M": "5min"
}

MTF_INTERVALS = ["1h", "15min", "10min", "5min"]


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.header("⚙️ Market Settings")

    market_name = st.selectbox(
        "Market",
        list(SYMBOLS.keys()),
        index=0
    )

    symbol = SYMBOLS[market_name]

    timeframe_name = st.selectbox(
        "Main Timeframe",
        list(INTERVALS.keys()),
        index=1
    )

    interval = INTERVALS[timeframe_name]

    candles = st.select_slider(
        "Historical candles",
        options=[150, 200, 300, 400, 500],
        value=300
    )

    signal_mode = st.radio(
        "Signal Mode",
        [
            "Closed Candle - Stable",
            "Live Candle - Faster"
        ],
        index=0
    )

    st.divider()

    st.header("🎯 Signal Settings")

    min_score = st.slider(
        "Minimum Score",
        5,
        9,
        6
    )

    use_mtf = st.checkbox(
        "Enable MTF Confirmation",
        value=True
    )

    strict_mtf = st.checkbox(
        "Strict MTF Filter",
        value=True
    )

    st.divider()

    st.header("💰 Risk Management")

    balance = st.number_input(
        "Account Balance ($)",
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
        "TP1 RR",
        min_value=0.5,
        max_value=10.0,
        value=1.0,
        step=0.5
    )

    rr2 = st.number_input(
        "TP2 RR",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5
    )

    rr3 = st.number_input(
        "TP3 RR",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.5
    )

    st.divider()

    if st.button(
        "🔄 Refresh Market",
        use_container_width=True
    ):
        st.cache_data.clear()
        st.rerun()


# =========================================================
# API KEY
# =========================================================
api_key = st.secrets.get(
    "TWELVE_DATA_API_KEY",
    ""
)

if not api_key:
    st.error(
        "❌ TWELVE_DATA_API_KEY পাওয়া যায়নি। "
        "Streamlit Secrets-এ API key যোগ করো।"
    )
    st.stop()


# =========================================================
# API TEST
# =========================================================
def test_api(key):

    url = "https://api.twelvedata.com/time_series"

    response = requests.get(
        url,
        params={
            "symbol": "XAU/USD",
            "interval": "15min",
            "outputsize": 2,
            "apikey": key,
            "format": "JSON"
        },
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") == "error":
        raise ValueError(
            data.get(
                "message",
                "Unknown Twelve Data error"
            )
        )

    values = data.get("values")

    if not values:
        raise ValueError(
            "API connected but candle data পাওয়া যায়নি।"
        )

    last = values[0]

    return {
        "symbol": data.get(
            "meta",
            {}
        ).get(
            "symbol",
            "XAU/USD"
        ),
        "interval": data.get(
            "meta",
            {}
        ).get(
            "interval",
            "15min"
        ),
        "candles": len(values),
        "time": last.get("datetime"),
        "close": float(last["close"])
    }


with st.sidebar:

    st.header("🔌 API Connection")

    if st.button(
        "🧪 Test Twelve Data API",
        use_container_width=True
    ):

        with st.spinner(
            "Testing Twelve Data..."
        ):

            try:

                result = test_api(api_key)

                st.success(
                    "✅ API Connected"
                )

                st.metric(
                    "XAU/USD",
                    f"{result['close']:,.3f}"
                )

                st.caption(
                    f"Candles: {result['candles']}"
                )

                st.caption(
                    f"Last candle: {result['time']}"
                )

            except requests.exceptions.Timeout:

                st.error(
                    "❌ API Timeout"
                )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"❌ Network error: {e}"
                )

            except Exception as e:

                st.error(
                    f"❌ API Test Failed: {e}"
                )


# =========================================================
# DATA FETCH
# =========================================================
@st.cache_data(ttl=45)
def get_data(
    key,
    sym,
    tf,
    size
):

    url = "https://api.twelvedata.com/time_series"

    response = requests.get(
        url,
        params={
            "symbol": sym,
            "interval": tf,
            "outputsize": size,
            "apikey": key,
            "format": "JSON"
        },
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") == "error":

        raise ValueError(
            data.get(
                "message",
                "API error"
            )
        )

    if "values" not in data:

        raise ValueError(
            "No candle values returned."
        )

    df = pd.DataFrame(
        data["values"]
    )

    required = [
        "datetime",
        "open",
        "high",
        "low",
        "close"
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    df = df.dropna(
        subset=required
    )

    df = df.drop_duplicates(
        "datetime"
    )

    df = df.sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    if len(df) < 80:

        raise ValueError(
            "Not enough candles."
        )

    return df


# =========================================================
# INDICATORS
# =========================================================
def indicators(df):

    d = df.copy()

    # EMA
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

    # RSI
    change = d["close"].diff()

    gain = change.clip(
        lower=0
    )

    loss = -change.clip(
        upper=0
    )

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

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    d["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # ATR
    previous_close = d["close"].shift(1)

    tr = pd.concat(
        [
            d["high"] - d["low"],
            (
                d["high"] -
                previous_close
            ).abs(),
            (
                d["low"] -
                previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    d["ATR"] = tr.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    # ADX
    up_move = d["high"].diff()

    down_move = -d["low"].diff()

    plus_dm = up_move.where(
        (
            (up_move > down_move) &
            (up_move > 0)
        ),
        0.0
    )

    minus_dm = down_move.where(
        (
            (down_move > up_move) &
            (down_move > 0)
        ),
        0.0
    )

    atr_safe = d["ATR"].replace(
        0,
        np.nan
    )

    plus_di = (
        100 *
        plus_dm.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean() /
        atr_safe
    )

    minus_di = (
        100 *
        minus_dm.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean() /
        atr_safe
    )

    dx = (
        100 *
        (plus_di - minus_di).abs() /
        (plus_di + minus_di).replace(
            0,
            np.nan
        )
    )

    d["ADX"] = dx.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    # MACD
    ema12 = d["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = d["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    d["MACD"] = ema12 - ema26

    d["MACD_SIGNAL"] = d["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    d["MACD_HIST"] = (
        d["MACD"] -
        d["MACD_SIGNAL"]
    )

    # Bollinger
    d["BB_MID"] = d["close"].rolling(
        20
    ).mean()

    std = d["close"].rolling(
        20
    ).std()

    d["BB_UPPER"] = (
        d["BB_MID"] +
        2 * std
    )

    d["BB_LOWER"] = (
        d["BB_MID"] -
        2 * std
    )

    # Candle data
    d["RANGE"] = (
        d["high"] -
        d["low"]
    )

    d["BODY"] = (
        d["close"] -
        d["open"]
    ).abs()

    d["BODY_RATIO"] = (
        d["BODY"] /
        d["RANGE"].replace(
            0,
            np.nan
        )
    )

    return d


# =========================================================
# CANDLE PATTERN
# =========================================================
def candle_pattern(row):

    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])

    rng = max(
        h - l,
        1e-10
    )

    body = abs(
        c - o
    )

    upper = h - max(
        o,
        c
    )

    lower = min(
        o,
        c
    ) - l

    ratio = body / rng

    if ratio < 0.15:
        return "DOJI"

    if (
        lower >= body * 2 and
        upper <= body
    ):

        return (
            "BULLISH PIN"
            if c >= o
            else "PIN BAR"
        )

    if (
        upper >= body * 2 and
        lower <= body
    ):

        return (
            "BEARISH PIN"
            if c <= o
            else "PIN BAR"
        )

    if ratio >= 0.65:

        if c > o:
            return "STRONG BULL"
        else:
            return "STRONG BEAR"

    return "NORMAL"


# =========================================================
# MARKET REGIME
# =========================================================
def market_regime(row):

    adx = float(row["ADX"])
    atr = float(row["ATR"])
    close = float(row["close"])

    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema200 = float(row["EMA200"])

    if not np.isfinite(
        adx
    ):

        return "UNKNOWN"

    if adx >= 30:

        if (
            ema20 >
            ema50 >
            ema200
        ):

            return "STRONG UPTREND"

        if (
            ema20 <
            ema50 <
            ema200
        ):

            return "STRONG DOWNTREND"

        return "STRONG TREND"

    if adx < 15:

        return "CHOPPY / RANGE"

    if (
        close > ema20 >
        ema50
    ):

        return "UPTREND"

    if (
        close < ema20 <
        ema50
    ):

        return "DOWNTREND"

    if atr <= 0:

        return "RANGE"

    return "RANGE"


# =========================================================
# SIGNAL ENGINE
# =========================================================
def calculate_signal(
    df,
    row,
    minimum_score=6
):

    close = float(row["close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema200 = float(row["EMA200"])

    rsi = float(row["RSI"])
    adx = float(row["ADX"])

    macd = float(row["MACD"])
    macd_signal = float(
        row["MACD_SIGNAL"]
    )

    pattern = candle_pattern(
        row
    )

    regime = market_regime(
        row
    )

    buy = 0
    sell = 0

    reasons_buy = []
    reasons_sell = []

    # 1 Price / EMA20
    if close > ema20:

        buy += 1
        reasons_buy.append(
            "Price > EMA20"
        )

    if close < ema20:

        sell += 1
        reasons_sell.append(
            "Price < EMA20"
        )

    # 2 EMA20 / EMA50
    if ema20 > ema50:

        buy += 1
        reasons_buy.append(
            "EMA20 > EMA50"
        )

    if ema20 < ema50:

        sell += 1
        reasons_sell.append(
            "EMA20 < EMA50"
        )

    # 3 EMA50 / EMA200
    if ema50 > ema200:

        buy += 1
        reasons_buy.append(
            "EMA50 > EMA200"
        )

    if ema50 < ema200:

        sell += 1
        reasons_sell.append(
            "EMA50 < EMA200"
        )

    # 4 RSI
    if rsi >= 55:

        buy += 1
        reasons_buy.append(
            f"RSI {rsi:.1f}"
        )

    if rsi <= 45:

        sell += 1
        reasons_sell.append(
            f"RSI {rsi:.1f}"
        )

    # 5 MACD
    if macd > macd_signal:

        buy += 1
        reasons_buy.append(
            "MACD bullish"
        )

    if macd < macd_signal:

        sell += 1
        reasons_sell.append(
            "MACD bearish"
        )

    # 6 ADX
    if adx >= 18:

        if buy > sell:

            buy += 1
            reasons_buy.append(
                "ADX confirms trend"
            )

        elif sell > buy:

            sell += 1
            reasons_sell.append(
                "ADX confirms trend"
            )

    # 7 Candle confirmation
    if pattern in [
        "BULLISH PIN",
        "STRONG BULL"
    ]:

        buy += 1
        reasons_buy.append(
            pattern
        )

    if pattern in [
        "BEARISH PIN",
        "STRONG BEAR"
    ]:

        sell += 1
        reasons_sell.append(
            pattern
        )

    # 8 Breakout
    recent_high = df["high"].iloc[
        -7:-1
    ].max()

    recent_low = df["low"].iloc[
        -7:-1
    ].min()

    if close > recent_high:

        buy += 1
        reasons_buy.append(
            "Range breakout"
        )

    if close < recent_low:

        sell += 1
        reasons_sell.append(
            "Range breakdown"
        )

    # 9 Momentum
    if len(df) >= 4:

        previous_close = float(
            df["close"].iloc[-4]
        )

        if close > previous_close:

            buy += 1

        elif close < previous_close:

            sell += 1

    max_score = 9

    if (
        buy >= minimum_score and
        buy > sell
    ):

        signal = (
            "STRONG BUY"
            if buy >= 8
            else "BUY"
        )

        confidence = round(
            buy / max_score * 100,
            1
        )

    elif (
        sell >= minimum_score and
        sell > buy
    ):

        signal = (
            "STRONG SELL"
            if sell >= 8
            else "SELL"
        )

        confidence = round(
            sell / max_score * 100,
            1
        )

    else:

        signal = "WAIT"

        confidence = round(
            max(
                buy,
                sell
            ) /
            max_score *
            100,
            1
        )

    return {
        "signal": signal,
        "buy_score": buy,
        "sell_score": sell,
        "max_score": max_score,
        "confidence": confidence,
        "pattern": pattern,
        "regime": regime,
        "buy_reasons": reasons_buy,
        "sell_reasons": reasons_sell
    }


# =========================================================
# SMART SL / TP
# =========================================================
def make_trade(
    row,
    signal,
    rr1,
    rr2,
    rr3
):

    if signal not in [
        "BUY",
        "STRONG BUY",
        "SELL",
        "STRONG SELL"
    ]:

        return None

    entry = float(
        row["close"]
    )

    atr = float(
        row["ATR"]
    )

    if not np.isfinite(
        atr
    ) or atr <= 0:

        return None

    recent_low = float(
        row.name
    ) if False else 0

    # ATR protection
    if "BUY" in signal:

        sl = entry - 1.5 * atr

        risk = entry - sl

        tp1 = entry + risk * rr1
        tp2 = entry + risk * rr2
        tp3 = entry + risk * rr3

        direction = "BUY"

    else:

        sl = entry + 1.5 * atr

        risk = sl - entry

        tp1 = entry - risk * rr1
        tp2 = entry - risk * rr2
        tp3 = entry - risk * rr3

        direction = "SELL"

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "risk": risk
    }


# =========================================================
# MTF
# =========================================================
def get_mtf(
    key,
    sym,
    size,
    minimum_score
):

    rows = []

    for tf in MTF_INTERVALS:

        try:

            raw = get_data(
                key,
                sym,
                tf,
                min(
                    size,
                    250
                )
            )

            data = indicators(
                raw
            )

            if len(data) >= 2:

                row = data.iloc[-2]

            else:

                row = data.iloc[-1]

            info = calculate_signal(
                data,
                row,
                minimum_score
            )

            rows.append(
                {
                    "Timeframe": tf,
                    "Signal": info["signal"],
                    "BUY": info["buy_score"],
                    "SELL": info["sell_score"],
                    "Confidence": f"{info['confidence']:.0f}%",
                    "Regime": info["regime"]
                }
            )

        except Exception as e:

            rows.append(
                {
                    "Timeframe": tf,
                    "Signal": "ERROR",
                    "BUY": "-",
                    "SELL": "-",
                    "Confidence": "-",
                    "Regime": str(e)[:35]
                }
            )

    return pd.DataFrame(
        rows
    )


def mtf_vote(
    table
):

    if table.empty:

        return "UNKNOWN"

    buy = 0
    sell = 0

    for signal in table["Signal"]:

        if "BUY" in str(signal):

            buy += 1

        elif "SELL" i
