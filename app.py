import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(
    page_title="AI Forex Market Analysis V2",
    page_icon="📊",
    layout="wide",
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
    "USD/BDT": "USD/BDT",
}

INTERVALS = {
    "1H": "1h",
    "15M": "15min",
    "10M": "10min",
    "5M": "5min",
}

MTF_INTERVALS = ["1h", "15min", "10min", "5min"]

st.title("📊 AI Forex Market Analysis V2")
st.caption(
    "Live technical analysis + MTF confirmation + market regime + historical backtest. "
    "Educational tool; not guaranteed financial advice."
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Market Settings")
    selected_name = st.selectbox("Market", list(SYMBOLS.keys()), index=0)
    symbol = SYMBOLS[selected_name]

    timeframe_label = st.selectbox(
        "Main Timeframe",
        list(INTERVALS.keys()),
        index=1,  # 15M default
    )
    interval = INTERVALS[timeframe_label]

    outputsize = st.select_slider(
        "Candles to load",
        options=[150, 200, 300, 400, 500],
        value=300,
    )

    signal_mode = st.radio(
        "Signal mode",
        ["Closed candle (stable)", "Live candle (faster)"],
        index=0,
    )

    st.divider()
    st.subheader("Risk / Targets")
    account_balance = st.number_input(
        "Account Balance (USD)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
    )
    risk_pct = st.number_input(
        "Risk per Trade (%)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )
    rr1 = st.number_input("TP1 RR", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
    rr2 = st.number_input("TP2 RR", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
    rr3 = st.number_input("TP3 RR", min_value=0.5, max_value=10.0, value=3.0, step=0.5)

    st.divider()
    st.subheader("Confirmation")
    mtf_enabled = st.checkbox(
        "Enable MTF confirmation",
        value=True,
        help="Uses extra API requests.",
    )
    strict_mtf = st.checkbox(
        "Strict MTF filter",
        value=True,
        help="If enabled, conflicting higher-timeframe direction becomes WAIT.",
    )

    min_score = st.slider(
        "Minimum technical score",
        min_value=5,
        max_value=9,
        value=6,
    )

    if st.button("🔄 Refresh market data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

api_key = st.secrets.get("TWELVE_DATA_API_KEY", "")
if not api_key:
    st.error(
        "Missing TWELVE_DATA_API_KEY. Add it in Streamlit → Settings → Secrets."
    )
    st.stop()

# -----------------------------
# Twelve Data API connection test
# -----------------------------
def test_twelve_data_api(api_key, symbol="XAU/USD", interval="15min"):
    """Small live request used only to verify API access and candle payload."""
    url = "https://api.twelvedata.com/time_series"
    response = requests.get(
        url,
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": 2,
            "apikey": api_key,
            "format": "JSON",
        },
        timeout=15,
    )
    response.raise_for_status()

    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError("Twelve Data returned a non-JSON response.") from exc

    if data.get("status") == "error":
        raise ValueError(data.get("message", "Twelve Data API returned an error."))

    values = data.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("API connected, but no candle values were returned.")

    required = {"datetime", "open", "high", "low", "close"}
    missing = required.difference(values[0].keys())
    if missing:
        raise ValueError(f"API response is missing fields: {sorted(missing)}")

    latest = values[0]
    return {
        "symbol": data.get("meta", {}).get("symbol", symbol),
        "interval": data.get("meta", {}).get("interval", interval),
        "candles": len(values),
        "datetime": latest.get("datetime"),
        "close": float(latest["close"]),
    }


with st.sidebar:
    st.divider()
    st.subheader("🔌 API Connection")

    if st.button("🧪 Test Twelve Data API", use_container_width=True):
        with st.spinner("Testing Twelve Data connection..."):
            try:
                api_test = test_twelve_data_api(api_key, "XAU/USD", "15min")
                st.success("✅ API Connected")
                st.caption(
                    f"XAU/USD • {api_test['interval']} • "
                    f"{api_test['candles']} candles received"
                )
                st.metric("Latest XAU/USD Close", f"{api_test['close']:,.3f}")
                st.caption(f"Latest candle: {api_test['datetime']}")
            except requests.exceptions.Timeout:
                st.error("❌ API timeout. Twelve Data did not respond in time.")
            except requests.exceptions.RequestException as exc:
                st.error(f"❌ Network/API request failed: {exc}")
            except Exception as exc:
                st.error(f"❌ API test failed: {exc}")


# -----------------------------
# Data
# -----------------------------
@st.cache_data(ttl=45, show_spinner=False)
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
        raise ValueError(data.get("message", "No candle data returned."))

    df = pd.DataFrame(data["values"])
    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC fields: {missing}")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = (
        df.dropna(subset=required)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if len(df) < 80:
        raise ValueError("Not enough candles returned by the data provider.")

    return df, data.get("meta", {})


# -----------------------------
# Indicators
# -----------------------------
def add_indicators(df):
    d = df.copy()

    d["EMA20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["EMA200"] = d["close"].ewm(span=200, adjust=False).mean()

    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    prev_close = d["close"].shift(1)
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    up_move = d["high"].diff()
    down_move = -d["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr_safe = d["ATR"].replace(0, np.nan)
    plus_di = (
        100
        * plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        / atr_safe
    )
    minus_di = (
        100
        * minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        / atr_safe
    )
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["ADX"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    d["MACD"] = (
        d["close"].ewm(span=12, adjust=False).mean()
        - d["close"].ewm(span=26, adjust=False).mean()
    )
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]

    d["BB_mid"] = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["BB_upper"] = d["BB_mid"] + 2 * bb_std
    d["BB_lower"] = d["BB_mid"] - 2 * bb_std

    d["volume_avg"] = (
        d["volume"].rolling(20).mean() if "volume" in d.columns else np.nan
    )

    d["body"] = (d["close"] - d["open"]).abs()
    d["range"] = (d["high"] - d["low"]).replace(0, np.nan)
    d["body_ratio"] = d["body"] / d["range"]
    d["upper_wick"] = d["high"] - d[["open", "close"]].max(axis=1)
    d["lower_wick"] = d[["open", "close"]].min(axis=1) - d["low"]

    return d


# -----------------------------
# Candle confirmation
# -----------------------------
def candle_confirmation(row):
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    body_ratio = body / rng

    bullish_engulf = (
        c > o
        and row.name is not None
    )
    bearish_engulf = c < o

    bullish_pin = lower >= body * 2.0 and lower >= upper * 1.2 and c > o
    bearish_pin = upper >= body * 2.0 and upper >= lower * 1.2 and c < o
    strong_bull = c > o and body_ratio >= 0.60
    strong_bear = c < o and body_ratio >= 0.60
    doji = body_ratio <= 0.18

    if bullish_pin or strong_bull or bullish_engulf:
        bull = True
    else:
        bull = False

    if bearish_pin or strong_bear or bearish_engulf:
        bear = True
    else:
        bear = False

    if doji:
        label = "Doji / weak confirmation"
    elif bull and not bear:
        label = "Bullish confirmation"
    elif bear and not bull:
        label = "Bearish confirmation"
    else:
        label = "Mixed candle"

    return {
        "bull": bull,
        "bear": bear,
        "doji": doji,
        "label": label,
        "body_ratio": body_ratio,
    }


# -----------------------------
# Market regime
# -----------------------------
def market_regime(row, df):
    atr = float(row["ATR"]) if pd.notna(row["ATR"]) else 0.0
    adx = float(row["ADX"]) if pd.notna(row["ADX"]) else 0.0
    ema20 = row["EMA20"]
    ema50 = row["EMA50"]
    ema200 = row["EMA200"]
    close = row["close"]

    if atr <= 0 or pd.isna(ema200):
        return "UNKNOWN"

    recent = df.tail(20)
    avg_atr = recent["ATR"].mean()
    atr_ratio = atr / avg_atr if avg_atr and pd.notna(avg_atr) else 1.0

    if adx >= 25 and close > ema20 > ema50 > ema200:
        return "STRONG UPTREND"
    if adx >= 25 and close < ema20 < ema50 < ema200:
        return "STRONG DOWNTREND"

    if adx >= 18 and close > ema50 and ema20 > ema50:
        return "UPTREND"
    if adx >= 18 and close < ema50 and ema20 < ema50:
        return "DOWNTREND"

    if adx < 15 and 0.75 <= atr_ratio <= 1.35:
        return "CHOPPY / RANGE"

    if atr_ratio >= 1.50:
        return "HIGH VOLATILITY"

    return "MIXED"


# -----------------------------
# Signal engine
# -----------------------------
def get_signal(row, df, min_score=6):
    required = [
        "close", "EMA20", "EMA50", "EMA200",
        "RSI", "MACD", "MACD_signal", "ADX",
        "ATR", "BB_upper", "BB_lower"
    ]
    if any(pd.isna(row.get(c)) for c in required):
        return {
            "signal": "WAIT",
            "score": 0,
            "max_score": 9,
            "confidence": 0,
            "buy_score": 0,
            "sell_score": 0,
            "reasons": [],
            "regime": "UNKNOWN",
            "candle": "Unknown",
        }

    candle = candle_confirmation(row)
    regime = market_regime(row, df)

    buy_checks = [
        bool(row["close"] > row["EMA20"]),
        bool(row["EMA20"] > row["EMA50"]),
        bool(row["EMA50"] > row["EMA200"]),
        bool(row["RSI"] > 52),
        bool(row["MACD"] > row["MACD_signal"]),
        bool(row["ADX"] >= 18),
        bool(candle["bull"] and not candle["doji"]),
        bool(row["close"] > row["BB_mid"]),
        bool(regime in ["UPTREND", "STRONG UPTREND"]),
    ]

    sell_checks = [
        bool(row["close"] < row["EMA20"]),
        bool(row["EMA20"] < row["EMA50"]),
        bool(row["EMA50"] < row["EMA200"]),
        bool(row["RSI"] < 48),
        bool(row["MACD"] < row["MACD_signal"]),
        bool(row["ADX"] >= 18),
        bool(candle["bear"] and not candle["doji"]),
        bool(row["close"] < row["BB_mid"]),
        bool(regime in ["DOWNTREND", "STRONG DOWNTREND"]),
    ]

    buy_score = int(sum(buy_checks))
    sell_score = int(sum(sell_checks))
    max_score = len(buy_checks)

    # WAIT is preferred in a choppy market unless the technical setup is unusually strong.
    if regime == "CHOPPY / RANGE" and max(buy_score, sell_score) < 8:
        signal = "WAIT"
    elif buy_score >= min_score and buy_score > sell_score:
        signal = "STRONG BUY" if buy_score >= 8 else "BUY"
    elif sell_score >= min_score and sell_score > buy_score:
        signal = "STRONG SELL" if sell_score >= 8 else "SELL"
    else:
        signal = "WAIT"

    score = max(buy_score, sell_score)
    confidence = int(round((score / max_score) * 100))

    reasons = []
    if row["close"] > row["EMA20"]:
        reasons.append("Price above EMA20")
    elif row["close"] < row["EMA20"]:
        reasons.append("Price below EMA20")

    if row["EMA20"] > row["EMA50"] > row["EMA200"]:
        reasons.append("Bullish EMA structure")
    elif row["EMA20"] < row["EMA50"] < row["EMA200"]:
        reasons.append("Bearish EMA structure")

    if row["RSI"] > 52:
        reasons.append("RSI bullish")
    elif row["RSI"] < 48:
        reasons.append("RSI bearish")

    if row["MACD"] > row["MACD_signal"]:
        reasons.append("MACD bullish")
    elif row["MACD"] < row["MACD_signal"]:
        reasons.append("MACD bearish")

    if row["ADX"] >= 18:
        reasons.append(f"ADX {row['ADX']:.1f}")

    if candle["label"]:
        reasons.append(candle["label"])

    return {
        "signal": signal,
        "score": score,
        "max_score": max_score,
        "confidence": confidence,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "reasons": reasons,
        "regime": regime,
        "candle": candle["label"],
    }


# -----------------------------
# Smart SL / TP
# -----------------------------
def build_trade(row, signal, rr1, rr2, rr3):
    direction = "BUY" if "BUY" in signal else "SELL" if "SELL" in signal else None
    if direction is None or pd.isna(row["ATR"]):
        return None

    entry = float(row["close"])
    atr = float(row["ATR"])

    recent_low = float(row.get("recent_low", np.nan))
    recent_high = float(row.get("recent_high", np.nan))

    if direction == "BUY":
        atr_sl = entry - 1.5 * atr
        swing_sl = recent_low - 0.10 * atr if np.isfinite(recent_low) else atr_sl
        sl = min(atr_sl, swing_sl)
        risk = entry - sl
        if risk <= 0:
            return None
        return {
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp1": entry + risk * rr1,
            "tp2": entry + risk * rr2,
            "tp3": entry + risk * rr3,
            "risk_distance": risk,
        }

    atr_sl = entry + 1.5 * atr
    swing_sl = recent_high + 0.10 * atr if np.isfinite(recent_high) else atr_sl
    sl = max(atr_sl, swing_sl)
    risk = sl - entry
    if risk <= 0:
        return None

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": entry - risk * rr1,
        "tp2": entry - risk * rr2,
        "tp3": entry - risk * rr3,
        "risk_distance": risk,
    }


def add_trade_levels(df):
    d = df.copy()
    d["recent_low"] = d["low"].rolling(6).min().shift(1)
    d["recent_high"] = d["high"].rolling(6).max().shift(1)
    return d


# -----------------------------
# Backtest
# -----------------------------
def backtest(df, min_score=6, rr=2.0):
    trades = []

    if len(df) < 100:
        return pd.DataFrame(), {}

    for i in range(100, len(df) - 1):
        row = df.iloc[i]
        result = get_signal(row, df.iloc[: i + 1], min_score=min_score)

        if result["signal"] not in ["BUY", "STRONG BUY", "SELL", "STRONG SELL"]:
            continue

        trade = build_trade(row, result["signal"], 1.0, rr, rr)
        if trade is None:
            continue

        # No look-ahead: entry is next candle open.
        next_bar = df.iloc[i + 1]
        entry = float(next_bar["open"])
        direction = trade["direction"]

        # Rebuild levels from the next-bar entry using the same risk distance.
        risk = trade["risk_distance"]
        if direction == "BUY":
            sl = entry - risk
            tp = entry + risk * rr
        else:
            sl = entry + risk
            tp = entry - risk * rr

        outcome = None
        exit_price = None
        exit_time = None

        for j in range(i + 1, len(df)):
            bar = df.iloc[j]

            if direction == "BUY":
                hit_sl = bar["low"] <= sl
                hit_tp = bar["high"] >= tp
            else:
                hit_sl = bar["high"] >= sl
                hit_tp = bar["low"] <= tp

            # Conservative rule: if both are touched in one candle, count SL first.
            if hit_sl:
                outcome = "LOSS"
                exit_price = sl
                exit_time = bar["datetime"]
                break
            if hit_tp:
                outcome = "WIN"
                exit_price = tp
                exit_time = bar["datetime"]
                break

        if outcome is None:
            continue

        r = 1.0 if outcome == "WIN" else -1.0

        trades.append(
            {
                "time": next_bar["datetime"],
                "direction": direction,
                "entry": entry,
                "SL": sl,
                "TP": tp,
                "result": outcome,
                "R": r,
                "exit_time": exit_time,
                "score": result["score"],
                "regime": result["regime"],
            }
        )

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return trades_df, {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "max_drawdown_r": 0.0,
        }

    wins = int((trades_df["result"] == "WIN").sum())
    losses = int((trades_df["result"] == "LOSS").sum())
    win_rate = wins / len(trades_df) * 100

    gross_profit = float(trades_df.loc[trades_df["R"] > 0, "R"].sum())
    gross_loss = abs(float(trades_df.loc[trades_df["R"] < 0, "R"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    equity = trades_df["R"].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = abs(float(drawdown.min())) if len(drawdown) else 0.0

        return trades_df, {
        "trades": len(trades_df),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_r": float(trades_df["R"].sum()),
        "max_drawdown_r": max_dd,
    }
  
