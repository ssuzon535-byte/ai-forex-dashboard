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

with st.sidebar:
    st.header("Market Settings")
    selected_name = st.selectbox("Market", list(SYMBOLS.keys()), index=0)
    symbol = SYMBOLS[selected_name]

    timeframe_label = st.selectbox(
        "Main Timeframe",
        list(INTERVALS.keys()),
        index=1,
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

    rr1 = st.number_input(
        "TP1 RR",
        min_value=0.5,
        max_value=10.0,
        value=1.0,
        step=0.5,
    )

    rr2 = st.number_input(
        "TP2 RR",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
    )

    rr3 = st.number_input(
        "TP3 RR",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.5,
    )

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


def test_twelve_data_api(api_key, symbol="XAU/USD", interval="15min"):
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
        raise ValueError(
            data.get(
                "message",
                "Twelve Data API returned an error."
            )
        )

    values = data.get("values")

    if not isinstance(values, list) or not values:
        raise ValueError(
            "API connected, but no candle values were returned."
        )

    required = {
        "datetime",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required.difference(values[0].keys())

    if missing:
        raise ValueError(
            f"API response is missing fields: {sorted(missing)}"
        )

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

    if st.button(
        "🧪 Test Twelve Data API",
        use_container_width=True,
    ):
        with st.spinner("Testing Twelve Data connection..."):
            try:
                api_test = test_twelve_data_api(
                    api_key,
                    "XAU/USD",
                    "15min",
                )

                st.success("✅ API Connected")

                st.caption(
                    f"XAU/USD • {api_test['interval']} • "
                    f"{api_test['candles']} candles received"
                )

                st.metric(
                    "Latest XAU/USD Close",
                    f"{api_test['close']:,.3f}",
                )

                st.caption(
                    f"Latest candle: {api_test['datetime']}"
                )

            except requests.exceptions.Timeout:
                st.error(
                    "❌ API timeout. Twelve Data did not respond in time."
                )

            except requests.exceptions.RequestException as exc:
                st.error(
                    f"❌ Network/API request failed: {exc}"
                )

            except Exception as exc:
                st.error(
                    f"❌ API test failed: {exc}"
                )


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
        raise ValueError(
            data.get(
                "message",
                "No candle data returned.",
            )
        )

    df = pd.DataFrame(data["values"])

    required = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing OHLC fields: {missing}"
        )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    df = (
        df.dropna(subset=required)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if len(df) < 80:
        raise ValueError(
            "Not enough candles returned by the data provider."
        )

    return df, data.get("meta", {})


def add_indicators(df):
    d = df.copy()

    d["EMA20"] = d["close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    d["EMA50"] = d["close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    d["EMA200"] = d["close"].ewm(
        span=200,
        adjust=False,
    ).mean()

    delta = d["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    d["RSI"] = 100 - (
        100 / (1 + rs)
    )

    prev_close = d["close"].shift(1)

    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    d["ATR"] = tr.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    up_move = d["high"].diff()
    down_move = -d["low"].diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0),
        0.0,
    )

    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0),
        0.0,
    )

    atr_safe = d["ATR"].replace(
        0,
        np.nan,
    )

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False,
        ).mean()
        / atr_safe
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False,
        ).mean()
        / atr_safe
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(
            0,
            np.nan,
        )
    )

    d["ADX"] = dx.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    d["MACD"] = (
        d["close"].ewm(
            span=12,
            adjust=False,
        ).mean()
        -
        d["close"].ewm(
            span=26,
            adjust=False,
        ).mean()
    )

    d["MACD_signal"] = d["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    d["MACD_hist"] = (
        d["MACD"] -
        d["MACD_signal"]
    )

    d["BB_mid"] = d["close"].rolling(
        20
    ).mean()

    bb_std = d["close"].rolling(
        20
    ).std()

    d["BB_upper"] = (
        d["BB_mid"] +
        2 * bb_std
    )

    d["BB_lower"] = (
        d["BB_mid"] -
        2 * bb_std
    )

    d["volume_avg"] = (
        d["volume"].rolling(20).mean()
        if "volume" in d.columns
        else np.nan
    )

    d["body"] = (
        d["close"] -
        d["open"]
    ).abs()

    d["range"] = (
        d["high"] -
        d["low"]
    ).replace(
        0,
        np.nan,
    )

    d["body_ratio"] = (
        d["body"] /
        d["range"]
    )

    d["upper_wick"] = (
        d["high"] -
        d[["open", "close"]].max(axis=1)
    )

    d["lower_wick"] = (
        d[["open", "close"]].min(axis=1) -
        d["low"]
    )

  def add_trade_levels(df):
    d = df.copy()

    d["recent_low"] = (
        d["low"]
        .rolling(6)
        .min()
        .shift(1)
    )

    d["recent_high"] = (
        d["high"]
        .rolling(6)
        .max()
        .shift(1)
    )

    d["buy_sl"] = np.minimum(
        d["close"] - 1.5 * d["ATR"],
        d["recent_low"] - 0.10 * d["ATR"],
    )

    d["sell_sl"] = np.maximum(
        d["close"] + 1.5 * d["ATR"],
        d["recent_high"] + 0.10 * d["ATR"],
    )

    return d


def market_regime(row):
    close = float(row["close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema200 = float(row["EMA200"])
    adx = float(row["ADX"])
    atr = float(row["ATR"])

    atr_ratio = (
        atr / close
        if close != 0
        else 0
    )

    if (
        adx >= 25
        and close > ema20 > ema50 > ema200
    ):
        return "STRONG UPTREND"

    if (
        adx >= 25
        and close < ema20 < ema50 < ema200
    ):
        return "STRONG DOWNTREND"

    if (
        adx >= 18
        and close > ema50
        and ema20 > ema50
    ):
        return "UPTREND"

    if (
        adx >= 18
        and close < ema50
        and ema20 < ema50
    ):
        return "DOWNTREND"

    if (
        adx < 15
        and 0.0075 <= atr_ratio <= 0.0135
    ):
        return "CHOPPY / RANGE"

    if atr_ratio >= 0.015:
        return "HIGH VOLATILITY"

    return "MIXED"


def get_signal(row, minimum_score=6):
    regime = market_regime(row)

    close = float(row["close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema200 = float(row["EMA200"])
    rsi = float(row["RSI"])
    macd = float(row["MACD"])
    macd_signal = float(row["MACD_signal"])
    adx = float(row["ADX"])
    bb_mid = float(row["BB_mid"])
    body_ratio = float(row["body_ratio"])

    bullish_candle = (
        close > float(row["open"])
        and body_ratio >= 0.25
    )

    bearish_candle = (
        close < float(row["open"])
        and body_ratio >= 0.25
    )

    buy_checks = {
        "Price > EMA20": close > ema20,
        "EMA20 > EMA50": ema20 > ema50,
        "EMA50 > EMA200": ema50 > ema200,
        "RSI > 52": rsi > 52,
        "MACD > Signal": macd > macd_signal,
        "ADX >= 18": adx >= 18,
        "Bullish Candle": bullish_candle,
        "Price > BB Mid": close > bb_mid,
        "Uptrend Regime": regime in [
            "UPTREND",
            "STRONG UPTREND",
        ],
    }

    sell_checks = {
        "Price < EMA20": close < ema20,
        "EMA20 < EMA50": ema20 < ema50,
        "EMA50 < EMA200": ema50 < ema200,
        "RSI < 48": rsi < 48,
        "MACD < Signal": macd < macd_signal,
        "ADX >= 18": adx >= 18,
        "Bearish Candle": bearish_candle,
        "Price < BB Mid": close < bb_mid,
        "Downtrend Regime": regime in [
            "DOWNTREND",
            "STRONG DOWNTREND",
        ],
    }

    buy_score = sum(
        bool(value)
        for value in buy_checks.values()
    )

    sell_score = sum(
        bool(value)
        for value in sell_checks.values()
    )

    max_score = len(buy_checks)

    if regime == "CHOPPY / RANGE":
        if max(buy_score, sell_score) < 8:
            signal = "WAIT"

        elif buy_score > sell_score:
            signal = "BUY"

        elif sell_score > buy_score:
            signal = "SELL"

        else:
            signal = "WAIT"

    else:
        if (
            buy_score >= minimum_score
            and buy_score > sell_score
        ):
            signal = (
                "STRONG BUY"
                if buy_score >= 8
                else "BUY"
            )

        elif (
            sell_score >= minimum_score
            and sell_score > buy_score
        ):
            signal = (
                "STRONG SELL"
                if sell_score >= 8
                else "SELL"
            )

        else:
            signal = "WAIT"

    if signal in ["BUY", "STRONG BUY"]:
        score = buy_score

    elif signal in ["SELL", "STRONG SELL"]:
        score = sell_score

    else:
        score = max(
            buy_score,
            sell_score,
        )

    confidence = (
        score / max_score * 100
        if max_score
        else 0
    )

    return {
        "signal": signal,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "score": score,
        "max_score": max_score,
        "confidence": confidence,
        "regime": regime,
        "buy_checks": buy_checks,
        "sell_checks": sell_checks,
    }


def build_trade(row, signal, rr_values):
    if signal not in [
        "BUY",
        "STRONG BUY",
        "SELL",
        "STRONG SELL",
    ]:
        return None

    entry = float(row["close"])
    atr = float(row["ATR"])

    rr1_value, rr2_value, rr3_value = rr_values

    if signal in [
        "BUY",
        "STRONG BUY",
    ]:
        sl = float(row["buy_sl"])

        risk_distance = entry - sl

        if risk_distance <= 0:
            risk_distance = 1.5 * atr

        tp1 = entry + risk_distance * rr1_value
        tp2 = entry + risk_distance * rr2_value
        tp3 = entry + risk_distance * rr3_value

        direction = "BUY"

    else:
        sl = float(row["sell_sl"])

        risk_distance = sl - entry

        if risk_distance <= 0:
            risk_distance = 1.5 * atr

        tp1 = entry - risk_distance * rr1_value
        tp2 = entry - risk_distance * rr2_value
        tp3 = entry - risk_distance * rr3_value

        direction = "SELL"

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "risk_distance": risk_distance,
    }


def backtest_strategy(
    df,
    minimum_score,
    rr,
):
    if len(df) < 120:
        return pd.DataFrame(), {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "max_drawdown_r": 0.0,
        }

    rows = []

    for i in range(100, len(df) - 1):
        signal_row = df.iloc[i]

        signal_info = get_signal(
            signal_row,
            minimum_score,
        )

        signal = signal_info["signal"]

        if signal not in [
            "BUY",
            "STRONG BUY",
            "SELL",
            "STRONG SELL",
        ]:
            continue

        entry_row = df.iloc[i + 1]

        entry = float(entry_row["open"])
        atr = float(signal_row["ATR"])

        if not np.isfinite(entry) or not np.isfinite(atr):
            continue

        if signal in [
            "BUY",
            "STRONG BUY",
        ]:
            recent_low = float(
                signal_row["recent_low"]
            )

            sl = min(
                entry - 1.5 * atr,
                recent_low - 0.10 * atr,
            )

            risk_distance = entry - sl

            if risk_distance <= 0:
                continue

            tp = entry + risk_distance * rr

            result = None
            exit_time = None

            for j in range(
                i + 1,
                len(df),
            ):
                bar = df.iloc[j]

                bar_low = float(bar["low"])
                bar_high = float(bar["high"])

                hit_sl = bar_low <= sl
                hit_tp = bar_high >= tp

                if hit_sl and hit_tp:
                    result = "LOSS"
                    exit_time = bar["datetime"]
                    break

                if hit_sl:
                    result = "LOSS"
                    exit_time = bar["datetime"]
                    break

                if hit_tp:
                    result = "WIN"
                    exit_time = bar["datetime"]
                    break

            if result is None:
                continue

            r_value = (
                1.0
                if result == "WIN"
                else -1.0
            )

        else:
            recent_high = float(
                signal_row["recent_high"]
            )

            sl = max(
                entry + 1.5 * atr,
                recent_high + 0.10 * atr,
            )

            risk_distance = sl - entry

            if risk_distance <= 0:
                continue

            tp = entry - risk_distance * rr

            result = None
            exit_time = None

            for j in range(
                i + 1,
                len(df),
            ):
                bar = df.iloc[j]

                bar_low = float(bar["low"])
                bar_high = float(bar["high"])

                hit_sl = bar_high >= sl
                hit_tp = bar_low <= tp

                if hit_sl and hit_tp:
                    result = "LOSS"
                    exit_time = bar["datetime"]
                    break

                if hit_sl:
                    result = "LOSS"
                    def add_trade_levels(df):
    d = df.copy()

    d["recent_low"] = (
        d["low"]
        .rolling(6)
        .min()
        .shift(1)
    )

    d["recent_high"] = (
        d["high"]
        .rolling(6)
        .max()
        .shift(1)
    )

    d["buy_sl"] = np.minimum(
        d["close"] - 1.5 * d["ATR"],
        d["recent_low"] - 0.10 * d["ATR"],
    )

    d["sell_sl"] = np.maximum(
        d["close"] + 1.5 * d["ATR"],
        d["recent_high"] + 0.10 * d["ATR"],
    )

    return d


def market_regime(row):
    close = float(row["close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema200 = float(row["EMA200"])
    adx = float(row["ADX"])
    atr = float(row["ATR"])

    atr_ratio = (
        atr / close
        if close != 0
        else 0
    )

    if (
        adx >= 25
        and close > ema20 > ema50 > ema200
    ):
        return "STRONG UPTREND"

    if (
        adx >= 25
        and close < ema20 < ema50 < ema200
    ):
        return "STRONG DOWNTREND"

    if (
        adx >= 18
        and close > ema50
        and ema20 > ema50
    ):
        return "UPTREND"

    if (
        adx >= 18
        and close < ema50
        and ema20 < ema50
    ):
        return "DOWNTREND"

    if (
        adx < 15
        and 0.0075 <= atr_ratio <= 0.0135
    ):
        return "CHOPPY / RANGE"

    if atr_ratio >= 0.015:
        return "HIGH VOLATILITY"

    return "MIXED"


def get_signal(row, minimum_score=6):
    regime = market_regime(row)

    close = float(row["close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema200 = float(row["EMA200"])
    rsi = float(row["RSI"])
    macd = float(row["MACD"])
    macd_signal = float(row["MACD_signal"])
    adx = float(row["ADX"])
    bb_mid = float(row["BB_mid"])
    body_ratio = float(row["body_ratio"])

    bullish_candle = (
        close > float(row["open"])
        and body_ratio >= 0.25
    )

    bearish_candle = (
        close < float(row["open"])
        and body_ratio >= 0.25
    )

    buy_checks = {
        "Price > EMA20": close > ema20,
        "EMA20 > EMA50": ema20 > ema50,
        "EMA50 > EMA200": ema50 > ema200,
        "RSI > 52": rsi > 52,
        "MACD > Signal": macd > macd_signal,
        "ADX >= 18": adx >= 18,
        "Bullish Candle": bullish_candle,
        "Price > BB Mid": close > bb_mid,
        "Uptrend Regime": regime in [
            "UPTREND",
            "STRONG UPTREND",
        ],
    }

    sell_checks = {
        "Price < EMA20": close < ema20,
        "EMA20 < EMA50": ema20 < ema50,
        "EMA50 < EMA200": ema50 < ema200,
        "RSI < 48": rsi < 48,
        "MACD < Signal": macd < macd_signal,
        "ADX >= 18": adx >= 18,
        "Bearish Candle": bearish_candle,
        "Price < BB Mid": close < bb_mid,
        "Downtrend Regime": regime in [
            "DOWNTREND",
            "STRONG DOWNTREND",
        ],
    }

    buy_score = sum(
        bool(value)
        for value in buy_checks.values()
    )

    sell_score = sum(
        bool(value)
        for value in sell_checks.values()
    )

    max_score = len(buy_checks)

    if regime == "CHOPPY / RANGE":
        if max(buy_score, sell_score) < 8:
            signal = "WAIT"

        elif buy_score > sell_score:
            signal = "BUY"

        elif sell_score > buy_score:
            signal = "SELL"

        else:
            signal = "WAIT"

    else:
        if (
            buy_score >= minimum_score
            and buy_score > sell_score
        ):
            signal = (
                "STRONG BUY"
                if buy_score >= 8
                else "BUY"
            )

        elif (
            sell_score >= minimum_score
            and sell_score > buy_score
        ):
            signal = (
                "STRONG SELL"
                if sell_score >= 8
                else "SELL"
            )

        else:
            signal = "WAIT"

    if signal in ["BUY", "STRONG BUY"]:
        score = buy_score

    elif signal in ["SELL", "STRONG SELL"]:
        score = sell_score

    else:
        score = max(
            buy_score,
            sell_score,
        )

    confidence = (
        score / max_score * 100
        if max_score
        else 0
    )

    return {
        "signal": signal,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "score": score,
        "max_score": max_score,
        "confidence": confidence,
        "regime": regime,
        "buy_checks": buy_checks,
        "sell_checks": sell_checks,
    }


def build_trade(row, signal, rr_values):
    if signal not in [
        "BUY",
        "STRONG BUY",
        "SELL",
        "STRONG SELL",
    ]:
        return None

    entry = float(row["close"])
    atr = float(row["ATR"])

    rr1_value, rr2_value, rr3_value = rr_values

    if signal in [
        "BUY",
        "STRONG BUY",
    ]:
        sl = float(row["buy_sl"])

        risk_distance = entry - sl

        if risk_distance <= 0:
            risk_distance = 1.5 * atr

        tp1 = entry + risk_distance * rr1_value
        tp2 = entry + risk_distance * rr2_value
        tp3 = entry + risk_distance * rr3_value

        direction = "BUY"

    else:
        sl = float(row["sell_sl"])

        risk_distance = sl - entry

        if risk_distance <= 0:
            risk_distance = 1.5 * atr

        tp1 = entry - risk_distance * rr1_value
        tp2 = entry - risk_distance * rr2_value
        tp3 = entry - risk_distance * rr3_value

        direction = "SELL"

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "risk_distance": risk_distance,
    }


def backtest_strategy(
    df,
    minimum_score,
    rr,
):
    if len(df) < 120:
        return pd.DataFrame(), {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "max_drawdown_r": 0.0,
        }

    rows = []

    for i in range(100, len(df) - 1):
        signal_row = df.iloc[i]

        signal_info = get_signal(
            signal_row,
            minimum_score,
        )

        signal = signal_info["signal"]

        if signal not in [
            "BUY",
            "STRONG BUY",
            "SELL",
            "STRONG SELL",
        ]:
            continue

        entry_row = df.iloc[i + 1]

        entry = float(entry_row["open"])
        atr = float(signal_row["ATR"])

        if not np.isfinite(entry) or not np.isfinite(atr):
            continue

        if signal in [
            "BUY",
            "STRONG BUY",
        ]:
            recent_low = float(
                signal_row["recent_low"]
            )

            sl = min(
                entry - 1.5 * atr,
                recent_low - 0.10 * atr,
            )

            risk_distance = entry - sl

            if risk_distance <= 0:
                continue

            tp = entry + risk_distance * rr

            result = None
            exit_time = None

            for j in range(
                i + 1,
                len(df),
            ):
                bar = df.iloc[j]

                bar_low = float(bar["low"])
                bar_high = float(bar["high"])

                hit_sl = bar_low <= sl
                hit_tp = bar_high >= tp

                if hit_sl and hit_tp:
                    result = "LOSS"
                    exit_time = bar["datetime"]
                    break

                if hit_sl:
                    result = "LOSS"
                    exit_time = bar["datetime"]
                    break

                if hit_tp:
                    result = "WIN"
                    exit_time = bar["datetime"]
                    break

            if result is None:
                continue

            r_value = (
                1.0
                if result == "WIN"
                else -1.0
            )

        else:
            recent_high = float(
                signal_row["recent_high"]
            )

            sl = max(
                entry + 1.5 * atr,
                recent_high + 0.10 * atr,
            )

            risk_distance = sl - entry

            if risk_distance <= 0:
                continue

            tp = entry - risk_distance * rr

            result = None
            exit_time = None

            for j in range(
                i + 1,
                len(df),
            ):
                bar = df.iloc[j]

                bar_low = float(bar["low"])
                bar_high = float(bar["high"])

                hit_sl = bar_high >= sl
                hit_tp = bar_low <= tp

                if hit_sl and hit_tp:
                    result = "LOSS"
                    exit_time = bar["datetime"]
                    break

                if hit_sl:
                    result = "LOSS"
                    exit_time = bar["datetime"]
                    break

                if hit_tp:
                    result = "WIN"
                    exit_time = bar["datetime"]
                    break

            if result is None:
                continue

            r_value = (
                1.0
                if result == "WIN"
                else -1.0
            )

        rows.append(
            {
                "time": signal_row["datetime"],
                "exit_time": exit_time,
                "direction": (
                    "BUY"
                    if signal in [
                        "BUY",
                        "STRONG BUY",
                    ]
                    else "SELL"
                ),
                "signal": signal,
                "score": signal_info["score"],
                "confidence": signal_info["confidence"],
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "result": result,
                "R": r_value,
            }
        )

    trades_df = pd.DataFrame(rows)

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

    wins = int(
        (trades_df["result"] == "WIN").sum()
    )

    losses = int(
        (trades_df["result"] == "LOSS").sum()
    )

    win_rate = (
        wins /
        len(trades_df) *
        100
    )

    gross_profit = float(
        trades_df.loc[
            trades_df["R"] > 0,
            "R",
        ].sum()
    )

    gross_loss = abs(
        float(
            trades_df.loc[
                trades_df["R"] < 0,
                "R",
            ].sum()
        )
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = trades_df["R"].cumsum()

    peak = equity.cummax()

    drawdown = equity - peak

    max_dd = (
        abs(float(drawdown.min()))
        if len(drawdown)
        else 0.0
    )

    return trades_df, {
        "trades": len(trades_df),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_r": float(
            trades_df["R"].sum()
        ),
        "max_drawdown_r": max_dd,
    }


def direction_stats(trades_df):
    if trades_df.empty:
        return pd.DataFrame()

    rows = []

    for direction in [
        "BUY",
        "SELL",
    ]:
        subset = trades_df[
            trades_df["direction"] == direction
        ]

        if subset.empty:
            rows.append(
                {
                    "Direction": direction,
                    "Trades": 0,
                    "Wins": 0,
                    "Losses": 0,
                    "Accuracy %": 0.0,
                    "Net R": 0.0,
                }
            )
            continue

        wins = int(
            (
                subset["result"] == "WIN"
            ).sum()
        )

        losses = int(
            (
                subset["result"] == "LOSS"
            ).sum()
        )

        rows.append(
            {
                "Direction": direction,
                "Trades": len(subset),
                "Wins": wins,
                "Losses": losses,
                "Accuracy %": (
                    wins /
                    len(subset) *
                    100
                ),
                "Net R": float(
                    subset["R"].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def mtf_analysis(
    api_key,
    symbol,
    outputsize,
    minimum_score,
):
    rows = []

    for tf in MTF_INTERVALS:
        try:
            raw, _ = fetch_candles(
                api_key,
                symbol,
                tf,
                outputsize,
            )

            data = add_trade_levels(
                add_indicators(raw)
            )

            if len(data) > 2:
                row = data.iloc[-2]
            else:
                row = data.iloc[-1]

            info = get_signal(
                row,
                minimum_score,
            )

            rows.append(
                {
                    "Timeframe": tf,
                    "Signal": info["signal"],
                    "Score": (
                        f"{info['score']}/"
                        f"{info['max_score']}"
                    ),
                    "Confidence": (
                        f"{info['confidence']:.0f}%"
                    ),
                    "Regime": info["regime"],
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "Timeframe": tf,
                    "Signal": "ERROR",
                    "Score": "-",
                    "Confidence": "-",
                    "Regime": str(exc)[:80],
                }
            )

    return pd.DataFrame(rows)


def mtf_direction(mtf_df):
    if mtf_df.empty:
        return "OFF"

    buy_count = 0
    sell_count = 0

    for value in mtf_df["Signal"].astype(str):
        if value in [
            "BUY",
            "STRONG BUY",
        ]:
            buy_count += 1

        elif value in [
            "SELL",
            "STRONG SELL",
        ]:
            sell_count += 1

    if buy_count >= 3:
        return "BUY"

    if sell_count >= 3:
        return "SELL"

    return "CONFLICT"


try:
    raw_df, meta = fetch_candles(
        api_key,
        symbol,
        interval,
        outputsize,
    )

    df = add_trade_levels(
        add_indicators(raw_df)
    )

except Exception as exc:
    st.error(
        f"Market data error: {exc}"
    )
    st.stop()


closed_candle = (
    signal_mode == "Closed candle (stable)"
)

if closed_candle:
    signal_index = -2
else:
    signal_index = -1

if len(df) < 3:
    st.error(
        "Not enough candles available for signal calculation."
    )
    st.stop()

signal_row = df.iloc[signal_index]

signal_info = get_signal(
    signal_row,
    min_score,
)

trade = build_trade(
    signal_row,
    signal_info["signal"],
    (rr1, rr2, rr3),
)

mtf_df = pd.DataFrame()
mtf_dir = "OFF"

if mtf_enabled:
    with st.spinner(
        "Checking multi-timeframe confirmation..."
    ):
        mtf_df = mtf_analysis(
            api_key,
            symbol,
            outputsize,
            min_score,
        )

    mtf_dir = mtf_direction(
        mtf_df
    )


raw_signal = signal_info["signal"]

final_signal = raw_signal

if (
    mtf_enabled
    and strict_mtf
    and raw_signal in [
        "BUY",
        "STRONG BUY",
    ]
    and mtf_dir == "SELL"
):
    final_signal = "WAIT"

    trade = None

elif (
    mtf_enabled
    and strict_mtf
    and raw_signal in [
        "SELL",
        "STRONG SELL",
    ]
    and mtf_dir == "BUY"
):
    final_signal = "WAIT"

    trade = None
    # =========================
# MAIN DASHBOARD
# =========================

st.divider()

latest_price = float(signal_row["close"])

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Price",
        f"{latest_price:,.3f}",
    )

with col2:
    st.metric(
        "Final Signal",
        final_signal,
    )

with col3:
    st.metric(
        "Technical Score",
        f"{signal_info['score']}/{signal_info['max_score']}",
    )

with col4:
    st.metric(
        "Confidence",
        f"{signal_info['confidence']:.0f}%",
    )

with col5:
    st.metric(
        "Market Regime",
        signal_info["regime"],
    )


st.subheader("🎯 Final Trade Analysis")

if trade is not None and final_signal != "WAIT":
    direction = trade["direction"]

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Direction",
            direction,
        )

    with c2:
        st.metric(
            "Entry",
            f"{trade['entry']:,.3f}",
        )

    with c3:
        st.metric(
            "Stop Loss",
            f"{trade['sl']:,.3f}",
        )

    with c4:
        st.metric(
            "TP1",
            f"{trade['tp1']:,.3f}",
        )

    with c5:
        st.metric(
            "TP2",
            f"{trade['tp2']:,.3f}",
        )

    st.metric(
        "TP3",
        f"{trade['tp3']:,.3f}",
    )

    risk_distance = trade["risk_distance"]

    risk_budget = (
        account_balance *
        risk_pct /
        100
    )

    st.info(
        f"Risk distance: {risk_distance:,.3f} | "
        f"Risk budget: ${risk_budget:,.2f} | "
        f"Target RR: 1:{rr2:g}"
    )

    if mtf_enabled:
        if mtf_dir == direction:
            st.success(
                f"✅ MTF confirmation agrees with {direction}."
            )

        elif mtf_dir == "CONFLICT":
            st.warning(
                "⚠️ MTF direction is mixed/conflicting."
            )

        else:
            st.warning(
                f"MTF direction: {mtf_dir}"
            )

else:
    st.warning(
        "⏸️ WAIT — current conditions do not meet "
        "the required technical confirmation."
    )


st.divider()

st.subheader("🧠 Decision Details")

factor_rows = []

for name, value in signal_info["buy_checks"].items():
    factor_rows.append(
        {
            "BUY Factor": name,
            "BUY": "✅" if value else "❌",
        }
    )

factor_df = pd.DataFrame(
    factor_rows
)

sell_factor_rows = []

for name, value in signal_info["sell_checks"].items():
    sell_factor_rows.append(
        {
            "SELL Factor": name,
            "SELL": "✅" if value else "❌",
        }
    )

sell_factor_df = pd.DataFrame(
    sell_factor_rows
)

factor_col1, factor_col2 = st.columns(2)

with factor_col1:
    st.markdown("**BUY Confirmation**")
    st.dataframe(
        factor_df,
        use_container_width=True,
        hide_index=True,
    )

with factor_col2:
    st.markdown("**SELL Confirmation**")
    st.dataframe(
        sell_factor_df,
        use_container_width=True,
        hide_index=True,
    )


st.divider()

st.subheader("📈 Historical Accuracy / Backtest")

with st.spinner("Running historical backtest..."):
    trades_df, backtest_stats = backtest_strategy(
        df,
        min_score,
        rr2,
    )

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric(
        "Trades",
        backtest_stats["trades"],
    )

with m2:
    st.metric(
        "Win Rate",
        f"{backtest_stats['win_rate']:.1f}%",
    )

with m3:
    pf = backtest_stats["profit_factor"]

    if np.isinf(pf):
        pf_text = "∞"
    else:
        pf_text = f"{pf:.2f}"

    st.metric(
        "Profit Factor",
        pf_text,
    )

with m4:
    st.metric(
        "Net R",
        f"{backtest_stats['net_r']:.2f}",
    )

with m5:
    st.metric(
        "Max DD",
        f"{backtest_stats['max_drawdown_r']:.2f}R",
    )


direction_df = direction_stats(
    trades_df
)

if not direction_df.empty:
    st.markdown("**BUY vs SELL Performance**")

    st.dataframe(
        direction_df,
        use_container_width=True,
        hide_index=True,
    )


if not trades_df.empty:
    with st.expander(
        "📋 View Backtest Trade History"
    ):
        display_df = trades_df.copy()

        if "time" in display_df.columns:
            display_df["time"] = (
                display_df["time"]
                .astype(str)
            )

        if "exit_time" in display_df.columns:
            display_df["exit_time"] = (
                display_df["exit_time"]
                .astype(str)
            )

        st.dataframe(
            display_df.sort_values(
                "time",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )


st.divider()

st.subheader("🌐 Multi-Timeframe Confirmation")

if mtf_enabled and not mtf_df.empty:
    st.dataframe(
        mtf_df,
        use_container_width=True,
        hide_index=True,
    )

    if mtf_dir == "BUY":
        st.success(
            "MTF overall direction: BUY"
        )

    elif mtf_dir == "SELL":
        st.error(
            "MTF overall direction: SELL"
        )

    else:
        st.warning(
            "MTF overall direction: CONFLICT / NO CLEAR DIRECTION"
        )

else:
    st.info(
        "MTF confirmation is disabled."
    )


st.divider()

st.subheader("📊 Price Chart")

chart_df = df.tail(
    min(180, len(df))
).copy()

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=chart_df["datetime"],
        open=chart_df["open"],
        high=chart_df["high"],
        low=chart_df["low"],
        close=chart_df["close"],
        name="Price",
    )
)

fig.add_trace(
    go.Scatter(
        x=chart_df["datetime"],
        y=chart_df["EMA20"],
        mode="lines",
        name="EMA20",
    )
)

fig.add_trace(
    go.Scatter(
        x=chart_df["datetime"],
        y=chart_df["EMA50"],
        mode="lines",
        name="EMA50",
    )
)

fig.add_trace(
    go.Scatter(
        x=chart_df["datetime"],
        y=chart_df["EMA200"],
        mode="lines",
        name="EMA200",
    )
)

if trade is not None:
    fig.add_hline(
        y=trade["entry"],
        line_dash="dash",
        annotation_text="ENTRY",
    )

    fig.add_hline(
        y=trade["sl"],
        line_dash="dot",
        annotation_text="SL",
    )

    fig.add_hline(
        y=trade["tp1"],
        line_dash="dot",
        annotation_text="TP1",
    )

    fig.add_hline(
        y=trade["tp2"],
        line_dash="dot",
        annotation_text="TP2",
    )

fig.update_layout(
    height=650,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


st.divider()

with st.expander(
    "📌 Current Indicator Values"
):
    indicator_values = {
        "Close": latest_price,
        "EMA20": float(
            signal_row["EMA20"]
        ),
        "EMA50": float(
            signal_row["EMA50"]
        ),
        "EMA200": float(
            signal_row["EMA200"]
        ),
        "RSI": float(
            signal_row["RSI"]
        ),
        "ATR": float(
            signal_row["ATR"]
        ),
        "ADX": float(
            signal_row["ADX"]
        ),
        "MACD": float(
            signal_row["MACD"]
        ),
        "MACD Signal": float(
            signal_row["MACD_signal"]
        ),
        "BB Mid": float(
            signal_row["BB_mid"]
        ),
    }

    indicator_df = pd.DataFrame(
        [
            {
                "Indicator": key,
                "Value": value,
            }
            for key, value in indicator_values.items()
        ]
    )

    st.dataframe(
        indicator_df,
        use_container_width=True,
        hide_index=True,
    )


st.divider()

st.subheader("ℹ️ System Status")

status_col1, status_col2 = st.columns(2)

with status_col1:
    st.write(
        f"**Market:** {selected_name}"
    )

    st.write(
        f"**Timeframe:** {timeframe_label}"
    )

    st.write(
        f"**Signal Mode:** {signal_mode}"
    )

    st.write(
        f"**Candles Loaded:** {len(df)}"
    )

with status_col2:
    st.write(
        f"**Raw Signal:** {raw_signal}"
    )

    st.write(
        f"**Final Signal:** {final_signal}"
    )

    st.write(
        f"**MTF Direction:** {mtf_dir}"
    )

    st.write(
        f"**Last Candle:** {signal_row['datetime']}"
    )


st.caption(
    "⚠️ This dashboard provides technical-analysis estimates only. "
    "It does not guarantee profitable trades. Always manage risk "
    "and verify market conditions before trading."
)

st.caption(
    f"Last dashboard update: "
    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
)
           
