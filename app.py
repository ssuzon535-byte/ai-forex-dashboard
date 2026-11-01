
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(
    page_title="AI Forex Multi-Timeframe Dashboard",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# SETTINGS
# =========================================================

TIMEFRAMES = {
    "Daily": "1day",
    "4H": "4h",
    "1H": "1h",
    "15M": "15min",
    "5M": "5min"
}

PAIRS = [
    "XAU/USD", "XAG/USD", "EUR/USD", "GBP/USD",
    "USD/JPY", "USD/CHF", "AUD/USD", "NZD/USD",
    "USD/CAD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
    "AUD/JPY", "EUR/AUD", "GBP/AUD"
]


# =========================================================
# API
# =========================================================

def safe_secrets(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def fetch_data(symbol, interval, outputsize, api_key):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON"
    }

    response = requests.get(
        url,
        params=params,
        timeout=25
    )

    response.raise_for_status()
    payload = response.json()

    if "values" not in payload:
        raise ValueError(
            payload.get(
                "message",
                payload.get("code", "No candle data returned")
            )
        )

    df = pd.DataFrame(payload["values"])

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    for col in ["open", "high", "low", "close", "volume"]:

        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        elif col == "volume":
            df[col] = np.nan

    df = df.dropna(
        subset=["datetime", "open", "high", "low", "close"]
    )

    df = (
        df.sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    if len(df) < 60:
        raise ValueError(
            f"Not enough candles for {interval}. Received {len(df)}"
        )

    return df


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def add_indicators(df):

    d = df.copy()

    c = d.close
    h = d.high
    l = d.low

    # EMA
    for n in [20, 50, 100, 200]:
        d[f"EMA{n}"] = c.ewm(
            span=n,
            adjust=False
        ).mean()

    # RSI
    delta = c.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)

    d["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    tr = pd.concat(
        [
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    d["ATR"] = tr.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    # ADX
    up = h.diff()
    down = -l.diff()

    plus_dm = pd.Series(
        np.where(
            (up > down) & (up > 0),
            up,
            0
        ),
        index=d.index
    )

    minus_dm = pd.Series(
        np.where(
            (down > up) & (down > 0),
            down,
            0
        ),
        index=d.index
    )

    atr_safe = d["ATR"].replace(0, np.nan)

    plus_di = (
        100
        * plus_dm.ewm(alpha=1/14, adjust=False).mean()
        / atr_safe
    )

    minus_di = (
        100
        * minus_dm.ewm(alpha=1/14, adjust=False).mean()
        / atr_safe
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    d["ADX"] = dx.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()

    d["MACD"] = ema12 - ema26

    d["MACD_SIGNAL"] = d.MACD.ewm(
        span=9,
        adjust=False
    ).mean()

    d["MACD_HIST"] = d.MACD - d.MACD_SIGNAL

    # Bollinger Bands
    d["BB_MID"] = c.rolling(20).mean()

    std = c.rolling(20).std()

    d["BB_UPPER"] = d.BB_MID + 2 * std
    d["BB_LOWER"] = d.BB_MID - 2 * std

    # VWAP, only when volume exists
    typical = (h + l + c) / 3

    if (
        d.volume.notna().sum() > 10
        and d.volume.fillna(0).sum() > 0
    ):

        date_key = d.datetime.dt.date

        pv = typical * d.volume.fillna(0)

        d["VWAP"] = (
            pv.groupby(date_key).cumsum()
            /
            d.volume.fillna(0)
            .groupby(date_key)
            .cumsum()
            .replace(0, np.nan)
        )

    else:
        d["VWAP"] = np.nan

    # Candle information
    d["BODY"] = (d.close - d.open).abs()

    d["RANGE"] = (
        d.high - d.low
    ).replace(0, np.nan)

    d["BULL_CANDLE"] = d.close > d.open
    d["BEAR_CANDLE"] = d.close < d.open

    # Market structure
    d["SWING_HIGH"] = d.high.rolling(
        5,
        center=True
    ).max()

    d["SWING_LOW"] = d.low.rolling(
        5,
        center=True
    ).min()

    d["PRIOR_HIGH"] = (
        d.high.shift(1).rolling(20).max()
    )

    d["PRIOR_LOW"] = (
        d.low.shift(1).rolling(20).min()
    )

    d["BOS_BULL"] = d.close > d.PRIOR_HIGH
    d["BOS_BEAR"] = d.close < d.PRIOR_LOW

    d["CHOCH_BULL"] = (
        d.BOS_BULL
        & (d.EMA20.shift(1) < d.EMA50.shift(1))
    )

    d["CHOCH_BEAR"] = (
        d.BOS_BEAR
        & (d.EMA20.shift(1) > d.EMA50.shift(1))
    )

    return d


# =========================================================
# TREND ANALYSIS
# =========================================================

def trend_bias(df):

    r = df.iloc[-1]

    if pd.isna(r.EMA50) or pd.isna(r.EMA200):
        return "NEUTRAL", 0

    score = 0

    score += 1 if r.close > r.EMA50 else -1
    score += 1 if r.EMA50 > r.EMA200 else -1
    score += 1 if r.MACD > r.MACD_SIGNAL else -1
    score += 1 if r.RSI >= 50 else -1

    if r.ADX >= 20:
        score += 1 if r.close > r.EMA50 else -1

    if score >= 2:
        bias = "BULLISH"

    elif score <= -2:
        bias = "BEARISH"

    else:
        bias = "NEUTRAL"

    return bias, score


# =========================================================
# CANDLE PATTERN
# =========================================================

def candle_pattern(row, prev=None):

    body = (
        float(row.BODY)
        if pd.notna(row.BODY)
        else 0
    )

    rng = (
        float(row.RANGE)
        if pd.notna(row.RANGE)
        else 0
    )

    if rng <= 0:
        return "None"

    upper = row.high - max(row.open, row.close)

    lower = min(row.open, row.close) - row.low

    if body / rng < 0.10:
        return "Doji"

    if lower > body * 2 and upper < body:
        return "Bullish Pin Bar"

    if upper > body * 2 and lower < body:
        return "Bearish Pin Bar"

    if prev is not None:

        bull_engulf = (
            row.close > row.open
            and prev.close < prev.open
            and row.close >= prev.open
            and row.open <= prev.close
        )

        bear_engulf = (
            row.close < row.open
            and prev.close > prev.open
            and row.open >= prev.close
            and row.close <= prev.open
        )

        if bull_engulf:
            return "Bullish Engulfing"

        if bear_engulf:
            return "Bearish Engulfing"

    return "None"


# =========================================================
# SUPPLY AND DEMAND
# =========================================================

def add_zones(df):

    d = df.copy()

    d["DEMAND_LOW"] = np.nan
    d["DEMAND_HIGH"] = np.nan

    d["SUPPLY_LOW"] = np.nan
    d["SUPPLY_HIGH"] = np.nan

    if len(d) >= 12:

        recent = d.tail(100)

        lows = recent.low.nsmallest(3).index
        highs = recent.high.nlargest(3).index

        for idx in lows:

            d.loc[idx, "DEMAND_LOW"] = d.loc[idx, "low"]

            d.loc[idx, "DEMAND_HIGH"] = (
                d.loc[idx, "low"]
                + d.loc[idx, "ATR"] * 0.5
            )

        for idx in highs:

            d.loc[idx, "SUPPLY_HIGH"] = d.loc[idx, "high"]

            d.loc[idx, "SUPPLY_LOW"] = (
                d.loc[idx, "high"]
                - d.loc[idx, "ATR"] * 0.5
            )

    return d


# =========================================================
# ENTRY, SL AND TARGETS
# =========================================================

def build_trade_plan(df, direction):

    r = df.iloc[-1]

    entry = float(r.close)

    atr = (
        float(r.ATR)
        if pd.notna(r.ATR) and r.ATR > 0
        else entry * 0.001
    )

    recent = df.tail(20)

    if direction == "BUY":

        swing_sl = float(recent.low.min())

        sl = min(
            entry - atr * 1.5,
            swing_sl - atr * 0.1
        )

        risk = max(entry - sl, atr)

        return {
            "Entry": entry,
            "SL": sl,
            "TP1": entry + risk,
            "TP2": entry + risk * 2,
            "TP3": entry + risk * 3,
            "RiskDistance": risk
        }

    swing_sl = float(recent.high.max())

    sl = max(
        entry + atr * 1.5,
        swing_sl + atr * 0.1
    )

    risk = max(sl - entry, atr)

    return {
        "Entry": entry,
        "SL": sl,
        "TP1": entry - risk,
        "TP2": entry - risk * 2,
        "TP3": entry - risk * 3,
        "RiskDistance": risk
    }


# =========================================================
# SIGNAL ENGINE
# =========================================================

def evaluate_signal(frames):

    biases = {
        name: trend_bias(df)[0]
        for name, df in frames.items()
    }

    scores = {
        name: trend_bias(df)[1]
        for name, df in frames.items()
    }

    m5 = frames["5M"]

    r = m5.iloc[-1]
    prev = m5.iloc[-2]

    pattern = candle_pattern(r, prev)

    bullish = sum(
        biases[x] == "BULLISH"
        for x in ["Daily", "4H", "1H"]
    )

    bearish = sum(
        biases[x] == "BEARISH"
        for x in ["Daily", "4H", "1H"]
    )

    aligned_buy = (
        biases["15M"] == "BULLISH"
        and r.close > r.EMA20
        and r.RSI > 50
        and r.MACD > r.MACD_SIGNAL
        and r.BULL_CANDLE
    )

    aligned_sell = (
        biases["15M"] == "BEARISH"
        and r.close < r.EMA20
        and r.RSI < 50
        and r.MACD < r.MACD_SIGNAL
        and r.BEAR_CANDLE
    )

    # Quality score
    quality = int(
        min(
            100,
            max(
                0,
                35
                + abs(sum(scores.values())) * 4
                + min(
                    float(r.ADX)
                    if pd.notna(r.ADX)
                    else 0,
                    35
                ) * 0.5
            )
        )
    )

    # False signal filters
    body_ratio = (
        r.BODY / r.RANGE
        if pd.notna(r.RANGE) and r.RANGE > 0
        else 0
    )

    if (
        bullish >= 2
        and aligned_buy
        and r.ADX >= 15
        and r.RSI < 75
        and body_ratio >= 0.25
    ):

        direction = "BUY"

        signal = (
            "STRONG BUY"
            if bullish == 3 and quality >= 75
            else "BUY"
        )

    elif (
        bearish >= 2
        and aligned_sell
        and r.ADX >= 15
        and r.RSI > 25
        and body_ratio >= 0.25
    ):

        direction = "SELL"

        signal = (
            "STRONG SELL"
            if bearish == 3 and quality >= 75
            else "SELL"
        )

    else:

        direction = "WAIT"
        signal = "WAIT"

        quality = min(quality, 69)

    return {
        "signal": signal,
        "direction": direction,
        "quality": quality,
        "biases": biases,
        "scores": scores,
        "pattern": pattern
    }


# =========================================================
# CHART
# =========================================================

def make_chart(df, plan=None, signal_rows=None):

    d = df.tail(250).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=d.datetime,
            open=d.open,
            high=d.high,
            low=d.low,
            close=d.close,
            name="Price"
        )
    )

    for col, color in [
        ("EMA20", "#f5a623"),
        ("EMA50", "#268bd2"),
        ("EMA200", "#d62728")
    ]:

        fig.add_trace(
            go.Scatter(
                x=d.datetime,
                y=d[col],
                name=col,
                mode="lines",
                line=dict(
                    width=1.3,
                    color=color
                )
            )
        )

    fig.add_trace(
        go.Scatter(
            x=d.datetime,
            y=d.BB_UPPER,
            name="BB Upper",
            line=dict(width=1, dash="dot")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d.datetime,
            y=d.BB_LOWER,
            name="BB Lower",
            line=dict(width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(100,100,100,0.06)"
        )
    )

    if d.VWAP.notna().any():

        fig.add_trace(
            go.Scatter(
                x=d.datetime,
                y=d.VWAP,
                name="VWAP",
                line=dict(width=1.2, dash="dash")
            )
        )

    if signal_rows is not None and not signal_rows.empty:

        buys = signal_rows[
            signal_rows.direction == "BUY"
        ]

        sells = signal_rows[
            signal_rows.direction == "SELL"
        ]

        fig.add_trace(
            go.Scatter(
                x=buys.datetime,
                y=buys.low,
                mode="markers",
                name="BUY History",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="#00c853"
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=sells.datetime,
                y=sells.high,
                mode="markers",
                name="SELL History",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="#ff1744"
                )
            )
        )

        # Historical SL/TP boxes
        for _, trade in signal_rows.tail(15).iterrows():

            entry = float(trade.close)

            atr = (
                float(trade.ATR)
                if pd.notna(trade.ATR) and trade.ATR > 0
                else entry * 0.001
            )

            risk = atr * 1.5

            is_buy = trade.direction == "BUY"

            stop = (
                entry - risk
                if is_buy
                else entry + risk
            )

            target = (
                entry + risk * 2
                if is_buy
                else entry - risk * 2
            )

            x0 = trade.datetime

            x1 = x0 + pd.Timedelta(minutes=60)

            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=min(entry, stop),
                y1=max(entry, stop),
                fillcolor="rgba(255,23,68,0.13)",
                line=dict(width=0),
                layer="below"
            )

            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=min(entry, target),
                y1=max(entry, target),
                fillcolor="rgba(0,200,83,0.10)",
                line=dict(width=0),
                layer="below"
            )

            fig.add_annotation(
                x=x0,
                y=entry,
                text="BUY ▶" if is_buy else "◀ SELL",
                showarrow=False,
                font=dict(
                    size=9,
                    color="#ffffff"
                ),
                bgcolor=(
                    "#087f3e"
                    if is_buy
                    else "#b71c1c"
                ),
                borderpad=2
            )

    if plan:

        for key, color in [
            ("Entry", "#2196f3"),
            ("SL", "#ff1744"),
            ("TP1", "#00c853"),
            ("TP2", "#00a152"),
            ("TP3", "#007a3d")
        ]:

            fig.add_hline(
                y=plan[key],
                line_dash="dash",
                line_color=color,
                annotation_text=key
            )

    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend_orientation="h",
        margin=dict(
            l=10,
            r=10,
            t=35,
            b=10
        )
    )

    return fig


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(token, chat_id, message):

    url = (
        f"https://api.telegram.org/bot{token}/sendMessage"
    )

    response = requests.post(
        url,
        data={
            "chat_id": chat_id,
            "text": message
        },
        timeout=15
    )

    response.raise_for_status()


# =========================================================
# BACKTEST
# =========================================================

def backtest_simple(
    df,
    risk_reward=2.0,
    fee_r=0.02
):

    d = df.copy().reset_index(drop=True)

    trades = []

    warmup = min(
        205,
        max(50, len(d) // 3)
    )

    for i in range(warmup, len(d) - 2):

        r = d.iloc[i]

        long_ok = (
            r.close > r.EMA50 > r.EMA200
            and r.RSI > 52
            and r.MACD > r.MACD_SIGNAL
            and r.ADX > 15
        )

        short_ok = (
            r.close < r.EMA50 < r.EMA200
            and r.RSI < 48
            and r.MACD < r.MACD_SIGNAL
            and r.ADX > 15
        )

        if not (long_ok or short_ok):
            continue

        side = "BUY" if long_ok else "SELL"

        entry = float(d.iloc[i + 1].open)

        atr = (
            float(r.ATR)
            if pd.notna(r.ATR) and r.ATR > 0
            else entry * 0.001
        )

        stop_dist = atr * 1.5

        sl = (
            entry - stop_dist
            if side == "BUY"
            else entry + stop_dist
        )

        tp = (
            entry + stop_dist * risk_reward
            if side == "BUY"
            else entry - stop_dist * risk_reward
        )

        outcome = "OPEN"
        r_multiple = 0.0
        exit_idx = None

        for j in range(i + 1, len(d)):

            bar = d.iloc[j]

            hit_sl = (
                bar.low <= sl
                if side == "BUY"
                else bar.high >= sl
            )

            hit_tp = (
                bar.high >= tp
                if side == "BUY"
                else bar.low <= tp
            )

            if hit_sl and hit_tp:

                outcome = "LOSS (same candle ambiguity)"
                r_multiple = -1 - fee_r
                exit_idx = j
                break

            if hit_sl:

                outcome = "LOSS"
                r_multiple = -1 - fee_r
                exit_idx = j
                break

            if hit_tp:

                outcome = "WIN"
                r_multiple = risk_reward - fee_r
                exit_idx = j
                break

        if exit_idx is None:
            continue

        trades.append({
            "Entry Time": d.iloc[i + 1].datetime,
            "Exit Time": d.iloc[exit_idx].datetime,
            "Side": side,
            "Entry": entry,
            "SL": sl,
            "TP": tp,
           
