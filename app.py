
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="AI Forex Swing Dashboard", page_icon="📈", layout="wide")

# ============================================================
# CONFIGURATION
# ============================================================
TIMEFRAMES = {
    "Daily": ("1day", pd.Timedelta(days=1)),
    "4H": ("4h", pd.Timedelta(hours=4)),
    "1H": ("1h", pd.Timedelta(hours=1)),
    "15M": ("15min", pd.Timedelta(minutes=15)),
    "5M": ("5min", pd.Timedelta(minutes=5)),
}

PAIRS = [
    "XAU/USD", "XAG/USD", "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "NZD/USD", "USD/CAD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
    "AUD/JPY", "EUR/AUD", "GBP/AUD"
]

DB_FILE = "forex_journal.db"


# ============================================================
# DATA ACCESS
# ============================================================
def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles(symbol, interval, outputsize, api_key):
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": int(outputsize),
        "apikey": api_key,
        "format": "JSON",
        "order": "ASC",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()
    payload = response.json()

    if "values" not in payload:
        raise RuntimeError(
            payload.get("message", str(payload))
        )

    df = pd.DataFrame(payload["values"])

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    for col in ["open", "high", "low", "close", "volume"]:

        if col in df:
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

    if len(df) < 210:
        raise RuntimeError(
            f"{interval}: API returned {len(df)} candles. "
            "At least 210 are needed for EMA200. "
            "Increase output size or check the API plan/symbol."
        )

    return df


# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def add_indicators(source):

    d = source.copy()

    c = d["close"]
    h = d["high"]
    l = d["low"]

    # EMA
    for period in (20, 50, 100, 200):
        d[f"EMA{period}"] = c.ewm(
            span=period,
            adjust=False
        ).mean()

    # RSI
    change = c.diff()

    avg_gain = change.clip(lower=0).ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = (-change.clip(upper=0)).ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    d["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    tr = pd.concat(
        [
            (h - l),
            (h - c.shift()).abs(),
            (l - c.shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    d["ATR"] = tr.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    # ADX
    up = h.diff()
    down = -l.diff()

    plus_dm = pd.Series(
        np.where(
            (up > down) & (up > 0),
            up,
            0.0
        ),
        index=d.index
    )

    minus_dm = pd.Series(
        np.where(
            (down > up) & (down > 0),
            down,
            0.0
        ),
        index=d.index
    )

    safe_atr = d["ATR"].replace(0, np.nan)

    plus_di = (
        100
        * plus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / safe_atr
    )

    minus_di = (
        100
        * minus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / safe_atr
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    d["ADX"] = dx.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    # MACD
    fast = c.ewm(span=12, adjust=False).mean()
    slow = c.ewm(span=26, adjust=False).mean()

    d["MACD"] = fast - slow

    d["MACD_SIGNAL"] = d["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    d["MACD_HIST"] = d["MACD"] - d["MACD_SIGNAL"]

    # Bollinger Bands
    d["BB_MID"] = c.rolling(20).mean()

    std = c.rolling(20).std()

    d["BB_UPPER"] = d["BB_MID"] + 2 * std
    d["BB_LOWER"] = d["BB_MID"] - 2 * std

    # VWAP: never fabricate volume for spot Forex.
    typical = (h + l + c) / 3
    vol = d["volume"].fillna(0)

    if vol.sum() > 0 and (vol > 0).sum() > 10:

        session = d["datetime"].dt.date

        cumulative_pv = (
            (typical * vol)
            .groupby(session)
            .cumsum()
        )

        cumulative_vol = (
            vol.groupby(session)
            .cumsum()
            .replace(0, np.nan)
        )

        d["VWAP"] = cumulative_pv / cumulative_vol

    else:
        d["VWAP"] = np.nan

    # Candle data
    d["BODY"] = (d["close"] - d["open"]).abs()

    d["RANGE"] = (
        d["high"] - d["low"]
    ).replace(0, np.nan)

    d["BODY_RATIO"] = (
        d["BODY"] / d["RANGE"]
    ).fillna(0)

    d["BULL_CANDLE"] = d["close"] > d["open"]
    d["BEAR_CANDLE"] = d["close"] < d["open"]

    # BOS and CHOCH
    d["PRIOR_HIGH_20"] = (
        d["high"].shift(1).rolling(20).max()
    )

    d["PRIOR_LOW_20"] = (
        d["low"].shift(1).rolling(20).min()
    )

    d["BOS_BULL"] = d["close"] > d["PRIOR_HIGH_20"]
    d["BOS_BEAR"] = d["close"] < d["PRIOR_LOW_20"]

    d["CHOCH_BULL"] = (
        d["BOS_BULL"]
        & (d["EMA20"].shift(1) < d["EMA50"].shift(1))
    )

    d["CHOCH_BEAR"] = (
        d["BOS_BEAR"]
        & (d["EMA20"].shift(1) > d["EMA50"].shift(1))
    )

    # Supply and demand approximations
    d["DEMAND_LOW"] = (
        d["low"].shift(1).rolling(10).min()
    )

    d["DEMAND_HIGH"] = (
        d["DEMAND_LOW"] + d["ATR"].shift(1) * 0.5
    )

    d["SUPPLY_HIGH"] = (
        d["high"].shift(1).rolling(10).max()
    )

    d["SUPPLY_LOW"] = (
        d["SUPPLY_HIGH"] - d["ATR"].shift(1) * 0.5
    )

    # Liquidity estimates
    d["LIQUIDITY_HIGH"] = (
        d["high"].shift(1).rolling(30).max()
    )

    d["LIQUIDITY_LOW"] = (
        d["low"].shift(1).rolling(30).min()
    )

    return d


def trend_series(df):

    score = pd.Series(
        0,
        index=df.index,
        dtype=float
    )

    score += np.where(
        df["close"] > df["EMA50"],
        1,
        -1
    )

    score += np.where(
        df["EMA50"] > df["EMA200"],
        1,
        -1
    )

    score += np.where(
        df["MACD"] > df["MACD_SIGNAL"],
        1,
        -1
    )

    score += np.where(
        df["RSI"] >= 50,
        1,
        -1
    )

    score += np.where(
        df["ADX"] >= 20,
        np.where(df["close"] > df["EMA50"], 1, -1),
        0
    )

    bias = pd.Series(
        np.where(
            score >= 2,
            1,
            np.where(score <= -2, -1, 0)
        ),
        index=df.index
    )

    return bias, score


def candle_pattern(row, previous=None):

    body = (
        float(row["BODY"])
        if pd.notna(row["BODY"])
        else 0.0
    )

    rng = (
        float(row["RANGE"])
        if pd.notna(row["RANGE"])
        else 0.0
    )

    if rng <= 0:
        return "None"

    upper = (
        row["high"]
        - max(row["open"], row["close"])
    )

    lower = (
        min(row["open"], row["close"])
        - row["low"]
    )

    if body / rng < 0.10:
        return "Doji"

    if lower > body * 2 and upper < body:
        return "Bullish Pin Bar"

    if upper > body * 2 and lower < body:
        return "Bearish Pin Bar"

    if previous is not None:

        bull = (
            row["close"] > row["open"]
            and previous["close"] < previous["open"]
            and row["close"] >= previous["open"]
            and row["open"] <= previous["close"]
        )

        bear = (
            row["close"] < row["open"]
            and previous["close"] > previous["open"]
            and row["open"] >= previous["close"]
            and row["close"] <= previous["open"]
        )

        if bull:
            return "Bullish Engulfing"

        if bear:
            return "Bearish Engulfing"

    return "None"


# ============================================================
# MULTI-TIMEFRAME ENGINE
# ============================================================
def build_mtf_frame(frames):

    base = (
        frames["5M"]
        .copy()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    base["available_at"] = (
        base["datetime"] + TIMEFRAMES["5M"][1]
    )

    merged = base

    for name in ("Daily", "4H", "1H", "15M"):

        tf = (
            frames[name]
            .copy()
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        bias, score = trend_series(tf)

        tf["available_at"] = (
            tf["datetime"] + TIMEFRAMES[name][1]
        )

        tf[f"{name}_BIAS"] = bias
        tf[f"{name}_SCORE"] = score

        right = tf[
            [
                "available_at",
                f"{name}_BIAS",
                f"{name}_SCORE"
            ]
        ].dropna().sort_values("available_at")

        merged = pd.merge_asof(
            merged.sort_values("available_at"),
            right,
            on="available_at",
            direction="backward"
        )

    b5, s5 = trend_series(base)

    merged["5M_BIAS"] = b5.values
    merged["5M_SCORE"] = s5.values

    return (
        merged
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def signal_row(row):

    htf = [
        row.get("Daily_BIAS", np.nan),
        row.get("4H_BIAS", np.nan),
        row.get("1H_BIAS", np.nan)
    ]

    bullish = sum(
        pd.notna(x) and int(x) == 1
        for x in htf
    )

    bearish = sum(
        pd.notna(x) and int(x) == -1
        for x in htf
    )

    m15 = row.get("15M_BIAS", 0)

    buy = (
        bullish >= 2
        and m15 == 1
        and row["close"] > row["EMA20"]
        and row["RSI"] > 50
        and row["RSI"] < 75
        and row["MACD"] > row["MACD_SIGNAL"]
        and row["ADX"] >= 15
        and row["BODY_RATIO"] >= 0.25
        and bool(row["BULL_CANDLE"])
    )

    sell = (
        bearish >= 2
        and m15 == -1
        and row["close"] < row["EMA20"]
        and row["RSI"] < 50
        and row["RSI"] > 25
        and row["MACD"] < row["MACD_SIGNAL"]
        and row["ADX"] >= 15
        and row["BODY_RATIO"] >= 0.25
        and bool(row["BEAR_CANDLE"])
    )

    raw_scores = [
        row.get(f"{x}_SCORE", 0)
        for x in ("Daily", "4H", "1H", "15M", "5M")
    ]

    if buy:
        alignment = sum(
            1 for x in raw_scores
            if pd.notna(x) and x > 0
        )
    else:
        alignment = sum(
            1 for x in raw_scores
            if pd.notna(x) and x < 0
        )

    adx_part = min(
        float(row["ADX"])
        if pd.notna(row["ADX"])
        else 0,
        35
    )

    quality = int(
        np.clip(
            30
            + alignment * 8
            + adx_part * 0.6
            + float(row["BODY_RATIO"]) * 10,
            0,
            100
        )
    )

    if buy:

        direction = "BUY"

        label = (
            "STRONG BUY"
            if bullish == 3 and quality >= 75
            else "BUY"
        )

    elif sell:

        direction = "SELL"

        label = (
            "STRONG SELL"
            if bearish == 3 and quality >= 75
            else "SELL"
        )

    else:

        direction = "WAIT"
        label = "WAIT"
        quality = min(quality, 69)

    return direction, label, quality


def current_analysis(frames):

    latest = {}
    biases = {}
    scores = {}

    for name, df in frames.items():

        # Avoid using a candle that may still be forming.
        usable = (
            df.iloc[:-1]
            if len(df) > 2
            else df
        )

        b, s = trend_series(usable)

        biases[name] = int(b.iloc[-1])
        scores[name] = int(s.iloc[-1])

        latest[name] = usable

    labels = {
        1: "BULLISH",
        -1: "BEARISH",
        0: "NEUTRAL"
    }

    aligned = {
        key: labels[value]
        for key, value in biases.items()
    }

    base = latest["5M"]

    row = base.iloc[-1]

    bull_count = sum(
        biases[x] == 1
        for x in ("Daily", "4H", "1H")
    )

    bear_count = sum(
        biases[x] == -1
        for x in ("Daily", "4H", "1H")
    )

    pattern = candle_pattern(
        row,
        base.iloc[-2]
    )

    buy = (
        bull_count >= 2
        and biases["15M"] == 1
        and row["close"] > row["EMA20"]
        and row["RSI"] > 50
        and row["RSI"] < 75
        and row["MACD"] > row["MACD_SIGNAL"]
        and row["ADX"] >= 15
        and row["BODY_RATIO"] >= 0.25
        and bool(row["BULL_CANDLE"])
    )

    sell = (
        bear_count >= 2
        and biases["15M"] == -1
        and row["close"] < row["EMA20"]
        and row["RSI"] < 50
        and row["RSI"] > 25
        and row["MACD"] < row["MACD_SIGNAL"]
        and row["ADX"] >= 15
        and row["BODY_RATIO"] >= 0.25
        and bool(row["BEAR_CANDLE"])
    )

    if buy:
        direction = "BUY"

    elif sell:
        direction = "SELL"

    else:
        direction = "WAIT"

    relevant_scores = [
        scores[x]
        for x in TIMEFRAMES
    ]

    if direction == "BUY":
        agree = sum(x > 0 for x in relevant_scores)
    elif direction == "SELL":
        agree = sum(x < 0 for x in relevant_scores)
    else:
        agree = 0

    quality = int(
        np.clip(
            30
            + agree * 8
            + min(float(row["ADX"]), 35) * 0.6
            + float(row["BODY_RATIO"]) * 10,
            0,
            100
        )
    )

    if direction == "WAIT":

        label = "WAIT"
        quality = min(quality, 69)

    elif direction == "BUY":

        label = (
            "STRONG BUY"
            if bull_count == 3 and quality >= 75
            else "BUY"
        )

    else:

        label = (
            "STRONG SELL"
            if bear_count == 3 and quality >= 75
            else "SELL"
        )

    return {
        "direction": direction,
        "signal": label,
        "quality": quality,
        "biases": aligned,
        "scores": scores,
        "pattern": pattern,
        "latest_frames": latest,
    }


# ============================================================
# TRADE LEVELS AND MTF BACKTEST
# ============================================================
def trade_levels(df, index, side, rr=2.0):

    row = df.iloc[index]

    entry = float(row["open"])

    atr = (
        float(row["ATR"])
        if pd.notna(row["ATR"]) and row["ATR"] > 0
        else entry * 0.001
    )

    lookback = df.iloc[
        max(0, index - 20):index
    ]

    if side == "BUY":

        swing = (
            float(lookback["low"].min())
            if not lookback.empty
            else entry - atr * 1.5
        )

        sl = min(
            entry - atr * 1.5,
            swing - atr * 0.1
        )

        risk = max(entry - sl, atr)

        return (
            entry,
            sl,
            entry + risk,
            entry + risk * 2,
            entry + risk * 3
        )

    swing = (
        float(lookback["high"].max())
        if not lookback.empty
        else entry + atr * 1.5
    )

    sl = max(
        entry + atr * 1.5,
        swing + atr * 0.1
    )

    risk = max(sl - entry, atr)

    return (
        entry,
        sl,
        entry - risk,
        entry - risk * 2,
        entry - risk * 3
    )


def run_mtf_backtest(
    frames,
    rr=2.0,
    holding_days=2,
    fee_r=0.03
):

    aligned = build_mtf_frame(frames)

    aligned = aligned.dropna(
        subset=[
            "Daily_BIAS",
            "4H_BIAS",
            "1H_BIAS",
            "15M_BIAS"
        ]
    ).reset_index(drop=True)

    trades = []

    i = 1

    max_bars = int(
        holding_days * 24 * 12
    )

    while i < len(aligned) - 2:

        row = aligned.iloc[i - 1]

        direction, label, quality = signal_row(row)

        if direction == "WAIT":
            i += 1
            continue

        entry_i = i

        entry, sl, tp1, tp2, tp3 = trade_levels(
            aligned,
            entry_i,
            direction,
            rr
        )

        risk = abs(entry - sl)

        backtest_target = (
            entry + risk * rr
            if direction == "BUY"
            else entry - risk * rr
        )

        if risk <= 0 or not np.isfinite(risk):
            i += 1
            continue

        end_i = min(
            len(aligned) - 1,
            entry_i + max_bars
        )

        outcome = "TIME EXIT"

        exit_price = float(
            aligned.iloc[end_i]["close"]
        )

        exit_i = end_i
        r_mult = 0.0

        for j in range(entry_i, end_i + 1):

            bar = aligned.iloc[j]

            hit_sl = (
                bar["low"] <= sl
                if direction == "BUY"
                else bar["high"] >= sl
            )

            hit_tp = (
                bar["high"] >= backtest_target
                if direction == "BUY"
                else bar["low"] <= backtest_target
            )

            if hit_sl and hit_tp:

                outcome = "LOSS (SL/TP same candle)"
                exit_price = sl
                exit_i = j
                r_mult = -1 - fee_r
                break

            if hit_sl:

                outcome = "LOSS"
                exit_price = sl
                exit_i = j
                r_mult = -1 - fee_r
                break

            if hit_tp:

                outcome = "WIN"
                exit_price = backtest_target
                exit_i = j
                r_mult = rr - fee_r
                break

        if outcome == "TIME EXIT":

            signed = (
                exit_price - entry
                if direction == "BUY"
                else entry - exit_price
            )

            r_mult = signed / risk - fee_r

            if r_mult > 0:
                outcome = "TIME EXIT WIN"
            elif r_mult < 0:
                outcome = "TIME EXIT LOSS"
            else:
                outcome = "BREAKEVEN"

        trades.append({
            "Entry Time": aligned.iloc[entry_i]["datetime"],
            "Exit Time": aligned.iloc[exit_i]["datetime"],
            "Side": direction,
            "Signal": label,
            "Quality": quality,
            "Entry": entry,
            "SL": sl,
            "TP1": tp1,
            "TP2": tp2,
            "TP3": tp3,
            "Backtest Target": backtest_target,
            "Exit Price": exit_price,
            "Result": outcome,
            "R Multiple": r_mult,
            "Bars Held": exit_i - entry_i,
        })

        # One position at a time.
        i = exit_i + 1

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:

        return aligned, trades_df, {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "net_r": 0.0,
            "avg_r": 0.0
        }

    pnl = trades_df["R Multiple"].astype(float)

    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())

    equity = pnl.cumsum()
    peak = equity.cummax().clip(lower=0)
    drawdown = equity - peak

    stats = {
        "trades": int(l
