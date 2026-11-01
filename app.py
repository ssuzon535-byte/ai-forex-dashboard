import sqlite3
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="AI Forex Swing Dashboard", page_icon="📈", layout="wide")

# CONFIGURATION
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


# DATA ACCESS
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
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if "values" not in payload:
        raise RuntimeError(payload.get("message", str(payload)))

    df = pd.DataFrame(payload["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col == "volume":
            df[col] = np.nan
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    if len(df) < 210:
        raise RuntimeError(
            f"{interval}: API returned {len(df)} candles. At least 210 are needed for EMA200. "
            "Increase output size or check the API plan/symbol."
        )
    return df


# INDICATORS
def add_indicators(source):
    d = source.copy()
    c, h, l = d["close"], d["high"], d["low"]

    for period in (20, 50, 100, 200):
        d[f"EMA{period}"] = c.ewm(span=period, adjust=False).mean()

    change = c.diff()
    avg_gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    up, down = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=d.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=d.index)
    safe_atr = d["ATR"].replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / safe_atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / safe_atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    fast = c.ewm(span=12, adjust=False).mean()
    slow = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = fast - slow
    d["MACD_SIGNAL"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d["MACD"] - d["MACD_SIGNAL"]

    d["BB_MID"] = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["BB_UPPER"] = d["BB_MID"] + 2 * std
    d["BB_LOWER"] = d["BB_MID"] - 2 * std

    # Never fabricate volume for spot FX.
    typical = (h + l + c) / 3
    vol = d["volume"].fillna(0)
    if vol.sum() > 0 and (vol > 0).sum() > 10:
        session = d["datetime"].dt.date
        cumulative_pv = (typical * vol).groupby(session).cumsum()
        cumulative_vol = vol.groupby(session).cumsum().replace(0, np.nan)
        d["VWAP"] = cumulative_pv / cumulative_vol
    else:
        d["VWAP"] = np.nan

    d["BODY"] = (d["close"] - d["open"]).abs()
    d["RANGE"] = (d["high"] - d["low"]).replace(0, np.nan)
    d["BODY_RATIO"] = (d["BODY"] / d["RANGE"]).fillna(0)
    d["BULL_CANDLE"] = d["close"] > d["open"]
    d["BEAR_CANDLE"] = d["close"] < d["open"]

    d["PRIOR_HIGH_20"] = d["high"].shift(1).rolling(20).max()
    d["PRIOR_LOW_20"] = d["low"].shift(1).rolling(20).min()
    d["BOS_BULL"] = d["close"] > d["PRIOR_HIGH_20"]
    d["BOS_BEAR"] = d["close"] < d["PRIOR_LOW_20"]
    d["CHOCH_BULL"] = d["BOS_BULL"] & (d["EMA20"].shift(1) < d["EMA50"].shift(1))
    d["CHOCH_BEAR"] = d["BOS_BEAR"] & (d["EMA20"].shift(1) > d["EMA50"].shift(1))

    d["DEMAND_LOW"] = d["low"].shift(1).rolling(10).min()
    d["DEMAND_HIGH"] = d["DEMAND_LOW"] + d["ATR"].shift(1) * 0.5
    d["SUPPLY_HIGH"] = d["high"].shift(1).rolling(10).max()
    d["SUPPLY_LOW"] = d["SUPPLY_HIGH"] - d["ATR"].shift(1) * 0.5

    d["LIQUIDITY_HIGH"] = d["high"].shift(1).rolling(30).max()
    d["LIQUIDITY_LOW"] = d["low"].shift(1).rolling(30).min()
    return d


def trend_series(df):
    score = pd.Series(0, index=df.index, dtype=float)
    score += np.where(df["close"] > df["EMA50"], 1, -1)
    score += np.where(df["EMA50"] > df["EMA200"], 1, -1)
    score += np.where(df["MACD"] > df["MACD_SIGNAL"], 1, -1)
    score += np.where(df["RSI"] >= 50, 1, -1)
    score += np.where(
        df["ADX"] >= 20,
        np.where(df["close"] > df["EMA50"], 1, -1),
        0
    )
    bias = pd.Series(
        np.where(score >= 2, 1, np.where(score <= -2, -1, 0)),
        index=df.index
    )
    return bias, score


def candle_pattern(row, previous=None):
    body = float(row["BODY"]) if pd.notna(row["BODY"]) else 0.0
    rng = float(row["RANGE"]) if pd.notna(row["RANGE"]) else 0.0
    if rng <= 0:
        return "None"
    upper = row["high"] - max(row["open"], row["close"])
    lower = min(row["open"], row["close"]) - row["low"]
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


# MULTI-TIMEFRAME ENGINE
def build_mtf_frame(frames):
    base = frames["5M"].copy().sort_values("datetime").reset_index(drop=True)
    base["available_at"] = base["datetime"] + TIMEFRAMES["5M"][1]
    merged = base

    for name in ("Daily", "4H", "1H", "15M"):
        tf = frames[name].copy().sort_values("datetime").reset_index(drop=True)
        bias, score = trend_series(tf)
        tf["available_at"] = tf["datetime"] + TIMEFRAMES[name][1]
        tf[f"{name}_BIAS"] = bias
        tf[f"{name}_SCORE"] = score
        right = (
            tf[["available_at", f"{name}_BIAS", f"{name}_SCORE"]]
            .dropna()
            .sort_values("available_at")
        )
        merged = pd.merge_asof(
            merged.sort_values("available_at"),
            right,
            on="available_at",
            direction="backward"
        )

    b5, s5 = trend_series(base)
    merged["5M_BIAS"] = b5.values
    merged["5M_SCORE"] = s5.values
    return merged.sort_values("datetime").reset_index(drop=True)


def signal_row(row):
    htf = [
        row.get("Daily_BIAS", np.nan),
        row.get("4H_BIAS", np.nan),
        row.get("1H_BIAS", np.nan)
    ]
    bullish = sum(pd.notna(x) and int(x) == 1 for x in htf)
    bearish = sum(pd.notna(x) and int(x) == -1 for x in htf)
    m15 = row.get("15M_BIAS", 0)

    buy = (
        bullish >= 2 and m15 == 1
        and row["close"] > row["EMA20"]
        and row["RSI"] > 50 and row["RSI"] < 75
        and row["MACD"] > row["MACD_SIGNAL"]
        and row["ADX"] >= 15 and row["BODY_RATIO"] >= 0.25
        and bool(row["BULL_CANDLE"])
    )
    sell = (
        bearish >= 2 and m15 == -1
        and row["close"] < row["EMA20"]
        and row["RSI"] < 50 and row["RSI"] > 25
        and row["MACD"] < row["MACD_SIGNAL"]
        and row["ADX"] >= 15 and row["BODY_RATIO"] >= 0.25
        and bool(row["BEAR_CANDLE"])
    )

    raw_scores = [
        row.get(f"{x}_SCORE", 0)
        for x in ("Daily", "4H", "1H", "15M", "5M")
    ]
    if buy:
        alignment = sum(
            1 for x in raw_scores if pd.notna(x) and x > 0
        )
    else:
        alignment = sum(
            1 for x in raw_scores if pd.notna(x) and x < 0
        )

    adx_part = min(float(row["ADX"]) if pd.notna(row["ADX"]) else 0, 35)
    quality = int(np.clip(
        30 + alignment * 8 + adx_part * 0.6
        + float(row["BODY_RATIO"]) * 10,
        0, 100
    ))

    if buy:
        direction = "BUY"
        label = "STRONG BUY" if bullish == 3 and quality >= 75 else "BUY"
    elif sell:
        direction = "SELL"
        label = "STRONG SELL" if bearish == 3 and quality >= 75 else "SELL"
    else:
        direction, label = "WAIT", "WAIT"
        quality = min(quality, 69)

    return direction, label, quality
    ```python
def current_analysis(frames):
    latest = {}

    for tf, df in frames.items():
        if df is not None and not df.empty:
            latest[tf] = df.iloc[-1]

    return latest
```

    for name, df in frames.items():
        usable = df.iloc[:-1] if len(df) > 2 else df
        b, s = trend_series(usable)
        biases[name] = int(b.iloc[-1])
        scores[name] = int(s.iloc[-1])
        latest[name] = usable

    labels = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}
    aligned = {k: labels[v] for k, v in biases.items()}
    base = latest["5M"]
    row = base.iloc[-1]
    bull_count = sum(biases[x] == 1 for x in ("Daily", "4H", "1H"))
    bear_count = sum(biases[x] == -1 for x in ("Daily", "4H", "1H"))
    pattern = candle_pattern(row, base.iloc[-2])

    buy = (
        bull_count >= 2 and biases["15M"] == 1
        and row["close"] > row["EMA20"]
        and row["RSI"] > 50 and row["RSI"] < 75
        and row["MACD"] > row["MACD_SIGNAL"]
        and row["ADX"] >= 15
        and row["BODY_RATIO"] >= 0.25
        and bool(row["BULL_CANDLE"])
    )

    sell = (
        bear_count >= 2 and biases["15M"] == -1
        and row["close"] < row["EMA20"]
        and row["RSI"] < 50 and row["RSI"] > 25
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

    relevant_scores = [scores[x] for x in TIMEFRAMES]

    if direction == "BUY":
        agree = sum(1 for x in relevant_scores if x > 0)
    elif direction == "SELL":
        agree = sum(1 for x in relevant_scores if x < 0)
    else:
        agree = 0

    quality = int(np.clip(
        30 + agree * 8 + min(float(row["ADX"]), 35) * 0.6
        + float(row["BODY_RATIO"]) * 10,
        0, 100
    ))

    if direction == "WAIT":
        label, quality = "WAIT", min(quality, 69)
    elif direction == "BUY":
        label = "STRONG BUY" if bull_count == 3 and quality >= 75 else "BUY"
    else:
        label = "STRONG SELL" if bear_count == 3 and quality >= 75 else "SELL"

    return {
        "direction": direction,
        "signal": label,
        "quality": quality,
        "biases": aligned,
        "scores": scores,
        "pattern": pattern,
        "latest_frames": latest,
    }


# TRADE LEVELS AND MTF BACKTEST
def trade_levels(df, index, side, rr=2.0):
    row = df.iloc[index]
    entry = float(row["open"])
    atr = (
        float(row["ATR"])
        if pd.notna(row["ATR"]) and row["ATR"] > 0
        else entry * 0.001
    )
    lookback = df.iloc[max(0, index - 20):index]

    if side == "BUY":
        swing = (
            float(lookback["low"].min())
            if not lookback.empty else entry - atr * 1.5
        )
        sl = min(entry - atr * 1.5, swing - atr * 0.1)
        risk = max(entry - sl, atr)
        return entry, sl, entry + risk, entry + risk * 2, entry + risk * 3

    swing = (
        float(lookback["high"].max())
        if not lookback.empty else entry + atr * 1.5
    )
    sl = max(entry + atr * 1.5, swing + atr * 0.1)
    risk = max(sl - entry, atr)
    return entry, sl, entry - risk, entry - risk * 2, entry - risk * 3


def run_mtf_backtest(frames, rr=2.0, holding_days=2, fee_r=0.03):
    aligned = build_mtf_frame(frames)
    aligned = aligned.dropna(
        subset=["Daily_BIAS", "4H_BIAS", "1H_BIAS", "15M_BIAS"]
    ).reset_index(drop=True)

    trades = []
    i = 1
    max_bars = int(holding_days * 24 * 12)

    while i < len(aligned) - 2:
        row = aligned.iloc[i - 1]
        direction, label, quality = signal_row(row)

        if direction == "WAIT":
            i += 1
            continue

        entry_i = i
        entry, sl, tp1, tp2, tp3 = trade_levels(
            aligned, entry_i, direction, rr
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

        end_i = min(len(aligned) - 1, entry_i + max_bars)
        outcome = "TIME EXIT"
        exit_price = float(aligned.iloc[end_i]["close"])
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
                exit_price, exit_i, r_mult = sl, j, -1 - fee_r
                break
            if hit_sl:
                outcome = "LOSS"
                exit_price, exit_i, r_mult = sl, j, -1 - fee_r
                break
            if hit_tp:
                outcome = "WIN"
                exit_price, exit_i, r_mult = backtest_target, j, rr - fee_r
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
        i = exit_i + 1

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return aligned, trades_df, {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "net_r": 0.0,
            "avg_r": 0.0,
        }

    pnl = trades_df["R Multiple"].astype(float)
    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    equity = pnl.cumsum()
    peak = equity.cummax().clip(lower=0)
    drawdown = equity - peak

    stats = {
        "trades": int(len(trades_df)),
        "win_rate": float((pnl > 0).mean() * 100),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "max_drawdown_r": float(abs(drawdown.min())),
        "net_r": float(pnl.sum()),
        "avg_r": float(pnl.mean()),
    }
    trades_df["Equity R"] = equity
    trades_df["Drawdown R"] = drawdown
    return aligned, trades_df, stats


# CHART
def make_chart(df, trades=None, plan=None):
    d = df.tail(300).copy()
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=d["datetime"],
        open=d["open"],
        high=d["high"],
        low=d["low"],
        close=d["close"],
        name="Price"
    ))

    for col in ("EMA20", "EMA50", "EMA200"):
        fig.add_trace(go.Scatter(
            x=d["datetime"], y=d[col], mode="lines", name=col
        ))

    fig.add_trace(go.Scatter(
        x=d["datetime"], y=d["BB_UPPER"],
        mode="lines", name="BB Upper",
        line=dict(dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=d["datetime"], y=d["BB_LOWER"],
        mode="lines", name="BB Lower",
        line=dict(dash="dot")
    ))

    if d["VWAP"].notna().any():
        fig.add_trace(go.Scatter(
            x=d["datetime"], y=d["VWAP"],
            mode="lines", name="VWAP"
        ))

    if trades is not None and not trades.empty:
        visible_start = d["datetime"].min()
        visible_end = d["datetime"].max()

        for _, t in trades.iterrows():
            if t["Entry Time"] < visible_start or t["Entry Time"] > visible_end:
                continue

            is_buy = t["Side"] == "BUY"
            marker_y = t["Entry"]

            fig.add_trace(go.Scatter(
                x=[t["Entry Time"]],
                y=[marker_y],
                mode="markers+text",
                text=["BUY" if is_buy else "SELL"],
                textposition="top center",
                name="Historical BUY" if is_buy else "Historical SELL",
                marker=dict(
                    symbol="triangle-up" if is_buy else "triangle-down",
                    size=12
                )
            ))

            x1 = t["Exit Time"]
            fig.add_shape(
                type="rect",
                x0=t["Entry Time"],
                x1=x1,
                y0=min(t["Entry"], t["SL"]),
                y1=max(t["Entry"], t["SL"]),
                fillcolor="rgba(255,40,60,0.12)",
                line=dict(width=1, color="rgba(255,40,60,0.5)"),
                layer="below"
            )

            target = t.get("Backtest Target", t["TP2"])
            fig.add_shape(
                type="rect",
                x0=t["Entry Time"],
                x1=x1,
                y0=min(t["Entry"], target),
                y1=max(t["Entry"], target),
                fillcolor="rgba(0,200,100,0.10)",
                line=dict(width=1, color="rgba(0,200,100,0.5)"),
                layer="below"
            )

            fig.add_annotation(
                x=t["Entry Time"],
                y=t["Entry"],
                text="▲ BUY" if is_buy else "▼ SELL",
                showarrow=False,
                bgcolor="#087f3e" if is_buy else "#b71c1c",
                font=dict(color="white", size=9)
            )

    if plan:
        for name, color in (
            ("Entry", "#2196f3"),
            ("SL", "#ff1744"),
            ("TP1", "#00c853"),
            ("TP2", "#00a152"),
            ("TP3", "#007a3d")
        ):
            fig.add_hline(
                y=plan[name],
                line_dash="dash",
                line_color=color,
                annotation_text=name
            )

    fig.update_layout(
        template="plotly_dark",
        height=680,
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        margin=dict(l=10, r=10, t=35, b=10)
    )
    return fig


# PERSISTENT JOURNAL
def init_db():
    with sqlite3.connect(DB_FILE) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry REAL NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL,
            result TEXT NOT NULL,
            notes TEXT
        )""")


def save_journal(symbol, side, entry, sl, tp, result, notes):
    with sqlite3.connect(DB_FILE) as con:
        con.execute(
            "INSERT INTO journal(created_at,symbol,side,entry,stop_loss,take_profit,result,notes) VALUES(?,?,?,?,?,?,?,?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                symbol, side, entry, sl, tp, result, notes
            )
        )


def read_journal():
    with sqlite3.connect(DB_FILE) as con:
        return pd.read_sql_query(
            "SELECT * FROM journal ORDER BY id DESC", con
        )


def delete_journal_row(row_id):
    with sqlite3.connect(DB_FILE) as con:
        con.execute(
            "DELETE FROM journal WHERE id=?",
            (int(row_id),)
        )


# TELEGRAM
def send_telegram(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(
        url,
        data={"chat_id": chat_id, "text": message},
        timeout=20
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("ok"):
        raise RuntimeError(result.get("description", "Telegram API error"))
        # USER INTERFACE
init_db()

st.title("📈 AI Forex Swing Trading Dashboard")
st.caption(
    "Strategy: 1–2 day swing | Daily + 4H + 1H context "
    "| 15M setup | 5M entry confirmation"
)

with st.sidebar:
    st.header("Market & Data")
    symbol = st.selectbox("Market", PAIRS, index=0)
    candle_count = st.selectbox(
        "History candles per timeframe",
        [300, 500, 800, 1000],
        index=1
    )
    api_key = st.text_input(
        "Twelve Data API Key",
        type="password",
        value=get_secret("TWELVE_DATA_API_KEY", "")
    )
    holding_days = st.selectbox(
        "Maximum holding duration",
        [1, 2],
        index=1,
        format_func=lambda x: f"{x} day(s)"
    )

    st.divider()
    st.header("Risk Settings")
    balance = st.number_input(
        "Account balance",
        min_value=1.0,
        value=1000.0,
        step=100.0
    )
    risk_pct = st.number_input(
        "Risk per trade (%)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1
    )
    rr = st.selectbox(
        "Backtest TP2 risk:reward",
        [1.0, 1.5, 2.0, 2.5, 3.0],
        index=2
    )
    st.caption(
        "Risk-to-lot conversion depends on your broker's contract size "
        "and account currency. Enter broker-specific values in the Risk Calculator."
    )


if not api_key:
    st.info("Sidebar-এ Twelve Data API Key দিন।")
    st.stop()


try:
    with st.spinner("Fetching and calculating all five timeframes..."):
        frames = {}
        for name, (interval, _) in TIMEFRAMES.items():
            frames[name] = add_indicators(
                fetch_candles(symbol, interval, candle_count, api_key)
            )
except Exception as exc:
    st.error(f"Market data error: {exc}")
    st.stop()


analysis = current_analysis(frames)
base = analysis["latest_frames"]["5M"]
last = base.iloc[-1]


# CURRENT TRADE PLAN
def make_current_plan(df, direction):
    row = df.iloc[-1]
    entry = float(row["close"])
    atr = (
        float(row["ATR"])
        if pd.notna(row["ATR"]) and row["ATR"] > 0
        else entry * 0.001
    )
    look = df.tail(20)

    if direction == "BUY":
        sl = min(
            entry - 1.5 * atr,
            float(look["low"].min()) - 0.1 * atr
        )
        risk = max(entry - sl, atr)
        return {
            "Entry": entry,
            "SL": sl,
            "TP1": entry + risk,
            "TP2": entry + 2 * risk,
            "TP3": entry + 3 * risk,
            "Risk": risk
        }

    if direction == "SELL":
        sl = max(
            entry + 1.5 * atr,
            float(look["high"].max()) + 0.1 * atr
        )
        risk = max(sl - entry, atr)
        return {
            "Entry": entry,
            "SL": sl,
            "TP1": entry - risk,
            "TP2": entry - 2 * risk,
            "TP3": entry - 3 * risk,
            "Risk": risk
        }

    return None


plan = make_current_plan(base, analysis["direction"])


# HISTORICAL MTF BACKTEST
with st.spinner("Running historical multi-timeframe backtest..."):
    aligned, bt_trades, stats = run_mtf_backtest(
        frames,
        rr=float(rr),
        holding_days=int(holding_days)
    )


# DASHBOARD METRICS
m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Current Price", f"{last['close']:.5f}")
m2.metric("Signal", analysis["signal"])
m3.metric("Signal Quality", f"{analysis['quality']}/100")
m4.metric("5M RSI", f"{last['RSI']:.1f}")
m5.metric("5M ADX", f"{last['ADX']:.1f}")


st.subheader("Multi-Timeframe Market Direction")

cols = st.columns(5)
for col, name in zip(cols, TIMEFRAMES.keys()):
    col.metric(
        name,
        analysis["biases"][name],
        f"Score {analysis['scores'][name]}"
    )


if analysis["direction"] == "WAIT":
    st.warning(
        "WAIT: Higher-timeframe alignment and 5M confirmation "
        "do not currently agree. No trade setup is issued."
    )
else:
    st.success(
        f"{analysis['signal']} | Quality {analysis['quality']}/100. "
        "Quality is a rule-confluence score, not a win probability."
    )


st.subheader("Current Trade Levels")

if plan:
    level_cols = st.columns(5)
    for col, key in zip(
        level_cols,
        ("Entry", "SL", "TP1", "TP2", "TP3")
    ):
        col.metric(key, f"{plan[key]:.5f}")
else:
    st.info(
        "বর্তমানে WAIT signal. তাই নতুন Entry, SL বা TP দেওয়া হচ্ছে না।"
    )


# TABS
chart_tab, structure_tab, risk_tab, backtest_tab, journal_tab, alert_tab = st.tabs([
    "📊 Chart & Indicators",
    "🧭 Price Action",
    "🧮 Risk Calculator",
    "🧪 MTF Backtest",
    "📒 Trade Journal",
    "🔔 Telegram Alerts"
])


# CHART TAB
with chart_tab:
    st.plotly_chart(
        make_chart(base, bt_trades, plan),
        use_container_width=True
    )

    st.caption(
        "Historical arrows and boxes come from the MTF backtest trades. "
        "They are simulated historical setups, not executed broker trades."
    )

    indicator_cols = [
        "datetime", "open", "high", "low", "close",
        "EMA20", "EMA50", "EMA200", "RSI", "MACD",
        "MACD_SIGNAL", "ADX", "ATR",
        "BB_UPPER", "BB_LOWER", "VWAP"
    ]

    st.dataframe(
        base.tail(30)[indicator_cols].sort_values(
            "datetime",
            ascending=False
        ),
        use_container_width=True
    )


# PRICE ACTION TAB
with structure_tab:
    row = last

    a, b, c, d = st.columns(4)

    a.metric(
        "BOS",
        "Bullish" if bool(row["BOS_BULL"])
        else "Bearish" if bool(row["BOS_BEAR"])
        else "None"
    )

    b.metric(
        "CHOCH",
        "Bullish" if bool(row["CHOCH_BULL"])
        else "Bearish" if bool(row["CHOCH_BEAR"])
        else "None"
    )

    c.metric("Candle Pattern", analysis["pattern"])

    d.metric(
        "VWAP",
        f"{row['VWAP']:.5f}"
        if pd.notna(row["VWAP"])
        else "Not supplied by feed"
    )

    st.caption(
        "BOS/CHOCH use a trailing 20-candle close-break rule. "
        "Supply/demand and liquidity are transparent price-action "
        "approximations, not institutional order-book data."
    )

    st.subheader("Support & Resistance")

    recent = base.tail(100)
    sr1, sr2 = st.columns(2)

    sr1.metric(
        "Resistance (100-candle high)",
        f"{recent['high'].max():.5f}"
    )
    sr2.metric(
        "Support (100-candle low)",
        f"{recent['low'].min():.5f}"
    )

    st.subheader("Supply & Demand")

    zone_df = base.tail(100).dropna(
        subset=[
            "DEMAND_LOW", "DEMAND_HIGH",
            "SUPPLY_LOW", "SUPPLY_HIGH"
        ],
        how="all"
    )

    if zone_df.empty:
        st.info("No zone values available in the selected history.")
    else:
        st.dataframe(
            zone_df[
                [
                    "datetime",
                    "DEMAND_LOW",
                    "DEMAND_HIGH",
                    "SUPPLY_LOW",
                    "SUPPLY_HIGH"
                ]
            ].tail(30),
            use_container_width=True
        )

    st.subheader("Liquidity Levels")

    liq1, liq2 = st.columns(2)

    liq1.metric(
        "Buy-side liquidity estimate",
        f"{row['LIQUIDITY_HIGH']:.5f}"
        if pd.notna(row["LIQUIDITY_HIGH"])
        else "N/A"
    )

    liq2.metric(
        "Sell-side liquidity estimate",
        f"{row['LIQUIDITY_LOW']:.5f}"
        if pd.notna(row["LIQUIDITY_LOW"])
        else "N/A"
    )


# RISK CALCULATOR TAB
with risk_tab:
    st.subheader("Risk and Position Size Calculator")

    default_entry = (
        float(plan["Entry"]) if plan
        else float(last["close"])
    )

    default_sl = (
        float(plan["SL"]) if plan
        else float(last["close"] - 1.5 * last["ATR"])
    )

    r1, r2, r3 = st.columns(3)

    entry_input = r1.number_input(
        "Entry price",
        value=default_entry,
        format="%.5f"
    )

    sl_input = r2.number_input(
        "Stop-loss price",
        value=default_sl,
        format="%.5f"
    )

    risk_balance = r3.number_input(
        "Account balance",
        min_value=1.0,
        value=float(balance),
        key="risk_balance"
    )

    r4, r5, r6 = st.columns(3)

    risk_input = r4.number_input(
        "Risk percent",
        min_value=0.1,
        max_value=10.0,
        value=float(risk_pct),
        step=0.1
    )

    contract_size = r5.number_input(
        "Contract size per 1 lot (broker setting)",
        min_value=0.000001,
        value=100.0,
        format="%.6f"
    )

    quote_conversion = r6.number_input(
        "Quote-currency to account-currency conversion",
        min_value=0.000001,
        value=1.0,
        format="%.6f"
    )

    cash_risk = risk_balance * risk_input / 100
    price_distance = abs(entry_input - sl_input)

    estimated_lots = (
        cash_risk /
        (price_distance * contract_size * quote_conversion)
        if price_distance > 0 else 0.0
    )

    q1, q2, q3 = st.columns(3)

    q1.metric("Cash risk", f"{cash_risk:.2f}")
    q2.metric("Price distance", f"{price_distance:.5f}")
    q3.metric("Estimated lots", f"{estimated_lots:.4f}")

    st.warning(
        "Set contract size and currency conversion from your broker's "
        "XAU/USD or FX contract specification. The default contract size "
        "is only an example and may be wrong for your broker. Confirm before trading."
    )


# BACKTEST TAB
with backtest_tab:
    st.subheader("Multi-Timeframe Historical Backtest")

    b1, b2, b3, b4, b5 = st.columns(5)

    b1.metric("Completed trades", stats["trades"])
    b2.metric("Win rate", f"{stats['win_rate']:.1f}%")

    b3.metric(
        "Profit factor",
        "∞" if np.isinf(stats["profit_factor"])
        else f"{stats['profit_factor']:.2f}"
    )

    b4.metric(
        "Max drawdown",
        f"{stats['max_drawdown_r']:.2f}R"
    )

    b5.metric("Net result", f"{stats['net_r']:.2f}R")

    st.caption(
        f"Backtest applies the Daily/4H/1H bias + 15M setup + 5M trigger. "
        f"Maximum holding period: {holding_days} day(s). "
        "Entry is next 5M candle open. Same-candle SL/TP ambiguity is counted "
        "as a loss. Fees are approximated as 0.03R per trade; broker spread/slippage can differ."
    )

    if not bt_trades.empty:
        st.line_chart(
            bt_trades.set_index("Entry Time")["Equity R"]
        )

        st.dataframe(
            bt_trades,
            use_container_width=True
        )

        st.download_button(
            "Download MTF Backtest CSV",
            bt_trades.to_csv(index=False).encode("utf-8"),
            file_name="mtf_backtest.csv",
            mime="text/csv"
        )
    else:
        st.info(
            "No completed trades were found in this data window using the selected rules."
        )


# TRADE JOURNAL TAB
with journal_tab:
    st.subheader("Persistent Trade Journal")

    with st.form("journal_entry_form"):
        j1, j2, j3 = st.columns(3)

        j_side = j1.selectbox(
            "Direction",
            ["BUY", "SELL"]
        )

        j_entry = j1.number_input(
            "Entry",
            value=float(last["close"]),
            format="%.5f"
        )

        j_sl = j2.number_input(
            "Stop loss",
            value=float(plan["SL"]) if plan else default_sl,
            format="%.5f"
        )

        j_tp = j2.number_input(
            "Take profit",
            value=(
                float(plan["TP2"]) if plan
                else float(last["close"] + 3 * last["ATR"])
            ),
            format="%.5f"
        )

        j_result = j3.selectbox(
            "Trade result",
            ["Open", "Win", "Loss", "Break-even"]
        )

        j_notes = j3.text_input("Notes")

        add_trade = st.form_submit_button("Save Trade")

    if add_trade:
        save_journal(
            symbol,
            j_side,
            j_entry,
            j_sl,
            j_tp,
            j_result,
            j_notes
        )
        st.success("Trade saved permanently in the app database.")

    journal = read_journal()

    if not journal.empty:
        st.dataframe(
            journal,
            use_container_width=True
        )

        st.download_button(
            "Download Journal CSV",
            journal.to_csv(index=False).encode("utf-8"),
            file_name="forex_journal.csv",
            mime="text/csv"
        )

        delete_id = st.number_input(
            "Journal row ID to delete",
            min_value=0,
            step=1,
            value=0
        )

        if st.button("Delete selected journal row"):
            if delete_id > 0:
                delete_journal_row(delete_id)
                st.success(
                    "Journal row deleted. Refresh the app to update the table."
                )
            else:
                st.warning("Enter a valid row ID.")
    else:
        st.info("No journal entries yet.")


# TELEGRAM ALERT TAB
with alert_tab:
    st.subheader("Telegram Signal Alert")

    tg_token = st.text_input(
        "Telegram Bot Token",
        type="password",
        value=get_secret("TELEGRAM_BOT_TOKEN", "")
    )

    tg_chat = st.text_input(
        "Telegram Chat ID",
        value=str(get_secret("TELEGRAM_CHAT_ID", ""))
    )

    if st.button("Send current signal to Telegram"):
        if not tg_token or not tg_chat:
            st.error("Bot Token and Chat ID enter করো।")
        else:
            message = (
                f"{symbol} | {analysis['signal']} | "
                f"Quality {analysis['quality']}/100 | "
                f"Price {last['close']:.5f}"
            )

            if plan:
                message += (
                    f"\nEntry: {plan['Entry']:.5f}"
                    f"\nSL: {plan['SL']:.5f}"
                    f"\nTP1: {plan['TP1']:.5f}"
                    f"\nTP2: {plan['TP2']:.5f}"
                    f"\nTP3: {plan['TP3']:.5f}"
                )

            try:
                send_telegram(tg_token, tg_chat, message)
                st.success("Telegram alert sent.")
            except Exception as exc:
                st.error(f"Telegram alert failed: {exc}")

    st.divider()

    st.write(
        "Signal Quality is a rule-confluence score, "
        "not the probability that a trade will win."
    )

    st.write(
        "This app does not place trades or connect to a broker for execution."
    )


st.caption(
    f"Last completed 5M candle: {last['datetime']} | "
    f"App refreshed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
)
