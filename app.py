import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(
    page_title="XAUUSD Gold Market Analysis",
    page_icon="📊",
    layout="wide",
)


def safe_accuracy_pct(wins, total):
    """Return a bounded win-rate percentage; invalid counts return 0."""
    try:
        wins = int(wins)
        total = int(total)
        if total <= 0 or wins < 0:
            return 0.0
        # A win rate cannot exceed 100%, even if a caller passes bad counts.
        return min(100.0, (wins / total) * 100.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0

def trade_result_from_path(entry, sl, tp, highs, lows, direction):
    """Determine which target/stop was hit first from OHLC paths."""
    direction = str(direction).upper()
    for hi, lo in zip(highs, lows):
        if direction == "BUY":
            if lo <= sl:
                return "LOSS"
            if hi >= tp:
                return "WIN"
        elif direction == "SELL":
            if hi >= sl:
                return "LOSS"
            if lo <= tp:
                return "WIN"
    return "OPEN"


def resolve_bar_exit(direction, sl, tp, bar_open, bar_high, bar_low):
    """Return conservative OHLC exit outcome/price, accounting for opening gaps."""
    direction = str(direction).upper()
    values = [float(sl), float(tp), float(bar_open), float(bar_high), float(bar_low)]
    if direction not in ("BUY", "SELL") or not all(np.isfinite(v) for v in values):
        return None, None
    sl, tp, bar_open, bar_high, bar_low = values

    if direction == "BUY":
        if bar_open <= sl:
            return "LOSS", bar_open
        if bar_open >= tp:
            return "WIN", bar_open
        if bar_low <= sl:
            return "LOSS", sl
        if bar_high >= tp:
            return "WIN", tp
    else:
        if bar_open >= sl:
            return "LOSS", bar_open
        if bar_open <= tp:
            return "WIN", bar_open
        if bar_high >= sl:
            return "LOSS", sl
        if bar_low <= tp:
            return "WIN", tp
    return None, None


SYMBOLS = {
    "Gold (XAU/USD)": "XAU/USD",
}

INTERVALS = {
    "1H": "1h",
    "15M": "15min",
    "5M": "5min",
}

def smart_mtf_intervals(main_interval):
    """Return only useful higher-timeframe checks; avoid duplicate/lower TF API calls."""
    mapping = {
        "5min": ["15min", "1h"],
        "15min": ["1h"],
        "1h": [],
    }
    return mapping.get(main_interval, ["1h"])

st.title("🥇 XAUUSD Gold Market Analysis")
st.caption(
    "XAUUSD only: closed-candle trend analysis + signal diagnostics + gap-aware backtest. "
    "Educational tool; historical results do not guarantee future performance."
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
        options=[150, 200, 300, 400, 500, 750, 1000],
        value=1000,
        help="Loads up to 1,000 candles for a larger historical backtest sample. More candles improve sample size, not signal accuracy by themselves.",
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
    estimated_cost_r = st.number_input(
        "Estimated round-trip cost (R)", min_value=0.0, max_value=0.50,
        value=0.05, step=0.01,
        help="Approximate spread + commission + slippage as a fraction of initial risk. This is an estimate, not broker-specific pricing."
    )

    st.divider()
    st.subheader("Confirmation")
    mtf_enabled = st.checkbox(
        "Enable MTF confirmation",
        value=False,
        help="Uses extra API requests. Keep off to save API credits; enable when needed.",
    )
    strict_mtf = st.checkbox(
        "Strict MTF filter",
        value=True,
        help="If enabled, conflicting higher-timeframe direction becomes WAIT.",
    )
    st.caption("Smart MTF: 5M→15M+1H, 15M→1H, 1H→no extra MTF request.")

    min_score = st.slider(
        "Minimum technical score",
        min_value=5,
        max_value=9,
        value=6,
        help="Default restored to 6 so the dashboard can compare BUY and SELL samples without prematurely filtering signals. Compare direction, score and multiple periods; this does not guarantee accuracy.",
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
def test_twelve_data_api(api_key, symbols, interval="15min"):
    """Rate-limit-aware connectivity test for supported markets."""
    results = {}
    url = "https://api.twelvedata.com/time_series"
    for name, test_symbol in symbols.items():
        try:
            response = requests.get(
                url,
                params={
                    "symbol": test_symbol,
                    "interval": interval,
                    "outputsize": 2,
                    "apikey": api_key,
                    "format": "JSON",
                },
                timeout=15,
            )
            if response.status_code == 429:
                results[name] = {"status": "RATE LIMIT", "message": "HTTP 429 — too many requests"}
                # Do not hammer the provider after a rate limit.
                break
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "error":
                results[name] = {"status": "ERROR", "message": data.get("message", "Provider error")}
                continue
            values = data.get("values") or []
            if not values:
                results[name] = {"status": "ERROR", "message": "No candle data returned"}
                continue
            results[name] = {
                "status": "OK",
                "message": f"{len(values)} candles · latest close {float(values[0]['close']):,.5f}",
            }
        except requests.exceptions.Timeout:
            results[name] = {"status": "ERROR", "message": "Timeout"}
        except requests.exceptions.RequestException as exc:
            results[name] = {"status": "ERROR", "message": str(exc)[:100]}
        except Exception as exc:
            results[name] = {"status": "ERROR", "message": str(exc)[:100]}
    return results


with st.sidebar:
    st.divider()
    st.subheader("🧪 Dashboard Accuracy Test")
    if st.button("Run dashboard logic self-test", use_container_width=True):
        test_results = []
        # Pure logic checks do not consume Twelve Data API credits.
        try:
            test_results.append(("Win-rate calculation", safe_accuracy_pct(2, 4) == 50.0))
            test_results.append(("Invalid win-rate guard", safe_accuracy_pct(5, 0) == 0.0))
            test_results.append(("BUY stop-loss precedence", trade_result_from_path(100, 95, 110, [111], [94], "BUY") == "LOSS"))
            test_results.append(("SELL stop-loss precedence", trade_result_from_path(100, 105, 90, [106], [89], "SELL") == "LOSS"))
            test_results.append(("BUY gap-through SL fill", resolve_bar_exit("BUY", 95, 110, 94, 102, 93) == ("LOSS", 94.0)))
            test_results.append(("SELL gap-through SL fill", resolve_bar_exit("SELL", 105, 90, 106, 107, 101) == ("LOSS", 106.0)))
            test_results.append(("BUY gap-through TP fill", resolve_bar_exit("BUY", 95, 110, 111, 113, 109) == ("WIN", 111.0)))
            test_results.append(("SELL gap-through TP fill", resolve_bar_exit("SELL", 105, 90, 89, 91, 87) == ("WIN", 89.0)))
            test_results.append(("Same-candle SL-first rule", resolve_bar_exit("BUY", 95, 110, 100, 111, 94) == ("LOSS", 95.0)))
            test_results.append(("MTF 15M mapping", smart_mtf_intervals("15min") == ["1h"]))
            test_results.append(("MTF 5M mapping", smart_mtf_intervals("5min") == ["15min", "1h"]))
            for test_name, passed in test_results:
                if passed:
                    st.success(f"PASS: {test_name}")
                else:
                    st.error(f"FAIL: {test_name}")
            if all(passed for _, passed in test_results):
                st.success(f"Dashboard logic self-test passed: {len(test_results)}/{len(test_results)} checks.")
            else:
                st.warning(f"Dashboard logic self-test: {sum(p for _, p in test_results)}/{len(test_results)} checks passed.")
        except Exception as exc:
            st.error(f"Dashboard self-test could not complete: {str(exc)[:150]}")

    st.divider()
    st.subheader("🔌 API Connection")
    st.caption("To reduce API-credit usage, this test checks only 3 markets per run.")

    if "api_test_offset" not in st.session_state:
        st.session_state.api_test_offset = 0

    if st.button("🧪 Test next 3 markets", use_container_width=True):
        market_items = list(SYMBOLS.items())
        batch_size = 3
        start_at = st.session_state.api_test_offset % len(market_items)
        batch = market_items[start_at:start_at + batch_size]
        if len(batch) < batch_size:
            batch += market_items[:batch_size - len(batch)]
        batch_symbols = dict(batch)
        with st.spinner("Testing 3 markets only..."):
            api_results = test_twelve_data_api(api_key, batch_symbols, "15min")
        for market_name, result in api_results.items():
            if result["status"] == "OK":
                st.success(f"✅ {market_name}: {result['message']}")
            elif result["status"] == "RATE LIMIT":
                st.warning(f"⏳ {market_name}: {result['message']}")
            else:
                st.error(f"❌ {market_name}: {result['message']}")

        ok_count = sum(1 for result in api_results.values() if result["status"] == "OK")
        rate_limited = any(result["status"] == "RATE LIMIT" for result in api_results.values())
        error_count = sum(1 for result in api_results.values() if result["status"] == "ERROR")
        st.divider()
        st.markdown("**API Test Summary (this batch only)**")
        st.write(f"Successful: {ok_count} / {len(batch_symbols)}")
        st.write(f"Failed: {error_count}")
        st.write(f"Not tested in this batch: {len(SYMBOLS) - len(batch_symbols)} other markets")
        if rate_limited:
            st.info("Rate limit reached. Stop testing and wait at least 60 seconds before trying again.")
        if error_count:
            st.caption("A 404 means the provider did not find that symbol/endpoint. Verify the exact symbol in Twelve Data's Commodities list; it does not automatically mean your API key is wrong.")
        if not rate_limited:
            st.session_state.api_test_offset = (start_at + batch_size) % len(market_items)
            st.caption("Press the button again to test the next batch of 3 markets.")


# -----------------------------
# Data
# -----------------------------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_candles(api_key, symbol, interval, outputsize):
    url = "https://api.twelvedata.com/time_series"

    request_interval = interval
    request_outputsize = int(outputsize)

    # Request timestamps in Bangladesh local time explicitly. This makes naive
    # provider timestamps interpretable and keeps the dashboard display consistent.
    requested_timezone = "Asia/Dhaka"
    response = requests.get(
        url,
        params={
            "symbol": symbol,
            "interval": request_interval,
            "outputsize": request_outputsize,
            "timezone": requested_timezone,
            "apikey": api_key,
            "format": "JSON",
        },
        timeout=20,
    )
    if response.status_code == 429:
        raise RuntimeError("Twelve Data rate limit (429). Please wait 30–60 seconds before retrying.")
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

    # Normalize timestamps before validation, but record defects BEFORE cleaning.
    meta = data.get("meta", {}) or {}
    raw_times = pd.to_datetime(df["datetime"], errors="coerce")
    if getattr(raw_times.dt, "tz", None) is None:
        try:
            raw_times = raw_times.dt.tz_localize(
                requested_timezone, ambiguous="NaT", nonexistent="shift_forward"
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Could not interpret provider timestamps in Asia/Dhaka timezone."
            ) from exc
    df["datetime"] = raw_times.dt.tz_convert("UTC")

    raw_issues = []
    if df[required].isna().any().any():
        raw_issues.append("Raw API response contains missing or invalid datetime/OHLC values.")
    if df["datetime"].duplicated().any():
        raw_issues.append("Raw API response contains duplicate candle timestamps.")
    if not (
        df["datetime"].is_monotonic_increasing
        or df["datetime"].is_monotonic_decreasing
    ):
        raw_issues.append("Raw API candles have mixed timestamp order; sorted before analysis.")
    raw_ohlc_bad = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    if raw_ohlc_bad.any():
        raw_issues.append(f"Raw API response contains {int(raw_ohlc_bad.sum())} invalid/non-positive OHLC candle(s).")
    meta["_raw_validation_issues"] = raw_issues

    df = (
        df.dropna(subset=required)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    df = df[
        (df[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (df["high"] >= df[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
    ].reset_index(drop=True)

    if len(df) < 80:
        raise ValueError("Not enough valid candles returned by the data provider after cleaning.")

    return df, meta


def validate_market_data(df, interval):
    """Validate candle chronology/OHLC consistency without making extra API calls."""
    problems = []
    if df is None or df.empty:
        return ["No candle rows received."]
    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"Missing fields: {', '.join(missing)}"]
    if df[required].isna().any().any():
        problems.append("Some candle rows contain missing values.")
    if df["datetime"].duplicated().any():
        problems.append("Duplicate candle timestamps detected.")
    if not df["datetime"].is_monotonic_increasing:
        problems.append("Candle timestamps are not in chronological order.")

    # Report unusually large gaps between candles. Forex weekend/holiday
    # closures can create legitimate gaps, so this is a warning, not a failure.
    expected_minutes = {
        "5min": 5,
        "15min": 15,
        "1h": 60,
    }.get(interval)
    if expected_minutes and len(df) > 1:
        candle_times = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        gaps = candle_times.diff().dropna().dt.total_seconds() / 60
        unusual_gaps = gaps[gaps > expected_minutes * 3]
        if not unusual_gaps.empty:
            largest_gap = float(unusual_gaps.max())
            problems.append(
                f"{len(unusual_gaps)} unusually large candle gap(s) found; "
                f"largest gap {largest_gap:.0f} minutes. This may be a normal "
                "Forex market closure or missing provider data; review before trading."
            )

    numeric = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    invalid_numeric = numeric.isna().any(axis=1) | ~np.isfinite(numeric).all(axis=1)
    non_positive = (numeric <= 0).any(axis=1)
    bad_ohlc = (
        (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1))
        | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))
    )
    if invalid_numeric.any():
        problems.append(f"{int(invalid_numeric.sum())} candle(s) contain non-numeric or infinite OHLC values.")
    if non_positive.any():
        problems.append(f"{int(non_positive.sum())} candle(s) contain zero/negative prices.")
    if bad_ohlc.any():
        problems.append(f"{int(bad_ohlc.sum())} candle(s) have inconsistent OHLC values.")
    latest = pd.to_datetime(df["datetime"].iloc[-1], errors="coerce")
    if pd.isna(latest):
        problems.append("Latest candle timestamp cannot be parsed.")
    elif latest.tzinfo is None:
        problems.append(
            "Timestamps are timezone-naive. The API request asks for Asia/Dhaka, "
            "but timezone normalization failed; age check is inconclusive."
        )
    else:
        now = pd.Timestamp.now(tz="UTC")
        age_hours = (now - latest.tz_convert("UTC")).total_seconds() / 3600
        if age_hours < -1:
            problems.append(f"Latest candle is {abs(age_hours):.1f} hours in the future; check API timezone metadata.")
        elif age_hours > 72:
            problems.append(f"Latest candle is stale ({age_hours:.1f} hours old).")
    return problems


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
    d.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100
    d.loc[(avg_loss == 0) & (avg_gain == 0), "RSI"] = 50

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

    # Directional movement confirmation (Wilder-style smoothed DI).
    d["PLUS_DI"] = plus_di
    d["MINUS_DI"] = minus_di

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
def candle_confirmation(row, previous_row=None):
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    body_ratio = body / rng

    # A real engulfing pattern must cover the previous candle body.
    bullish_engulf = False
    bearish_engulf = False
    if previous_row is not None:
        prev_o = float(previous_row["open"])
        prev_c = float(previous_row["close"])
        bullish_engulf = (
            c > o
            and prev_c < prev_o
            and o <= prev_c
            and c >= prev_o
        )
        bearish_engulf = (
            c < o
            and prev_c > prev_o
            and o >= prev_c
            and c <= prev_o
        )

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

    # Use only candles up to the row being evaluated. In live closed-candle
    # mode the dataframe may also contain a newer, still-forming candle.
    recent = df
    try:
        if row.name in df.index:
            recent = df.loc[:row.name].tail(20)
    except Exception:
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
def get_signal(row, df, min_score=6, experiment=None):
    cfg = {"rsi_buy":52.0,"rsi_sell":48.0,"adx_score":18.0,"adx_gate":20.0,"extension_atr":1.25,"use_candle":True,"use_di":True,"use_slope":True,"use_ema_structure":True}
    if experiment: cfg.update(experiment)
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
            "technical_score_pct": 0,
            "buy_score": 0,
            "sell_score": 0,
            "reasons": [],
            "regime": "UNKNOWN",
            "candle": "Unknown",
        }

    previous_row = None
    try:
        # In the dashboard and backtest, df contains candles up to this row.
        if len(df) >= 2 and df.index[-1] == row.name:
            previous_row = df.iloc[-2]
        elif row.name in df.index:
            row_pos = df.index.get_loc(row.name)
            if isinstance(row_pos, (int, np.integer)) and row_pos > 0:
                previous_row = df.iloc[row_pos - 1]
    except Exception:
        previous_row = None

    candle = candle_confirmation(row, previous_row)
    regime = market_regime(row, df)

    buy_checks = [
        bool(row["close"] > row["EMA20"]),
        bool(row["EMA20"] > row["EMA50"]),
        bool(row["EMA50"] > row["EMA200"]),
        bool(row["RSI"] > cfg["rsi_buy"]),
        bool(row["MACD"] > row["MACD_signal"]),
        bool(row["ADX"] >= cfg["adx_score"]),
        bool((candle["bull"] and not candle["doji"]) if cfg["use_candle"] else True),
        bool(row["close"] > row["BB_mid"]),
        bool(regime in ["UPTREND", "STRONG UPTREND"]),
    ]

    sell_checks = [
        bool(row["close"] < row["EMA20"]),
        bool(row["EMA20"] < row["EMA50"]),
        bool(row["EMA50"] < row["EMA200"]),
        bool(row["RSI"] < cfg["rsi_sell"]),
        bool(row["MACD"] < row["MACD_signal"]),
        bool(row["ADX"] >= cfg["adx_score"]),
        bool((candle["bear"] and not candle["doji"]) if cfg["use_candle"] else True),
        bool(row["close"] < row["BB_mid"]),
        bool(regime in ["DOWNTREND", "STRONG DOWNTREND"]),
    ]

    buy_score = int(sum(buy_checks))
    sell_score = int(sum(sell_checks))
    max_score = len(buy_checks)

    # Quality gate: require trend alignment before allowing a scored signal.
    # This intentionally reduces signal frequency to avoid counter-trend entries.
    # V8 quality gate: require directional trend structure, DI agreement,
    # positive/negative EMA20 slope, and avoid chasing an overextended candle.
    # These are filters, not a claim of a particular future win probability.
    prior_ema20 = (
        float(previous_row["EMA20"])
        if previous_row is not None and pd.notna(previous_row.get("EMA20"))
        else float(row["EMA20"])
    )
    ema20_slope_up = float(row["EMA20"]) > prior_ema20
    ema20_slope_down = float(row["EMA20"]) < prior_ema20
    atr_value = float(row["ATR"])
    extension_atr = (
        abs(float(row["close"]) - float(row["EMA20"])) / atr_value
        if atr_value > 0 else np.inf
    )
    plus_di_value = float(row.get("PLUS_DI", np.nan))
    minus_di_value = float(row.get("MINUS_DI", np.nan))
    di_valid = np.isfinite(plus_di_value) and np.isfinite(minus_di_value)

    buy_quality = (
        (row["close"] > row["EMA50"] > row["EMA200"] and row["EMA20"] > row["EMA50"] if cfg["use_ema_structure"] else True)
        and (ema20_slope_up if cfg["use_slope"] else True)
        and row["RSI"] > (cfg["rsi_buy"] - 2.0)
        and row["MACD"] > row["MACD_signal"]
        and row["ADX"] >= cfg["adx_gate"]
        and ((di_valid and plus_di_value > minus_di_value) if cfg["use_di"] else True)
        and extension_atr <= cfg["extension_atr"]
        and regime in ["UPTREND", "STRONG UPTREND"]
    )
    sell_quality = (
        (row["close"] < row["EMA50"] < row["EMA200"] and row["EMA20"] < row["EMA50"] if cfg["use_ema_structure"] else True)
        and (ema20_slope_down if cfg["use_slope"] else True)
        and row["RSI"] < (cfg["rsi_sell"] + 2.0)
        and row["MACD"] < row["MACD_signal"]
        and row["ADX"] >= cfg["adx_gate"]
        and ((di_valid and minus_di_value > plus_di_value) if cfg["use_di"] else True)
        and extension_atr <= cfg["extension_atr"]
        and regime in ["DOWNTREND", "STRONG DOWNTREND"]
    )

    # A high score alone is not enough if the directional quality gate fails.
    if buy_score >= min_score and buy_score > sell_score and buy_quality:
        signal = "STRONG BUY" if buy_score >= 8 else "BUY"
    elif sell_score >= min_score and sell_score > buy_score and sell_quality:
        signal = "STRONG SELL" if sell_score >= 8 else "SELL"
    else:
        signal = "WAIT"

    score = max(buy_score, sell_score)
    confluence_pct = int(round((score / max_score) * 100))

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
        "technical_score_pct": confluence_pct,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "reasons": reasons,
        "regime": regime,
        "candle": candle["label"],
        "quality_failures": [
            label for label, passed in [
                ("EMA trend structure", (row["close"] > row["EMA50"] > row["EMA200"] and row["EMA20"] > row["EMA50"]) if cfg["use_ema_structure"] else True),
                ("EMA20 slope", ema20_slope_up if buy_score >= sell_score else (ema20_slope_down if cfg["use_slope"] else True)),
                ("RSI threshold", row["RSI"] > (cfg["rsi_buy"] - 2.0) if buy_score >= sell_score else row["RSI"] < (cfg["rsi_sell"] + 2.0)),
                ("MACD direction", row["MACD"] > row["MACD_signal"] if buy_score >= sell_score else row["MACD"] < row["MACD_signal"]),
                ("ADX strength", row["ADX"] >= cfg["adx_gate"]),
                ("Directional DI", (di_valid and (plus_di_value > minus_di_value if buy_score >= sell_score else minus_di_value > plus_di_value)) if cfg["use_di"] else True),
                ("Not overextended", extension_atr <= cfg["extension_atr"]),
                ("Trend regime", regime in (["UPTREND", "STRONG UPTREND"] if buy_score >= sell_score else ["DOWNTREND", "STRONG DOWNTREND"])),
            ] if not passed
        ],
    }


# -----------------------------
# Smart SL / TP
# -----------------------------
def valid_directional_levels(direction, entry, sl, tp):
    """Reject malformed levels before they can appear in a live setup or backtest."""
    try:
        entry, sl, tp = float(entry), float(sl), float(tp)
    except (TypeError, ValueError, OverflowError):
        return False
    if not all(np.isfinite(v) for v in (entry, sl, tp)):
        return False
    direction = str(direction).upper()
    if direction == "BUY":
        return sl < entry < tp
    if direction == "SELL":
        return tp < entry < sl
    return False


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
        # Reject structurally distant stops instead of creating unrealistic,
        # oversized targets from an old swing low.
        if risk <= 0 or risk > 2.5 * atr:
            return None
        trade = {
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp1": entry + risk * rr1,
            "tp2": entry + risk * rr2,
            "tp3": entry + risk * rr3,
            "risk_distance": risk,
        }
        if not all(valid_directional_levels(direction, entry, sl, trade[k]) for k in ("tp1", "tp2", "tp3")):
            return None
        return trade

    atr_sl = entry + 1.5 * atr
    swing_sl = recent_high + 0.10 * atr if np.isfinite(recent_high) else atr_sl
    sl = max(atr_sl, swing_sl)
    risk = sl - entry
    # Reject structurally distant stops instead of creating unrealistic,
    # oversized targets from an old swing high.
    if risk <= 0 or risk > 2.5 * atr:
        return None

    trade = {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": entry - risk * rr1,
        "tp2": entry - risk * rr2,
        "tp3": entry - risk * rr3,
        "risk_distance": risk,
    }
    if not all(valid_directional_levels(direction, entry, sl, trade[k]) for k in ("tp1", "tp2", "tp3")):
        return None
    return trade


def add_trade_levels(df):
    d = df.copy()
    d["recent_low"] = d["low"].rolling(6).min().shift(1)
    d["recent_high"] = d["high"].rolling(6).max().shift(1)
    return d


# -----------------------------
# Backtest
# -----------------------------
def expected_interval_minutes(interval):
    return {"5min": 5, "15min": 15, "1h": 60}.get(interval)


def backtest(df, min_score=6, rr=2.0, cost_r=0.05, interval=None, start_index=None, end_index=None, experiment=None):
    """Conservative non-overlapping bar-by-bar backtest with estimated costs in R."""
    try:
        rr = float(rr)
        cost_r = float(cost_r)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("rr and cost_r must be numeric.")
    if not np.isfinite(rr) or rr <= 0:
        raise ValueError("rr must be a finite number greater than zero.")
    if not np.isfinite(cost_r) or cost_r < 0:
        raise ValueError("cost_r must be a finite, non-negative number.")
    empty_stats = {
        "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
        "profit_factor": 0.0, "net_r": 0.0, "max_drawdown_r": 0.0,
        "gross_r": 0.0, "cost_r": float(cost_r), "open_trades": 0,
    }
    if len(df) < 100:
        return pd.DataFrame(), empty_stats

    trades = []
    # Optional index bounds allow a chronological holdout test while preserving
    # indicator warm-up candles that occurred before the test period.
    start_i = max(100, int(start_index)) if start_index is not None else 100
    stop_i = min(len(df), int(end_index)) if end_index is not None else len(df)
    if start_i < 100 or stop_i > len(df) or stop_i <= start_i + 1:
        return pd.DataFrame(), empty_stats
    i = start_i
    while i < stop_i - 1:
        row = df.iloc[i]
        result = get_signal(row, df.iloc[:i + 1], min_score=min_score, experiment=experiment)
        if result["signal"] not in ["BUY", "STRONG BUY", "SELL", "STRONG SELL"]:
            i += 1
            continue

        setup = build_trade(row, result["signal"], 1.0, rr, rr)
        if setup is None:
            i += 1
            continue

        next_i = i + 1
        next_bar = df.iloc[next_i]
        # Do not open a backtest position after a provider/session gap. The
        # signal may be stale after a weekend, missing candles, or market halt.
        expected_minutes = expected_interval_minutes(interval) if interval else None
        if expected_minutes:
            signal_time = pd.to_datetime(row["datetime"], utc=True, errors="coerce")
            entry_time = pd.to_datetime(next_bar["datetime"], utc=True, errors="coerce")
            elapsed_minutes = (
                (entry_time - signal_time).total_seconds() / 60
                if pd.notna(signal_time) and pd.notna(entry_time) else np.inf
            )
            if elapsed_minutes <= 0 or elapsed_minutes > expected_minutes * 1.5:
                i += 1
                continue
        entry = float(next_bar["open"])
        direction = setup["direction"]
        risk = float(setup["risk_distance"])
        if not np.isfinite(entry) or not np.isfinite(risk) or risk <= 0:
            i += 1
            continue

        sl = entry - risk if direction == "BUY" else entry + risk
        tp = entry + risk * rr if direction == "BUY" else entry - risk * rr
        # Never record a trade if its SL/TP are on the wrong side of entry.
        if not valid_directional_levels(direction, entry, sl, tp):
            i += 1
            continue
        outcome, exit_price, exit_time, exit_i = None, None, None, next_i

        for j in range(next_i, stop_i):
            bar = df.iloc[j]
            bar_open = float(bar["open"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])

            outcome, exit_price = resolve_bar_exit(
                direction, sl, tp, bar_open, bar_high, bar_low
            )
            if outcome is None:
                continue

            exit_time, exit_i = bar["datetime"], j
            break

        if outcome is None:
            trades.append({
                "time": next_bar["datetime"], "direction": direction, "entry": entry,
                "SL": sl, "TP": tp, "result": "OPEN", "gross_R": np.nan,
                "cost_R": float(cost_r), "R": np.nan, "exit_time": None,
                "score": result["score"], "regime": result["regime"],
            })
            break

        # Use actual modeled exit fill. Gap-through stops can lose more than 1R;
        # favorable gaps can earn more than the planned RR.
        signed_move = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
        gross_r = float(signed_move / risk)
        net_r = gross_r - float(cost_r)
        trades.append({
            "time": next_bar["datetime"], "direction": direction, "entry": entry,
            "SL": sl, "TP": tp, "result": outcome, "gross_R": gross_r,
            "cost_R": float(cost_r), "R": net_r,
            "exit_time": exit_time, "score": result["score"], "regime": result["regime"],
        })
        i = max(exit_i + 1, i + 1)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, empty_stats

    closed = trades_df[trades_df["result"].isin(["WIN", "LOSS"])].copy()
    wins = int((closed["result"] == "WIN").sum())
    losses = int((closed["result"] == "LOSS").sum())
    # Profit factor is conventionally calculated from gross winning/losing trades.
    # Keep estimated costs in Net R, but do not mix them into the gross PF calculation.
    gross_profit = float(closed.loc[closed["gross_R"] > 0, "gross_R"].sum()) if not closed.empty else 0.0
    gross_loss = abs(float(closed.loc[closed["gross_R"] < 0, "gross_R"].sum())) if not closed.empty else 0.0
    equity = closed["R"].cumsum() if not closed.empty else pd.Series(dtype=float)
    max_dd = abs(float((equity - equity.cummax()).min())) if len(equity) else 0.0
    return trades_df, {
        "trades": len(closed), "wins": wins, "losses": losses,
        "win_rate": wins / len(closed) * 100 if len(closed) else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0),
        "net_r": float(closed["R"].sum()) if not closed.empty else 0.0,
        "gross_r": float(closed["gross_R"].sum()) if not closed.empty else 0.0,
        "max_drawdown_r": max_dd, "cost_r": float(cost_r),
        "open_trades": int((trades_df["result"] == "OPEN").sum()),
    }


def direction_stats(trades_df):
    """Compare BUY and SELL outcomes using closed trades only."""
    if trades_df.empty:
        return pd.DataFrame()

    closed_all = trades_df[trades_df["result"].isin(["WIN", "LOSS"])].copy()
    rows = []
    for direction in ["BUY", "SELL"]:
        part = closed_all[closed_all["direction"] == direction].copy()
        open_count = int(
            ((trades_df["direction"] == direction) & (trades_df["result"] == "OPEN")).sum()
        )
        wins = int((part["result"] == "WIN").sum())
        losses = int((part["result"] == "LOSS").sum())
        gross_profit = float(part.loc[part["gross_R"] > 0, "gross_R"].sum()) if not part.empty else 0.0
        gross_loss = abs(float(part.loc[part["gross_R"] < 0, "gross_R"].sum())) if not part.empty else 0.0
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (np.inf if gross_profit > 0 else 0.0)
        )
        equity = part["R"].cumsum() if not part.empty else pd.Series(dtype=float)
        max_drawdown = (
            abs(float((equity - equity.cummax()).min())) if len(equity) else 0.0
        )
        rows.append({
            "Direction": direction,
            "Closed Trades": len(part),
            "Wins": wins,
            "Losses": losses,
            "Open": open_count,
            "Win Rate %": round(wins / len(part) * 100, 1) if len(part) else 0.0,
            "Gross R": round(float(part["gross_R"].sum()), 2) if not part.empty else 0.0,
            "Net R": round(float(part["R"].sum()), 2) if not part.empty else 0.0,
            "Profit Factor": "∞" if np.isinf(profit_factor) else round(profit_factor, 2),
            "Max DD R": round(max_drawdown, 2),
        })
    return pd.DataFrame(rows)


def mtf_analysis(api_key, symbol, outputsize, main_interval):
    results = []

    for tf in smart_mtf_intervals(main_interval):
        try:
            raw, _ = fetch_candles(api_key, symbol, tf, min(outputsize, 200))
            data = add_trade_levels(add_indicators(raw))
            row = data.iloc[-2] if len(data) > 2 else data.iloc[-1]
            sig = get_signal(row, data.iloc[: data.index.get_loc(row.name) + 1], min_score=6)

            results.append(
                {
                    "Timeframe": tf,
                    "Signal": sig["signal"],
                    "Score": f"{sig['score']}/{sig['max_score']}",
                    "Confluence": f"{sig['technical_score_pct']}%",
                    "Regime": sig["regime"],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "Timeframe": tf,
                    "Signal": "ERROR",
                    "Score": "-",
                    "Confluence": "-",
                    "Regime": str(exc)[:45],
                }
            )

    return pd.DataFrame(results)


def mtf_direction(mtf_df):
    if mtf_df.empty:
        return "UNKNOWN"

    directions = []
    for s in mtf_df["Signal"]:
        if "BUY" in str(s):
            directions.append("BUY")
        elif "SELL" in str(s):
            directions.append("SELL")
        else:
            directions.append("WAIT")

    buy = directions.count("BUY")
    sell = directions.count("SELL")

    # Smart MTF uses at most two higher timeframes. Require a clear majority.
    if buy > sell and buy >= max(1, int(np.ceil(len(directions) * 0.67))):
        return "BUY"
    if sell > buy and sell >= max(1, int(np.ceil(len(directions) * 0.67))):
        return "SELL"
    if buy == 0 and sell == 0:
        return "CONFLICT"
    return "CONFLICT"


# -----------------------------
# Load main data
# -----------------------------
try:
    raw_df, meta = fetch_candles(api_key, symbol, interval, outputsize)
    df = add_trade_levels(add_indicators(raw_df))
except Exception as exc:
    if "429" in str(exc):
        st.warning(f"⚠️ Market data temporarily rate-limited: {exc}")
        st.info("Wait 30–60 seconds, then use Refresh market data. Avoid repeatedly pressing API Test or Refresh.")
    else:
        st.error(f"Market data error: {exc}")
    st.stop()

# API response/data integrity test uses already-loaded cached data (no extra API credits).
with st.expander("🧪 Twelve Data API / Market Data Validation", expanded=False):
    st.caption("Checks the candles already loaded by the dashboard. This does not send another API request.")
    if st.button("Run Market Data Validation", key="validate_loaded_market_data"):
        _data_issues = list(meta.get("_raw_validation_issues", []))
        _data_issues.extend(validate_market_data(raw_df, interval))
        if _data_issues:
            st.warning("Market data validation found warnings/issues:")
            for _issue in dict.fromkeys(_data_issues):
                st.write(f"• {_issue}")
            st.info("Raw API defects are reported before cleaning. The analysis uses cleaned candles only.")
        else:
            st.success(f"PASS: {len(raw_df)} cleaned candles passed timestamp, ordering, positive-price and OHLC checks.")

# Use closed candle by default to reduce signal flipping/repainting.
if signal_mode == "Closed candle (stable)" and len(df) >= 2:
    signal_index = -2
else:
    signal_index = -1

signal_row = df.iloc[signal_index]
signal_history_df = df.iloc[: len(df) - 1] if signal_index == -2 else df

signal_info = get_signal(
    signal_row,
    signal_history_df,
    min_score=min_score,
)

trade = build_trade(
    signal_row,
    signal_info["signal"],
    rr1,
    rr2,
    rr3,
)

# In stable closed-candle mode, enter at the next candle's open, matching backtest.
# Live-candle mode remains provisional and uses the currently forming candle close.
if trade and signal_mode == "Closed candle (stable)" and len(df) >= 2:
    next_open = float(df.iloc[-1]["open"])
    risk_distance = float(trade["risk_distance"])
    if np.isfinite(next_open) and risk_distance > 0:
        trade["entry"] = next_open
        if trade["direction"] == "BUY":
            trade["sl"] = next_open - risk_distance
            trade["tp1"] = next_open + risk_distance * rr1
            trade["tp2"] = next_open + risk_distance * rr2
            trade["tp3"] = next_open + risk_distance * rr3
        else:
            trade["sl"] = next_open + risk_distance
            trade["tp1"] = next_open - risk_distance * rr1
            trade["tp2"] = next_open - risk_distance * rr2
            trade["tp3"] = next_open - risk_distance * rr3

# -----------------------------
# MTF confirmation
# -----------------------------
mtf_df = pd.DataFrame()
mtf_dir = "OFF"

if mtf_enabled:
    try:
        mtf_df = mtf_analysis(api_key, symbol, outputsize, interval)
        mtf_dir = mtf_direction(mtf_df)
    except Exception as exc:
        st.warning(f"MTF data unavailable: {exc}")
        mtf_dir = "ERROR"

raw_direction = (
    "BUY"
    if "BUY" in signal_info["signal"]
    else "SELL"
    if "SELL" in signal_info["signal"]
    else "WAIT"
)

final_signal = signal_info["signal"]
# MTF is not confirmed unless it was enabled and returned an aligned direction.
# Never show PASS when the filter is disabled or the higher-timeframe data is unavailable.
mtf_status = "OFF" if not mtf_enabled else "UNAVAILABLE"

if mtf_enabled and mtf_dir in ["BUY", "SELL"]:
    if raw_direction in ["BUY", "SELL"] and raw_direction != mtf_dir:
        mtf_status = "CONFLICT"
        if strict_mtf:
            final_signal = "WAIT"
    elif raw_direction == mtf_dir:
        mtf_status = "PASS"
    else:
        mtf_status = "UNAVAILABLE"
elif mtf_enabled:
    mtf_status = "CONFLICT" if mtf_dir == "CONFLICT" else "UNAVAILABLE"

# Rebuild trade only when final signal survives MTF filter.
if final_signal != signal_info["signal"]:
    trade = None

# -----------------------------
# Header metrics
# -----------------------------
current_price = float(signal_row["close"])
price_time = signal_row["datetime"]
price_time_display = pd.Timestamp(price_time).tz_convert("Asia/Dhaka").strftime("%Y-%m-%d %H:%M:%S %Z")
regime = signal_info["regime"]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Price", f"{current_price:.5f}")
col2.metric("Final Signal", final_signal)
col3.metric("Technical Score", f"{signal_info['score']}/{signal_info['max_score']}")
col4.metric("Confluence (not win probability)", f"{signal_info['technical_score_pct']}%")
col5.metric("Regime", regime)

st.caption(
    f"Analysis candle (Bangladesh time): {price_time_display} | Data refresh cache: 120 sec | "
    f"Signal mode: {signal_mode} | "
    + ("Entry reference: next candle open" if signal_mode == "Closed candle (stable)" else "Entry reference: live candle close (provisional)")
)

# -----------------------------
# Final setup card
# -----------------------------
def format_price(value, symbol_name):
    # Keep enough decimals so Entry/SL/TP never look identical because of rounding.
    if "XAU" in symbol_name or "XAG" in symbol_name:
        return f"{float(value):.2f}"
    if "JPY" in symbol_name:
        return f"{float(value):.3f}"
    return f"{float(value):.5f}"


def validate_trade_levels(trade_obj):
    if not trade_obj:
        return False
    try:
        e = float(trade_obj["entry"])
        sl = float(trade_obj["sl"])
        tp1 = float(trade_obj["tp1"])
        tp2 = float(trade_obj["tp2"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    direction = str(trade_obj.get("direction", "")).upper()
    if direction == "BUY":
        return valid_directional_levels(direction, e, sl, tp1) and valid_directional_levels(direction, e, sl, tp2) and tp1 <= tp2
    if direction == "SELL":
        return valid_directional_levels(direction, e, sl, tp1) and valid_directional_levels(direction, e, sl, tp2) and tp2 <= tp1
    return False


st.subheader("🎯 Final Trade Analysis")

if final_signal in ["BUY", "STRONG BUY", "SELL", "STRONG SELL"] and trade:
    # Hard safety check: Entry, SL and TP must be on the correct sides of Entry.
    if not validate_trade_levels(trade):
        st.error("⚠️ Entry/SL/TP calculation invalid — trade levels hidden to prevent a wrong setup.")
        trade = None
    else:
        risk_amount = account_balance * risk_pct / 100.0
        entry_txt = format_price(trade["entry"], symbol)
        sl_txt = format_price(trade["sl"], symbol)
        tp1_txt = format_price(trade["tp1"], symbol)
        tp2_txt = format_price(trade["tp2"], symbol)
        tp3_txt = format_price(trade["tp3"], symbol)

        # Use two compact rows instead of five narrow columns so values remain
        # readable on mobile screens and are not visually clipped/wrapped.
        a, b, c = st.columns(3)
        a.metric("Direction", trade["direction"])
        b.metric("Entry", entry_txt)
        c.metric("Stop Loss (SL)", sl_txt)

        d, e, f = st.columns(3)
        d.metric("Take Profit 1", tp1_txt)
        e.metric("Take Profit 2", tp2_txt)
        f.metric("Take Profit 3", tp3_txt)

        st.caption(
            f"Risk distance: `{trade['risk_distance']:.5f}` | "
            f"Risk budget: `${risk_amount:.2f}`"
        )

    if mtf_enabled:
        st.write(
            f"**MTF:** `{mtf_status}`  |  **MTF direction:** `{mtf_dir}`"
        )
else:
    st.warning(
        f"🟡 WAIT — setup is not strong enough. "
        f"Technical: {signal_info['score']}/{signal_info['max_score']} | "
        f"MTF: {mtf_status}"
    )

# -----------------------------
# Decision details
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("🧠 Decision Factors")
    factors = {
        "Price / EMA20": "PASS" if (
            (raw_direction == "BUY" and signal_row["close"] > signal_row["EMA20"])
            or
            (raw_direction == "SELL" and signal_row["close"] < signal_row["EMA20"])
        ) else "NO",
        "EMA structure": "PASS" if (
            (raw_direction == "BUY" and signal_row["EMA20"] > signal_row["EMA50"] > signal_row["EMA200"])
            or
            (raw_direction == "SELL" and signal_row["EMA20"] < signal_row["EMA50"] < signal_row["EMA200"])
        ) else "NO",
        "RSI": f"{signal_row['RSI']:.1f}",
        "ADX": f"{signal_row['ADX']:.1f}",
        "MACD": "Bullish" if signal_row["MACD"] > signal_row["MACD_signal"] else "Bearish",
        "Candle": signal_info["candle"],
        "Market regime": regime,
        "MTF": mtf_status,
    }
    st.table(pd.DataFrame(list(factors.items()), columns=["Factor", "Status"]))

    if signal_info.get("signal") == "WAIT" and signal_info.get("quality_failures"):
        st.warning("WAIT কারণ: " + " · ".join(signal_info["quality_failures"]))
    elif signal_info.get("signal") == "WAIT":
        st.info("WAIT: Technical score বা BUY/SELL direction-এর শর্ত পূরণ হয়নি.")

with right:
    st.subheader("📈 Historical Backtest")
    st.caption("Historical win rate is wins divided by closed trades in this sample. Technical confluence counts matching rules; it is NOT a probability or future accuracy forecast.")
    with st.spinner("Running historical backtest..."):
        bt_trades, bt_stats = backtest(
            df,
            min_score=min_score,
            rr=rr2,
            cost_r=estimated_cost_r,
            interval=interval,
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trades", bt_stats["trades"])
    m2.metric("Win rate", f"{bt_stats['win_rate']:.1f}%")
    m3.metric(
        "Profit factor",
        "∞" if np.isinf(bt_stats["profit_factor"]) else f"{bt_stats['profit_factor']:.2f}",
    )
    m4.metric("Net R (after estimated cost)", f"{bt_stats['net_r']:.2f}")

    st.write(
        f"Gross R: **{bt_stats['gross_r']:.2f}R** | "
        f"Estimated cost: **{bt_stats['cost_r']:.2f}R per closed trade** | "
        f"Max drawdown: **{bt_stats['max_drawdown_r']:.2f}R**"
    )
    if bt_stats.get("open_trades", 0):
        st.info("One trade remains OPEN at the end of loaded data. It is shown in history but excluded from win rate and net R.")

    # Chronological holdout: signals are only generated in the final 30% of
    # candles. Earlier candles remain available solely for indicator warm-up.
    _holdout_start = max(200, int(len(df) * 0.70))
    if len(df) - _holdout_start >= 30:
        _holdout_trades, _holdout_stats = backtest(
            df,
            min_score=min_score,
            rr=rr2,
            cost_r=estimated_cost_r,
            interval=interval,
            start_index=_holdout_start,
            end_index=len(df),
        )
        st.markdown("**🧪 Chronological Holdout Check (final 30% of candles)**")
        st.caption("No strategy parameters are fitted on this segment. Earlier candles are used only to warm up indicators; only signals after the split are counted. This is a first-pass holdout, not proof of future accuracy.")
        _h1, _h2, _h3, _h4 = st.columns(4)
        _h1.metric("Holdout closed trades", _holdout_stats["trades"])
        _h2.metric("Holdout win rate", f"{_holdout_stats['win_rate']:.1f}%")
        _h3.metric("Holdout Net R", f"{_holdout_stats['net_r']:.2f}R")
        _h4.metric("Holdout Profit Factor", "∞" if np.isinf(_holdout_stats["profit_factor"]) else f"{_holdout_stats['profit_factor']:.2f}")
        if _holdout_stats["trades"] < 20:
            st.warning("Holdout sample is too small for a reliable conclusion. More historical candles are needed.")
    else:
        st.info("Not enough candles for a separate chronological holdout check; load more history.")

    # Three chronological walk-forward evaluation windows. Each window is
    # evaluated only on its own candles; earlier data is used for indicator
    # warm-up. This exposes whether results depend on one short market regime.
    if len(df) >= 400:
        st.markdown("**🧭 Walk-Forward Consistency Check (3 chronological windows)**")
        st.caption("Same fixed strategy settings in each period. This is a stability check, not parameter optimization. Fewer than 20 closed trades per window is insufficient for a useful conclusion.")
        _wf_rows = []
        _wf_start = max(200, int(len(df) * 0.40))
        _wf_edges = np.linspace(_wf_start, len(df), 4, dtype=int)
        for _wf_n in range(3):
            _wf_a, _wf_b = int(_wf_edges[_wf_n]), int(_wf_edges[_wf_n + 1])
            if _wf_b - _wf_a < 30:
                continue
            _, _wf_stats = backtest(
                df, min_score=min_score, rr=rr2, cost_r=estimated_cost_r,
                interval=interval, start_index=_wf_a, end_index=_wf_b,
            )
            _wf_rows.append({
                "Period": f"Window {_wf_n + 1}",
                "Candles": _wf_b - _wf_a,
                "Closed trades": _wf_stats["trades"],
                "Win rate %": round(_wf_stats["win_rate"], 1),
                "Net R": round(_wf_stats["net_r"], 2),
                "Profit Factor": "∞" if np.isinf(_wf_stats["profit_factor"]) else round(_wf_stats["profit_factor"], 2),
                "Max DD R": round(_wf_stats["max_drawdown_r"], 2),
            })
        if _wf_rows:
            st.dataframe(pd.DataFrame(_wf_rows), use_container_width=True, hide_index=True)
            _wf_total = sum(x["Closed trades"] for x in _wf_rows)
            _wf_positive = sum(1 for x in _wf_rows if x["Net R"] > 0)
            if _wf_total < 30:
                st.warning(f"Only {_wf_total} closed trades across these windows. Not enough evidence to call the strategy accurate.")
            elif _wf_positive < 2:
                st.warning("The strategy was not consistently profitable across the three windows. Do not treat the current settings as validated.")
            else:
                st.info("Review all three windows and validate on a later unseen period before considering live use.")

    # Compare signal-score thresholds on the same chronological holdout.
    # Diagnostic only; it does not automatically change live signals.
    if len(df) - _holdout_start >= 30:
        st.markdown("**Signal Threshold Comparison (same holdout period)**")
        st.caption("Compares minimum scores 6, 7 and 8 on the same final 30% period. Evaluate Net R, Profit Factor, drawdown and trade count together, not win rate alone.")
        _threshold_rows = []
        for _threshold in (6, 7, 8):
            _threshold_trades, _threshold_stats = backtest(
                df, min_score=_threshold, rr=rr2, cost_r=estimated_cost_r,
                interval=interval, start_index=_holdout_start, end_index=len(df),
            )
            _threshold_rows.append({
                "Minimum score": _threshold,
                "Closed trades": _threshold_stats["trades"],
                "Win rate %": round(_threshold_stats["win_rate"], 1),
                "Net R": round(_threshold_stats["net_r"], 2),
                "Profit Factor": "infinite" if np.isinf(_threshold_stats["profit_factor"]) else round(_threshold_stats["profit_factor"], 2),
                "Max drawdown R": round(_threshold_stats["max_drawdown_r"], 2),
            })
        st.dataframe(pd.DataFrame(_threshold_rows), use_container_width=True, hide_index=True)
        st.caption("This is a threshold sensitivity check, not proof of future performance. Very small trade samples are unreliable.")

    # Decision-factor sensitivity tests on the identical chronological holdout.
    if len(df) - _holdout_start >= 30:
        with st.expander("🧪 Decision Factor Experiment (same holdout)", expanded=True):
            st.caption("One factor changes per row. Compare Net R, Profit Factor, drawdown and trade count, not win rate alone. Diagnostic only; live settings remain unchanged.")
            _experiment_cases = [
                ("Baseline", min_score, rr2, {}),
                ("Minimum score 5", 5, rr2, {}), ("Minimum score 7", 7, rr2, {}),
                ("Minimum score 8", 8, rr2, {}), ("Minimum score 9", 9, rr2, {}),
                ("RSI 50/50", min_score, rr2, {"rsi_buy":50.0,"rsi_sell":50.0}),
                ("RSI 55/45", min_score, rr2, {"rsi_buy":55.0,"rsi_sell":45.0}),
                ("ADX 18", min_score, rr2, {"adx_score":18.0,"adx_gate":18.0}),
                ("ADX 22", min_score, rr2, {"adx_score":22.0,"adx_gate":22.0}),
                ("ADX 25", min_score, rr2, {"adx_score":25.0,"adx_gate":25.0}),
                ("Extension 1.0 ATR", min_score, rr2, {"extension_atr":1.0}),
                ("Extension 1.5 ATR", min_score, rr2, {"extension_atr":1.5}),
                ("Extension 2.0 ATR", min_score, rr2, {"extension_atr":2.0}),
                ("Candle filter OFF", min_score, rr2, {"use_candle":False}),
                ("DI filter OFF", min_score, rr2, {"use_di":False}),
                ("EMA slope OFF", min_score, rr2, {"use_slope":False}),
                ("EMA structure OFF", min_score, rr2, {"use_ema_structure":False}),
                ("RR 1:1.5", min_score, 1.5, {}), ("RR 1:2.5", min_score, 2.5, {}),
            ]
            _exp_rows = []
            for _label, _score, _rr, _cfg in _experiment_cases:
                _, _st = backtest(df, min_score=_score, rr=_rr, cost_r=estimated_cost_r,
                    interval=interval, start_index=_holdout_start, end_index=len(df), experiment=_cfg)
                _exp_rows.append({"Test":_label,"Trades":_st["trades"],"Win rate %":round(_st["win_rate"],1),
                    "Net R":round(_st["net_r"],2),"Profit Factor":"∞" if np.isinf(_st["profit_factor"]) else round(_st["profit_factor"],2),
                    "Max DD R":round(_st["max_drawdown_r"],2)})
            st.dataframe(pd.DataFrame(_exp_rows), use_container_width=True, hide_index=True)
            st.caption("This is a one-factor-at-a-time test. Combined interactions are not tested. Small samples are unreliable; validate promising settings on a separate later period.")

    if not bt_trades.empty:
        stats_df = direction_stats(bt_trades)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Score-wise report: closed outcomes only; OPEN trades are counted separately.
        st.markdown("**📊 Score-wise Backtest Report**")
        _score_rows = []
        for _score_value in range(5, 10):
            _score_part = bt_trades[bt_trades["score"] == _score_value]
            _score_closed = _score_part[_score_part["result"].isin(["WIN", "LOSS"])]
            _score_wins = int((_score_closed["result"] == "WIN").sum())
            _score_losses = int((_score_closed["result"] == "LOSS").sum())
            _score_gross_profit = float(_score_closed.loc[_score_closed["gross_R"] > 0, "gross_R"].sum()) if not _score_closed.empty else 0.0
            _score_gross_loss = abs(float(_score_closed.loc[_score_closed["gross_R"] < 0, "gross_R"].sum())) if not _score_closed.empty else 0.0
            _score_pf = (
                _score_gross_profit / _score_gross_loss
                if _score_gross_loss > 0
                else (np.inf if _score_gross_profit > 0 else 0.0)
            )
            _score_rows.append({
                "Score": _score_value,
                "Closed Trades": len(_score_closed),
                "Wins": _score_wins,
                "Losses": _score_losses,
                "Open": int((_score_part["result"] == "OPEN").sum()),
                "Win Rate %": round(_score_wins / len(_score_closed) * 100, 1) if len(_score_closed) else 0.0,
                "Gross R": round(float(_score_closed["gross_R"].sum()), 2) if not _score_closed.empty else 0.0,
                "Net R": round(float(_score_closed["R"].sum()), 2) if not _score_closed.empty else 0.0,
                "Profit Factor": "∞" if np.isinf(_score_pf) else round(_score_pf, 2),
            })
        st.dataframe(pd.DataFrame(_score_rows), use_container_width=True, hide_index=True)
        st.caption("Each row shows trades with exactly that score. OPEN trades are excluded from Win Rate, Gross R, Net R and Profit Factor. A small number of trades is not enough to establish a reliable edge.")
        if bt_stats["trades"] < 20:
            st.warning(f"⚠️ Only {bt_stats['trades']} closed trades. Accuracy is not statistically reliable yet; aim for at least 20–30 trades before judging the strategy.")
        else:
            st.success("Backtest sample size is large enough for a more useful first-pass comparison (still not a guarantee of future performance).")

    with st.expander("🧪 Historical Signal Outcome Check", expanded=False):
        st.caption("Reports outcomes from the loaded historical sample only. It does not measure or predict future signal accuracy. No extra Twelve Data API request is made.")
        if st.button("Run Historical Outcome Check", key="run_signal_accuracy_test"):
            if bt_trades.empty:
                st.warning("No historical trades were generated. Try more candles or a less strict minimum score.")
            else:
                _closed_test = bt_trades[bt_trades["result"].isin(["WIN", "LOSS"])].copy()
                _total_test = len(_closed_test)
                _wins_test = int((_closed_test["result"] == "WIN").sum())
                _losses_test = int((_closed_test["result"] == "LOSS").sum())
                _overall_test = (_wins_test / _total_test * 100) if _total_test else 0.0
                _buy_test = _closed_test[_closed_test["direction"] == "BUY"]
                _sell_test = _closed_test[_closed_test["direction"] == "SELL"]
                _m1, _m2, _m3, _m4 = st.columns(4)
                _m1.metric("Closed trades", _total_test)
                _m2.metric("Wins / Losses", f"{_wins_test} / {_losses_test}")
                _m3.metric("Overall win rate", f"{_overall_test:.1f}%")
                _m4.metric("Net R", f"{_closed_test['R'].sum():.2f}R" if _total_test else "0.00R")
                _buy_n = len(_buy_test)
                _sell_n = len(_sell_test)
                _performance = pd.DataFrame([
                    {"Direction": "BUY", "Closed trades": _buy_n, "Wins": int((_buy_test["result"] == "WIN").sum()), "Losses": int((_buy_test["result"] == "LOSS").sum()), "Win rate": f"{((_buy_test['result'] == 'WIN').sum() / _buy_n * 100) if _buy_n else 0:.1f}%"},
                    {"Direction": "SELL", "Closed trades": _sell_n, "Wins": int((_sell_test["result"] == "WIN").sum()), "Losses": int((_sell_test["result"] == "LOSS").sum()), "Win rate": f"{((_sell_test['result'] == 'WIN').sum() / _sell_n * 100) if _sell_n else 0:.1f}%"},
                ])
                st.dataframe(_performance, use_container_width=True, hide_index=True)
                if _total_test < 30:
                    st.warning("Small sample: this is a preliminary historical result, not a reliable estimate of future win probability. Aim for 30+ closed trades and test other periods too.")
                st.info("This test evaluates the strategy on loaded historical candles. It does not prove future accuracy and does not execute live trades.")

# -----------------------------
# MTF table
# -----------------------------
if mtf_enabled:
    st.subheader("🧭 Multi-Timeframe Confirmation")
    if not mtf_df.empty:
        st.dataframe(mtf_df, use_container_width=True, hide_index=True)
        st.caption(
            "Smart MTF checks only useful higher timeframes to reduce API requests. Higher timeframe alignment is a filter, not a profit guarantee."
        )

# -----------------------------
# Chart
# -----------------------------
st.subheader("📉 Price + Indicators")

chart_df = df.tail(min(180, len(df))).copy()
# Normalize chart inputs explicitly and discard rows Plotly cannot render.
chart_df["datetime"] = pd.to_datetime(
    chart_df["datetime"], utc=True, errors="coerce"
).dt.tz_convert("Asia/Dhaka").dt.tz_localize(None)
_chart_numeric = ["open", "high", "low", "close", "EMA20", "EMA50", "EMA200"]
for _col in _chart_numeric:
    chart_df[_col] = pd.to_numeric(chart_df[_col], errors="coerce")
chart_df = chart_df.replace([np.inf, -np.inf], np.nan)
chart_df = chart_df.dropna(subset=["datetime", "open", "high", "low", "close"])
chart_df = chart_df.sort_values("datetime").reset_index(drop=True)

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
        name="EMA20",
        line=dict(width=1),
    )
)
fig.add_trace(
    go.Scatter(
        x=chart_df["datetime"],
        y=chart_df["EMA50"],
        name="EMA50",
        line=dict(width=1),
    )
)
fig.add_trace(
    go.Scatter(
        x=chart_df["datetime"],
        y=chart_df["EMA200"],
        name="EMA200",
        line=dict(width=1),
    )
)

if trade:
    fig.add_hline(y=trade["entry"], annotation_text="Entry")
    fig.add_hline(y=trade["sl"], annotation_text="SL")
    fig.add_hline(y=trade["tp1"], annotation_text="TP1")
    fig.add_hline(y=trade["tp2"], annotation_text="TP2")

fig.update_layout(
    height=650,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=30, b=10),
    template="plotly_white",
    hovermode="x unified",
)
if chart_df.empty:
    st.warning("Price chart unavailable: no valid timestamp/OHLC rows remained after chart validation.")
else:
    st.caption(
        f"Chart candles: {len(chart_df)} | "
        f"{chart_df['datetime'].iloc[0]} to {chart_df['datetime'].iloc[-1]} (Bangladesh time)"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Backtest details
# -----------------------------
with st.expander("📊 Backtest trade history"):
    if bt_trades.empty:
        st.info("No completed historical trades for the current settings.")
    else:
        display_cols = [
            "time", "direction", "entry", "SL", "TP",
            "result", "gross_R", "cost_R", "R", "score", "regime"
        ]
        history_display = bt_trades[display_cols].tail(100).copy()
        if "time" in history_display.columns:
            history_display["time"] = pd.to_datetime(
                history_display["time"], utc=True
            ).dt.tz_convert("Asia/Dhaka").dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(
            history_display,
            use_container_width=True,
            hide_index=True,
        )

# -----------------------------
# Live indicator values
# -----------------------------
with st.expander("🔎 Current indicator values"):
    indicator_cols = [
        "datetime", "open", "high", "low", "close",
        "EMA20", "EMA50", "EMA200",
        "RSI", "ATR", "ADX", "PLUS_DI", "MINUS_DI", "MACD", "MACD_signal",
        "BB_upper", "BB_mid", "BB_lower"
    ]
    indicator_display = df[indicator_cols].tail(10).copy()
    indicator_display["datetime"] = (
        indicator_display["datetime"].dt.tz_convert("Asia/Dhaka")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    st.dataframe(
        indicator_display,
        use_container_width=True,
        hide_index=True,
    )

# -----------------------------
# Status
# -----------------------------
st.divider()
st.caption(
    "Data source: Twelve Data API. The dashboard uses technical rules and historical "
    "simulation; it does not predict the market with certainty and does not execute trades."
)


# ================= STALE DATA WARNING =================
# Candle timestamps are normalized to UTC in fetch_candles after requesting
# Asia/Dhaka explicitly. If parsing/localization fails, keep the warning visible.
try:
    if "df" in locals() and not df.empty and "datetime" in df.columns:
        latest_value = pd.to_datetime(df["datetime"].iloc[-1], errors="coerce")
        if pd.isna(latest_value):
            st.warning("Could not validate candle age: latest candle timestamp is invalid.")
        elif latest_value.tzinfo is None:
            st.warning(
                "Candle timestamp has no timezone information. Check Twelve Data "
                "metadata/timezone before relying on live signals."
            )
        else:
            now_value = pd.Timestamp.now(tz="UTC")
            latest_utc = latest_value.tz_convert("UTC")
            age_hours = (now_value - latest_utc).total_seconds() / 3600
            # Allow a small future tolerance for provider clock/candle-boundary differences.
            if age_hours < -1.0:
                st.error(
                    f"⚠️ Candle timestamp is {abs(age_hours):.1f} hours in the future "
                    f"({latest_utc.tz_convert('Asia/Dhaka')}). Check provider timezone metadata."
                )
            elif age_hours > 72:
                st.error(
                    f"⚠️ STALE DATA WARNING: latest candle is about {age_hours:.1f} "
                    f"hours old ({latest_utc}). Do not trade from this data."
                )
            elif age_hours > 2:
                st.warning(
                    f"Latest candle is {age_hours:.1f} hours old. Check market hours "
                    "and provider updates before relying on the signal."
                )
except Exception as exc:
    st.warning(f"Could not validate latest candle timestamp: {str(exc)[:120]}")
