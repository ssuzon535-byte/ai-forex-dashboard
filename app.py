import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
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

st.title("🥇 XAUUSD Gold Live Analysis")
st.caption(
    "XAUUSD only · 5M analysis · closed-candle signals · gap-aware backtest. "
    "Educational tool; historical results do not guarantee future performance."
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Market Settings")
    symbol = "XAU/USD"
    st.info("Market: XAUUSD Gold only")

    st.info("Analysis timeframe: 5M only")
    timeframe_label = "5M"
    interval = "5min"

    outputsize = st.select_slider(
        "Candles to load",
        options=[150, 200, 300, 400, 500, 750, 1000],
        value=1000,
        help="Loads up to 1,000 candles for a larger historical backtest sample. More candles improve sample size, not signal accuracy by themselves.",
    )

    signal_mode = st.radio(
        "Signal mode",
        ["Closed candle (stable)", "Live candle (provisional)"],
        index=0,
        help="Default uses the last completed 5M candle to avoid signals changing inside an unfinished candle.",
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
    st.subheader("Analysis timeframe")
    st.caption("শুধু 5M candle ব্যবহার হবে। 1H/15M বা MTF confirmation নেই।")
    mtf_enabled = False
    strict_mtf = False

    min_score = st.slider(
        "Minimum technical score",
        min_value=3,
        max_value=5,
        value=3,
        help="Minimum confluence required out of the available factors. Volume is omitted when the data provider does not supply it.",
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
            test_results.append(("5M-only timeframe", INTERVALS == {"5M": "5min"}))
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
    st.subheader("🔌 XAUUSD API Connection")
    if st.button("🧪 Test XAUUSD API", use_container_width=True):
        with st.spinner("Testing XAUUSD 5M data..."):
            api_results = test_twelve_data_api(
                api_key, {"XAUUSD Gold": "XAU/USD"}, "5min"
            )
        result = api_results["XAUUSD Gold"]
        if result["status"] == "OK":
            st.success(f"✅ XAUUSD: {result['message']}")
        elif result["status"] == "RATE LIMIT":
            st.warning(f"⏳ XAUUSD: {result['message']}")
        else:
            st.error(f"❌ XAUUSD: {result['message']}")


# -----------------------------
# Data
# -----------------------------
@st.cache_data(ttl=55, show_spinner=False)
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
    """Classify recent 5M structure without using future candles."""
    try:
        pos = df.index.get_loc(row.name)
        history = df.iloc[:pos]
        if len(history) < 10:
            return "BUILDING STRUCTURE"
        recent = history.tail(5)
        previous = history.iloc[-10:-5]
        rh, rl = float(recent["high"].max()), float(recent["low"].min())
        ph, pl = float(previous["high"].max()), float(previous["low"].min())
        close = float(row["close"])
        if close > ph:
            return "BULLISH BOS"
        if close < pl:
            return "BEARISH BOS"
        if rh > ph and rl > pl:
            return "BULLISH HH / HL"
        if rh < ph and rl < pl:
            return "BEARISH LH / LL"
        return "RANGE / STRUCTURE FORMING"
    except Exception:
        return "STRUCTURE UNKNOWN"


# -----------------------------
# Signal engine
# -----------------------------
def get_signal(row, df, min_score=3, experiment=None):
    """Five-factor XAUUSD 5M engine using confirmed candles and past-only levels."""
    empty = {"signal":"WAIT","score":0,"max_score":4,"technical_score_pct":0,
             "buy_score":0,"sell_score":0,"reasons":[],
             "regime":"STRUCTURE UNKNOWN","candle":"Unknown","quality_failures":[]}
    try:
        required = ["open", "high", "low", "close", "ATR"]
        if any(k not in row.index or pd.isna(row[k]) for k in required):
            return empty
        pos = df.index.get_loc(row.name)
        if not isinstance(pos, (int, np.integer)) or pos < 20:
            return empty

        history = df.iloc[:pos]
        recent5 = history.tail(5)
        prior5 = history.iloc[-10:-5]
        recent10 = history.tail(10)
        recent20 = history.tail(20)
        o, h, l, c = [float(row[k]) for k in ("open", "high", "low", "close")]
        atr = float(row["ATR"])
        if not np.isfinite(atr) or atr <= 0:
            return empty

        candle_range = max(h - l, 1e-12)
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        bullish_rejection = c > o and lower_wick >= max(body * 0.8, candle_range * 0.20)
        bearish_rejection = c < o and upper_wick >= max(body * 0.8, candle_range * 0.20)

        # Levels are calculated only from candles before the signal candle.
        resistance10 = float(recent10["high"].max())
        support10 = float(recent10["low"].min())
        resistance20 = float(recent20["high"].max())
        support20 = float(recent20["low"].min())
        previous_close = float(history.iloc[-1]["close"])

        buy_sweep = l < support10 and c > support10
        sell_sweep = h > resistance10 and c < resistance10
        buy_breakout = c > resistance20 and previous_close <= resistance20
        sell_breakout = c < support20 and previous_close >= support20
        # Retest must use the breakout level that existed BEFORE the
        # previous candle. Using resistance20/support20 above would include
        # the previous candle itself, making a close beyond that level impossible.
        retest_history = df.iloc[max(0, pos - 21):pos - 1]
        if len(retest_history) >= 10:
            retest_resistance = float(retest_history["high"].tail(20).max())
            retest_support = float(retest_history["low"].tail(20).min())
            buy_retest = (
                previous_close > retest_resistance
                and l <= retest_resistance + 0.25 * atr
                and c > retest_resistance
            )
            sell_retest = (
                previous_close < retest_support
                and h >= retest_support - 0.25 * atr
                and c < retest_support
            )
        else:
            buy_retest = False
            sell_retest = False

        # Short-term HH/HL and LH/LL structure, plus breakout/CHOCH confirmation.
        recent_high, recent_low = float(recent5["high"].max()), float(recent5["low"].min())
        prior_high, prior_low = float(prior5["high"].max()), float(prior5["low"].min())
        bullish_structure = recent_high > prior_high and recent_low > prior_low
        bearish_structure = recent_high < prior_high and recent_low < prior_low
        buy_structure = bullish_structure or buy_breakout or (buy_sweep and c > previous_close)
        sell_structure = bearish_structure or sell_breakout or (sell_sweep and c < previous_close)

        # Approximate demand/supply from recent swing extremes, with a proximity filter.
        demand_candle = recent20.loc[recent20["low"].idxmin()]
        supply_candle = recent20.loc[recent20["high"].idxmax()]
        demand_low = float(demand_candle["low"])
        demand_high = max(float(demand_candle["open"]), float(demand_candle["close"]))
        supply_low = min(float(supply_candle["open"]), float(supply_candle["close"]))
        supply_high = float(supply_candle["high"])
        buy_zone = l <= demand_high + 0.50 * atr and c >= demand_low and (c - demand_low) <= 1.25 * atr
        sell_zone = h >= supply_low - 0.50 * atr and c <= supply_high and (supply_high - c) <= 1.25 * atr

        # Tick volume is optional. If provider volume is absent, score out of four.
        volume_available = (
            "volume" in df.columns
            and history["volume"].notna().sum() >= 10
            and pd.notna(row.get("volume"))
        )
        volume_spike = False
        if volume_available:
            vol_hist = pd.to_numeric(history["volume"].tail(20), errors="coerce").dropna()
            avg_volume = float(vol_hist.mean()) if not vol_hist.empty else 0.0
            current_volume = float(row["volume"])
            volume_spike = avg_volume > 0 and current_volume >= 1.30 * avg_volume

        buy_price_action = bullish_rejection or buy_breakout or buy_retest
        sell_price_action = bearish_rejection or sell_breakout or sell_retest
        buy_checks = {
            "Price action (rejection / breakout / retest)": bool(buy_price_action),
            "Liquidity sweep": bool(buy_sweep),
            "Market structure (HH/HL or BOS/CHOCH)": bool(buy_structure),
            "Demand zone reaction": bool(buy_zone),
        }
        sell_checks = {
            "Price action (rejection / breakout / retest)": bool(sell_price_action),
            "Liquidity sweep": bool(sell_sweep),
            "Market structure (LH/LL or BOS/CHOCH)": bool(sell_structure),
            "Supply zone reaction": bool(sell_zone),
        }
        if volume_available:
            buy_checks["Volume spike + bullish candle"] = bool(volume_spike and c > o)
            sell_checks["Volume spike + bearish candle"] = bool(volume_spike and c < o)

        buy_score = sum(buy_checks.values())
        sell_score = sum(sell_checks.values())
        max_score = len(buy_checks)
        threshold = max(3, min(int(min_score), max_score))

        # A setup needs confluence plus at least one actual price-action/liquidity trigger.
        buy_trigger = buy_price_action or buy_sweep
        sell_trigger = sell_price_action or sell_sweep
        if buy_score >= threshold and buy_score > sell_score and buy_trigger:
            signal = "STRONG BUY" if buy_score >= max_score - 1 else "BUY"
        elif sell_score >= threshold and sell_score > buy_score and sell_trigger:
            signal = "STRONG SELL" if sell_score >= max_score - 1 else "SELL"
        else:
            signal = "WAIT"

        if signal.endswith("BUY"):
            selected = buy_checks
        elif signal.endswith("SELL"):
            selected = sell_checks
        else:
            selected = buy_checks if buy_score >= sell_score else sell_checks

        reasons = [name for name, passed in selected.items() if passed]
        if not volume_available:
            reasons.append("Volume unavailable from data provider; volume factor omitted")
        if buy_breakout or sell_breakout:
            reasons.append("Confirmed close beyond prior 20-candle level")
        if buy_retest or sell_retest:
            reasons.append("Breakout retest detected")
        candle = (
            "Bullish rejection" if bullish_rejection else
            "Bearish rejection" if bearish_rejection else
            "Strong bullish candle" if c > o and body / candle_range >= 0.55 else
            "Strong bearish candle" if c < o and body / candle_range >= 0.55 else
            "No clear rejection"
        )
        score = max(buy_score, sell_score)
        return {
            "signal": signal,
            "score": int(score),
            "max_score": int(max_score),
            "technical_score_pct": int(round(score / max_score * 100)) if max_score else 0,
            "buy_score": int(buy_score),
            "sell_score": int(sell_score),
            "reasons": reasons,
            "regime": market_regime(row, df),
            "candle": candle,
            "quality_failures": [name for name, passed in selected.items() if not passed],
        }
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return empty


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
                "factors": list(result.get("reasons", [])),
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
            "factors": list(result.get("reasons", [])),
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
# Live refresh
# -----------------------------
refresh_count = st_autorefresh(interval=60 * 1000, limit=None, key="xauusd_live_refresh")

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

live_now = pd.Timestamp.now(tz="Asia/Dhaka")
candle_open_time = pd.Timestamp(df.iloc[-1]["datetime"]).tz_convert("Asia/Dhaka")
next_candle_time = candle_open_time + pd.Timedelta(minutes=5)
seconds_to_close = max(0, int((next_candle_time - live_now).total_seconds()))
countdown_text = f"{seconds_to_close // 60:02d}:{seconds_to_close % 60:02d}"

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Live Price", f"{float(df.iloc[-1]['close']):.5f}")
col2.metric("Signal", final_signal)
col3.metric("Technical Score", f"{signal_info['score']}/{signal_info['max_score']}")
col4.metric("Confluence", f"{signal_info['technical_score_pct']}%")
col5.metric("Market Regime", regime)
col6.metric("Bangladesh Time", live_now.strftime("%H:%M:%S"))
col2.metric("Final Signal", final_signal)
col3.metric("Technical Score", f"{signal_info['score']}/{signal_info['max_score']}")
col4.metric("Confluence (not win probability)", f"{signal_info['technical_score_pct']}%")
col5.metric("Regime", regime)

st.caption(
    f"Current 5M candle: {candle_open_time.strftime('%Y-%m-%d %H:%M:%S')} BD time | "
    f"Next candle in: {countdown_text} | Auto-refresh: 60 sec | "
    f"Last analysis candle: {price_time_display} | Mode: {signal_mode}"
)
st.info("Live candle signal provisional. Candle close হওয়ার আগে BUY/SELL/WAIT বদলাতে পারে। Stable signal চাইলে sidebar থেকে Closed candle mode নাও।")

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
    st.subheader("🧠 Price Action Decision Factors")

    # Show the exact factors used by the current price-action signal engine.
    # The selected side is the generated signal, or the stronger side when WAIT.
    _signal_name = str(signal_info.get("signal", "WAIT"))
    _reasons = set(signal_info.get("reasons", []))
    _failed = set(signal_info.get("quality_failures", []))
    _selected_direction = (
        "BUY" if _signal_name.endswith("BUY")
        else "SELL" if _signal_name.endswith("SELL")
        else ("BUY" if signal_info.get("buy_score", 0) >= signal_info.get("sell_score", 0) else "SELL")
    )
    _factor_names = [
        "Price action (rejection / breakout / retest)",
        "Liquidity sweep",
        "Market structure (HH/HL or BOS/CHOCH)" if _selected_direction == "BUY"
            else "Market structure (LH/LL or BOS/CHOCH)",
        "Demand zone reaction" if _selected_direction == "BUY" else "Supply zone reaction",
    ]
    _factor_labels = [
        "Price Action",
        "Liquidity Sweep",
        "Market Structure (HH/HL or LH/LL)",
        "Demand Zone" if _selected_direction == "BUY" else "Supply Zone",
    ]
    _factor_rows = []
    for _label, _name in zip(_factor_labels, _factor_names):
        if _name in _reasons:
            _status = "PASS"
        elif _name in _failed:
            _status = "NO"
        else:
            _status = "N/A"
        _factor_rows.append({"Factor": _label, "Status": _status})

    _volume_note = any("Volume unavailable" in str(x) for x in _reasons)
    if _volume_note:
        _volume_status = "N/A (provider data unavailable)"
    else:
        _volume_buy = "Volume spike + bullish candle" in _reasons
        _volume_sell = "Volume spike + bearish candle" in _reasons
        _volume_fail_buy = "Volume spike + bullish candle" in _failed
        _volume_fail_sell = "Volume spike + bearish candle" in _failed
        _volume_status = (
            "PASS" if (_volume_buy or _volume_sell)
            else "NO" if (_volume_fail_buy or _volume_fail_sell)
            else "N/A"
        )
    _factor_rows.append({"Factor": "Volume Confirmation", "Status": _volume_status})
    _factor_rows.append({"Factor": "Market Regime", "Status": regime})
    _factor_rows.append({"Factor": "Candle", "Status": signal_info.get("candle", "Unknown")})
    _factor_rows.append({
        "Factor": "Selected side",
        "Status": f"{_selected_direction} · score {signal_info.get('buy_score', 0) if _selected_direction == 'BUY' else signal_info.get('sell_score', 0)}/{signal_info.get('max_score', 0)}",
    })
    st.table(pd.DataFrame(_factor_rows))

    if signal_info.get("signal") == "WAIT" and signal_info.get("quality_failures"):
        st.warning("WAIT কারণ: " + " · ".join(signal_info["quality_failures"]))
    elif signal_info.get("signal") == "WAIT":
        st.info("WAIT: Price action score বা direction-এর শর্ত পূরণ হয়নি.")

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
        for _threshold in (3, 4, 5):
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

    # Legacy RSI/ADX/EMA experiment rows were removed because the active
    # price-action engine does not implement those experiment parameters.
    st.caption("Only the active Price Action rules and score-threshold sensitivity are evaluated here. No inactive EMA/RSI/ADX experiment is presented as a valid test.")


    if not bt_trades.empty:
        stats_df = direction_stats(bt_trades)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Factor attribution: outcomes for trades whose selected signal carried
        # each factor. This is co-occurrence analysis, not proof of causation.
        st.markdown("**🔬 Price Action Factor Win/Loss Analysis**")
        st.caption("একটি ট্রেডে একাধিক ফ্যাক্টর থাকতে পারে। তাই একই ট্রেড একাধিক সারিতে গণনা হতে পারে। এই রিপোর্ট সম্পর্ক দেখায়, কোনো ফ্যাক্টর একা লাভের কারণ প্রমাণ করে না।")
        _closed_factor_trades = bt_trades[bt_trades["result"].isin(["WIN", "LOSS"])].copy()
        _factor_defs = [
            ("Price Action", "Price action (rejection / breakout / retest)"),
            ("Liquidity Sweep", "Liquidity sweep"),
            ("Market Structure", "Market structure"),
            ("Demand / Supply Zone", "zone reaction"),
            ("Volume Confirmation", "Volume spike +"),
        ]
        _factor_rows = []
        for _factor_label, _match in _factor_defs:
            _mask = _closed_factor_trades["factors"].apply(
                lambda _items: any(_match.lower() in str(_item).lower() for _item in (_items if isinstance(_items, list) else []))
            ) if "factors" in _closed_factor_trades.columns else pd.Series(False, index=_closed_factor_trades.index)
            _part = _closed_factor_trades[_mask]
            _n = len(_part)
            _w = int((_part["result"] == "WIN").sum())
            _l = int((_part["result"] == "LOSS").sum())
            _gross_win = float(_part.loc[_part["gross_R"] > 0, "gross_R"].sum()) if _n else 0.0
            _gross_loss = abs(float(_part.loc[_part["gross_R"] < 0, "gross_R"].sum())) if _n else 0.0
            _pf = _gross_win / _gross_loss if _gross_loss > 0 else (np.inf if _gross_win > 0 else 0.0)
            _factor_rows.append({
                "Factor": _factor_label,
                "Trades": _n,
                "Wins": _w,
                "Losses": _l,
                "Win Rate %": round(_w / _n * 100, 1) if _n else 0.0,
                "Net R": round(float(_part["R"].sum()), 2) if _n else 0.0,
                "Profit Factor": "∞" if np.isinf(_pf) else round(_pf, 2),
            })
        st.dataframe(pd.DataFrame(_factor_rows), use_container_width=True, hide_index=True)
        st.caption("Only closed trades are counted. A factor is credited when it appeared in the selected signal's reason list; factors are correlated and overlapping.")

        # Score-wise report: closed outcomes only; OPEN trades are counted separately.
        st.markdown("**📊 Score-wise Backtest Report**")
        _score_rows = []
        for _score_value in range(1, 6):
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
# Chart
# -----------------------------
st.subheader("📉 Live XAUUSD 5M Candlestick Chart")

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

# Legacy EMA / RSI / ADX / MACD / Bollinger values intentionally hidden.
# The active dashboard strategy is Price Action, Liquidity, Structure,
# Supply/Demand and Volume confirmation.

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
