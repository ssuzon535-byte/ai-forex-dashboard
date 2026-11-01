import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Forex Multi-Market Dashboard", page_icon="📈", layout="wide")
st.title("📈 Forex Multi-Market Dashboard")
st.caption("Rule-based technical analysis for supported Forex pairs and metals. Signals are not guaranteed outcomes.")

PAIRS = {
    "EUR/USD": "EUR/USD", "GBP/USD": "GBP/USD", "USD/JPY": "USD/JPY",
    "AUD/USD": "AUD/USD", "USD/CAD": "USD/CAD", "NZD/USD": "NZD/USD",
    "USD/CHF": "USD/CHF", "EUR/GBP": "EUR/GBP", "EUR/JPY": "EUR/JPY",
    "GBP/JPY": "GBP/JPY", "EUR/AUD": "EUR/AUD", "GBP/AUD": "GBP/AUD",
    "XAU/USD Gold": "XAU/USD", "XAG/USD Silver": "XAG/USD"
}
TFS = {"5 minutes": "5min", "15 minutes": "15min", "1 hour": "1h", "4 hours": "4h", "Daily": "1day"}
MTF = ["Daily", "4 hours", "1 hour", "15 minutes", "5 minutes"]
DB = "forex_journal.db"


def api_key_saved():
    try:
        return st.secrets.get("TWELVE_DATA_API_KEY", "") or os.getenv("TWELVE_DATA_API_KEY", "")
    except Exception:
        return os.getenv("TWELVE_DATA_API_KEY", "")


@st.cache_data(ttl=60, show_spinner=False)
def candles(symbol, interval, count, key):
    if not key:
        raise ValueError("Twelve Data API key দিন। Streamlit Secrets-এ TWELVE_DATA_API_KEY-ও সেট করতে পারেন।")
    r = requests.get("https://api.twelvedata.com/time_series", params={
        "symbol": symbol, "interval": interval, "outputsize": count,
        "apikey": key, "format": "JSON", "timezone": "UTC"
    }, timeout=25)
    r.raise_for_status()
    j = r.json()
    if "values" not in j:
        raise ValueError(j.get("message", "Market data পাওয়া যায়নি।"))
    d = pd.DataFrame(j["values"]).rename(columns={
        "datetime": "datetime", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume"
    })
    for c in ["Open", "High", "Low", "Close"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce", utc=True)
    d = d.dropna(subset=["datetime", "Open", "High", "Low", "Close"])
    d = d.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    if len(d) < 60:
        raise ValueError(f"যথেষ্ট candle নেই: {len(d)}টি পাওয়া গেছে।")
    return d


def indicators(df):
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    for n in [20, 50, 200]:
        d[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["RSI"] = (100 - 100/(1 + gain/loss.replace(0, np.nan))).fillna(50)
    d["MACD"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["MACDSignal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    plus = h.diff().where((h.diff() > -l.diff()) & (h.diff() > 0), 0)
    minus = (-l.diff()).where((-l.diff() > h.diff()) & (-l.diff() > 0), 0)
    pdi = 100 * plus.ewm(alpha=1/14, adjust=False).mean()/d["ATR"].replace(0, np.nan)
    mdi = 100 * minus.ewm(alpha=1/14, adjust=False).mean()/d["ATR"].replace(0, np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    d["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0)
    d["BBMid"] = c.rolling(20).mean()
    sd = c.rolling(20).std()
    d["BBUpper"], d["BBLower"] = d["BBMid"] + 2*sd, d["BBMid"] - 2*sd
    return d


def trend(df):
    r = df.iloc[-1]
    if r["EMA20"] > r["EMA50"] and r["Close"] > r["EMA50"]:
        return "BULLISH", 1
    if r["EMA20"] < r["EMA50"] and r["Close"] < r["EMA50"]:
        return "BEARISH", -1
    return "NEUTRAL", 0


def analyse(frames):
    dirs = {tf: trend(frames[tf])[1] if tf in frames and not frames[tf].empty else 0 for tf in MTF}
    d = frames.get("5 minutes")
    if d is None or d.empty:
        return {"signal": "WAIT", "score": 0, "directions": dirs, "reason": "5M data unavailable"}
    r = d.iloc[-1]
    bulls = sum(dirs[x] == 1 for x in MTF[:4])
    bears = sum(dirs[x] == -1 for x in MTF[:4])
    buy_ok = r["Close"] > r["EMA20"] and r["RSI"] >= 52 and r["MACD"] > r["MACDSignal"]
    sell_ok = r["Close"] < r["EMA20"] and r["RSI"] <= 48 and r["MACD"] < r["MACDSignal"]
    bs = min(100, bulls*20 + int(buy_ok)*10 + int(r["ADX"] >= 18)*10)
    ss = min(100, bears*20 + int(sell_ok)*10 + int(r["ADX"] >= 18)*10)
    if bulls == 4 and buy_ok:
        sig, score, reason = ("STRONG BUY" if bs >= 90 else "BUY"), bs, "Higher timeframes bullish and 5M confirmation present."
    elif bulls >= 3 and buy_ok:
        sig, score, reason = "BUY", bs, "Most higher timeframes bullish with 5M confirmation."
    elif bears == 4 and sell_ok:
        sig, score, reason = ("STRONG SELL" if ss >= 90 else "SELL"), ss, "Higher timeframes bearish and 5M confirmation present."
    elif bears >= 3 and sell_ok:
        sig, score, reason = "SELL", ss, "Most higher timeframes bearish with 5M confirmation."
    else:
        sig, score, reason = "WAIT", max(bs, ss), "Timeframe alignment or 5M confirmation is incomplete."
    return {"signal": sig, "score": score, "directions": dirs, "reason": reason}


def levels(df, sig, rr):
    r = df.iloc[-1]
    entry, atr = float(r["Close"]), float(r["ATR"])
    if sig in ["BUY", "STRONG BUY"]:
        sl = min(float(df.tail(10)["Low"].min()), entry - 1.2*atr)
        side = 1
    elif sig in ["SELL", "STRONG SELL"]:
        sl = max(float(df.tail(10)["High"].max()), entry + 1.2*atr)
        side = -1
    else:
        return {"Entry": entry, "SL": np.nan, "TP1": np.nan, "TP2": np.nan, "TP3": np.nan, "side": 0}
    risk = abs(entry-sl)
    if risk <= 0 or not np.isfinite(risk):
        risk = max(atr, abs(entry)*0.001)
        sl = entry-side*risk
    return {"Entry": entry, "SL": sl, "TP1": entry+side*risk*rr[0],
            "TP2": entry+side*risk*rr[1], "TP3": entry+side*risk*rr[2], "side": side}


def chart(df, lev, sig):
    d = df.tail(250)
    f = go.Figure(data=[go.Candlestick(x=d["datetime"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price")])
    for name in ["EMA20", "EMA50", "EMA200"]:
        f.add_trace(go.Scatter(x=d["datetime"], y=d[name], mode="lines", name=name))
    if lev["side"]:
        buy = lev["side"] == 1
        f.add_trace(go.Scatter(x=[d["datetime"].iloc[-1]], y=[d["Low"].iloc[-1] if buy else d["High"].iloc[-1]],
            mode="markers+text", text=["▲ BUY" if buy else "▼ SELL"], textposition="bottom center" if buy else "top center",
            marker={"size": 16, "symbol": "triangle-up" if buy else "triangle-down"}, name="Signal"))
        for key, color in [("Entry", "white"), ("SL", "red"), ("TP1", "green"), ("TP2", "green"), ("TP3", "green")]:
            f.add_hline(y=lev[key], line_dash="dot", line_color=color, annotation_text=f"{key}: {lev[key]:.5f}")
        f.add_hrect(y0=min(lev["SL"], lev["TP1"]), y1=max(lev["SL"], lev["TP1"]),
                    fillcolor="green" if buy else "red", opacity=0.08, line_width=0)
        f.add_annotation(x=d["datetime"].iloc[-1], y=(lev["SL"]+lev["TP1"])/2,
                         text="➤ BUY" if buy else "SELL ➤", showarrow=False)
    f.update_layout(template="plotly_dark", height=620, xaxis_rangeslider_visible=False)
    return f


def db_init():
    with sqlite3.connect(DB) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, symbol TEXT, signal TEXT,
            entry REAL, sl REAL, tp1 REAL, tp2 REAL, tp3 REAL, notes TEXT)""")


def journal_read():
    db_init()
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query("SELECT * FROM journal ORDER BY id DESC", con)


def journal_save(symbol, sig, lev, notes):
    db_init()
    with sqlite3.connect(DB) as con:
        con.execute("INSERT INTO journal (created_at,symbol,signal,entry,sl,tp1,tp2,tp3,notes) VALUES (?,?,?,?,?,?,?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), symbol, sig, lev["Entry"],
                     lev["SL"], lev["TP1"], lev["TP2"], lev["TP3"], notes))


def backtest(df):
    d = df.dropna(subset=["EMA20", "EMA50", "ATR"]).reset_index(drop=True)
    trades, i = [], 2
    while i < len(d)-1:
        r = d.iloc[i]
        side = 1 if r["EMA20"] > r["EMA50"] and r["RSI"] > 52 else (-1 if r["EMA20"] < r["EMA50"] and r["RSI"] < 48 else 0)
        if not side or r["ATR"] <= 0:
            i += 1
            continue
        entry, risk = float(r["Close"]), max(float(r["ATR"])*1.5, abs(float(r["Close"]))*0.0005)
        sl, tp = entry-side*risk, entry+side*risk*2
        exit_i, exit_price = min(i+48, len(d)-1), float(d.iloc[min(i+48, len(d)-1)]["Close"])
        result = "TIME EXIT"
        for j in range(i+1, min(i+49, len(d))):
            c = d.iloc[j]
            if (side == 1 and c["Low"] <= sl) or (side == -1 and c["High"] >= sl):
                exit_i, exit_price, result = j, sl, "LOSS"
                break
            if (side == 1 and c["High"] >= tp) or (side == -1 and c["Low"] <= tp):
                exit_i, exit_price, result = j, tp, "WIN"
                break
        pnl_r = (exit_price-entry)*side/risk
        trades.append({"Entry Time": d.iloc[i]["datetime"], "Exit Time": d.iloc[exit_i]["datetime"],
                       "Side": "BUY" if side == 1 else "SELL", "Entry": entry, "Exit": exit_price,
                       "Result": result, "R Multiple": pnl_r, "Bars Held": exit_i-i})
        i = exit_i+1
    t = pd.DataFrame(trades)
    if t.empty:
        return t, {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "drawdown_r": 0.0}
    wins = t.loc[t["R Multiple"] > 0, "R Multiple"].sum()
    losses = abs(t.loc[t["R Multiple"] < 0, "R Multiple"].sum())
    eq = t["R Multiple"].cumsum()
    dd = eq - eq.cummax()
    return t, {"trades": len(t), "win_rate": (t["R Multiple"] > 0).mean()*100,
               "profit_factor": wins/losses if losses else (float("inf") if wins else 0.0),
               "drawdown_r": abs(float(dd.min()))}


with st.sidebar:
    st.header("Market Settings")
    name = st.selectbox("Forex pair / market", list(PAIRS.keys()))
    symbol = PAIRS[name]
    custom = st.text_input("Custom Twelve Data symbol", "")
    if custom.strip():
        symbol = custom.strip().upper()
    tf_name = st.selectbox("Chart timeframe", list(TFS.keys()), index=0)
    count = st.slider("Candles", 100, 5000, 500, 100)
    balance = st.number_input("Account balance", min_value=1.0, value=1000.0, step=100.0)
    risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
    rr1 = st.number_input("TP1 R:R", 0.5, 10.0, 1.0, 0.5)
    rr2 = st.number_input("TP2 R:R", 0.5, 10.0, 2.0, 0.5)
    rr3 = st.number_input("TP3 R:R", 0.5, 15.0, 3.0, 0.5)
    key = st.text_input("Twelve Data API key", type="password") or api_key_saved()
    if st.button("Refresh data"):
        st.cache_data.clear()

if not key:
    st.info('Sidebar-এ Twelve Data API key দিন। Streamlit Secrets-এ `TWELVE_DATA_API_KEY = "your_key"` ব্যবহার করতে পারেন।')
    st.stop()

frames, errors = {}, []
with st.spinner("Fetching market candles..."):
    for tf in MTF:
        try:
            frames[tf] = indicators(candles(symbol, TFS[tf], count, key))
        except Exception as e:
            errors.append(f"{tf}: {e}")
if errors:
    with st.expander("Data/API errors", expanded=True):
        for e in errors:
            st.warning(e)
if "5 minutes" not in frames:
    st.error("5-minute data পাওয়া যায়নি। API key, pair symbol বা provider limits পরীক্ষা করুন।")
    st.stop()

a = analyse(frames)
df = frames.get(tf_name, frames["5 minutes"])
lev = levels(df, a["signal"], [rr1, rr2, rr3])
cols = st.columns(5)
cols[0].metric("Market", symbol)
cols[1].metric("Signal", a["signal"])
cols[2].metric("Quality score", f"{a['score']}/100")
cols[3].metric("Latest close", f"{df['Close'].iloc[-1]:.5f}")
cols[4].metric("5M RSI", f"{frames['5 minutes']['RSI'].iloc[-1]:.1f}")
st.caption("Quality score is a rule-based alignment score, not a win probability.")
st.info(a["reason"])

tabs = st.tabs(["Chart & Signal", "Multi-Timeframe", "Backtest", "Risk Calculator", "Trade Journal", "Telegram Alerts", "CSV Export"])
with tabs[0]:
    st.plotly_chart(chart(df, lev, a["signal"]), use_container_width=True)
    st.dataframe(pd.DataFrame([{"Level": k, "Price": v} for k, v in lev.items() if k != "side"]), use_container_width=True, hide_index=True)
with tabs[1]:
    rows = []
    for tf in MTF:
        if tf in frames:
            r = frames[tf].iloc[-1]
            rows.append({"Timeframe": tf, "Trend": trend(frames[tf])[0], "Close": r["Close"], "RSI": r["RSI"], "ADX": r["ADX"]})
        else:
            rows.append({"Timeframe": tf, "Trend": "DATA UNAVAILABLE", "Close": np.nan, "RSI": np.nan, "ADX": np.nan})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    r = df.iloc[-1]
    st.write("Indicators", {k: round(float(r[k]), 5) for k in ["EMA20", "EMA50", "EMA200", "MACD", "MACDSignal", "ADX", "ATR"]})
    st.write("Approximate support:", round(float(df.tail(50)["Low"].min()), 5))
    st.write("Approximate resistance:", round(float(df.tail(50)["High"].max()), 5))
    st.caption("Supply/demand, BOS/CHOCH and liquidity need additional structure rules and should be treated as approximations, not order-flow data.")
with tabs[2]:
    st.caption("Simplified EMA/RSI backtest on the selected chart timeframe. Spread, commission and slippage are not included.")
    if st.button("Run backtest"):
        bt, stats = backtest(df)
        st.session_state["bt"] = bt
        st.session_state["bt_stats"] = stats
    if "bt_stats" in st.session_state:
        s = st.session_state["bt_stats"]
        c = st.columns(4)
        c[0].metric("Trades", s["trades"])
        c[1].metric("Win rate", f'{s["win_rate"]:.1f}%')
        c[2].metric("Profit factor", "∞" if np.isinf(s["profit_factor"]) else f'{s["profit_factor"]:.2f}')
        c[3].metric("Max drawdown", f'{s["drawdown_r"]:.2f}R')
        st.dataframe(st.session_state["bt"], use_container_width=True, hide_index=True)
with tabs[3]:
    dist = abs(lev["Entry"]-lev["SL"]) if pd.notna(lev["SL"]) else float(df["ATR"].iloc[-1])
    risk_cash = balance*risk_pct/100
    st.metric("Money risk budget", f"{risk_cash:.2f}")
    st.metric("Risk budget / price distance", f"{risk_cash/max(dist, 1e-9):.4f} units")
    st.caption("Simplified estimate only. Correct lot size requires broker contract size, pip value and account currency.")
with tabs[4]:
    notes = st.text_input("Trade notes")
    if st.button("Save signal to journal"):
        journal_save(symbol, a["signal"], lev, notes)
        st.success("Saved.")
    j = journal_read()
    if not j.empty:
        st.dataframe(j, use_container_width=True, hide_index=True)
        delete_id = st.number_input("Journal ID to delete", min_value=0, value=0, step=1)
        if st.button("Delete journal row") and delete_id:
            with sqlite3.connect(DB) as con:
                con.execute("DELETE FROM journal WHERE id=?", (int(delete_id),))
            st.rerun()
with tabs[5]:
    token = st.text_input("Telegram bot token", type="password")
    chat = st.text_input("Telegram chat ID")
    msg = f"{symbol} {a['signal']} | score {a['score']}/100 | Entry {lev['Entry']:.5f} | SL {lev['SL']:.5f} | TP1 {lev['TP1']:.5f} | TP2 {lev['TP2']:.5f} | TP3 {lev['TP3']:.5f}"
    if st.button("Send Telegram alert"):
        if not token or not chat:
            st.error("Bot token ও chat ID দিন।")
        else:
            try:
                res = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={"chat_id": chat, "text": msg}, timeout=15)
                data = res.json()
                st.success("Telegram message sent.") if res.ok and data.get("ok") else st.error(data.get("description", "Telegram send failed."))
            except Exception as e:
                st.error(str(e))
with tabs[6]:
    st.download_button("Download candles CSV", df.to_csv(index=False).encode("utf-8"), f"{symbol.replace('/', '_')}_{TFS[tf_name]}.csv", "text/csv")
    st.download_button("Download journal CSV", journal_read().to_csv(index=False).encode("utf-8"), "forex_journal.csv", "text/csv")
    if "bt" in st.session_state and not st.session_state["bt"].empty:
        st.download_button("Download backtest CSV", st.session_state["bt"].to_csv(index=False).encode("utf-8"), "forex_backtest.csv", "text/csv")

with st.expander("Data source and limitations"):
    st.write("Candles are supplied by Twelve Data. Check API limits and symbol coverage. Forex volume may be unavailable or tick-based, so VWAP is not fabricated. This dashboard is for research and does not guarantee profitable signals.")
