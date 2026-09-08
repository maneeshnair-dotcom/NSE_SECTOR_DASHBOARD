# -*- coding: utf-8 -*-
"""
Agentic Nifty Sector Opportunity Dashboard
────────────────────────────────────────────────────────────────────────────
A standalone Streamlit app that:
  1. Knows the constituent stocks of 17 Nifty sectoral/thematic indices.
  2. Pulls live OHLCV data for every stock in every sector via yfinance.
  3. Computes a technical fingerprint for each stock (trend, momentum,
     volume, relative strength vs its own sector index, proximity to
     52-week high/low).
  4. Runs an autonomous "Agent" that scores every stock, decides a verdict
     (Strong Opportunity / Watch / Avoid), writes out its reasoning in
     plain English, and independently ranks the 17 sectors by aggregate
     opportunity score (sector rotation view) — nobody tells it which
     stocks or sectors to prefer; it decides that itself, every run.

Run locally:   streamlit run agentic_sector_dashboard.py
Requires:      pip install streamlit yfinance pandas numpy
────────────────────────────────────────────────────────────────────────────
"""

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ═════════════════════════════════════════════════════════════════════════
# 1. SECTOR UNIVERSE
#    Snapshot of constituents as of the most recent semi-annual NSE
#    rebalance known at build time. Sector indices are reviewed by NSE
#    every 6 months, so this list can drift — use the "🔄 Refresh from
#    NSE live" toggle in the sidebar to attempt a live pull; it falls
#    back to this snapshot automatically if NSE blocks the request
#    (common when running on cloud hosts).
# ═════════════════════════════════════════════════════════════════════════

SECTOR_STOCKS = {
    "Nifty IT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
        "LTIM", "PERSISTENT", "COFORGE", "MPHASIS", "OFSS",
    ],
    "Nifty Auto": [
        "MARUTI", "M&M", "TMPV", "BAJAJ-AUTO", "EICHERMOT",
        "HEROMOTOCO", "TVSMOTOR", "ASHOKLEY", "BHARATFORG", "MOTHERSON",
        "TIINDIA", "UNOMINDA", "SONACOMS", "EXIDEIND", "BALKRISIND",
    ],
    "Nifty Pharma": [
        "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "TORNTPHARM",
        "LUPIN", "AUROPHARMA", "ZYDUSLIFE", "ALKEM", "MANKIND",
        "GLENMARK", "BIOCON", "ABBOTINDIA", "IPCALAB", "LAURUSLABS",
        "GLAND", "NATCOPHARM", "AJANTPHARM", "PFIZER", "SANOFI",
    ],
    "Nifty FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM",
        "VBL", "DABUR", "GODREJCP", "MARICO", "COLPAL",
        "UBL", "EMAMILTD", "PATANJALI", "RADICO", "GILLETTE",
    ],
    "Nifty Metal": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "JINDALSTEL",
        "SAIL", "NMDC", "HINDZINC", "NATIONALUM", "APLAPOLLO",
        "RATNAMANI", "HINDCOPPER", "WELCORP", "JSL", "LLOYDSME",
    ],
    "Nifty Realty": [
        "DLF", "GODREJPROP", "OBEROIRLTY", "PHOENIXLTD", "PRESTIGE",
        "LODHA", "BRIGADE", "SOBHA", "MAHLIFE", "SUNTECK",
    ],
    "Nifty Energy": [
        "RELIANCE", "NTPC", "POWERGRID", "ONGC", "COALINDIA",
        "BPCL", "IOC", "GAIL", "ADANIGREEN", "TATAPOWER",
    ],
    "Nifty Media": [
        "ZEEL", "SUNTV", "PVRINOX", "NETWORK18", "TV18BRDCST",
        "SAREGAMA", "NAZARA", "DISHTV", "HATHWAY", "TIPSMUSIC",
    ],
    "Nifty PSU Bank": [
        "SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK",
        "INDIANB", "BANKINDIA", "CENTRALBK", "IOB", "UCOBANK",
        "MAHABANK", "PSB",
    ],
    "Nifty Infra": [
        "LT", "ADANIPORTS", "ULTRACEMCO", "POWERGRID", "NTPC",
        "RELIANCE", "BHARTIARTL", "GRASIM", "SHREECEM", "SIEMENS",
        "ABB", "GMRAIRPORT", "IRB", "NBCC", "CONCOR",
        "GAIL", "PETRONET", "JSWENERGY", "TATAPOWER", "ADANIENT",
    ],
    "Nifty Consumption": [
        "HINDUNILVR", "ITC", "TITAN", "ASIANPAINT", "MARUTI",
        "BAJAJ-AUTO", "NESTLEIND", "TATACONSUM", "TRENT", "DMART",
        "BRITANNIA", "PAGEIND", "VBL", "GODREJCP", "MARICO",
        "INDIGO", "JUBLFOOD", "HAVELLS", "VOLTAS", "DABUR",
    ],
    "Nifty Financial Services": [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
        "BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "SHRIRAMFIN",
        "CHOLAFIN", "PFC", "RECLTD", "ICICIGI", "ICICIPRULI",
        "SBICARD", "MUTHOOTFIN", "LICHSGFIN", "JIOFIN", "INDUSINDBK",
    ],
    "Nifty MNC": [
        "NESTLEIND", "HINDUNILVR", "MARUTI", "SIEMENS", "ABB",
        "BOSCHLTD", "CIPLA", "GLAXO", "PGHH", "COLPAL",
        "UBL", "HONAUT", "WHIRLPOOL", "3MINDIA", "OFSS",
        "BATAINDIA", "SCHAEFFLER", "LINDEINDIA", "GILLETTE", "TIMKEN",
    ],
    "Nifty Commodities": [
        "RELIANCE", "ONGC", "COALINDIA", "TATASTEEL", "JSWSTEEL",
        "HINDALCO", "VEDL", "UPL", "GRASIM", "ULTRACEMCO",
        "SHREECEM", "JINDALSTEL", "SAIL", "NTPC", "POWERGRID",
        "GAIL", "IOC", "BPCL", "PIIND", "SRF",
    ],
    "Nifty Services Sector": [
        "HDFCBANK", "ICICIBANK", "BHARTIARTL", "SBIN", "KOTAKBANK",
        "AXISBANK", "LT", "BAJFINANCE", "TCS", "INFY",
        "HCLTECH", "INDIGO", "IRCTC", "ETERNAL", "ADANIPORTS",
        "DMART", "TITAN", "HDFCLIFE", "SBILIFE", "BAJAJFINSV",
    ],
    "Nifty Midcap Select": [
        "PERSISTENT", "SUPREMEIND", "VOLTAS", "MAXHEALTH", "CUMMINSIND",
        "POLYCAB", "INDHOTEL", "PAGEIND", "GODREJPROP", "LUPIN",
        "FEDERALBNK", "MFSL", "COFORGE", "OBEROIRLTY", "IDFCFIRSTB",
        "GMRAIRPORT", "PRESTIGE", "AUBANK", "PIIND", "SOLARINDS",
        "YESBANK", "BANKBARODA", "HDFCAMC", "LTF", "NMDC",
    ],
    "Nifty PSE": [
        "ONGC", "NTPC", "POWERGRID", "COALINDIA", "BEL",
        "BPCL", "GAIL", "IOC", "SAIL", "NBCC",
        "NMDC", "HAL", "BHEL", "CONCOR", "IRCTC",
        "RVNL", "NHPC", "SJVN", "OIL", "RECLTD",
    ],
}

# Yahoo Finance ticker for each sector's own benchmark index, used to
# measure whether a stock is beating or lagging its own sector.
SECTOR_INDEX_TICKER = {
    "Nifty IT": "^CNXIT", "Nifty Auto": "^CNXAUTO", "Nifty Pharma": "^CNXPHARMA",
    "Nifty FMCG": "^CNXFMCG", "Nifty Metal": "^CNXMETAL", "Nifty Realty": "^CNXREALTY",
    "Nifty Energy": "^CNXENERGY", "Nifty Media": "^CNXMEDIA", "Nifty PSU Bank": "^CNXPSUBANK",
    "Nifty Infra": "^CNXINFRA", "Nifty Consumption": "^CNXCONSUM", "Nifty Financial Services": "^CNXFIN",
    "Nifty MNC": "^CNXMNC", "Nifty Commodities": "^CNXCMDT", "Nifty Services Sector": "^CNXSERVICE",
    "Nifty Midcap Select": "NIFTY_MID_SELECT.NS", "Nifty PSE": "^CNXPSE",
}

# NSE's own internal index name, used only for the optional live-refresh call.
NSE_LIVE_INDEX_NAME = {
    "Nifty IT": "NIFTY IT", "Nifty Auto": "NIFTY AUTO", "Nifty Pharma": "NIFTY PHARMA",
    "Nifty FMCG": "NIFTY FMCG", "Nifty Metal": "NIFTY METAL", "Nifty Realty": "NIFTY REALTY",
    "Nifty Energy": "NIFTY ENERGY", "Nifty Media": "NIFTY MEDIA", "Nifty PSU Bank": "NIFTY PSU BANK",
    "Nifty Infra": "NIFTY INFRA", "Nifty Consumption": "NIFTY INDIA CONSUMPTION",
    "Nifty Financial Services": "NIFTY FINANCIAL SERVICES", "Nifty MNC": "NIFTY MNC",
    "Nifty Commodities": "NIFTY COMMODITIES", "Nifty Services Sector": "NIFTY SERV SECTOR",
    "Nifty Midcap Select": "NIFTY MIDCAP SELECT", "Nifty PSE": "NIFTY PSE",
}


def fetch_live_constituents(sector: str) -> list | None:
    """Best-effort live pull of a sector's current constituents straight
    from NSE. Returns None (silently) on any failure so the caller can
    fall back to the built-in snapshot — NSE aggressively blocks
    scripted/cloud requests, so this is a bonus, not a dependency."""
    try:
        index_name = NSE_LIVE_INDEX_NAME[sector]
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "application/json",
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=6)
        resp = session.get(
            "https://www.nseindia.com/api/equity-stockIndices",
            params={"index": index_name}, headers=headers, timeout=6,
        )
        data = resp.json()
        syms = [row["symbol"] for row in data["data"] if row["symbol"] != index_name]
        return syms if syms else None
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════
# 2. DATA FETCH + TECHNICAL INDICATORS
# ═════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def download_history(tickers: tuple, period: str = "1y") -> dict:
    """Single batched yfinance call for every ticker; returns
    {ticker: DataFrame} so a bad/delisted ticker can't break the rest."""
    def to_yf(t: str) -> str:
        if t.startswith("^") or t.endswith(".NS"):
            return t
        return t + ".NS"

    yf_tickers = [to_yf(t) for t in tickers]
    raw = yf.download(yf_tickers, period=period, interval="1d",
                       group_by="ticker", threads=True, progress=False, auto_adjust=True)
    out = {}
    for orig, yft in zip(tickers, yf_tickers):
        try:
            df = raw[yft].dropna(how="all") if len(yf_tickers) > 1 else raw.dropna(how="all")
            if df is not None and not df.empty and "Close" in df.columns:
                out[orig] = df
        except Exception:
            continue
    return out


def compute_indicators(hist: pd.DataFrame) -> dict | None:
    """Compute the technical fingerprint the Agent reasons over."""
    if hist is None or len(hist) < 30 or "Close" not in hist:
        return None
    close, vol = hist["Close"], hist["Volume"]

    sma20, sma50 = close.rolling(20).mean(), close.rolling(50).mean()
    sma200 = close.rolling(200).mean() if len(close) >= 200 else pd.Series([np.nan] * len(close))

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()

    vol_avg20 = vol.rolling(20).mean()
    vol_ratio = float(vol.iloc[-1] / vol_avg20.iloc[-1]) if vol_avg20.iloc[-1] else np.nan

    def ret(n):
        return float(close.iloc[-1] / close.iloc[-n] - 1) if len(close) > n else np.nan

    lookback = min(len(close), 252)
    high52, low52 = close.iloc[-lookback:].max(), close.iloc[-lookback:].min()

    return {
        "close": float(close.iloc[-1]),
        "sma20": float(sma20.iloc[-1]), "sma50": float(sma50.iloc[-1]),
        "sma200": float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else None,
        "rsi": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0,
        "macd": float(macd.iloc[-1]), "macd_signal": float(macd_signal.iloc[-1]),
        "vol_ratio": vol_ratio if not np.isnan(vol_ratio) else 1.0,
        "ret_1w": ret(5), "ret_1m": ret(21), "ret_3m": ret(63), "ret_6m": ret(126),
        "pct_from_high": float(close.iloc[-1] / high52 - 1),
        "pct_from_low": float(close.iloc[-1] / low52 - 1),
    }


# ═════════════════════════════════════════════════════════════════════════
# 3. THE AGENT — autonomous scoring + plain-English reasoning
# ═════════════════════════════════════════════════════════════════════════

def agent_evaluate(ind: dict, sector_ret_1m: float) -> tuple:
    """Given one stock's indicators and its sector's 1-month return,
    the Agent independently scores it, assigns a verdict, and writes
    out WHY — this is the decision-making core; nothing here is fed a
    preset answer, it is derived fresh from the numbers every run."""
    score, reasons = 0.0, []

    # ── Trend structure ──
    if ind["sma200"] is not None:
        if ind["close"] > ind["sma50"] > ind["sma200"]:
            score += 2.0
            reasons.append("Price is above both the 50- and 200-day average — an established uptrend.")
        elif ind["close"] > ind["sma200"]:
            score += 1.0
            reasons.append("Price sits above its 200-day average — long-term trend still up.")
        else:
            score -= 1.0
            reasons.append("Price is below its 200-day average — long-term trend is down.")
    elif ind["close"] > ind["sma50"]:
        score += 0.5

    # ── Momentum (RSI) ──
    if 45 <= ind["rsi"] <= 65:
        score += 1.0
        reasons.append(f"RSI at {ind['rsi']:.0f} shows healthy momentum without being overbought.")
    elif ind["rsi"] > 70:
        score -= 1.0
        reasons.append(f"RSI at {ind['rsi']:.0f} is overbought — near-term pullback risk.")
    elif ind["rsi"] < 30:
        score += 0.5
        reasons.append(f"RSI at {ind['rsi']:.0f} is oversold — a potential reversal setup.")

    # ── MACD ──
    if ind["macd"] > ind["macd_signal"]:
        score += 1.0
        reasons.append("MACD is above its signal line — bullish crossover in force.")
    else:
        score -= 0.5

    # ── Volume confirmation ──
    if ind["vol_ratio"] > 1.5:
        score += 1.0
        reasons.append(f"Volume is running {ind['vol_ratio']:.1f}x its 20-day average — real participation behind the move.")

    # ── Relative strength vs its own sector ──
    if not np.isnan(sector_ret_1m) and not np.isnan(ind["ret_1m"]):
        if ind["ret_1m"] > sector_ret_1m:
            score += 1.0
            reasons.append("Outperforming its own sector index over the past month.")
        else:
            score -= 0.3

    # ── Breakout proximity ──
    if -0.03 <= ind["pct_from_high"] <= 0:
        score += 1.0
        reasons.append("Trading within 3% of its 52-week high — breakout zone.")
    elif ind["pct_from_low"] <= 0.05:
        score -= 0.5
        reasons.append("Trading near its 52-week low — momentum is weak.")

    # ── 3-month trend confirmation ──
    if not np.isnan(ind["ret_3m"]) and ind["ret_3m"] > 0.08:
        score += 0.5
        reasons.append(f"Up {ind['ret_3m']*100:.1f}% over 3 months — durable move, not a one-day spike.")

    if score >= 4.0:
        verdict = "🟢 Strong Opportunity"
    elif score >= 1.5:
        verdict = "🟡 Watch"
    else:
        verdict = "🔴 Avoid / Weak"

    return round(score, 2), verdict, reasons


# ═════════════════════════════════════════════════════════════════════════
# 4. STREAMLIT APP
# ═════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Agentic Sector Opportunity Dashboard", layout="wide", page_icon="🤖")

st.markdown("""
<style>
  .agent-pill-strong { background:#DCFCE7; color:#166534; padding:3px 10px; border-radius:20px; font-weight:700; font-size:12px; }
  .agent-pill-watch  { background:#FEF9C3; color:#854D0E; padding:3px 10px; border-radius:20px; font-weight:700; font-size:12px; }
  .agent-pill-avoid  { background:#FEE2E2; color:#991B1B; padding:3px 10px; border-radius:20px; font-weight:700; font-size:12px; }
  .dash-title { font-size:26px; font-weight:800; }
  .dash-sub   { font-size:13px; color:#666; margin-top:-6px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="dash-title">🤖 Agentic Nifty Sector Opportunity Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="dash-sub">The agent below independently scores every stock across 17 Nifty '
            'sector indices and decides — with its own reasoning — where the opportunities are.</div>',
            unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    selected_sectors = st.multiselect("Sectors to scan", list(SECTOR_STOCKS.keys()),
                                       default=list(SECTOR_STOCKS.keys()))
    history_period = st.selectbox("History window", ["6mo", "1y", "2y"], index=1)
    use_live = st.checkbox("🔄 Try live constituent refresh from NSE", value=False,
                            help="Attempts to pull today's official constituent list from NSE. "
                                 "Falls back to the built-in snapshot if NSE blocks the request "
                                 "(common on cloud hosts).")
    top_n = st.slider("How many top picks to show", 5, 30, 12)
    min_score_watch = st.slider("Score threshold for 'Watch'", 0.0, 3.0, 1.5, 0.25)
    auto_refresh = st.checkbox("Auto-refresh every 5 min", value=False)
    run_btn = st.button("▶ Run Agent", type="primary", width="stretch")

if "agent_ran" not in st.session_state:
    st.session_state.agent_ran = False

if run_btn:
    st.session_state.agent_ran = True

if not st.session_state.agent_ran:
    st.info("Pick your sectors on the left and hit **▶ Run Agent** to let it scan the market and "
            "report back with its own picks and reasoning.")
    st.stop()

if not selected_sectors:
    st.warning("Select at least one sector in the sidebar.")
    st.stop()

# ── Resolve constituents (live or snapshot) ──
sector_universe = {}
with st.spinner("🕵️ Agent is identifying the current stock universe for each sector..."):
    for sec in selected_sectors:
        live = fetch_live_constituents(sec) if use_live else None
        sector_universe[sec] = live if live else SECTOR_STOCKS[sec]

all_tickers = sorted({t for lst in sector_universe.values() for t in lst})
all_sector_index_tickers = sorted({SECTOR_INDEX_TICKER[s] for s in selected_sectors})

# ── Pull data ──
with st.spinner(f"📡 Fetching price history for {len(all_tickers)} stocks across "
                 f"{len(selected_sectors)} sectors..."):
    stock_hist = download_history(tuple(all_tickers), period=history_period)
    index_hist = download_history(tuple(all_sector_index_tickers), period=history_period)

# ── Compute indicators ──
stock_ind = {t: compute_indicators(h) for t, h in stock_hist.items()}
index_ind = {t: compute_indicators(h) for t, h in index_hist.items()}

# ── Run the Agent over every stock ──
records = []
for sec in selected_sectors:
    idx_ticker = SECTOR_INDEX_TICKER[sec]
    sector_ret_1m = index_ind.get(idx_ticker, {}).get("ret_1m", np.nan) if index_ind.get(idx_ticker) else np.nan
    for t in sector_universe[sec]:
        ind = stock_ind.get(t)
        if ind is None:
            continue
        score, verdict, reasons = agent_evaluate(ind, sector_ret_1m)
        records.append({
            "Sector": sec, "Ticker": t, "Close": round(ind["close"], 2),
            "1M %": round(ind["ret_1m"] * 100, 1) if not np.isnan(ind["ret_1m"]) else None,
            "3M %": round(ind["ret_3m"] * 100, 1) if not np.isnan(ind["ret_3m"]) else None,
            "RSI": round(ind["rsi"], 1), "Vol x20d": round(ind["vol_ratio"], 2),
            "From 52W High": round(ind["pct_from_high"] * 100, 1),
            "Agent Score": score, "Verdict": verdict, "Why": " ".join(reasons),
        })

if not records:
    st.error("No data could be fetched — this can happen if yfinance is temporarily rate-limited. "
              "Try again in a minute.")
    st.stop()

df = pd.DataFrame(records).sort_values("Agent Score", ascending=False).reset_index(drop=True)

# ═════════════════════════════════════════════════════════════════════════
# 5. AGENT'S TOP PICKS
# ═════════════════════════════════════════════════════════════════════════

st.markdown("## 🏆 Agent's Top Picks — System-Wide")
st.caption("Ranked purely by the agent's own composite score. It was not told which sectors or "
           "stocks to prefer.")

top = df.head(top_n)
for _, row in top.iterrows():
    pill_class = ("agent-pill-strong" if "Strong" in row["Verdict"]
                  else "agent-pill-watch" if "Watch" in row["Verdict"] else "agent-pill-avoid")
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1, 5])
        c1.markdown(f"**{row['Ticker']}**  \n*{row['Sector']}*")
        c2.markdown(f'<span class="{pill_class}">{row["Verdict"]}</span><br>'
                     f'Score: <b>{row["Agent Score"]}</b>', unsafe_allow_html=True)
        c3.markdown(f"₹{row['Close']}  ·  1M: {row['1M %']}%  ·  3M: {row['3M %']}%  ·  "
                    f"RSI {row['RSI']}  ·  {row['Vol x20d']}x volume  \n{row['Why']}")

st.divider()

# ═════════════════════════════════════════════════════════════════════════
# 6. SECTOR ROTATION — agent's own sector ranking
# ═════════════════════════════════════════════════════════════════════════

st.markdown("## 🔄 Sector Rotation — Where the Agent Sees Strength")
sector_summary = (df.groupby("Sector")
                   .agg(Avg_Score=("Agent Score", "mean"),
                        Strong=("Verdict", lambda s: (s.str.contains("Strong")).sum()),
                        Watch=("Verdict", lambda s: (s.str.contains("Watch")).sum()),
                        Count=("Ticker", "count"))
                   .sort_values("Avg_Score", ascending=False).reset_index())
sector_summary["Avg_Score"] = sector_summary["Avg_Score"].round(2)

col1, col2 = st.columns([3, 2])
with col1:
    st.dataframe(sector_summary, width="stretch", hide_index=True)
with col2:
    best = sector_summary.iloc[0]
    worst = sector_summary.iloc[-1]
    st.success(f"**Strongest sector right now:** {best['Sector']} "
               f"(avg score {best['Avg_Score']}, {int(best['Strong'])} strong picks)")
    st.error(f"**Weakest sector right now:** {worst['Sector']} "
             f"(avg score {worst['Avg_Score']}, {int(worst['Strong'])} strong picks)")

st.divider()

# ═════════════════════════════════════════════════════════════════════════
# 7. DRILL-DOWN PER SECTOR
# ═════════════════════════════════════════════════════════════════════════

st.markdown("## 🔍 Full Breakdown by Sector")
tabs = st.tabs(selected_sectors)
for tab, sec in zip(tabs, selected_sectors):
    with tab:
        sec_df = df[df["Sector"] == sec].sort_values("Agent Score", ascending=False)
        st.dataframe(
            sec_df[["Ticker", "Close", "1M %", "3M %", "RSI", "Vol x20d",
                    "From 52W High", "Agent Score", "Verdict"]],
            width="stretch", hide_index=True,
        )
        with st.expander("Agent's reasoning for each stock in this sector"):
            for _, r in sec_df.iterrows():
                st.markdown(f"**{r['Ticker']}** — {r['Verdict']} (score {r['Agent Score']})  \n{r['Why']}")

st.divider()

# ═════════════════════════════════════════════════════════════════════════
# 8. DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Full Agent Report (CSV)", csv,
    file_name=f"agentic_sector_report_{datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv", width="stretch",
)
st.caption(f"{len(df)} stocks evaluated across {len(selected_sectors)} sectors. "
           f"Sector constituent lists are a snapshot and may drift from NSE's live index after "
           f"a semi-annual rebalance — enable live refresh in the sidebar to reduce that risk. "
           f"This tool surfaces technical signals only; it is not investment advice.")

if auto_refresh:
    st.caption("⏱ Auto-refreshing in 5 minutes...")
    time.sleep(300)
    st.cache_data.clear()
    st.rerun()
