import os
import re
import time
from datetime import datetime
import concurrent.futures
from functools import partial

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from plotly.subplots import make_subplots
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Version-compatible width kwarg for st.dataframe / st.download_button / etc.
# Streamlit >=1.54 uses width="stretch", older uses use_container_width=True
# ---------------------------------------------------------------------------
_st_ver = tuple(int(x) for x in st.__version__.split(".")[:2])
FULL_WIDTH = {"width": "stretch"} if _st_ver >= (1, 54) else {"use_container_width": True}

# ---------------------------------------------------------------------------
# Page config (MUST be first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NSE Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(28,131,225,0.08), rgba(28,131,225,0.02));
        border: 1px solid rgba(28,131,225,0.15);
        border-radius: 0.6rem;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"] { font-size: 0.82rem; font-weight: 600; }

    /* Signal badges */
    .signal-strong-buy {
        background: #00c853; color: white;
        padding: 3px 10px; border-radius: 4px;
        font-weight: 700; font-size: 0.78rem;
    }
    .signal-buy {
        background: #69f0ae; color: #1b5e20;
        padding: 3px 10px; border-radius: 4px;
        font-weight: 700; font-size: 0.78rem;
    }
    .signal-hold {
        background: #ffd740; color: #33291a;
        padding: 3px 10px; border-radius: 4px;
        font-weight: 700; font-size: 0.78rem;
    }
    .signal-sell {
        background: #ff1744; color: white;
        padding: 3px 10px; border-radius: 4px;
        font-weight: 700; font-size: 0.78rem;
    }

    /* Value colors */
    .pos { color: #00c853; font-weight: 600; }
    .neg { color: #ff1744; font-weight: 600; }

    /* Momentum table */
    .mom-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .mom-table th {
        text-align: left; padding: 8px 10px;
        border-bottom: 2px solid rgba(128,128,128,0.4);
        font-weight: 700;
    }
    .mom-table td { padding: 7px 10px; border-bottom: 1px solid rgba(128,128,128,0.15); }
    .mom-table tr:hover { background: rgba(28,131,225,0.06); }
    .mom-table .r { text-align: right; }
    .mom-table .c { text-align: center; }

    /* Hide branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_DIR = os.path.join(APP_DIR, "html")
HIST_FILE = os.path.join(APP_DIR, "nse_historical.csv")

TICKERS = [
    "EGAD","KAPC","KUKZ","LIMT","SASN","WTK",
"CGEN",
"ABSA","SBIC","IMH","DTK","SCBK","EQTY","COOP","BKG","HFCK","KCB","NCBA",
"XPRS","SMER","KQ","NMG","SGL","TPSE","SCAN","UCHM","LKL","NBV","CRWN","CABL","PORT",
"TOTL","KEGN","KPLC","UMME",
"JUB","SLAM","KNRE","LBTY","BRIT","CIC",
"OCH","CTUM","TCL","HAFR","KURV",
"NSE",
"BOC","BAT","CARB","EABL","UNGA","EVRD","AMAC","FTGH","SKL",
"SCOM",
"LAPR","GLD","SMWF"
]

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def fmt_num(n, decimals=1):
    """Format large numbers: 1.32T, 158B, 63.8M, 623K."""
    if n is None or pd.isna(n):
        return "---"
    sign = "-" if n < 0 else ""
    a = abs(n)
    if a >= 1e12:
        return f"{sign}{a/1e12:.{decimals}f}T"
    if a >= 1e9:
        return f"{sign}{a/1e9:.{decimals}f}B"
    if a >= 1e6:
        return f"{sign}{a/1e6:.{decimals}f}M"
    if a >= 1e3:
        return f"{sign}{a/1e3:.{decimals}f}K"
    return f"{sign}{a:,.{decimals}f}"


def fmt_pct(v):
    if v is None or pd.isna(v):
        return "---"
    return f"{v:+.1f}%"


def colored_pct(v):
    if v is None or pd.isna(v):
        return "---"
    cls = "pos" if v >= 0 else "neg"
    return f'<span class="{cls}">{v:+.1f}%</span>'


def signal_badge(sig):
    cls_map = {
        "STRONG BUY": "signal-strong-buy",
        "BUY": "signal-buy",
        "HOLD": "signal-hold",
        "SELL": "signal-sell",
    }
    cls = cls_map.get(sig, "signal-hold")
    return f'<span class="{cls}">{sig}</span>'


# ---------------------------------------------------------------------------
# Data parsing helpers (from notebook)
# ---------------------------------------------------------------------------

def parse_number(s):
    if not s or s in ("—", "---", ""):
        return None
    s = s.strip().replace(",", "")
    for suffix, mult in {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}.items():
        if s.upper().endswith(suffix):
            try:
                return float(s[:-1]) * mult
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_pct(s):
    if not s:
        return None
    m = re.search(r"([+-]?[\d.]+)%", s)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def download_single_ticker(ticker, session, base_url, headers, html_dir):
    """Download a single ticker's HTML page."""
    try:
        r = session.get(base_url.format(ticker.lower()),
                        headers=headers, timeout=10)
        r.raise_for_status()
        with open(os.path.join(html_dir, f"{ticker}.html"), "w",
                  encoding="utf-8") as f:
            f.write(r.text)
        return ticker, True
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")
        return ticker, False


def download_html_pages(tickers, progress_container):
    """Download HTML pages from afx.kwayisi.org using concurrent downloads."""
    BASE_URL = "https://afx.kwayisi.org/nse/{}.html"
    HEADERS = {"User-Agent": "Mozilla/5.0"}
    os.makedirs(HTML_DIR, exist_ok=True)

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    bar = progress_container.progress(0, text="Starting download...")
    results = {}
    
    # Create a partial function with pre-filled parameters
    download_func = partial(download_single_ticker, session=session, 
                           base_url=BASE_URL, headers=HEADERS, html_dir=HTML_DIR)
    
    # Use ThreadPoolExecutor for concurrent downloads
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_ticker = {executor.submit(download_func, ticker): ticker 
                           for ticker in tickers}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker, success = future.result()
            results[ticker] = success
            completed += 1
            bar.progress(completed / len(tickers),
                        text=f"Downloaded {ticker}... ({completed}/{len(tickers)})")
            time.sleep(0.1)  # Small delay between processing results

    bar.empty()
    return results


@st.cache_data(ttl=300, show_spinner=False)
def parse_all_html():
    """Parse all HTML files and return (snapshot_df, history_df)."""
    if not os.path.isdir(HTML_DIR):
        return pd.DataFrame(), pd.DataFrame()

    results = []
    history_rows = []

    for filename in sorted(os.listdir(HTML_DIR)):
        if not filename.endswith(".html"):
            continue
        ticker = filename.replace(".html", "")
        try:
            with open(os.path.join(HTML_DIR, filename), "r",
                      encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "lxml")
        except Exception:
            continue

        text = soup.get_text(" ", strip=True)

        # Price & change
        pm = re.search(
            rf"{ticker}\s*[•·]\s*([\d.]+)\s*[▴▾▵▿]?\s*([+-]?[\d.]+)\s*\(([\d.]+%)\)",
            text, re.IGNORECASE,
        )
        price = float(pm.group(1)) if pm else None
        change = float(pm.group(2)) if pm else None
        change_pct = parse_pct(pm.group(3)) if pm else None

        # Company name
        h1 = soup.find("h1")
        company = h1.text.split(" - ", 1)[1].strip() if h1 and " - " in h1.text else None

        # Table value helper
        def tv(label):
            for td in soup.find_all("td"):
                if td.get_text(strip=True) == label:
                    nxt = td.find_next_sibling("td")
                    return nxt.get_text(strip=True) if nxt else None
            return None

        opening = parse_number(tv("Opening Price"))
        low = parse_number(tv("Day's Low Price"))
        high = parse_number(tv("Day's High Price"))
        volume = parse_number(tv("Traded Volume"))
        deals = parse_number(tv("Number of Deals"))
        turnover = parse_number(tv("Gross Turnover"))
        eps = parse_number(tv("Earnings Per Share"))
        pe = parse_number(tv("Price/Earning Ratio"))
        dps = parse_number(tv("Dividend Per Share"))
        dy_raw = tv("Dividend Yield")
        div_yield = parse_pct(dy_raw) if dy_raw else None
        shares_out = parse_number(tv("Shares Outstanding"))
        mcap = parse_number(tv("Market Capitalization"))

        # Performance
        perf_div = soup.find("div", attrs={"data-perf": True})
        perf = {}
        if perf_div:
            hdrs = [th.get("title", th.text) for th in perf_div.find_all("th")]
            vals = [td.get_text(strip=True) for td in perf_div.find_all("td")]
            for h, v in zip(hdrs, vals):
                perf[h] = parse_pct(v)

        # Sector / Industry
        sector = industry = None
        fact = soup.find("div", attrs={"data-fact": True})
        if fact:
            for dt in fact.find_all("dt"):
                dd = dt.find_next_sibling("dd")
                if dd:
                    lbl = dt.get_text(strip=True)
                    if lbl == "Sector":
                        sector = dd.get_text(strip=True)
                    elif lbl == "Industry":
                        industry = dd.get_text(strip=True)

        # 10-day history
        hist = soup.find("table", attrs={"data-hist": True})
        if hist and hist.find("tbody"):
            for tr in hist.find("tbody").find_all("tr"):
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) >= 3:
                    history_rows.append({
                        "Ticker": ticker,
                        "Date": tds[0],
                        "Volume": parse_number(tds[1]),
                        "Close": parse_number(tds[2]),
                        "Change": parse_number(tds[3]) if len(tds) > 3 else None,
                        "Change%": parse_pct(tds[4]) if len(tds) > 4 else None,
                    })

        results.append({
            "Ticker": ticker, "Company": company,
            "Sector": sector, "Industry": industry,
            "Price": price, "Change": change, "Change%": change_pct,
            "Open": opening, "Low": low, "High": high,
            "Volume": volume, "Deals": deals, "Turnover": turnover,
            "EPS": eps, "P/E": pe, "DPS": dps, "Div Yield%": div_yield,
            "Shares Out": shares_out, "Market Cap": mcap,
            "1WK%": perf.get("1-Week"), "4WK%": perf.get("4-Week"),
            "3MO%": perf.get("3-Month"), "6MO%": perf.get("6-Month"),
            "1YR%": perf.get("1-Year"), "YTD%": perf.get("Year-to-Date"),
        })

    df = pd.DataFrame(results)
    df_hist = pd.DataFrame(history_rows)
    if not df_hist.empty:
        df_hist["Date"] = pd.to_datetime(df_hist["Date"], format="%Y-%m-%d")
    return df, df_hist


def accumulate_historical(df, df_hist):
    """Merge 10-day history into wide-format CSV that grows over time."""
    if df_hist.empty:
        return pd.DataFrame()

    wide = df_hist.copy()
    wide["DateStr"] = wide["Date"].dt.strftime("%Y-%m-%d")

    close_piv = wide.pivot(index="Ticker", columns="DateStr", values="Close")
    close_piv.columns = [f"Close_{d}" for d in close_piv.columns]
    vol_piv = wide.pivot(index="Ticker", columns="DateStr", values="Volume")
    vol_piv.columns = [f"Volume_{d}" for d in vol_piv.columns]

    new_data = close_piv.join(vol_piv)
    static = df.set_index("Ticker")[["Company", "Sector", "Industry"]]
    new_data = static.join(new_data)
    new_data.index.name = "Ticker"

    if os.path.exists(HIST_FILE):
        existing = pd.read_csv(HIST_FILE, index_col="Ticker")
        new_cols = [c for c in new_data.columns
                    if c.startswith(("Close_", "Volume_"))]
        for col in new_cols:
            existing[col] = new_data[col]
        for col in ["Company", "Sector", "Industry"]:
            if col in existing.columns:
                existing[col] = new_data[col].combine_first(existing[col])
            else:
                existing[col] = new_data[col]
        new_tickers = new_data.index.difference(existing.index)
        if len(new_tickers):
            existing = pd.concat([existing, new_data.loc[new_tickers]])
        merged = existing
    else:
        merged = new_data

    stat_cols = ["Company", "Sector", "Industry"]
    date_cols = sorted(c for c in merged.columns if c not in stat_cols)
    merged = merged[stat_cols + date_cols]
    merged.to_csv(HIST_FILE)
    return merged


def compute_momentum(df):
    """Compute momentum score, rank, and signal for each stock."""
    a = df[["Ticker", "Company", "Sector", "Price", "Open", "Low", "High",
            "Volume", "Turnover", "Market Cap",
            "1WK%", "4WK%", "3MO%", "6MO%", "1YR%", "YTD%"]].copy()

    price = pd.to_numeric(a["Price"], errors="coerce").fillna(0)
    high = pd.to_numeric(a["High"], errors="coerce").fillna(0)
    low = pd.to_numeric(a["Low"], errors="coerce").fillna(0)
    a["Intraday Range%"] = np.where(
        price > 0,
        ((high - low) / price * 100).round(2),
        None,
    )
    a["Momentum Score"] = (
        a["1WK%"].fillna(0) * 0.30 +
        a["4WK%"].fillna(0) * 0.25 +
        a["3MO%"].fillna(0) * 0.20 +
        a["YTD%"].fillna(0) * 0.15 +
        a["6MO%"].fillna(0) * 0.10
    ).round(2)
    a["Momentum Rank"] = a["Momentum Score"].rank(ascending=False).astype(int)

    def sig(r):
        w1 = r["1WK%"] or 0
        w4 = r["4WK%"] or 0
        m3 = r["3MO%"] or 0
        ytd = r["YTD%"] or 0
        if w1 > 0 and w4 > 0 and m3 > 0 and ytd > 0 and w1 > (w4 / 4):
            return "STRONG BUY"
        if w1 > 0 and w4 > 0 and m3 > 0:
            return "BUY"
        if w1 > 0 or w4 > 0:
            return "HOLD"
        if w1 < 0 and w4 < 0:
            return "SELL"
        return "HOLD"

    a["Signal"] = a.apply(sig, axis=1)
    return a.sort_values("Momentum Rank")


def compute_trend_summary(df_hist):
    """10-day trend stats per ticker."""
    rows = []
    for ticker, grp in df_hist.groupby("Ticker"):
        grp = grp.sort_values("Date")
        if len(grp) < 2:
            continue
        oldest = grp.iloc[0]["Close"]
        latest = grp.iloc[-1]["Close"]
        pct = ((latest - oldest) / oldest * 100) if oldest else 0
        avg_v = grp["Volume"].mean()
        max_v = grp["Volume"].max()
        spike = max_v / avg_v if avg_v else 0
        up = (grp["Change"].fillna(0) > 0).sum()
        down = (grp["Change"].fillna(0) < 0).sum()
        rows.append({
            "Ticker": ticker,
            "10D Change%": round(pct, 2),
            "Avg Volume": int(avg_v),
            "Max Volume": int(max_v),
            "Vol Spike Ratio": round(spike, 1),
            "Up Days": int(up),
            "Down Days": int(down),
            "Win Rate%": round(up / len(grp) * 100, 1),
        })
    return pd.DataFrame(rows).sort_values("10D Change%", ascending=False)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df, df_hist = parse_all_html()
has_data = not df.empty

if has_data:
    historical = accumulate_historical(df, df_hist)
    momentum = compute_momentum(df)
    trend = compute_trend_summary(df_hist)
else:
    historical = pd.DataFrame()
    momentum = pd.DataFrame()
    trend = pd.DataFrame()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 NSE Analyzer")
    st.caption("Nairobi Securities Exchange")
    st.divider()

    # Refresh All Data button
    st.markdown("### 🔄 Data Refresh")
    if st.button("🔄 Refresh All Data", type="primary", **FULL_WIDTH):
        progress_area = st.empty()
        results = download_html_pages(TICKERS, progress_area)
        ok = sum(1 for v in results.values() if v)
        st.success(f"Downloaded {ok}/{len(TICKERS)} stocks")
        st.cache_data.clear()
        st.session_state["last_refresh"] = datetime.now()
        time.sleep(0.5)
        st.rerun()
    
    # Selective refresh option
    st.markdown("**— OR —**")
    st.caption("Refresh specific tickers only")
    
    selected_tickers_refresh = st.multiselect(
        "Select tickers to refresh",
        options=TICKERS,
        default=[],
        key="refresh_ticker_select",
        help="Choose one or more tickers to refresh only those"
    )
    
    refresh_selected_disabled = len(selected_tickers_refresh) == 0
    if st.button(
        "🔄 Refresh Selected", 
        disabled=refresh_selected_disabled,
        type="secondary" if not refresh_selected_disabled else "secondary",
        **FULL_WIDTH
    ):
        progress_area = st.empty()
        results = download_html_pages(selected_tickers_refresh, progress_area)
        ok = sum(1 for v in results.values() if v)
        st.success(f"Downloaded {ok}/{len(selected_tickers_refresh)} selected stocks")
        st.cache_data.clear()
        st.session_state["last_refresh"] = datetime.now()
        time.sleep(0.5)
        st.rerun()

    if "last_refresh" in st.session_state:
        st.caption(
            f"Last refresh: {st.session_state['last_refresh'].strftime('%b %d, %Y %H:%M')}"
        )
    elif has_data:
        # Show file modification time as proxy
        sample = os.path.join(HTML_DIR, f"{TICKERS[0]}.html")
        if os.path.exists(sample):
            mtime = datetime.fromtimestamp(os.path.getmtime(sample))
            st.caption(f"Data from: {mtime.strftime('%b %d, %Y %H:%M')}")

    st.divider()

    # Filters
    if has_data:
        sectors = ["All Sectors"] + sorted(df["Sector"].dropna().unique().tolist())
        sel_sector = st.selectbox("Sector", sectors)

        avail = df["Ticker"].tolist()
        if sel_sector != "All Sectors":
            avail = df[df["Sector"] == sel_sector]["Ticker"].tolist()
        sel_tickers = st.multiselect("Stocks", avail, default=avail)

        sig_opts = ["All Signals", "STRONG BUY", "BUY", "HOLD", "SELL"]
        sel_signal = st.selectbox("Signal Filter", sig_opts)
    else:
        sel_tickers = []
        sel_signal = "All Signals"

    st.divider()
    st.caption("Data: afx.kwayisi.org")
    st.caption("Built with Streamlit + Plotly")

# ---------------------------------------------------------------------------
# Filter helper
# ---------------------------------------------------------------------------

def filt(frame, ticker_col="Ticker"):
    out = frame[frame[ticker_col].isin(sel_tickers)] if sel_tickers else frame
    return out

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 NSE Stock Analyzer")
st.markdown(
    "Real-time analysis of **Nairobi Securities Exchange** stocks — "
    "momentum scoring, trend detection & sector breakdown"
)

if not has_data:
    st.info(
        "No stock data found. Click **🔄 Refresh Data** in the sidebar to "
        "download the latest NSE prices."
    )
    st.stop()

# Apply filters
f_df = filt(df)
f_mom = filt(momentum)
if sel_signal != "All Signals":
    f_mom = f_mom[f_mom["Signal"] == sel_signal]
f_hist = filt(df_hist)
f_trend = filt(trend)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)
total_mcap = f_df["Market Cap"].sum()
avg_ytd = f_df["YTD%"].mean()
avg_1wk = f_df["1WK%"].mean()
total_vol = f_df["Volume"].sum()
gainers = int((f_df["Change%"].fillna(0) > 0).sum())
losers = int((f_df["Change%"].fillna(0) < 0).sum())
total_turn = f_df["Turnover"].sum()

c1.metric("Total Market Cap", fmt_num(total_mcap))
c2.metric("Avg YTD Return", fmt_pct(avg_ytd),
          delta=fmt_pct(avg_ytd), delta_color="normal")
c3.metric("Avg 1-Week", fmt_pct(avg_1wk),
          delta=fmt_pct(avg_1wk), delta_color="normal")
c4.metric("Today's Volume", fmt_num(total_vol))
c5.metric("Gainers / Losers", f"{gainers} / {losers}",
          delta=f"{gainers - losers:+d}", delta_color="normal")
c6.metric("Turnover", fmt_num(total_turn))

st.markdown("")  # spacer

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🚀 Momentum & Signals",
    "📈 Price Trends",
    "🏢 Sectors",
    "📋 Historical Data",
])

# ========================== TAB 1: OVERVIEW ================================
with tab1:
    st.subheader("Market Snapshot")
    st.dataframe(
        f_df[["Ticker", "Company", "Price", "Change", "Change%",
              "Open", "Low", "High", "Volume", "Turnover", "Market Cap",
              "EPS", "P/E", "DPS", "Div Yield%"]],
        column_config={
            "Price": st.column_config.NumberColumn("Price (KES)", format="%.2f"),
            "Change": st.column_config.NumberColumn("Change", format="%+.2f"),
            "Change%": st.column_config.NumberColumn("Change %", format="%.2f%%"),
            "Market Cap": st.column_config.NumberColumn("Mkt Cap", format="%.0f"),
            "EPS": st.column_config.NumberColumn("EPS", format="%.2f"),
            "P/E": st.column_config.NumberColumn("P/E", format="%.1f"),
            "DPS": st.column_config.NumberColumn("DPS", format="%.2f"),
            "Div Yield%": st.column_config.NumberColumn("Div Yield %", format="%.2f%%"),
        },
        hide_index=True,
        height=min(600, 45 + len(f_df) * 35),
        **FULL_WIDTH,
    )

    left, right = st.columns(2)

    # YTD bar chart
    with left:
        st.subheader("Year-to-Date Performance")
        ytd_data = f_df[["Ticker", "YTD%"]].dropna().sort_values("YTD%")
        fig_ytd = go.Figure(go.Bar(
            x=ytd_data["YTD%"], y=ytd_data["Ticker"], orientation="h",
            marker_color=[
                "#00c853" if v >= 0 else "#ff1744" for v in ytd_data["YTD%"]
            ],
            text=[f"{v:+.1f}%" for v in ytd_data["YTD%"]],
            textposition="outside",
        ))
        fig_ytd.update_layout(
            height=max(400, len(ytd_data) * 32),
            xaxis_title="YTD %", yaxis_title="",
            margin=dict(l=0, r=60, t=10, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ytd, **FULL_WIDTH)

    # Performance heatmap
    with right:
        st.subheader("Performance Heatmap")
        perf_cols = ["1WK%", "4WK%", "3MO%", "6MO%", "1YR%", "YTD%"]
        heat_df = f_df.set_index("Ticker")[perf_cols].copy()

        fig_heat = go.Figure(go.Heatmap(
            z=heat_df.values,
            x=["1WK", "4WK", "3MO", "6MO", "1YR", "YTD"],
            y=heat_df.index,
            colorscale=[[0, "#ff1744"], [0.5, "#f5f5f5"], [1, "#00c853"]],
            zmid=0,
            text=heat_df.map(
                lambda x: f"{x:+.1f}%" if pd.notna(x) else "---"
            ).values,
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        ))
        fig_heat.update_layout(
            height=max(400, len(heat_df) * 30),
            margin=dict(l=0, r=0, t=10, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat, **FULL_WIDTH)


# ======================== TAB 2: MOMENTUM ==================================
with tab2:
    st.subheader("Momentum Rankings & Trading Signals")

    # Signal count cards
    sig_counts = f_mom["Signal"].value_counts() if not f_mom.empty else pd.Series()
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("🟢 STRONG BUY", int(sig_counts.get("STRONG BUY", 0)))
    sc2.metric("🟢 BUY", int(sig_counts.get("BUY", 0)))
    sc3.metric("🟡 HOLD", int(sig_counts.get("HOLD", 0)))
    sc4.metric("🔴 SELL", int(sig_counts.get("SELL", 0)))

    st.markdown("")

    # Momentum HTML table
    if not f_mom.empty:
        rows_html = ""
        for _, r in f_mom.iterrows():
            rows_html += f"""<tr>
                <td class="c" style="font-weight:700;">{r['Momentum Rank']}</td>
                <td><strong>{r['Ticker']}</strong></td>
                <td>{r['Company'] or '---'}</td>
                <td class="r">{r['Price']:.2f}</td>
                <td class="r">{colored_pct(r.get('1WK%'))}</td>
                <td class="r">{colored_pct(r.get('4WK%'))}</td>
                <td class="r">{colored_pct(r.get('3MO%'))}</td>
                <td class="r">{colored_pct(r.get('YTD%'))}</td>
                <td class="r" style="font-weight:700;">{r['Momentum Score']:.1f}</td>
                <td class="c">{signal_badge(r['Signal'])}</td>
            </tr>"""

        st.markdown(f"""<table class="mom-table">
            <thead><tr>
                <th class="c">Rank</th><th>Ticker</th><th>Company</th>
                <th class="r">Price</th><th class="r">1WK</th><th class="r">4WK</th>
                <th class="r">3MO</th><th class="r">YTD</th>
                <th class="r">Score</th><th class="c">Signal</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

    st.markdown("")

    # Momentum bar chart
    st.subheader("Momentum Score Distribution")
    color_map = {"STRONG BUY": "#00c853", "BUY": "#69f0ae",
                 "HOLD": "#ffd740", "SELL": "#ff1744"}
    if not f_mom.empty:
        fig_bar = px.bar(
            f_mom.sort_values("Momentum Score"),
            x="Momentum Score", y="Ticker", orientation="h",
            color="Signal", color_discrete_map=color_map,
            text="Momentum Score",
        )
        fig_bar.update_layout(
            height=max(400, len(f_mom) * 30),
            margin=dict(l=0, r=0, t=10, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        st.plotly_chart(fig_bar, **FULL_WIDTH)

    # Top picks callout
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🟢 Top Picks")
        sbuys = momentum[momentum["Signal"] == "STRONG BUY"]
        buys = momentum[momentum["Signal"] == "BUY"]
        if not sbuys.empty:
            for _, r in sbuys.iterrows():
                st.success(
                    f"**{r['Ticker']}** — {r['Company']}  \n"
                    f"Score: **{r['Momentum Score']:.1f}** | "
                    f"Price: KES {r['Price']:.2f} | "
                    f"YTD: {fmt_pct(r['YTD%'])}"
                )
        if not buys.empty:
            for _, r in buys.iterrows():
                st.info(
                    f"**{r['Ticker']}** — {r['Company']}  \n"
                    f"Score: {r['Momentum Score']:.1f} | "
                    f"Price: KES {r['Price']:.2f}"
                )
        if sbuys.empty and buys.empty:
            st.caption("No BUY signals currently.")

    with col_b:
        st.subheader("🔴 Underperformers")
        sells = momentum[momentum["Signal"] == "SELL"]
        if not sells.empty:
            for _, r in sells.iterrows():
                st.error(
                    f"**{r['Ticker']}** — {r['Company']}  \n"
                    f"Score: {r['Momentum Score']:.1f} | "
                    f"Price: KES {r['Price']:.2f}"
                )
        else:
            st.caption("No SELL signals currently.")

    st.caption(
        "⚠️ Signals are based on momentum trends only. "
        "Always consider fundamentals (EPS, P/E, dividends) before trading."
    )


# ======================== TAB 3: TRENDS ====================================
with tab3:
    st.subheader("10-Day Trend Analysis")

    if not f_trend.empty:
        st.dataframe(
            f_trend,
            column_config={
                "10D Change%": st.column_config.NumberColumn(format="%.2f%%"),
                "Vol Spike Ratio": st.column_config.NumberColumn(format="%.1f"),
                "Win Rate%": st.column_config.NumberColumn(format="%.1f%%"),
            },
            hide_index=True,
            **FULL_WIDTH,
        )

        # Volume spike alerts
        spikes = f_trend[f_trend["Vol Spike Ratio"] > 2.0]
        if not spikes.empty:
            st.warning(
                f"⚡ **Volume spike detected:** "
                f"{', '.join(spikes['Ticker'].tolist())} "
                f"(ratio > 2.0x — possible breakout)"
            )

    # Top movers subplots
    if not f_hist.empty and not f_trend.empty:
        st.subheader("Top Movers — 10-Day Price Trends")
        top_n = min(6, len(f_trend))
        movers = f_trend.head(top_n)["Ticker"].tolist()
        rows_n = (top_n + 2) // 3
        cols_n = min(3, top_n)

        fig_sub = make_subplots(
            rows=rows_n, cols=cols_n,
            subplot_titles=[
                f"{t} ({f_trend[f_trend['Ticker']==t]['10D Change%'].values[0]:+.1f}%)"
                for t in movers
            ],
            vertical_spacing=0.14,
            horizontal_spacing=0.06,
        )
        for i, ticker in enumerate(movers):
            r = i // 3 + 1
            c = i % 3 + 1
            grp = f_hist[f_hist["Ticker"] == ticker].sort_values("Date")
            chg = f_trend[f_trend["Ticker"] == ticker]["10D Change%"].values[0]
            color = "#00c853" if chg >= 0 else "#ff1744"
            fig_sub.add_trace(go.Scatter(
                x=grp["Date"], y=grp["Close"],
                mode="lines+markers",
                fill="tozeroy",
                fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba") if "rgb" in color
                    else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)",
                line=dict(color=color, width=2),
                marker=dict(size=5),
                showlegend=False,
            ), row=r, col=c)

        fig_sub.update_layout(
            height=280 * rows_n,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_sub.update_xaxes(tickformat="%b %d")
        st.plotly_chart(fig_sub, **FULL_WIDTH)

    # Individual stock detail
    if not f_hist.empty:
        st.subheader("Individual Stock Detail")
        stock_pick = st.selectbox("Select stock", sorted(f_hist["Ticker"].unique()))
        sdata = f_hist[f_hist["Ticker"] == stock_pick].sort_values("Date")

        fig_det = make_subplots(specs=[[{"secondary_y": True}]])
        fig_det.add_trace(
            go.Scatter(x=sdata["Date"], y=sdata["Close"],
                       mode="lines+markers", name="Close Price",
                       line=dict(color="#2196f3", width=2.5)),
            secondary_y=False,
        )
        fig_det.add_trace(
            go.Bar(x=sdata["Date"], y=sdata["Volume"],
                   name="Volume", marker_color="rgba(255,167,38,0.45)"),
            secondary_y=True,
        )
        fig_det.update_layout(
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig_det.update_xaxes(title_text="Date", tickformat="%b %d")
        fig_det.update_yaxes(title_text="Price (KES)", secondary_y=False)
        fig_det.update_yaxes(title_text="Volume", secondary_y=True)
        st.plotly_chart(fig_det, **FULL_WIDTH)


# ======================== TAB 4: SECTORS ===================================
with tab4:
    st.subheader("Sector Breakdown")

    left_s, right_s = st.columns(2)

    with left_s:
        sec_mcap = f_df.groupby("Sector")["Market Cap"].sum().reset_index()
        sec_mcap.columns = ["Sector", "Market Cap"]
        sec_mcap = sec_mcap.sort_values("Market Cap", ascending=False)

        fig_pie = px.pie(
            sec_mcap, values="Market Cap", names="Sector",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont_size=12,
        )
        fig_pie.update_layout(
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, **FULL_WIDTH)

    with right_s:
        sec_sum = f_df.groupby("Sector").agg(
            Stocks=("Ticker", "count"),
            Total_MCap=("Market Cap", "sum"),
            Avg_YTD=("YTD%", "mean"),
            Avg_1WK=("1WK%", "mean"),
        ).round(2)
        sec_sum["Market Cap"] = sec_sum["Total_MCap"].apply(fmt_num)
        sec_sum["Avg YTD%"] = sec_sum["Avg_YTD"].apply(fmt_pct)
        sec_sum["Avg 1WK%"] = sec_sum["Avg_1WK"].apply(fmt_pct)
        st.dataframe(
            sec_sum[["Stocks", "Market Cap", "Avg YTD%", "Avg 1WK%"]],
            **FULL_WIDTH,
        )

    # Sector performance comparison
    st.subheader("Sector Performance Comparison")
    sec_perf = f_df.groupby("Sector").agg(
        Avg_1WK=("1WK%", "mean"), Avg_YTD=("YTD%", "mean")
    ).reset_index().round(2)

    fig_grp = go.Figure()
    fig_grp.add_trace(go.Bar(
        name="Avg 1-Week %", x=sec_perf["Sector"], y=sec_perf["Avg_1WK"],
        marker_color="#2196f3",
        text=sec_perf["Avg_1WK"].apply(lambda x: f"{x:+.1f}%"),
        textposition="outside",
    ))
    fig_grp.add_trace(go.Bar(
        name="Avg YTD %", x=sec_perf["Sector"], y=sec_perf["Avg_YTD"],
        marker_color="#ff9800",
        text=sec_perf["Avg_YTD"].apply(lambda x: f"{x:+.1f}%"),
        textposition="outside",
    ))
    fig_grp.update_layout(
        barmode="group", height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_grp, **FULL_WIDTH)

    # Most liquid stocks
    st.subheader("Most Liquid Stocks (easiest to trade)")
    liq = f_df.nlargest(5, "Turnover")[
        ["Ticker", "Company", "Price", "Volume", "Turnover", "Deals"]
    ].copy()
    liq["Volume"] = liq["Volume"].apply(fmt_num)
    liq["Turnover"] = liq["Turnover"].apply(fmt_num)
    st.dataframe(liq, hide_index=True, **FULL_WIDTH)


# ======================== TAB 5: HISTORICAL ================================
with tab5:
    st.subheader("Accumulated Historical Data")

    if not historical.empty:
        close_dates = sorted(
            c.replace("Close_", "")
            for c in historical.columns if c.startswith("Close_")
        )
        st.caption(
            f"Tracking **{historical.shape[0]}** stocks across "
            f"**{len(close_dates)}** trading days  \n"
            f"Date range: **{close_dates[0]}** to **{close_dates[-1]}**"
        )
        st.dataframe(historical, height=500, **FULL_WIDTH)
        st.download_button(
            label="📥 Download Historical Data (CSV)",
            data=historical.to_csv(),
            file_name=f"nse_historical_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            **FULL_WIDTH,
        )
    else:
        st.info("No historical data yet. Click **Refresh Data** to start building history.")

    st.divider()

    st.subheader("Current Snapshot")
    st.dataframe(f_df, hide_index=True, **FULL_WIDTH)
    st.download_button(
        label="📥 Download Snapshot (CSV)",
        data=f_df.to_csv(index=False),
        file_name=f"nse_snapshot_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    st.divider()

    st.subheader("10-Day History (Long Format)")
    st.dataframe(f_hist, hide_index=True, **FULL_WIDTH)
    if not f_hist.empty:
        st.download_button(
            label="📥 Download 10-Day History (CSV)",
            data=f_hist.to_csv(index=False),
            file_name=f"nse_10d_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
