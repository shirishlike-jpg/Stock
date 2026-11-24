import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NSE EOD Stock Screener",
    page_icon="📈",
    layout="wide",
)

IST = ZoneInfo("Asia/Kolkata")
NSE_ARCHIVE_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"
)
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}

st.title("NSE EOD Stock Screener")
st.caption(
    "Data source: NSE India official CM-UDiFF Bhavcopy Final only | "
    "Daily EMA 20/50 | RSI 14 | Volume SMA 5 | Previous completed month's R1"
)


def _find_col(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def download_one_nse_bhavcopy(date_yyyymmdd: str) -> pd.DataFrame:
    """Download and parse one official NSE CM UDiFF final bhavcopy."""
    url = NSE_ARCHIVE_URL.format(date=date_yyyymmdd)
    try:
        response = requests.get(url, headers=NSE_HEADERS, timeout=(8, 25))
        if response.status_code != 200 or not response.content:
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                return pd.DataFrame()
            with archive.open(csv_names[0]) as f:
                raw = pd.read_csv(f, low_memory=False)
        raw.columns = [str(c).strip() for c in raw.columns]

        mapping = {
            "Symbol": _find_col(raw.columns, ["TckrSymb", "SYMBOL"]),
            "Series": _find_col(raw.columns, ["SctySrs", "SERIES"]),
            "Open": _find_col(raw.columns, ["OpnPric", "OPEN"]),
            "High": _find_col(raw.columns, ["HghPric", "HIGH"]),
            "Low": _find_col(raw.columns, ["LwPric", "LOW"]),
            "Close": _find_col(raw.columns, ["ClsPric", "CLOSE"]),
            "Volume": _find_col(raw.columns, ["TtlTradgVol", "TOTTRDQTY", "VOLUME"]),
            "TradeDate": _find_col(raw.columns, ["TradDt", "DATE1", "TIMESTAMP"]),
        }
        if any(mapping[k] is None for k in ["Symbol", "Series", "Open", "High", "Low", "Close", "Volume"]):
            return pd.DataFrame()

        out = pd.DataFrame({new: raw[old] for new, old in mapping.items() if old is not None})
        out["Symbol"] = out["Symbol"].astype(str).str.strip().str.upper()
        out["Series"] = out["Series"].astype(str).str.strip().str.upper()
        out = out[out["Series"] == "EQ"].copy()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        out = out[
            (out["Open"] > 0) & (out["High"] > 0) & (out["Low"] > 0)
            & (out["Close"] > 0) & (out["Volume"] >= 0)
            & (out["High"] >= out["Low"])
        ]
        if "TradeDate" in out.columns:
            parsed = pd.to_datetime(out["TradeDate"], errors="coerce", dayfirst=True)
            if parsed.notna().any():
                out["Date"] = parsed
            else:
                out["Date"] = pd.to_datetime(date_yyyymmdd, format="%Y%m%d")
        else:
            out["Date"] = pd.to_datetime(date_yyyymmdd, format="%Y%m%d")
        out["Date"] = out["Date"].dt.normalize()
        return out[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]]
    except (requests.RequestException, zipfile.BadZipFile, pd.errors.ParserError, ValueError, KeyError):
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def load_nse_history(lookback_calendar_days: int = 170):
    """Build daily history for all NSE EQ symbols from NSE's official bhavcopies."""
    now = datetime.now(IST)
    end_date = now.date()
    # Do not use today's unfinished candle before the EOD cutoff.
    if now.hour < 15 or (now.hour == 15 and now.minute < 45):
        end_date -= timedelta(days=1)

    dates = [
        (end_date - timedelta(days=i)).strftime("%Y%m%d")
        for i in range(lookback_calendar_days)
    ]
    frames = []
    successful_days = []
    failed_days = []

    # NSE publishes files only for trading sessions. Parallel downloads keep
    # the first load practical; individual files are cached for later reruns.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_one_nse_bhavcopy, d): d for d in dates}
        for future in as_completed(futures):
            d = futures[future]
            try:
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
                    successful_days.append(d)
                else:
                    failed_days.append(d)
            except Exception:
                failed_days.append(d)

    if not frames:
        return pd.DataFrame(), successful_days, failed_days

    history = pd.concat(frames, ignore_index=True)
    history = history.drop_duplicates(["Date", "Symbol"], keep="last")
    history = history.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return history, successful_days, failed_days


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
    return rsi


def add_previous_month_r1(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().set_index("Date").sort_index()
    monthly = data.resample("ME").agg(
        {"High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    pivot = (monthly["High"] + monthly["Low"] + monthly["Close"]) / 3.0
    monthly["R1"] = 2.0 * pivot - monthly["Low"]
    monthly["R1_for_next_month"] = monthly["R1"].shift(1)
    month_r1 = monthly["R1_for_next_month"]
    data["Monthly_R1"] = data.index.to_period("M").map(month_r1.to_dict())
    return data.reset_index()


def analyze_one_symbol(symbol: str, data: pd.DataFrame):
    df = data.sort_values("Date").drop_duplicates("Date").copy()
    if len(df) < 65:
        return None

    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False, min_periods=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False, min_periods=50).mean()
    df["RSI_14"] = calculate_rsi(df["Close"], 14)
    df["Volume_SMA_5"] = df["Volume"].rolling(5, min_periods=5).mean()
    df = add_previous_month_r1(df)
    df["Change_Percent"] = df["Close"].pct_change() * 100.0

    checks = {
        "High > Previous High": df["High"] > df["High"].shift(1),
        "Close > Open": df["Close"] > df["Open"],
        "Close > EMA 20": df["Close"] > df["EMA_20"],
        "Close > EMA 50": df["Close"] > df["EMA_50"],
        "Change > 2%": df["Change_Percent"] > 2.0,
        "Volume > SMA 5": df["Volume"] > df["Volume_SMA_5"],
        "RSI > 65": df["RSI_14"] > 65.0,
        "Close > Monthly R1": df["Close"] > df["Monthly_R1"],
        "Previous Close <= Monthly R1": df["Close"].shift(1) <= df["Monthly_R1"],
    }
    for name, values in checks.items():
        df[name] = values.fillna(False)

    last = df.iloc[-1]
    required = ["EMA_20", "EMA_50", "RSI_14", "Volume_SMA_5", "Monthly_R1", "Change_Percent"]
    if last[required].isna().any():
        return None

    signal = all(bool(last[name]) for name in checks)
    return {
        "Symbol": symbol,
        "Candle Date": pd.Timestamp(last["Date"]).strftime("%Y-%m-%d"),
        "Previous Close": round(float(df.iloc[-2]["Close"]), 2),
        "Latest Open": round(float(last["Open"]), 2),
        "Latest High": round(float(last["High"]), 2),
        "Latest Low": round(float(last["Low"]), 2),
        "Latest Close": round(float(last["Close"]), 2),
        "Change %": round(float(last["Change_Percent"]), 2),
        "Previous Day High": round(float(df.iloc[-2]["High"]), 2),
        "EMA 20": round(float(last["EMA_20"]), 2),
        "EMA 50": round(float(last["EMA_50"]), 2),
        "Volume": int(last["Volume"]),
        "Volume SMA 5": round(float(last["Volume_SMA_5"]), 0),
        "RSI 14": round(float(last["RSI_14"]), 2),
        "Previous Completed Month R1": round(float(last["Monthly_R1"]), 2),
        **{name: bool(last[name]) for name in checks},
        "Signal": "BUY" if signal else "WAIT",
    }


with st.sidebar:
    st.header("Scanner Settings")
    universe = st.selectbox(
        "NSE Cash Market Universe",
        ["All NSE EQ stocks", "NIFTY 500", "NIFTY 50"],
    )
    lookback_days = st.selectbox(
        "NSE historical data lookback",
        [150, 170, 200],
        index=1,
        help="NSE bhavcopy files are downloaded and cached. About 100+ trading sessions are targeted for EMA/RSI warm-up.",
    )
    uploaded_symbols = st.file_uploader("Optional symbols CSV", type=["csv"])
    st.caption("Optional CSV must have a Symbol column, e.g. RELIANCE.")
    st.divider()
    st.markdown("**BUY conditions (all 9 required)**")
    st.write("1. Daily High > previous day's High")
    st.write("2. Daily Close > Daily Open")
    st.write("3. Daily Close > EMA 20 and EMA 50")
    st.write("4. Daily change > 2%")
    st.write("5. Daily Volume > Volume SMA 5")
    st.write("6. Daily RSI 14 > 65")
    st.write("7. Daily Close > previous completed month's R1")
    st.write("8. Previous day's Close <= the same month's R1")
    st.caption("Monthly R1 = 2 × ((Previous Month H + L + C) / 3) − Previous Month Low")

st.info(
    "This version uses NSE India CM-UDiFF Bhavcopy Final files for OHLCV. "
    "Yahoo Finance and yfinance are not used."
)

if st.button("DOWNLOAD NSE DATA & RUN SCANNER", type="primary", use_container_width=True):
    progress = st.progress(0)
    status = st.empty()
    with st.spinner("Downloading NSE official daily bhavcopy files and building history..."):
        history, ok_days, missed_days = load_nse_history(int(lookback_days))
    progress.progress(0.55)
    status.write(f"NSE bhavcopy sessions downloaded: {len(ok_days)}")

    if history.empty:
        progress.empty()
        status.empty()
        st.error(
            "NSE bhavcopy data could not be downloaded. No Yahoo fallback is used. "
            "Please retry later; NSE archive access may be temporarily unavailable."
        )
        st.stop()

    # Keep the most recent trading dates that were successfully returned.
    latest_date = history["Date"].max()
    history = history[history["Date"] <= latest_date].copy()

    if universe == "All NSE EQ stocks":
        symbols = sorted(history["Symbol"].dropna().unique().tolist())
    else:
        # Constituents are used only to choose the universe; all price/volume
        # history still comes exclusively from NSE bhavcopy files.
        constituent_urls = {
            "NIFTY 500": "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
            "NIFTY 50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
        }
        try:
            r = requests.get(
                constituent_urls[universe],
                headers={"User-Agent": NSE_HEADERS["User-Agent"], "Referer": "https://www.niftyindices.com/"},
                timeout=20,
            )
            r.raise_for_status()
            constituent_df = pd.read_csv(io.BytesIO(r.content))
            sym_col = _find_col(constituent_df.columns, ["Symbol"])
            if sym_col is None:
                raise ValueError("Symbol column missing from index constituent file.")
            symbols = sorted(set(constituent_df[sym_col].dropna().astype(str).str.upper()) & set(history["Symbol"].unique()))
        except Exception as exc:
            st.error(f"Could not load {universe} constituent list: {exc}")
            st.stop()

    if uploaded_symbols is not None:
        try:
            custom_df = pd.read_csv(uploaded_symbols)
            sym_col = _find_col(custom_df.columns, ["Symbol"])
            if sym_col is None:
                st.error("Uploaded CSV must contain a Symbol column.")
                st.stop()
            custom = set(custom_df[sym_col].dropna().astype(str).str.strip().str.upper())
            symbols = sorted(set(symbols) & custom) if universe != "All NSE EQ stocks" else sorted(set(history["Symbol"].unique()) & custom)
        except Exception as exc:
            st.error(f"Could not read custom symbols CSV: {exc}")
            st.stop()

    progress.progress(0.65)
    status.write(f"Analyzing {len(symbols)} NSE EQ symbols using NSE-only data...")
    rows = []
    failed = []
    grouped = {s: g for s, g in history.groupby("Symbol", sort=False)}
    for i, symbol in enumerate(symbols, start=1):
        stock_history = grouped.get(symbol)
        if stock_history is None:
            failed.append(symbol)
        else:
            try:
                result = analyze_one_symbol(symbol, stock_history)
                if result is None:
                    failed.append(symbol)
                else:
                    rows.append(result)
            except Exception:
                failed.append(symbol)
        progress.progress(0.65 + 0.35 * i / max(len(symbols), 1))
        status.write(f"Analyzing {symbol} ({i}/{len(symbols)})")

    progress.empty()
    status.empty()
    if not rows:
        st.error("No symbols had enough valid NSE history to calculate all indicators.")
        st.info(f"NSE sessions available: {len(ok_days)}. Increase lookback or retry if files were missing.")
        st.stop()

    results = pd.DataFrame(rows).sort_values(
        ["Signal", "RSI 14"], ascending=[True, False]
    ).reset_index(drop=True)
    buys = results[results["Signal"] == "BUY"].copy()
    st.session_state["results"] = results
    st.session_state["buys"] = buys
    st.session_state["failed"] = failed
    st.session_state["nse_latest_date"] = latest_date.strftime("%Y-%m-%d")
    st.session_state["nse_sessions"] = len(ok_days)
    st.session_state["missed_days"] = len(missed_days)
    st.success(
        f"Scan complete using NSE data: {len(buys)} BUY candidates from "
        f"{len(results)} analyzed stocks. Latest bhavcopy: {latest_date:%d-%b-%Y}."
    )

if "results" in st.session_state:
    results = st.session_state["results"]
    buys = st.session_state["buys"]
    failed = st.session_state["failed"]

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks analyzed", len(results))
    c2.metric("BUY candidates", len(buys))
    c3.metric("NSE sessions loaded", st.session_state.get("nse_sessions", 0))
    c4.metric("Latest NSE bhavcopy", st.session_state.get("nse_latest_date", "N/A"))

    st.subheader("BUY watchlist")
    if buys.empty:
        st.info("No stocks met all nine conditions on the latest NSE bhavcopy date.")
    else:
        st.dataframe(buys, use_container_width=True, hide_index=True)
        st.download_button(
            "Download BUY watchlist CSV",
            data=buys.to_csv(index=False).encode("utf-8"),
            file_name="NSE_EOD_BUY_Watchlist.csv",
            mime="text/csv",
        )

    st.subheader("All scanned stocks")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "Download full scan CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="NSE_EOD_Full_Scan.csv",
        mime="text/csv",
    )

    st.subheader("Condition-by-condition validation")
    selected = st.selectbox("Select stock", results["Symbol"].tolist())
    row = results.loc[results["Symbol"] == selected].iloc[0]
    check_names = [
        "High > Previous High", "Close > Open", "Close > EMA 20",
        "Close > EMA 50", "Change > 2%", "Volume > SMA 5",
        "RSI > 65", "Close > Monthly R1", "Previous Close <= Monthly R1",
    ]
    metric_cols = st.columns(3)
    for idx, name in enumerate(check_names):
        metric_cols[idx % 3].metric(name, "PASS" if bool(row[name]) else "FAIL")
    st.write(f"**Signal: {row['Signal']}**")
    st.caption(
        "All OHLCV values are sourced from NSE India CM-UDiFF Common Bhavcopy Final "
        "for series EQ. No Yahoo Finance fallback is present. The first scan may take "
        "some time because NSE daily files are collected and cached. Indicator results "
        "should be checked against your charting platform before live trading."
    )
    if failed:
        with st.expander(f"Symbols with insufficient/missing NSE history ({len(failed)})"):
            st.write(", ".join(failed))
else:
    st.info("Select the NSE cash-market universe and click DOWNLOAD NSE DATA & RUN SCANNER.")
