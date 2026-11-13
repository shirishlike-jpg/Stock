import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import BytesIO
from datetime import datetime, time
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="NSE EOD CCI 200 + R1 Screener",
    page_icon="📈",
    layout="wide",
)

st.title("NSE EOD CCI 200 + Standard Pivot R1 Screener")
st.caption(
    "After-market scanner | Daily candles | CCI(14) cross above 200 | "
    "Close crosses above previous session's Standard/Classic R1 | Daily change 0% to +3%"
)

NIFTY_URLS = {
    "NIFTY 50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "NIFTY 500": "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
}

@st.cache_data(ttl=86400, show_spinner=False)
def get_nse_symbols(universe: str) -> list[str]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.niftyindices.com/",
    }
    response = requests.get(NIFTY_URLS[universe], headers=headers, timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(BytesIO(response.content))
    frame.columns = frame.columns.astype(str).str.strip()
    if "Symbol" not in frame.columns:
        raise ValueError("Constituent CSV did not contain a Symbol column.")
    return sorted(
        frame["Symbol"].dropna().astype(str).str.strip()
        .str.replace(".NS", "", regex=False).unique().tolist()
    )

def calculate_cci(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    typical = (frame["High"] + frame["Low"] + frame["Close"]) / 3.0
    average = typical.rolling(period, min_periods=period).mean()
    mean_deviation = typical.rolling(period, min_periods=period).apply(
        lambda values: np.mean(np.abs(values - np.mean(values))),
        raw=True,
    )
    denominator = 0.015 * mean_deviation
    return ((typical - average) / denominator).replace([np.inf, -np.inf], np.nan)

def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    # yfinance may return MultiIndex columns even for a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(col).strip().title() for col in data.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(data.columns):
        return pd.DataFrame()
    data = data[required].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["High", "Low", "Close"])
    data = data[(data["Close"] > 0) & (data["High"] >= data["Low"])]
    data = data[~data.index.duplicated(keep="last")].sort_index()
    return data

def analyze_stock(symbol: str, raw_frame: pd.DataFrame) -> dict | None:
    frame = normalize_ohlcv(raw_frame)
    if len(frame) < 30:
        return None

    frame["CCI_14"] = calculate_cci(frame, 14)

    # Standard / Classic Pivot R1 for session D:
    # Pivot(D) uses High, Low, Close from session D-1.
    # R1(D) = 2 * Pivot(D) - Low(D-1)
    previous_pivot = (
        frame["High"].shift(1)
        + frame["Low"].shift(1)
        + frame["Close"].shift(1)
    ) / 3.0
    frame["Previous_R1"] = 2.0 * previous_pivot - frame["Low"].shift(1)

    frame["Change_Percent"] = (
        (frame["Close"] - frame["Close"].shift(1))
        / frame["Close"].shift(1)
    ) * 100.0

    # Cross above 200: yesterday <= 200 and latest completed session > 200.
    frame["CCI_Cross"] = (
        (frame["CCI_14"].shift(1) <= 200)
        & (frame["CCI_14"] > 200)
    )

    # Close crosses above the R1 calculated from the previous session.
    # Both the prior close and current close are compared with the same R1 level.
    frame["R1_Cross"] = (
        (frame["Close"].shift(1) <= frame["Previous_R1"])
        & (frame["Close"] > frame["Previous_R1"])
    )

    frame["Price_Condition"] = frame["Change_Percent"].between(
        0.0, 3.0, inclusive="both"
    )
    frame["BUY_SIGNAL"] = (
        frame["CCI_Cross"] & frame["R1_Cross"] & frame["Price_Condition"]
    )

    latest = frame.iloc[-1]
    previous = frame.iloc[-2]

    if pd.isna(latest["CCI_14"]) or pd.isna(latest["Previous_R1"]) or pd.isna(latest["Change_Percent"]):
        return None

    return {
        "Symbol": symbol,
        "Candle Date": pd.Timestamp(frame.index[-1]).strftime("%Y-%m-%d"),
        "Close": round(float(latest["Close"]), 2),
        "Previous Close": round(float(previous["Close"]), 2),
        "Change %": round(float(latest["Change_Percent"]), 2),
        "CCI 14": round(float(latest["CCI_14"]), 2),
        "Previous Day Standard R1": round(float(latest["Previous_R1"]), 2),
        "CCI Cross Above 200": bool(latest["CCI_Cross"]),
        "Close Cross Above R1": bool(latest["R1_Cross"]),
        "Change 0% to 3%": bool(latest["Price_Condition"]),
        "Signal": "BUY" if bool(latest["BUY_SIGNAL"]) else "WAIT",
    }

@st.cache_data(ttl=900, show_spinner=False)
def download_data(symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Download in modest batches to reduce large-request failures."""
    output: dict[str, pd.DataFrame] = {}
    batch_size = 40

    for start in range(0, len(symbols), batch_size):
        batch = list(symbols[start:start + batch_size])
        tickers = [f"{symbol}.NS" for symbol in batch]
        try:
            downloaded = yf.download(
                tickers=tickers,
                period="1y",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=30,
            )
            if downloaded is None or downloaded.empty:
                continue

            if isinstance(downloaded.columns, pd.MultiIndex):
                top = downloaded.columns.get_level_values(0)
                bottom = downloaded.columns.get_level_values(1)
                for ticker in tickers:
                    if ticker in top:
                        stock_frame = downloaded[ticker].copy()
                    elif ticker in bottom:
                        stock_frame = downloaded.xs(ticker, axis=1, level=1).copy()
                    else:
                        continue
                    output[ticker.removesuffix(".NS")] = stock_frame
            elif len(tickers) == 1:
                output[batch[0]] = downloaded
        except Exception:
            # Missing symbols are reported in the results section.
            continue
    return output

def is_market_session_complete() -> bool:
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    # This is only a reminder, not an exchange-calendar validation.
    return now.time() >= time(15, 45)

with st.sidebar:
    st.header("Scanner settings")
    universe = st.selectbox("NSE universe", ["NIFTY 500", "NIFTY 50"])
    uploaded = st.file_uploader("Optional custom symbols CSV", type=["csv"])
    st.caption("Custom CSV must contain a column named Symbol.")
    st.divider()
    st.markdown("**Fixed screening rules**")
    st.write("1. CCI(14) crosses above 200")
    st.write("2. Close crosses above previous session's Standard R1")
    st.write("3. Daily change is between 0% and +3%")

if uploaded is not None:
    try:
        custom = pd.read_csv(uploaded)
        if "Symbol" not in custom.columns:
            st.error("CSV must contain a column named Symbol.")
            st.stop()
        symbols = (
            custom["Symbol"].dropna().astype(str).str.strip()
            .str.replace(".NS", "", regex=False).unique().tolist()
        )
        source_label = "Custom CSV"
    except Exception as exc:
        st.error(f"Could not read custom CSV: {exc}")
        st.stop()
else:
    try:
        symbols = get_nse_symbols(universe)
        source_label = universe
    except Exception as exc:
        st.error(f"Could not retrieve {universe} constituents: {exc}")
        st.info("Try again later or upload a CSV with a Symbol column.")
        st.stop()

left, right = st.columns([2, 1])
left.write(f"**Universe:** {source_label}  |  **Symbols:** {len(symbols)}")
right.caption("For EOD use, run after the daily candle is complete.")
if not is_market_session_complete():
    st.warning("It is before 3:45 PM IST. Today's daily candle may still be incomplete. For a next-session watchlist, run after market close and verify the candle date.")

if st.button("Run EOD Scanner", type="primary", use_container_width=True):
    progress = st.progress(0)
    status = st.empty()
    with st.spinner("Downloading daily historical candles..."):
        market_data = download_data(tuple(symbols))

    rows = []
    failed = []
    total = max(len(symbols), 1)

    for index, symbol in enumerate(symbols, start=1):
        stock_data = market_data.get(symbol)
        if stock_data is None:
            failed.append(symbol)
        else:
            try:
                result = analyze_stock(symbol, stock_data)
                if result is None:
                    failed.append(symbol)
                else:
                    rows.append(result)
            except Exception:
                failed.append(symbol)
        progress.progress(index / total)
        status.text(f"Analyzed {index}/{total}: {symbol}")

    progress.empty()
    status.empty()

    if rows:
        results = pd.DataFrame(rows).sort_values(
            ["Signal", "CCI 14"], ascending=[True, False]
        ).reset_index(drop=True)
        buys = results[results["Signal"] == "BUY"].copy()
        st.session_state["results"] = results
        st.session_state["buys"] = buys
        st.session_state["failed"] = failed
        st.success(f"Scan complete: {len(buys)} BUY candidate(s) from {len(rows)} analyzed stock(s).")
    else:
        st.error("No valid stock data was analyzed. Yahoo Finance may be unavailable or the download may have failed.")

if "results" in st.session_state:
    results = st.session_state["results"]
    buys = st.session_state["buys"]
    failed = st.session_state["failed"]

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Stocks analyzed", len(results))
    m2.metric("BUY candidates", len(buys))
    m3.metric("Missing / failed", len(failed))

    st.subheader("BUY watchlist for next trading session")
    if buys.empty:
        st.info("No stocks met all three conditions on the latest available candle.")
    else:
        st.dataframe(buys, use_container_width=True, hide_index=True)
        st.download_button(
            "Download BUY watchlist CSV",
            data=buys.to_csv(index=False).encode("utf-8"),
            file_name="NSE_EOD_CCI_R1_BUY_Watchlist.csv",
            mime="text/csv",
        )

    st.subheader("All scanned stocks")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "Download full scan CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="NSE_EOD_CCI_R1_Full_Scan.csv",
        mime="text/csv",
    )

    st.subheader("Condition-by-condition check")
    selected_symbol = st.selectbox("Choose a stock", results["Symbol"].tolist())
    selected = results.loc[results["Symbol"] == selected_symbol].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("CCI crossed above 200", "PASS" if selected["CCI Cross Above 200"] else "FAIL")
    c2.metric("Close crossed above R1", "PASS" if selected["Close Cross Above R1"] else "FAIL")
    c3.metric("Daily change 0% to 3%", "PASS" if selected["Change 0% to 3%"] else "FAIL")
    st.write(f"**Signal: {selected['Signal']}**")

    if failed:
        with st.expander(f"Missing / failed symbols ({len(failed)})"):
            st.write(", ".join(failed))

    st.caption(
        "Data source: Yahoo Finance via yfinance. Adjusted daily OHLCV is used. "
        "The latest available candle is not guaranteed to be today's completed NSE candle. "
        "Verify the date and prices before making any trading decision."
    )
else:
    st.info("Select your universe and click **Run EOD Scanner** to create the watchlist.")
