import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import BytesIO
from datetime import datetime, time
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="NSE EOD CCI 200 + Daily R1 Screener",
    page_icon="📈",
    layout="wide",
)

st.title("NSE EOD CCI 200 + Traditional Daily Pivot R1 Screener")
st.caption(
    "EOD scanner | Traditional / Classic pivots | Explicit DAILY pivot timeframe | "
    "CCI(14) cross above 200 | Close cross above daily R1 | Change 0% to +3%"
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
    constituents = pd.read_csv(BytesIO(response.content))
    constituents.columns = constituents.columns.astype(str).str.strip()
    if "Symbol" not in constituents.columns:
        raise ValueError("NSE constituent CSV has no Symbol column.")
    return sorted(
        constituents["Symbol"].dropna().astype(str).str.strip()
        .str.replace(".NS", "", regex=False).unique().tolist()
    )


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(c).strip().title() for c in data.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(data.columns):
        return pd.DataFrame()
    data = data[required].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["High", "Low", "Close"])
    data = data[
        (data["Close"] > 0)
        & (data["High"] >= data["Low"])
        & (data["High"] >= data["Close"])
        & (data["Low"] <= data["Close"])
    ]
    return data[~data.index.duplicated(keep="last")].sort_index()


def calculate_cci(data: pd.DataFrame, period: int = 14) -> pd.Series:
    typical = (data["High"] + data["Low"] + data["Close"]) / 3.0
    sma = typical.rolling(period, min_periods=period).mean()
    mean_deviation = typical.rolling(period, min_periods=period).apply(
        lambda values: np.mean(np.abs(values - np.mean(values))),
        raw=True,
    )
    result = (typical - sma) / (0.015 * mean_deviation)
    return result.replace([np.inf, -np.inf], np.nan)


def add_daily_traditional_pivots(data: pd.DataFrame) -> pd.DataFrame:
    """
    Explicit DAILY Traditional/Classic pivots.

    For trading session t, use the immediately preceding trading
    session's H/L/C:
        Pivot_t = (H_(t-1) + L_(t-1) + C_(t-1)) / 3
        R1_t = 2 * Pivot_t - L_(t-1)

    This intentionally does NOT use TradingView's Auto timeframe.
    On some chart/timeframe combinations, Auto can resolve to weekly
    pivots. This scanner explicitly uses daily pivot periods.
    """
    df = data.copy()
    previous_pivot = (
        df["High"].shift(1)
        + df["Low"].shift(1)
        + df["Close"].shift(1)
    ) / 3.0

    df["Daily_Pivot"] = previous_pivot
    df["Daily_R1"] = 2.0 * previous_pivot - df["Low"].shift(1)
    df["Prior_Session_Daily_R1"] = df["Daily_R1"].shift(1)
    return df


def analyze_stock(symbol: str, raw: pd.DataFrame) -> dict | None:
    df = normalize_ohlcv(raw)
    if len(df) < 35:
        return None

    df["CCI_14"] = calculate_cci(df, 14)
    df = add_daily_traditional_pivots(df)

    df["Change_Percent"] = (
        (df["Close"] - df["Close"].shift(1))
        / df["Close"].shift(1)
    ) * 100.0

    # Condition 1: CCI crosses from <= 200 to > 200.
    df["CCI_Cross"] = (
        (df["CCI_14"].shift(1) <= 200)
        & (df["CCI_14"] > 200)
    )

    # Condition 2: close crosses above the daily R1 series.
    # Yesterday's close is compared with yesterday's daily R1;
    # today's close is compared with today's daily R1.
    df["R1_Cross"] = (
        (df["Close"].shift(1) <= df["Prior_Session_Daily_R1"])
        & (df["Close"] > df["Daily_R1"])
    )

    # Condition 3: daily close-to-close return in [0%, +3%].
    df["Price_Condition"] = df["Change_Percent"].between(
        0.0, 3.0, inclusive="both"
    )

    df["BUY_SIGNAL"] = (
        df["CCI_Cross"] & df["R1_Cross"] & df["Price_Condition"]
    )

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    needed = [
        latest["CCI_14"],
        latest["Daily_Pivot"],
        latest["Daily_R1"],
        latest["Prior_Session_Daily_R1"],
        latest["Change_Percent"],
    ]
    if any(pd.isna(v) for v in needed):
        return None

    candle_date = pd.Timestamp(df.index[-1])
    if candle_date.tzinfo is not None:
        candle_date = candle_date.tz_convert("Asia/Kolkata").tz_localize(None)

    return {
        "Symbol": symbol,
        "Candle Date": candle_date.strftime("%Y-%m-%d"),
        "Previous Close": round(float(previous["Close"]), 2),
        "Latest Close": round(float(latest["Close"]), 2),
        "Change %": round(float(latest["Change_Percent"]), 2),
        "CCI 14": round(float(latest["CCI_14"]), 2),
        "Previous Day High": round(float(previous["High"]), 2),
        "Previous Day Low": round(float(previous["Low"]), 2),
        "Previous Day Close": round(float(previous["Close"]), 2),
        "Today's Daily Pivot": round(float(latest["Daily_Pivot"]), 2),
        "Today's Daily R1": round(float(latest["Daily_R1"]), 2),
        "Previous Session Daily R1": round(float(latest["Prior_Session_Daily_R1"]), 2),
        "Close - Today's R1": round(float(latest["Close"] - latest["Daily_R1"]), 2),
        "CCI Cross Above 200": bool(latest["CCI_Cross"]),
        "Close Cross Above Daily R1": bool(latest["R1_Cross"]),
        "Change 0% to 3%": bool(latest["Price_Condition"]),
        "Signal": "BUY" if bool(latest["BUY_SIGNAL"]) else "WAIT",
    }


@st.cache_data(ttl=900, show_spinner=False)
def download_market_data(symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    batch_size = 40

    for start in range(0, len(symbols), batch_size):
        batch = list(symbols[start:start + batch_size])
        tickers = [f"{symbol}.NS" for symbol in batch]
        try:
            data = yf.download(
                tickers=tickers,
                period="1y",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=30,
            )
            if data is None or data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                level0 = data.columns.get_level_values(0)
                level1 = data.columns.get_level_values(1)
                for ticker in tickers:
                    try:
                        if ticker in level0:
                            one = data[ticker].copy()
                        elif ticker in level1:
                            one = data.xs(ticker, axis=1, level=1).copy()
                        else:
                            continue
                        output[ticker[:-3]] = one
                    except Exception:
                        continue
            elif len(batch) == 1:
                output[batch[0]] = data
        except Exception:
            continue

    return output


def after_market_close() -> bool:
    return datetime.now(ZoneInfo("Asia/Kolkata")).time() >= time(15, 45)


with st.sidebar:
    st.header("Scanner Settings")
    universe = st.selectbox("NSE Universe", ["NIFTY 500", "NIFTY 50"])
    upload = st.file_uploader("Optional custom symbols CSV", type=["csv"])
    st.caption("CSV must contain a column named Symbol, e.g. RELIANCE.")
    st.divider()
    st.markdown("**Pivot configuration**")
    st.write("Type: Traditional / Classic")
    st.write("Timeframe: DAILY (explicit, not Auto)")
    st.write("R1 = 2 × Daily Pivot − Previous Day Low")
    st.divider()
    st.markdown("**BUY conditions**")
    st.write("1. CCI(14) crosses above 200")
    st.write("2. Close crosses above the daily R1 series")
    st.write("3. Daily change is 0% to +3%")


if upload is not None:
    try:
        custom = pd.read_csv(upload)
        if "Symbol" not in custom.columns:
            st.error("CSV must contain a 'Symbol' column.")
            st.stop()
        symbols = (
            custom["Symbol"].dropna().astype(str).str.strip()
            .str.replace(".NS", "", regex=False).unique().tolist()
        )
        source = "Custom CSV"
    except Exception as exc:
        st.error(f"Unable to read CSV: {exc}")
        st.stop()
else:
    try:
        symbols = get_nse_symbols(universe)
        source = universe
    except Exception as exc:
        st.error(f"Unable to fetch {universe} constituents: {exc}")
        st.info("Try again later or upload a CSV containing a Symbol column.")
        st.stop()

top1, top2 = st.columns([2, 1])
top1.write(f"**Universe:** {source}  |  **Symbols:** {len(symbols)}")
top2.caption("Recommended scan time: after 3:45 PM IST.")

if not after_market_close():
    st.warning(
        "It is before 3:45 PM IST. Today's daily candle may be incomplete. "
        "Run after the NSE market session has closed."
    )

if st.button("RUN EOD SCANNER", type="primary", use_container_width=True):
    progress = st.progress(0)
    status = st.empty()

    with st.spinner("Downloading daily OHLCV candles from Yahoo Finance..."):
        market_data = download_market_data(tuple(symbols))

    rows = []
    failed = []
    total = max(len(symbols), 1)

    for i, symbol in enumerate(symbols, start=1):
        raw = market_data.get(symbol)
        if raw is None:
            failed.append(symbol)
        else:
            try:
                result = analyze_stock(symbol, raw)
                if result is None:
                    failed.append(symbol)
                else:
                    rows.append(result)
            except Exception:
                failed.append(symbol)
        progress.progress(i / total)
        status.text(f"Analyzing {symbol} ({i}/{total})")

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
        st.success(
            f"Scan complete: {len(buys)} BUY candidate(s) "
            f"from {len(results)} analyzed stocks."
        )
    else:
        st.error("No valid data was analyzed. Yahoo Finance may be unavailable.")


if "results" in st.session_state:
    results = st.session_state["results"]
    buys = st.session_state["buys"]
    failed = st.session_state["failed"]

    st.divider()
    a, b, c = st.columns(3)
    a.metric("Stocks analyzed", len(results))
    b.metric("BUY candidates", len(buys))
    c.metric("Missing / failed", len(failed))

    st.subheader("BUY watchlist for next trading session")
    if buys.empty:
        st.info("No stocks met all three conditions.")
    else:
        st.dataframe(buys, use_container_width=True, hide_index=True)
        st.download_button(
            "Download BUY watchlist CSV",
            data=buys.to_csv(index=False).encode("utf-8"),
            file_name="NSE_EOD_CCI_Daily_R1_BUY.csv",
            mime="text/csv",
        )

    st.subheader("All scanned stocks")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "Download full scan CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="NSE_EOD_CCI_Daily_R1_Full.csv",
        mime="text/csv",
    )

    st.subheader("Condition-by-condition validation")
    chosen = st.selectbox("Select stock", results["Symbol"].tolist())
    row = results.loc[results["Symbol"] == chosen].iloc[0]
    x, y, z = st.columns(3)
    x.metric("CCI crossed above 200", "PASS" if row["CCI Cross Above 200"] else "FAIL")
    y.metric("Close crossed above daily R1", "PASS" if row["Close Cross Above Daily R1"] else "FAIL")
    z.metric("Daily change 0% to +3%", "PASS" if row["Change 0% to 3%"] else "FAIL")

    st.write(f"**Signal: {row['Signal']}**")
    st.caption(
        "Check the displayed candle date and daily OHLC values. "
        "Data is sourced from Yahoo Finance and may be delayed, missing, "
        "or different from TradingView's adjustment/session settings. "
        "This is a research scanner, not an execution recommendation."
    )

    if failed:
        with st.expander(f"Missing / failed symbols ({len(failed)})"):
            st.write(", ".join(failed))
else:
    st.info("Choose a universe and click RUN EOD SCANNER.")
