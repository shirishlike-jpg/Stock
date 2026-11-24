import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import BytesIO
from datetime import datetime, time
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="NSE EOD Monthly R1 Breakout Screener",
    page_icon="📈",
    layout="wide",
)

st.title("NSE EOD Monthly R1 Breakout Screener")
st.caption(
    "Cash-market EOD scanner | Previous completed month's Traditional R1 | "
    "EMA 20/50 | RSI 14 | Volume SMA 5"
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
    data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    data = data[
        (data["Open"] > 0)
        & (data["Close"] > 0)
        & (data["High"] >= data["Low"])
        & (data["High"] >= data["Open"])
        & (data["High"] >= data["Close"])
        & (data["Low"] <= data["Open"])
        & (data["Low"] <= data["Close"])
    ]
    data = data[~data.index.duplicated(keep="last")].sort_index()

    # Use timezone-naive dates after converting Yahoo timestamps to India time.
    idx = pd.to_datetime(data.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    data.index = idx.normalize()
    return data


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's RSI (RMA smoothing), matching the common charting convention.
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
    return rsi


def add_previous_month_r1(data: pd.DataFrame) -> pd.DataFrame:
    """
    For every daily candle in a calendar month, calculate R1 from the
    immediately preceding COMPLETED calendar month's H/L/C.

    Monthly Pivot = (Previous Month High + Low + Close) / 3
    Monthly R1 = 2 * Monthly Pivot - Previous Month Low
    """
    df = data.copy()
    month_key = df.index.to_period("M")

    monthly = df.resample("ME").agg(
        {"High": "max", "Low": "min", "Close": "last"}
    ).dropna()

    monthly["Monthly_Pivot"] = (
        monthly["High"] + monthly["Low"] + monthly["Close"]
    ) / 3.0
    monthly["Monthly_R1"] = 2.0 * monthly["Monthly_Pivot"] - monthly["Low"]

    # Shift one completed monthly value forward, then map it to daily rows.
    monthly["R1_For_Next_Month"] = monthly["Monthly_R1"].shift(1)
    monthly.index = monthly.index.to_period("M")
    r1_map = monthly["R1_For_Next_Month"]
    df["Monthly_R1"] = month_key.map(r1_map.to_dict())
    return df


def analyze_stock(symbol: str, raw: pd.DataFrame) -> dict | None:
    df = normalize_ohlcv(raw)
    if len(df) < 70:
        return None

    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False, min_periods=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False, min_periods=50).mean()
    df["RSI_14"] = calculate_rsi(df["Close"], 14)
    df["Volume_SMA_5"] = df["Volume"].rolling(5, min_periods=5).mean()
    df = add_previous_month_r1(df)

    df["Change_Percent"] = (
        (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)
    ) * 100.0

    # User's updated screener conditions.
    df["High_Above_Prior_High"] = df["High"] > df["High"].shift(1)
    df["Bullish_Candle"] = df["Close"] > df["Open"]
    df["Close_Above_EMA20"] = df["Close"] > df["EMA_20"]
    df["Close_Above_EMA50"] = df["Close"] > df["EMA_50"]
    df["Change_Above_2_Percent"] = df["Change_Percent"] > 2.0
    df["Volume_Above_SMA5"] = df["Volume"] > df["Volume_SMA_5"]
    df["RSI_Above_65"] = df["RSI_14"] > 65.0
    df["Close_Above_Monthly_R1"] = df["Close"] > df["Monthly_R1"]

    # Yesterday's close must be at/below the same current-month R1 level.
    df["Prior_Close_Below_Monthly_R1"] = (
        df["Close"].shift(1) <= df["Monthly_R1"]
    )

    condition_columns = [
        "High_Above_Prior_High",
        "Bullish_Candle",
        "Close_Above_EMA20",
        "Close_Above_EMA50",
        "Change_Above_2_Percent",
        "Volume_Above_SMA5",
        "RSI_Above_65",
        "Close_Above_Monthly_R1",
        "Prior_Close_Below_Monthly_R1",
    ]
    df["BUY_SIGNAL"] = df[condition_columns].all(axis=1)

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    required_values = [
        latest["EMA_20"],
        latest["EMA_50"],
        latest["RSI_14"],
        latest["Volume_SMA_5"],
        latest["Monthly_R1"],
        latest["Change_Percent"],
    ]
    if any(pd.isna(value) for value in required_values):
        return None

    candle_date = pd.Timestamp(df.index[-1])
    return {
        "Symbol": symbol,
        "Candle Date": candle_date.strftime("%Y-%m-%d"),
        "Previous Close": round(float(previous["Close"]), 2),
        "Latest Open": round(float(latest["Open"]), 2),
        "Latest High": round(float(latest["High"]), 2),
        "Latest Low": round(float(latest["Low"]), 2),
        "Latest Close": round(float(latest["Close"]), 2),
        "Change %": round(float(latest["Change_Percent"]), 2),
        "Previous Day High": round(float(previous["High"]), 2),
        "EMA 20": round(float(latest["EMA_20"]), 2),
        "EMA 50": round(float(latest["EMA_50"]), 2),
        "Volume": int(latest["Volume"]),
        "Volume SMA 5": round(float(latest["Volume_SMA_5"]), 0),
        "RSI 14": round(float(latest["RSI_14"]), 2),
        "Previous Completed Month R1": round(float(latest["Monthly_R1"]), 2),
        "Close - Monthly R1": round(
            float(latest["Close"] - latest["Monthly_R1"]), 2
        ),
        "High > Previous High": bool(latest["High_Above_Prior_High"]),
        "Close > Open": bool(latest["Bullish_Candle"]),
        "Close > EMA 20": bool(latest["Close_Above_EMA20"]),
        "Close > EMA 50": bool(latest["Close_Above_EMA50"]),
        "Change > 2%": bool(latest["Change_Above_2_Percent"]),
        "Volume > SMA 5": bool(latest["Volume_Above_SMA5"]),
        "RSI > 65": bool(latest["RSI_Above_65"]),
        "Close > Monthly R1": bool(latest["Close_Above_Monthly_R1"]),
        "Previous Close <= Monthly R1": bool(
            latest["Prior_Close_Below_Monthly_R1"]
        ),
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
                period="2y",
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
    st.markdown("**BUY conditions**")
    st.write("1. Daily High > previous day's High")
    st.write("2. Daily Close > Daily Open")
    st.write("3. Daily Close > EMA 20 and EMA 50")
    st.write("4. Daily change > 2%")
    st.write("5. Daily Volume > Volume SMA 5")
    st.write("6. Daily RSI 14 > 65")
    st.write("7. Daily Close > previous completed month's R1")
    st.write("8. Previous day's Close <= that month's R1")
    st.caption(
        "Monthly R1 = 2 × ((Previous Month H + L + C) / 3) "
        "− Previous Month Low"
    )


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

symbols = [s for s in symbols if s and s.lower() != "nan"]

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
            ["Signal", "RSI 14"], ascending=[True, False]
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

    st.subheader("BUY watchlist")
    if buys.empty:
        st.info("No stocks met all nine conditions.")
    else:
        st.dataframe(buys, use_container_width=True, hide_index=True)
        st.download_button(
            "Download BUY watchlist CSV",
            data=buys.to_csv(index=False).encode("utf-8"),
            file_name="NSE_EOD_Monthly_R1_BUY.csv",
            mime="text/csv",
        )

    st.subheader("All scanned stocks")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "Download full scan CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="NSE_EOD_Monthly_R1_Full.csv",
        mime="text/csv",
    )

    st.subheader("Condition-by-condition validation")
    chosen = st.selectbox("Select stock", results["Symbol"].tolist())
    row = results.loc[results["Symbol"] == chosen].iloc[0]

    # Display labels are mapped to the exact column names returned by analyze_stock().
    # This avoids KeyError caused by looking up a display label that is not a DataFrame column.
    condition_checks = [
        ("High > Previous High", "High > Previous High"),
        ("Close > Open", "Close > Open"),
        ("Close > EMA 20", "Close > EMA 20"),
        ("Close > EMA 50", "Close > EMA 50"),
        ("Change > 2%", "Change > 2%"),
        ("Volume > SMA 5", "Volume > SMA 5"),
        ("RSI > 65", "RSI > 65"),
        ("Close > Monthly R1", "Close > Monthly R1"),
        ("Previous Close <= Monthly R1", "Previous Close <= Monthly R1"),
    ]
    cols = st.columns(3)
    for i, (label, column_name) in enumerate(condition_checks):
        if column_name not in row.index:
            cols[i % 3].metric(label, "N/A")
        else:
            cols[i % 3].metric(
                label,
                "PASS" if bool(row[column_name]) else "FAIL",
            )

    st.write(f"**Signal: {row['Signal']}**")
    st.caption(
        "Monthly R1 uses the previous completed calendar month's H/L/C and "
        "is held constant throughout the current month. The previous day's "
        "close is compared with the same current-month R1. Data is sourced "
        "from Yahoo Finance, may be delayed or missing, and uses auto-adjusted "
        "prices. Verify signals against your chart/data provider before trading. "
        "This is a research scanner, not an execution recommendation."
    )

    if failed:
        with st.expander(f"Missing / failed symbols ({len(failed)})"):
            st.write(", ".join(failed))
else:
    st.info("Choose a universe and click RUN EOD SCANNER.")
