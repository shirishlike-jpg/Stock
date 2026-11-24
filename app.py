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
