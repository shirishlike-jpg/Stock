# NSE EOD CCI 200 + Standard Pivot R1 Screener

A Streamlit end-of-day (EOD) scanner for NSE stocks. It does not require a broker API.

## Screening rules

All three conditions must be true on the latest available daily candle:

1. **CCI(14) crosses above 200**: previous CCI <= 200 and current CCI > 200.
2. **Close crosses above previous session's Standard/Classic Pivot R1**:
   - Previous session pivot = (previous High + previous Low + previous Close) / 3
   - R1 = (2 × pivot) - previous Low
   - Previous close <= that R1 and current close > that same R1.
3. **Daily percentage change** is between 0% and +3%, inclusive.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and the `.streamlit/config.toml` file.
3. In Streamlit Community Cloud, connect the repository.
4. Set the main file path to `app.py` and deploy.

No API key or Streamlit secrets are required.

## Data notes

- Data is downloaded from Yahoo Finance using `yfinance`; NSE symbols are suffixed with `.NS`.
- The app uses daily adjusted OHLCV data and is intended for after-market research.
- Yahoo Finance data availability, timestamps, and completeness are not guaranteed. Check the displayed candle date and independently verify prices.
- Run after the NSE session is complete. On holidays, the latest available candle will be the previous trading session.
- This is a screening tool, not a trade execution system or investment advice. Backtest and paper trade before risking capital.
