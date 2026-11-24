
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
