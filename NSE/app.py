    # Refresh options
    st.markdown("### 🔄 Data Refresh")
    
    # Refresh All button
    if st.button("🔄 Refresh All Data", type="primary", **FULL_WIDTH):
        progress_area = st.empty()
        results = download_html_pages(TICKERS, progress_area)
        ok = sum(1 for v in results.values() if v)
        st.success(f"Downloaded {ok}/{len(TICKERS)} stocks")
        st.cache_data.clear()
        st.session_state["last_refresh"] = datetime.now()
        time.sleep(0.5)
        st.rerun()
    
    # Selective refresh
    st.markdown("**OR**")
    selected_tickers = st.multiselect(
        "Select specific tickers to refresh",
        options=TICKERS,
        default=[],
        help="Choose one or more tickers to refresh only those"
    )
    
    if st.button("🔄 Refresh Selected", disabled=len(selected_tickers) == 0, **FULL_WIDTH):
        progress_area = st.empty()
        results = download_html_pages(selected_tickers, progress_area)
        ok = sum(1 for v in results.values() if v)
        st.success(f"Downloaded {ok}/{len(selected_tickers)} selected stocks")
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
