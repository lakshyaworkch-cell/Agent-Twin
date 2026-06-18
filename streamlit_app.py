# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI  —  Agent Twin v3.1
# Paste this entire block at the END of your existing agent_twin_v3.py file,
# replacing the "st.write('App loaded successfully')" test line you added.
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _regime_pill(regime: str) -> str:
    color = REGIME_COLORS.get(regime, C["text_dim"])
    return (f'<span class="regime-pill" '
            f'style="background:{color}22;color:{color};border:1px solid {color}55;">'
            f'{regime}</span>')

def _delta_color(v: float) -> str:
    if v > 0.5:  return C["green"]
    if v < -0.5: return C["red"]
    return C["text_dim"]

def _fmt_pct(v: float, decimals: int = 1) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{decimals}f}%"

def _metric_html(label: str, value: str, color: str = None) -> str:
    col = color or C["text"]
    return (f'<div><div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{col};">{value}</div></div>')

def _sparkline_fig(series: list, color: str, height: int = 80) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=series, mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba")
             if "rgb" in color else color + "14",
    ))
    fig.update_layout(
        **LAYOUT,
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="at-header">🏛 Agent Twin</div>', unsafe_allow_html=True)
    st.markdown('<div class="at-sub">v3.1 · Institutional Market Simulator</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Scenario**")
    scenario_name = st.selectbox(
        "Preset", list(SCENARIOS.keys()), label_visibility="collapsed")

    st.markdown("**Simulation parameters**")
    periods   = st.slider("Periods (months)", 24, 200, 100, 12)
    mc_seeds  = st.slider("Monte Carlo seeds", 20, 200, 60, 10)
    rand_seed = st.number_input("Random seed", value=42, step=1)

    st.markdown("---")
    st.markdown("**Override macro**")
    sc = SCENARIOS[scenario_name]
    inflation     = st.slider("Inflation (%)",      -2.0, 15.0, float(sc["inflation"]),     0.1)
    interest_rate = st.slider("Interest rate (%)",   0.0, 12.0, float(sc["interest_rate"]), 0.1)
    gdp_growth    = st.slider("GDP growth (%)",    -10.0, 10.0, float(sc["gdp_growth"]),    0.1)
    oil_shock     = st.slider("Oil shock (%)",     -50.0,150.0, float(sc["oil_shock"]),     1.0)
    sentiment     = st.select_slider("Sentiment", SENTIMENT_LEVELS,
                                     value=sc["sentiment"])
    regime        = st.selectbox("Starting regime", REGIMES,
                                 index=REGIMES.index(sc["regime"]))

    st.markdown("---")
    run_btn = st.button("▶  Run Simulation", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────

if "engine" not in st.session_state:
    st.session_state.engine    = None
    st.session_state.mc        = None
    st.session_state.ablation  = None
    st.session_state.last_sc   = None

if run_btn:
    base_env = MacroEnvironment(
        inflation=inflation, interest_rate=interest_rate,
        gdp_growth=gdp_growth, oil_shock=oil_shock,
        sentiment=sentiment, regime=regime,
    )
    with st.spinner("Running simulation…"):
        eng = SimulationEngine(base_env, periods=periods, seed=int(rand_seed))
        eng.run()
        st.session_state.engine   = eng
        st.session_state.last_sc  = scenario_name

    with st.spinner("Running Monte Carlo…"):
        st.session_state.mc = run_monte_carlo(base_env, periods, n_seeds=mc_seeds)

    with st.spinner("Running agent ablation…"):
        st.session_state.ablation = run_agent_ablation(base_env, periods, seed=int(rand_seed))

# ──────────────────────────────────────────────────────────────────────────────
# LANDING SCREEN (no run yet)
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.engine is None:
    st.markdown('<div class="at-header">🏛 Agent Twin v3.1</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="at-sub">Institutional Market Simulator — '
        'heterogeneous agents · realistic price formation · return attribution</div>',
        unsafe_allow_html=True)
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="panel"><div class="metric-label">Agents</div>'
            '<div style="font-family:IBM Plex Mono,monospace;margin-top:.4rem;">'
            '🏦 Pension Fund<br>📈 Hedge Fund<br>👤 Retail Investor</div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="panel"><div class="metric-label">Asset classes</div>'
            '<div style="font-family:IBM Plex Mono,monospace;margin-top:.4rem;">'
            '📊 Stocks<br>💵 Bonds<br>🪙 Gold</div></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            '<div class="panel"><div class="metric-label">Analysis modules</div>'
            '<div style="font-family:IBM Plex Mono,monospace;margin-top:.4rem;">'
            '🔬 Return attribution<br>🎲 Monte Carlo<br>🔩 Agent ablation</div></div>',
            unsafe_allow_html=True)
    st.info("👈  Select a scenario in the sidebar and click **Run Simulation** to begin.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# LIVE ENGINE + MC RESULTS
# ──────────────────────────────────────────────────────────────────────────────

eng: SimulationEngine  = st.session_state.engine
mc:  Dict              = st.session_state.mc
abl: Dict              = st.session_state.ablation
scene_name             = st.session_state.last_sc

market   = eng.market
pension  = next(a for a in eng.agents if isinstance(a, PensionFund))
hedge    = next(a for a in eng.agents if isinstance(a, HedgeFund))
retail   = next(a for a in eng.agents if isinstance(a, RetailInvestor))
agent_names = [a.name for a in eng.agents]

sr = market.assets["Stocks"].total_return_pct()
br = market.assets["Bonds"].total_return_pct()
gr = market.assets["Gold"].total_return_pct()

stocks_prices = market.assets["Stocks"].price_history
bonds_prices  = market.assets["Bonds"].price_history
gold_prices   = market.assets["Gold"].price_history

confidence  = compute_confidence(mc)
eco_dom     = compute_economic_dominance(eng)
ct          = run_consistency_tests(mc, scene_name)
attr_df     = compute_attribution_table(eng, "Stocks")
attr_summ   = summarize_attribution(attr_df, agent_names)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ──────────────────────────────────────────────────────────────────────────────

h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(f'<div class="at-header">🏛 {scene_name}</div>', unsafe_allow_html=True)
    regime_counts = pd.Series(market.regime_history).value_counts()
    pills = " ".join(_regime_pill(r) for r in regime_counts.index[:4])
    st.markdown(f'<div class="at-sub">{periods} periods · {pills}</div>',
                unsafe_allow_html=True)
with h2:
    conf_html = (f'<div style="text-align:right;padding-top:.5rem;">'
                 f'<span class="metric-label">Confidence </span>'
                 f'<span class="{confidence["label_class"]}">{confidence["level"]}</span>'
                 f'</div>')
    st.markdown(conf_html, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ──────────────────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5, k6 = st.columns(6)
stocks_vol = pd.Series(stocks_prices).pct_change().std() * 100
stocks_dd  = ((pd.Series(stocks_prices) / pd.Series(stocks_prices).cummax()) - 1).min() * 100
final_fr   = pension.funding_ratio_history[-1] if pension.funding_ratio_history else 100.0
max_lev    = max(hedge.leverage_ratio_history) if hedge.leverage_ratio_history else 1.0

for col, label, val, color in [
    (k1, "Stocks",        _fmt_pct(sr),       C["stock"]),
    (k2, "Bonds",         _fmt_pct(br),        C["bond"]),
    (k3, "Gold",          _fmt_pct(gr),         C["gold"]),
    (k4, "Equity Vol",    f"{stocks_vol:.2f}%/p", C["text"]),
    (k5, "Max Drawdown",  _fmt_pct(stocks_dd),  C["red"]),
    (k6, "Pension FR",    f"{final_fr:.1f}%",   C["green"] if final_fr >= 100 else C["amber"]),
]:
    with col:
        st.markdown(f'<div class="panel">{_metric_html(label, val, color)}</div>',
                    unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📈 Markets",
    "🤖 Agents",
    "🔬 Attribution",
    "🎲 Monte Carlo",
    "🔩 Agent Impact",
    "📋 Event Log",
    "🩺 Diagnostics",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKETS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown("### Price history")

    # Price chart
    fig = go.Figure()
    for name, color in [("Stocks", C["stock"]), ("Bonds", C["bond"]), ("Gold", C["gold"])]:
        ph = market.assets[name].price_history
        fv = market.assets[name].fair_value_history
        fig.add_trace(go.Scatter(
            x=list(range(len(ph))), y=ph,
            name=name, line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(
            x=list(range(len(fv))), y=fv,
            name=f"{name} Fair Value",
            line=dict(color=color, width=1, dash="dot"),
            opacity=0.5))
    fig.update_layout(**LAYOUT, height=340, title="Price vs Fair Value (base=100)",
                      legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig, use_container_width=True)

    # Regime shading overlay
    st.markdown("### Macro environment")
    c1, c2 = st.columns(2)

    with c1:
        fig2 = go.Figure()
        env_hist = market.env_history
        periods_x = list(range(len(env_hist)))
        fig2.add_trace(go.Scatter(x=periods_x, y=[e.inflation for e in env_hist],
                                  name="Inflation", line=dict(color=C["red"], width=1.5)))
        fig2.add_trace(go.Scatter(x=periods_x, y=[e.interest_rate for e in env_hist],
                                  name="Interest Rate", line=dict(color=C["blue"], width=1.5)))
        fig2.add_trace(go.Scatter(x=periods_x, y=[e.gdp_growth for e in env_hist],
                                  name="GDP Growth", line=dict(color=C["green"], width=1.5)))
        fig2.update_layout(**LAYOUT, height=260, title="Macro indicators",
                           legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        fig3 = go.Figure()
        regime_hist = market.regime_history
        for regime_name, color in REGIME_COLORS.items():
            probs = [rp.get(regime_name, 0) for rp in eng.regime_prob_history]
            fig3.add_trace(go.Scatter(
                x=list(range(len(probs))), y=probs,
                name=regime_name, stackgroup="one",
                line=dict(color=color, width=0),
                fillcolor=color + "88"))
        fig3.update_layout(**LAYOUT, height=260, title="Regime probability",
                           legend=dict(orientation="h", y=-0.22),
                           yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig3, use_container_width=True)

    # Rolling correlations
    st.markdown("### Rolling cross-asset correlation (20-period)")
    sp = pd.Series(stocks_prices).pct_change()
    bp = pd.Series(bonds_prices).pct_change()
    gp = pd.Series(gold_prices).pct_change()
    sb_corr = sp.rolling(20).corr(bp).dropna()
    sg_corr = sp.rolling(20).corr(gp).dropna()
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=sb_corr.tolist(), name="Stocks–Bonds",
                              line=dict(color=C["blue"], width=1.5)))
    fig4.add_trace(go.Scatter(y=sg_corr.tolist(), name="Stocks–Gold",
                              line=dict(color=C["gold"], width=1.5)))
    fig4.add_hline(y=0, line=dict(color=C["border"], dash="dot"))
    fig4.update_layout(**LAYOUT, height=220, legend=dict(orientation="h", y=-0.22))
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AGENTS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    a1, a2, a3 = st.columns(3)

    # ── Pension Fund ──
    with a1:
        st.markdown(f'<div style="color:{C["pension"]};font-family:IBM Plex Mono,monospace;'
                    f'font-weight:700;margin-bottom:.5rem;">🏦 Pension Fund</div>',
                    unsafe_allow_html=True)
        fr_hist = pension.funding_ratio_history
        if fr_hist:
            fig_fr = go.Figure()
            fig_fr.add_trace(go.Scatter(y=fr_hist, mode="lines",
                                        line=dict(color=C["pension"], width=1.5), fill="tozeroy",
                                        fillcolor=C["pension"] + "18"))
            fig_fr.add_hline(y=100, line=dict(color=C["amber"], dash="dot", width=1))
            fig_fr.update_layout(**LAYOUT, height=160, title="Funding ratio (%)",
                                 margin=dict(l=32, r=8, t=32, b=8))
            st.plotly_chart(fig_fr, use_container_width=True)

        alloc_h = pension.allocation_history
        if alloc_h:
            fig_pa = go.Figure()
            for asset, color in [("Stocks", C["stock"]), ("Bonds", C["bond"]), ("Gold", C["gold"])]:
                fig_pa.add_trace(go.Scatter(
                    y=[a.get(asset, 0) for a in alloc_h],
                    name=asset, stackgroup="one",
                    line=dict(color=color, width=0), fillcolor=color + "88"))
            fig_pa.update_layout(**LAYOUT, height=160, title="Allocation",
                                 legend=dict(orientation="h", y=-0.28),
                                 margin=dict(l=32, r=8, t=32, b=8))
            st.plotly_chart(fig_pa, use_container_width=True)

        pv = pension.portfolio_value_history
        if len(pv) > 1:
            pv_ret = (pv[-1] / pv[0] - 1) * 100
            st.markdown(_metric_html("Portfolio return", _fmt_pct(pv_ret),
                                     C["green"] if pv_ret > 0 else C["red"]),
                        unsafe_allow_html=True)

    # ── Hedge Fund ──
    with a2:
        st.markdown(f'<div style="color:{C["hedge"]};font-family:IBM Plex Mono,monospace;'
                    f'font-weight:700;margin-bottom:.5rem;">📈 Hedge Fund</div>',
                    unsafe_allow_html=True)

        fig_lev = go.Figure()
        fig_lev.add_trace(go.Scatter(
            y=hedge.gross_exposure_history, name="Gross",
            line=dict(color=C["hedge"], width=1.5)))
        fig_lev.add_trace(go.Scatter(
            y=hedge.net_exposure_history, name="Net",
            line=dict(color=C["amber"], width=1.5, dash="dash")))
        fig_lev.add_hline(y=1, line=dict(color=C["border"], dash="dot"))
        fig_lev.update_layout(**LAYOUT, height=160, title="Leverage (gross/net)",
                              legend=dict(orientation="h", y=-0.28),
                              margin=dict(l=32, r=8, t=32, b=8))
        st.plotly_chart(fig_lev, use_container_width=True)

        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(
            y=hedge.trend_signal_history, name="Trend",
            line=dict(color=C["green"], width=1.5)))
        fig_sig.add_trace(go.Scatter(
            y=hedge.valuation_signal_history, name="Valuation",
            line=dict(color=C["blue"], width=1.5)))
        fig_sig.add_hline(y=0, line=dict(color=C["border"], dash="dot"))
        fig_sig.update_layout(**LAYOUT, height=160, title="Signals",
                              legend=dict(orientation="h", y=-0.28),
                              margin=dict(l=32, r=8, t=32, b=8))
        st.plotly_chart(fig_sig, use_container_width=True)

        hv = hedge.portfolio_value_history
        if len(hv) > 1:
            hv_ret = (hv[-1] / hv[0] - 1) * 100
            st.markdown(_metric_html("Portfolio return", _fmt_pct(hv_ret),
                                     C["green"] if hv_ret > 0 else C["red"]),
                        unsafe_allow_html=True)

    # ── Retail ──
    with a3:
        st.markdown(f'<div style="color:{C["retail"]};font-family:IBM Plex Mono,monospace;'
                    f'font-weight:700;margin-bottom:.5rem;">👤 Retail Investor</div>',
                    unsafe_allow_html=True)

        fig_fg = go.Figure()
        fig_fg.add_trace(go.Scatter(
            y=retail.greed_history, name="Greed",
            line=dict(color=C["green"], width=1.5)))
        fig_fg.add_trace(go.Scatter(
            y=retail.fear_history, name="Fear",
            line=dict(color=C["red"], width=1.5)))
        fig_fg.update_layout(**LAYOUT, height=160, title="Fear / Greed",
                             legend=dict(orientation="h", y=-0.28),
                             margin=dict(l=32, r=8, t=32, b=8))
        st.plotly_chart(fig_fg, use_container_width=True)

        alloc_r = retail.allocation_history
        if alloc_r:
            fig_ra = go.Figure()
            for asset, color in [("Stocks", C["stock"]), ("Cash", C["text_dim"])]:
                fig_ra.add_trace(go.Scatter(
                    y=[a.get(asset, 0) for a in alloc_r],
                    name=asset, stackgroup="one",
                    line=dict(color=color, width=0), fillcolor=color + "88"))
            fig_ra.update_layout(**LAYOUT, height=160, title="Allocation",
                                 legend=dict(orientation="h", y=-0.28),
                                 margin=dict(l=32, r=8, t=32, b=8))
            st.plotly_chart(fig_ra, use_container_width=True)

        rv = retail.portfolio_value_history
        if len(rv) > 1:
            rv_ret = (rv[-1] / rv[0] - 1) * 100
            st.markdown(_metric_html("Portfolio return", _fmt_pct(rv_ret),
                                     C["green"] if rv_ret > 0 else C["red"]),
                        unsafe_allow_html=True)

    # ── Decision rules reference ──
    st.markdown("---")
    st.markdown("### Agent decision rules (reference)")
    dr1, dr2, dr3 = st.columns(3)
    with dr1:
        st.markdown(
            '<div class="panel-dark"><div class="metric-label" style="color:#3b82a6;">Pension Fund</div>'
            '<div class="rule-row">FR &lt; 90% → emergency de-risk to 15/70/X</div>'
            '<div class="rule-row">FR &lt; 100% → tilt bonds 62%, stocks 28%</div>'
            '<div class="rule-row">FR &gt; 120% → surplus equity tilt 55%</div>'
            '<div class="rule-row">Yield gap &lt; 1.5pp → trim equities</div>'
            '<div class="rule-row">Inflation &gt; 5% → add gold inflation hedge</div>'
            '<div class="rule-row">Regime = Recession → defensive 22/65</div></div>',
            unsafe_allow_html=True)
    with dr2:
        st.markdown(
            '<div class="panel-dark"><div class="metric-label" style="color:#b3473a;">Hedge Fund</div>'
            '<div class="rule-row">Trend + valuation aligned bull → 1.60–1.85× long</div>'
            '<div class="rule-row">Trend + valuation aligned bear → 0.05–0.30× long</div>'
            '<div class="rule-row">Vol target 12% ann. → scale exposure by σ</div>'
            '<div class="rule-row">Rates &gt; 5% → reduce financing exposure –0.15×</div>'
            '<div class="rule-row">Recession cap → gross long ≤ 0.50×</div>'
            '<div class="rule-row">Recession/oil → add up to 0.30× gold</div></div>',
            unsafe_allow_html=True)
    with dr3:
        st.markdown(
            '<div class="panel-dark"><div class="metric-label" style="color:#c08a2e;">Retail Investor</div>'
            '<div class="rule-row">F/G score &gt; 0.8 → euphoric buying, 90–95% stocks</div>'
            '<div class="rule-row">F/G score &lt; –0.8 → panic sell, 15–30% stocks</div>'
            '<div class="rule-row">5p drawdown &gt; 6% → panic sell trigger –15pp</div>'
            '<div class="rule-row">Regime = Recession → cap stocks at 45%</div>'
            '<div class="rule-row">Mean-reverts toward 80/20 stocks/cash</div></div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown("### Return attribution — Stocks")

    attr_colors = {
        "Valuation_Macro": C["valuation_attr"],
        "Pension Fund":    C["pension_attr"],
        "Hedge Fund":      C["hedge_attr"],
        "Retail Investor": C["retail_attr"],
        "Noise":           C["noise_attr"],
    }

    # Waterfall summary
    summ_labels = ["Valuation_Macro"] + agent_names + ["Noise", "Total"]
    summ_values = [attr_summ.get(k, 0) for k in summ_labels]
    measure     = ["relative"] * (len(summ_labels)-1) + ["total"]
    bar_colors  = [attr_colors.get(l, C["text_dim"]) for l in summ_labels[:-1]] + [C["green"] if summ_values[-1] > 0 else C["red"]]

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=measure,
        x=summ_labels,
        y=summ_values,
        connector=dict(line=dict(color=C["border"], width=1)),
        increasing=dict(marker_color=C["green"]),
        decreasing=dict(marker_color=C["red"]),
        totals=dict(marker_color=C["blue"]),
    ))
    fig_wf.update_layout(**LAYOUT, height=320, title="Cumulative return attribution (pp)")
    st.plotly_chart(fig_wf, use_container_width=True)

    # Rolling stacked area
    st.markdown("### Period-by-period attribution")
    fig_attr = go.Figure()
    for col in ["Valuation_Macro"] + agent_names + ["Noise"]:
        if col in attr_df.columns:
            fig_attr.add_trace(go.Scatter(
                x=attr_df["Period"], y=attr_df[col],
                name=col, stackgroup="positive" if attr_df[col].mean() >= 0 else "negative",
                line=dict(color=attr_colors.get(col, C["text_dim"]), width=0),
                fillcolor=attr_colors.get(col, C["text_dim"]) + "99"))
    fig_attr.add_trace(go.Scatter(
        x=attr_df["Period"], y=attr_df["Total"],
        name="Total", line=dict(color=C["text"], width=1.5, dash="dot")))
    fig_attr.update_layout(**LAYOUT, height=320, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_attr, use_container_width=True)

    # Attribution for Bonds and Gold
    st.markdown("### Attribution — Bonds & Gold")
    bc1, bc2 = st.columns(2)
    for col_widget, asset_name in [(bc1, "Bonds"), (bc2, "Gold")]:
        with col_widget:
            adf = compute_attribution_table(eng, asset_name)
            asumm = summarize_attribution(adf, agent_names)
            labels = ["Valuation_Macro"] + agent_names + ["Noise"]
            vals   = [asumm.get(k, 0) for k in labels]
            meas   = ["relative"] * len(labels)
            fig_a2 = go.Figure(go.Waterfall(
                orientation="v", measure=meas + ["total"],
                x=labels + ["Total"],
                y=vals + [asumm.get("Total", 0)],
                increasing=dict(marker_color=C["green"]),
                decreasing=dict(marker_color=C["red"]),
                totals=dict(marker_color=C["blue"]),
                connector=dict(line=dict(color=C["border"])),
            ))
            fig_a2.update_layout(**LAYOUT, height=280,
                                 title=f"{asset_name} attribution (pp)")
            st.plotly_chart(fig_a2, use_container_width=True)

    # Raw attribution table
    with st.expander("Raw attribution data"):
        st.dataframe(attr_df.style.format("{:.3f}"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown(f"### Monte Carlo — {mc['n_seeds']} seeds · {periods} periods")

    mc_s = monte_carlo_stats(mc["stock_returns"])
    mc_b = monte_carlo_stats(mc["bond_returns"])
    mc_g = monte_carlo_stats(mc["gold_returns"])

    # KPI strip
    mc1, mc2, mc3 = st.columns(3)
    for col, label, stats, color in [
        (mc1, "Stocks", mc_s, C["stock"]),
        (mc2, "Bonds",  mc_b, C["bond"]),
        (mc3, "Gold",   mc_g, C["gold"]),
    ]:
        with col:
            st.markdown(
                f'<div class="panel">'
                f'<div class="metric-label" style="color:{color};">{label}</div>'
                f'<div class="metric-value" style="color:{color};">{_fmt_pct(stats["mean"])}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:.78rem;color:{C["text_dim"]};margin-top:.3rem;">'
                f'Median {_fmt_pct(stats["median"])} · σ {stats["std"]:.1f}pp<br>'
                f'Best {_fmt_pct(stats["best"])} · Worst {_fmt_pct(stats["worst"])}<br>'
                f'P(+) {stats["p_pos"]:.0f}%</div></div>',
                unsafe_allow_html=True)

    # Distribution histograms
    st.markdown("### Return distributions")
    fig_dist = make_subplots(rows=1, cols=3, subplot_titles=["Stocks", "Bonds", "Gold"])
    for i, (arr, color) in enumerate([(mc["stock_returns"], C["stock"]),
                                       (mc["bond_returns"],  C["bond"]),
                                       (mc["gold_returns"],  C["gold"])], 1):
        fig_dist.add_trace(go.Histogram(x=arr, nbinsx=30, marker_color=color,
                                        opacity=0.75, showlegend=False), row=1, col=i)
        fig_dist.add_vline(x=float(np.mean(arr)), line=dict(color=C["text"], dash="dot"),
                           row=1, col=i)
    fig_dist.update_layout(**LAYOUT, height=280, showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Hedge leverage & retail greed distributions
    hg1, hg2 = st.columns(2)
    with hg1:
        fig_lv = go.Figure(go.Histogram(
            x=mc["hedge_max_leverages"], nbinsx=25,
            marker_color=C["hedge"], opacity=0.75))
        fig_lv.update_layout(**LAYOUT, height=240,
                             title="Hedge fund peak gross leverage (×)")
        st.plotly_chart(fig_lv, use_container_width=True)
    with hg2:
        fig_gr = go.Figure(go.Histogram(
            x=mc["retail_avg_greeds"], nbinsx=25,
            marker_color=C["retail"], opacity=0.75))
        fig_gr.update_layout(**LAYOUT, height=240,
                             title="Retail avg greed score")
        st.plotly_chart(fig_gr, use_container_width=True)

    # Scenario consistency tests
    st.markdown("### Scenario consistency tests")
    for test in ct:
        badge = ('<span class="pass-badge">PASS</span>' if test["passed"]
                 else '<span class="fail-badge">FAIL</span>')
        st.markdown(
            f'<div class="panel" style="padding:.6rem 1rem;">'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:.85rem;">'
            f'{test["name"]}</span>{badge}'
            f'<span style="color:{C["text_dim"]};font-size:.8rem;font-family:IBM Plex Mono,monospace;margin-left:.8rem;">'
            f'{test["condition"]} → {test["value"]}</span></div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AGENT IMPACT (ABLATION)
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("### Agent impact analysis — ablation study")
    st.markdown(
        '<div class="at-sub">Each row removes one agent from the demand aggregation. '
        '"Return Contribution" shows how much that agent added or subtracted '
        'from total equity return vs the full model.</div>',
        unsafe_allow_html=True)

    abl_df = summarize_ablation(abl, "Stocks")

    # Bar chart
    contrib_df = abl_df[abl_df["Configuration"] != "Full Model"].copy()
    fig_abl = go.Figure()
    for _, row in contrib_df.iterrows():
        label = row["Configuration"].replace("Without ", "−")
        color = C["green"] if row["Return Contribution (pp vs Full)"] > 0 else C["red"]
        fig_abl.add_trace(go.Bar(
            name=label,
            x=[label],
            y=[row["Return Contribution (pp vs Full)"]],
            marker_color=color,
        ))
    fig_abl.update_layout(**LAYOUT, height=280,
                          title="Equity return contribution by agent (pp vs full model)",
                          showlegend=False)
    st.plotly_chart(fig_abl, use_container_width=True)

    # Price paths comparison
    st.markdown("### Price paths across configurations")
    fig_ab2 = go.Figure()
    abl_colors = {
        "Full Model":           C["stock"],
        "Without Pension Fund": C["pension_attr"],
        "Without Hedge Fund":   C["hedge_attr"],
        "Without Retail":       C["retail_attr"],
    }
    for label, ab_eng in abl.items():
        ph = ab_eng.market.assets["Stocks"].price_history
        fig_ab2.add_trace(go.Scatter(
            x=list(range(len(ph))), y=ph,
            name=label,
            line=dict(color=abl_colors.get(label, C["text_dim"]),
                      width=2 if label == "Full Model" else 1.2,
                      dash="solid" if label == "Full Model" else "dash")))
    fig_ab2.update_layout(**LAYOUT, height=320,
                          legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_ab2, use_container_width=True)

    # Summary table
    st.markdown("### Ablation summary table")
    display_cols = ["Configuration", "Total Return", "Volatility (per period)",
                    "Max Drawdown", "Return Contribution (pp vs Full)"]
    st.dataframe(
        abl_df[display_cols].style.format({
            "Total Return": "{:.1f}%",
            "Volatility (per period)": "{:.3f}%",
            "Max Drawdown": "{:.1f}%",
            "Return Contribution (pp vs Full)": "{:+.1f}pp",
        }),
        use_container_width=True)

    # Economic dominance
    st.markdown("---")
    st.markdown("### Economic dominance score")
    if eco_dom["r2"] is not None:
        ed1, ed2, ed3 = st.columns(3)
        with ed1:
            st.markdown(
                _metric_html("Macro R²", f"{eco_dom['signal_pct']:.1f}%",
                             C["green"] if eco_dom["signal_pct"] > 40 else C["amber"]),
                unsafe_allow_html=True)
        with ed2:
            st.markdown(
                _metric_html("Noise share", f"{eco_dom['noise_pct']:.1f}%",
                             C["red"] if eco_dom["warning"] else C["text"]),
                unsafe_allow_html=True)
        with ed3:
            warn_txt = "⚠ High noise" if eco_dom["warning"] else "✓ OK"
            st.markdown(_metric_html("Signal quality", warn_txt), unsafe_allow_html=True)
        if "asset_r2" in eco_dom:
            for asset_name, r2_val in eco_dom["asset_r2"].items():
                st.markdown(
                    f'<span class="badge">{asset_name}</span>'
                    f'<span style="font-family:IBM Plex Mono,monospace;font-size:.82rem;">'
                    f'R² = {r2_val*100:.1f}%</span><br>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EVENT LOG
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown("### Simulation event log")

    log_filter = st.multiselect(
        "Filter by regime", REGIMES, default=REGIMES, key="log_filter")

    shown = 0
    for log in reversed(eng.logs):
        if log.regime not in log_filter:
            continue
        shown += 1
        if shown > 80:
            st.markdown(f'<div class="log-line" style="color:{C["text_dim"]};">'
                        f'… {len(eng.logs) - 80} earlier periods hidden</div>',
                        unsafe_allow_html=True)
            break

        sc_color = _delta_color(log.stock_change)
        bc_color = _delta_color(log.bond_change)
        gc_color = _delta_color(log.gold_change)

        details_html = "".join(
            f'<div class="log-line">↳ {d}</div>' for d in log.details)

        st.markdown(
            f'<div class="panel" style="padding:.55rem 1rem;margin-bottom:.35rem;">'
            f'<span class="log-period">T{log.period:03d}</span> '
            f'{_regime_pill(log.regime)} '
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:.82rem;">'
            f'{log.headline}</span><br>'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:.78rem;">'
            f'S <span style="color:{sc_color};">{_fmt_pct(log.stock_change)}</span>  '
            f'B <span style="color:{bc_color};">{_fmt_pct(log.bond_change)}</span>  '
            f'G <span style="color:{gc_color};">{_fmt_pct(log.gold_change)}</span>  '
            f'FR {log.funding_ratio:.1f}%</span>'
            f'{details_html}</div>',
            unsafe_allow_html=True)

    st.markdown(f'<div class="log-line">{len(eng.logs)} total events</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.markdown("### Fair-value drift diagnostics (v3.1 sanity panel)")
    st.markdown(
        '<div class="at-sub">Tracks the fair-value anchor separately from price. '
        'If FV total return ≈ price total return, agents had little net impact. '
        'Clip events = periods where the raw model-implied annual return exceeded '
        f'±{MAX_ANNUAL_FAIR_VALUE_DRIFT_PCT:.0f}%/year and was clamped.</div>',
        unsafe_allow_html=True)

    # FV vs price return comparison
    d1, d2, d3 = st.columns(3)
    for col_w, name, price_color in [
        (d1, "Stocks", C["stock"]),
        (d2, "Bonds",  C["bond"]),
        (d3, "Gold",   C["gold"]),
    ]:
        with col_w:
            asset = market.assets[name]
            fv_ret   = asset.fair_value_total_return_pct()
            px_ret   = asset.total_return_pct()
            clip_cnt = market.fv_drift_clip_events.get(name, 0)
            clip_pct = clip_cnt / max(periods, 1) * 100
            badge_cls = "pass-badge" if clip_cnt == 0 else ("warn-badge" if clip_pct < 15 else "fail-badge")
            st.markdown(
                f'<div class="panel">'
                f'<div class="metric-label" style="color:{price_color};">{name}</div>'
                f'<div style="margin-top:.4rem;font-family:IBM Plex Mono,monospace;font-size:.82rem;">'
                f'Price return: <span style="color:{_delta_color(px_ret)};">{_fmt_pct(px_ret)}</span><br>'
                f'FV return:    <span style="color:{_delta_color(fv_ret)};">{_fmt_pct(fv_ret)}</span><br>'
                f'Clip events:  <span class="{badge_cls}">{clip_cnt} ({clip_pct:.0f}%)</span>'
                f'</div></div>',
                unsafe_allow_html=True)

    # FV history chart
    st.markdown("### Fair-value anchor paths")
    fig_fv = go.Figure()
    for name, color in [("Stocks", C["stock"]), ("Bonds", C["bond"]), ("Gold", C["gold"])]:
        asset = market.assets[name]
        fig_fv.add_trace(go.Scatter(
            y=asset.fair_value_history, name=f"{name} FV",
            line=dict(color=color, width=1.8)))
        fig_fv.add_trace(go.Scatter(
            y=asset.price_history, name=f"{name} Price",
            line=dict(color=color, width=1, dash="dot"), opacity=0.5))
    fig_fv.update_layout(**LAYOUT, height=320,
                         title="Fair value vs price (base=100)",
                         legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_fv, use_container_width=True)

    # Raw annual return signal and clipped drift
    st.markdown("### Per-period fair-value drift — raw vs clipped")
    fig_drift = make_subplots(rows=1, cols=3,
                              subplot_titles=["Stocks", "Bonds", "Gold"])
    for i, (name, color) in enumerate([("Stocks", C["stock"]),
                                        ("Bonds",  C["bond"]),
                                        ("Gold",   C["gold"])], 1):
        asset = market.assets[name]
        raw   = asset.fv_annual_return_raw_history
        clipped = [v * PERIODS_PER_YEAR for v in asset.fv_period_drift_applied_history]
        fig_drift.add_trace(go.Scatter(y=raw, name="Raw annual %",
                                       line=dict(color=color, width=1.5)), row=1, col=i)
        fig_drift.add_trace(go.Scatter(y=clipped, name="Clipped (×12)",
                                       line=dict(color=C["amber"], width=1,
                                                 dash="dash")), row=1, col=i)
        fig_drift.add_hline(y=MAX_ANNUAL_FAIR_VALUE_DRIFT_PCT,
                            line=dict(color=C["red"], dash="dot", width=1), row=1, col=i)
        fig_drift.add_hline(y=-MAX_ANNUAL_FAIR_VALUE_DRIFT_PCT,
                            line=dict(color=C["red"], dash="dot", width=1), row=1, col=i)
    fig_drift.update_layout(**LAYOUT, height=280, showlegend=False)
    st.plotly_chart(fig_drift, use_container_width=True)

    # StockValuation state
    st.markdown("### StockValuation internal state")
    sv: StockValuation = market.valuations["Stocks"]
    sv1, sv2, sv3 = st.columns(3)
    with sv1:
        st.markdown(_metric_html("Current PE", f"{sv.pe_ratio:.1f}×",
                                 C["red"] if sv.is_expensive() else
                                 C["green"] if sv.is_cheap() else C["text"]),
                    unsafe_allow_html=True)
    with sv2:
        st.markdown(_metric_html("Earnings", f"{sv.earnings:.2f}"), unsafe_allow_html=True)
    with sv3:
        st.markdown(_metric_html("Earnings growth", f"{sv.earnings_growth:.2f}%"),
                    unsafe_allow_html=True)

    # Narrative summary
    st.markdown("---")
    st.markdown("### Institutional narrative summary")
    narrative = generate_institutional_summary(eng)
    for para in narrative.split("\n\n"):
        st.markdown(
            f'<div class="panel" style="font-family:IBM Plex Sans,sans-serif;'
            f'font-size:.88rem;line-height:1.6;color:{C["text"]};">{para}</div>',
            unsafe_allow_html=True)
