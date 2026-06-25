"""
================================================================================
  INSTITUTIONAL-GRADE DCF MODEL  |  Streamlit App
  Data Source: yfinance (no API key required)
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DCF Valuation Model",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
  .dcf-header {
    background: linear-gradient(135deg, #0a1628 0%, #112244 60%, #1a3a6e 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #1e3a5f;
  }
  .dcf-header h1 { color: #e8f0fe; font-size: 1.8rem; font-weight: 700; margin: 0 0 0.25rem 0; }
  .dcf-header .subtitle { color: #7ea8d8; font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .metric-card {
    background: #0e1f38; border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 1.1rem 1.3rem; text-align: center;
  }
  .metric-card .label { color: #7ea8d8; font-size: 0.72rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.4rem; }
  .metric-card .value { color: #e8f0fe; font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .metric-card .delta { font-size: 0.78rem; font-weight: 500; margin-top: 0.2rem; }
  .positive { color: #4ade80; }
  .negative { color: #f87171; }
  .section-label {
    color: #7ea8d8; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.15em;
    text-transform: uppercase; margin: 1.5rem 0 0.6rem 0;
    padding-bottom: 0.4rem; border-bottom: 1px solid #1e3a5f;
  }
  .disclaimer { color: #4a6a8a; font-size: 0.72rem; text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #1a2f4a; }
  #MainMenu, footer, header { visibility: hidden; }
  [data-testid="stSidebar"] { background: #080f1e; border-right: 1px solid #1e3a5f; }
  [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stSlider p { color: #a0c4e8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Inputs")
    st.markdown("---")
    TICKER         = st.text_input("Ticker Symbol", value="GOOG").upper().strip()
    WACC           = st.slider("WACC (%)", 6.0, 20.0, 11.2, 0.1) / 100
    FORECAST_YEARS = st.slider("Forecast Years", 3, 10, 5)
    st.markdown("---")
    run = st.button("▶ Run Model", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption("Data: Yahoo Finance · yfinance\nFor analytical purposes only.")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def safe(val, fallback=np.nan):
    try:
        v = float(val)
        return v if np.isfinite(v) else fallback
    except Exception:
        return fallback

def trailing_avg(series, n=3):
    return series.dropna().tail(n).mean()

def cagr(series, n):
    s = series.dropna()
    if len(s) < n + 1:
        n = len(s) - 1
    if n <= 0 or s.iloc[-n - 1] <= 0:
        return np.nan
    return (s.iloc[-1] / s.iloc[-n - 1]) ** (1 / n) - 1

def fetch_treasury_rate():
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "GS10", "api_key": "demo",
                    "sort_order": "desc", "limit": 1, "file_type": "json"},
            timeout=10,
        )
        return float(r.json()["observations"][0]["value"]) / 100
    except Exception:
        return 0.04

def dark_layout(fig, title=""):
    DARK_BG  = "#0a1628"
    GRID_CLR = "#1e3a5f"
    TEXT_CLR = "#7ea8d8"
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e8f0fe", size=13), x=0),
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_CLR, family="Inter"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_CLR)),
        xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, tickfont=dict(color=TEXT_CLR)),
        yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, tickfont=dict(color=TEXT_CLR)),
    )
    return fig

# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all(ticker):
    t = yf.Ticker(ticker)
    return t.info, t.financials, t.cashflow, t.balance_sheet

def build_income(fin, limit=5):
    if fin is None or fin.empty:
        return []
    df = fin.T.sort_index().tail(limit)
    rows = []
    for date, row in df.iterrows():
        op  = row.get("Operating Income", np.nan)
        da  = row.get("Reconciled Depreciation", row.get("Depreciation And Amortization", np.nan))
        ebi = row.get("EBITDA", np.nan)
        if pd.isna(ebi) and not pd.isna(op) and not pd.isna(da):
            ebi = op + da
        rows.append({
            "calendarYear"     : str(date.year),
            "revenue"          : row.get("Total Revenue", np.nan),
            "operatingIncome"  : op,
            "ebitda"           : ebi,
            "netIncome"        : row.get("Net Income", np.nan),
            "incomeTaxExpense" : row.get("Tax Provision", np.nan),
            "incomeBeforeTax"  : row.get("Pretax Income", np.nan),
            "interestExpense"  : row.get("Interest Expense", np.nan),
        })
    return rows

def build_cashflow(cf, limit=5):
    if cf is None or cf.empty:
        return []
    df = cf.T.sort_index().tail(limit)
    rows = []
    for date, row in df.iterrows():
        da = row.get("Depreciation And Amortization",
             row.get("Reconciled Depreciation", np.nan))
        rows.append({
            "calendarYear"                : str(date.year),
            "depreciationAndAmortization" : da,
            "capitalExpenditure"          : row.get("Capital Expenditure", np.nan),
            "operatingCashFlow"           : row.get("Operating Cash Flow", np.nan),
        })
    return rows

def build_balance(bal, limit=5):
    if bal is None or bal.empty:
        return []
    df = bal.T.sort_index().tail(limit)
    rows = []
    for date, row in df.iterrows():
        cash = row.get("Cash And Cash Equivalents",
               row.get("Cash Cash Equivalents And Short Term Investments", np.nan))
        rows.append({
            "calendarYear"              : str(date.year),
            "netReceivables"            : row.get("Net Receivables", row.get("Accounts Receivable", np.nan)),
            "inventory"                 : row.get("Inventory", np.nan),
            "accountPayables"           : row.get("Accounts Payable", np.nan),
            "propertyPlantEquipmentNet" : row.get("Net PPE", np.nan),
            "totalCurrentAssets"        : row.get("Current Assets", np.nan),
            "totalCurrentLiabilities"   : row.get("Current Liabilities", np.nan),
            "totalDebt"                 : row.get("Total Debt", np.nan),
            "cashAndCashEquivalents"    : cash,
        })
    return rows

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dcf-header">
  <div class="subtitle">Equity Research · Institutional DCF</div>
  <h1>📊 Discounted Cash Flow Valuation</h1>
</div>
""", unsafe_allow_html=True)

if not run:
    st.info("👈 Set your inputs in the sidebar and click **Run Model** to begin.")
    st.stop()

# ─────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────
with st.spinner(f"Fetching data for **{TICKER}**…"):
    try:
        info, fin_raw, cf_raw, bal_raw = fetch_all(TICKER)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

inc_raw  = build_income(fin_raw)
cf_list  = build_cashflow(cf_raw)
bal_list = build_balance(bal_raw)

if not inc_raw:
    st.error(f"No financial data found for **{TICKER}**. Check the ticker and try again.")
    st.stop()

inc_raw  = list(reversed(inc_raw))
cf_list  = list(reversed(cf_list))
bal_list = list(reversed(bal_list))

common_yrs = sorted(
    {r["calendarYear"] for r in inc_raw} &
    {r["calendarYear"] for r in cf_list} &
    {r["calendarYear"] for r in bal_list}
)
if not common_yrs:
    st.error("Could not align financial statement years. Try a different ticker.")
    st.stop()

inc_by_yr = {r["calendarYear"]: r for r in inc_raw}
cf_by_yr  = {r["calendarYear"]: r for r in cf_list}
bal_by_yr = {r["calendarYear"]: r for r in bal_list}

# Build historical df
rows = []
for yr in common_yrs:
    i = inc_by_yr[yr]; c = cf_by_yr[yr]; b = bal_by_yr[yr]
    revenue = safe(i.get("revenue"))
    ebit    = safe(i.get("operatingIncome"))
    da      = safe(c.get("depreciationAndAmortization"))
    ebitda  = safe(i.get("ebitda"))
    if np.isnan(ebitda) and not np.isnan(ebit) and not np.isnan(da):
        ebitda = ebit + da
    rows.append({
        "Year"          : int(yr),
        "Revenue"       : revenue / 1e9,
        "EBIT"          : ebit / 1e9,
        "EBITDA"        : ebitda / 1e9,
        "D&A"           : da / 1e9,
        "CAPEX"         : abs(safe(c.get("capitalExpenditure", 0))) / 1e9,
        "OCF"           : safe(c.get("operatingCashFlow")) / 1e9,
        "Tax_Provision" : abs(safe(i.get("incomeTaxExpense", 0))) / 1e9,
        "Pretax_Income" : safe(i.get("incomeBeforeTax")) / 1e9,
        "Net_Income"    : safe(i.get("netIncome")) / 1e9,
        "Receivables"   : safe(b.get("netReceivables")) / 1e9,
        "Inventory"     : safe(b.get("inventory")) / 1e9,
        "Payables"      : safe(b.get("accountPayables")) / 1e9,
        "Net_PPE"       : safe(b.get("propertyPlantEquipmentNet")) / 1e9,
        "Current_Assets": safe(b.get("totalCurrentAssets")) / 1e9,
        "Current_Liab"  : safe(b.get("totalCurrentLiabilities")) / 1e9,
        "Total_Debt"    : safe(b.get("totalDebt", 0)) / 1e9,
        "Cash"          : safe(b.get("cashAndCashEquivalents", 0)) / 1e9,
    })

df = pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)
df["NWC"]           = df["Current_Assets"] - df["Current_Liab"]
df["DELTA_NWC"]     = df["NWC"].diff()
df["EBIT_Margin"]   = df["EBIT"]   / df["Revenue"].replace(0, np.nan)
df["EBITDA_Margin"] = df["EBITDA"] / df["Revenue"].replace(0, np.nan)
df["DA_pct"]        = df["D&A"]    / df["Revenue"].replace(0, np.nan)
df["Capex_pct"]     = df["CAPEX"]  / df["Revenue"].replace(0, np.nan)
df["Tax_Rate"]      = np.where(
    (df["Pretax_Income"] > 0) & df["Tax_Provision"].notna(),
    df["Tax_Provision"] / df["Pretax_Income"], np.nan)
df["DSO"] = (df["Receivables"] / df["Revenue"].replace(0, np.nan)) * 365
df["DIO"] = (df["Inventory"]   / df["Revenue"].replace(0, np.nan)) * 365
df["DPO"] = (df["Payables"]    / df["Revenue"].replace(0, np.nan)) * 365
df["NOPAT"]      = df["EBIT"] * (1 - df["Tax_Rate"])
df["FCFF"]       = df["NOPAT"] + df["D&A"] - df["CAPEX"] - df["DELTA_NWC"].fillna(0)
df["Rev_Growth"] = df["Revenue"].pct_change()

# Market data
company_name  = info.get("longName", TICKER)
sector        = info.get("sector",   "N/A")
industry      = info.get("industry", "N/A")
current_price = safe(info.get("currentPrice") or info.get("regularMarketPrice"))
week52_low    = safe(info.get("fiftyTwoWeekLow"))
week52_high   = safe(info.get("fiftyTwoWeekHigh"))
shares_out    = safe(info.get("sharesOutstanding", 0)) / 1e9
market_cap    = safe(info.get("marketCap", 0)) / 1e9
fwd_pe        = safe(info.get("trailingPE") or info.get("forwardPE"))
ev_ebitda_mult= safe(info.get("enterpriseToEbitda"))
rev_growth_fwd= safe(info.get("revenueGrowth"))

latest_bal = bal_by_yr[common_yrs[-1]]
NET_DEBT   = (safe(latest_bal.get("totalDebt", 0)) -
              safe(latest_bal.get("cashAndCashEquivalents", 0))) / 1e9
SHARES     = shares_out if shares_out > 0 else (market_cap / current_price if current_price else 1)

if np.isnan(rev_growth_fwd) or not (0.0 < rev_growth_fwd < 0.60):
    rev_growth_fwd = np.nan

RISK_FREE_RATE  = fetch_treasury_rate()
TERMINAL_GROWTH = RISK_FREE_RATE

# Forecast assumptions
eff_tax  = trailing_avg(df["Tax_Rate"], 3)
TAX_RATE = float(np.clip(eff_tax, 0.10, 0.35))

margin_base = max(trailing_avg(df["EBIT_Margin"], 3), df["EBIT_Margin"].iloc[-1])
EBIT_MARGINS = []
m = margin_base
for _ in range(FORECAST_YEARS):
    m = min(m + 0.005, 0.35)
    EBIT_MARGINS.append(round(m, 4))

base_g = rev_growth_fwd if not np.isnan(rev_growth_fwd) else cagr(df["Revenue"], 3)
if np.isnan(base_g): base_g = 0.08
floor_g = TERMINAL_GROWTH + 0.01
REVENUE_GROWTH = []
g = base_g
for _ in range(FORECAST_YEARS):
    REVENUE_GROWTH.append(round(max(g, floor_g), 4))
    g *= 0.88

da_pct_hist    = trailing_avg(df["DA_pct"], 3)
capex_pct_hist = trailing_avg(df["Capex_pct"], 3)
if np.isnan(capex_pct_hist): capex_pct_hist = (da_pct_hist + 0.05) if not np.isnan(da_pct_hist) else 0.10
if np.isnan(da_pct_hist):    da_pct_hist = 0.07
maint_capex_pct  = da_pct_hist
growth_capex_pct = max(capex_pct_hist - maint_capex_pct, 0)

ppe_last = df["Net_PPE"].iloc[-1]
if np.isnan(ppe_last): ppe_last = df["CAPEX"].sum() * 0.5
da_rate = trailing_avg(df["D&A"] / df["Net_PPE"].replace(0, np.nan), 3)
if np.isnan(da_rate) or da_rate <= 0: da_rate = da_pct_hist

dso_hist = trailing_avg(df["DSO"], 3); dso_hist = 30.0 if np.isnan(dso_hist) else dso_hist
dio_hist = trailing_avg(df["DIO"], 3); dio_hist = 45.0 if np.isnan(dio_hist) else dio_hist
dpo_hist = trailing_avg(df["DPO"], 3); dpo_hist = 45.0 if np.isnan(dpo_hist) else dpo_hist

# Forecast engine
last_rev = df["Revenue"].iloc[-1]; last_yr = df["Year"].iloc[-1]
ppe = ppe_last; prev_rev = last_rev
dso, dio, dpo = dso_hist, dio_hist, dpo_hist
prev_rec = df["Receivables"].iloc[-1]; prev_rec = (dso_hist/365)*last_rev if np.isnan(prev_rec) else prev_rec
prev_inv = df["Inventory"].iloc[-1];   prev_inv = (dio_hist/365)*last_rev if np.isnan(prev_inv) else prev_inv
prev_pay = df["Payables"].iloc[-1];    prev_pay = (dpo_hist/365)*last_rev if np.isnan(prev_pay) else prev_pay

forecast_rows = []
for i in range(FORECAST_YEARS):
    yr    = last_yr + i + 1
    rev   = prev_rev * (1 + REVENUE_GROWTH[i])
    ebit  = rev * EBIT_MARGINS[i]
    nopat = ebit * (1 - TAX_RATE)
    capex = rev * maint_capex_pct + (rev - prev_rev) * growth_capex_pct
    da    = ppe * da_rate
    ppe   = ppe + capex - da
    ebitda = ebit + da
    dso = max(dso - 0.5, 1); dio = max(dio - 0.3, 0); dpo = max(dpo + 0.5, 1)
    rec = (dso/365)*rev; inv = (dio/365)*rev; pay = (dpo/365)*rev
    delta_nwc = (rec - prev_rec) + (inv - prev_inv) - (pay - prev_pay)
    fcff = nopat + da - capex - delta_nwc
    forecast_rows.append({
        "Year": yr, "Revenue": rev, "Rev_Growth": REVENUE_GROWTH[i],
        "EBIT": ebit, "EBIT_Margin": EBIT_MARGINS[i], "EBITDA": ebitda,
        "D&A": da, "CAPEX": capex, "NOPAT": nopat, "DELTA_NWC": delta_nwc,
        "DSO": dso, "DIO": dio, "DPO": dpo, "FCFF": fcff,
    })
    prev_rev=rev; prev_rec=rec; prev_inv=inv; prev_pay=pay

fc = pd.DataFrame(forecast_rows)
fc["Year_Index"]      = range(1, FORECAST_YEARS + 1)
fc["Discount_Factor"] = 1 / (1 + WACC) ** fc["Year_Index"]
fc["PV_FCFF"]         = fc["FCFF"] * fc["Discount_Factor"]

# Valuation
last_fcff   = fc["FCFF"].iloc[-1]
TV          = (last_fcff * (1 + TERMINAL_GROWTH)) / (WACC - TERMINAL_GROWTH)
PV_TV       = TV / (1 + WACC) ** fc["Year_Index"].iloc[-1]
sum_pv_fcff = fc["PV_FCFF"].sum()
EV          = sum_pv_fcff + PV_TV
EQUITY_VAL  = EV - NET_DEBT
PRICE_TARGET= EQUITY_VAL / SHARES
TV_PCT      = PV_TV / EV * 100
updown      = (PRICE_TARGET / current_price - 1) * 100 if not np.isnan(current_price) else np.nan

# Football field
fwd_ebitda   = fc["EBITDA"].iloc[0]
last_ni      = df["Net_Income"].iloc[-1]
ev_mult_low  = ev_ebitda_mult * 0.80 if not np.isnan(ev_ebitda_mult) else np.nan
ev_mult_high = ev_ebitda_mult * 1.20 if not np.isnan(ev_ebitda_mult) else np.nan
price_ev_low = ((fwd_ebitda * ev_mult_low  - NET_DEBT) / SHARES) if not np.isnan(ev_mult_low)  else np.nan
price_ev_hi  = ((fwd_ebitda * ev_mult_high - NET_DEBT) / SHARES) if not np.isnan(ev_mult_high) else np.nan
pe_low       = fwd_pe * 0.85 * (last_ni / SHARES) if not np.isnan(fwd_pe) else np.nan
pe_high      = fwd_pe * 1.15 * (last_ni / SHARES) if not np.isnan(fwd_pe) else np.nan

# ─────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────
BLUE     = "#3b82f6"
TEAL     = "#2dd4bf"
AMBER    = "#fbbf24"
RED      = "#f87171"
GRID_CLR = "#1e3a5f"

st.markdown(f"### {company_name} &nbsp; `{TICKER}` &nbsp; · &nbsp; {sector} · {industry}")

# Metric cards
col1, col2, col3, col4, col5 = st.columns(5)
updown_color = "positive" if (not np.isnan(updown) and updown >= 0) else "negative"
updown_str   = f"{'▲' if updown >= 0 else '▼'} {abs(updown):.1f}%" if not np.isnan(updown) else "N/A"

cards = [
    (col1, "DCF Price Target",  f"${PRICE_TARGET:,.2f}", updown_str, updown_color),
    (col2, "Current Price",     f"${current_price:,.2f}" if not np.isnan(current_price) else "N/A",
     f"52W: ${week52_low:,.0f}–${week52_high:,.0f}", ""),
    (col3, "Enterprise Value",  f"${EV:,.1f}B", f"Equity: ${EQUITY_VAL:,.1f}B", ""),
    (col4, "WACC / TGR",        f"{WACC:.1%}", f"TGR: {TERMINAL_GROWTH:.2%}", ""),
    (col5, "TV % of EV",        f"{TV_PCT:.1f}%",
     "OK" if 40 < TV_PCT < 85 else "⚠ Review",
     "positive" if 40 < TV_PCT < 85 else "negative"),
]
for col, label, value, delta, dclass in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="delta {dclass}">{delta}</div>
        </div>""", unsafe_allow_html=True)

# Chart row 1
st.markdown('<div class="section-label">Historical Performance</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    fig = go.Figure()
    fig.add_bar(x=df["Year"], y=df["Revenue"], name="Revenue", marker_color=BLUE, opacity=0.85)
    fig.add_bar(x=df["Year"], y=df["EBITDA"],  name="EBITDA",  marker_color=TEAL, opacity=0.85)
    fig.add_bar(x=df["Year"], y=df["FCFF"],    name="FCFF",    marker_color=AMBER, opacity=0.85)
    fig.update_layout(barmode="group")
    dark_layout(fig, "Revenue · EBITDA · FCFF  (USD B)")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig2 = go.Figure()
    fig2.add_scatter(x=df["Year"], y=df["EBIT_Margin"]*100,   mode="lines+markers", name="EBIT %",   line=dict(color=BLUE, width=2))
    fig2.add_scatter(x=df["Year"], y=df["EBITDA_Margin"]*100, mode="lines+markers", name="EBITDA %", line=dict(color=TEAL, width=2))
    fig2.add_scatter(x=df["Year"], y=df["Tax_Rate"]*100,      mode="lines+markers", name="Tax Rate %", line=dict(color=AMBER, width=2, dash="dot"))
    dark_layout(fig2, "Margin & Tax Rate  (%)")
    st.plotly_chart(fig2, use_container_width=True)

# Chart row 2
st.markdown('<div class="section-label">Forecast & DCF Bridge</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)

with c3:
    fig3 = go.Figure()
    fig3.add_scatter(x=list(df["Year"]),  y=list(df["Revenue"]),
                     mode="lines+markers", name="Historical Rev", line=dict(color=BLUE, width=2))
    fig3.add_scatter(x=list(fc["Year"]),  y=list(fc["Revenue"]),
                     mode="lines+markers", name="Forecast Rev",   line=dict(color=BLUE, width=2, dash="dash"))
    fig3.add_scatter(x=list(fc["Year"]),  y=list(fc["FCFF"]),
                     mode="lines+markers", name="Forecast FCFF",  line=dict(color=AMBER, width=2, dash="dash"))
    fig3.add_scatter(x=list(fc["Year"]),  y=list(fc["PV_FCFF"]),
                     mode="lines+markers", name="PV of FCFF",     line=dict(color=TEAL, width=2, dash="dot"))
    dark_layout(fig3, "Revenue & FCFF Forecast  (USD B)")
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    fig4 = go.Figure(go.Pie(
        labels=["PV Forecast FCFFs", "PV Terminal Value"],
        values=[sum_pv_fcff, PV_TV],
        hole=0.55,
        marker=dict(colors=[BLUE, TEAL]),
        textfont=dict(color="#e8f0fe"),
    ))
    fig4.add_annotation(text=f"EV<br>${EV:,.0f}B", x=0.5, y=0.5,
                        font=dict(color="#e8f0fe", size=14), showarrow=False)
    dark_layout(fig4, "Enterprise Value Composition")
    st.plotly_chart(fig4, use_container_width=True)

# Football field
st.markdown('<div class="section-label">Football Field  (Price per Share, USD)</div>', unsafe_allow_html=True)
ff_rows, ff_names = [], []
if not np.isnan(week52_low):
    ff_rows.append((week52_low, week52_high)); ff_names.append("52-Week Range")
if not np.isnan(price_ev_low):
    ff_rows.append((price_ev_low, price_ev_hi)); ff_names.append(f"EV/EBITDA ({ev_mult_low:.1f}x–{ev_mult_high:.1f}x)")
if not np.isnan(pe_low):
    ff_rows.append((pe_low, pe_high)); ff_names.append(f"P/E ({fwd_pe*0.85:.1f}x–{fwd_pe*1.15:.1f}x)")
ff_rows.append((PRICE_TARGET, PRICE_TARGET)); ff_names.append("DCF Base Case")

fig5 = go.Figure()
bar_colors = [GRID_CLR, TEAL, AMBER, BLUE]
for idx, ((lo, hi), name) in enumerate(zip(ff_rows, ff_names)):
    width = max(hi - lo, 0.01)
    fig5.add_trace(go.Bar(
        x=[width], base=[lo], y=[name], orientation="h",
        marker=dict(color=bar_colors[idx % len(bar_colors)], opacity=0.85),
        name=name,
        text=[f"${lo:,.0f}" if lo != hi else f"${lo:,.2f}"],
        textposition="outside",
        textfont=dict(color="#e8f0fe", size=11),
    ))
if not np.isnan(current_price):
    fig5.add_vline(x=current_price, line_color=RED, line_dash="dash",
                   annotation_text=f"Current ${current_price:,.0f}",
                   annotation_font_color=RED)
dark_layout(fig5, "Valuation Range by Methodology")
fig5.update_layout(showlegend=False, height=220,
                   yaxis=dict(autorange="reversed"),
                   xaxis=dict(tickprefix="$"))
st.plotly_chart(fig5, use_container_width=True)

# Sensitivity heatmap
st.markdown('<div class="section-label">Sensitivity Analysis  (Price per Share · rows = Terminal Growth · cols = WACC)</div>', unsafe_allow_html=True)
STEP, POINTS = 0.005, 5
wacc_range   = np.round(np.linspace(WACC - STEP*POINTS, WACC + STEP*POINTS, 2*POINTS+1), 4)
growth_range = np.round(np.linspace(TERMINAL_GROWTH - STEP*POINTS,
                                    TERMINAL_GROWTH + STEP*POINTS, 2*POINTS+1), 4)
n = fc["Year_Index"].iloc[-1]
z_vals, text_vals = [], []
for g in reversed(growth_range):
    z_row, t_row = [], []
    for w in wacc_range:
        if w <= g:
            z_row.append(np.nan); t_row.append("N/M")
        else:
            tv_s    = (last_fcff * (1 + g)) / (w - g)
            pv_tv_s = tv_s / (1 + w) ** n
            pv_f_s  = (fc["FCFF"] / (1 + w) ** fc["Year_Index"]).sum()
            price   = (pv_f_s + pv_tv_s - NET_DEBT) / SHARES
            z_row.append(price); t_row.append(f"${price:,.0f}")
    z_vals.append(z_row); text_vals.append(t_row)

fig6 = go.Figure(go.Heatmap(
    z=z_vals,
    x=[f"{w:.1%}" for w in wacc_range],
    y=[f"{g:.2%}" for g in reversed(growth_range)],
    text=text_vals, texttemplate="%{text}",
    colorscale=[[0, "#0a1628"], [0.5, "#1e3a5f"], [1, "#3b82f6"]],
    showscale=False,
    textfont=dict(size=11, color="#e8f0fe"),
))
dark_layout(fig6, "")
fig6.update_layout(height=350, xaxis_title="WACC", yaxis_title="Terminal Growth Rate")
st.plotly_chart(fig6, use_container_width=True)

# Data tables
st.markdown('<div class="section-label">Financial Detail</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📅 Historical Financials", "🔭 Forecast", "⚙️ Assumptions"])

with tab1:
    hd = df[["Year","Revenue","Rev_Growth","EBIT","EBIT_Margin","EBITDA","D&A","CAPEX","Tax_Rate","NOPAT","FCFF"]].copy()
    hd["Rev_Growth"]  = hd["Rev_Growth"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    hd["EBIT_Margin"] = hd["EBIT_Margin"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    hd["Tax_Rate"]    = hd["Tax_Rate"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    for col in ["Revenue","EBIT","EBITDA","D&A","CAPEX","NOPAT","FCFF"]:
        hd[col] = hd[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) and np.isfinite(x) else "—")
    st.dataframe(hd, use_container_width=True, hide_index=True)

with tab2:
    fd = fc[["Year","Revenue","Rev_Growth","EBIT","EBIT_Margin","EBITDA","D&A","CAPEX","NOPAT","FCFF","PV_FCFF"]].copy()
    fd["Rev_Growth"]  = fd["Rev_Growth"].map(lambda x: f"{x:.1%}")
    fd["EBIT_Margin"] = fd["EBIT_Margin"].map(lambda x: f"{x:.1%}")
    for col in ["Revenue","EBIT","EBITDA","D&A","CAPEX","NOPAT","FCFF","PV_FCFF"]:
        fd[col] = fd[col].map(lambda x: f"{x:,.2f}")
    st.dataframe(fd, use_container_width=True, hide_index=True)

with tab3:
    assump = {
        "WACC": f"{WACC:.2%}",
        "Terminal Growth Rate": f"{TERMINAL_GROWTH:.2%}",
        "Risk-Free Rate (10Y UST)": f"{RISK_FREE_RATE:.2%}",
        "Effective Tax Rate (3yr avg)": f"{TAX_RATE:.2%}",
        "Shares Outstanding": f"{SHARES:.3f}B",
        "Net Debt": f"${NET_DEBT:,.2f}B",
        "Base Revenue Growth (Yr 1)": f"{REVENUE_GROWTH[0]:.2%}",
        "Revenue Growth Path": str([f"{x:.1%}" for x in REVENUE_GROWTH]),
        "EBIT Margin Path": str([f"{x:.1%}" for x in EBIT_MARGINS]),
        "Maintenance CapEx % Rev": f"{maint_capex_pct:.2%}",
        "Growth CapEx % Rev-Increment": f"{growth_capex_pct:.2%}",
        "D&A Rate (on PP&E)": f"{da_rate:.2%}",
        "DSO / DIO / DPO (base, days)": f"{dso_hist:.1f} / {dio_hist:.1f} / {dpo_hist:.1f}",
    }
    st.dataframe(
        pd.DataFrame(assump.items(), columns=["Assumption", "Value"]),
        use_container_width=True, hide_index=True
    )

st.markdown("""
<div class="disclaimer">
  Data: Yahoo Finance via yfinance &nbsp;·&nbsp; For analytical purposes only — not investment advice.
</div>
""", unsafe_allow_html=True)
