"""
================================================================================
  INSTITUTIONAL-GRADE DCF MODEL  |  JPMorgan Equity Research Style
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

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# ============================================================
# CHANGE THESE INPUTS
# ============================================================
TICKER         = "GOOG"
WACC           = 0.112
FORECAST_YEARS = 5
# ============================================================


# -------------------------------------------------------------
# YFINANCE API HELPERS
# -------------------------------------------------------------

def fmp_income(ticker, limit=4):
    t = yf.Ticker(ticker)
    inc = t.financials  # rows=line items, cols=dates
    if inc is None or inc.empty:
        return []
    inc = inc.T.sort_index().tail(limit)
    rows = []
    for date, row in inc.iterrows():
        ebitda_val = row.get("EBITDA", np.nan)
        op_income  = row.get("Operating Income", np.nan)
        da_val     = row.get("Reconciled Depreciation", row.get("Depreciation And Amortization", np.nan))
        # Fallback: EBIT + D&A
        if pd.isna(ebitda_val) and not pd.isna(op_income) and not pd.isna(da_val):
            ebitda_val = op_income + da_val
        rows.append({
            "calendarYear"     : str(date.year),
            "revenue"          : row.get("Total Revenue", np.nan),
            "operatingIncome"  : op_income,
            "ebitda"           : ebitda_val,
            "netIncome"        : row.get("Net Income", np.nan),
            "incomeTaxExpense" : row.get("Tax Provision", np.nan),
            "incomeBeforeTax"  : row.get("Pretax Income", np.nan),
            "interestExpense"  : row.get("Interest Expense", np.nan),
        })
    return rows


def fmp_cashflow(ticker, limit=4):
    t = yf.Ticker(ticker)
    cf = t.cashflow
    if cf is None or cf.empty:
        return []
    cf = cf.T.sort_index().tail(limit)
    rows = []
    for date, row in cf.iterrows():
        da_val = row.get(
            "Depreciation And Amortization",
            row.get("Reconciled Depreciation", np.nan)
        )
        rows.append({
            "calendarYear"                : str(date.year),
            "depreciationAndAmortization" : da_val,
            "capitalExpenditure"          : row.get("Capital Expenditure", np.nan),
            "operatingCashFlow"           : row.get("Operating Cash Flow", np.nan),
        })
    return rows


def fmp_balance(ticker, limit=4):
    t = yf.Ticker(ticker)
    bal = t.balance_sheet
    if bal is None or bal.empty:
        return []
    bal = bal.T.sort_index().tail(limit)
    rows = []
    for date, row in bal.iterrows():
        cash = row.get("Cash And Cash Equivalents",
               row.get("Cash Cash Equivalents And Short Term Investments", np.nan))
        rows.append({
            "calendarYear"              : str(date.year),
            "netReceivables"            : row.get("Net Receivables",
                                          row.get("Accounts Receivable", np.nan)),
            "inventory"                 : row.get("Inventory", np.nan),
            "accountPayables"           : row.get("Accounts Payable", np.nan),
            "propertyPlantEquipmentNet" : row.get("Net PPE", np.nan),
            "totalCurrentAssets"        : row.get("Current Assets", np.nan),
            "totalCurrentLiabilities"   : row.get("Current Liabilities", np.nan),
            "totalDebt"                 : row.get("Total Debt", np.nan),
            "cashAndCashEquivalents"    : cash,
        })
    return rows


def fmp_profile(ticker):
    info = yf.Ticker(ticker).info
    return {
        "companyName"   : info.get("longName", ticker),
        "sector"        : info.get("sector",   "N/A"),
        "industry"      : info.get("industry", "N/A"),
        "revenueGrowth" : info.get("revenueGrowth", np.nan),
    }


def fmp_quote(ticker):
    info = yf.Ticker(ticker).info
    return {
        "price"             : info.get("currentPrice") or info.get("regularMarketPrice", np.nan),
        "yearLow"           : info.get("fiftyTwoWeekLow",  np.nan),
        "yearHigh"          : info.get("fiftyTwoWeekHigh", np.nan),
        "sharesOutstanding" : info.get("sharesOutstanding", np.nan),
        "marketCap"         : info.get("marketCap", np.nan),
    }


def fmp_ratios(ticker):
    info = yf.Ticker(ticker).info
    return {
        "peRatioTTM" : info.get("trailingPE") or info.get("forwardPE", np.nan),
    }


def fmp_key_metrics(ticker):
    info = yf.Ticker(ticker).info
    return {
        "evToEbitdaTTM" : info.get("enterpriseToEbitda", np.nan),
    }


def fetch_treasury_rate():
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "GS10", "api_key": "demo",
                    "sort_order": "desc", "limit": 1, "file_type": "json"},
            timeout=10
        )
        val = r.json()["observations"][0]["value"]
        return float(val) / 100
    except Exception:
        return 0.04


# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------

def safe(val, fallback=np.nan):
    try:
        v = float(val)
        return v if np.isfinite(v) else fallback
    except Exception:
        return fallback

def to_b(val):
    return safe(val) / 1e9

def trailing_avg(series, n=3):
    return series.dropna().tail(n).mean()

def cagr(series, n):
    s = series.dropna()
    if len(s) < n + 1:
        n = len(s) - 1
    if n <= 0 or s.iloc[-n-1] <= 0:
        return np.nan
    return (s.iloc[-1] / s.iloc[-n-1]) ** (1/n) - 1

def print_header(title):
    w = 90
    print("\n" + "-" * w)
    print(f"  {title}")
    print("-" * w)

def print_section(title):
    print(f"\n  {'-'*85}")
    print(f"  {title}")
    print(f"  {'-'*85}")


# -------------------------------------------------------------
# 1. RISK-FREE RATE
# -------------------------------------------------------------
print("\n  Fetching 10Y Treasury yield...")
RISK_FREE_RATE  = fetch_treasury_rate()
TERMINAL_GROWTH = RISK_FREE_RATE
print(f"  10Y Treasury : {RISK_FREE_RATE:.2%}  (used as terminal growth rate)")


# -------------------------------------------------------------
# 2. FETCH RAW DATA FROM YFINANCE
# -------------------------------------------------------------
print(f"\n  Fetching yfinance data for {TICKER}...")

inc_raw  = fmp_income(TICKER)
cf_raw   = fmp_cashflow(TICKER)
bal_raw  = fmp_balance(TICKER)
profile  = fmp_profile(TICKER)
quote    = fmp_quote(TICKER)
ratios   = fmp_ratios(TICKER)
metrics  = fmp_key_metrics(TICKER)

if not inc_raw:
    raise ValueError(
        f"No income statement data for {TICKER}. "
        "Check that the ticker is valid and yfinance can reach Yahoo Finance."
    )

inc_raw  = list(reversed(inc_raw))
cf_raw   = list(reversed(cf_raw))
bal_raw  = list(reversed(bal_raw))

inc_years  = {r["calendarYear"] for r in inc_raw}
cf_years   = {r["calendarYear"] for r in cf_raw}
bal_years  = {r["calendarYear"] for r in bal_raw}
common_yrs = sorted(inc_years & cf_years & bal_years)

if not common_yrs:
    raise ValueError("Could not align income / cashflow / balance sheet years.")

inc_by_yr  = {r["calendarYear"]: r for r in inc_raw}
cf_by_yr   = {r["calendarYear"]: r for r in cf_raw}
bal_by_yr  = {r["calendarYear"]: r for r in bal_raw}


# -------------------------------------------------------------
# 3. BUILD HISTORICAL DataFrame
# -------------------------------------------------------------
rows = []
for yr in common_yrs:
    i = inc_by_yr[yr]
    c = cf_by_yr[yr]
    b = bal_by_yr[yr]

    revenue       = safe(i.get("revenue"))
    ebit          = safe(i.get("operatingIncome"))
    ebitda        = safe(i.get("ebitda"))
    net_income    = safe(i.get("netIncome"))
    tax_provision = safe(i.get("incomeTaxExpense"))
    pretax_income = safe(i.get("incomeBeforeTax"))
    int_exp       = abs(safe(i.get("interestExpense", 0)))

    da            = safe(c.get("depreciationAndAmortization"))
    capex         = abs(safe(c.get("capitalExpenditure", 0)))
    ocf           = safe(c.get("operatingCashFlow"))

    receivables   = safe(b.get("netReceivables"))
    inventory     = safe(b.get("inventory"))
    payables      = safe(b.get("accountPayables"))
    net_ppe       = safe(b.get("propertyPlantEquipmentNet"))
    cur_assets    = safe(b.get("totalCurrentAssets"))
    cur_liab      = safe(b.get("totalCurrentLiabilities"))
    total_debt    = safe(b.get("totalDebt"))
    cash          = safe(b.get("cashAndCashEquivalents"))

    # Fallback: if EBITDA still NaN, derive it
    if np.isnan(ebitda) and not np.isnan(ebit) and not np.isnan(da):
        ebitda = ebit + da

    rows.append({
        "Year"             : int(yr),
        "Revenue"          : revenue   / 1e9,
        "EBIT"             : ebit      / 1e9,
        "EBITDA"           : ebitda    / 1e9,
        "D&A"              : da        / 1e9,
        "CAPEX"            : capex     / 1e9,
        "OCF"              : ocf       / 1e9,
        "Tax_Provision"    : abs(tax_provision) / 1e9,
        "Pretax_Income"    : pretax_income / 1e9,
        "Net_Income"       : net_income / 1e9,
        "Interest_Expense" : int_exp   / 1e9,
        "Receivables"      : receivables / 1e9,
        "Inventory"        : inventory  / 1e9,
        "Payables"         : payables   / 1e9,
        "Net_PPE"          : net_ppe    / 1e9,
        "Current_Assets"   : cur_assets / 1e9,
        "Current_Liab"     : cur_liab   / 1e9,
        "Total_Debt"       : total_debt / 1e9,
        "Cash"             : cash       / 1e9,
    })

df = pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)

df["NWC"]           = df["Current_Assets"] - df["Current_Liab"]
df["DELTA_NWC"]     = df["NWC"].diff()
df["EBIT_Margin"]   = df["EBIT"]   / df["Revenue"].replace(0, np.nan)
df["EBITDA_Margin"] = df["EBITDA"] / df["Revenue"].replace(0, np.nan)
df["DA_pct"]        = df["D&A"]    / df["Revenue"].replace(0, np.nan)
df["Capex_pct"]     = df["CAPEX"]  / df["Revenue"].replace(0, np.nan)
df["NWC_pct"]       = df["NWC"]    / df["Revenue"].replace(0, np.nan)

df["Tax_Rate"] = np.where(
    (df["Pretax_Income"] > 0) & df["Tax_Provision"].notna(),
    df["Tax_Provision"] / df["Pretax_Income"],
    np.nan
)
df["DSO"] = (df["Receivables"] / df["Revenue"].replace(0, np.nan)) * 365
df["DIO"] = (df["Inventory"]   / df["Revenue"].replace(0, np.nan)) * 365
df["DPO"] = (df["Payables"]    / df["Revenue"].replace(0, np.nan)) * 365

df["NOPAT"]      = df["EBIT"] * (1 - df["Tax_Rate"])
df["FCFF"]       = df["NOPAT"] + df["D&A"] - df["CAPEX"] - df["DELTA_NWC"].fillna(0)
df["Rev_Growth"] = df["Revenue"].pct_change()


# -------------------------------------------------------------
# 4. MARKET / COMPANY DATA
# -------------------------------------------------------------
company_name   = profile.get("companyName", TICKER)
sector         = profile.get("sector",   "N/A")
industry       = profile.get("industry", "N/A")
current_price  = safe(quote.get("price"))
week52_low     = safe(quote.get("yearLow"))
week52_high    = safe(quote.get("yearHigh"))
shares_out     = safe(quote.get("sharesOutstanding", 0)) / 1e9
market_cap     = safe(quote.get("marketCap", 0))         / 1e9

latest_bal     = bal_by_yr[common_yrs[-1]]
NET_DEBT       = (safe(latest_bal.get("totalDebt", 0)) -
                  safe(latest_bal.get("cashAndCashEquivalents", 0))) / 1e9

SHARES         = shares_out if shares_out > 0 else market_cap / current_price

fwd_pe         = safe(ratios.get("peRatioTTM"))
ev_ebitda_mult = safe(metrics.get("evToEbitdaTTM"))
rev_growth_fwd = safe(profile.get("revenueGrowth"))

if np.isnan(rev_growth_fwd) or not (0.0 < rev_growth_fwd < 0.60):
    rev_growth_fwd = np.nan


# -------------------------------------------------------------
# 5. DERIVE FORECAST ASSUMPTIONS
# -------------------------------------------------------------
eff_tax  = trailing_avg(df["Tax_Rate"], 3)
TAX_RATE = float(np.clip(eff_tax, 0.10, 0.35))

margin_hist = trailing_avg(df["EBIT_Margin"], 3)
margin_last = df["EBIT_Margin"].iloc[-1]
margin_base = max(margin_hist, margin_last)
EBIT_MARGINS = []
m = margin_base
for _ in range(FORECAST_YEARS):
    m = min(m + 0.005, 0.35)
    EBIT_MARGINS.append(round(m, 4))

if not np.isnan(rev_growth_fwd):
    base_g = rev_growth_fwd
else:
    base_g = cagr(df["Revenue"], 3)
    if np.isnan(base_g):
        base_g = 0.08

floor_g = TERMINAL_GROWTH + 0.01
REVENUE_GROWTH = []
g = base_g
for _ in range(FORECAST_YEARS):
    REVENUE_GROWTH.append(round(max(g, floor_g), 4))
    g *= 0.88

da_pct_hist    = trailing_avg(df["DA_pct"],    3)
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

DSO_TREND, DIO_TREND, DPO_TREND = -0.5, -0.3, +0.5


# -------------------------------------------------------------
# 6. FORECAST ENGINE
# -------------------------------------------------------------
last_rev = df["Revenue"].iloc[-1]
last_yr  = df["Year"].iloc[-1]
ppe      = ppe_last
prev_rev = last_rev
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
    dso   = max(dso + DSO_TREND, 1)
    dio   = max(dio + DIO_TREND, 0)
    dpo   = max(dpo + DPO_TREND, 1)
    rec   = (dso/365)*rev; inv = (dio/365)*rev; pay = (dpo/365)*rev
    delta_nwc = (rec - prev_rec) + (inv - prev_inv) - (pay - prev_pay)
    fcff  = nopat + da - capex - delta_nwc
    forecast_rows.append({"Year":yr,"Revenue":rev,"Rev_Growth":REVENUE_GROWTH[i],
        "EBIT":ebit,"EBIT_Margin":EBIT_MARGINS[i],"EBITDA":ebitda,
        "D&A":da,"CAPEX":capex,"NOPAT":nopat,"DELTA_NWC":delta_nwc,
        "DSO":dso,"DIO":dio,"DPO":dpo,"FCFF":fcff})
    prev_rev=rev; prev_rec=rec; prev_inv=inv; prev_pay=pay

fc = pd.DataFrame(forecast_rows)
fc["Year_Index"]      = range(1, FORECAST_YEARS + 1)
fc["Discount_Factor"] = 1 / (1 + WACC) ** fc["Year_Index"]
fc["PV_FCFF"]         = fc["FCFF"] * fc["Discount_Factor"]


# -------------------------------------------------------------
# 7. TERMINAL VALUE
# -------------------------------------------------------------
last_fcff    = fc["FCFF"].iloc[-1]
TV           = (last_fcff * (1 + TERMINAL_GROWTH)) / (WACC - TERMINAL_GROWTH)
PV_TV        = TV / (1 + WACC) ** fc["Year_Index"].iloc[-1]
sum_pv_fcff  = fc["PV_FCFF"].sum()
EV           = sum_pv_fcff + PV_TV
EQUITY_VAL   = EV - NET_DEBT
PRICE_TARGET = EQUITY_VAL / SHARES
TV_PCT       = PV_TV / EV * 100


# -------------------------------------------------------------
# 8. FOOTBALL FIELD
# -------------------------------------------------------------
last_ni      = df["Net_Income"].iloc[-1]
fwd_ebitda   = fc["EBITDA"].iloc[0]
ev_mult_low  = ev_ebitda_mult * 0.80 if not np.isnan(ev_ebitda_mult) else np.nan
ev_mult_high = ev_ebitda_mult * 1.20 if not np.isnan(ev_ebitda_mult) else np.nan
price_ev_low = ((fwd_ebitda * ev_mult_low  - NET_DEBT) / SHARES) if not np.isnan(ev_mult_low)  else np.nan
price_ev_hi  = ((fwd_ebitda * ev_mult_high - NET_DEBT) / SHARES) if not np.isnan(ev_mult_high) else np.nan
pe_low       = fwd_pe * 0.85 * (last_ni / SHARES) if not np.isnan(fwd_pe) else np.nan
pe_high      = fwd_pe * 1.15 * (last_ni / SHARES) if not np.isnan(fwd_pe) else np.nan


# -------------------------------------------------------------
# 9. SENSITIVITY MATRIX
# -------------------------------------------------------------
STEP, POINTS = 0.005, 5
wacc_range   = np.round(np.linspace(WACC - STEP*POINTS, WACC + STEP*POINTS, 2*POINTS+1), 4)
growth_range = np.round(np.linspace(TERMINAL_GROWTH - STEP*POINTS,
                                    TERMINAL_GROWTH + STEP*POINTS, 2*POINTS+1), 4)
sens = pd.DataFrame(index=[f"{g:.2%}" for g in growth_range],
                    columns=[f"{w:.2%}" for w in wacc_range])
n = fc["Year_Index"].iloc[-1]
for g, gl in zip(growth_range, [f"{g:.2%}" for g in growth_range]):
    for w, wl in zip(wacc_range, [f"{w:.2%}" for w in wacc_range]):
        if w <= g:
            sens.loc[gl, wl] = "N/M"; continue
        tv_s    = (last_fcff * (1 + g)) / (w - g)
        pv_tv_s = tv_s / (1 + w) ** n
        pv_f_s  = (fc["FCFF"] / (1 + w) ** fc["Year_Index"]).sum()
        sens.loc[gl, wl] = f"${(pv_f_s + pv_tv_s - NET_DEBT)/SHARES:,.1f}"
sens = sens.sort_index(ascending=False)


# -------------------------------------------------------------
# 10. OUTPUT
# -------------------------------------------------------------
print_header(f"EQUITY RESEARCH  |  {company_name} ({TICKER})  |  DCF VALUATION")
print(f"  Sector: {sector}  |  Industry: {industry}")
print(f"  Current Price: ${current_price:,.2f}  |  52W Range: ${week52_low:,.2f} - ${week52_high:,.2f}")

print_section("KEY ASSUMPTIONS")
for label, val in [
    ("WACC",                         f"{WACC:.2%}"),
    ("Terminal Growth Rate (= RFR)", f"{TERMINAL_GROWTH:.2%}"),
    ("Effective Tax Rate (3yr avg)", f"{TAX_RATE:.2%}"),
    ("Shares Outstanding",           f"{SHARES:.3f}B"),
    ("Net Debt",                     f"${NET_DEBT:,.2f}B"),
    ("Base Revenue Growth (Yr 1)",   f"{REVENUE_GROWTH[0]:.2%}"),
    ("Revenue Growth Path",          str([f"{x:.1%}" for x in REVENUE_GROWTH])),
    ("EBIT Margin Path",             str([f"{x:.1%}" for x in EBIT_MARGINS])),
    ("Maintenance CapEx % Rev",      f"{maint_capex_pct:.2%}"),
    ("Growth CapEx % Rev-Increment", f"{growth_capex_pct:.2%}"),
    ("D&A Rate (on PP&E)",           f"{da_rate:.2%}"),
    ("DSO / DIO / DPO (base, days)", f"{dso_hist:.1f} / {dio_hist:.1f} / {dpo_hist:.1f}"),
]:
    print(f"  {label:<38} {val}")

print_section("HISTORICAL FINANCIALS  (USD Billions)")
hd = df[["Year","Revenue","Rev_Growth","EBIT","EBIT_Margin","EBITDA",
          "D&A","CAPEX","Tax_Rate","NOPAT","DELTA_NWC","FCFF"]].copy()
hd["Rev_Growth"]  = hd["Rev_Growth"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
hd["EBIT_Margin"] = hd["EBIT_Margin"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
hd["Tax_Rate"]    = hd["Tax_Rate"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
for col in ["Revenue","EBIT","EBITDA","D&A","CAPEX","NOPAT","DELTA_NWC","FCFF"]:
    hd[col] = hd[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) and np.isfinite(x) else "-")
print(hd.to_string(index=False))

print_section("FORECAST  (USD Billions)")
fd = fc[["Year","Revenue","Rev_Growth","EBIT","EBIT_Margin","EBITDA",
          "D&A","CAPEX","NOPAT","DSO","DIO","DPO","DELTA_NWC","FCFF","PV_FCFF"]].copy()
fd["Rev_Growth"]  = fd["Rev_Growth"].map(lambda x: f"{x:.1%}")
fd["EBIT_Margin"] = fd["EBIT_Margin"].map(lambda x: f"{x:.1%}")
for c in ["DSO","DIO","DPO"]: fd[c] = fd[c].map(lambda x: f"{x:.1f}")
for col in ["Revenue","EBIT","EBITDA","D&A","CAPEX","NOPAT","DELTA_NWC","FCFF","PV_FCFF"]:
    fd[col] = fd[col].map(lambda x: f"{x:,.2f}")
print(fd.to_string(index=False))

print_section("VALUATION BRIDGE  (USD Billions)")
for label, val in [
    ("PV of Forecast FCFFs",  f"${sum_pv_fcff:>12,.2f}B"),
    ("PV of Terminal Value",   f"${PV_TV:>12,.2f}B"),
    ("Terminal Value % of EV", f"{TV_PCT:>11.1f}%"),
    ("Enterprise Value",       f"${EV:>12,.2f}B"),
    ("Less: Net Debt",         f"${NET_DEBT:>12,.2f}B"),
    ("Equity Value",           f"${EQUITY_VAL:>12,.2f}B"),
    ("Shares Outstanding",     f"{SHARES:>12,.3f}B"),
]:
    print(f"  {label:<40} {val}")
print(f"\n  {'DCF PRICE TARGET':<40} ${PRICE_TARGET:>12,.2f}")
print(f"  {'   Current Price':<40} ${current_price:>12,.2f}")
if not np.isnan(current_price):
    updown = (PRICE_TARGET / current_price - 1) * 100
    print(f"  {'   Implied Upside / Downside':<40} {'+'if updown>=0 else '-'} {abs(updown):>10.1f}%")

print_section("FOOTBALL FIELD  (Price per Share, USD)")
rows_ff = []
if not np.isnan(week52_low):   rows_ff.append(("52-Week Trading Range", f"${week52_low:,.2f}", f"${week52_high:,.2f}"))
if not np.isnan(price_ev_low): rows_ff.append((f"EV/EBITDA ({ev_mult_low:.1f}x-{ev_mult_high:.1f}x)", f"${price_ev_low:,.2f}", f"${price_ev_hi:,.2f}"))
if not np.isnan(pe_low):       rows_ff.append((f"P/E ({fwd_pe*0.85:.1f}x-{fwd_pe*1.15:.1f}x fwd)", f"${pe_low:,.2f}", f"${pe_high:,.2f}"))
rows_ff.append(("DCF (Base Case)", f"${PRICE_TARGET:,.2f}", f"${PRICE_TARGET:,.2f}"))
print(f"\n  {'Methodology':<42} {'Low':>12}  {'High':>12}")
print(f"  {'-'*68}")
for label, lo, hi in rows_ff:
    print(f"  {label:<42} {lo:>12}  {hi:>12}")

print_section("SENSITIVITY  (Price per Share, rows=Terminal Growth, cols=WACC)")
print(sens.to_string())

print_section("SANITY CHECKS")
for label, val in [
    ("TV % of EV (target 50-80%)",    f"{TV_PCT:.1f}%  {'OK' if 40 < TV_PCT < 85 else 'REVIEW'}"),
    ("WACC > Terminal Growth",         'OK' if WACC > TERMINAL_GROWTH else 'INVALID'),
    ("Tax Rate (10-35%)",              f"{'OK' if 0.10 <= TAX_RATE <= 0.35 else 'REVIEW'}"),
    ("Final yr EBIT Margin",           f"{EBIT_MARGINS[-1]:.1%}"),
    ("Final yr Revenue Growth",        f"{REVENUE_GROWTH[-1]:.1%}"),
    ("Forecast FCFF all positive",     'OK' if (fc['FCFF'] > 0).all() else 'NEGATIVE FCFF YEARS'),
]:
    print(f"  {label:<45} {val}")

print("\n" + "-"*90)
print("  Data source: Yahoo Finance via yfinance")
print("  Model complete. Values in USD billions unless stated.")
print("  For analytical purposes only — not investment advice.")
print("-"*90 + "\n")
