"""
================================================================================
  INSTITUTIONAL-GRADE DCF MODEL  |  JPMorgan Equity Research Style
================================================================================
  Instructions:
    1. Set TICKER  -> any US-listed stock
    2. Set WACC    -> your cost of capital assumption
    3. Run: python jpmorgan_dcf.py

  Everything else is auto-extracted from yfinance + live 10Y Treasury.

  Fix applied: Rate-limit handling with retry + browser session headers.
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import time
import random
import requests
import pandas as pd
import numpy as np
import yfinance as yf

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# ============================================================
# CHANGE ONLY THESE TWO INPUTS
# ============================================================
TICKER        = "GOOG"
WACC          = 0.112        # your cost-of-capital assumption

FORECAST_YEARS = 5           # standard sell-side horizon
# ============================================================


# -------------------------------------------------------------
# RATE-LIMIT SAFE FETCH WRAPPER
# -------------------------------------------------------------

def make_session():
    """Create a requests Session that looks like a real browser.
    This dramatically reduces Yahoo Finance rate-limit rejections."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
    })
    return session


def fetch_ticker(symbol, max_retries=5, base_delay=2.0):
    """Return a yf.Ticker object, retrying on rate-limit errors."""
    session = make_session()
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            tk = yf.Ticker(symbol, session=session)
            # Trigger a lightweight call to confirm the session works
            _ = tk.fast_info
            return tk
        except Exception as exc:
            msg = str(exc).lower()
            is_rate_limit = any(k in msg for k in
                                ["rate limit", "429", "too many requests",
                                 "yfratelimiterror", "encountered an error"])
            if not is_rate_limit:
                raise   # non-rate-limit error -> surface immediately

            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"  [Rate limit] attempt {attempt}/{max_retries}. "
                  f"Retrying in {delay:.1f}s ...")
            time.sleep(delay)
            last_exc = exc

    raise RuntimeError(
        f"Still rate-limited after {max_retries} attempts for {symbol}.\n"
        "  Tips:\n"
        "    - Wait a few minutes and re-run.\n"
        "    - Use a VPN or different network.\n"
        "    - Reduce how often you call this script.\n"
        f"  Original error: {last_exc}"
    )


def safe_fast_info(tk, key, fallback=None):
    """Read a key from fast_info with a fallback."""
    try:
        return tk.fast_info[key]
    except Exception:
        return fallback


def fetch_statement(tk, attr, max_retries=5, base_delay=2.0):
    """Fetch a statement (financials / cashflow / balance_sheet) with retry."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            data = getattr(tk, attr)
            if data is not None and not data.empty:
                return data
            return data   # empty but valid
        except Exception as exc:
            msg = str(exc).lower()
            is_rate_limit = any(k in msg for k in
                                ["rate limit", "429", "too many requests",
                                 "yfratelimiterror", "encountered an error"])
            if not is_rate_limit:
                raise

            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"  [Rate limit / {attr}] attempt {attempt}/{max_retries}. "
                  f"Retrying in {delay:.1f}s ...")
            time.sleep(delay)
            last_exc = exc

    raise RuntimeError(
        f"Could not fetch '{attr}' after {max_retries} retries. "
        f"Original error: {last_exc}"
    )


# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------

def get_row(stmt, *keys):
    for k in keys:
        if k in stmt.index:
            return stmt.loc[k]
    return None

def to_b(series_val):
    try:
        return float(series_val) / 1e9
    except Exception:
        return np.nan

def safe_div(a, b, fallback=np.nan):
    try:
        return a / b if b != 0 else fallback
    except Exception:
        return fallback

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
# 1. FETCH RISK-FREE RATE  (10Y US Treasury -> terminal growth)
# -------------------------------------------------------------
print("\n  Fetching live Risk-Free Rate (^TNX)...")
try:
    rfr_tk         = fetch_ticker("^TNX")
    rfr_raw        = safe_fast_info(rfr_tk, "lastPrice")
    RISK_FREE_RATE = round(float(rfr_raw) / 100, 4)
    print(f"  10Y Treasury Yield : {RISK_FREE_RATE:.2%}")
except Exception as e:
    RISK_FREE_RATE = 0.04
    print(f"  Could not fetch RFR ({e}). Using fallback: {RISK_FREE_RATE:.2%}")

TERMINAL_GROWTH = RISK_FREE_RATE


# -------------------------------------------------------------
# 2. FETCH ALL RAW DATA
# -------------------------------------------------------------
print(f"\n  Fetching financial data for {TICKER} ...")
print("  (Using browser-like session + retry on rate limits)\n")

tk = fetch_ticker(TICKER)

# Small delay between statement fetches to avoid burst rate-limits
financials  = fetch_statement(tk, "financials");   time.sleep(0.5)
cashflow    = fetch_statement(tk, "cashflow");     time.sleep(0.5)
balance     = fetch_statement(tk, "balance_sheet"); time.sleep(0.5)
info        = tk.info   # dict -- single call, no retry needed

# -- Income statement rows ----------------------------------
rev_row     = get_row(financials, "Total Revenue", "Revenue")
ebit_row    = get_row(financials, "Operating Income", "EBIT",
                      "Operating Income Loss", "Ebit")
ebitda_row  = get_row(financials, "EBITDA", "Normalized EBITDA",
                      "Reconciled Depreciation")
tax_row     = get_row(financials, "Tax Provision", "Income Tax Expense",
                      "Income Before Tax", "Tax Effect Of Unusual Items")
pretax_row  = get_row(financials, "Pretax Income", "Income Before Tax",
                      "Pretax Income Loss Adjustment")
ni_row      = get_row(financials, "Net Income", "Net Income Common Stockholders",
                      "Net Income From Continuing Operations")
int_exp_row = get_row(financials, "Interest Expense",
                      "Interest Expense Non Operating",
                      "Net Interest Income")

# -- Cash flow rows -----------------------------------------
da_row      = get_row(cashflow, "Depreciation And Amortization",
                      "Depreciation Amortization Depletion",
                      "Depreciation And Amortization In Income Statement",
                      "Reconciled Depreciation", "Depreciation")
capex_row   = get_row(cashflow, "Capital Expenditure",
                      "Purchase Of Property Plant And Equipment",
                      "Capital Expenditures",
                      "Purchases Of Property And Equipment")
ocf_row     = get_row(cashflow, "Operating Cash Flow",
                      "Cash Flow From Continuing Operating Activities",
                      "Net Cash Provided By Operating Activities")

# -- Balance sheet rows ------------------------------------
ca_row      = get_row(balance, "Current Assets", "Total Current Assets",
                      "Current Assets Total")
cl_row      = get_row(balance, "Current Liabilities", "Total Current Liabilities",
                      "Current Liabilities Total")
rec_row     = get_row(balance, "Accounts Receivable", "Net Receivables",
                      "Receivables", "Accounts Receivable Net",
                      "Net Accounts Receivable")
inv_row     = get_row(balance, "Inventory", "Inventories",
                      "Finished Goods", "Net Inventory")
pay_row     = get_row(balance, "Accounts Payable", "Payables",
                      "Accounts Payable Current", "Payables And Accrued Expenses")
ppe_row     = get_row(balance, "Net PPE", "Property Plant And Equipment Net",
                      "Net Property Plant And Equipment",
                      "Properties", "Gross PPE")
debt_row    = get_row(balance, "Total Debt", "Long Term Debt And Capital Lease Obligation",
                      "Long Term Debt", "Short Long Term Debt Total")
cash_b_row  = get_row(balance, "Cash And Cash Equivalents",
                      "Cash Cash Equivalents And Short Term Investments",
                      "Cash And Short Term Investments",
                      "Cash Equivalents")

# -- Debug: print any rows still None ----------------------
_check = {
    "Revenue": rev_row, "EBIT": ebit_row, "D&A": da_row,
    "CapEx": capex_row, "Receivables": rec_row, "Inventory": inv_row,
    "Payables": pay_row, "Net PPE": ppe_row,
}
_missing = [k for k, v in _check.items() if v is None]
if _missing:
    print(f"  Could not find rows for: {_missing}")
    print("  Available balance sheet keys:", list(balance.index[:20]))
    print("  Available cashflow keys:", list(cashflow.index[:20]))
    print("  Available income stmt keys:", list(financials.index[:20]))

for name, row in [("Revenue", rev_row), ("EBIT", ebit_row),
                  ("D&A", da_row), ("CapEx", capex_row)]:
    if row is None:
        raise ValueError(
            f"Could not fetch mandatory row '{name}' for {TICKER}.\n"
            "  This may be a data availability issue with yfinance.\n"
            "  Try a different ticker, or wait and retry."
        )


# -------------------------------------------------------------
# 3. BUILD HISTORICAL DataFrame
# -------------------------------------------------------------
years  = sorted([d.year for d in rev_row.index])
dates  = {yr: next(d for d in rev_row.index if d.year == yr) for yr in years}

def hist_val(row, yr, abs_val=False):
    if row is None:
        return np.nan
    v = to_b(row[dates[yr]])
    return abs(v) if abs_val else v

rows = []
for yr in years:
    rows.append({
        "Year"             : yr,
        "Revenue"          : hist_val(rev_row,     yr),
        "EBIT"             : hist_val(ebit_row,     yr),
        "EBITDA"           : hist_val(ebitda_row,   yr),
        "D&A"              : hist_val(da_row,       yr, abs_val=True),
        "CAPEX"            : hist_val(capex_row,    yr, abs_val=True),
        "OCF"              : hist_val(ocf_row,      yr),
        "Tax_Provision"    : hist_val(tax_row,      yr, abs_val=True),
        "Pretax_Income"    : hist_val(pretax_row,   yr),
        "Net_Income"       : hist_val(ni_row,       yr),
        "Interest_Expense" : hist_val(int_exp_row,  yr, abs_val=True),
        "Receivables"      : hist_val(rec_row,      yr),
        "Inventory"        : hist_val(inv_row,      yr),
        "Payables"         : hist_val(pay_row,      yr),
        "Net_PPE"          : hist_val(ppe_row,      yr),
        "Current_Assets"   : hist_val(ca_row,       yr),
        "Current_Liab"     : hist_val(cl_row,       yr),
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

df["NOPAT"] = df["EBIT"] * (1 - df["Tax_Rate"])
df["FCFF"]  = df["NOPAT"] + df["D&A"] - df["CAPEX"] - df["DELTA_NWC"].fillna(0)
df["Rev_Growth"] = df["Revenue"].pct_change()


# -------------------------------------------------------------
# 4. DERIVE FORECAST ASSUMPTIONS
# -------------------------------------------------------------
eff_tax  = trailing_avg(df["Tax_Rate"], 3)
TAX_RATE = float(np.clip(eff_tax, 0.10, 0.35))

margin_hist = trailing_avg(df["EBIT_Margin"], 3)
margin_last = df["EBIT_Margin"].iloc[-1]
margin_base = max(margin_hist, margin_last)

EBIT_MARGINS = []
m = margin_base
for i in range(FORECAST_YEARS):
    m = min(m + 0.005, 0.35)
    EBIT_MARGINS.append(round(m, 4))

fwd_growth = info.get("revenueGrowth")
if fwd_growth and 0.0 < fwd_growth < 0.60:
    base_g = fwd_growth
else:
    base_g = cagr(df["Revenue"], 3)
    if np.isnan(base_g):
        base_g = 0.08

floor_g = TERMINAL_GROWTH + 0.01
REVENUE_GROWTH = []
g = base_g
for i in range(FORECAST_YEARS):
    REVENUE_GROWTH.append(round(max(g, floor_g), 4))
    g *= 0.88

da_pct_hist    = trailing_avg(df["DA_pct"],    3)
capex_pct_hist = trailing_avg(df["Capex_pct"], 3)

if np.isnan(capex_pct_hist):
    capex_pct_hist = da_pct_hist + 0.05 if not np.isnan(da_pct_hist) else 0.10
if np.isnan(da_pct_hist):
    da_pct_hist = trailing_avg(df["D&A"] / df["Revenue"].replace(0, np.nan), 3)
if np.isnan(da_pct_hist):
    da_pct_hist = 0.07

maint_capex_pct  = da_pct_hist
growth_capex_pct = max(capex_pct_hist - maint_capex_pct, 0)

ppe_last = df["Net_PPE"].iloc[-1]
if np.isnan(ppe_last):
    ppe_last = df["CAPEX"].sum() * 0.5

da_rate = trailing_avg(df["D&A"] / df["Net_PPE"].replace(0, np.nan), 3)
if np.isnan(da_rate) or da_rate <= 0:
    da_rate = da_pct_hist

dso_hist = trailing_avg(df["DSO"], 3)
dio_hist = trailing_avg(df["DIO"], 3)
dpo_hist = trailing_avg(df["DPO"], 3)

if np.isnan(dso_hist): dso_hist = 30.0
if np.isnan(dio_hist): dio_hist = 45.0
if np.isnan(dpo_hist): dpo_hist = 45.0

DSO_TREND = -0.5
DIO_TREND = -0.3
DPO_TREND = +0.5

shares_raw = (info.get("sharesOutstanding")
              or info.get("impliedSharesOutstanding") or 0)
SHARES     = shares_raw / 1e9

if debt_row is not None and cash_b_row is not None:
    NET_DEBT = to_b(debt_row.iloc[0]) - to_b(cash_b_row.iloc[0])
else:
    NET_DEBT = ((info.get("totalDebt") or 0) - (info.get("totalCash") or 0)) / 1e9

current_price = info.get("currentPrice") or info.get("regularMarketPrice") or np.nan
week52_low    = info.get("fiftyTwoWeekLow")  or np.nan
week52_high   = info.get("fiftyTwoWeekHigh") or np.nan
fwd_pe        = info.get("forwardPE")        or np.nan
trailing_pe   = info.get("trailingPE")       or np.nan
ev_ebitda     = info.get("enterpriseToEbitda") or np.nan
sector        = info.get("sector",   "N/A")
industry      = info.get("industry", "N/A")
company_name  = info.get("longName", TICKER)


# -------------------------------------------------------------
# 5. FORECAST ENGINE
# -------------------------------------------------------------
last_rev = df["Revenue"].iloc[-1]
last_yr  = df["Year"].iloc[-1]
ppe      = ppe_last if not np.isnan(ppe_last) else (last_rev * da_pct_hist / da_rate)
prev_rev = last_rev

dso = dso_hist
dio = dio_hist
dpo = dpo_hist

prev_rec = df["Receivables"].iloc[-1]
prev_inv = df["Inventory"].iloc[-1]
prev_pay = df["Payables"].iloc[-1]
if np.isnan(prev_rec): prev_rec = (dso_hist / 365) * last_rev
if np.isnan(prev_inv): prev_inv = (dio_hist / 365) * last_rev
if np.isnan(prev_pay): prev_pay = (dpo_hist / 365) * last_rev

forecast_rows = []
for i in range(FORECAST_YEARS):
    yr  = last_yr + i + 1
    rev = prev_rev * (1 + REVENUE_GROWTH[i])

    ebit  = rev * EBIT_MARGINS[i]
    nopat = ebit * (1 - TAX_RATE)

    capex  = rev * maint_capex_pct + (rev - prev_rev) * growth_capex_pct
    da     = ppe * da_rate
    ppe    = ppe + capex - da
    ebitda = ebit + da

    dso = max(dso + DSO_TREND, 1)
    dio = max(dio + DIO_TREND, 0)
    dpo = max(dpo + DPO_TREND, 1)

    rec = (dso / 365) * rev
    inv = (dio / 365) * rev
    pay = (dpo / 365) * rev

    delta_nwc = (rec - prev_rec) + (inv - prev_inv) - (pay - prev_pay)
    fcff      = nopat + da - capex - delta_nwc

    forecast_rows.append({
        "Year": yr, "Revenue": rev, "Rev_Growth": REVENUE_GROWTH[i],
        "EBIT": ebit, "EBIT_Margin": EBIT_MARGINS[i],
        "EBITDA": ebitda, "D&A": da, "CAPEX": capex,
        "NOPAT": nopat, "DELTA_NWC": delta_nwc,
        "DSO": dso, "DIO": dio, "DPO": dpo, "FCFF": fcff,
    })

    prev_rev = rev
    prev_rec = rec
    prev_inv = inv
    prev_pay = pay

fc = pd.DataFrame(forecast_rows)


# -------------------------------------------------------------
# 6. DISCOUNTING
# -------------------------------------------------------------
fc["Year_Index"]      = range(1, FORECAST_YEARS + 1)
fc["Discount_Factor"] = 1 / (1 + WACC) ** fc["Year_Index"]
fc["PV_FCFF"]         = fc["FCFF"] * fc["Discount_Factor"]


# -------------------------------------------------------------
# 7. TERMINAL VALUE
# -------------------------------------------------------------
last_fcff   = fc["FCFF"].iloc[-1]
TV          = (last_fcff * (1 + TERMINAL_GROWTH)) / (WACC - TERMINAL_GROWTH)
PV_TV       = TV / (1 + WACC) ** fc["Year_Index"].iloc[-1]
sum_pv_fcff = fc["PV_FCFF"].sum()
EV          = sum_pv_fcff + PV_TV
EQUITY_VAL  = EV - NET_DEBT
PRICE_TARGET = EQUITY_VAL / SHARES
TV_PCT      = PV_TV / EV * 100


# -------------------------------------------------------------
# 8. TRADING MULTIPLES
# -------------------------------------------------------------
last_ni      = df["Net_Income"].iloc[-1]
fwd_ebitda   = fc["EBITDA"].iloc[0]

ev_mult_low  = ev_ebitda * 0.80 if not np.isnan(ev_ebitda) else np.nan
ev_mult_high = ev_ebitda * 1.20 if not np.isnan(ev_ebitda) else np.nan
ev_low       = fwd_ebitda * ev_mult_low  if not np.isnan(ev_mult_low)  else np.nan
ev_high      = fwd_ebitda * ev_mult_high if not np.isnan(ev_mult_high) else np.nan
price_ev_low = (ev_low  - NET_DEBT) / SHARES if not np.isnan(ev_low)  else np.nan
price_ev_hi  = (ev_high - NET_DEBT) / SHARES if not np.isnan(ev_high) else np.nan

pe_low  = fwd_pe * 0.85 * (last_ni / SHARES) if not np.isnan(fwd_pe) else np.nan
pe_high = fwd_pe * 1.15 * (last_ni / SHARES) if not np.isnan(fwd_pe) else np.nan


# -------------------------------------------------------------
# 9. SENSITIVITY MATRIX  (11 x 11)
# -------------------------------------------------------------
STEP   = 0.005
POINTS = 5

wacc_range   = np.round(np.linspace(WACC - STEP*POINTS, WACC + STEP*POINTS, 2*POINTS+1), 4)
growth_range = np.round(np.linspace(TERMINAL_GROWTH - STEP*POINTS,
                                    TERMINAL_GROWTH + STEP*POINTS, 2*POINTS+1), 4)

wacc_labels  = [f"{w:.2%}" for w in wacc_range]
g_labels     = [f"{g:.2%}" for g in growth_range]
sens         = pd.DataFrame(index=g_labels, columns=wacc_labels)

n = fc["Year_Index"].iloc[-1]
for g, gl in zip(growth_range, g_labels):
    for w, wl in zip(wacc_range, wacc_labels):
        if w <= g:
            sens.loc[gl, wl] = "N/M"
            continue
        tv_s    = (last_fcff * (1 + g)) / (w - g)
        pv_tv_s = tv_s / (1 + w) ** n
        pv_f_s  = (fc["FCFF"] / (1 + w) ** fc["Year_Index"]).sum()
        price_s = (pv_f_s + pv_tv_s - NET_DEBT) / SHARES
        sens.loc[gl, wl] = f"${price_s:,.1f}"

sens = sens.sort_index(ascending=False)


# -------------------------------------------------------------
# 10. OUTPUT
# -------------------------------------------------------------

print_header(f"EQUITY RESEARCH  |  {company_name} ({TICKER})  |  DCF VALUATION")
print(f"  Sector: {sector}  |  Industry: {industry}")
print(f"  Current Price: ${current_price:,.2f}  |  "
      f"52W Range: ${week52_low:,.2f} - ${week52_high:,.2f}")

print_section("KEY ASSUMPTIONS")
print(f"  {'WACC':<35} {WACC:.2%}")
print(f"  {'Terminal Growth Rate (= RFR)':<35} {TERMINAL_GROWTH:.2%}")
print(f"  {'Effective Tax Rate (3yr avg)':<35} {TAX_RATE:.2%}")
print(f"  {'Shares Outstanding':<35} {SHARES:.3f}B")
print(f"  {'Net Debt':<35} ${NET_DEBT:,.2f}B")
print(f"  {'Base Revenue Growth (Yr 1)':<35} {REVENUE_GROWTH[0]:.2%}")
print(f"  {'Revenue Growth Path':<35} {[f'{x:.1%}' for x in REVENUE_GROWTH]}")
print(f"  {'EBIT Margin Path':<35} {[f'{x:.1%}' for x in EBIT_MARGINS]}")
print(f"  {'Maintenance CapEx % Rev':<35} {maint_capex_pct:.2%}")
print(f"  {'Growth CapEx % Rev-Increment':<35} {growth_capex_pct:.2%}")
print(f"  {'D&A Rate (on PP&E)':<35} {da_rate:.2%}")
print(f"  {'DSO / DPO / DIO (base, days)':<35} {dso_hist:.1f} / {dpo_hist:.1f} / {dio_hist:.1f}")

print_section("HISTORICAL FINANCIALS  (USD Billions)")
hist_disp = df[["Year","Revenue","Rev_Growth","EBIT","EBIT_Margin",
                 "EBITDA","D&A","CAPEX","Tax_Rate","NOPAT","DELTA_NWC","FCFF"]].copy()
hist_disp["Rev_Growth"]  = hist_disp["Rev_Growth"].map(lambda x: f"{x:.1%}" if not np.isnan(x) else "-")
hist_disp["EBIT_Margin"] = hist_disp["EBIT_Margin"].map(lambda x: f"{x:.1%}" if not np.isnan(x) else "-")
hist_disp["Tax_Rate"]    = hist_disp["Tax_Rate"].map(lambda x: f"{x:.1%}" if not np.isnan(x) else "-")
for col in ["Revenue","EBIT","EBITDA","D&A","CAPEX","NOPAT","DELTA_NWC","FCFF"]:
    hist_disp[col] = hist_disp[col].map(lambda x: f"{x:,.2f}" if not np.isnan(x) else "-")
print(hist_disp.to_string(index=False))

print_section("FORECAST  (USD Billions)")
fc_disp = fc[["Year","Revenue","Rev_Growth","EBIT","EBIT_Margin",
               "EBITDA","D&A","CAPEX","NOPAT","DSO","DIO","DPO",
               "DELTA_NWC","FCFF","PV_FCFF"]].copy()
fc_disp["Rev_Growth"]  = fc_disp["Rev_Growth"].map(lambda x: f"{x:.1%}")
fc_disp["EBIT_Margin"] = fc_disp["EBIT_Margin"].map(lambda x: f"{x:.1%}")
fc_disp["DSO"]         = fc_disp["DSO"].map(lambda x: f"{x:.1f}")
fc_disp["DIO"]         = fc_disp["DIO"].map(lambda x: f"{x:.1f}")
fc_disp["DPO"]         = fc_disp["DPO"].map(lambda x: f"{x:.1f}")
for col in ["Revenue","EBIT","EBITDA","D&A","CAPEX","NOPAT","DELTA_NWC","FCFF","PV_FCFF"]:
    fc_disp[col] = fc_disp[col].map(lambda x: f"{x:,.2f}")
print(fc_disp.to_string(index=False))

print_section("VALUATION BRIDGE  (USD Billions)")
print(f"  {'PV of Forecast FCFFs':<40} ${sum_pv_fcff:>12,.2f}B")
print(f"  {'PV of Terminal Value':<40} ${PV_TV:>12,.2f}B")
print(f"  {'Terminal Value % of EV':<40} {TV_PCT:>11.1f}%")
print(f"  {'Enterprise Value':<40} ${EV:>12,.2f}B")
print(f"  {'Less: Net Debt':<40} ${NET_DEBT:>12,.2f}B")
print(f"  {'Equity Value':<40} ${EQUITY_VAL:>12,.2f}B")
print(f"  {'Shares Outstanding':<40} {SHARES:>12,.3f}B")
print(f"\n  {'DCF PRICE TARGET':<40} ${PRICE_TARGET:>12,.2f}")
print(f"  {'   Current Price':<40} ${current_price:>12,.2f}")
if not np.isnan(current_price):
    updown = (PRICE_TARGET / current_price - 1) * 100
    arrow  = "+" if updown >= 0 else "-"
    print(f"  {'   Implied Upside / Downside':<40} {arrow} {abs(updown):>10.1f}%")

print_section("FOOTBALL FIELD VALUATION SUMMARY  (Price per Share, USD)")
rows_ff = []
if not np.isnan(week52_low):
    rows_ff.append(("52-Week Trading Range", f"${week52_low:,.2f}", f"${week52_high:,.2f}"))
if not np.isnan(price_ev_low):
    rows_ff.append((f"EV/EBITDA  ({ev_mult_low:.1f}x - {ev_mult_high:.1f}x)",
                    f"${price_ev_low:,.2f}", f"${price_ev_hi:,.2f}"))
if not np.isnan(pe_low):
    rows_ff.append((f"P/E  ({fwd_pe*0.85:.1f}x - {fwd_pe*1.15:.1f}x fwd)",
                    f"${pe_low:,.2f}", f"${pe_high:,.2f}"))
rows_ff.append(("DCF (Base Case)", f"${PRICE_TARGET:,.2f}", f"${PRICE_TARGET:,.2f}"))
print(f"\n  {'Methodology':<40} {'Low':>12}  {'High':>12}")
print(f"  {'-'*66}")
for label, lo, hi in rows_ff:
    print(f"  {label:<40} {lo:>12}  {hi:>12}")

print_section(f"SENSITIVITY ANALYSIS  --  Price per Share")
print(f"  Base Case: WACC = {WACC:.2%}  |  Terminal Growth = {TERMINAL_GROWTH:.2%}\n")
print(f"  Rows = Terminal Growth Rate  |  Columns = WACC\n")
print(sens.to_string())

print_section("MODEL SANITY CHECKS")
checks = {
    "TV % of EV  (target: 50-80%)"       : f"{TV_PCT:.1f}%   {'OK' if 40 < TV_PCT < 85 else 'review'}",
    "WACC > Terminal Growth"              : f"{'OK' if WACC > TERMINAL_GROWTH else 'INVALID'}",
    "Tax Rate reasonable (10-35%)"        : f"{'OK' if 0.10 <= TAX_RATE <= 0.35 else 'review'}",
    "Final yr EBIT Margin"                : f"{EBIT_MARGINS[-1]:.1%}",
    "Final yr Revenue Growth"             : f"{REVENUE_GROWTH[-1]:.1%}",
    "Forecast FCFF all positive"          : f"{'OK' if (fc['FCFF'] > 0).all() else 'negative FCFF years'}",
}
for k, v in checks.items():
    print(f"  {k:<45} {v}")

print("\n" + "-"*90)
print("  Model complete. All values in USD billions unless stated otherwise.")
print("  This model is for analytical purposes only and does not constitute investment advice.")
print("-"*90 + "\n")
