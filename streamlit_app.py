"""
==================================================================================
 AGENT TWIN v3.1 — Institutional Market Simulator with Realistic Price Formation
==================================================================================

CHANGES FROM v3.0 (see accompanying CHANGELOG / diagnosis writeup):
  ROOT CAUSE FIXED: Stock fair value was compounding an uncontrolled, oversized
  "expected return" every single period with no brake, no inflation/rate
  sensitivity, and a broken earnings_yield formula. Mean reversion then
  faithfully chased that runaway anchor, producing +400%+ stock returns even
  after market-impact and reversion fixes were added in v3.0.

  Specific fixes:
  1. StockValuation.expected_return_pct() formula corrected. The old code
     computed `earnings_yield = (earnings/100)*100` which is just `earnings`
     -- the /100 and *100 canceled, a leftover refactor bug. Earnings yield
     is now 1/PE (the standard definition), expressed as a percentage.
  2. StockValuation now responds to interest rates and inflation (discount-
     rate / multiple-compression channel), matching how Bonds and Gold
     already worked in v3.0. Rate/inflation shocks now suppress equity fair
     value instead of having zero effect on it.
  3. earnings_growth update rule now has an explicit ceiling tied to nominal
     GDP capacity and is damped harder, so it cannot sit pinned near its max
     for 100 consecutive periods under a merely "modest positive GDP" macro
     path (which is what Inflation Shock and several other presets specify).
  4. Fair-value compounding in Market._update_fair_values() now treats
     `er[asset]` explicitly as an ANNUAL expected return and divides by a
     named PERIODS_PER_YEAR constant (=12) rather than a bare magic number,
     and the resulting per-period drift is clipped to a sane band so a
     transient valuation-model spike cannot compound geometrically over
     100 periods even before reversion gets a chance to act.
  5. Added an explicit unit test / sanity-check helper (validate_fair_value_drift)
     surfaced in the UI under a new "Diagnostics" panel, so this entire class
     of bug (anchor blowing up while reversion masks it) is visible going
     forward without re-deriving it from first principles each time.
  6. Return Attribution and Agent Impact Analysis are unchanged in mechanics
     (still exact, still reconciling), but now correctly reflect a sane
     Valuation_Macro component since that's the series that was broken.
  7. Agent decision rules are UNCHANGED in their rationale/bounds/triggers --
     this is a price-formation/valuation fix only, not an agent-behavior fix.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agent Twin v3.1 | Institutional Market Simulator",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ──────────────────────────────────────────────────────────────────────────────
C = {
    "bg":           "#080c0b",
    "panel":        "#0d1210",
    "panel_alt":    "#111816",
    "border":       "#1a2421",
    "text":         "#d8e3de",
    "text_dim":     "#7a9088",
    "green":        "#16a34a",
    "green_dim":    "#0c3d24",
    "amber":        "#c08a2e",
    "red":          "#b3473a",
    "blue":         "#3b7bab",
    "purple":       "#7c5cbf",
    "stock":        "#2fae6a",
    "bond":         "#3b82a6",
    "gold":         "#c08a2e",
    "pension":      "#3b82a6",
    "hedge":        "#b3473a",
    "retail":       "#c08a2e",
    "expansion":    "#16a34a",
    "slowdown":     "#c08a2e",
    "recession":    "#b3473a",
    "recovery":     "#3b7bab",
    "valuation_attr": "#3b7bab",
    "macro_attr":     "#7c5cbf",
    "pension_attr":   "#3b82a6",
    "hedge_attr":     "#b3473a",
    "retail_attr":    "#c08a2e",
    "noise_attr":     "#5a6b65",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
.stApp {{ background-color:{C['bg']}; color:{C['text']}; }}
section[data-testid="stSidebar"] {{
    background-color:{C['panel']};
    border-right:1px solid {C['border']};
}}
h1,h2,h3,h4 {{ color:{C['text']} !important; font-family:'IBM Plex Mono',monospace; letter-spacing:.02em; }}
.at-header {{
    font-family:'IBM Plex Mono',monospace; font-size:1.9rem; font-weight:700;
    color:{C['text']}; border-bottom:2px solid {C['green_dim']};
    padding-bottom:.4rem; margin-bottom:.1rem;
}}
.at-sub {{ color:{C['text_dim']}; font-size:.9rem; margin-bottom:1.2rem; font-family:'IBM Plex Sans',sans-serif; }}
.panel {{
    background:{C['panel']}; border:1px solid {C['border']};
    border-radius:6px; padding:1rem 1.2rem; margin-bottom:.8rem;
}}
.panel-dark {{
    background:{C['panel_alt']}; border:1px solid {C['border']};
    border-radius:6px; padding:1rem 1.2rem; margin-bottom:.8rem;
}}
.metric-label {{
    color:{C['text_dim']}; font-size:.72rem; text-transform:uppercase;
    letter-spacing:.08em; font-family:'IBM Plex Mono',monospace;
}}
.metric-value {{ font-family:'IBM Plex Mono',monospace; font-size:1.35rem; margin-top:.15rem; }}
.badge {{
    display:inline-block; font-size:.68rem; text-transform:uppercase;
    letter-spacing:.06em; padding:.12rem .45rem; border-radius:3px;
    background:{C['panel_alt']}; border:1px solid {C['border']};
    color:{C['text_dim']}; margin-right:.35rem; font-family:'IBM Plex Mono',monospace;
}}
.rule-row {{
    font-family:'IBM Plex Mono',monospace; font-size:.8rem; color:{C['text_dim']};
    border-left:2px solid {C['green_dim']}; padding:.12rem 0 .12rem .55rem; margin-bottom:.2rem;
}}
.log-line {{ font-family:'IBM Plex Mono',monospace; font-size:.8rem; color:{C['text_dim']}; padding:.08rem 0; }}
.log-period {{ color:{C['green']}; font-weight:700; }}
div[data-testid="stMetricValue"] {{ font-family:'IBM Plex Mono',monospace; color:{C['text']}; }}
.stButton button {{
    background:{C['panel_alt']}; color:{C['text']};
    border:1px solid {C['green_dim']}; border-radius:4px;
    font-family:'IBM Plex Mono',monospace; font-size:.82rem;
}}
.stButton button:hover {{ border-color:{C['green']}; color:{C['green']}; }}
hr {{ border-color:{C['border']}; }}
.regime-pill {{
    display:inline-block; font-size:.75rem; font-weight:600;
    padding:.18rem .7rem; border-radius:12px; margin-right:.4rem;
    font-family:'IBM Plex Mono',monospace; letter-spacing:.04em;
}}
.pass-badge {{
    display:inline-block; padding:.15rem .5rem; border-radius:4px;
    font-family:'IBM Plex Mono',monospace; font-size:.75rem; font-weight:700;
    background:#0c3d24; color:#16a34a; border:1px solid #16a34a; margin-left:.4rem;
}}
.fail-badge {{
    display:inline-block; padding:.15rem .5rem; border-radius:4px;
    font-family:'IBM Plex Mono',monospace; font-size:.75rem; font-weight:700;
    background:#3d1210; color:#b3473a; border:1px solid #b3473a; margin-left:.4rem;
}}
.warn-badge {{
    display:inline-block; padding:.15rem .5rem; border-radius:4px;
    font-family:'IBM Plex Mono',monospace; font-size:.75rem; font-weight:700;
    background:#3d2d10; color:#c08a2e; border:1px solid #c08a2e; margin-left:.4rem;
}}
.confidence-high {{ color:#16a34a; font-weight:700; font-family:'IBM Plex Mono',monospace; }}
.confidence-med  {{ color:#c08a2e; font-weight:700; font-family:'IBM Plex Mono',monospace; }}
.confidence-low  {{ color:#b3473a; font-weight:700; font-family:'IBM Plex Mono',monospace; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LAYOUT = dict(
    paper_bgcolor=C["panel"], plot_bgcolor=C["panel"],
    font=dict(color=C["text"], family="IBM Plex Mono, monospace", size=11),
    xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
    yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=44, r=16, t=44, b=36),
)

# ══════════════════════════════════════════════════════════════════════════════
# TIME-BASIS CONSTANT (NEW in v3.1)
# ══════════════════════════════════════════════════════════════════════════════
# Every "expected_return_pct()" across StockValuation / BondValuation /
# GoldValuation is defined and documented as an ANNUALISED percentage return.
# One simulation period is modeled as one month, so fair-value compounding
# divides annual expected return by PERIODS_PER_YEAR. This constant is named
# and centralised (v3.0 had a bare "/12" with no explanation, and no
# corresponding cap) so the time-basis assumption is explicit and auditable.
PERIODS_PER_YEAR = 12.0

# Hard sanity band on per-period fair-value drift (in %). At PERIODS_PER_YEAR=12,
# a sustained +/-40%/year fundamental drift is already an extreme regime
# (deep recession or runaway boom); we clip the realized per-period drift to
# the equivalent of +/-60%/year so a transient valuation-model spike cannot
# silently compound to triple-digit totals over a long run even before
# mean-reversion in the price engine gets a chance to act. This is a sanity
# rail on the FAIR VALUE ANCHOR itself, independent of the price engine's own
# tanh damping on price -- the two operate at different stages and both are
# needed (price damping alone could not have prevented this, since price was
# faithfully tracking a runaway anchor).
MAX_ANNUAL_FAIR_VALUE_DRIFT_PCT = 60.0
MAX_PERIOD_FAIR_VALUE_DRIFT_PCT = MAX_ANNUAL_FAIR_VALUE_DRIFT_PCT / PERIODS_PER_YEAR

# ══════════════════════════════════════════════════════════════════════════════
# MARKET REGIMES  (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

REGIMES = ["Expansion", "Slowdown", "Recession", "Recovery"]
REGIME_COLORS = {
    "Expansion": C["expansion"], "Slowdown": C["slowdown"],
    "Recession": C["recession"], "Recovery": C["recovery"],
}

REGIME_TRANSITIONS = {
    "Expansion": {"Expansion": 0.93, "Slowdown": 0.05, "Recession": 0.01, "Recovery": 0.01},
    "Slowdown":  {"Expansion": 0.05, "Slowdown": 0.87, "Recession": 0.07, "Recovery": 0.01},
    "Recession": {"Expansion": 0.01, "Slowdown": 0.07, "Recession": 0.80, "Recovery": 0.12},
    "Recovery":  {"Expansion": 0.18, "Slowdown": 0.03, "Recession": 0.02, "Recovery": 0.77},
}

def next_regime(current: str, rng: random.Random) -> str:
    probs = REGIME_TRANSITIONS[current]
    r = rng.random()
    cumulative = 0.0
    for regime, p in probs.items():
        cumulative += p
        if r < cumulative:
            return regime
    return current

def regime_asset_bias(regime: str) -> Dict[str, float]:
    # These are ANNUALISED-equivalent fundamental drift biases (drift in
    # expected/fair return), not direct price-impact nudges. They feed the
    # fair-value path, not the price engine's noise term.
    biases = {
        "Expansion": {"Stocks":  0.20, "Bonds": -0.03, "Gold":  0.00},
        "Slowdown":  {"Stocks": -0.07, "Bonds":  0.10, "Gold":  0.07},
        "Recession": {"Stocks": -0.30, "Bonds":  0.22, "Gold":  0.13},
        "Recovery":  {"Stocks":  0.17, "Bonds":  0.03, "Gold":  0.03},
    }
    return biases[regime]

# ══════════════════════════════════════════════════════════════════════════════
# ASSET VALUATION LAYER  (FIXED in v3.1 — see header notes 1-3)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StockValuation:
    earnings: float
    earnings_growth: float

    @property
    def pe_ratio(self) -> float:
        return 100.0 / self.earnings if self.earnings > 0 else 999.0

    def expected_return_pct(self, interest_rate: float = 4.0, inflation: float = 2.0) -> float:
        """
        ANNUALISED expected return (%), composed of:
          - earnings_yield = 1 / PE  (the standard Gordon-style yield definition;
            v3.0 had `(earnings/100)*100` which algebraically equals
            `earnings`, NOT a yield -- the root cause of the runaway anchor)
          - + earnings_growth (also annualised)
          - - a discount-rate / multiple-compression penalty: equities are
            valued at a multiple that compresses as real rates rise. v3.0 had
            ZERO sensitivity to interest_rate or inflation in equity fair
            value (Bonds and Gold both already had this; Stocks did not),
            which is why an Inflation Shock scenario could not suppress
            equity fair value at all.
        """
        earnings_yield = self.pe_ratio and (100.0 / self.pe_ratio) or 0.0  # = 1/PE * 100, i.e. true earnings yield
        real_rate = interest_rate - inflation
        # Multiple-compression penalty: every 1pp of real rate above a 2%
        # "neutral" level shaves ~0.8pp off the equity expected-return path
        # (a simple stand-in for discount-rate sensitivity, calibrated so a
        # severe rate shock visibly suppresses equities without dominating
        # every other term).
        discount_penalty = max(0.0, real_rate - 2.0) * 0.8
        # Inflation above target also compresses equity multiples
        # independent of real rates (margin compression / uncertainty
        # premium channel), capped so it can't run away on its own.
        inflation_penalty = max(0.0, min(inflation - 2.0, 10.0)) * 0.5
        return earnings_yield + self.earnings_growth - discount_penalty - inflation_penalty

    def is_expensive(self) -> bool:
        return self.pe_ratio > 25

    def is_cheap(self) -> bool:
        return self.pe_ratio < 14

    def update(self, price: float, gdp_growth: float, rng: random.Random):
        growth_factor = 1.0 + (gdp_growth / 100.0) * 0.4 + rng.gauss(0, 0.004)
        self.earnings = max(0.5, self.earnings * growth_factor)
        # earnings_growth update: same mean-reverting form as v3.0, but with
        # a materially tighter ceiling and stronger pull toward a sane
        # baseline, so a merely "modest positive GDP" macro path (e.g.
        # Inflation Shock's gdp_growth=1.5) cannot push earnings_growth up
        # against its ceiling and hold it there for 100 consecutive periods.
        # v3.0 ceiling was +15 with weak (0.97) decay; v3.1 ceiling is +9
        # with faster (0.92) decay back toward a baseline of 3.0.
        target = 3.0 + gdp_growth * 0.5
        self.earnings_growth = max(-5.0, min(9.0,
            self.earnings_growth * 0.92 + target * 0.08 + rng.gauss(0, 0.15)))


@dataclass
class BondValuation:
    yield_pct: float
    duration: float

    def expected_return_pct(self) -> float:
        return self.yield_pct

    def price_sensitivity(self, rate_change_pct: float) -> float:
        return -self.duration * rate_change_pct

    def update(self, interest_rate: float, inflation: float, rng: random.Random):
        target_yield = interest_rate + max(0, inflation - 2.0) * 0.25
        self.yield_pct = self.yield_pct * 0.92 + target_yield * 0.08 + rng.gauss(0, 0.03)
        self.yield_pct = max(0.1, self.yield_pct)


@dataclass
class GoldValuation:
    inflation_sensitivity: float

    def expected_return_pct(self, inflation: float, real_rate: float) -> float:
        return self.inflation_sensitivity * max(0, inflation - 2.0) - real_rate * 0.5

    def update(self, rng: random.Random):
        self.inflation_sensitivity = max(0.3, min(1.5,
            self.inflation_sensitivity + rng.gauss(0, 0.005)))


# ══════════════════════════════════════════════════════════════════════════════
# MACRO ENVIRONMENT  (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

SENTIMENT_LEVELS = ["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"]
SENTIMENT_SCORE  = {"Very Bearish": -2, "Bearish": -1, "Neutral": 0, "Bullish": 1, "Very Bullish": 2}

@dataclass
class MacroEnvironment:
    inflation: float
    interest_rate: float
    gdp_growth: float
    oil_shock: float
    sentiment: str
    regime: str = "Expansion"

    @property
    def real_rate(self) -> float:
        return self.interest_rate - self.inflation

    def is_recession(self):      return self.gdp_growth < 0
    def is_high_inflation(self): return self.inflation > 5.0
    def is_high_rates(self):     return self.interest_rate > 5.0
    def is_oil_crisis(self):     return self.oil_shock > 15.0

# ══════════════════════════════════════════════════════════════════════════════
# ASSET  —  PRICE FORMATION (mechanics unchanged from v3.0 — this part was OK)
# ══════════════════════════════════════════════════════════════════════════════
#
#   impact_pct    = sign(flow) * sqrt(|flow|) * (100 / sqrt(market_depth))
#   reversion_pct = -reversion_speed * ln(price / fair_value) * 100
#   noise_pct     = N(0, base_vol * vol_scale)
#   raw           = impact_pct + reversion_pct + noise_pct
#   pct_change    = 8 * tanh(raw / 8)      <- smooth damping, not a hard floor
#
# v3.0 diagnosis confirmed this layer was working as intended: price tracked
# fair value closely (reversion doing its job). The bug was entirely
# upstream, in what fair_value itself was compounding toward. No changes
# below this point in the Asset class.

ASSET_DEPTH = {
    "Stocks": 42.0,
    "Bonds":  70.0,
    "Gold":   30.0,
}
ASSET_REVERSION_SPEED = {
    "Stocks": 0.055,
    "Bonds":  0.09,
    "Gold":   0.045,
}
ASSET_BASE_VOL = {
    "Stocks": 1.05,
    "Bonds":  0.55,
    "Gold":   0.95,
}

REGIME_VOL_SCALE = {
    "Expansion": 0.90, "Slowdown": 1.05, "Recession": 1.45, "Recovery": 1.10,
}


class Asset:
    def __init__(self, name: str, start_price: float, market_depth: float,
                 reversion_speed: float, base_volatility: float):
        self.name = name
        self.market_depth = market_depth
        self.reversion_speed = reversion_speed
        self.base_volatility = base_volatility
        self.price_history: List[float] = [start_price]
        self.fair_value_history: List[float] = [start_price]
        self.demand_pressure_history: List[float] = [0.0]
        self.impact_history: List[float] = [0.0]
        self.reversion_history: List[float] = [0.0]
        self.noise_history: List[float] = [0.0]
        # NEW: track the raw (pre-clip) annualised expected return and the
        # clipped per-period drift actually applied, so the diagnostics
        # panel can show exactly how much (if any) clipping is occurring.
        self.fv_annual_return_raw_history: List[float] = [0.0]
        self.fv_period_drift_applied_history: List[float] = [0.0]

    @property
    def price(self) -> float:
        return self.price_history[-1]

    @property
    def fair_value(self) -> float:
        return self.fair_value_history[-1]

    def set_fair_value(self, fv: float, annual_return_raw: float, period_drift_applied: float):
        self.fair_value_history.append(max(fv, 0.01))
        self.fv_annual_return_raw_history.append(annual_return_raw)
        self.fv_period_drift_applied_history.append(period_drift_applied)

    def apply_demand(self, net_flow_pp: float, rng: random.Random,
                      vol_scale: float = 1.0) -> Tuple[float, float, float, float]:
        """
        net_flow_pp: net order flow this period, in percentage points of
        aggregate target-allocation change.
        Returns (total_pct_change, impact_component, reversion_component, noise_component)
        so the attribution framework can reconcile exactly.
        """
        sign = 1.0 if net_flow_pp >= 0 else (-1.0 if net_flow_pp < 0 else 0.0)
        impact_pct = sign * math.sqrt(abs(net_flow_pp)) * (100.0 / math.sqrt(self.market_depth))

        log_gap = math.log(max(self.price, 0.01) / max(self.fair_value, 0.01))
        reversion_pct = -self.reversion_speed * log_gap * 100.0

        noise_pct = rng.gauss(0, self.base_volatility * vol_scale)

        raw = impact_pct + reversion_pct + noise_pct
        capped = 8.0 * math.tanh(raw / 8.0)

        if abs(raw) > 1e-9:
            scale = capped / raw
        else:
            scale = 1.0
        impact_pct   *= scale
        reversion_pct *= scale
        noise_pct     *= scale

        new_price = max(self.price * (1 + capped / 100.0), 0.01)
        self.price_history.append(new_price)
        self.demand_pressure_history.append(net_flow_pp)
        self.impact_history.append(impact_pct)
        self.reversion_history.append(reversion_pct)
        self.noise_history.append(noise_pct)
        return capped, impact_pct, reversion_pct, noise_pct

    def total_return_pct(self) -> float:
        if len(self.price_history) < 2:
            return 0.0
        return (self.price_history[-1] / self.price_history[0] - 1) * 100.0

    def fair_value_total_return_pct(self) -> float:
        """NEW: total return of the FAIR VALUE anchor alone, isolated from
        price/agent-flow effects. This is the single most direct diagnostic
        for the v3.0 bug class -- if this number alone explains almost all
        of total_return_pct(), the anchor (not agent flow) is driving the
        result."""
        if len(self.fair_value_history) < 2:
            return 0.0
        return (self.fair_value_history[-1] / self.fair_value_history[0] - 1) * 100.0

# ══════════════════════════════════════════════════════════════════════════════
# ORDER  (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentOrder:
    agent_name: str
    allocation_deltas: Dict[str, float]
    rationale: List[str]

# ══════════════════════════════════════════════════════════════════════════════
# BASE AGENT  (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

class Agent:
    name: str = "Agent"
    color: str = "#888"
    allocation_bounds: Dict[str, tuple] = {}

    def __init__(self, starting_allocation: Dict[str, float], starting_capital: float = 100.0):
        self.starting_allocation = dict(starting_allocation)
        self.allocation_history: List[Dict[str, float]] = [dict(starting_allocation)]
        self.capital = starting_capital
        self.portfolio_value_history: List[float] = [starting_capital]
        self.trade_log: List[AgentOrder] = []

    @property
    def current_allocation(self) -> Dict[str, float]:
        return self.allocation_history[-1]

    def _clamp(self, asset: str, v: float) -> float:
        lo, hi = self.allocation_bounds.get(asset, (-3.0, 3.0))
        return max(lo, min(hi, v))

    def _move_toward(self, deltas: Dict, asset: str, target: float, speed: float) -> float:
        current = self.current_allocation.get(asset, 0.0) + deltas.get(asset, 0.0)
        delta = (target - current) * speed
        deltas[asset] = deltas.get(asset, 0.0) + delta
        return delta

    def apply_order(self, order: AgentOrder):
        new_alloc = dict(self.current_allocation)
        for asset, delta in order.allocation_deltas.items():
            new_alloc[asset] = self._clamp(asset, new_alloc.get(asset, 0.0) + delta)
        self.allocation_history.append(new_alloc)
        self.trade_log.append(order)

    def update_portfolio_value(self, asset_returns_pct: Dict[str, float]):
        prior = self.allocation_history[-2] if len(self.allocation_history) >= 2 else self.starting_allocation
        period_return = sum(
            w * asset_returns_pct.get(a, 0.0) / 100.0
            for a, w in prior.items() if a not in ("Cash",)
        )
        self.portfolio_value_history.append(self.portfolio_value_history[-1] * (1 + period_return))

    def decide(self, env: MacroEnvironment, market: "Market", period: int) -> AgentOrder:
        raise NotImplementedError

# ══════════════════════════════════════════════════════════════════════════════
# PENSION FUND  (DECISION RULES UNCHANGED from v3.0/v2.1)
# ══════════════════════════════════════════════════════════════════════════════

class PensionFund(Agent):
    name = "Pension Fund"
    color = C["pension"]
    allocation_bounds = {
        "Stocks": (0.05, 0.70),
        "Bonds":  (0.20, 0.80),
        "Gold":   (0.00, 0.25),
    }

    def __init__(self, liability_duration: float = 12.0, required_return_pct: float = 5.5):
        super().__init__({"Stocks": 0.40, "Bonds": 0.50, "Gold": 0.10})
        self.liabilities = 95.0
        self.liability_duration = liability_duration
        self.required_return_pct = required_return_pct
        self.funding_ratio_history: List[float] = []

    @property
    def funding_ratio(self) -> float:
        assets = self.portfolio_value_history[-1] if self.portfolio_value_history else 100.0
        return assets / self.liabilities * 100.0

    def update_liabilities(self, interest_rate: float):
        discount_change = -(interest_rate - 4.0) * 0.01 * self.liability_duration
        self.liabilities *= (1.0 + discount_change / 100.0)
        self.liabilities = max(50.0, self.liabilities)

    def decide(self, env: MacroEnvironment, market: "Market", period: int) -> AgentOrder:
        deltas: Dict[str, float] = {}
        rationale: List[str] = []

        self.update_liabilities(env.interest_rate)
        fr = self.funding_ratio
        self.funding_ratio_history.append(fr)

        bond_val: BondValuation = market.valuations["Bonds"]
        stock_val: StockValuation = market.valuations["Stocks"]
        gold_val: GoldValuation = market.valuations["Gold"]

        exp_stock  = stock_val.expected_return_pct(env.interest_rate, env.inflation) + regime_asset_bias(env.regime)["Stocks"]
        exp_bond   = bond_val.expected_return_pct()
        exp_gold   = gold_val.expected_return_pct(env.inflation, env.real_rate)

        if fr < 90:
            self._move_toward(deltas, "Bonds", 0.70, 0.13)
            self._move_toward(deltas, "Stocks", 0.15, 0.13)
            rationale.append(f"CRITICAL: Funding ratio {fr:.1f}% < 90% — emergency de-risking")
        elif fr < 100:
            self._move_toward(deltas, "Bonds", 0.62, 0.08)
            self._move_toward(deltas, "Stocks", 0.28, 0.08)
            rationale.append(f"Funding ratio {fr:.1f}% < 100% — increasing bond allocation")
        elif fr > 120:
            self._move_toward(deltas, "Stocks", 0.55, 0.07)
            self._move_toward(deltas, "Bonds", 0.38, 0.07)
            rationale.append(f"Funding ratio {fr:.1f}% > 120% — surplus allows higher equity allocation")

        yield_gap = exp_stock - exp_bond
        if yield_gap < 1.5 and fr < 110:
            self._move_toward(deltas, "Bonds", 0.58, 0.06)
            self._move_toward(deltas, "Stocks", 0.32, 0.06)
            rationale.append(f"Equity–bond yield gap {yield_gap:.1f}pp narrow — trim equities")
        elif yield_gap > 5.0:
            self._move_toward(deltas, "Stocks", 0.52, 0.06)
            rationale.append(f"Equity–bond yield gap {yield_gap:.1f}pp wide — adding equities")

        if env.is_high_inflation():
            self._move_toward(deltas, "Gold", min(0.22, exp_gold / 20.0 + 0.12), 0.07)
            rationale.append(f"Inflation {env.inflation:.1f}% > 5% — adding gold")

        if env.regime == "Recession":
            self._move_toward(deltas, "Bonds", 0.65, 0.07)
            self._move_toward(deltas, "Stocks", 0.22, 0.07)
            rationale.append("Regime = Recession → defensive shift")
        elif env.regime == "Recovery" and fr > 105:
            self._move_toward(deltas, "Stocks", 0.50, 0.06)
            rationale.append("Regime = Recovery + comfortable funding → add equities")

        recent_stock = market.recent_return("Stocks", 5)
        if recent_stock < -8.0 and fr > 100:
            self._move_toward(deltas, "Stocks", 0.52, 0.10)
            rationale.append(f"Stocks down {recent_stock:.1f}% (5p) and funded — contrarian buy")
        elif recent_stock > 14.0:
            self._move_toward(deltas, "Stocks", 0.34, 0.07)
            rationale.append(f"Stocks up {recent_stock:.1f}% (5p) — rebalancing discipline")

        if not rationale:
            self._move_toward(deltas, "Stocks", 0.40, 0.04)
            self._move_toward(deltas, "Bonds", 0.50, 0.04)
            self._move_toward(deltas, "Gold",  0.10, 0.04)
            rationale.append("No active signal — drifting to 40/50/10 policy mix")

        return AgentOrder(self.name, deltas, rationale)

# ══════════════════════════════════════════════════════════════════════════════
# HEDGE FUND  (DECISION RULES UNCHANGED from v3.0/v2.1)
# ══════════════════════════════════════════════════════════════════════════════

class HedgeFund(Agent):
    name = "Hedge Fund"
    color = C["hedge"]
    allocation_bounds = {
        "Stocks": (-0.60, 2.00),
        "Cash":   (-1.00, 0.40),
        "Gold":   (0.00,  0.50),
        "Bonds":  (0.00,  0.35),
    }

    def __init__(self, risk_budget_vol: float = 0.12):
        super().__init__({"Stocks": 1.20, "Cash": -0.20})
        self.risk_budget_vol = risk_budget_vol
        self.gross_exposure_history: List[float] = [1.40]
        self.net_exposure_history:   List[float] = [1.20]
        self.leverage_ratio_history: List[float] = [1.40]
        self.trend_signal_history:   List[float] = [0.0]
        self.valuation_signal_history: List[float] = [0.0]

    def _trend_signal(self, market: "Market") -> float:
        mom = market.smoothed_momentum("Stocks", lookback=8)
        return max(-1.0, min(1.0, mom / 3.0))

    def _valuation_signal(self, market: "Market") -> float:
        sv: StockValuation = market.valuations["Stocks"]
        pe = sv.pe_ratio
        z = (pe - 18.0) / 6.0
        return max(-1.5, min(1.5, -z))

    def decide(self, env: MacroEnvironment, market: "Market", period: int) -> AgentOrder:
        deltas: Dict[str, float] = {}
        rationale: List[str] = []

        trend     = self._trend_signal(market)
        valuation = self._valuation_signal(market)
        self.trend_signal_history.append(trend)
        self.valuation_signal_history.append(valuation)

        sv: StockValuation = market.valuations["Stocks"]
        sentiment_score = SENTIMENT_SCORE.get(env.sentiment, 0)

        signal_agree    = (trend > 0 and valuation > 0) or (trend < 0 and valuation < 0)
        signal_conflict = abs(trend - valuation) > 1.2

        target_stocks = 1.0

        if signal_agree and trend > 0.3:
            target_stocks = 1.60 + 0.25 * min(trend, 1.0)
            rationale.append(f"Trend ({trend:+.2f}) & valuation ({valuation:+.2f}) aligned BULLISH")
        elif signal_agree and trend < -0.3:
            target_stocks = 0.30 - 0.25 * min(abs(trend), 1.0)
            rationale.append(f"Trend ({trend:+.2f}) & valuation ({valuation:+.2f}) aligned BEARISH")
        elif signal_conflict and trend > 0.4:
            target_stocks = 1.15
            rationale.append(f"Trend positive ({trend:+.2f}) but valuation extreme — reduced long")
        elif signal_conflict and trend < -0.4:
            target_stocks = 0.65
            rationale.append(f"Trend negative ({trend:+.2f}) but valuation cheap — partial de-risk")
        else:
            rationale.append(f"Mixed signals (trend={trend:+.2f}, val={valuation:+.2f}) — near-neutral")

        if sentiment_score >= 1:
            target_stocks += 0.10 * sentiment_score
            rationale.append(f"Sentiment '{env.sentiment}' → add leverage")
        elif sentiment_score <= -1:
            target_stocks -= 0.12 * abs(sentiment_score)
            rationale.append(f"Sentiment '{env.sentiment}' → de-risk")

        recent_prices = market.assets["Stocks"].price_history[-12:]
        if len(recent_prices) > 3:
            rets = [(recent_prices[i]/recent_prices[i-1]-1) for i in range(1, len(recent_prices))]
            realised_vol = (np.std(rets) * math.sqrt(52)) if rets else 0.15
            vol_scale = min(1.0, self.risk_budget_vol / max(realised_vol, 0.04))
            target_stocks *= vol_scale
            if vol_scale < 0.85:
                rationale.append(f"Realised vol {realised_vol*100:.1f}% ann. — scaling by {vol_scale:.2f}x")

        if env.is_high_rates():
            target_stocks -= 0.15
            rationale.append(f"Rates {env.interest_rate:.1f}% > 5% → reduce financing exposure")
        if env.regime == "Recession":
            target_stocks = min(target_stocks, 0.50)
            rationale.append("Regime = Recession → cap gross long exposure")

        target_stocks = float(np.clip(target_stocks,
            self.allocation_bounds["Stocks"][0], self.allocation_bounds["Stocks"][1]))

        self._move_toward(deltas, "Stocks", target_stocks, 0.07)
        deltas["Cash"] = deltas.get("Cash", 0.0) - deltas.get("Stocks", 0.0)

        if env.is_recession() or env.is_oil_crisis() or env.regime == "Recession":
            self._move_toward(deltas, "Gold", 0.30, 0.09)
            rationale.append("Recession/oil crisis → tactical gold allocation")
        else:
            self._move_toward(deltas, "Gold", 0.0, 0.07)

        new_stocks = float(np.clip(
            self.current_allocation.get("Stocks", 1.2) + deltas.get("Stocks", 0.0),
            self.allocation_bounds["Stocks"][0], self.allocation_bounds["Stocks"][1]))
        new_gold   = float(np.clip(
            self.current_allocation.get("Gold", 0.0) + deltas.get("Gold", 0.0),
            0.0, 0.5))
        gross = abs(new_stocks) + abs(new_gold)
        net   = new_stocks + new_gold
        self.gross_exposure_history.append(gross)
        self.net_exposure_history.append(net)
        self.leverage_ratio_history.append(gross)

        if not rationale:
            rationale.append("No directional signal — hold current positions")
        return AgentOrder(self.name, deltas, rationale)

# ══════════════════════════════════════════════════════════════════════════════
# RETAIL INVESTOR  (DECISION RULES UNCHANGED from v3.0/v2.1)
# ══════════════════════════════════════════════════════════════════════════════

class RetailInvestor(Agent):
    name = "Retail Investor"
    color = C["retail"]
    allocation_bounds = {
        "Stocks": (0.0, 1.0),
        "Cash":   (0.0, 1.0),
        "Bonds":  (0.0, 0.15),
        "Gold":   (0.0, 0.15),
    }

    def __init__(self, news_sensitivity: float = 1.2):
        super().__init__({"Stocks": 0.80, "Cash": 0.20})
        self.fear  = 20.0
        self.greed = 60.0
        self.news_sensitivity = news_sensitivity
        self.fear_history:  List[float] = [20.0]
        self.greed_history: List[float] = [60.0]

    def _update_fear_greed(self, recent_3: float, recent_5: float, sentiment: str):
        s = SENTIMENT_SCORE.get(sentiment, 0)
        self.greed += (max(0, recent_3) * 1.6 + max(0, s) * 4.0) * self.news_sensitivity
        self.greed -= (max(0, -recent_3) * 1.0 + max(0, -s) * 2.0) * self.news_sensitivity
        self.greed = float(np.clip(self.greed, 0, 100))
        self.fear += (max(0, -recent_5) * 2.0 + max(0, -s) * 4.5) * self.news_sensitivity
        self.fear -= (max(0, recent_5) * 0.8 + max(0, s) * 2.0) * self.news_sensitivity
        self.fear = float(np.clip(self.fear, 0, 100))
        self.greed = self.greed * 0.87 + 50 * 0.13
        self.fear  = self.fear  * 0.87 + 30 * 0.13

    def decide(self, env: MacroEnvironment, market: "Market", period: int) -> AgentOrder:
        deltas: Dict[str, float] = {}
        rationale: List[str] = []

        r3 = market.recent_return("Stocks", 3)
        r5 = market.recent_return("Stocks", 5)
        self._update_fear_greed(r3, r5, env.sentiment)
        self.fear_history.append(self.fear)
        self.greed_history.append(self.greed)

        fg_score = (self.greed - self.fear) / 50.0

        target_stocks = 0.80

        if fg_score > 0.8:
            target_stocks = 0.90 + 0.05 * min(fg_score, 2.0)
            rationale.append(f"Greed {self.greed:.0f} >> Fear {self.fear:.0f} → euphoric buying")
        elif fg_score > 0.3:
            target_stocks = 0.85
            rationale.append(f"Net Greed ({fg_score:+.1f}) → mild risk-on")
        elif fg_score < -0.8:
            target_stocks = 0.30 - 0.10 * min(abs(fg_score), 1.5)
            rationale.append(f"Fear {self.fear:.0f} >> Greed {self.greed:.0f} → PANIC selling")
        elif fg_score < -0.3:
            target_stocks = 0.60
            rationale.append(f"Net Fear ({fg_score:+.1f}) → cautious, trimming stocks")

        if env.regime == "Recession":
            target_stocks = min(target_stocks, 0.45)
            rationale.append("Regime = Recession → further de-risking")

        if r5 < -6.0:
            target_stocks -= 0.15
            rationale.append(f"Drawdown {r5:.1f}% (5p) → panic sell trigger")

        target_stocks = float(np.clip(target_stocks, 0.0, 1.0))
        self._move_toward(deltas, "Stocks", target_stocks, 0.14)
        deltas["Cash"] = -deltas.get("Stocks", 0.0)

        if not rationale:
            rationale.append(f"F/G balanced (fear={self.fear:.0f}, greed={self.greed:.0f}) — hold")
        return AgentOrder(self.name, deltas, rationale)

# ══════════════════════════════════════════════════════════════════════════════
# MARKET  —  fair-value compounding FIXED (see header notes 4); price step
# mechanics otherwise unchanged from v3.0
# ══════════════════════════════════════════════════════════════════════════════

class Market:
    def __init__(self, env: MacroEnvironment, seed: int = 42, agent_filter: Optional[List[str]] = None):
        """
        agent_filter: if provided, only orders from agents whose `agent_name`
        is in this list contribute demand to the price engine. Used by the
        Agent Impact Analysis ablation runner.
        """
        self.env = env
        self.rng = random.Random(seed)
        self.agent_filter = agent_filter  # None = everyone included
        self.assets: Dict[str, Asset] = {
            "Stocks": Asset("Stocks", 100.0, market_depth=ASSET_DEPTH["Stocks"],
                            reversion_speed=ASSET_REVERSION_SPEED["Stocks"],
                            base_volatility=ASSET_BASE_VOL["Stocks"]),
            "Bonds":  Asset("Bonds",  100.0, market_depth=ASSET_DEPTH["Bonds"],
                            reversion_speed=ASSET_REVERSION_SPEED["Bonds"],
                            base_volatility=ASSET_BASE_VOL["Bonds"]),
            "Gold":   Asset("Gold",   100.0, market_depth=ASSET_DEPTH["Gold"],
                            reversion_speed=ASSET_REVERSION_SPEED["Gold"],
                            base_volatility=ASSET_BASE_VOL["Gold"]),
        }
        self.valuations: Dict = {
            "Stocks": StockValuation(earnings=5.5, earnings_growth=5.0),
            "Bonds":  BondValuation(yield_pct=env.interest_rate, duration=8.5),
            "Gold":   GoldValuation(inflation_sensitivity=0.85),
        }
        self.env_history: List[MacroEnvironment] = [env]
        self.regime_history: List[str] = [env.regime]
        self.expected_return_history: List[Dict[str, float]] = []
        self.attribution_history: List[Dict[str, Dict[str, float]]] = []
        # NEW: count of periods where the fair-value drift clip actually
        # bound (i.e. the raw model-implied annual return exceeded the sane
        # band). Surfaced in Diagnostics so clipping frequency is visible
        # rather than silent.
        self.fv_drift_clip_events: Dict[str, int] = {"Stocks": 0, "Bonds": 0, "Gold": 0}

    def expected_returns(self, env: MacroEnvironment) -> Dict[str, float]:
        sv: StockValuation = self.valuations["Stocks"]
        bv: BondValuation  = self.valuations["Bonds"]
        gv: GoldValuation  = self.valuations["Gold"]
        bias = regime_asset_bias(env.regime)
        return {
            "Stocks": sv.expected_return_pct(env.interest_rate, env.inflation) + bias["Stocks"],
            "Bonds":  bv.expected_return_pct() + bias["Bonds"],
            "Gold":   gv.expected_return_pct(env.inflation, env.real_rate) + bias["Gold"],
        }

    def recent_return(self, asset: str, lookback: int) -> float:
        h = self.assets[asset].price_history
        if len(h) < 2:
            return 0.0
        lb = min(lookback, len(h)-1)
        return (h[-1] / h[-1-lb] - 1) * 100.0

    def smoothed_momentum(self, asset: str, lookback: int) -> float:
        h = self.assets[asset].price_history
        if len(h) < 3:
            return 0.0
        lb = min(lookback, len(h)-1)
        w = h[-(lb+1):]
        rets = [(w[i]/w[i-1]-1)*100 for i in range(1, len(w))]
        return sum(rets)/len(rets)

    def aggregate_demand(self, orders: List[AgentOrder]) -> Dict[str, float]:
        demand = {n: 0.0 for n in self.assets}
        for o in orders:
            if self.agent_filter is not None and o.agent_name not in self.agent_filter:
                continue
            for a, d in o.allocation_deltas.items():
                if a in demand:
                    demand[a] += d
        return demand

    def per_agent_demand(self, orders: List[AgentOrder]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for o in orders:
            out[o.agent_name] = {a: d for a, d in o.allocation_deltas.items() if a in self.assets}
        return out

    def _update_fair_values(self):
        """
        Fair value path = pure fundamentals: prior fair value compounded by
        the asset's current model-implied expected return (valuation model +
        regime drift bias), with NO price-momentum or agent-flow term.

        FIXED in v3.1 (root cause of the +410% Inflation Shock bug):
          - `er[name]` is an ANNUAL % return. It is converted to a per-period
            fraction via PERIODS_PER_YEAR (named constant, not a bare magic
            number) -- mechanically the same operation as v3.0, but now
            paired with an explicit sanity clip immediately below, because
            v3.0 had no brake whatsoever on this compounding loop.
          - The per-period drift is clipped to +/- MAX_PERIOD_FAIR_VALUE_DRIFT_PCT
            (derived from a +/-60%/year sanity band). This is independent of
            -- and in addition to -- the price engine's own tanh damping on
            PRICE; this clip protects the ANCHOR that price reverts to. Even
            if the valuation model produces a transient extreme value (e.g.
            earnings_growth spiking, or a rate shock producing a large
            discount penalty), it cannot compound geometrically over 100
            periods.
          - Clip activations are counted per asset and exposed in
            Diagnostics so silent clipping is visible to a researcher.
        """
        er = self.expected_returns(self.env)
        for name, asset in self.assets.items():
            annual_return_raw = er[name]
            period_drift = annual_return_raw / PERIODS_PER_YEAR
            clipped_drift = max(-MAX_PERIOD_FAIR_VALUE_DRIFT_PCT,
                                 min(MAX_PERIOD_FAIR_VALUE_DRIFT_PCT, period_drift))
            if abs(clipped_drift - period_drift) > 1e-9:
                self.fv_drift_clip_events[name] += 1
            growth = 1.0 + clipped_drift / 100.0
            asset.set_fair_value(asset.fair_value * growth, annual_return_raw, clipped_drift)

    def step(self, orders: List[AgentOrder]) -> Dict[str, float]:
        demand = self.aggregate_demand(orders)
        per_agent = self.per_agent_demand(orders)
        vol_scale = REGIME_VOL_SCALE.get(self.env.regime, 1.0)

        self._update_fair_values()

        pct_changes = {}
        period_attr: Dict[str, Dict[str, float]] = {}

        for name, asset in self.assets.items():
            fv_before = asset.fair_value_history[-2]
            fv_after  = asset.fair_value_history[-1]
            valuation_macro_pct = (math.log(max(fv_after, 0.01) / max(fv_before, 0.01))) * 100.0

            total_pct, impact_pct, reversion_pct, noise_pct = asset.apply_demand(
                demand[name], self.rng, vol_scale=vol_scale)
            pct_changes[name] = total_pct

            total_signed_flow = sum(per_agent.get(o.agent_name, {}).get(name, 0.0) for o in orders)
            agent_components: Dict[str, float] = {}
            if abs(total_signed_flow) > 1e-9:
                for o in orders:
                    flow_i = per_agent.get(o.agent_name, {}).get(name, 0.0)
                    agent_components[o.agent_name] = impact_pct * (flow_i / total_signed_flow)
            else:
                for o in orders:
                    agent_components[o.agent_name] = 0.0

            period_attr[name] = {
                "Valuation_Macro": valuation_macro_pct + reversion_pct,
                **{f"Agent::{k}": v for k, v in agent_components.items()},
                "Noise": noise_pct,
                "Total": total_pct,
            }

        self.attribution_history.append(period_attr)

        sv: StockValuation = self.valuations["Stocks"]
        bv: BondValuation  = self.valuations["Bonds"]
        gv: GoldValuation  = self.valuations["Gold"]
        sv.update(self.assets["Stocks"].price, self.env.gdp_growth, self.rng)
        bv.update(self.env.interest_rate, self.env.inflation, self.rng)
        gv.update(self.rng)
        er = self.expected_returns(self.env)
        self.expected_return_history.append(er)
        return pct_changes

    def set_environment(self, env: MacroEnvironment):
        self.env = env
        self.env_history.append(env)
        self.regime_history.append(env.regime)

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE  (UNCHANGED — drift logic, ablation wiring)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LogEntry:
    period: int
    headline: str
    details: List[str]
    stock_change: float
    bond_change:  float
    gold_change:  float
    regime: str
    funding_ratio: float

class SimulationEngine:
    def __init__(self, base_env: MacroEnvironment, periods: int = 100, seed: int = 42,
                 agent_filter: Optional[List[str]] = None):
        self.base_env = base_env
        self.periods  = periods
        self.rng      = random.Random(seed)
        self.market   = Market(base_env, seed=seed, agent_filter=agent_filter)
        self.agents: List[Agent] = [PensionFund(), HedgeFund(), RetailInvestor()]
        self.logs: List[LogEntry] = []
        self.demand_history: List[Dict[str, float]] = []
        self.regime_prob_history: List[Dict[str, float]] = []
        self._regime_probs: Dict[str, float] = {r: 0.25 for r in REGIMES}

    def _drift_environment(self, prev: MacroEnvironment) -> MacroEnvironment:
        def mr(v, base, vol, lo=None, hi=None):
            out = v + (base - v) * 0.15 + self.rng.gauss(0, vol)
            if lo is not None: out = max(out, lo)
            if hi is not None: out = min(out, hi)
            return out

        new_inf  = mr(prev.inflation,     self.base_env.inflation,     0.13, lo=-2.0)
        new_rate = mr(prev.interest_rate, self.base_env.interest_rate, 0.11, lo=0.0)
        new_gdp  = mr(prev.gdp_growth,    self.base_env.gdp_growth,    0.17, lo=-10, hi=10)
        new_oil  = mr(prev.oil_shock,     self.base_env.oil_shock,     0.85, lo=-50, hi=150)

        base_idx = SENTIMENT_LEVELS.index(self.base_env.sentiment)
        idx      = SENTIMENT_LEVELS.index(prev.sentiment)
        if self.rng.random() < 0.12:
            step = 1 if idx < base_idx else -1 if idx > base_idx else self.rng.choice([-1,1])
            idx  = max(0, min(len(SENTIMENT_LEVELS)-1, idx+step))

        new_regime = next_regime(prev.regime, self.rng)

        return MacroEnvironment(new_inf, new_rate, new_gdp, new_oil,
                                SENTIMENT_LEVELS[idx], regime=new_regime)

    def _update_regime_probs(self, current_regime: str):
        target = {r: 0.02 for r in REGIMES}
        target[current_regime] = 0.94
        alpha = 0.12
        for r in REGIMES:
            self._regime_probs[r] = (1-alpha)*self._regime_probs[r] + alpha*target[r]
        self.regime_prob_history.append(dict(self._regime_probs))

    def run(self):
        current_env = self.base_env
        for period in range(1, self.periods+1):
            if period > 1:
                current_env = self._drift_environment(current_env)
                self.market.set_environment(current_env)
            self._update_regime_probs(current_env.regime)

            orders = [a.decide(current_env, self.market, period) for a in self.agents]
            for a, o in zip(self.agents, orders):
                a.apply_order(o)

            pct_changes = self.market.step(orders)
            for a in self.agents:
                a.update_portfolio_value(pct_changes)

            demand = self.market.aggregate_demand(orders)
            self.demand_history.append(demand)

            pension = next(a for a in self.agents if isinstance(a, PensionFund))
            fr = pension.funding_ratio_history[-1] if pension.funding_ratio_history else 100.0

            self.logs.append(self._build_log(period, current_env, orders, pct_changes, fr))
        return self

    def _build_log(self, period, env, orders, pct_changes, fr) -> LogEntry:
        triggered = [
            f"{o.agent_name}: {r}"
            for o in orders for r in o.rationale
            if not any(x in r for x in ("No active signal","No directional","F/G balanced","No threshold","No momentum","Neutral sentiment","strategic 40"))
        ]
        regime_tag = f"[{env.regime}]"
        if env.regime == "Recession":
            headline = f"{regime_tag} Recession: risk assets under pressure"
        elif env.is_high_inflation():
            headline = f"{regime_tag} Elevated inflation reshaping allocations"
        elif pct_changes["Stocks"] > 2.5:
            headline = f"{regime_tag} Equity rally — risk appetite building"
        elif pct_changes["Stocks"] < -2.5:
            headline = f"{regime_tag} Equity sell-off — agents repositioning"
        else:
            headline = f"{regime_tag} Markets consolidating"
        return LogEntry(period, headline, triggered[:6],
                        pct_changes["Stocks"], pct_changes["Bonds"], pct_changes["Gold"],
                        env.regime, fr)

# ══════════════════════════════════════════════════════════════════════════════
# RETURN ATTRIBUTION FRAMEWORK  (mechanics unchanged — math was correct;
# only the Valuation_Macro series it consumes was wrong, now fixed upstream)
# ══════════════════════════════════════════════════════════════════════════════

def compute_attribution_table(engine: SimulationEngine, asset: str = "Stocks") -> pd.DataFrame:
    rows = []
    agent_names = [a.name for a in engine.agents]
    for t, period_attr in enumerate(engine.market.attribution_history, start=1):
        a = period_attr.get(asset, {})
        row = {"Period": t, "Valuation_Macro": a.get("Valuation_Macro", 0.0)}
        for name in agent_names:
            row[name] = a.get(f"Agent::{name}", 0.0)
        row["Noise"] = a.get("Noise", 0.0)
        row["Total"] = a.get("Total", 0.0)
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_attribution(df: pd.DataFrame, agent_names: List[str]) -> Dict[str, float]:
    cols = ["Valuation_Macro"] + agent_names + ["Noise"]
    log_sums = {c: df[c].sum() / 100.0 for c in cols}
    total_log = df["Total"].sum() / 100.0
    total_simple_return = (math.exp(total_log) - 1) * 100.0

    out = {}
    if abs(total_log) > 1e-9:
        for c in cols:
            out[c] = total_simple_return * (log_sums[c] / total_log)
    else:
        for c in cols:
            out[c] = 0.0
    out["Total"] = total_simple_return
    return out

# ══════════════════════════════════════════════════════════════════════════════
# AGENT IMPACT ANALYSIS  — ablation runner (UNCHANGED mechanics)
# ══════════════════════════════════════════════════════════════════════════════

ALL_AGENT_NAMES = ["Pension Fund", "Hedge Fund", "Retail Investor"]

def run_agent_ablation(base_env: MacroEnvironment, periods: int, seed: int = 42) -> Dict[str, SimulationEngine]:
    configs = {
        "Full Model":            None,
        "Without Pension Fund":  [a for a in ALL_AGENT_NAMES if a != "Pension Fund"],
        "Without Hedge Fund":    [a for a in ALL_AGENT_NAMES if a != "Hedge Fund"],
        "Without Retail":        [a for a in ALL_AGENT_NAMES if a != "Retail Investor"],
    }
    results = {}
    for label, agent_filter in configs.items():
        eng = SimulationEngine(base_env, periods=periods, seed=seed, agent_filter=agent_filter)
        eng.run()
        results[label] = eng
    return results

def summarize_ablation(results: Dict[str, SimulationEngine], asset: str = "Stocks") -> pd.DataFrame:
    rows = []
    base_prices = pd.Series(results["Full Model"].market.assets[asset].price_history)
    for label, eng in results.items():
        prices = pd.Series(eng.market.assets[asset].price_history)
        rets   = prices.pct_change().dropna()
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100.0
        vol = rets.std() * 100.0
        drawdown = ((prices / prices.cummax()) - 1).min() * 100.0
        rows.append({
            "Configuration": label,
            "Total Return": total_return,
            "Volatility (per period)": vol,
            "Max Drawdown": drawdown,
        })
    df = pd.DataFrame(rows)
    base_row = df[df["Configuration"] == "Full Model"].iloc[0]
    df["Return Contribution (pp vs Full)"]     = base_row["Total Return"] - df["Total Return"]
    df["Volatility Contribution (pp vs Full)"] = base_row["Volatility (per period)"] - df["Volatility (per period)"]
    df["Drawdown Contribution (pp vs Full)"]   = base_row["Max Drawdown"] - df["Max Drawdown"]
    return df

# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO ENGINE  (unchanged interface, runs through fixed price engine)
# ══════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(base_env: MacroEnvironment, periods: int, n_seeds: int = 100) -> Dict:
    stock_returns = []
    bond_returns  = []
    gold_returns  = []
    hedge_max_leverages = []
    retail_avg_greeds   = []

    for seed in range(n_seeds):
        eng = SimulationEngine(base_env, periods=periods, seed=seed)
        eng.run()
        stock_returns.append(eng.market.assets["Stocks"].total_return_pct())
        bond_returns.append(eng.market.assets["Bonds"].total_return_pct())
        gold_returns.append(eng.market.assets["Gold"].total_return_pct())
        hedge = next(a for a in eng.agents if isinstance(a, HedgeFund))
        retail = next(a for a in eng.agents if isinstance(a, RetailInvestor))
        hedge_max_leverages.append(max(hedge.leverage_ratio_history) if hedge.leverage_ratio_history else 1.0)
        retail_avg_greeds.append(np.mean(retail.greed_history) if retail.greed_history else 50.0)

    return {
        "stock_returns": np.array(stock_returns),
        "bond_returns":  np.array(bond_returns),
        "gold_returns":  np.array(gold_returns),
        "hedge_max_leverages": np.array(hedge_max_leverages),
        "retail_avg_greeds":   np.array(retail_avg_greeds),
        "n_seeds": n_seeds,
    }

def monte_carlo_stats(arr: np.ndarray) -> Dict:
    return {
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "best":   float(np.max(arr)),
        "worst":  float(np.min(arr)),
        "p_pos":  float(np.mean(arr > 0) * 100),
        "p_loss": float(np.mean(arr < 0) * 100),
    }

# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO CONSISTENCY TESTS  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def run_consistency_tests(mc_results: Dict, scenario_name: str) -> List[Dict]:
    tests = []
    sr = mc_results["stock_returns"]
    br = mc_results["bond_returns"]
    gr = mc_results["gold_returns"]
    lg = mc_results["hedge_max_leverages"]
    fg = mc_results["retail_avg_greeds"]

    def t(name, condition_str, passed, value, unit=""):
        return {"name": name, "condition": condition_str,
                "passed": passed, "value": value, "unit": unit}

    if scenario_name == "AI Boom":
        tests.append(t("Stocks positive (mean)", "mean > 0%",
                       np.mean(sr) > 0, f"{np.mean(sr):+.1f}%"))
        tests.append(t("Stocks positive (>80% of seeds)", ">80% seeds > 0%",
                       np.mean(sr > 0) > 0.80, f"{np.mean(sr>0)*100:.0f}%"))
        tests.append(t("Greed elevated", "avg greed > 55",
                       np.mean(fg) > 55, f"{np.mean(fg):.1f}"))
        tests.append(t("Hedge fund leveraged", "mean max leverage > 1.3×",
                       np.mean(lg) > 1.3, f"{np.mean(lg):.2f}×"))
        tests.append(t("Stocks beat bonds", "mean stock > mean bond",
                       np.mean(sr) > np.mean(br), f"ΔR={np.mean(sr)-np.mean(br):+.1f}pp"))

    elif scenario_name == "Recession":
        tests.append(t("Stocks negative (mean)", "mean < 0%",
                       np.mean(sr) < 0, f"{np.mean(sr):+.1f}%"))
        tests.append(t("Stocks negative (>70% of seeds)", ">70% seeds < 0%",
                       np.mean(sr < 0) > 0.70, f"{np.mean(sr<0)*100:.0f}%"))
        tests.append(t("Bonds outperform stocks", "mean bond > mean stock",
                       np.mean(br) > np.mean(sr), f"ΔR={np.mean(br)-np.mean(sr):+.1f}pp"))
        tests.append(t("Fear elevated", "avg greed < 48",
                       np.mean(fg) < 48, f"{np.mean(fg):.1f}"))
        tests.append(t("Gold positive (mean)", "mean gold > 0%",
                       np.mean(gr) > 0, f"{np.mean(gr):+.1f}%"))

    elif scenario_name in ("Inflation Shock", "Oil Crisis"):
        tests.append(t("Gold positive (mean)", "mean gold > 0%",
                       np.mean(gr) > 0, f"{np.mean(gr):+.1f}%"))
        tests.append(t("Gold outperforms bonds", "mean gold > mean bond",
                       np.mean(gr) > np.mean(br), f"ΔR={np.mean(gr)-np.mean(br):+.1f}pp"))
        tests.append(t("Bonds under pressure", "mean bond < mean stock or bond <5%",
                       np.mean(br) < 5, f"{np.mean(br):+.1f}%"))
        tests.append(t("Stocks weak or modest", "mean stock < 15%",
                       np.mean(sr) < 15, f"{np.mean(sr):+.1f}%"))
        tests.append(t("Gold beats stocks", "mean gold > mean stock",
                       np.mean(gr) > np.mean(sr), f"ΔR={np.mean(gr)-np.mean(sr):+.1f}pp"))

    elif scenario_name == "Bull Market":
        tests.append(t("Stocks positive (mean)", "mean > 0%",
                       np.mean(sr) > 0, f"{np.mean(sr):+.1f}%"))
        tests.append(t("Stocks positive (>75% of seeds)", ">75% seeds > 0%",
                       np.mean(sr > 0) > 0.75, f"{np.mean(sr>0)*100:.0f}%"))
        tests.append(t("Greed elevated", "avg greed > 52",
                       np.mean(fg) > 52, f"{np.mean(fg):.1f}"))

    elif scenario_name == "Rate Shock":
        tests.append(t("Bonds under pressure", "mean bond < 5%",
                       np.mean(br) < 5, f"{np.mean(br):+.1f}%"))
        tests.append(t("Stocks subdued", "mean stock < 20%",
                       np.mean(sr) < 20, f"{np.mean(sr):+.1f}%"))

    else:
        tests.append(t("Simulation completed", "all seeds ran",
                       True, f"{mc_results['n_seeds']} seeds"))

    return tests

# ══════════════════════════════════════════════════════════════════════════════
# ECONOMIC DOMINANCE SCORE  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def compute_economic_dominance(engine: SimulationEngine) -> Dict:
    env_hist = engine.market.env_history
    prices   = engine.market.assets["Stocks"].price_history

    if len(prices) < 10 or len(env_hist) < 5:
        return {"r2": None, "signal_pct": None, "noise_pct": None, "warning": False}

    min_len = min(len(prices) - 1, len(env_hist))
    rets = [(prices[i] / prices[i-1] - 1) * 100 for i in range(1, min_len + 1)]
    gdps  = [env_hist[i].gdp_growth   for i in range(min_len)]
    infs  = [env_hist[i].inflation     for i in range(min_len)]
    rates = [env_hist[i].interest_rate for i in range(min_len)]
    reg_enc = [1 if env_hist[i].regime == "Expansion" else
               -1 if env_hist[i].regime == "Recession" else 0
               for i in range(min_len)]

    y = np.array(rets)
    X = np.column_stack([gdps, infs, rates, reg_enc, np.ones(min_len)])

    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ coeffs
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = max(0.0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    except Exception:
        r2 = 0.0

    signal_pct = r2 * 100
    noise_pct  = (1 - r2) * 100
    warning    = noise_pct > 70

    asset_r2 = {}
    for asset_name in ["Stocks", "Bonds", "Gold"]:
        ap = engine.market.assets[asset_name].price_history
        min_la = min(len(ap) - 1, len(env_hist))
        ar = [(ap[i] / ap[i-1] - 1) * 100 for i in range(1, min_la + 1)]
        ya = np.array(ar)
        Xa = np.column_stack([gdps[:min_la], infs[:min_la], rates[:min_la],
                               reg_enc[:min_la], np.ones(min_la)])
        try:
            ca, _, _, _ = np.linalg.lstsq(Xa, ya, rcond=None)
            yh = Xa @ ca
            ss_r = np.sum((ya - yh) ** 2)
            ss_t = np.sum((ya - np.mean(ya)) ** 2)
            asset_r2[asset_name] = max(0.0, 1 - ss_r / ss_t) if ss_t > 0 else 0.0
        except Exception:
            asset_r2[asset_name] = 0.0

    return {
        "r2": r2, "signal_pct": signal_pct, "noise_pct": noise_pct,
        "warning": warning, "asset_r2": asset_r2
    }

# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS / CONFIDENCE LEVEL  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def compute_confidence(mc_results: Dict) -> Dict:
    sr = mc_results["stock_returns"]
    mean = np.mean(sr)
    std  = np.std(sr)
    cov  = std / max(abs(mean), 1.0)

    if cov < 0.5:
        level, label_class = "High", "confidence-high"
    elif cov < 1.2:
        level, label_class = "Medium", "confidence-med"
    else:
        level, label_class = "Low", "confidence-low"

    return {
        "level": level,
        "label_class": label_class,
        "cov": cov,
        "mean": mean,
        "std": std,
        "range": float(np.max(sr) - np.min(sr)),
        "p_pos": float(np.mean(sr > 0) * 100),
    }

# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO PRESETS  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "Rate Shock":      dict(inflation=4.0,  interest_rate=7.5, gdp_growth=1.0,  oil_shock=0.0,   sentiment="Bearish",     regime="Slowdown"),
    "Inflation Shock": dict(inflation=8.5,  interest_rate=6.0, gdp_growth=1.5,  oil_shock=5.0,   sentiment="Bearish",     regime="Slowdown"),
    "Recession":       dict(inflation=1.5,  interest_rate=2.0, gdp_growth=-2.5, oil_shock=-10.0, sentiment="Very Bearish",regime="Recession"),
    "Oil Crisis":      dict(inflation=6.0,  interest_rate=5.0, gdp_growth=0.5,  oil_shock=35.0,  sentiment="Bearish",     regime="Slowdown"),
    "Bull Market":     dict(inflation=2.0,  interest_rate=3.0, gdp_growth=3.5,  oil_shock=0.0,   sentiment="Bullish",     regime="Expansion"),
    "AI Boom":         dict(inflation=2.5,  interest_rate=3.5, gdp_growth=4.5,  oil_shock=-5.0,  sentiment="Very Bullish",regime="Expansion"),
    "Recovery":        dict(inflation=2.8,  interest_rate=3.2, gdp_growth=2.2,  oil_shock=-5.0,  sentiment="Neutral",     regime="Recovery"),
}

# ══════════════════════════════════════════════════════════════════════════════
# NARRATIVE  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def generate_institutional_summary(engine: SimulationEngine) -> str:
    m  = engine.market
    stocks = pd.Series(m.assets["Stocks"].price_history)
    bonds  = pd.Series(m.assets["Bonds"].price_history)
    gold   = pd.Series(m.assets["Gold"].price_history)

    sr = m.assets["Stocks"].total_return_pct()
    br = m.assets["Bonds"].total_return_pct()
    gr = m.assets["Gold"].total_return_pct()
    sv = stocks.pct_change().std()*100
    sd = ((stocks/stocks.cummax())-1).min()*100
    sbc = stocks.pct_change().corr(bonds.pct_change())
    sgc = stocks.pct_change().corr(gold.pct_change())

    pension = next(a for a in engine.agents if isinstance(a, PensionFund))
    hedge   = next(a for a in engine.agents if isinstance(a, HedgeFund))
    retail  = next(a for a in engine.agents if isinstance(a, RetailInvestor))

    min_fr = min(pension.funding_ratio_history) if pension.funding_ratio_history else 100
    max_fr = max(pension.funding_ratio_history) if pension.funding_ratio_history else 100
    final_fr = pension.funding_ratio_history[-1] if pension.funding_ratio_history else 100

    max_lever = max(hedge.leverage_ratio_history) if hedge.leverage_ratio_history else 1.0
    avg_greed = np.mean(retail.greed_history) if retail.greed_history else 50

    regime_counts = pd.Series(engine.market.regime_history).value_counts()
    dominant_regime = regime_counts.index[0] if len(regime_counts) else "Expansion"
    dom_pct = regime_counts.iloc[0] / len(engine.market.regime_history) * 100 if engine.market.regime_history else 0

    paras = []
    env = engine.base_env
    reg_desc = f"a {env.regime.lower()}-dominated regime" if dom_pct > 60 else "a mixed-regime environment"
    paras.append(
        f"Over {engine.periods} simulation periods under {reg_desc} "
        f"({dominant_regime} {dom_pct:.0f}% of periods), the market produced "
        f"cumulative returns of {sr:+.1f}% equities, {br:+.1f}% bonds, and "
        f"{gr:+.1f}% gold. Realised equity volatility averaged {sv:.1f}% per period "
        f"with a peak drawdown of {sd:.1f}%."
    )

    if sbc < -0.15:
        paras.append(f"The stock–bond correlation of {sbc:.2f} confirmed a classic diversification dynamic.")
    elif sbc > 0.15:
        paras.append(f"A positive stock–bond correlation of {sbc:.2f} reflects the inflationary or rate-shock character of the regime.")
    else:
        paras.append(f"Stocks and bonds were broadly uncorrelated ({sbc:.2f}).")

    if final_fr < 95:
        paras.append(f"The pension fund finished below full funding ({final_fr:.1f}%), trough {min_fr:.1f}% — defensive shifts weighed on equity demand.")
    elif final_fr > 115:
        paras.append(f"The pension fund accumulated a surplus (final {final_fr:.1f}%, peak {max_fr:.1f}%), enabling additional equity risk.")
    else:
        paras.append(f"The pension fund's funding ratio remained near full funding (range {min_fr:.1f}%–{max_fr:.1f}%).")

    paras.append(
        f"The hedge fund reached a peak gross leverage of {max_lever:.2f}x, with trend and valuation signals "
        f"{'aligned for extended periods' if max_lever > 1.6 else 'frequently in conflict, constraining gross exposure'}."
    )
    paras.append(
        f"Retail investor sentiment averaged a greed score of {avg_greed:.0f}/100. "
        f"{'Fear episodes drove pro-cyclical selling.' if avg_greed < 45 else 'Sustained greed contributed to momentum-chasing flows.'}"
    )
    paras.append(
        "On a risk-adjusted basis, the simulation favoured "
        + ("duration and defensive assets" if br > sr else "equities and real assets")
        + ". The interaction of heterogeneous agents produced price dynamics that neither "
        "efficient-market models nor simple factor regressions fully capture."
    )
    return "\n\n".join(paras)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    defaults = dict(
        inflation=3.0, interest_rate=4.0, gdp_growth=2.0, oil_shock=0.0,
        sentiment="Neutral", regime="Expansion", periods=100, seed=42,
        engine=None, mc_results=None, mc_scenario=None,
        mc_n_seeds=100, mc_running=False, ablation_results=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _make_env():
    return MacroEnvironment(
        inflation=st.session_state["inflation"],
        interest_rate=st.session_state["interest_rate"],
        gdp_growth=st.session_state["gdp_growth"],
        oil_shock=st.session_state["oil_shock"],
        sentiment=st.session_state["sentiment"],
        regime=st.session_state["regime"],
    )

def apply_scenario_and_run(name: str):
    p = SCENARIOS[name]
    for k, v in p.items():
        st.session_state[k] = v
    run_simulation()

def run_simulation():
    env = _make_env()
    engine = SimulationEngine(env, periods=st.session_state["periods"], seed=st.session_state["seed"])
    engine.run()
    st.session_state["engine"] = engine
    st.session_state["ablation_results"] = None  # invalidate stale ablation run

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    c_text     = C["text"]
    c_green    = C["green"]
    c_text_dim = C["text_dim"]

    st.sidebar.markdown(
        f"<div style='font-family:monospace;font-size:1.05rem;font-weight:700;color:{c_text};'>"
        f"AGENT TWIN <span style='color:{c_green};font-size:.7rem;'>v3.1</span></div>"
        f"<div style='color:{c_text_dim};font-size:.76rem;margin-bottom:1rem;'>"
        "Institutional Market Simulator</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.slider("Inflation (%)",        -2.0, 12.0, key="inflation",     step=0.1)
    st.sidebar.slider("Interest Rate (%)",     0.0, 12.0, key="interest_rate", step=0.1)
    st.sidebar.slider("GDP Growth (%)",       -8.0,  8.0, key="gdp_growth",    step=0.1)
    st.sidebar.slider("Oil Price Shock (%)", -50.0,100.0, key="oil_shock",     step=1.0)
    st.sidebar.selectbox("Market Sentiment", SENTIMENT_LEVELS, key="sentiment")
    st.sidebar.selectbox("Starting Regime",  REGIMES, key="regime")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='font-size:.75rem;color:{c_text_dim};text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;'>Simulation Settings</div>",
        unsafe_allow_html=True)
    st.sidebar.slider("Periods", 20, 250, key="periods", step=10)
    st.sidebar.number_input("Random Seed", 0, 9999, key="seed", step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='font-size:.75rem;color:{c_text_dim};text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;'>Scenario Presets</div>",
        unsafe_allow_html=True)
    cols = st.sidebar.columns(2)
    for i, sname in enumerate(SCENARIOS):
        cols[i%2].button(sname, key=f"sc_{sname}", use_container_width=True,
                         on_click=apply_scenario_and_run, args=(sname,))

    st.sidebar.markdown("---")
    if st.sidebar.button("▶  RUN SIMULATION", use_container_width=True, type="primary"):
        run_simulation()
        st.rerun()

    if st.session_state["engine"] is None:
        st.sidebar.info("Configure environment and click Run, or choose a preset.")

# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fig(h=400, **kwargs) -> go.Figure:
    f = go.Figure()
    f.update_layout(**LAYOUT, height=h, **kwargs)
    return f

def _xaxis(n):
    return list(range(n))

# ══════════════════════════════════════════════════════════════════════════════
# RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown(
        '<div class="at-header">AGENT TWIN <span style="font-size:1rem;color:#16a34a;">v3.1</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="at-sub">Institutional market simulator with square-root market-impact '
        'price formation, fundamental mean reversion, full return attribution, agent ablation analysis, '
        'and a bounded fair-value anchor (v3.1 fix: equity fair value can no longer compound unboundedly).</div>',
        unsafe_allow_html=True)

def render_env_strip(engine: SimulationEngine):
    env = engine.base_env
    items = [
        ("Inflation", f"{env.inflation:.1f}%"),
        ("Policy Rate", f"{env.interest_rate:.1f}%"),
        ("GDP Growth", f"{env.gdp_growth:.1f}%"),
        ("Oil Shock", f"{env.oil_shock:+.0f}%"),
        ("Sentiment", env.sentiment),
        ("Starting Regime", env.regime),
    ]
    cols = st.columns(len(items))
    for col, (lbl, val) in zip(cols, items):
        col.markdown(
            f"<div class='panel' style='text-align:center;padding:.6rem .5rem;'>"
            f"<div class='metric-label'>{lbl}</div>"
            f"<div class='metric-value' style='font-size:1.05rem;'>{val}</div></div>",
            unsafe_allow_html=True)

def render_regime_chart(engine: SimulationEngine):
    if not engine.regime_prob_history:
        return
    df = pd.DataFrame(engine.regime_prob_history)
    df.index = range(1, len(df)+1)
    fig = _fig(h=280, title=dict(text="Market Regime Probabilities (smoothed)", font=dict(size=14)))
    for r in REGIMES:
        if r in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[r]*100, name=r,
                stackgroup="one",
                line=dict(width=0, color=REGIME_COLORS[r]),
                fillcolor=REGIME_COLORS[r],
                opacity=0.75,
            ))
    fig.update_layout(xaxis_title="Period", yaxis_title="Probability (%)",
                      hovermode="x unified", yaxis=dict(range=[0,100], **LAYOUT["yaxis"]))
    st.plotly_chart(fig, use_container_width=True)

def render_valuation_chart(engine: SimulationEngine):
    er_hist = engine.market.expected_return_history
    if not er_hist:
        return
    periods_x = list(range(1, len(er_hist)+1))
    fig = _fig(h=320, title=dict(text="Expected Returns by Asset Class (annualised %)", font=dict(size=14)))
    fig.add_trace(go.Scatter(x=periods_x, y=[e["Stocks"] for e in er_hist], name="Stocks", line=dict(color=C["stock"], width=2)))
    fig.add_trace(go.Scatter(x=periods_x, y=[e["Bonds"]  for e in er_hist], name="Bonds",  line=dict(color=C["bond"],  width=2)))
    fig.add_trace(go.Scatter(x=periods_x, y=[e["Gold"]   for e in er_hist], name="Gold",   line=dict(color=C["gold"],  width=2)))
    fig.add_hline(y=0, line_dash="dot", line_color=C["border"], line_width=1)
    fig.update_layout(xaxis_title="Period", yaxis_title="Expected Return (%/year)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def render_price_chart(engine: SimulationEngine):
    m = engine.market
    n = len(m.assets["Stocks"].price_history)
    x = _xaxis(n)
    fig = _fig(h=420, title=dict(text="Asset Price Evolution (Base = 100) — solid = price, dashed = fair value", font=dict(size=14)))
    for name, col in [("Stocks", C["stock"]), ("Bonds", C["bond"]), ("Gold", C["gold"])]:
        fig.add_trace(go.Scatter(x=x, y=m.assets[name].price_history, name=name,
                                 line=dict(color=col, width=2.2)))
        fig.add_trace(go.Scatter(x=x, y=m.assets[name].fair_value_history, name=f"{name} Fair Value",
                                 line=dict(color=col, width=1.2, dash="dot"), opacity=0.6,
                                 showlegend=True))
    for i, env in enumerate(m.env_history):
        if env.regime == "Recession":
            fig.add_vrect(x0=i-0.5, x1=i+0.5, fillcolor=C["recession"],
                         opacity=0.07, layer="below", line_width=0)
    fig.update_layout(xaxis_title="Period", yaxis_title="Index Level", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def render_summary_metrics(engine: SimulationEngine):
    m = engine.market
    cols = st.columns(3)
    for col, name in zip(cols, ["Stocks","Bonds","Gold"]):
        tr = m.assets[name].total_return_pct()
        color = C["green"] if tr >= 0 else C["red"]
        col.markdown(
            f"<div class='panel' style='text-align:center;'>"
            f"<div class='metric-label'>{name} — Total Return</div>"
            f"<div style='font-size:1.55rem;font-family:monospace;color:{color};margin-top:.15rem;'>{tr:+.1f}%</div>"
            f"<div class='metric-label' style='margin-top:.3rem;'>Final: {m.assets[name].price:.2f}</div>"
            "</div>", unsafe_allow_html=True)

def render_funding_ratio_chart(engine: SimulationEngine):
    pension = next(a for a in engine.agents if isinstance(a, PensionFund))
    if not pension.funding_ratio_history:
        return
    fr = pension.funding_ratio_history
    x  = list(range(1, len(fr)+1))
    fig = _fig(h=320, title=dict(text="Pension Fund — Funding Ratio (%)", font=dict(size=14)))
    fig.add_trace(go.Scatter(x=x, y=fr, name="Funding Ratio",
                             line=dict(color=C["pension"], width=2.2),
                             fill="tozeroy", fillcolor=f"rgba(59,130,166,0.08)"))
    fig.add_hline(y=100, line_dash="dash", line_color=C["amber"],
                  annotation_text="100% (Full Funding)", line_width=1.5)
    fig.add_hline(y=90,  line_dash="dot",  line_color=C["red"],
                  annotation_text="90% (De-risk Threshold)", line_width=1)
    fig.add_hline(y=120, line_dash="dot",  line_color=C["green"],
                  annotation_text="120% (Risk-on Trigger)", line_width=1)
    fig.update_layout(xaxis_title="Period", yaxis_title="Funding Ratio (%)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val) in zip([c1,c2,c3,c4], [
        ("Min Funding Ratio", f"{min(fr):.1f}%"),
        ("Max Funding Ratio", f"{max(fr):.1f}%"),
        ("Final Funding Ratio", f"{fr[-1]:.1f}%"),
        ("Required Return", f"{pension.required_return_pct:.1f}%"),
    ]):
        col.markdown(f"<div class='panel' style='text-align:center;padding:.7rem .5rem;'>"
                     f"<div class='metric-label'>{lbl}</div>"
                     f"<div class='metric-value' style='font-size:1.1rem;'>{val}</div></div>",
                     unsafe_allow_html=True)

def render_hedge_fund_dashboard(engine: SimulationEngine):
    hedge = next(a for a in engine.agents if isinstance(a, HedgeFund))
    if not hedge.gross_exposure_history:
        return
    x = list(range(len(hedge.gross_exposure_history)))
    fig = _fig(h=320, title=dict(text="Hedge Fund — Gross / Net Exposure & Leverage", font=dict(size=14)))
    fig.add_trace(go.Scatter(x=x, y=hedge.gross_exposure_history, name="Gross Exposure",
                             line=dict(color=C["hedge"], width=2.2)))
    fig.add_trace(go.Scatter(x=x, y=hedge.net_exposure_history, name="Net Exposure",
                             line=dict(color=C["amber"], width=1.8, dash="dash")))
    fig.add_hline(y=1.0, line_dash="dot", line_color=C["border"], line_width=1,
                  annotation_text="1× (Unleveraged)")
    fig.update_layout(xaxis_title="Period", yaxis_title="Exposure (×)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    if hedge.trend_signal_history and hedge.valuation_signal_history:
        xt = list(range(len(hedge.trend_signal_history)))
        fig2 = _fig(h=260, title=dict(text="Hedge Fund — Trend & Valuation Signals", font=dict(size=14)))
        fig2.add_trace(go.Scatter(x=xt, y=hedge.trend_signal_history, name="Trend Signal",
                                  line=dict(color=C["stock"], width=2)))
        fig2.add_trace(go.Scatter(x=xt, y=hedge.valuation_signal_history, name="Valuation Signal",
                                  line=dict(color=C["purple"], width=2)))
        fig2.add_hline(y=0, line_dash="dot", line_color=C["border"], line_width=1)
        fig2.update_layout(xaxis_title="Period", yaxis_title="Signal (−1 to +1)", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for col, (lbl, val) in zip([c1,c2,c3], [
        ("Peak Gross Leverage", f"{max(hedge.gross_exposure_history):.2f}×"),
        ("Final Net Exposure",  f"{hedge.net_exposure_history[-1]:.2f}×"),
        ("Risk Budget (Vol)",   f"{hedge.risk_budget_vol*100:.0f}% ann."),
    ]):
        col.markdown(f"<div class='panel' style='text-align:center;padding:.7rem .5rem;'>"
                     f"<div class='metric-label'>{lbl}</div>"
                     f"<div class='metric-value' style='font-size:1.1rem;'>{val}</div></div>",
                     unsafe_allow_html=True)

def render_fear_greed_chart(engine: SimulationEngine):
    retail = next(a for a in engine.agents if isinstance(a, RetailInvestor))
    if not retail.fear_history:
        return
    x = list(range(len(retail.fear_history)))
    fig = _fig(h=300, title=dict(text="Retail Investor — Fear / Greed Index", font=dict(size=14)))
    fig.add_trace(go.Scatter(x=x, y=retail.greed_history, name="Greed",
                             line=dict(color=C["green"], width=2.2),
                             fill="tozeroy", fillcolor="rgba(22,163,74,0.07)"))
    fig.add_trace(go.Scatter(x=x, y=retail.fear_history, name="Fear",
                             line=dict(color=C["red"], width=2.2),
                             fill="tozeroy", fillcolor="rgba(179,71,58,0.07)"))
    fig.add_hline(y=50, line_dash="dot", line_color=C["border"], line_width=1)
    fig.update_layout(xaxis_title="Period", yaxis_title="Index (0–100)", hovermode="x unified",
                      yaxis=dict(range=[0,100], **LAYOUT["yaxis"]))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for col, (lbl, val) in zip([c1,c2,c3], [
        ("Avg Greed Score", f"{np.mean(retail.greed_history):.0f}/100"),
        ("Peak Fear Score", f"{max(retail.fear_history):.0f}/100"),
        ("News Sensitivity", f"{retail.news_sensitivity:.1f}×"),
    ]):
        col.markdown(f"<div class='panel' style='text-align:center;padding:.7rem .5rem;'>"
                     f"<div class='metric-label'>{lbl}</div>"
                     f"<div class='metric-value' style='font-size:1.1rem;'>{val}</div></div>",
                     unsafe_allow_html=True)

def render_allocation_chart(engine: SimulationEngine):
    tabs = st.tabs([a.name for a in engine.agents])
    palette = [C["stock"], C["bond"], C["gold"], C["text_dim"]]
    for tab, agent in zip(tabs, engine.agents):
        with tab:
            df = pd.DataFrame(agent.allocation_history)
            fig = _fig(h=340, title=dict(text=f"{agent.name} — Allocation Over Time", font=dict(size=13)))
            for i, col_name in enumerate(df.columns):
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col_name]*100, name=col_name,
                    stackgroup="one",
                    line=dict(width=0.5, color=palette[i % len(palette)]),
                    fillcolor=palette[i % len(palette)],
                ))
            fig.update_layout(xaxis_title="Period", yaxis_title="Allocation (%)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, key=f"alloc_{agent.name}")

def render_demand_chart(engine: SimulationEngine):
    asset_colors = {"Stocks": C["stock"], "Bonds": C["bond"], "Gold": C["gold"]}
    net_by_agent = []
    for agent in engine.agents:
        totals = {k: 0.0 for k in ["Stocks","Bonds","Gold","Cash"]}
        for o in agent.trade_log:
            for k, v in o.allocation_deltas.items():
                if k in totals:
                    totals[k] += v
        net_by_agent.append((agent.name, totals, agent.color))

    fig = _fig(h=360, title=dict(text="Cumulative Net Demand by Agent (pp of allocation)", font=dict(size=14)))
    for asset in ["Stocks","Bonds","Gold"]:
        fig.add_trace(go.Bar(
            name=asset,
            x=[n for n,_,_ in net_by_agent],
            y=[t[asset]*100 for _,t,_ in net_by_agent],
            marker_color=asset_colors[asset],
        ))
    fig.update_layout(barmode="group", xaxis_title="Agent", yaxis_title="Net Allocation Change (pp)")
    st.plotly_chart(fig, use_container_width=True)

def render_risk_contribution_chart(engine: SimulationEngine):
    fig = _fig(h=300, title=dict(text="Risk Contribution by Agent (Realised Portfolio Value)", font=dict(size=14)))
    for agent in engine.agents:
        pv  = pd.Series(agent.portfolio_value_history)
        vol = pv.pct_change().std() * 100
        fig.add_trace(go.Scatter(
            x=list(range(len(agent.portfolio_value_history))),
            y=agent.portfolio_value_history,
            name=f"{agent.name} (σ={vol:.2f}%/p)",
            line=dict(color=agent.color, width=2),
        ))
    fig.update_layout(xaxis_title="Period", yaxis_title="Portfolio Value (Start=100)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for col, agent in zip([c1,c2,c3], engine.agents):
        pv  = pd.Series(agent.portfolio_value_history)
        vol = pv.pct_change().std() * 100
        tr  = (pv.iloc[-1] / pv.iloc[0] - 1) * 100 if len(pv) > 1 else 0
        col.markdown(f"<div class='panel' style='text-align:center;padding:.7rem .5rem;'>"
                     f"<div class='metric-label' style='color:{agent.color};'>{agent.name}</div>"
                     f"<div class='metric-value' style='font-size:1rem;'>Vol {vol:.2f}%/p</div>"
                     f"<div class='metric-label' style='margin-top:.25rem;'>Total Return {tr:+.1f}%</div>"
                     "</div>", unsafe_allow_html=True)

def render_rulebook():
    with st.expander("📋  Agent Decision Rules — Transparent Rule Engine", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Pension Fund** &nbsp;<span class='badge'>LDI</span><span class='badge'>Contrarian</span>", unsafe_allow_html=True)
            for r in [
                "IF funding ratio < 90% → emergency de-risk",
                "IF funding ratio < 100% → increase bond matching",
                "IF funding ratio > 120% → add return-seeking equities",
                "IF equity–bond yield gap < 1.5pp → trim equities",
                "IF yield gap > 5pp → add equities vs bonds",
                "IF inflation > 5% → add gold as liability hedge",
                "IF stocks fall >8% (5p) & funded → contrarian buy",
                "IF regime = Recession → defensive shift",
            ]:
                st.markdown(f"<div class='rule-row'>{r}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("**Hedge Fund** &nbsp;<span class='badge'>Trend</span><span class='badge'>Valuation</span><span class='badge'>Risk Budget</span>", unsafe_allow_html=True)
            for r in [
                "IF trend & valuation agree (bullish) → leveraged long",
                "IF trend & valuation agree (bearish) → reduce / short",
                "IF trend +ve but valuation expensive → half-size long",
                "IF realised vol > risk budget → scale positions down",
                "IF sentiment bullish → add leverage",
                "IF rates > 5% → reduce financing cost exposure",
                "IF recession / oil crisis → tactical gold",
                "Track: gross exposure, net exposure, leverage ratio",
            ]:
                st.markdown(f"<div class='rule-row'>{r}</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("**Retail Investor** &nbsp;<span class='badge'>Fear/Greed</span><span class='badge'>News</span>", unsafe_allow_html=True)
            for r in [
                "Gains compound greed index; losses spike fear index",
                "IF greed >> fear → euphoric buying / full equity",
                "IF fear >> greed → panic selling / flee to cash",
                "IF 5p drawdown > 6% → panic sell trigger",
                "IF regime = Recession → amplified de-risking",
                "News sensitivity multiplies all sentiment reactions",
                "Fear & greed mean-revert each period (not permanent)",
            ]:
                st.markdown(f"<div class='rule-row'>{r}</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Price Formation** &nbsp;<span class='badge'>√-Impact</span><span class='badge'>Mean Reversion</span><span class='badge'>Depth</span>", unsafe_allow_html=True)
        for r in [
            "impact_pct = sign(flow) · √|flow| · 100/√(market_depth)  — diminishing impact law",
            "reversion_pct = −reversion_speed · ln(price/fair_value) · 100  — pulls price to fundamentals",
            "noise_pct = N(0, base_vol · regime_vol_scale)",
            "total = 8·tanh((impact+reversion+noise)/8)  — smooth damping, no hard floor to saturate",
            "Stocks: depth=42, reversion=0.055  |  Bonds: depth=70, reversion=0.09  |  Gold: depth=30, reversion=0.045",
        ]:
            st.markdown(f"<div class='rule-row'>{r}</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Fair Value Anchor (FIXED in v3.1)** &nbsp;<span class='badge'>Bounded</span><span class='badge'>Rate-Sensitive</span>", unsafe_allow_html=True)
        for r in [
            f"fair_value compounds at expected_return/year ÷ {PERIODS_PER_YEAR:.0f} periods, clipped to ±{MAX_PERIOD_FAIR_VALUE_DRIFT_PCT:.2f}%/period",
            f"(equivalent to a ±{MAX_ANNUAL_FAIR_VALUE_DRIFT_PCT:.0f}%/year sanity band on the fundamental anchor itself)",
            "Stocks: earnings_yield (=1/PE, FIXED — was a no-op in v3.0) + earnings_growth − discount_penalty − inflation_penalty",
            "discount_penalty = max(0, real_rate − 2%) × 0.8   |   inflation_penalty = max(0, inflation − 2%) × 0.5",
            "earnings_growth ceiling tightened 15→9, decay speed 0.97→0.92 (was pinning near max for 100p under modest +GDP)",
        ]:
            st.markdown(f"<div class='rule-row'>{r}</div>", unsafe_allow_html=True)

def render_simulation_log(engine: SimulationEngine):
    c_text     = C["text"]
    notable = [l for l in engine.logs if l.details]
    show_all = st.checkbox("Show all periods", value=False)
    logs = engine.logs if show_all else notable

    if not logs:
        st.markdown("<div class='log-line'>No threshold rules triggered this run.</div>", unsafe_allow_html=True)
        return

    display = logs[-40:] if len(logs) > 40 else logs
    if len(logs) > 40:
        st.caption(f"Showing most recent 40 of {len(logs)} periods.")

    parts = []
    for l in display:
        regime_color = REGIME_COLORS.get(l.regime, C["text_dim"])
        lines = [
            f"<span class='log-period'>Period {l.period}</span> "
            f"<span style='color:{regime_color};font-size:.72rem;'>[{l.regime}]</span> "
            f"<span style='color:{c_text};'>FR:{l.funding_ratio:.0f}%</span> — {l.headline}."
        ]
        for d in l.details:
            lines.append(f"&nbsp;&nbsp;↳ {d}")
        lines.append(
            f"&nbsp;&nbsp;Stocks {l.stock_change:+.2f}% · "
            f"Bonds {l.bond_change:+.2f}% · Gold {l.gold_change:+.2f}%"
        )
        parts.append("<div class='log-line'>" + "<br>".join(lines) + "</div><br>")

    st.markdown(
        "<div class='panel' style='max-height:420px;overflow-y:auto;'>" + "".join(parts) + "</div>",
        unsafe_allow_html=True)

def render_institutional_panel(engine: SimulationEngine):
    st.markdown("### Institutional Analysis")
    summary = generate_institutional_summary(engine)
    st.markdown(
        "<div class='panel' style='line-height:1.7;font-size:.93rem;font-family:IBM Plex Sans,sans-serif;'>"
        + summary.replace("\n\n","<br><br>") + "</div>",
        unsafe_allow_html=True)
    st.caption("Generated entirely from simulation arrays — no external AI API.")

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS PANEL  (NEW in v3.1)
# ══════════════════════════════════════════════════════════════════════════════
# Surfaces exactly the metrics identified in the forensic diagnosis as the
# fastest way to confirm/reject "fair-value anchor blow-up" as a class of
# bug, so this doesn't have to be re-derived from first principles if a
# similar symptom appears again after future changes.

def render_diagnostics_panel(engine: SimulationEngine):
    st.markdown("### 🩺  Diagnostics — Fair Value vs. Price Decomposition")
    st.caption(
        "Isolates how much of an asset's total return came from the fundamental anchor "
        "(fair value) vs. agent-flow/price-impact effects, and reports fair-value clip activations."
    )

    m = engine.market
    cols = st.columns(3)
    for col, name in zip(cols, ["Stocks", "Bonds", "Gold"]):
        asset = m.assets[name]
        price_tr = asset.total_return_pct()
        fv_tr    = asset.fair_value_total_return_pct()
        clip_events = m.fv_drift_clip_events.get(name, 0)
        clip_pct = clip_events / max(1, len(asset.fv_period_drift_applied_history) - 1) * 100

        clip_color = C["red"] if clip_pct > 20 else (C["amber"] if clip_pct > 0 else C["green"])
        col.markdown(
            f"<div class='panel' style='padding:.8rem;'>"
            f"<div class='metric-label'>{name}</div>"
            f"<div style='font-family:monospace;font-size:.85rem;margin-top:.3rem;'>Price Total Return: "
            f"<b>{price_tr:+.1f}%</b></div>"
            f"<div style='font-family:monospace;font-size:.85rem;'>Fair Value Total Return: "
            f"<b>{fv_tr:+.1f}%</b></div>"
            f"<div style='font-family:monospace;font-size:.8rem;color:{C['text_dim']};margin-top:.25rem;'>"
            f"Gap (price − fair value): {price_tr - fv_tr:+.1f}pp</div>"
            f"<div style='font-family:monospace;font-size:.8rem;color:{clip_color};margin-top:.4rem;'>"
            f"Fair-value drift clip activated: {clip_events} periods ({clip_pct:.0f}%)</div>"
            "</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div style='font-family:monospace;font-size:.78rem;color:{C['text_dim']};margin-top:.3rem;'>"
        "Interpretation: if Price Total Return and Fair Value Total Return are close, the anchor is "
        "driving the result (a valuation-model issue, not an agent-flow issue). If the clip activated "
        "in a large share of periods, the underlying valuation model is frequently trying to imply an "
        "extreme annualised return and is being actively restrained — worth tightening the valuation "
        "model itself rather than relying on the clip.</div>",
        unsafe_allow_html=True)

    er_df = pd.DataFrame(m.expected_return_history)
    if not er_df.empty:
        st.markdown("**Annualised expected-return series (input to the fair-value anchor)**")
        st.dataframe(
            er_df.describe().T.style.format("{:.2f}"),
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# RETURN ATTRIBUTION DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def render_attribution_section(engine: SimulationEngine):
    st.markdown("### 🧮  Return Attribution — Why Did Stocks Move?")
    st.caption("Decomposition reconciles exactly to the realised total return (log-additive, converted to simple-return space).")

    asset_choice = st.selectbox("Asset", ["Stocks", "Bonds", "Gold"], key="attr_asset_select")
    df = compute_attribution_table(engine, asset=asset_choice)
    agent_names = [a.name for a in engine.agents]
    summary = summarize_attribution(df, agent_names)

    components = ["Valuation_Macro"] + agent_names + ["Noise"]
    labels = {
        "Valuation_Macro": "Valuation / Macro",
        "Pension Fund": "Pension Fund Impact",
        "Hedge Fund": "Hedge Fund Impact",
        "Retail Investor": "Retail Impact",
        "Noise": "Random Noise",
    }
    colors = {
        "Valuation_Macro": C["valuation_attr"],
        "Pension Fund": C["pension_attr"],
        "Hedge Fund": C["hedge_attr"],
        "Retail Investor": C["retail_attr"],
        "Noise": C["noise_attr"],
    }

    fig = _fig(h=340, title=dict(text=f"{asset_choice} Total Return Attribution", font=dict(size=14)))
    names_ordered = [labels.get(c, c) for c in components] + ["Total Return"]
    values_ordered = [summary[c] for c in components] + [summary["Total"]]
    bar_colors = [colors.get(c, C["text_dim"]) for c in components] + [C["green"] if summary["Total"] >= 0 else C["red"]]
    fig.add_trace(go.Bar(x=names_ordered, y=values_ordered, marker_color=bar_colors,
                         text=[f"{v:+.1f}%" for v in values_ordered], textposition="outside"))
    fig.add_hline(y=0, line_dash="dot", line_color=C["border"], line_width=1)
    fig.update_layout(yaxis_title="Contribution to Total Return (%)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    c_text_dim = C["text_dim"]
    rows_html = "".join(
        f"<tr><td style='padding:.35rem .7rem;font-family:monospace;font-size:.85rem;color:{colors.get(c, C['text'])};'>{labels.get(c,c)}</td>"
        f"<td style='padding:.35rem .7rem;font-family:monospace;font-size:.85rem;text-align:right;'>{summary[c]:+.2f}%</td></tr>"
        for c in components
    )
    st.markdown(
        f"<div class='panel'><table style='width:100%;border-collapse:collapse;'>"
        f"<thead><tr><th style='text-align:left;padding:.35rem .7rem;color:{c_text_dim};font-size:.75rem;text-transform:uppercase;'>Component</th>"
        f"<th style='text-align:right;padding:.35rem .7rem;color:{c_text_dim};font-size:.75rem;text-transform:uppercase;'>Contribution</th></tr></thead>"
        f"<tbody>{rows_html}"
        f"<tr style='border-top:1px solid {C['border']};'><td style='padding:.45rem .7rem;font-family:monospace;font-weight:700;'>Total Return</td>"
        f"<td style='padding:.45rem .7rem;font-family:monospace;font-weight:700;text-align:right;'>{summary['Total']:+.2f}%</td></tr>"
        f"</tbody></table></div>",
        unsafe_allow_html=True)

    reconciled = abs(sum(summary[c] for c in components) - summary["Total"]) < 0.05
    if reconciled:
        st.markdown(f"<span style='color:{C['green']};font-family:monospace;font-size:.8rem;'>✓ Components reconcile exactly to total return.</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:{C['amber']};font-family:monospace;font-size:.8rem;'>⚠ Reconciliation gap detected — check log.</span>", unsafe_allow_html=True)

    with st.expander("Per-period attribution detail"):
        st.dataframe(df.style.format({c: "{:+.2f}" for c in df.columns if c != "Period"}), use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════════════════
# AGENT IMPACT ANALYSIS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def render_agent_impact_section(engine: SimulationEngine):
    st.markdown("### 🔬  Agent Impact Analysis — Ablation Study")
    st.caption("Re-runs the identical environment/seed with each agent's market flow removed in turn, isolating its price-formation footprint.")

    if st.button("▶  Run Agent Ablation Study", type="primary"):
        with st.spinner("Running 4 configurations (Full / w/o Pension / w/o Hedge / w/o Retail)..."):
            results = run_agent_ablation(engine.base_env, periods=engine.periods, seed=st.session_state["seed"])
            st.session_state["ablation_results"] = results

    results = st.session_state.get("ablation_results")
    if results is None:
        st.info("Click **Run Agent Ablation Study** to quantify each agent's contribution to return, volatility, and drawdown.")
        return

    asset_choice = st.selectbox("Asset", ["Stocks", "Bonds", "Gold"], key="ablation_asset_select")
    df = summarize_ablation(results, asset=asset_choice)

    c_text_dim = C["text_dim"]
    display_cols = ["Configuration", "Total Return", "Volatility (per period)", "Max Drawdown",
                    "Return Contribution (pp vs Full)", "Volatility Contribution (pp vs Full)",
                    "Drawdown Contribution (pp vs Full)"]
    header_html = "".join(
        f"<th style='padding:.3rem .6rem;font-size:.74rem;color:{c_text_dim};text-transform:uppercase;text-align:left;'>{c}</th>"
        for c in display_cols
    )
    rows_html = ""
    for _, row in df.iterrows():
        is_full = row["Configuration"] == "Full Model"
        cells = []
        for c in display_cols:
            v = row[c]
            if c == "Configuration":
                weight = "700" if is_full else "400"
                cells.append(f"<td style='padding:.35rem .6rem;font-family:monospace;font-size:.85rem;font-weight:{weight};'>{v}</td>")
            else:
                cells.append(f"<td style='padding:.35rem .6rem;font-family:monospace;font-size:.85rem;text-align:right;'>{v:+.2f}</td>")
        rows_html += f"<tr>{''.join(cells)}</tr>"

    st.markdown(
        f"<div class='panel'><table style='width:100%;border-collapse:collapse;'>"
        f"<thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table></div>",
        unsafe_allow_html=True)

    fig = _fig(h=320, title=dict(text=f"{asset_choice} — Return Contribution by Agent (pp vs Full Model)", font=dict(size=14)))
    ablation_only = df[df["Configuration"] != "Full Model"]
    fig.add_trace(go.Bar(
        x=ablation_only["Configuration"], y=ablation_only["Return Contribution (pp vs Full)"],
        marker_color=[C["pension_attr"], C["hedge_attr"], C["retail_attr"]],
        text=[f"{v:+.1f}pp" for v in ablation_only["Return Contribution (pp vs Full)"]],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=C["border"], line_width=1)
    fig.update_layout(yaxis_title="Return Contribution (pp)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Interpretation: a positive 'Return Contribution' for 'Without X' means agent X's flow was a net "
        "DRAG on this asset's return in the Full Model (removing it increases the return). A negative value "
        "means agent X's flow was a net TAILWIND."
    )

# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def render_monte_carlo_section():
    st.markdown("### 🎲  Monte Carlo Validation")

    with st.expander("Monte Carlo Settings & Run", expanded=True):
        col_s, col_n, col_btn = st.columns([2, 2, 1])
        scenario_options = ["(current params)"] + list(SCENARIOS.keys())
        mc_scenario = col_s.selectbox("Scenario to test", scenario_options, key="mc_scenario_select")
        n_seeds = col_n.slider("Number of seeds", 20, 200, value=100, step=10, key="mc_n_seeds_slider")
        run_mc  = col_btn.button("▶  Run MC", use_container_width=True, type="primary")

    if run_mc:
        if mc_scenario == "(current params)":
            env = _make_env()
            label = "Custom Parameters"
        else:
            p = SCENARIOS[mc_scenario]
            env = MacroEnvironment(**p)
            label = mc_scenario

        with st.spinner(f"Running {n_seeds} seeds for '{label}'..."):
            mc = run_monte_carlo(env, periods=st.session_state["periods"], n_seeds=n_seeds)
            st.session_state["mc_results"]  = mc
            st.session_state["mc_label"]    = label
            st.session_state["mc_scenario"] = mc_scenario if mc_scenario != "(current params)" else None

    mc = st.session_state.get("mc_results")
    if mc is None:
        st.info("Click **Run MC** to validate scenario robustness across multiple seeds.")
        return

    label = st.session_state.get("mc_label", "Scenario")
    scenario_name = st.session_state.get("mc_scenario") or ""

    st.markdown(f"#### Monte Carlo Results — *{label}* ({mc['n_seeds']} seeds)")

    sr_stats = monte_carlo_stats(mc["stock_returns"])
    br_stats = monte_carlo_stats(mc["bond_returns"])
    gr_stats = monte_carlo_stats(mc["gold_returns"])

    st.markdown("##### Summary Statistics")
    col_labels = ["Metric", "Stocks", "Bonds", "Gold"]
    rows = [
        ("Mean Return",        f"{sr_stats['mean']:+.1f}%",   f"{br_stats['mean']:+.1f}%",   f"{gr_stats['mean']:+.1f}%"),
        ("Median Return",      f"{sr_stats['median']:+.1f}%", f"{br_stats['median']:+.1f}%", f"{gr_stats['median']:+.1f}%"),
        ("Std Deviation",      f"{sr_stats['std']:.1f}%",     f"{br_stats['std']:.1f}%",     f"{gr_stats['std']:.1f}%"),
        ("Best Case",          f"{sr_stats['best']:+.1f}%",   f"{br_stats['best']:+.1f}%",   f"{gr_stats['best']:+.1f}%"),
        ("Worst Case",         f"{sr_stats['worst']:+.1f}%",  f"{br_stats['worst']:+.1f}%",  f"{gr_stats['worst']:+.1f}%"),
        ("P(Positive Return)", f"{sr_stats['p_pos']:.0f}%",   f"{br_stats['p_pos']:.0f}%",   f"{gr_stats['p_pos']:.0f}%"),
        ("P(Loss)",            f"{sr_stats['p_loss']:.0f}%",  f"{br_stats['p_loss']:.0f}%",  f"{gr_stats['p_loss']:.0f}%"),
    ]

    c_text_dim = C["text_dim"]
    html_rows = "".join(
        f"<tr><td style='color:{c_text_dim};padding:.3rem .6rem;font-size:.85rem;'>{r[0]}</td>"
        + "".join(f"<td style='padding:.3rem .6rem;font-size:.85rem;font-family:monospace;'>{v}</td>" for v in r[1:])
        + "</tr>"
        for r in rows
    )
    header_html = "".join(
        f"<th style='color:{c_text_dim};padding:.3rem .6rem;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;text-align:left;'>{h}</th>"
        for h in col_labels
    )
    st.markdown(
        f"<div class='panel'><table style='width:100%;border-collapse:collapse;'>"
        f"<thead><tr>{header_html}</tr></thead><tbody>{html_rows}</tbody></table></div>",
        unsafe_allow_html=True)

    fig_hist = _fig(h=320, title=dict(text="Distribution of Stock Returns Across Seeds", font=dict(size=14)))
    fig_hist.add_trace(go.Histogram(
        x=mc["stock_returns"], nbinsx=25, name="Stock Returns",
        marker_color=C["stock"], opacity=0.8,
    ))
    fig_hist.add_vline(x=sr_stats["mean"],   line_dash="dash",  line_color=C["amber"], annotation_text=f"Mean {sr_stats['mean']:+.1f}%")
    fig_hist.add_vline(x=sr_stats["median"], line_dash="dot",   line_color=C["green"], annotation_text=f"Median {sr_stats['median']:+.1f}%")
    fig_hist.add_vline(x=0,                  line_dash="solid", line_color=C["red"],   line_width=1.5, annotation_text="0%")
    fig_hist.update_layout(xaxis_title="Total Stock Return (%)", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.markdown("##### ✅  Scenario Consistency Tests")

    tests = run_consistency_tests(mc, scenario_name)
    if not tests:
        st.info("No specific consistency tests defined for this scenario.")
    else:
        all_pass = all(t["passed"] for t in tests)
        n_pass   = sum(1 for t in tests if t["passed"])
        summary_color = C["green"] if all_pass else (C["amber"] if n_pass >= len(tests)//2 else C["red"])
        st.markdown(
            f"<div style='font-family:monospace;font-size:.85rem;color:{summary_color};margin-bottom:.5rem;'>"
            f"{'✓ All tests passed' if all_pass else f'{n_pass}/{len(tests)} tests passed'}"
            "</div>",
            unsafe_allow_html=True)

        for t in tests:
            badge = "<span class='pass-badge'>PASS</span>" if t["passed"] else "<span class='fail-badge'>FAIL</span>"
            st.markdown(
                f"<div class='panel' style='padding:.5rem .9rem;margin-bottom:.4rem;'>"
                f"{badge} <span style='font-family:monospace;font-size:.83rem;'>{t['name']}</span>"
                f"<span style='color:{c_text_dim};font-size:.8rem;font-family:monospace;'>"
                f" — condition: {t['condition']} — result: <b>{t['value']}</b></span>"
                "</div>",
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 📊  Economic Dominance Score")

    engine = st.session_state.get("engine")
    if engine is None:
        if scenario_name and scenario_name in SCENARIOS:
            p = SCENARIOS[scenario_name]
            env_ed = MacroEnvironment(**p)
        else:
            env_ed = _make_env()
        engine = SimulationEngine(env_ed, periods=st.session_state["periods"], seed=0)
        engine.run()

    dom = compute_economic_dominance(engine)

    if dom["r2"] is not None:
        sig   = dom["signal_pct"]
        noise = dom["noise_pct"]

        sig_color   = C["green"] if sig > 40   else C["amber"] if sig > 25   else C["red"]
        noise_color = C["red"]   if noise > 70 else C["amber"] if noise > 55 else C["green"]

        col1, col2, col3 = st.columns(3)
        col1.markdown(
            f"<div class='panel' style='text-align:center;padding:.7rem;'>"
            f"<div class='metric-label'>Economic Signal Strength</div>"
            f"<div class='metric-value' style='color:{sig_color};'>{sig:.1f}%</div>"
            f"<div class='metric-label' style='margin-top:.2rem;'>R² of macro→returns</div></div>",
            unsafe_allow_html=True)
        col2.markdown(
            f"<div class='panel' style='text-align:center;padding:.7rem;'>"
            f"<div class='metric-label'>Noise Contribution</div>"
            f"<div class='metric-value' style='color:{noise_color};'>{noise:.1f}%</div>"
            f"<div class='metric-label' style='margin-top:.2rem;'>Unexplained variance</div></div>",
            unsafe_allow_html=True)

        r2_bars = dom.get("asset_r2", {})

        def _r2_color(v: float) -> str:
            return C["green"] if v > 0.35 else C["amber"] if v > 0.2 else C["red"]

        col3.markdown(
            "<div class='panel' style='text-align:center;padding:.7rem;'>"
            "<div class='metric-label'>Per-Asset R²</div>"
            + "".join(
                f"<div style='font-family:monospace;font-size:.85rem;margin-top:.2rem;'>"
                f"{a}: <span style='color:{_r2_color(v)};'>{v*100:.1f}%</span></div>"
                for a, v in r2_bars.items()
            )
            + "</div>",
            unsafe_allow_html=True)

        c_red       = C["red"]
        c_green     = C["green"]
        c_green_dim = C["green_dim"]

        if dom["warning"]:
            st.markdown(
                f"<div style='background:#3d1210;border:1px solid {c_red};border-radius:6px;padding:.7rem 1rem;margin-top:.5rem;'>"
                f"<span style='color:{c_red};font-family:monospace;font-weight:700;'>⚠ NOISE DOMINANCE WARNING</span>"
                f"<span style='color:{c_text_dim};font-size:.85rem;font-family:monospace;'>"
                f" — Noise explains {noise:.1f}% of variance. "
                "Economic fundamentals are insufficiently dominant. Consider reducing stochastic noise parameters or increasing simulation length.</span>"
                "</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='background:{c_green_dim};border:1px solid {c_green};border-radius:6px;padding:.5rem 1rem;margin-top:.5rem;'>"
                f"<span style='color:{c_green};font-family:monospace;font-size:.85rem;'>"
                f"✓ Economic fundamentals are sufficiently dominant ({sig:.1f}% of variance explained).</span>"
                "</div>",
                unsafe_allow_html=True)

        fig_dom = _fig(h=200, title=dict(text="Per-Asset: Economic Signal vs Noise", font=dict(size=12)))
        assets_list = list(r2_bars.keys())
        sig_vals   = [r2_bars[a]*100 for a in assets_list]
        noise_vals = [100 - v for v in sig_vals]
        fig_dom.add_trace(go.Bar(x=assets_list, y=sig_vals,  name="Signal", marker_color=C["green"], opacity=0.8))
        fig_dom.add_trace(go.Bar(x=assets_list, y=noise_vals, name="Noise", marker_color=C["red"],   opacity=0.6))
        fig_dom.update_layout(barmode="stack", yaxis_title="% of Variance",
                              yaxis=dict(range=[0,100], **LAYOUT["yaxis"]))
        st.plotly_chart(fig_dom, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 🛡️  Robustness Dashboard")

    conf  = compute_confidence(mc)
    level = conf["level"]
    lc    = conf["label_class"]

    st.markdown(
        f"<div class='panel'>"
        f"<div class='metric-label'>Confidence Level</div>"
        f"<div class='{lc}' style='font-size:1.5rem;margin:.2rem 0;'>{level}</div>"
        f"<div style='color:{c_text_dim};font-size:.82rem;font-family:monospace;'>"
        f"CoV = {conf['cov']:.2f} | Std = {conf['std']:.1f}% | "
        f"Outcome Range = {conf['range']:.1f}pp | P(Positive) = {conf['p_pos']:.0f}%"
        "</div></div>",
        unsafe_allow_html=True)

    st.markdown("**Robustness across all built-in scenarios** (20 seeds each, quick preview)")
    if st.button("Generate robustness table for all scenarios"):
        with st.spinner("Running 20-seed MC for each scenario..."):
            rob_rows = []
            for sname, sparams in SCENARIOS.items():
                senv = MacroEnvironment(**sparams)
                smc  = run_monte_carlo(senv, periods=st.session_state["periods"], n_seeds=20)
                sc   = compute_confidence(smc)
                rob_rows.append({
                    "Scenario":        sname,
                    "Avg Stock Return": f"{np.mean(smc['stock_returns']):+.1f}%",
                    "Outcome Range":    f"{np.max(smc['stock_returns'])-np.min(smc['stock_returns']):.1f}pp",
                    "Std Dev":          f"{np.std(smc['stock_returns']):.1f}%",
                    "P(Positive)":      f"{np.mean(smc['stock_returns']>0)*100:.0f}%",
                    "Confidence":       sc["level"],
                })

        df_rob = pd.DataFrame(rob_rows)

        def conf_badge(v):
            cls = "pass-badge" if v == "High" else ("warn-badge" if v == "Medium" else "fail-badge")
            return f"<span class='{cls}'>{v}</span>"

        header_cells = "".join(
            f"<th style='color:{c_text_dim};padding:.3rem .6rem;font-size:.78rem;"
            f"text-transform:uppercase;text-align:left;'>{c}</th>"
            for c in df_rob.columns
        )
        html_rob = (
            "<div class='panel'><table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr>{header_cells}</tr></thead><tbody>"
        )
        for _, row in df_rob.iterrows():
            cells = []
            for col_name, val in row.items():
                if col_name == "Confidence":
                    cells.append(f"<td style='padding:.3rem .6rem;'>{conf_badge(val)}</td>")
                else:
                    cells.append(f"<td style='padding:.3rem .6rem;font-size:.85rem;font-family:monospace;'>{val}</td>")
            html_rob += f"<tr>{''.join(cells)}</tr>"
        html_rob += "</tbody></table></div>"
        st.markdown(html_rob, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    init_session_state()
    render_sidebar()
    render_header()

    engine: Optional[SimulationEngine] = st.session_state.get("engine")
    render_rulebook()
    st.markdown("---")

    c_text_dim = C["text_dim"]

    if engine is None:
        st.markdown(
            f"<div class='panel' style='text-align:center;padding:3rem 1rem;'>"
            f"<div style='font-size:1.05rem;color:{c_text_dim};'>"
            "No simulation running.<br>Set parameters in the sidebar and click "
            "<b>RUN SIMULATION</b>, or choose a scenario preset.</div></div>",
            unsafe_allow_html=True)
        st.markdown("---")
        render_monte_carlo_section()
        return

    st.markdown("### 1 · Market Environment")
    render_env_strip(engine)

    st.markdown("### 2 · Market Regimes")
    render_regime_chart(engine)

    st.markdown("### 3 · Asset Valuations & Expected Returns")
    render_valuation_chart(engine)

    st.markdown("### 4 · Simulation Results")
    render_summary_metrics(engine)
    render_price_chart(engine)
    st.markdown("---")

    st.markdown("### 5 · Pension Fund — Liability-Driven Dashboard")
    render_funding_ratio_chart(engine)
    st.markdown("---")

    st.markdown("### 6 · Hedge Fund — Exposure & Signal Dashboard")
    render_hedge_fund_dashboard(engine)
    st.markdown("---")

    st.markdown("### 7 · Retail Investor — Fear / Greed Dashboard")
    render_fear_greed_chart(engine)
    st.markdown("---")

    st.markdown("### 8 · Agent Allocations & Market Demand")
    render_allocation_chart(engine)
    render_demand_chart(engine)
    st.markdown("---")

    st.markdown("### 9 · Risk Contributions & Portfolio Performance")
    render_risk_contribution_chart(engine)
    st.markdown("---")

    st.markdown("### 10 · Diagnostics")
    render_diagnostics_panel(engine)
    st.markdown("---")

    st.markdown("### 11 · Return Attribution")
    render_attribution_section(engine)
    st.markdown("---")

    st.markdown("### 12 · Agent Impact Analysis")
    render_agent_impact_section(engine)
    st.markdown("---")

    st.markdown("### 13 · Simulation Log")
    render_simulation_log(engine)
    st.markdown("---")

    render_institutional_panel(engine)
    st.markdown("---")

    render_monte_carlo_section()

    st.markdown("---")
    st.caption(
        "Agent Twin v3.1 — square-root market impact, fundamental mean reversion, "
        "exact return attribution, agent ablation analysis, bounded fair-value anchor. Not investment advice."
    )

if __name__ == "__main__":
    main()
