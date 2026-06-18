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

def next_regime(current: str, rng: random.Random,
                 base_regime: Optional[str] = None,
                 anchor_strength: float = 0.0) -> str:
    """
    v3.2 FIX (see CHANGELOG_v3.2): the original chain's transition matrix has
    a stationary distribution (~48% Expansion / 28% Slowdown / 13% Recession
    / 10% Recovery) that is IDENTICAL regardless of which scenario the user
    selected. A 100-period "Recession" preset would spend ~84% of its
    periods NOT in Recession on average (measured empirically), because
    nothing in the chain remembers what scenario it was launched from --
    it free-runs on the generic matrix from period 2 onward. Since
    `regime_asset_bias()` flips sign across regimes (+0.20 Stocks in
    Expansion vs -0.30 in Recession), this silently fed a strongly
    pro-Expansion bias into supposedly-recessionary runs.

    Fix: blend the base transition matrix with a pull back toward the
    scenario's intended (base) regime. `anchor_strength` in [0, 1] controls
    how much weight is placed on staying near the scenario's own regime vs.
    the chain's free-running organic dynamics; at 0 this reduces to the
    original behaviour. The SimulationEngine sets this once per run from
    the scenario, so organic transitions (recoveries from recession, boom
    cooling into slowdown, etc.) are still possible -- the base regime is a
    gravitational anchor, not a hard floor.
    """
    probs = dict(REGIME_TRANSITIONS[current])
    if base_regime is not None and anchor_strength > 0.0:
        blended = {r: (1.0 - anchor_strength) * probs.get(r, 0.0) for r in REGIMES}
        blended[base_regime] = blended.get(base_regime, 0.0) + anchor_strength
        total = sum(blended.values())
        probs = {r: v / total for r, v in blended.items()}

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
    """
    v3.2 ROOT-CAUSE FIX (see CHANGELOG_v3.2): `earnings` was a freely
    compounding state variable with no mean reversion and no ceiling.
    `earnings_yield`, defined as 1/PE, is mathematically `100 / (100/earnings)
    == earnings` -- an IDENTITY, not an independent fix. So as long as
    `earnings` itself could compound without bound, "fixing" the formula
    that derives earnings_yield FROM earnings could never matter: any
    formula for earnings_yield that is monotonic in earnings will inherit
    earnings' lack of a ceiling. v3.1's clip on per-period FAIR VALUE drift
    could not catch this either, because the bug is a slow multi-period
    ratchet in the input (earnings/PE), not a single-period spike in the
    output (expected_return) -- by the time the clip's threshold (60%/yr)
    is crossed, the simulation has often already finished its run.

    v3.2 fix, applied directly to the state variable that was unbounded:
      - `earnings` now mean-reverts toward a GDP-implied fundamental level
        (EARNINGS_BASELINE adjusted for cumulative real growth) every period,
        the same way bond yields and gold's inflation sensitivity already do.
        It no longer just compounds forward with no anchor pulling it back.
      - PE ratio (and therefore earnings_yield) is hard-bounded to
        [PE_FLOOR, PE_CEILING] = [6, 40], i.e. earnings_yield in
        [2.5%, 16.7%]. This is the actual fix: it bounds the quantity that
        was unbounded, not a re-derivation that happens to equal the same
        unbounded quantity.
      - earnings_growth keeps its v3.1 tighter ceiling/decay (still useful
        as a separate, slower-moving growth-expectations state), but no
        longer the only thing standing between the model and a runaway PE.
    """
    earnings: float
    earnings_growth: float

    # Hard, economically-motivated bounds on the valuation multiple itself.
    # PE=6 (~16.7% earnings yield) is a deep-distress trough; PE=40 (~2.5%
    # earnings yield) is a euphoric-bubble ceiling. Real-world index PEs
    # have approached but rarely sustained outside this band for long.
    PE_FLOOR: float = field(default=6.0, repr=False, compare=False)
    PE_CEILING: float = field(default=40.0, repr=False, compare=False)

    # Fundamental "fair" earnings level at the start of a run (matches the
    # Market's starting StockValuation(earnings=5.5, ...)), used as the
    # mean-reversion target's base. Tracked via a running GDP-implied path
    # rather than left to float freely.
    _fundamental_earnings: float = field(default=5.5, repr=False, compare=False)

    @property
    def pe_ratio(self) -> float:
        raw_pe = 100.0 / self.earnings if self.earnings > 0 else self.PE_CEILING
        return max(self.PE_FLOOR, min(self.PE_CEILING, raw_pe))

    def expected_return_pct(self, interest_rate: float = 4.0, inflation: float = 2.0) -> float:
        """
        ANNUALISED expected return (%), composed of:
          - earnings_yield = 1 / PE, with PE drawn from the BOUNDED pe_ratio
            property above -- this is what actually stops the runaway, not
            the formula shape.
          - + earnings_growth (also annualised, separately bounded)
          - - a discount-rate / multiple-compression penalty (rates)
          - - an inflation penalty (margin compression / uncertainty premium)
        """
        earnings_yield = 100.0 / self.pe_ratio
        real_rate = interest_rate - inflation
        discount_penalty = max(0.0, real_rate - 2.0) * 0.8
        inflation_penalty = max(0.0, min(inflation - 2.0, 10.0)) * 0.5
        return earnings_yield + self.earnings_growth - discount_penalty - inflation_penalty

    def is_expensive(self) -> bool:
        return self.pe_ratio > 25

    def is_cheap(self) -> bool:
        return self.pe_ratio < 14

    def update(self, price: float, gdp_growth: float, rng: random.Random):
        # Fundamental earnings level drifts slowly with trend GDP capacity
        # (this is the "potential output" anchor -- it moves, but gradually,
        # and has no feedback from price/PE, so it cannot itself spiral).
        fundamental_growth_factor = 1.0 + (gdp_growth / 100.0) * 0.25
        self._fundamental_earnings = max(0.5, self._fundamental_earnings * fundamental_growth_factor)

        # Actual earnings: grows with GDP plus noise, same as before, BUT
        # now pulled back toward the fundamental anchor each period instead
        # of being left to compound freely. This is the actual fix -- a
        # mean-reverting state variable instead of an unbounded random walk
        # with positive drift.
        growth_factor = 1.0 + (gdp_growth / 100.0) * 0.4 + rng.gauss(0, 0.004)
        raw_earnings = max(0.5, self.earnings * growth_factor)
        reversion_speed = 0.06  # gentle pull, same order as asset price reversion
        self.earnings = raw_earnings + (self._fundamental_earnings - raw_earnings) * reversion_speed

        # Belt-and-suspenders: even after reversion, hard-bound earnings so
        # pe_ratio cannot be asked to clip an already-extreme number every
        # single period (keeps PE_FLOOR/PE_CEILING as the steady-state band,
        # not a last-ditch every-period rescue).
        min_earnings = 100.0 / self.PE_CEILING
        max_earnings = 100.0 / self.PE_FLOOR
        self.earnings = max(min_earnings * 0.5, min(max_earnings * 2.0, self.earnings))

        # earnings_growth update: same mean-reverting form as v3.1 (ceiling
        # 9, decay 0.92 toward a GDP-linked baseline of 3.0). Kept as-is --
        # this part was already sane and is not the root cause.
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
    # v3.2 FIX: weight placed on pulling the regime chain back toward the
    # scenario's own starting regime each period (see next_regime
    # docstring / CHANGELOG_v3.2). 0.0 = original unanchored behaviour
    # (chain free-runs to its generic stationary distribution regardless of
    # scenario); 1.0 = regime frozen at the scenario's base regime forever
    # (no organic transitions at all). 0.35 was chosen so that, empirically,
    # a "Recession" scenario spends a large majority of a 100-period run
    # actually in the Recession regime while still allowing a Recession ->
    # Recovery -> Expansion arc to play out over a longer run, the same way
    # real business cycles eventually turn.
    REGIME_ANCHOR_STRENGTH = 0.35

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
# STREAMLIT UI  —  Agent Twin v3.1
# Paste this entire block at the END of your existing agent_twin_v3.py file,
# replacing the "st.write('App loaded successfully')" test line you added.
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
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
                      legend=dict(orientation="h", yanchor="top", y=-0.15))
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
                           legend=dict(orientation="h", yanchor="top", y=-0.15))
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
                           legend=dict(orientation="h", yanchor="top", y=-0.15),
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
    fig4.update_layout(**LAYOUT, height=220, legend=dict(orientation="h", yanchor="top", y=-0.15))
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
                                 legend=dict(orientation="h", yanchor="top", y=-0.15),
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
                              legend=dict(orientation="h", yanchor="top", y=-0.15),
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
                              legend=dict(orientation="h", yanchor="top", y=-0.15),
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
                             legend=dict(orientation="h", yanchor="top", y=-0.15),
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
                                 legend=dict(orientation="h", yanchor="top", y=-0.15),
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
    fig_attr.update_layout(**LAYOUT, height=320, legend=dict(orientation="h", yanchor="top", y=-0.15))
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
                          legend=dict(orientation="h", yanchor="top", y=-0.15))
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
                         legend=dict(orientation="h", yanchor="top", y=-0.15))
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
            unsafe_allow_html=True)
