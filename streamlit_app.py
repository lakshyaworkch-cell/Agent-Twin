"""
==================================================================================
 AGENT TWIN v2.1 — Robust Institutional Market Simulator
==================================================================================

Changes over v2:
  PART 1 · Noise Reduction
    · Regime transitions: stickier (30% lower transition probabilities off-diagonal)
    · Price noise: base_volatility halved; demand scaling reduced
    · Earnings growth: noise std halved; mean-reversion strengthened
    · Macro drift: vol reduced ~40%; stronger mean-reversion (0.12 → 0.18)
    · Agent _move_toward speeds reduced to prevent overreaction per period
    · Regime bias scaled down so fundamentals carry more weight than luck

  PART 2 · Monte Carlo Validation Mode
    · Run 100 seeds for any scenario; histogram + summary statistics

  PART 3 · Scenario Consistency Tests
    · Automated pass/fail for AI Boom, Recession, Inflation Shock

  PART 4 · Economic Dominance Score
    · R² of economic variables vs asset returns; noise-dominance warning

  PART 5 · Robustness Dashboard
    · Confidence level (High/Medium/Low) from Monte Carlo CoV
==================================================================================
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
    page_title="Agent Twin v2.1 | Robust Institutional Market Simulator",
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
# MARKET REGIMES — PART 1: stickier transitions
# ══════════════════════════════════════════════════════════════════════════════

REGIMES = ["Expansion", "Slowdown", "Recession", "Recovery"]
REGIME_COLORS = {
    "Expansion": C["expansion"], "Slowdown": C["slowdown"],
    "Recession": C["recession"], "Recovery": C["recovery"],
}

# CHANGE: Off-diagonal probabilities reduced ~30%.
# Self-transition probabilities raised to compensate.
# This prevents the regime from randomly flipping every few periods,
# which was the primary driver of outcome variance.
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
    """
    CHANGE: Per-asset expected-return bias scaled down by ~35%.
    Regime bias is now a modifier, not the dominant driver.
    Fundamental valuation signals carry more relative weight.
    """
    biases = {
        "Expansion": {"Stocks":  0.20, "Bonds": -0.03, "Gold":  0.00},
        "Slowdown":  {"Stocks": -0.07, "Bonds":  0.10, "Gold":  0.07},
        "Recession": {"Stocks": -0.30, "Bonds":  0.22, "Gold":  0.13},
        "Recovery":  {"Stocks":  0.17, "Bonds":  0.03, "Gold":  0.03},
    }
    return biases[regime]

# ══════════════════════════════════════════════════════════════════════════════
# ASSET VALUATION LAYER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StockValuation:
    earnings: float
    earnings_growth: float

    @property
    def pe_ratio(self) -> float:
        return 100.0 / self.earnings if self.earnings > 0 else 999.0

    def expected_return_pct(self) -> float:
        earnings_yield = (self.earnings / 100.0) * 100.0
        return earnings_yield + self.earnings_growth

    def is_expensive(self) -> bool:
        return self.pe_ratio > 25

    def is_cheap(self) -> bool:
        return self.pe_ratio < 14

    def update(self, price: float, gdp_growth: float, rng: random.Random):
        # CHANGE: noise std 0.008 → 0.004; stronger mean-reversion on growth
        growth_factor = 1.0 + (gdp_growth / 100.0) * 0.4 + rng.gauss(0, 0.004)
        self.earnings = max(0.5, self.earnings * growth_factor)
        self.earnings_growth = max(-5.0, min(15.0,
            self.earnings_growth * 0.97 + gdp_growth * 0.3 + rng.gauss(0, 0.15)))


@dataclass
class BondValuation:
    yield_pct: float
    duration: float

    def expected_return_pct(self) -> float:
        return self.yield_pct

    def price_sensitivity(self, rate_change_pct: float) -> float:
        return -self.duration * rate_change_pct

    def update(self, interest_rate: float, inflation: float, rng: random.Random):
        # CHANGE: noise std 0.06 → 0.03
        target_yield = interest_rate + max(0, inflation - 2.0) * 0.25
        self.yield_pct = self.yield_pct * 0.92 + target_yield * 0.08 + rng.gauss(0, 0.03)
        self.yield_pct = max(0.1, self.yield_pct)


@dataclass
class GoldValuation:
    inflation_sensitivity: float

    def expected_return_pct(self, inflation: float, real_rate: float) -> float:
        return self.inflation_sensitivity * max(0, inflation - 2.0) - real_rate * 0.5

    def update(self, rng: random.Random):
        # CHANGE: noise std 0.01 → 0.005
        self.inflation_sensitivity = max(0.3, min(1.5,
            self.inflation_sensitivity + rng.gauss(0, 0.005)))


# ══════════════════════════════════════════════════════════════════════════════
# MACRO ENVIRONMENT
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
# ASSET
# ══════════════════════════════════════════════════════════════════════════════

class Asset:
    def __init__(self, name: str, start_price: float, sensitivity: float, base_volatility: float):
        self.name = name
        self.sensitivity = sensitivity
        self.base_volatility = base_volatility
        self.price_history: List[float] = [start_price]
        self.demand_pressure_history: List[float] = [0.0]

    @property
    def price(self) -> float:
        return self.price_history[-1]

    def apply_demand(self, demand_pressure: float, rng: random.Random,
                     regime_bias: float = 0.0) -> float:
        # CHANGE: noise halved; cap tightened from ±9% to ±6%
        noise = rng.gauss(0, self.base_volatility)
        pct_change = demand_pressure * self.sensitivity + noise + regime_bias
        pct_change = max(min(pct_change, 6.0), -6.0)
        new_price = max(self.price * (1 + pct_change / 100.0), 0.01)
        self.price_history.append(new_price)
        self.demand_pressure_history.append(demand_pressure)
        return pct_change

    def total_return_pct(self) -> float:
        if len(self.price_history) < 2:
            return 0.0
        return (self.price_history[-1] / self.price_history[0] - 1) * 100.0

# ══════════════════════════════════════════════════════════════════════════════
# ORDER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentOrder:
    agent_name: str
    allocation_deltas: Dict[str, float]
    rationale: List[str]

# ══════════════════════════════════════════════════════════════════════════════
# BASE AGENT
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
# PENSION FUND
# PART 1: slower rebalancing speeds to reduce period-to-period churn
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

        exp_stock  = stock_val.expected_return_pct() + regime_asset_bias(env.regime)["Stocks"]
        exp_bond   = bond_val.expected_return_pct()
        exp_gold   = gold_val.expected_return_pct(env.inflation, env.real_rate)

        # CHANGE: speeds reduced ~30% to dampen period-to-period allocation churn
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
# HEDGE FUND
# PART 1: slower signal-to-position translation; tighter vol scaling
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
        # CHANGE: lookback extended 6→8 for smoother trend
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

        # CHANGE: execution speed 0.10 → 0.07 (slower position changes per period)
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
# RETAIL INVESTOR
# PART 1: dampened fear/greed update magnitudes
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
        # CHANGE: update coefficients reduced ~35% to prevent fear/greed swings from
        # dominating. Mean reversion strengthened 0.90→0.87 (slightly faster decay).
        s = SENTIMENT_SCORE.get(sentiment, 0)
        self.greed += (max(0, recent_3) * 1.6 + max(0, s) * 4.0) * self.news_sensitivity
        self.greed -= (max(0, -recent_3) * 1.0 + max(0, -s) * 2.0) * self.news_sensitivity
        self.greed = float(np.clip(self.greed, 0, 100))
        self.fear += (max(0, -recent_5) * 2.0 + max(0, -s) * 4.5) * self.news_sensitivity
        self.fear -= (max(0, recent_5) * 0.8 + max(0, s) * 2.0) * self.news_sensitivity
        self.fear = float(np.clip(self.fear, 0, 100))
        # Stronger mean reversion: damps outlier fear/greed spikes
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
        # CHANGE: execution speed 0.20 → 0.14 (smoother allocation changes)
        self._move_toward(deltas, "Stocks", target_stocks, 0.14)
        deltas["Cash"] = -deltas.get("Stocks", 0.0)

        if not rationale:
            rationale.append(f"F/G balanced (fear={self.fear:.0f}, greed={self.greed:.0f}) — hold")
        return AgentOrder(self.name, deltas, rationale)

# ══════════════════════════════════════════════════════════════════════════════
# MARKET
# ══════════════════════════════════════════════════════════════════════════════

class Market:
    def __init__(self, env: MacroEnvironment, seed: int = 42):
        self.env = env
        self.rng = random.Random(seed)
        # CHANGE: base_volatility halved for all assets (was 0.55/0.30/0.50)
        self.assets: Dict[str, Asset] = {
            "Stocks": Asset("Stocks", 100.0, sensitivity=13.0, base_volatility=0.28),
            "Bonds":  Asset("Bonds",  100.0, sensitivity=8.5,  base_volatility=0.15),
            "Gold":   Asset("Gold",   100.0, sensitivity=10.0, base_volatility=0.25),
        }
        self.valuations: Dict = {
            "Stocks": StockValuation(earnings=5.5, earnings_growth=5.0),
            "Bonds":  BondValuation(yield_pct=env.interest_rate, duration=8.5),
            "Gold":   GoldValuation(inflation_sensitivity=0.85),
        }
        self.env_history: List[MacroEnvironment] = [env]
        self.regime_history: List[str] = [env.regime]
        self.expected_return_history: List[Dict[str, float]] = []

    def expected_returns(self, env: MacroEnvironment) -> Dict[str, float]:
        sv: StockValuation = self.valuations["Stocks"]
        bv: BondValuation  = self.valuations["Bonds"]
        gv: GoldValuation  = self.valuations["Gold"]
        bias = regime_asset_bias(env.regime)
        return {
            "Stocks": sv.expected_return_pct() + bias["Stocks"],
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
            for a, d in o.allocation_deltas.items():
                if a in demand:
                    demand[a] += d
        return demand

    def step(self, orders: List[AgentOrder]) -> Dict[str, float]:
        demand = self.aggregate_demand(orders)
        bias = regime_asset_bias(self.env.regime)
        pct_changes = {}
        # CHANGE: demand scaling 10 → 7 to reduce agent-demand price impact
        for name, asset in self.assets.items():
            scaled = demand[name] * 7
            pct_changes[name] = asset.apply_demand(scaled, self.rng, bias[name] * 0.12)

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
# SIMULATION ENGINE
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
    def __init__(self, base_env: MacroEnvironment, periods: int = 100, seed: int = 42):
        self.base_env = base_env
        self.periods  = periods
        self.rng      = random.Random(seed)
        self.market   = Market(base_env, seed=seed)
        self.agents: List[Agent] = [PensionFund(), HedgeFund(), RetailInvestor()]
        self.logs: List[LogEntry] = []
        self.demand_history: List[Dict[str, float]] = []
        self.regime_prob_history: List[Dict[str, float]] = []
        self._regime_probs: Dict[str, float] = {r: 0.25 for r in REGIMES}

    def _drift_environment(self, prev: MacroEnvironment) -> MacroEnvironment:
        def mr(v, base, vol, lo=None, hi=None):
            # CHANGE: mean-reversion strength 0.08 → 0.15; noise vol reduced ~40%
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
        # CHANGE: sentiment shift probability 0.17 → 0.12
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
# PART 2: MONTE CARLO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(base_env: MacroEnvironment, periods: int, n_seeds: int = 100) -> Dict:
    """Run simulation across n_seeds seeds and collect key outcome metrics."""
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

    sr = np.array(stock_returns)
    br = np.array(bond_returns)
    gr = np.array(gold_returns)

    return {
        "stock_returns": sr,
        "bond_returns":  br,
        "gold_returns":  gr,
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
# PART 3: SCENARIO CONSISTENCY TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_consistency_tests(mc_results: Dict, scenario_name: str) -> List[Dict]:
    """Return list of {name, condition_str, pass, value}."""
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
        # Generic tests for any scenario
        tests.append(t("Simulation completed", "all seeds ran",
                       True, f"{mc_results['n_seeds']} seeds"))

    return tests

# ══════════════════════════════════════════════════════════════════════════════
# PART 4: ECONOMIC DOMINANCE SCORE
# ══════════════════════════════════════════════════════════════════════════════

def compute_economic_dominance(engine: SimulationEngine) -> Dict:
    """
    Proxy for economic signal strength:
    R² of a simple linear model: macro vars → asset return per period.
    Uses GDP growth, inflation, interest rate as predictors for stock returns.
    """
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
    warning    = noise_pct > 70  # noise explains >70% → issue warning

    # Per-asset breakdown
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
# PART 5: ROBUSTNESS / CONFIDENCE LEVEL
# ══════════════════════════════════════════════════════════════════════════════

def compute_confidence(mc_results: Dict) -> Dict:
    """
    Confidence based on coefficient of variation (CoV = std/|mean|) of stock returns.
    High:   CoV < 0.5  (outcomes tightly clustered around mean)
    Medium: CoV 0.5–1.2
    Low:    CoV > 1.2  (noise dominates)
    """
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
# SCENARIO PRESETS
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
# NARRATIVE
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
        mc_n_seeds=100, mc_running=False,
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

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    st.sidebar.markdown(
        f"<div style='font-family:monospace;font-size:1.05rem;font-weight:700;color:{C['text']};'>"
        f"AGENT TWIN <span style='color:{C['green']};font-size:.7rem;'>v2.1</span></div>"
        f"<div style='color:{C['text_dim']};font-size:.76rem;margin-bottom:1rem;'>"
        "Robust Institutional Market Simulator</div>",
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
        f"<div style='font-size:.75rem;color:{C['text_dim']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;'>Simulation Settings</div>",
        unsafe_allow_html=True)
    st.sidebar.slider("Periods", 20, 250, key="periods", step=10)
    st.sidebar.number_input("Random Seed", 0, 9999, key="seed", step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='font-size:.75rem;color:{C['text_dim']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;'>Scenario Presets</div>",
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
# ORIGINAL SECTION RENDERERS (unchanged in v2.1)
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown(
        '<div class="at-header">AGENT TWIN <span style="font-size:1rem;color:#16a34a;">v2.1</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="at-sub">Robust valuation-aware institutional market simulator — '
        'noise-reduced, Monte Carlo validated, economically consistent across seeds.</div>',
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
    fig = _fig(h=320, title=dict(text="Expected Returns by Asset Class", font=dict(size=14)))
    fig.add_trace(go.Scatter(x=periods_x, y=[e["Stocks"] for e in er_hist], name="Stocks", line=dict(color=C["stock"], width=2)))
    fig.add_trace(go.Scatter(x=periods_x, y=[e["Bonds"]  for e in er_hist], name="Bonds",  line=dict(color=C["bond"],  width=2)))
    fig.add_trace(go.Scatter(x=periods_x, y=[e["Gold"]   for e in er_hist], name="Gold",   line=dict(color=C["gold"],  width=2)))
    fig.add_hline(y=0, line_dash="dot", line_color=C["border"], line_width=1)
    fig.update_layout(xaxis_title="Period", yaxis_title="Expected Return (%/period)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def render_price_chart(engine: SimulationEngine):
    m = engine.market
    n = len(m.assets["Stocks"].price_history)
    x = _xaxis(n)
    fig = _fig(h=420, title=dict(text="Asset Price Evolution (Base = 100)", font=dict(size=14)))
    for name, col in [("Stocks", C["stock"]), ("Bonds", C["bond"]), ("Gold", C["gold"])]:
        fig.add_trace(go.Scatter(x=x, y=m.assets[name].price_history, name=name,
                                 line=dict(color=col, width=2.2)))
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
            st.markdown(f"**Pension Fund** &nbsp;<span class='badge'>LDI</span><span class='badge'>Contrarian</span>", unsafe_allow_html=True)
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
            st.markdown(f"**Hedge Fund** &nbsp;<span class='badge'>Trend</span><span class='badge'>Valuation</span><span class='badge'>Risk Budget</span>", unsafe_allow_html=True)
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
            st.markdown(f"**Retail Investor** &nbsp;<span class='badge'>Fear/Greed</span><span class='badge'>News</span>", unsafe_allow_html=True)
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

def render_simulation_log(engine: SimulationEngine):
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
            f"<span style='color:{C['text']};'>FR:{l.funding_ratio:.0f}%</span> — {l.headline}."
        ]
        for d in l.details:
            lines.append(f"&nbsp;&nbsp;↳ {d}")
        lines.append(
            f"&nbsp;&nbsp;Stocks {l.stock_change:+.2f}% · "
            f"Bonds {l.bond_change:+.2f}% · Gold {l.gold_change:+.2f}%"
        )
        parts.append("<div class='log-line'>" + "<br>".join(lines) + "</div><br>")

    st.markdown(
        f"<div class='panel' style='max-height:420px;overflow-y:auto;'>" + "".join(parts) + "</div>",
        unsafe_allow_html=True)

def render_institutional_panel(engine: SimulationEngine):
    st.markdown("### Institutional Analysis")
    summary = generate_institutional_summary(engine)
    st.markdown(
        f"<div class='panel' style='line-height:1.7;font-size:.93rem;font-family:IBM Plex Sans,sans-serif;'>"
        + summary.replace("\n\n","<br><br>") + "</div>",
        unsafe_allow_html=True)
    st.caption("Generated entirely from simulation arrays — no external AI API.")

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: MONTE CARLO DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def render_monte_carlo_section():
    st.markdown("### 🎲  Monte Carlo Validation (Parts 2–5)")

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

    # ── Part 2: Summary Statistics ──────────────────────────────────────────
    sr_stats = monte_carlo_stats(mc["stock_returns"])
    br_stats = monte_carlo_stats(mc["bond_returns"])
    gr_stats = monte_carlo_stats(mc["gold_returns"])

    st.markdown("##### Summary Statistics")
    col_labels = ["Metric", "Stocks", "Bonds", "Gold"]
    rows = [
        ("Mean Return",           f"{sr_stats['mean']:+.1f}%",   f"{br_stats['mean']:+.1f}%",   f"{gr_stats['mean']:+.1f}%"),
        ("Median Return",         f"{sr_stats['median']:+.1f}%", f"{br_stats['median']:+.1f}%", f"{gr_stats['median']:+.1f}%"),
        ("Std Deviation",         f"{sr_stats['std']:.1f}%",     f"{br_stats['std']:.1f}%",     f"{gr_stats['std']:.1f}%"),
        ("Best Case",             f"{sr_stats['best']:+.1f}%",   f"{br_stats['best']:+.1f}%",   f"{gr_stats['best']:+.1f}%"),
        ("Worst Case",            f"{sr_stats['worst']:+.1f}%",  f"{br_stats['worst']:+.1f}%",  f"{gr_stats['worst']:+.1f}%"),
        ("P(Positive Return)",    f"{sr_stats['p_pos']:.0f}%",   f"{br_stats['p_pos']:.0f}%",   f"{gr_stats['p_pos']:.0f}%"),
        ("P(Loss)",               f"{sr_stats['p_loss']:.0f}%",  f"{br_stats['p_loss']:.0f}%",  f"{gr_stats['p_loss']:.0f}%"),
    ]

    html_rows = "".join(
        f"<tr><td style='color:{C['text_dim']};padding:.3rem .6rem;font-size:.85rem;'>{r[0]}</td>"
        + "".join(f"<td style='padding:.3rem .6rem;font-size:.85rem;font-family:monospace;'>{v}</td>" for v in r[1:])
        + "</tr>"
        for r in rows
    )
    header_html = "".join(
        f"<th style='color:{C['text_dim']};padding:.3rem .6rem;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;text-align:left;'>{h}</th>"
        for h in col_labels
    )
    st.markdown(
        f"<div class='panel'><table style='width:100%;border-collapse:collapse;'>"
        f"<thead><tr>{header_html}</tr></thead><tbody>{html_rows}</tbody></table></div>",
        unsafe_allow_html=True)

    # ── Histogram ────────────────────────────────────────────────────────────
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

    # ── Part 3: Consistency Tests ────────────────────────────────────────────
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
            f"</div>",
            unsafe_allow_html=True)

        for t in tests:
            badge = "<span class='pass-badge'>PASS</span>" if t["passed"] else "<span class='fail-badge'>FAIL</span>"
            st.markdown(
                f"<div class='panel' style='padding:.5rem .9rem;margin-bottom:.4rem;'>"
                f"{badge} <span style='font-family:monospace;font-size:.83rem;'>{t['name']}</span>"
                f"<span style='color:{C['text_dim']};font-size:.8rem;font-family:monospace;'> — condition: {t['condition']} — result: <b>{t['value']}</b></span>"
                f"</div>",
                unsafe_allow_html=True)

    # ── Part 4: Economic Dominance Score ─────────────────────────────────────
st.markdown("---")
st.markdown("##### 📊  Economic Dominance Score")

# Use the single-run engine if available; otherwise run seed 0
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
    sig = dom["signal_pct"]
    noise = dom["noise_pct"]

    col1, col2, col3 = st.columns(3)

    col1.markdown(
        f"<div class='panel' style='text-align:center;padding:.7rem;'>"
        f"<div class='metric-label'>Economic Signal Strength</div>"
        f"<div class='metric-value'>{sig:.1f}%</div>"
        f"<div class='metric-label' style='margin-top:.2rem;'>R² of macro→returns</div></div>",
        unsafe_allow_html=True)

    col2.markdown(
        f"<div class='panel' style='text-align:center;padding:.7rem;'>"
        f"<div class='metric-label'>Noise Contribution</div>"
        f"<div class='metric-value'>{noise:.1f}%</div>"
        f"<div class='metric-label' style='margin-top:.2rem;'>Unexplained variance</div></div>",
        unsafe_allow_html=True)

    r2_bars = dom.get("asset_r2", {})

    col3.markdown(
        f"<div class='panel' style='text-align:center;padding:.7rem;'>"
        f"<div class='metric-label'>Per-Asset R²</div>"
        + "".join(
            f"<div style='font-family:monospace;font-size:.85rem;margin-top:.2rem;'>"
            f"{a}: {v*100:.1f}%</div>"
            for a, v in r2_bars.items()
        )
        + "</div>",
        unsafe_allow_html=True)

    if dom["warning"]:
        st.markdown(
            f"<div style='background:#3d1210;border:1px solid {C['red']};border-radius:6px;padding:.7rem 1rem;margin-top:.5rem;'>"
            f"<span style='color:{C['red']};font-family:monospace;font-weight:700;'>⚠ NOISE DOMINANCE WARNING</span>"
            f"<span style='color:{C['text_dim']};font-size:.85rem;font-family:monospace;'> — Noise explains {noise:.1f}% of variance. "
            "Economic fundamentals are insufficiently dominant. Consider reducing stochastic noise parameters or increasing simulation length.</span>"
            "</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='background:{C['green_dim']};border:1px solid {C['green']};border-radius:6px;padding:.5rem 1rem;margin-top:.5rem;'>"
            f"<span style='color:{C['green']};font-family:monospace;font-size:.85rem;'>✓ Economic fundamentals are sufficiently dominant ({sig:.1f}% of variance explained).</span>"
            "</div>",
            unsafe_allow_html=True)

    # Small bar chart
    fig_dom = _fig(h=200, title=dict(text="Per-Asset: Economic Signal vs Noise", font=dict(size=12)))
    assets_list = list(r2_bars.keys())
    sig_vals = [r2_bars[a] * 100 for a in assets_list]
    noise_vals = [100 - v for v in sig_vals]

    fig_dom.add_trace(go.Bar(
        x=assets_list,
        y=sig_vals,
        name="Signal",
        marker_color=C["green"],
        opacity=0.8))

    fig_dom.add_trace(go.Bar(
        x=assets_list,
        y=noise_vals,
        name="Noise",
        marker_color=C["red"],
        opacity=0.6))

    fig_dom.update_layout(
        barmode="stack",
        yaxis_title="% of Variance",
        yaxis=dict(range=[0, 100], **LAYOUT["yaxis"])
    )

    st.plotly_chart(fig_dom, use_container_width=True)

    # ── Part 5: Robustness Dashboard ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### 🛡️  Robustness Dashboard")

    conf = compute_confidence(mc)
    level = conf["level"]
    lc    = conf["label_class"]

    st.markdown(
        f"<div class='panel'>"
        f"<div class='metric-label'>Confidence Level</div>"
        f"<div class='{lc}' style='font-size:1.5rem;margin:.2rem 0;'>{level}</div>"
        f"<div style='color:{C['text_dim']};font-size:.82rem;font-family:monospace;'>"
        f"CoV = {conf['cov']:.2f} | Std = {conf['std']:.1f}% | "
        f"Outcome Range = {conf['range']:.1f}pp | P(Positive) = {conf['p_pos']:.0f}%"
        f"</div></div>",
        unsafe_allow_html=True)

    # All scenarios quick table
    st.markdown("**Robustness across all built-in scenarios** (20 seeds each, quick preview)")
    if st.button("Generate robustness table for all scenarios"):
        with st.spinner("Running 20-seed MC for each scenario..."):
            rob_rows = []
            for sname, sparams in SCENARIOS.items():
                senv = MacroEnvironment(**sparams)
                smc  = run_monte_carlo(senv, periods=st.session_state["periods"], n_seeds=20)
                sc   = compute_confidence(smc)
                rob_rows.append({
                    "Scenario": sname,
                    "Avg Stock Return": f"{np.mean(smc['stock_returns']):+.1f}%",
                    "Outcome Range":    f"{np.max(smc['stock_returns'])-np.min(smc['stock_returns']):.1f}pp",
                    "Std Dev":          f"{np.std(smc['stock_returns']):.1f}%",
                    "P(Positive)":      f"{np.mean(smc['stock_returns']>0)*100:.0f}%",
                    "Confidence":       sc["level"],
                })

        df_rob = pd.DataFrame(rob_rows)

        def conf_badge(v):
            cls = "pass-badge" if v=="High" else ("warn-badge" if v=="Medium" else "fail-badge")
            return f"<span class='{cls}'>{v}</span>"

        html_rob = (
            "<div class='panel'><table style='width:100%;border-collapse:collapse;'>"
            "<thead><tr>" +
            "".join(f"<th style='color:{C['text_dim']};padding:.3rem .6rem;font-size:.78rem;text-transform:uppercase;text-align:left;'>{c}</th>"
                    for c in df_rob.columns) +
            "</tr></thead><tbody>"
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

    if engine is None:
        st.markdown(
            f"<div class='panel' style='text-align:center;padding:3rem 1rem;'>"
            f"<div style='font-size:1.05rem;color:{C['text_dim']};'>"
            "No simulation running.<br>Set parameters in the sidebar and click "
            "<b>RUN SIMULATION</b>, or choose a scenario preset.</div></div>",
            unsafe_allow_html=True)
        # Still allow Monte Carlo section even without a single run
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

    st.markdown("### 10 · Simulation Log")
    render_simulation_log(engine)
    st.markdown("---")

    render_institutional_panel(engine)
    st.markdown("---")

    # ── NEW: Robustness & Validation (Parts 2–5) ──────────────────────────────
    render_monte_carlo_section()

    st.markdown("---")
    st.caption(
        "Agent Twin v2.1 — noise-reduced, Monte Carlo validated. "
        "Same scenario now produces consistent directional outcomes across seeds. "
        "Not investment advice."
    )

if __name__ == "__main__":
    main()
