"""
==================================================================================
 AGENT TWIN — A Digital Twin of Financial Markets
==================================================================================

WHAT THIS APPLICATION IS
-------------------------
Agent Twin is NOT a stock screener, factor model, or portfolio optimizer.
It is an agent-based simulation in which asset prices EMERGE from the
interaction of distinct investor archetypes (Pension Fund, Hedge Fund,
Retail Investor) reacting to a shared macroeconomic environment.

Instead of fitting historical correlations, we simulate behavior:
    macro environment  -->  agent beliefs  -->  agent orders  -->
    aggregate demand    -->  price changes  -->  new environment  -->  ...

WHY THIS MATTERS
-----------------
Classic finance answers "what was the historical correlation between
inflation and bonds?". Agent Twin instead asks "if pension funds behave
like risk-averse long-term allocators, and hedge funds behave like
momentum chasers, and retail investors behave like sentiment-driven
panic sellers — what price path does that produce?" This is closer to
how real markets actually form prices: through the interaction of
heterogeneous, boundedly-rational participants.

ARCHITECTURE (OOP)
-------------------
    Asset             -- a tradeable instrument with a price history
    Agent (abstract)   -- base class for any market participant
        PensionFund    -- long-horizon, risk-averse, contrarian
        HedgeFund      -- momentum-driven, leveraged, trend-chasing
        RetailInvestor -- sentiment-driven, pro-cyclical, panic-prone
    Market             -- owns the assets and the macro environment
    SimulationEngine   -- orchestrates the period-by-period simulation loop

The whole app is a single file by design (per spec) so it can be deployed
to Streamlit Community Cloud with zero configuration beyond
`streamlit_app.py` + `requirements.txt`.

NO external AI APIs are used anywhere. The "Institutional Analysis Panel"
is generated entirely from the simulation's own numbers via templated,
rule-based natural-language generation.
==================================================================================
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ==================================================================================
# PAGE CONFIG & GLOBAL THEME
# ==================================================================================

st.set_page_config(
    page_title="Agent Twin | Digital Twin of Financial Markets",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------------
# COLOR PALETTE — dark institutional theme (Bloomberg / AQR / Bridgewater inspired)
# Dark slate background, deep green accents, muted gold/amber for warnings.
# ----------------------------------------------------------------------------------
COLORS = {
    "bg": "#0b0f0e",
    "panel": "#111614",
    "panel_alt": "#151b18",
    "border": "#1f2a25",
    "text": "#dfe6e2",
    "text_dim": "#8a9690",
    "accent_green": "#16a34a",
    "accent_green_dim": "#0f5132",
    "accent_amber": "#c08a2e",
    "accent_red": "#b3473a",
    "accent_blue": "#3b6e91",
    "stock": "#2fae6a",
    "bond": "#3b82a6",
    "gold": "#c08a2e",
    "pension": "#3b82a6",
    "hedge": "#b3473a",
    "retail": "#c08a2e",
}

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {COLORS["bg"]};
        color: {COLORS["text"]};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {COLORS["panel"]};
        border-right: 1px solid {COLORS["border"]};
    }}
    h1, h2, h3, h4 {{
        color: {COLORS["text"]} !important;
        font-family: 'IBM Plex Mono', 'Courier New', monospace;
        letter-spacing: 0.02em;
    }}
    .agent-twin-header {{
        font-family: 'IBM Plex Mono', 'Courier New', monospace;
        font-size: 1.9rem;
        font-weight: 700;
        color: {COLORS["text"]};
        border-bottom: 2px solid {COLORS["accent_green_dim"]};
        padding-bottom: 0.4rem;
        margin-bottom: 0.1rem;
    }}
    .agent-twin-subheader {{
        color: {COLORS["text_dim"]};
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }}
    .panel-card {{
        background-color: {COLORS["panel"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 6px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
    }}
    .metric-label {{
        color: {COLORS["text_dim"]};
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .rule-row {{
        font-family: 'IBM Plex Mono', 'Courier New', monospace;
        font-size: 0.83rem;
        color: {COLORS["text_dim"]};
        border-left: 2px solid {COLORS["accent_green_dim"]};
        padding: 0.15rem 0 0.15rem 0.6rem;
        margin-bottom: 0.25rem;
    }}
    .log-line {{
        font-family: 'IBM Plex Mono', 'Courier New', monospace;
        font-size: 0.82rem;
        color: {COLORS["text_dim"]};
        padding: 0.1rem 0;
    }}
    .log-period {{
        color: {COLORS["accent_green"]};
        font-weight: 700;
    }}
    .badge {{
        display: inline-block;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.15rem 0.55rem;
        border-radius: 3px;
        background-color: {COLORS["panel_alt"]};
        border: 1px solid {COLORS["border"]};
        color: {COLORS["text_dim"]};
        margin-right: 0.4rem;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'IBM Plex Mono', 'Courier New', monospace;
        color: {COLORS["text"]};
    }}
    .stButton button {{
        background-color: {COLORS["panel_alt"]};
        color: {COLORS["text"]};
        border: 1px solid {COLORS["accent_green_dim"]};
        border-radius: 4px;
        font-family: 'IBM Plex Mono', 'Courier New', monospace;
        font-size: 0.85rem;
    }}
    .stButton button:hover {{
        border-color: {COLORS["accent_green"]};
        color: {COLORS["accent_green"]};
    }}
    hr {{
        border-color: {COLORS["border"]};
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="IBM Plex Mono, Courier New, monospace", size=12),
        xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
)


# ==================================================================================
# DOMAIN MODEL
# ==================================================================================

@dataclass
class MacroEnvironment:
    """
    Snapshot of the macroeconomic environment at a given simulation period.

    This is the shared 'world state' that every agent observes before making
    a decision. Agents do not see each other's portfolios directly — they
    only see this environment plus current asset prices/returns, which is
    a reasonable simplification of how real-world participants act on
    public macro signals plus observed price action.
    """
    inflation: float          # in percent, e.g. 2.0 means 2%
    interest_rate: float      # in percent
    gdp_growth: float         # in percent, can be negative (recession)
    oil_shock: float          # in percent change applied to oil-sensitive logic
    sentiment: str            # one of SENTIMENT_LEVELS

    def is_recession(self) -> bool:
        """A simple recession heuristic: two consecutive quarters of contraction
        is the textbook definition; here we use a single negative-growth read
        as a simplifying proxy since this is a per-period macro snapshot."""
        return self.gdp_growth < 0

    def is_high_inflation(self) -> bool:
        return self.inflation > 5.0

    def is_high_rates(self) -> bool:
        return self.interest_rate > 5.0

    def is_oil_crisis(self) -> bool:
        return self.oil_shock > 15.0


SENTIMENT_LEVELS = ["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"]
SENTIMENT_SCORE = {  # maps qualitative sentiment to a numeric score in [-2, 2]
    "Very Bearish": -2,
    "Bearish": -1,
    "Neutral": 0,
    "Bullish": 1,
    "Very Bullish": 2,
}


class Asset:
    """
    A single tradeable instrument (Stocks, Bonds, or Gold).

    Each asset tracks its own price history and the per-period demand
    pressure exerted on it by all agents combined. Price formation uses a
    deliberately simple, transparent linear-impact model:

        price_change_pct = demand_pressure * sensitivity_coefficient

    This is not meant to be a microstructure-accurate order book model.
    It is meant to make the link between agent behavior and price outcomes
    legible to the user — the entire point of the "digital twin" framing
    is that you can trace *why* a price moved back to *which agent* moved it.
    """

    def __init__(self, name: str, start_price: float, sensitivity: float, base_volatility: float):
        self.name = name
        self.sensitivity = sensitivity            # how reactive price is to demand pressure
        self.base_volatility = base_volatility     # idiosyncratic noise term (per period, in %)
        self.price_history: List[float] = [start_price]
        self.demand_pressure_history: List[float] = [0.0]

    @property
    def price(self) -> float:
        return self.price_history[-1]

    def apply_demand(self, demand_pressure: float, rng: random.Random) -> float:
        """
        Update the asset price given net demand pressure from all agents.

        demand_pressure is a unitless net-buy-minus-sell signal, roughly in
        [-1, 1] in normal conditions (it can exceed that under extreme
        scenario shocks, which is intentional — crises produce fat tails).

        Returns the realized percentage price change for this period.
        """
        noise = rng.gauss(0, self.base_volatility)
        pct_change = demand_pressure * self.sensitivity + noise
        # Cap per-period moves at [-8%, +8%] — large enough to register a
        # clear "crisis period" or "melt-up period" but small enough that
        # even several consecutive extreme periods compound to a severe-but-
        # plausible drawdown/rally rather than a mathematically degenerate
        # price level over a 100-250 period run.
        pct_change = max(min(pct_change, 8.0), -8.0)
        new_price = self.price * (1 + pct_change / 100.0)
        new_price = max(new_price, 0.01)
        self.price_history.append(new_price)
        self.demand_pressure_history.append(demand_pressure)
        return pct_change

    def total_return_pct(self) -> float:
        if len(self.price_history) < 2:
            return 0.0
        return (self.price_history[-1] / self.price_history[0] - 1) * 100.0


# ==================================================================================
# AGENTS
# ==================================================================================

@dataclass
class AgentOrder:
    """
    The result of one agent's decision process for one period: how much it
    wants to change its allocation to each asset, plus a human-readable
    rationale string used to populate the transparent rule log.
    """
    agent_name: str
    allocation_deltas: Dict[str, float]   # e.g. {"Stocks": -0.05, "Bonds": +0.05, "Cash": 0.0}
    rationale: List[str]                   # list of triggered-rule descriptions


class Agent:
    """
    Abstract base class for all market participants.

    Subclasses implement `decide()`, which inspects the MacroEnvironment and
    recent asset performance and returns an AgentOrder describing how the
    agent wants to shift its allocation this period. The base class handles
    the bookkeeping common to all agents: tracking allocation history and
    converting allocation changes into portfolio value and demand pressure.
    """

    name: str = "Agent"
    color: str = "#888888"

    # Per-asset (min, max) allocation bounds. Subclasses override this to
    # reflect realistic mandate constraints — e.g. a pension fund cannot run
    # away to -300% bonds, and a hedge fund's leverage is capped. Without
    # bounds, small per-period rule triggers compound over 100 periods into
    # economically meaningless allocations (and prices that explode/collapse
    # to nonsense), since the same rule can keep firing every period a
    # condition persists (e.g. "bullish sentiment" lasting 30 periods in a row).
    allocation_bounds: Dict[str, tuple] = {}

    def __init__(self, starting_allocation: Dict[str, float], starting_capital: float = 100.0):
        # Allocation is expressed as weights that can exceed 100% (leverage)
        # or go negative (shorting / negative cash, i.e. borrowing), but is
        # always clamped to allocation_bounds to keep the simulation realistic.
        self.starting_allocation = dict(starting_allocation)
        self.allocation_history: List[Dict[str, float]] = [dict(starting_allocation)]
        self.capital = starting_capital
        self.portfolio_value_history: List[float] = [starting_capital]
        self.trade_log: List[AgentOrder] = []

    @property
    def current_allocation(self) -> Dict[str, float]:
        return self.allocation_history[-1]

    def decide(self, env: MacroEnvironment, market: "Market", period: int) -> AgentOrder:
        """Must be implemented by subclasses. Returns an AgentOrder."""
        raise NotImplementedError

    def _clamp(self, asset_name: str, value: float) -> float:
        lo, hi = self.allocation_bounds.get(asset_name, (-3.0, 3.0))
        return max(lo, min(hi, value))

    def _move_toward(self, deltas: Dict[str, float], asset_name: str, target: float, speed: float):
        """
        Nudge `asset_name`'s allocation a fraction (`speed`) of the way from
        its CURRENT level toward `target`, and record the resulting delta in
        `deltas` (additively, so multiple rules touching the same asset in
        one decide() call combine sensibly).

        This is the key negative-feedback mechanism that keeps the
        simulation well-behaved: a rule that fires every period a condition
        holds (e.g. "sentiment stays bearish for 40 periods straight") does
        NOT keep pushing allocation by a constant amount forever — the move
        shrinks as the agent approaches its target and stops once it
        arrives, exactly like a real allocator adjusting toward a desired
        position rather than trading in one direction indefinitely.
        """
        current = self.current_allocation.get(asset_name, 0.0) + deltas.get(asset_name, 0.0)
        delta = (target - current) * speed
        deltas[asset_name] = deltas.get(asset_name, 0.0) + delta
        return delta

    def apply_order(self, order: AgentOrder):
        """Apply the allocation deltas to produce this period's new allocation,
        clamped to this agent's realistic mandate bounds. Clamping is what
        keeps a persistent rule trigger (e.g. 30 bullish periods in a row)
        from compounding into an unbounded allocation."""
        new_alloc = dict(self.current_allocation)
        for asset_name, delta in order.allocation_deltas.items():
            raw = new_alloc.get(asset_name, 0.0) + delta
            new_alloc[asset_name] = self._clamp(asset_name, raw)
        self.allocation_history.append(new_alloc)
        self.trade_log.append(order)

    def update_portfolio_value(self, asset_returns_pct: Dict[str, float]):
        """
        Mark the agent's portfolio to market using this period's asset
        returns and its allocation weights *prior* to this period's trade
        (i.e. the allocation it held going into the period).
        """
        prior_alloc = self.allocation_history[-2] if len(self.allocation_history) >= 2 else self.starting_allocation
        period_return = 0.0
        for asset_name, weight in prior_alloc.items():
            if asset_name == "Cash":
                continue
            r = asset_returns_pct.get(asset_name, 0.0) / 100.0
            period_return += weight * r
        new_value = self.portfolio_value_history[-1] * (1 + period_return)
        self.portfolio_value_history.append(new_value)

    def demand_for(self, asset_name: str, last_order: AgentOrder) -> float:
        """How much net buy/sell pressure this agent exerted on a given asset
        this period, used by the Market to aggregate total demand pressure."""
        return last_order.allocation_deltas.get(asset_name, 0.0)


class PensionFund(Agent):
    """
    Long-horizon, risk-averse, fundamentally-driven allocator.

    Behavioral rules (transparent, deliberately simple):
      - High inflation  -> increase bond allocation (seeking real-yield protection
        is debatable in real markets, but the simplifying convention here is
        "inflation triggers a flight to duration-matched safety," which is how
        pension funds often behave even if it is not always optimal).
      - Weak GDP growth -> reduce equities (de-risking ahead of earnings downturns).
      - Large equity drawdown -> buys the dip (contrarian, long-horizon rebalancing).
      - Elevated equity valuations after a strong rally -> trims equities
        (rebalancing discipline, "sells when valuations become excessive").
    """

    name = "Pension Fund"
    color = COLORS["pension"]
    # A pension fund's mandate keeps it long-only and modestly diversified;
    # it can tilt meaningfully toward bonds/gold but never gets close to a
    # hedge-fund-style leverage profile.
    allocation_bounds = {
        "Stocks": (0.10, 0.65),
        "Bonds": (0.25, 0.75),
        "Gold": (0.0, 0.30),
    }

    def __init__(self):
        super().__init__({"Stocks": 0.40, "Bonds": 0.50, "Gold": 0.10})

    def decide(self, env: MacroEnvironment, market: "Market", period: int) -> AgentOrder:
        deltas: Dict[str, float] = {}
        rationale: List[str] = []
        base_stocks, base_bonds, base_gold = 0.40, 0.50, 0.10

        # Each rule below sets a TARGET allocation and moves a fraction of
        # the way toward it each period (see Agent._move_toward). This
        # means a persistent condition (e.g. inflation staying above 5% for
        # 40 periods straight) pulls the portfolio toward, and then holds
        # it at, a new equilibrium — rather than pushing it the same amount
        # every single period forever, which would compound to nonsense.

        # Rule 1: inflation hedge — target a higher bond / lower equity mix
        if env.is_high_inflation():
            self._move_toward(deltas, "Bonds", target=0.62, speed=0.12)
            self._move_toward(deltas, "Stocks", target=0.28, speed=0.12)
            rationale.append(f"Inflation {env.inflation:.1f}% > 5% threshold → drifting toward higher bond allocation")

        # Rule 2: weak growth -> target lower equities
        if env.gdp_growth < 1.0:
            self._move_toward(deltas, "Stocks", target=0.30, speed=0.10)
            self._move_toward(deltas, "Bonds", target=0.58, speed=0.10)
            rationale.append(f"GDP growth {env.gdp_growth:.1f}% weak (<1%) → drifting toward reduced equity exposure")

        # Rule 3: contrarian buy-the-dip after a sharp stock drawdown
        recent_stock_return = market.recent_return("Stocks", lookback=5)
        if recent_stock_return < -8.0:
            self._move_toward(deltas, "Stocks", target=0.55, speed=0.15)
            self._move_toward(deltas, "Bonds", target=0.35, speed=0.15)
            rationale.append(f"Stocks down {recent_stock_return:.1f}% over 5 periods → contrarian buy, drifting toward higher equities")

        # Rule 4: trim equities after a large rally (valuation discipline)
        elif recent_stock_return > 15.0:
            self._move_toward(deltas, "Stocks", target=0.32, speed=0.12)
            self._move_toward(deltas, "Gold", target=0.22, speed=0.12)
            rationale.append(f"Stocks up {recent_stock_return:.1f}% over 5 periods → valuations excessive, trimming equities")

        # Rule 5: oil crisis -> modest flight to gold as a real-asset hedge
        if env.is_oil_crisis():
            self._move_toward(deltas, "Gold", target=0.22, speed=0.08)
            self._move_toward(deltas, "Stocks", target=0.33, speed=0.08)
            rationale.append(f"Oil shock {env.oil_shock:.1f}% → modest reallocation toward gold")

        # Rule 6 (default / mean reversion): absent any active signal, drift
        # gently back toward the strategic 40/50/10 policy mix — this is
        # what gives a pension fund its long-horizon, anchor-like character.
        if not rationale:
            self._move_toward(deltas, "Stocks", target=base_stocks, speed=0.06)
            self._move_toward(deltas, "Bonds", target=base_bonds, speed=0.06)
            self._move_toward(deltas, "Gold", target=base_gold, speed=0.06)
            rationale.append("No threshold breached → drifting back toward strategic 40/50/10 policy mix")

        return AgentOrder(self.name, deltas, rationale)


class HedgeFund(Agent):
    """
    Momentum-driven, leveraged, trend-following allocator.

    Behavioral rules:
      - Stocks rising -> increase (leveraged) stock exposure, chasing the trend.
      - Stocks falling -> reduce exposure or actively short ("shorts weak assets").
      - Bullish sentiment -> add leverage.
      - Recession or oil crisis -> rotate toward gold as a tactical hedge,
        consistent with "chases trends" applied to commodities too.
    """

    name = "Hedge Fund"
    color = COLORS["hedge"]
    # Hedge funds run leverage (stock weight > 100%, cash negative i.e.
    # borrowed) but real-world prime-broker margin limits cap how far that
    # can run. -50% stocks represents an aggressive net-short stance.
    allocation_bounds = {
        "Stocks": (-0.50, 1.80),
        "Cash": (-0.80, 0.30),
        "Gold": (0.0, 0.40),
        "Bonds": (0.0, 0.30),
    }

    def __init__(self):
        super().__init__({"Stocks": 1.20, "Cash": -0.20})

    def decide(self, env: MacroEnvironment, market: "Market", period: int) -> AgentOrder:
        deltas: Dict[str, float] = {}
        rationale: List[str] = []
        base_stocks, base_cash = 1.20, -0.20

        # Use a smoothed momentum signal (average per-period return over the
        # last 6 periods) rather than a raw point-to-point comparison, so a
        # single noisy down-period near a peak doesn't immediately look like
        # a trend reversal — real momentum desks confirm a trend over
        # multiple observations before flipping a leveraged position.
        recent_stock_return = market.smoothed_momentum("Stocks", lookback=6)
        sentiment_score = SENTIMENT_SCORE.get(env.sentiment, 0)

        # Combine momentum + sentiment into a single target stock weight,
        # then move toward it. Using one target (rather than several
        # competing additive deltas) avoids rules fighting each other every
        # period and gives a clean, bounded equilibrium for any persistent
        # regime — exactly the "chases trends, uses leverage" character of
        # a momentum hedge fund, but one that settles rather than runs away.
        target_stocks = base_stocks
        if recent_stock_return > 1.5:
            target_stocks = 1.50
            rationale.append(f"Stock momentum +{recent_stock_return:.2f}%/period (6-period avg) → target increased leveraged exposure")
        elif recent_stock_return < -1.5:
            target_stocks = 0.70
            rationale.append(f"Stock momentum {recent_stock_return:.2f}%/period (6-period avg) → target cut exposure / shift toward shorting weak assets")

        if sentiment_score >= 1:
            target_stocks += 0.12 * sentiment_score
            rationale.append(f"Sentiment '{env.sentiment}' → target additional leverage")
        elif sentiment_score <= -1:
            target_stocks -= 0.15 * abs(sentiment_score)
            rationale.append(f"Sentiment '{env.sentiment}' → target de-risking")

        if env.is_high_rates():
            target_stocks -= 0.15
            rationale.append(f"Interest rate {env.interest_rate:.1f}% > 5% → target deleveraging (higher financing cost)")

        target_stocks = max(self.allocation_bounds["Stocks"][0], min(self.allocation_bounds["Stocks"][1], target_stocks))
        # Speed governs how much of the gap to the target is closed in one
        # period. 0.12 means a full momentum reversal (e.g. target swinging
        # from 1.65 to 0.20) plays out over several periods rather than as a
        # single violent reallocation — consistent with even aggressive
        # funds unwinding leveraged positions over days/weeks, not instantly.
        speed = 0.08
        self._move_toward(deltas, "Stocks", target=target_stocks, speed=speed)
        # Cash is the financing leg: it moves opposite to stocks 1:1 so that
        # leverage (stocks > 100%) is funded by negative cash (borrowing).
        deltas["Cash"] = deltas.get("Cash", 0.0) - deltas.get("Stocks", 0.0)

        # Tactical gold rotation during recession / oil crisis — kept as a
        # separate small target-seeking move on a different asset so it
        # doesn't interact with the stocks/cash financing pair above.
        if env.is_recession() or env.is_oil_crisis():
            self._move_toward(deltas, "Gold", target=0.30, speed=0.12)
            trigger = "Recession" if env.is_recession() else "Oil crisis"
            rationale.append(f"{trigger} detected → tactical rotation into gold")
        else:
            self._move_toward(deltas, "Gold", target=0.0, speed=0.08)

        if not rationale:
            rationale.append("No momentum or sentiment signal → hold current trend position")

        return AgentOrder(self.name, deltas, rationale)


class RetailInvestor(Agent):
    """
    Emotion-driven, sentiment-following, pro-cyclical allocator.

    Behavioral rules:
      - Bullish sentiment -> buys stocks (chases winners).
      - Bearish sentiment / drawdown -> panic sells.
      - Recession -> sells aggressively, retreats to cash.
      - Rising stock prices -> performance-chasing into the rally.
    """

    name = "Retail Investor"
    color = COLORS["retail"]
    # Retail investors are long-only and unleveraged: they can go anywhere
    # from "all cash, fully panicked" to "fully invested, euphoric", but
    # cannot short or use margin in this simplified model.
    allocation_bounds = {
        "Stocks": (0.0, 1.0),
        "Cash": (0.0, 1.0),
        "Bonds": (0.0, 0.20),
        "Gold": (0.0, 0.20),
    }

    def __init__(self):
        super().__init__({"Stocks": 0.80, "Cash": 0.20})

    def decide(self, env: MacroEnvironment, market: "Market", period: int) -> AgentOrder:
        deltas: Dict[str, float] = {}
        rationale: List[str] = []
        base_stocks = 0.80

        sentiment_score = SENTIMENT_SCORE.get(env.sentiment, 0)
        recent_stock_return_3 = market.recent_return("Stocks", lookback=3)
        recent_stock_return_5 = market.recent_return("Stocks", lookback=5)

        # As with the Hedge Fund, all signals are blended into one target
        # stock weight and the agent moves a fraction of the way toward it
        # each period. This produces the right qualitative behavior — buy
        # winners, panic-sell drawdowns, chase rallies — without a
        # persistent signal (e.g. 30 periods of "Bearish" sentiment in a
        # row) causing stock weight to run past 0% or compound the price
        # into an unrealistic spiral.
        target_stocks = base_stocks

        if sentiment_score >= 1:
            target_stocks += 0.10 * sentiment_score
            rationale.append(f"Sentiment '{env.sentiment}' → target buying winners, higher stock weight")
        elif sentiment_score <= -1:
            target_stocks -= 0.18 * abs(sentiment_score)
            rationale.append(f"Sentiment '{env.sentiment}' → target panic reduction in stock weight")

        if recent_stock_return_3 > 5.0:
            target_stocks += 0.10
            rationale.append(f"Stocks up {recent_stock_return_3:.1f}% (3-period) → target performance-chasing, higher stock weight")

        if recent_stock_return_5 < -6.0:
            target_stocks -= 0.25
            rationale.append(f"Drawdown {recent_stock_return_5:.1f}% (5-period) → target panic sell, much lower stock weight")

        if env.is_recession():
            target_stocks -= 0.20
            rationale.append(f"Recession (GDP {env.gdp_growth:.1f}%) → target aggressive de-risking to cash")

        target_stocks = max(self.allocation_bounds["Stocks"][0], min(self.allocation_bounds["Stocks"][1], target_stocks))
        speed = 0.18  # retail investors react quickly and emotionally, but not instantly
        self._move_toward(deltas, "Stocks", target=target_stocks, speed=speed)
        # Cash absorbs whatever stocks gave up/took — retail here is
        # unleveraged and long-only, so cash is simply 1 - stocks.
        deltas["Cash"] = deltas.get("Cash", 0.0) - deltas.get("Stocks", 0.0)

        if not rationale:
            rationale.append("Neutral sentiment, no strong signal → hold current position")

        return AgentOrder(self.name, deltas, rationale)


# ==================================================================================
# MARKET
# ==================================================================================

class Market:
    """
    Owns the set of tradeable Assets and the current MacroEnvironment.

    Responsible for:
      - aggregating all agents' per-period demand into a net demand-pressure
        figure for each asset,
      - applying that pressure to update asset prices,
      - exposing helper queries (like recent_return) that agents use to make
        decisions, so all agent "perception" of the market is centralized
        and consistent.
    """

    def __init__(self, env: MacroEnvironment, seed: int = 42):
        self.env = env
        self.rng = random.Random(seed)
        self.assets: Dict[str, Asset] = {
            "Stocks": Asset("Stocks", start_price=100.0, sensitivity=14.0, base_volatility=0.5),
            "Bonds": Asset("Bonds", start_price=100.0, sensitivity=9.0, base_volatility=0.35),
            "Gold": Asset("Gold", start_price=100.0, sensitivity=11.0, base_volatility=0.55),
        }
        self.env_history: List[MacroEnvironment] = [env]

    def recent_return(self, asset_name: str, lookback: int) -> float:
        """Percent price change over the last `lookback` periods (or fewer if
        the simulation hasn't run that long yet)."""
        history = self.assets[asset_name].price_history
        if len(history) < 2:
            return 0.0
        lb = min(lookback, len(history) - 1)
        return (history[-1] / history[-1 - lb] - 1) * 100.0

    def smoothed_momentum(self, asset_name: str, lookback: int) -> float:
        """
        A smoothed momentum signal: the average of the per-period returns
        over the lookback window, rather than a single point-to-point
        comparison. This filters out one-off noisy periods from looking
        like a trend reversal, which is what real momentum strategies do
        (e.g. trade on a moving average of returns, not the latest tick).
        """
        history = self.assets[asset_name].price_history
        if len(history) < 3:
            return 0.0
        lb = min(lookback, len(history) - 1)
        window = history[-(lb + 1):]
        period_returns = [
            (window[i] / window[i - 1] - 1) * 100.0 for i in range(1, len(window))
        ]
        return sum(period_returns) / len(period_returns)

    def aggregate_demand(self, orders: List[AgentOrder]) -> Dict[str, float]:
        """
        Sum every agent's allocation delta for each asset to get net demand
        pressure. This is the core "emergence" mechanism: no single agent
        sets the price — the price change is a function of everyone's
        combined behavior.
        """
        demand = {name: 0.0 for name in self.assets}
        for order in orders:
            for asset_name, delta in order.allocation_deltas.items():
                if asset_name in demand:
                    demand[asset_name] += delta
        return demand

    def step(self, orders: List[AgentOrder]) -> Dict[str, float]:
        """Apply one period's aggregated demand to all assets and return the
        realized percentage price changes, keyed by asset name."""
        demand = self.aggregate_demand(orders)
        # Demand deltas are individual agents' allocation changes for this
        # period only (typically a few percentage points, e.g. 0.03 for a
        # 3pp shift). We scale by 10 so a typical single-rule trigger (a few
        # pp) maps to a visible but realistic few-percent price move, rather
        # than the *100 scaling used previously which made price changes an
        # order of magnitude too large and caused multi-period compounding
        # to blow up into economically meaningless price levels.
        scaled_demand = {k: v * 10 for k, v in demand.items()}
        pct_changes = {}
        for name, asset in self.assets.items():
            pct_changes[name] = asset.apply_demand(scaled_demand[name], self.rng)
        return pct_changes

    def set_environment(self, env: MacroEnvironment):
        self.env = env
        self.env_history.append(env)


# ==================================================================================
# SIMULATION ENGINE
# ==================================================================================

@dataclass
class LogEntry:
    period: int
    headline: str
    details: List[str]
    stock_change: float
    bond_change: float
    gold_change: float


class SimulationEngine:
    """
    Orchestrates the full period-by-period simulation:

        for each period:
            1. agents observe the environment (and may drift it slightly,
               representing realistic macro persistence + noise)
            2. each agent decides on an order
            3. market aggregates demand and updates prices
            4. agents mark their portfolios to market
            5. a human-readable log entry is recorded

    The engine also supports macro "drift": rather than holding inflation,
    rates, GDP growth, and oil shock perfectly constant for 100 periods
    (unrealistic), each environment variable does a small mean-reverting
    random walk around the user-set baseline, which gives the simulation
    enough texture for momentum/contrarian rules to actually trigger.
    """

    def __init__(self, base_env: MacroEnvironment, periods: int = 100, seed: int = 42):
        self.base_env = base_env
        self.periods = periods
        self.rng = random.Random(seed)
        self.market = Market(base_env, seed=seed)
        self.agents: List[Agent] = [PensionFund(), HedgeFund(), RetailInvestor()]
        self.logs: List[LogEntry] = []
        self.demand_history: List[Dict[str, float]] = []

    def _drift_environment(self, prev_env: MacroEnvironment, period: int) -> MacroEnvironment:
        """Small mean-reverting random walk around the user-set baseline so
        the 100-period run has realistic texture instead of a flat macro
        backdrop. Sentiment occasionally shifts by one notch."""
        def mean_revert(value, base, vol, lower=None, upper=None):
            reverted = value + (base - value) * 0.08 + self.rng.gauss(0, vol)
            if lower is not None:
                reverted = max(reverted, lower)
            if upper is not None:
                reverted = min(reverted, upper)
            return reverted

        new_inflation = mean_revert(prev_env.inflation, self.base_env.inflation, 0.25, lower=-2.0)
        new_rate = mean_revert(prev_env.interest_rate, self.base_env.interest_rate, 0.2, lower=0.0)
        new_gdp = mean_revert(prev_env.gdp_growth, self.base_env.gdp_growth, 0.3, lower=-10.0, upper=10.0)
        new_oil = mean_revert(prev_env.oil_shock, self.base_env.oil_shock, 1.5, lower=-50.0, upper=150.0)

        # Sentiment: mean-reverts toward the user-selected BASELINE sentiment,
        # not a pure unbiased random walk. Without this pull, sentiment would
        # wander away from the chosen scenario over a 100-period run (e.g. a
        # "Bull Market" baseline of Bullish could drift all the way to Very
        # Bearish purely by chance), which would silently defeat the
        # scenario selector. The bias direction is computed relative to the
        # BASE index so the simulation reliably reflects the chosen regime
        # while still allowing realistic period-to-period texture.
        base_idx = SENTIMENT_LEVELS.index(self.base_env.sentiment)
        idx = SENTIMENT_LEVELS.index(prev_env.sentiment)
        if self.rng.random() < 0.18:
            if idx < base_idx:
                step = 1
            elif idx > base_idx:
                step = -1
            else:
                step = self.rng.choice([-1, 1])
            idx = max(0, min(len(SENTIMENT_LEVELS) - 1, idx + step))
        new_sentiment = SENTIMENT_LEVELS[idx]

        return MacroEnvironment(new_inflation, new_rate, new_gdp, new_oil, new_sentiment)

    def run(self):
        """Execute the full simulation and populate logs/history in place."""
        current_env = self.base_env
        for period in range(1, self.periods + 1):
            if period > 1:
                current_env = self._drift_environment(current_env, period)
                self.market.set_environment(current_env)

            orders = [agent.decide(current_env, self.market, period) for agent in self.agents]
            for agent, order in zip(self.agents, orders):
                agent.apply_order(order)

            pct_changes = self.market.step(orders)

            for agent in self.agents:
                agent.update_portfolio_value(pct_changes)

            demand = self.market.aggregate_demand(orders)
            self.demand_history.append(demand)

            self.logs.append(self._build_log_entry(period, current_env, orders, pct_changes))

        return self

    def _build_log_entry(self, period, env, orders, pct_changes) -> LogEntry:
        triggered = []
        for order in orders:
            for r in order.rationale:
                if "No threshold" not in r and "No momentum" not in r and "Neutral sentiment" not in r:
                    triggered.append(f"{order.agent_name}: {r}")

        if env.is_recession():
            headline = "Recession conditions weighing on risk assets"
        elif env.is_high_inflation():
            headline = "Elevated inflation reshaping allocations"
        elif env.is_oil_crisis():
            headline = "Oil price shock rippling through portfolios"
        elif pct_changes["Stocks"] > 2:
            headline = "Broad-based equity strength"
        elif pct_changes["Stocks"] < -2:
            headline = "Equity markets under pressure"
        else:
            headline = "Markets trading in a narrow range"

        return LogEntry(
            period=period,
            headline=headline,
            details=triggered[:6],  # cap for readability
            stock_change=pct_changes["Stocks"],
            bond_change=pct_changes["Bonds"],
            gold_change=pct_changes["Gold"],
        )


# ==================================================================================
# INSTITUTIONAL ANALYSIS — rule-based natural language generation
# (NO external AI API calls; purely derived from simulation arrays/results)
# ==================================================================================

def generate_institutional_summary(engine: SimulationEngine) -> str:
    """
    Build a short, professional-sounding narrative summary purely from
    simulation statistics: correlations, volatility, drawdowns, and which
    agent contributed the most net demand to which asset. This mirrors a
    real "risk commentary" paragraph a multi-asset desk might write, but is
    entirely templated/rule-based — there is no generative model call here.
    """
    market = engine.market
    stocks = pd.Series(market.assets["Stocks"].price_history)
    bonds = pd.Series(market.assets["Bonds"].price_history)
    gold = pd.Series(market.assets["Gold"].price_history)

    stock_total_return = market.assets["Stocks"].total_return_pct()
    bond_total_return = market.assets["Bonds"].total_return_pct()
    gold_total_return = market.assets["Gold"].total_return_pct()

    stock_vol = stocks.pct_change().dropna().std() * 100
    stock_dd = ((stocks / stocks.cummax()) - 1).min() * 100

    stock_bond_corr = stocks.pct_change().corr(bonds.pct_change())
    stock_gold_corr = stocks.pct_change().corr(gold.pct_change())

    # Net demand contribution by agent, summed over the whole run. We use
    # net STOCKS demand specifically (rather than summing across all assets,
    # including Cash) because Cash deltas are mechanically the offsetting
    # leg of a Stocks trade for several agents — summing them together would
    # cancel out the very signal we're trying to measure ("who pushed
    # equities the hardest"). Stocks is the asset every agent trades, so it
    # is the most meaningful common basis for comparison.
    net_stock_demand_by_agent = {agent.name: 0.0 for agent in engine.agents}
    for agent in engine.agents:
        for order in agent.trade_log:
            net_stock_demand_by_agent[agent.name] += order.allocation_deltas.get("Stocks", 0.0)

    biggest_net_buyer = max(net_stock_demand_by_agent, key=net_stock_demand_by_agent.get)
    biggest_net_seller = min(net_stock_demand_by_agent, key=net_stock_demand_by_agent.get)

    base_env = engine.base_env
    regime_bits = []
    if base_env.is_high_inflation():
        regime_bits.append("an inflationary backdrop")
    if base_env.is_recession():
        regime_bits.append("recessionary growth conditions")
    if base_env.is_high_rates():
        regime_bits.append("a restrictive rate environment")
    if base_env.is_oil_crisis():
        regime_bits.append("an energy price shock")
    if not regime_bits:
        regime_bits.append("a broadly neutral macro backdrop")
    regime_desc = " combined with ".join(regime_bits)

    paragraphs = []

    paragraphs.append(
        f"Under {regime_desc}, the simulated market produced a {stock_total_return:+.1f}% "
        f"cumulative return in equities, {bond_total_return:+.1f}% in bonds, and "
        f"{gold_total_return:+.1f}% in gold over {engine.periods} periods. Realized equity "
        f"volatility was approximately {stock_vol:.1f}% per period, with a maximum drawdown "
        f"of {stock_dd:.1f}%."
    )

    if stock_bond_corr < -0.15:
        corr_desc = (
            f"Stocks and bonds exhibited a negative correlation of {stock_bond_corr:.2f}, "
            "consistent with bonds functioning as a diversifying hedge during episodes of "
            "equity weakness — a classic 'flight to safety' dynamic."
        )
    elif stock_bond_corr > 0.15:
        corr_desc = (
            f"Stocks and bonds moved together with a correlation of {stock_bond_corr:.2f}, "
            "suggesting both asset classes were being driven by a common factor — most "
            "consistent with an inflation or rate-shock regime where both risk assets and "
            "duration sell off together."
        )
    else:
        corr_desc = (
            f"Stocks and bonds were largely uncorrelated ({stock_bond_corr:.2f}), implying "
            "diversification benefits held up over the simulated horizon."
        )
    paragraphs.append(corr_desc)

    if gold_total_return > stock_total_return and gold_total_return > 0:
        gold_desc = (
            "Gold outperformed equities over the period, consistent with its role as a "
            "real-asset hedge that agents rotated into during stress episodes."
        )
    elif stock_gold_corr < -0.1:
        gold_desc = (
            f"Gold showed a {stock_gold_corr:.2f} correlation with equities, behaving as a "
            "partial hedge against risk-asset drawdowns."
        )
    else:
        gold_desc = (
            "Gold's performance tracked broader risk sentiment rather than acting as a "
            "consistent hedge in this run."
        )
    paragraphs.append(gold_desc)

    if biggest_net_buyer == biggest_net_seller:
        behavior_desc = (
            f"At the agent level, net positioning shifts were modest and broadly balanced across "
            f"the three agent types over this run, with {biggest_net_buyer} showing the largest "
            f"absolute swing in equity allocation. This is broadly consistent with the calibrated "
            f"archetypes: pension funds provide a stabilizing, counter-cyclical bid during "
            f"drawdowns and rotate toward bonds as inflation protection; hedge funds amplify "
            f"directional moves through leveraged, momentum-driven positioning; and retail "
            f"investors tend to chase winners during rallies and exit aggressively during "
            f"drawdowns, contributing disproportionately to realized volatility."
        )
    else:
        behavior_desc = (
            f"At the agent level, {biggest_net_buyer} was the largest net source of buying "
            f"pressure on equities over the simulation, while {biggest_net_seller} was the "
            f"largest net seller. This is broadly consistent with the calibrated archetypes: "
            f"pension funds provide a stabilizing, counter-cyclical bid during drawdowns and "
            f"rotate toward bonds as inflation protection; hedge funds amplify directional "
            f"moves through leveraged, momentum-driven positioning; and retail investors tend "
            f"to chase winners during rallies and exit aggressively during drawdowns, "
            f"contributing disproportionately to realized volatility."
        )
    paragraphs.append(behavior_desc)

    closing = (
        "Taken together, the model suggests that the simulated regime favored "
        + ("duration and defensive positioning" if bond_total_return > stock_total_return else "risk assets")
        + " on a risk-adjusted basis, with retail flow acting as a volatility amplifier "
        "rather than a stabilizing force — a pattern broadly consistent with how "
        "heterogeneous investor behavior is understood to shape realized market dynamics."
    )
    paragraphs.append(closing)

    return "\n\n".join(paragraphs)


# ==================================================================================
# SCENARIO PRESETS
# ==================================================================================

SCENARIOS = {
    "Rate Shock": dict(inflation=4.0, interest_rate=7.5, gdp_growth=1.0, oil_shock=0.0, sentiment="Bearish"),
    "Inflation Shock": dict(inflation=8.5, interest_rate=6.0, gdp_growth=1.5, oil_shock=5.0, sentiment="Bearish"),
    "Recession": dict(inflation=1.5, interest_rate=2.0, gdp_growth=-2.5, oil_shock=-10.0, sentiment="Very Bearish"),
    "Oil Crisis": dict(inflation=6.0, interest_rate=5.0, gdp_growth=0.5, oil_shock=35.0, sentiment="Bearish"),
    "Bull Market": dict(inflation=2.0, interest_rate=3.0, gdp_growth=3.5, oil_shock=0.0, sentiment="Bullish"),
    "AI Boom": dict(inflation=2.5, interest_rate=3.5, gdp_growth=4.5, oil_shock=-5.0, sentiment="Very Bullish"),
}


# ==================================================================================
# STREAMLIT APP STATE HELPERS
# ==================================================================================

def init_session_state():
    defaults = dict(
        inflation=3.0,
        interest_rate=4.0,
        gdp_growth=2.0,
        oil_shock=0.0,
        sentiment="Neutral",
        periods=100,
        seed=42,
        engine=None,
        last_run_params=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def apply_scenario(name: str):
    params = SCENARIOS[name]
    st.session_state["inflation"] = params["inflation"]
    st.session_state["interest_rate"] = params["interest_rate"]
    st.session_state["gdp_growth"] = params["gdp_growth"]
    st.session_state["oil_shock"] = params["oil_shock"]
    st.session_state["sentiment"] = params["sentiment"]
    st.session_state["pending_scenario"] = name


def run_simulation():
    env = MacroEnvironment(
        inflation=st.session_state["inflation"],
        interest_rate=st.session_state["interest_rate"],
        gdp_growth=st.session_state["gdp_growth"],
        oil_shock=st.session_state["oil_shock"],
        sentiment=st.session_state["sentiment"],
    )
    engine = SimulationEngine(env, periods=st.session_state["periods"], seed=st.session_state["seed"])
    engine.run()
    st.session_state["engine"] = engine
    st.session_state["last_run_params"] = dict(
        inflation=env.inflation,
        interest_rate=env.interest_rate,
        gdp_growth=env.gdp_growth,
        oil_shock=env.oil_shock,
        sentiment=env.sentiment,
    )


# ==================================================================================
# SIDEBAR — MARKET ENVIRONMENT CONTROLS
# ==================================================================================

def render_sidebar():
    st.sidebar.markdown(
        f"<div style='font-family:monospace;font-size:1.1rem;font-weight:700;color:{COLORS['text']};'>"
        "AGENT TWIN</div>"
        f"<div style='color:{COLORS['text_dim']};font-size:0.78rem;margin-bottom:1rem;'>"
        "Market Environment Controls</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.slider("Inflation (%)", -2.0, 12.0, key="inflation", step=0.1)
    st.sidebar.slider("Interest Rate (%)", 0.0, 12.0, key="interest_rate", step=0.1)
    st.sidebar.slider("GDP Growth (%)", -8.0, 8.0, key="gdp_growth", step=0.1)
    st.sidebar.slider("Oil Price Shock (%)", -50.0, 100.0, key="oil_shock", step=1.0)
    st.sidebar.selectbox("Market Sentiment", SENTIMENT_LEVELS, key="sentiment")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='font-size:0.78rem;color:{COLORS['text_dim']};text-transform:uppercase;"
        "letter-spacing:0.08em;margin-bottom:0.4rem;'>Simulation Settings</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.slider("Periods to Simulate", 20, 250, key="periods", step=10)
    st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, key="seed", step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='font-size:0.78rem;color:{COLORS['text_dim']};text-transform:uppercase;"
        "letter-spacing:0.08em;margin-bottom:0.4rem;'>Scenario Engine</div>",
        unsafe_allow_html=True,
    )

    scenario_cols = st.sidebar.columns(2)
    scenario_names = list(SCENARIOS.keys())
    for i, sname in enumerate(scenario_names):
        col = scenario_cols[i % 2]
        if col.button(sname, key=f"scenario_{sname}", use_container_width=True):
            apply_scenario(sname)
            run_simulation()
            st.rerun()

    st.sidebar.markdown("---")
    run_clicked = st.sidebar.button("▶  RUN SIMULATION", use_container_width=True, type="primary")
    if run_clicked:
        run_simulation()
        st.rerun()

    if st.session_state["engine"] is None:
        st.sidebar.info("Set parameters and click Run Simulation, or choose a preset scenario above.")


# ==================================================================================
# MAIN PANEL RENDERERS
# ==================================================================================

def render_header():
    st.markdown('<div class="agent-twin-header">AGENT TWIN</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="agent-twin-subheader">A digital twin of financial markets — '
        'prices emerge from the interaction of heterogeneous investor agents, '
        'not from fitted historical correlations.</div>',
        unsafe_allow_html=True,
    )


def render_agent_rulebook():
    with st.expander("📋  Agent Decision Rules (transparent rule engine)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Pension Fund** &nbsp;<span class='badge'>Risk-averse</span>", unsafe_allow_html=True)
            rules = [
                "IF inflation > 5% → increase bond allocation",
                "IF GDP growth < 1% → reduce equity exposure",
                "IF stocks fall > 8% (5-period) → contrarian buy",
                "IF stocks rise > 15% (5-period) → trim equities",
                "IF oil shock > 15% → rotate modestly to gold",
            ]
            for r in rules:
                st.markdown(f"<div class='rule-row'>{r}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"**Hedge Fund** &nbsp;<span class='badge'>Momentum</span>", unsafe_allow_html=True)
            rules = [
                "IF stocks rising (3-period) → add leveraged exposure",
                "IF stocks falling (3-period) → cut exposure / short",
                "IF sentiment bullish → add leverage",
                "IF sentiment bearish → de-risk",
                "IF recession or oil crisis → rotate to gold",
                "IF rates > 5% → deleverage",
            ]
            for r in rules:
                st.markdown(f"<div class='rule-row'>{r}</div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"**Retail Investor** &nbsp;<span class='badge'>Sentiment</span>", unsafe_allow_html=True)
            rules = [
                "IF sentiment bullish → buy winners",
                "IF sentiment bearish → panic sell",
                "IF stocks rising (3-period) → chase performance",
                "IF stocks fall > 6% (5-period) → panic sell",
                "IF recession → sell aggressively",
            ]
            for r in rules:
                st.markdown(f"<div class='rule-row'>{r}</div>", unsafe_allow_html=True)


def render_environment_summary(engine: SimulationEngine):
    env = engine.base_env
    cols = st.columns(5)
    labels = ["Inflation", "Interest Rate", "GDP Growth", "Oil Shock", "Sentiment"]
    values = [f"{env.inflation:.1f}%", f"{env.interest_rate:.1f}%", f"{env.gdp_growth:.1f}%",
              f"{env.oil_shock:+.1f}%", env.sentiment]
    for col, label, value in zip(cols, labels, values):
        col.markdown(
            f"<div class='panel-card' style='text-align:center;padding:0.7rem;'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div style='font-size:1.3rem;font-family:monospace;margin-top:0.2rem;'>{value}</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def render_price_chart(engine: SimulationEngine):
    market = engine.market
    periods = list(range(len(market.assets["Stocks"].price_history)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periods, y=market.assets["Stocks"].price_history, name="Stocks",
        line=dict(color=COLORS["stock"], width=2.2),
    ))
    fig.add_trace(go.Scatter(
        x=periods, y=market.assets["Bonds"].price_history, name="Bonds",
        line=dict(color=COLORS["bond"], width=2.2),
    ))
    fig.add_trace(go.Scatter(
        x=periods, y=market.assets["Gold"].price_history, name="Gold",
        line=dict(color=COLORS["gold"], width=2.2),
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text="Asset Price Evolution", font=dict(size=15)),
        xaxis_title="Period",
        yaxis_title="Index Level (Start = 100)",
        height=420,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_allocation_chart(engine: SimulationEngine):
    st.markdown("##### Agent Allocation Evolution")
    tabs = st.tabs([a.name for a in engine.agents])
    for tab, agent in zip(tabs, engine.agents):
        with tab:
            df = pd.DataFrame(agent.allocation_history)
            df.index.name = "Period"
            fig = go.Figure()
            palette = [COLORS["stock"], COLORS["bond"], COLORS["gold"], COLORS["text_dim"]]
            for i, col_name in enumerate(df.columns):
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col_name] * 100, name=col_name,
                    stackgroup="one", line=dict(width=0.5, color=palette[i % len(palette)]),
                    fillcolor=palette[i % len(palette)],
                ))
            fig.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                title=dict(text=f"{agent.name} — Allocation Weight Over Time", font=dict(size=14)),
                xaxis_title="Period",
                yaxis_title="Allocation (%)",
                height=360,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True, key=f"alloc_{agent.name}")


def render_demand_chart(engine: SimulationEngine):
    st.markdown("##### Demand Pressure — Who Is Driving the Market")
    demand_df = pd.DataFrame(engine.demand_history)
    demand_df.index = range(1, len(demand_df) + 1)

    net_by_agent_asset = []
    for agent in engine.agents:
        totals = {"Stocks": 0.0, "Bonds": 0.0, "Gold": 0.0, "Cash": 0.0}
        for order in agent.trade_log:
            for k, v in order.allocation_deltas.items():
                if k in totals:
                    totals[k] += v
        net_by_agent_asset.append((agent.name, totals, agent.color))

    fig = go.Figure()
    asset_list = ["Stocks", "Bonds", "Gold"]
    for asset in asset_list:
        fig.add_trace(go.Bar(
            name=asset,
            x=[name for name, _, _ in net_by_agent_asset],
            y=[totals[asset] * 100 for _, totals, _ in net_by_agent_asset],
            marker_color=COLORS[asset.lower()],
        ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text="Cumulative Net Demand Pressure by Agent (percentage points of allocation)", font=dict(size=14)),
        barmode="group",
        xaxis_title="Agent",
        yaxis_title="Net Allocation Change (pp, summed over run)",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_portfolio_chart(engine: SimulationEngine):
    st.markdown("##### Agent Portfolio Value")
    fig = go.Figure()
    for agent in engine.agents:
        fig.add_trace(go.Scatter(
            x=list(range(len(agent.portfolio_value_history))),
            y=agent.portfolio_value_history,
            name=agent.name,
            line=dict(color=agent.color, width=2),
        ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text="Portfolio Value by Agent (Start = 100)", font=dict(size=14)),
        xaxis_title="Period",
        yaxis_title="Portfolio Value",
        height=380,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_simulation_log(engine: SimulationEngine):
    st.markdown("##### Simulation Log")
    notable_logs = [log for log in engine.logs if log.details]
    show_all = st.checkbox("Show every period (otherwise only periods with triggered rules)", value=False)
    logs_to_show = engine.logs if show_all else notable_logs

    if not logs_to_show:
        st.markdown(
            f"<div class='log-line'>No threshold rules were triggered during this run — "
            "the macro environment stayed within neutral ranges for every agent's rule set.</div>",
            unsafe_allow_html=True,
        )
        return

    max_display = 40
    display_logs = logs_to_show[-max_display:] if len(logs_to_show) > max_display else logs_to_show
    if len(logs_to_show) > max_display:
        st.caption(f"Showing most recent {max_display} of {len(logs_to_show)} matching periods.")

    log_html_parts = []
    for log in display_logs:
        lines = [f"<span class='log-period'>Period {log.period}.</span> {log.headline}."]
        for d in log.details:
            lines.append(f"&nbsp;&nbsp;↳ {d}")
        lines.append(
            f"&nbsp;&nbsp;Stocks {log.stock_change:+.2f}% · Bonds {log.bond_change:+.2f}% · Gold {log.gold_change:+.2f}%"
        )
        log_html_parts.append("<div class='log-line'>" + "<br>".join(lines) + "</div><br>")

    st.markdown(
        f"<div class='panel-card' style='max-height:420px;overflow-y:auto;'>" + "".join(log_html_parts) + "</div>",
        unsafe_allow_html=True,
    )


def render_summary_stats(engine: SimulationEngine):
    market = engine.market
    cols = st.columns(3)
    for col, asset_name in zip(cols, ["Stocks", "Bonds", "Gold"]):
        asset = market.assets[asset_name]
        total_ret = asset.total_return_pct()
        color = COLORS["accent_green"] if total_ret >= 0 else COLORS["accent_red"]
        col.markdown(
            f"<div class='panel-card' style='text-align:center;'>"
            f"<div class='metric-label'>{asset_name} — Total Return</div>"
            f"<div style='font-size:1.6rem;font-family:monospace;color:{color};margin-top:0.2rem;'>{total_ret:+.1f}%</div>"
            f"<div class='metric-label' style='margin-top:0.3rem;'>Final Level: {asset.price:.2f}</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def render_institutional_panel(engine: SimulationEngine):
    st.markdown("### Institutional Analysis")
    summary = generate_institutional_summary(engine)
    st.markdown(
        f"<div class='panel-card' style='line-height:1.65;font-size:0.95rem;'>"
        + summary.replace("\n\n", "<br><br>")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Generated entirely from this simulation's own price, demand, and allocation arrays. "
        "No external AI API is used to produce this text."
    )


# ==================================================================================
# MAIN APP FLOW
# ==================================================================================

def main():
    init_session_state()
    render_sidebar()
    render_header()

    engine: Optional[SimulationEngine] = st.session_state.get("engine")

    render_agent_rulebook()

    st.markdown("---")

    if engine is None:
        st.markdown(
            f"<div class='panel-card' style='text-align:center;padding:3rem 1rem;'>"
            f"<div style='font-size:1.1rem;color:{COLORS['text_dim']};'>"
            "No simulation has been run yet.<br>Configure the market environment in the sidebar "
            "and click <b>RUN SIMULATION</b>, or choose a scenario preset.</div></div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown("### 1 · Market Environment")
    render_environment_summary(engine)

    st.markdown("### 2 · Simulation Results")
    render_summary_stats(engine)
    render_price_chart(engine)

    st.markdown("---")
    st.markdown("### 3 · Investor Agents")
    render_allocation_chart(engine)
    render_demand_chart(engine)
    render_portfolio_chart(engine)

    st.markdown("---")
    st.markdown("### 4 · Simulation Log")
    render_simulation_log(engine)

    st.markdown("---")
    render_institutional_panel(engine)

    st.markdown("---")
    st.caption(
        "Agent Twin is a pedagogical / illustrative simulation. Agent rules, sensitivities, and "
        "pricing mechanics are simplified by design to make the link between investor behavior "
        "and price formation transparent. This is not investment advice and does not forecast "
        "real markets."
    )


if __name__ == "__main__":
    main()
