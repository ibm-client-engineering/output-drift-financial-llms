#!/usr/bin/env python3
"""
Portfolio Constraint Checking Benchmark Task

Evaluates LLM agent determinism and faithfulness on portfolio constraint
verification. The agent must recommend whether to approve, reject, or modify
proposed trades based on position limits, sector caps, and regulatory constraints.

Tools:
- get_current_holdings(portfolio_id) - Current portfolio positions
- get_market_data(ticker) - Current price and volume
- check_position_limit(ticker, quantity) - Verify against limits
- calculate_sector_exposure(sector) - Current sector exposure %
- get_regulatory_constraints(region) - Regulatory limits

Ground Truth Labels:
- approve: Trade satisfies all constraints
- reject: Trade violates constraints (with violation type)
- modify: Trade requires adjustment (with suggested modification)

Metrics:
- Constraint Satisfaction: Agent checks all relevant constraints
- Position Limit Adherence: Violations correctly identified
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class TradeDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"


class ConstraintViolation(Enum):
    POSITION_LIMIT = "position_limit_exceeded"
    SECTOR_CAP = "sector_cap_exceeded"
    LIQUIDITY = "insufficient_liquidity"
    CASH_RESERVE = "cash_reserve_violation"
    CONCENTRATION = "concentration_risk"
    NONE = "none"


@dataclass
class ProposedTrade:
    """A proposed trade to be validated against constraints."""
    trade_id: str
    portfolio_id: str
    action: str  # buy or sell
    ticker: str
    quantity: int
    price: float
    reason: str

    @property
    def notional_value(self) -> float:
        return self.quantity * self.price

    def to_prompt(self) -> str:
        """Format trade for agent prompt."""
        return f"""TRADE VALIDATION REQUEST: {self.trade_id}

Portfolio: {self.portfolio_id}
Proposed Action: {self.action.upper()} {self.quantity:,} shares of {self.ticker}
Price: ${self.price:,.2f}
Notional Value: ${self.notional_value:,.2f}
Reason: {self.reason}

Please validate this trade against all portfolio constraints:
1. Position limits (max 5% single stock)
2. Sector caps (max 25% any sector)
3. Liquidity requirements (3-day volume coverage)
4. Cash reserves (min 2% cash)

Use the available tools to verify compliance and provide your recommendation:
- APPROVE: Trade satisfies all constraints
- REJECT: Trade violates constraints (specify which)
- MODIFY: Suggest adjustment to make trade compliant"""


@dataclass
class MockPortfolioContext:
    """Simulated portfolio data for deterministic evaluation."""
    holdings: Dict[str, Dict] = field(default_factory=dict)
    market_data: Dict[str, Dict] = field(default_factory=dict)
    sector_mapping: Dict[str, str] = field(default_factory=dict)
    position_limits: Dict[str, float] = field(default_factory=dict)
    regulatory_constraints: Dict[str, Dict] = field(default_factory=dict)


class PortfolioConstraintTools:
    """Tool implementations for the portfolio constraint task."""

    def __init__(self, context: MockPortfolioContext):
        self.context = context
        self.call_log: List[Dict] = []

    def get_current_holdings(self, portfolio_id: str) -> Dict:
        """Get current portfolio holdings."""
        self.call_log.append({
            "tool": "get_current_holdings",
            "args": {"portfolio_id": portfolio_id}
        })
        holdings = self.context.holdings.get(portfolio_id, {})
        total_value = sum(h.get("market_value", 0) for h in holdings.values())
        return {
            "portfolio_id": portfolio_id,
            "holdings": holdings,
            "total_value": total_value,
            "cash": holdings.get("CASH", {}).get("market_value", 0),
            "cash_pct": holdings.get("CASH", {}).get("market_value", 0) / total_value * 100 if total_value > 0 else 0
        }

    def get_market_data(self, ticker: str) -> Dict:
        """Get current market data for a ticker."""
        self.call_log.append({
            "tool": "get_market_data",
            "args": {"ticker": ticker}
        })
        data = self.context.market_data.get(ticker, {
            "ticker": ticker,
            "price": 100.0,
            "volume_3d_avg": 1000000,
            "sector": "Unknown"
        })
        return data

    def check_position_limit(self, ticker: str, quantity: int, portfolio_value: float) -> Dict:
        """Check if position would exceed limits."""
        self.call_log.append({
            "tool": "check_position_limit",
            "args": {"ticker": ticker, "quantity": quantity, "portfolio_value": portfolio_value}
        })
        market_data = self.context.market_data.get(ticker, {"price": 100.0})
        position_value = quantity * market_data.get("price", 100.0)
        position_pct = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0
        limit = self.context.position_limits.get("single_stock", 5.0)

        return {
            "ticker": ticker,
            "position_value": position_value,
            "position_pct": position_pct,
            "limit_pct": limit,
            "within_limit": position_pct <= limit,
            "excess_pct": max(0, position_pct - limit)
        }

    def calculate_sector_exposure(self, sector: str, portfolio_id: str) -> Dict:
        """Calculate current sector exposure."""
        self.call_log.append({
            "tool": "calculate_sector_exposure",
            "args": {"sector": sector, "portfolio_id": portfolio_id}
        })
        holdings = self.context.holdings.get(portfolio_id, {})
        total_value = sum(h.get("market_value", 0) for h in holdings.values())

        sector_value = 0
        sector_holdings = []
        for ticker, holding in holdings.items():
            if self.context.sector_mapping.get(ticker) == sector:
                sector_value += holding.get("market_value", 0)
                sector_holdings.append(ticker)

        sector_pct = (sector_value / total_value * 100) if total_value > 0 else 0
        limit = self.context.position_limits.get("sector_cap", 25.0)

        return {
            "sector": sector,
            "exposure_value": sector_value,
            "exposure_pct": sector_pct,
            "limit_pct": limit,
            "within_limit": sector_pct <= limit,
            "holdings": sector_holdings
        }

    def get_regulatory_constraints(self, region: str) -> Dict:
        """Get regulatory constraints for a region."""
        self.call_log.append({
            "tool": "get_regulatory_constraints",
            "args": {"region": region}
        })
        return self.context.regulatory_constraints.get(region, {
            "region": region,
            "cash_reserve_min_pct": 2.0,
            "single_stock_max_pct": 5.0,
            "sector_max_pct": 25.0,
            "liquidity_coverage_days": 3
        })

    def get_tools_schema(self) -> List[Dict]:
        """Return JSON schema for all tools."""
        return [
            {
                "name": "get_current_holdings",
                "description": "Get current portfolio holdings and cash position",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "portfolio_id": {"type": "string"}
                    },
                    "required": ["portfolio_id"]
                }
            },
            {
                "name": "get_market_data",
                "description": "Get current market data (price, volume) for a ticker",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"}
                    },
                    "required": ["ticker"]
                }
            },
            {
                "name": "check_position_limit",
                "description": "Check if proposed position exceeds limits",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "portfolio_value": {"type": "number"}
                    },
                    "required": ["ticker", "quantity", "portfolio_value"]
                }
            },
            {
                "name": "calculate_sector_exposure",
                "description": "Calculate current sector exposure percentage",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sector": {"type": "string"},
                        "portfolio_id": {"type": "string"}
                    },
                    "required": ["sector", "portfolio_id"]
                }
            },
            {
                "name": "get_regulatory_constraints",
                "description": "Get regulatory limits for a region",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "region": {"type": "string"}
                    },
                    "required": ["region"]
                }
            }
        ]


def create_test_context() -> MockPortfolioContext:
    """Create a mock context for testing."""
    return MockPortfolioContext(
        holdings={
            "FUND-2025-ALPHA": {
                "AAPL": {"quantity": 5000, "market_value": 975000, "avg_cost": 180.0},
                "MSFT": {"quantity": 3000, "market_value": 1200000, "avg_cost": 380.0},
                "GOOGL": {"quantity": 2000, "market_value": 340000, "avg_cost": 160.0},
                "JPM": {"quantity": 4000, "market_value": 800000, "avg_cost": 190.0},
                "XOM": {"quantity": 3000, "market_value": 330000, "avg_cost": 105.0},
                "CASH": {"quantity": 1, "market_value": 355000, "avg_cost": 1.0}
            }
        },
        market_data={
            "AAPL": {"ticker": "AAPL", "price": 195.00, "volume_3d_avg": 50000000, "sector": "Technology"},
            "MSFT": {"ticker": "MSFT", "price": 400.00, "volume_3d_avg": 20000000, "sector": "Technology"},
            "GOOGL": {"ticker": "GOOGL", "price": 170.00, "volume_3d_avg": 25000000, "sector": "Technology"},
            "JPM": {"ticker": "JPM", "price": 200.00, "volume_3d_avg": 10000000, "sector": "Financial"},
            "XOM": {"ticker": "XOM", "price": 110.00, "volume_3d_avg": 15000000, "sector": "Energy"},
            "NVDA": {"ticker": "NVDA", "price": 480.00, "volume_3d_avg": 40000000, "sector": "Technology"},
            "SMALL_CAP": {"ticker": "SMALL_CAP", "price": 25.00, "volume_3d_avg": 50000, "sector": "Technology"}
        },
        sector_mapping={
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "NVDA": "Technology",
            "SMALL_CAP": "Technology",
            "JPM": "Financial",
            "XOM": "Energy"
        },
        position_limits={
            "single_stock": 5.0,  # 5% max
            "sector_cap": 25.0,   # 25% max
        },
        regulatory_constraints={
            "US": {
                "region": "US",
                "cash_reserve_min_pct": 2.0,
                "single_stock_max_pct": 5.0,
                "sector_max_pct": 25.0,
                "liquidity_coverage_days": 3
            }
        }
    )


# Sample test trades
SAMPLE_TRADES = [
    ProposedTrade(
        trade_id="TRADE-2025-001",
        portfolio_id="FUND-2025-ALPHA",
        action="buy",
        ticker="AAPL",
        quantity=1000,
        price=195.00,
        reason="Increase core tech holding"
    ),
    ProposedTrade(
        trade_id="TRADE-2025-002",
        portfolio_id="FUND-2025-ALPHA",
        action="buy",
        ticker="NVDA",
        quantity=2000,
        price=480.00,
        reason="Add AI exposure"
    ),
    ProposedTrade(
        trade_id="TRADE-2025-003",
        portfolio_id="FUND-2025-ALPHA",
        action="buy",
        ticker="SMALL_CAP",
        quantity=100000,
        price=25.00,
        reason="Small cap opportunity"
    )
]

# Ground truth for test trades
# Portfolio total: ~$4M, Tech exposure already ~$2.5M (62%)
GROUND_TRUTH = {
    "TRADE-2025-001": (TradeDecision.APPROVE, ConstraintViolation.NONE),
    "TRADE-2025-002": (TradeDecision.REJECT, ConstraintViolation.SECTOR_CAP),  # Would push tech > 25%
    "TRADE-2025-003": (TradeDecision.REJECT, ConstraintViolation.LIQUIDITY),  # Low volume stock
}


def example_portfolio_constraint():
    """Demonstrate the portfolio constraint benchmark."""
    print("=" * 60)
    print("PORTFOLIO CONSTRAINT BENCHMARK - EXAMPLE")
    print("=" * 60)

    context = create_test_context()
    tools = PortfolioConstraintTools(context)

    for trade in SAMPLE_TRADES:
        print(f"\n{'='*60}")
        print(f"Trade: {trade.trade_id}")
        print(f"Action: {trade.action.upper()} {trade.quantity:,} {trade.ticker} @ ${trade.price:,.2f}")
        print(f"Notional: ${trade.notional_value:,.2f}")
        print(f"Reason: {trade.reason}")
        decision, violation = GROUND_TRUTH[trade.trade_id]
        print(f"Ground Truth: {decision.value} ({violation.value})")
        print("-" * 40)

        # Simulate agent tool calls
        tools.call_log = []

        # Get current holdings
        holdings = tools.get_current_holdings(trade.portfolio_id)
        print(f"Portfolio value: ${holdings['total_value']:,.2f}")
        print(f"Cash: ${holdings['cash']:,.2f} ({holdings['cash_pct']:.1f}%)")

        # Get market data
        market = tools.get_market_data(trade.ticker)
        print(f"Market data: {market['ticker']} @ ${market['price']:.2f}, vol: {market['volume_3d_avg']:,}")

        # Check position limit
        pos_check = tools.check_position_limit(trade.ticker, trade.quantity, holdings['total_value'])
        print(f"Position limit: {pos_check['position_pct']:.2f}% (limit: {pos_check['limit_pct']}%) -> {'OK' if pos_check['within_limit'] else 'VIOLATED'}")

        # Check sector exposure
        sector = tools.calculate_sector_exposure(market['sector'], trade.portfolio_id)
        new_sector_pct = sector['exposure_pct'] + (trade.notional_value / holdings['total_value'] * 100)
        print(f"Sector ({market['sector']}): {sector['exposure_pct']:.1f}% current, {new_sector_pct:.1f}% after (limit: {sector['limit_pct']}%)")

        # Check liquidity
        liquidity_days = trade.notional_value / (market['volume_3d_avg'] * market['price'])
        print(f"Liquidity: {liquidity_days:.2f} days to trade (limit: 3 days)")

        print(f"\nTools called: {[c['tool'] for c in tools.call_log]}")

    print("\n" + "=" * 60)
    print("METRICS TO MEASURE:")
    print("-" * 60)
    print("1. Constraint Satisfaction: All constraints checked")
    print("2. Position Limit Adherence: Correct violation identification")
    print("3. Decision Determinism: Same approve/reject across runs")
    print("4. Evidence Grounding: Decision cites tool results")


if __name__ == "__main__":
    example_portfolio_constraint()
