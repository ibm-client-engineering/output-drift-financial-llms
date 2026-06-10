#!/usr/bin/env python3
"""
Unified Agentic Benchmark Runner

Runs all three financial benchmarks from the paper:
  1. Compliance Triage (§5.1) - 4 tools, escalate/dismiss/investigate
  2. Portfolio Constraint (§5.2) - 5 tools, approve/reject/modify
  3. DataOps Exception (§5.3) - 6 tools, auto_fix/escalate/quarantine

Supports Ollama (local) and Anthropic (API) providers.

Usage:
    # Run all benchmarks, 10 cases each (dev mode)
    python run_unified_benchmark.py --model qwen2.5:7b-instruct

    # Full 50-case runs for paper
    python run_unified_benchmark.py --model qwen2.5:7b-instruct --full

    # Single benchmark
    python run_unified_benchmark.py --model qwen2.5:7b-instruct --benchmark compliance

    # Anthropic provider
    python run_unified_benchmark.py --model claude-3-5-haiku-latest --provider anthropic --full
"""

import argparse
import json
import os
import re
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

try:
    from econometrics.benchmarks.run_logger import RunLogger, BatchMetadata
except ImportError:
    from run_logger import RunLogger, BatchMetadata

try:
    import ollama
except ImportError:
    ollama = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import httpx
except ImportError:
    httpx = None


# ============================================================================
# Sampling configuration — single source of truth for replay protocol
# ============================================================================
# The replay protocol (paper §3.2) requires temperature = 0.0 for every
# provider, and a fixed seed where the provider supports one. These constants
# are passed to each provider call AND threaded into the run logs, so the
# logged metadata always reflects the request that was actually made.
#
# NOTE (audit 2026-06-09): prior to this fix the Anthropic runner omitted
# `temperature` entirely (provider default 1.0) while logs hardcoded 0.0.
# Episodes logged before this date with provider == anthropic were sampled at
# the provider-default temperature. See REPRODUCIBILITY.md.

OLLAMA_TEMPERATURE = 0.0
OLLAMA_SEED = 42
ANTHROPIC_TEMPERATURE = 0.0   # Anthropic API exposes no seed parameter
GEMINI_TEMPERATURE = 0.0      # Gemini API exposes no seed parameter


def sampling_params_for(provider: str) -> Dict[str, Any]:
    """Return the sampling parameters actually sent to a provider.

    Used both for making requests and for logging, so the two cannot drift.
    Seed is None for API providers that do not expose one.
    """
    if provider == "ollama":
        return {"temperature": OLLAMA_TEMPERATURE, "seed": OLLAMA_SEED}
    if provider == "anthropic":
        return {"temperature": ANTHROPIC_TEMPERATURE, "seed": None}
    if provider == "gemini":
        return {"temperature": GEMINI_TEMPERATURE, "seed": None}
    raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# Secrets helper — prefers env vars, falls back to vault if configured
# ============================================================================

def get_secret(key: str) -> Optional[str]:
    """Retrieve a secret from environment or vault.

    Priority:
        1. Environment variable (e.g., ANTHROPIC_API_KEY)
        2. HashiCorp Vault (if VAULT_ADDR is set)
        3. None
    """
    # 1. Environment variable
    val = os.environ.get(key)
    if val:
        return val

    # 2. HashiCorp Vault (optional integration)
    vault_addr = os.environ.get("VAULT_ADDR")
    vault_token = os.environ.get("VAULT_TOKEN")
    if vault_addr and vault_token:
        try:
            import urllib.request
            vault_path = os.environ.get("VAULT_SECRET_PATH", "secret/data/drift-runner")
            url = f"{vault_addr}/v1/{vault_path}"
            req = urllib.request.Request(url, headers={"X-Vault-Token": vault_token})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return data.get("data", {}).get("data", {}).get(key)
        except Exception:
            pass  # Vault unavailable, fall through

    return None


# ============================================================================
# Tool definitions — Ollama format
# ============================================================================

# --- Compliance Triage (3 tools) ---
COMPLIANCE_TOOLS = [
    {"type": "function", "function": {
        "name": "check_sanctions",
        "description": "Check if an entity name appears on OFAC sanctions list",
        "parameters": {"type": "object", "properties": {
            "entity_name": {"type": "string", "description": "The entity name to screen"}
        }, "required": ["entity_name"]}
    }},
    {"type": "function", "function": {
        "name": "get_customer_profile",
        "description": "Get risk profile and KYC status for a customer",
        "parameters": {"type": "object", "properties": {
            "customer_id": {"type": "string", "description": "Customer name or ID"}
        }, "required": ["customer_id"]}
    }},
    {"type": "function", "function": {
        "name": "calculate_risk_score",
        "description": "Calculate overall risk score for a transaction",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number", "description": "Transaction amount"},
            "is_offshore": {"type": "boolean", "description": "Is destination offshore"},
            "is_new_customer": {"type": "boolean", "description": "Is customer new"},
            "sanctions_hit": {"type": "boolean", "description": "Any sanctions matches"}
        }, "required": ["amount"]}
    }},
]

# --- Portfolio Constraint (5 tools) ---
PORTFOLIO_TOOLS = [
    {"type": "function", "function": {
        "name": "get_current_holdings",
        "description": "Get current portfolio holdings and cash position",
        "parameters": {"type": "object", "properties": {
            "portfolio_id": {"type": "string", "description": "Portfolio identifier"}
        }, "required": ["portfolio_id"]}
    }},
    {"type": "function", "function": {
        "name": "get_market_data",
        "description": "Get current market data (price, volume) for a ticker",
        "parameters": {"type": "object", "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol"}
        }, "required": ["ticker"]}
    }},
    {"type": "function", "function": {
        "name": "check_position_limit",
        "description": "Check if proposed position exceeds single-stock limits",
        "parameters": {"type": "object", "properties": {
            "ticker": {"type": "string", "description": "Stock ticker"},
            "quantity": {"type": "integer", "description": "Proposed shares"},
            "portfolio_value": {"type": "number", "description": "Total portfolio value"}
        }, "required": ["ticker", "quantity", "portfolio_value"]}
    }},
    {"type": "function", "function": {
        "name": "calculate_sector_exposure",
        "description": "Calculate current sector exposure percentage",
        "parameters": {"type": "object", "properties": {
            "sector": {"type": "string", "description": "Sector name"},
            "portfolio_id": {"type": "string", "description": "Portfolio identifier"}
        }, "required": ["sector", "portfolio_id"]}
    }},
    {"type": "function", "function": {
        "name": "get_regulatory_constraints",
        "description": "Get regulatory limits for a region",
        "parameters": {"type": "object", "properties": {
            "region": {"type": "string", "description": "Regulatory region (e.g., US, EU)"}
        }, "required": ["region"]}
    }},
]

# --- DataOps Exception (6 tools) ---
DATAOPS_TOOLS = [
    {"type": "function", "function": {
        "name": "get_exception_details",
        "description": "Get full context about a data exception including priority and SLA deadline",
        "parameters": {"type": "object", "properties": {
            "exception_id": {"type": "string", "description": "Exception ID to look up"}
        }, "required": ["exception_id"]}
    }},
    {"type": "function", "function": {
        "name": "query_reference_data",
        "description": "Look up canonical value in reference data (tickers, CUSIPs, currencies)",
        "parameters": {"type": "object", "properties": {
            "field": {"type": "string", "description": "Field type to query"},
            "value": {"type": "string", "description": "Value to look up"}
        }, "required": ["field", "value"]}
    }},
    {"type": "function", "function": {
        "name": "get_historical_fixes",
        "description": "Search past fixes for similar exception patterns",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Pattern to search for"}
        }, "required": ["pattern"]}
    }},
    {"type": "function", "function": {
        "name": "validate_fix",
        "description": "Validate a proposed fix before applying",
        "parameters": {"type": "object", "properties": {
            "field": {"type": "string", "description": "Field to fix"},
            "old_value": {"type": "string", "description": "Current invalid value"},
            "new_value": {"type": "string", "description": "Proposed new value"}
        }, "required": ["field", "old_value", "new_value"]}
    }},
    {"type": "function", "function": {
        "name": "apply_fix",
        "description": "Apply and log a fix to the exception",
        "parameters": {"type": "object", "properties": {
            "exception_id": {"type": "string", "description": "Exception ID"},
            "fix": {"type": "object", "description": "Fix details"}
        }, "required": ["exception_id", "fix"]}
    }},
    {"type": "function", "function": {
        "name": "escalate_to_human",
        "description": "Escalate exception to human review",
        "parameters": {"type": "object", "properties": {
            "exception_id": {"type": "string", "description": "Exception ID"},
            "reason": {"type": "string", "description": "Reason for escalation"}
        }, "required": ["exception_id", "reason"]}
    }},
]


# ============================================================================
# Anthropic tool format converter
# ============================================================================

def to_anthropic_tools(ollama_tools: List[Dict]) -> List[Dict]:
    """Convert Ollama tool format to Anthropic format."""
    result = []
    for tool in ollama_tools:
        func = tool["function"]
        result.append({
            "name": func["name"],
            "description": func["description"],
            "input_schema": func["parameters"],
        })
    return result


COMPLIANCE_TOOLS_ANTHROPIC = to_anthropic_tools(COMPLIANCE_TOOLS)
PORTFOLIO_TOOLS_ANTHROPIC = to_anthropic_tools(PORTFOLIO_TOOLS)
DATAOPS_TOOLS_ANTHROPIC = to_anthropic_tools(DATAOPS_TOOLS)


# ============================================================================
# Mock tool execution — Compliance
# ============================================================================

SANCTIONS_DB = {
    "shadow corp": True, "blocked entity": True, "suspicious ltd": True,
    "offshore holdings": True, "sanctioned bank": True,
}

CUSTOMER_DB = {
    "new customer llc": {"risk_level": "high", "kyc_status": "pending", "years": 0},
    "abc corp": {"risk_level": "low", "kyc_status": "complete", "years": 5},
    "legitimate inc": {"risk_level": "low", "kyc_status": "complete", "years": 3},
}


def execute_compliance_tool(name: str, args: Dict) -> Dict:
    if name == "check_sanctions":
        entity = args.get("entity_name", "").lower()
        is_hit = SANCTIONS_DB.get(entity, False)
        return {"entity": args.get("entity_name"), "is_sanctioned": is_hit,
                "list": "OFAC SDN" if is_hit else None, "match_score": 1.0 if is_hit else 0.0}
    elif name == "get_customer_profile":
        cust = args.get("customer_id", "").lower()
        profile = CUSTOMER_DB.get(cust, {"risk_level": "unknown", "kyc_status": "incomplete", "years": 0})
        return {"customer": args.get("customer_id"), "risk_level": profile["risk_level"],
                "kyc_status": profile["kyc_status"], "relationship_years": profile["years"]}
    elif name == "calculate_risk_score":
        score = 0.0
        if args.get("amount", 0) > 50000: score += 0.3
        if args.get("is_offshore", False): score += 0.2
        if args.get("is_new_customer", False): score += 0.2
        if args.get("sanctions_hit", False): score += 0.4
        return {"risk_score": min(score, 1.0),
                "risk_level": "HIGH" if score > 0.6 else "MEDIUM" if score > 0.3 else "LOW"}
    return {"error": f"Unknown tool: {name}"}


# ============================================================================
# Mock tool execution — Portfolio
# ============================================================================

# Loaded lazily from trades.json
_PORTFOLIO_DATA = None

def _load_portfolio_data():
    global _PORTFOLIO_DATA
    if _PORTFOLIO_DATA is None:
        path = Path(__file__).parent / "portfolio_constraint" / "data" / "trades.json"
        with open(path) as f:
            _PORTFOLIO_DATA = json.load(f)
    return _PORTFOLIO_DATA


def execute_portfolio_tool(name: str, args: Dict) -> Dict:
    data = _load_portfolio_data()
    holdings = data.get("current_holdings", {})
    market = data.get("market_data", {})
    meta = data.get("metadata", {})
    constraints = meta.get("constraints", {})

    if name == "get_current_holdings":
        total_value = sum(h.get("market_value", 0) for h in holdings.values())
        cash = holdings.get("CASH", {}).get("market_value", 0)
        return {"portfolio_id": args.get("portfolio_id", "FUND-2025-ALPHA"),
                "total_value": total_value, "cash": cash,
                "cash_pct": round(cash / total_value * 100, 2) if total_value > 0 else 0,
                "num_positions": len(holdings) - 1}

    elif name == "get_market_data":
        ticker = args.get("ticker", "")
        md = market.get(ticker, {"price": 100.0, "volume_3d_avg": 1000000, "sector": "Unknown"})
        return {"ticker": ticker, "price": md.get("price", 100.0),
                "volume_3d_avg": md.get("volume_3d_avg", 1000000),
                "sector": md.get("sector", "Unknown")}

    elif name == "check_position_limit":
        ticker = args.get("ticker", "")
        quantity = args.get("quantity", 0)
        portfolio_value = args.get("portfolio_value", 10000000)
        price = market.get(ticker, {}).get("price", 100.0)
        position_value = quantity * price
        position_pct = position_value / portfolio_value * 100 if portfolio_value > 0 else 0
        limit = constraints.get("single_stock_max_pct", 5.0)
        return {"ticker": ticker, "position_value": position_value,
                "position_pct": round(position_pct, 2), "limit_pct": limit,
                "within_limit": position_pct <= limit}

    elif name == "calculate_sector_exposure":
        sector = args.get("sector", "")
        total_value = sum(h.get("market_value", 0) for h in holdings.values())
        sector_value = sum(h.get("market_value", 0) for t, h in holdings.items()
                          if h.get("sector") == sector)
        sector_pct = sector_value / total_value * 100 if total_value > 0 else 0
        limit = constraints.get("sector_max_pct", 25.0)
        return {"sector": sector, "exposure_pct": round(sector_pct, 2),
                "limit_pct": limit, "within_limit": sector_pct <= limit}

    elif name == "get_regulatory_constraints":
        return {"region": args.get("region", "US"),
                "cash_reserve_min_pct": constraints.get("cash_reserve_min_pct", 2.0),
                "single_stock_max_pct": constraints.get("single_stock_max_pct", 5.0),
                "sector_max_pct": constraints.get("sector_max_pct", 25.0),
                "liquidity_coverage_days": constraints.get("liquidity_coverage_days", 3)}

    return {"error": f"Unknown tool: {name}"}


# ============================================================================
# Mock tool execution — DataOps
# ============================================================================

REFERENCE_DATA = {
    "ticker": {"MSFT": "MSFT", "AAPL": "AAPL", "GOOG": "GOOGL", "GOOGL": "GOOGL"},
    "cusip": {"594918104": "MSFT", "037833100": "AAPL"},
}

HISTORICAL_FIXES = {
    "negative_price": [{"resolution": "absolute_value", "success_rate": 0.95, "count": 150}],
    "date_format": [{"resolution": "convert_to_ISO", "success_rate": 1.0, "count": 500}],
    "price_must_be_positive": [{"resolution": "absolute_value", "success_rate": 0.95, "count": 150}],
    "ticker_mismatch": [{"resolution": "map_to_canonical", "success_rate": 0.98, "count": 200}],
    "missing_cusip": [{"resolution": "escalate", "success_rate": 0.0, "count": 50}],
    "missing_field": [{"resolution": "escalate", "success_rate": 0.0, "count": 50}],
}


def execute_dataops_tool(name: str, args: Dict) -> Dict:
    if name == "get_exception_details":
        return {"exception_id": args.get("exception_id", ""),
                "created_at": "2025-01-15T10:30:00Z", "priority": "high",
                "sla_deadline": "2025-01-15T11:30:00Z", "similar_exceptions_today": 3}

    elif name == "query_reference_data":
        field = args.get("field", "")
        value = args.get("value", "").upper()
        ref = REFERENCE_DATA.get(field, {})
        canonical = ref.get(value)
        return {"field": field, "query_value": value,
                "canonical_value": canonical, "match_found": canonical is not None}

    elif name == "get_historical_fixes":
        pattern = args.get("pattern", "").lower()
        results = []
        for key, fixes in HISTORICAL_FIXES.items():
            if pattern in key or key in pattern:
                results.extend(fixes)
        return {"pattern": pattern, "fixes_found": len(results), "fixes": results[:5]}

    elif name == "validate_fix":
        field = args.get("field", "")
        new_value = args.get("new_value", "")
        is_valid = True
        errors = []
        if field == "trade_price":
            try:
                v = float(new_value) if isinstance(new_value, str) else new_value
                if v <= 0: is_valid, errors = False, ["Price must be positive"]
            except (ValueError, TypeError):
                is_valid, errors = False, ["Invalid number"]
        return {"field": field, "is_valid": is_valid, "validation_errors": errors}

    elif name == "apply_fix":
        return {"exception_id": args.get("exception_id", ""), "status": "fixed",
                "fix_applied": args.get("fix", {})}

    elif name == "escalate_to_human":
        return {"exception_id": args.get("exception_id", ""), "status": "escalated",
                "escalation_queue": "data_quality_team"}

    return {"error": f"Unknown tool: {name}"}


# ============================================================================
# Benchmark configurations
# ============================================================================

BENCHMARKS = {
    "compliance": {
        "name": "Compliance Triage",
        "tools_ollama": COMPLIANCE_TOOLS,
        "tools_anthropic": COMPLIANCE_TOOLS_ANTHROPIC,
        "execute_tool": execute_compliance_tool,
        "decisions": ["escalate", "dismiss", "investigate"],
        "system_prompt": """You are a compliance analyst. Analyze the alert and decide:
- ESCALATE: Forward to compliance team (high risk indicators)
- DISMISS: Close as false positive (normal business)
- INVESTIGATE: Need more information

IMPORTANT: Use the tools to gather evidence BEFORE deciding.
After gathering evidence, state your final decision clearly as: DECISION: [ESCALATE/DISMISS/INVESTIGATE]""",
        "data_loader": lambda: _load_json("compliance_triage/data/alerts.json", "alerts"),
        "case_formatter": lambda c: (
            f"COMPLIANCE ALERT: {c['alert_id']}\n"
            f"Amount: ${c['amount']:,.2f} {c['currency']}\n"
            f"Sender: {c['sender']} -> Receiver: {c['receiver']}\n"
            f"Destination: {c['country']}\n"
            f"Flags: {', '.join(c['flags'])}\n"
            f"Use the available tools to investigate, then provide your decision."
        ),
        "decision_key": "ground_truth",
    },
    "portfolio": {
        "name": "Portfolio Constraint",
        "tools_ollama": PORTFOLIO_TOOLS,
        "tools_anthropic": PORTFOLIO_TOOLS_ANTHROPIC,
        "execute_tool": execute_portfolio_tool,
        "decisions": ["approve", "reject", "modify"],
        "system_prompt": """You are a portfolio compliance officer. Validate the proposed trade against constraints:
- APPROVE: Trade satisfies all constraints
- REJECT: Trade violates constraints (specify which)
- MODIFY: Suggest adjustment to make trade compliant

Constraints: single stock max 5%, sector max 25%, cash reserve min 2%, 3-day liquidity coverage.
Use the available tools to verify compliance, then state: DECISION: [APPROVE/REJECT/MODIFY]""",
        "data_loader": lambda: _load_json("portfolio_constraint/data/trades.json", "trades"),
        "case_formatter": lambda c: (
            f"TRADE VALIDATION: {c['trade_id']}\n"
            f"Portfolio: {c.get('portfolio_id', 'FUND-2025-ALPHA')}\n"
            f"Action: {c['action'].upper()} {c['quantity']:,} shares of {c['ticker']}\n"
            f"Price: ${c['price']:,.2f}\n"
            f"Notional: ${c['quantity'] * c['price']:,.2f}\n"
            f"Reason: {c.get('reason', 'N/A')}\n"
            f"Validate against all portfolio constraints using the available tools."
        ),
        "decision_key": "ground_truth",
    },
    "dataops": {
        "name": "DataOps Exception",
        "tools_ollama": DATAOPS_TOOLS,
        "tools_anthropic": DATAOPS_TOOLS_ANTHROPIC,
        "execute_tool": execute_dataops_tool,
        "decisions": ["auto_fix", "escalate", "quarantine"],
        "system_prompt": """You are a data quality engineer. Analyze the data exception and decide:
- AUTO_FIX: Apply automatic correction (provide the fix details)
- ESCALATE: Requires human review (provide reason)
- QUARANTINE: Cannot determine action, needs investigation

Use the available tools to research the issue, then state: DECISION: [AUTO_FIX/ESCALATE/QUARANTINE]""",
        "data_loader": lambda: _load_json("dataops_exception/data/exceptions.json", "exceptions"),
        "case_formatter": lambda c: (
            f"DATA EXCEPTION: {c['exception_id']}\n"
            f"Source: {c['source']}\n"
            f"Type: {c['exception_type']}\n"
            f"Field: {c['field']} = {c['value']}\n"
            f"Rule: {c['rule_violated']}\n"
            f"Record: {json.dumps(c.get('record', {}))}\n"
            f"Use the available tools to investigate, then provide your decision."
        ),
        "decision_key": "ground_truth",
    },
}


def _load_json(rel_path: str, key: str) -> List[Dict]:
    """Load JSON data file relative to benchmarks directory."""
    path = Path(__file__).parent / rel_path
    if not path.exists():
        print(f"Warning: {path} not found")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get(key, data if isinstance(data, list) else [])


# ============================================================================
# Generic agent runner — Ollama
# ============================================================================

# Models known not to support Ollama tool calling
_NO_TOOL_MODELS = set()


def run_agent_ollama(
    client, model: str, case: Dict, benchmark_cfg: Dict, max_turns: int = 5
) -> Dict:
    """Run an agent on a case using Ollama.

    Falls back to text-only mode if model doesn't support tools.
    """
    system_prompt = benchmark_cfg["system_prompt"]
    user_prompt = benchmark_cfg["case_formatter"](case)
    execute_tool = benchmark_cfg["execute_tool"]
    valid_decisions = benchmark_cfg["decisions"]

    use_tools = model not in _NO_TOOL_MODELS

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tools_used = []
    final_decision = None
    content = ""

    for turn in range(max_turns):
        try:
            kwargs = {
                "model": model, "messages": messages,
                "options": {"temperature": OLLAMA_TEMPERATURE, "seed": OLLAMA_SEED},
            }
            if use_tools:
                kwargs["tools"] = benchmark_cfg["tools_ollama"]
            resp = client.chat(**kwargs)
        except Exception as e:
            if "does not support tools" in str(e):
                _NO_TOOL_MODELS.add(model)
                use_tools = False
                resp = client.chat(
                    model=model, messages=messages,
                    options={"temperature": OLLAMA_TEMPERATURE, "seed": OLLAMA_SEED},
                )
            else:
                raise

        msg = resp.get("message", {})
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # Parse decision from content
        if content:
            final_decision = _extract_decision(content, valid_decisions)

        # Process tool calls
        if tool_calls:
            messages.append(msg)
            for tc in tool_calls:
                name, args = _parse_ollama_tool_call(tc)
                if name:
                    result = execute_tool(name, args)
                    tools_used.append({"tool": name, "args": args, "result": result})
                    messages.append({"role": "tool", "content": json.dumps(result)})

        if final_decision and not tool_calls:
            break
        if not tool_calls:
            break

    if not final_decision:
        final_decision = _extract_decision(content, valid_decisions)
    decision_source = "parsed" if final_decision else "fallback_last_ontology_label"
    if not final_decision:
        # Explicit, flagged fallback: the run never stated a recognizable
        # decision. Recorded so analysis can exclude or reweight these runs
        # instead of silently counting a fabricated label toward DAR/DCB.
        final_decision = valid_decisions[-1]

    return {"decision": final_decision, "tools_used": tools_used,
            "num_turns": turn + 1, "final_content": content[:500] if content else "",
            "decision_source": decision_source}


def _parse_ollama_tool_call(tc) -> tuple:
    """Parse tool call from Ollama response."""
    try:
        if hasattr(tc, 'function'):
            func = tc.function
            return (getattr(func, 'name', '') or '', getattr(func, 'arguments', {}) or {})
        return (tc.get("function", {}).get("name", ""),
                tc.get("function", {}).get("arguments", {}))
    except Exception:
        return ("", {})


# ============================================================================
# Generic agent runner — Anthropic
# ============================================================================

def run_agent_anthropic(
    client, model: str, case: Dict, benchmark_cfg: Dict, max_turns: int = 5
) -> Dict:
    """Run an agent on a case using Anthropic API."""
    system_prompt = benchmark_cfg["system_prompt"]
    user_prompt = benchmark_cfg["case_formatter"](case)
    execute_tool = benchmark_cfg["execute_tool"]
    valid_decisions = benchmark_cfg["decisions"]

    messages = [{"role": "user", "content": user_prompt}]
    tools_used = []
    final_decision = None
    content = ""

    for turn in range(max_turns):
        resp = client.messages.create(
            model=model, max_tokens=1024, system=system_prompt,
            temperature=ANTHROPIC_TEMPERATURE,
            tools=benchmark_cfg["tools_anthropic"], messages=messages,
        )

        assistant_content = []
        tool_use_blocks = []

        for block in resp.content:
            if block.type == "text":
                content = block.text
                assistant_content.append({"type": "text", "text": content})
                final_decision = _extract_decision(content, valid_decisions) or final_decision
            elif block.type == "tool_use":
                tool_use_blocks.append(block)
                assistant_content.append({
                    "type": "tool_use", "id": block.id,
                    "name": block.name, "input": block.input,
                })

        messages.append({"role": "assistant", "content": assistant_content})

        if tool_use_blocks:
            tool_results = []
            for block in tool_use_blocks:
                result = execute_tool(block.name, block.input)
                tools_used.append({"tool": block.name, "args": block.input, "result": result})
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
            messages.append({"role": "user", "content": tool_results})

        if resp.stop_reason == "end_turn" and not tool_use_blocks:
            break
        if final_decision and not tool_use_blocks:
            break

    if not final_decision:
        final_decision = _extract_decision(content, valid_decisions)
    decision_source = "parsed" if final_decision else "fallback_last_ontology_label"
    if not final_decision:
        # Explicit, flagged fallback — see run_agent_ollama for rationale.
        final_decision = valid_decisions[-1]

    return {"decision": final_decision, "tools_used": tools_used,
            "num_turns": turn + 1, "final_content": content[:500] if content else "",
            "decision_source": decision_source}


# ============================================================================
# Gemini tool format converter
# ============================================================================

def to_gemini_tools(ollama_tools: List[Dict]) -> List[Dict]:
    """Convert Ollama tool format to Gemini function declarations."""
    declarations = []
    for tool in ollama_tools:
        func = tool["function"]
        declarations.append({
            "name": func["name"],
            "description": func["description"],
            "parameters": func["parameters"],
        })
    return [{"function_declarations": declarations}]


# ============================================================================
# Generic agent runner — Gemini
# ============================================================================

def run_agent_gemini(
    api_key: str, model: str, case: Dict, benchmark_cfg: Dict, max_turns: int = 5
) -> Dict:
    """Run an agent on a case using Gemini Generative Language API with tool calling."""
    system_prompt = benchmark_cfg["system_prompt"]
    user_prompt = benchmark_cfg["case_formatter"](case)
    execute_tool = benchmark_cfg["execute_tool"]
    valid_decisions = benchmark_cfg["decisions"]

    gemini_tools = to_gemini_tools(benchmark_cfg["tools_ollama"])
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    model_path = f"models/{model}"

    contents = [{"role": "user", "parts": [{"text": user_prompt}]}]
    tools_used = []
    final_decision = None
    content = ""

    for turn in range(max_turns):
        payload = {
            "contents": contents,
            "tools": gemini_tools,
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {"temperature": GEMINI_TEMPERATURE, "maxOutputTokens": 1024},
        }

        url = f"{base_url}/{model_path}:generateContent?key={api_key}"
        # Retry with backoff for rate limits
        data = None
        for attempt in range(5):
            try:
                with httpx.Client(timeout=180) as http_client:
                    resp = http_client.post(url, json=payload, headers={"Content-Type": "application/json"})
                    resp.raise_for_status()
                    data = resp.json()
                    break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < 4:
                    wait = 2 ** attempt * 5  # 5, 10, 20, 40s
                    print(f"      Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        if data is None:
            raise RuntimeError("Gemini API failed after 5 retries")

        # Parse response
        candidate = data.get("candidates", [{}])[0]
        resp_content = candidate.get("content", {})
        parts = resp_content.get("parts", [])

        assistant_parts = []
        function_calls = []

        for part in parts:
            if "text" in part:
                content = part["text"]
                assistant_parts.append(part)
                final_decision = _extract_decision(content, valid_decisions) or final_decision
            elif "functionCall" in part:
                function_calls.append(part["functionCall"])
                assistant_parts.append(part)

        # Add assistant response to conversation
        contents.append({"role": "model", "parts": assistant_parts})

        if function_calls:
            tool_response_parts = []
            for fc in function_calls:
                name = fc["name"]
                args = fc.get("args", {})
                result = execute_tool(name, args)
                tools_used.append({"tool": name, "args": args, "result": result})
                tool_response_parts.append({
                    "functionResponse": {"name": name, "response": result}
                })
            contents.append({"role": "user", "parts": tool_response_parts})

        finish_reason = candidate.get("finishReason", "")
        if finish_reason == "STOP" and not function_calls:
            break
        if final_decision and not function_calls:
            break

    if not final_decision:
        final_decision = _extract_decision(content, valid_decisions)
    decision_source = "parsed" if final_decision else "fallback_last_ontology_label"
    if not final_decision:
        # Explicit, flagged fallback — see run_agent_ollama for rationale.
        final_decision = valid_decisions[-1]

    return {"decision": final_decision, "tools_used": tools_used,
            "num_turns": turn + 1, "final_content": content[:500] if content else "",
            "decision_source": decision_source}


# ============================================================================
# Decision extraction
# ============================================================================

def _extract_decision(text: str, valid_decisions: List[str]) -> Optional[str]:
    """Extract a decision from response text.

    Matching is word-boundary based: a label only matches as a whole word,
    so e.g. "approve" does NOT match inside "disapprove". When multiple
    DECISION: markers exist, the last one wins (models that revise mid-answer
    state the final decision last).
    """
    if not text:
        return None
    upper = text.upper()

    def _word_match(label: str, haystack: str) -> bool:
        return re.search(rf"\b{re.escape(label.upper())}\b", haystack) is not None

    # Look for explicit DECISION: marker (last marker wins)
    if "DECISION:" in upper:
        after = upper.split("DECISION:")[-1].strip()[:50]
        for d in valid_decisions:
            if _word_match(d, after):
                return d

    # Look for decision words in last 20 words
    tail = " ".join(upper.split()[-20:])
    for d in valid_decisions:
        if _word_match(d, tail):
            return d

    # Look anywhere (word-boundary, not substring)
    for d in valid_decisions:
        if _word_match(d, upper):
            return d

    return None


# ============================================================================
# Experiment runner
# ============================================================================

def run_experiment(
    model: str,
    cases: List[Dict],
    benchmark_key: str,
    num_runs: int = 8,
    provider: str = "ollama",
) -> Dict:
    """Run a full experiment for one benchmark."""
    cfg = BENCHMARKS[benchmark_key]
    decision_key = cfg["decision_key"]

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {cfg['name']} | Model: {model} ({provider})")
    print(f"Cases: {len(cases)}, Runs/case: {num_runs}")
    print(f"{'='*60}")

    # Set up client
    if provider == "ollama":
        if ollama is None:
            raise ImportError("ollama package not installed")
        client = ollama.Client()
        agent_fn = lambda case: run_agent_ollama(client, model, case, cfg)
    elif provider == "anthropic":
        if anthropic is None:
            raise ImportError("anthropic package not installed")
        api_key = get_secret("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in env or vault")
        client = anthropic.Anthropic(api_key=api_key)
        agent_fn = lambda case: run_agent_anthropic(client, model, case, cfg)
    elif provider == "gemini":
        if httpx is None:
            raise ImportError("httpx package not installed")
        gemini_key = get_secret("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not found in env or vault")
        agent_fn = lambda case: run_agent_gemini(gemini_key, model, case, cfg)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Structured run logger
    logger = RunLogger(benchmark=benchmark_key, model=model)

    all_results = []
    for case in cases:
        case_id = case.get("alert_id") or case.get("trade_id") or case.get("exception_id") or "unknown"
        gt = case.get(decision_key, "unknown")
        print(f"\n  Case {case_id} (GT: {gt}):")

        decisions = []
        tool_seqs = []

        # Sampling params actually sent to the provider — logged verbatim so
        # metadata cannot drift from the request (audit finding C2).
        params = sampling_params_for(provider)

        run_tool_outputs: List[List[Any]] = []
        run_latencies: List[float] = []
        run_sources: List[str] = []

        for run in range(num_runs):
            start = time.time()
            result = agent_fn(case)
            latency = time.time() - start

            decisions.append(result["decision"])
            tool_names = [t["tool"] for t in result["tools_used"]]
            tool_outputs = [t["result"] for t in result["tools_used"]]
            tool_seqs.append(tool_names)
            run_tool_outputs.append(tool_outputs)
            run_latencies.append(latency)
            run_sources.append(result.get("decision_source", "parsed"))

            # Faithfulness: 1.0 if decision matches GT, 0.0 otherwise
            faithfulness = 1.0 if result["decision"] == gt else 0.0

            print(f"    Run {run+1}: {result['decision']} | Tools: {tool_names} | {latency:.1f}s")

            # Log this run (deterministic computed after all runs, use provisional)
            logger.log_run(
                case_id=case_id, run_id=run,
                seed=params["seed"], temperature=params["temperature"],
                tool_sequence=tool_names,
                tool_outputs=tool_outputs,
                decision_output=result["decision"],
                deterministic=True,  # provisional; updated after all runs
                faithfulness_score=faithfulness,
                runtime_seconds=latency,
                extra={"decision_source": run_sources[-1]},
            )

        # Compute determinism metrics
        decision_det = len(set(decisions)) == 1
        # Action determinism: same set of tools (ignoring order)
        action_sets = [frozenset(seq) for seq in tool_seqs]
        action_det = len(set(action_sets)) == 1
        # Signature determinism: same tools with same args in same order
        sig_strs = [str(seq) for seq in tool_seqs]
        sig_det = len(set(sig_strs)) == 1

        # Re-log with the correct deterministic flag if not deterministic.
        # IMPORTANT: re-logs carry the ORIGINAL tool outputs and runtimes.
        # (Audit finding C4: the previous version re-logged tool_outputs=[]
        # and runtime_seconds=0.0, destroying the evidence channel for every
        # decision-divergent case group in the corpus.)
        if not decision_det:
            for run in range(num_runs):
                tool_names_r = tool_seqs[run]
                faithfulness_r = 1.0 if decisions[run] == gt else 0.0
                logger.log_run(
                    case_id=case_id, run_id=run,
                    seed=params["seed"], temperature=params["temperature"],
                    tool_sequence=tool_names_r,
                    tool_outputs=run_tool_outputs[run],
                    decision_output=decisions[run],
                    deterministic=False,
                    faithfulness_score=faithfulness_r,
                    runtime_seconds=run_latencies[run],
                    extra={"note": "deterministic_correction",
                           "decision_source": run_sources[run]},
                )

        all_results.append({
            "case_id": case_id,
            "ground_truth": gt,
            "decisions": decisions,
            "is_deterministic": decision_det,
            "is_correct": decisions[0] == gt,
            "action_deterministic": action_det,
            "signature_deterministic": sig_det,
            "tool_sequences": tool_seqs,
            "avg_tools": sum(len(s) for s in tool_seqs) / len(tool_seqs),
        })

    print(f"\n  Logging: {logger.summary()}")

    n = len(all_results)
    return {
        "benchmark": benchmark_key,
        "benchmark_name": cfg["name"],
        "model": model,
        "provider": provider,
        "num_cases": n,
        "num_runs": num_runs,
        "decision_determinism": 100.0 * sum(1 for r in all_results if r["is_deterministic"]) / n,
        "action_determinism": 100.0 * sum(1 for r in all_results if r["action_deterministic"]) / n,
        "signature_determinism": 100.0 * sum(1 for r in all_results if r["signature_deterministic"]) / n,
        "accuracy": 100.0 * sum(1 for r in all_results if r["is_correct"]) / n,
        "avg_tools_per_run": sum(r["avg_tools"] for r in all_results) / n,
        "details": all_results,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Agentic Benchmark Runner")
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct")
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "anthropic", "gemini"])
    parser.add_argument("--benchmark", type=str, default="all",
                        choices=["all", "compliance", "portfolio", "dataops"])
    parser.add_argument("--n-cases", type=int, default=10,
                        help="Number of cases per benchmark (default: 10)")
    parser.add_argument("--full", action="store_true",
                        help="Run all 50 cases per benchmark (overrides --n-cases)")
    parser.add_argument("--n-runs", type=int, default=8,
                        help="Number of runs per case (default: 8)")
    parser.add_argument("--all-models", action="store_true",
                        help="Run all preconfigured models")
    args = parser.parse_args()

    n_cases = 50 if args.full else args.n_cases

    benchmarks_to_run = (
        list(BENCHMARKS.keys()) if args.benchmark == "all"
        else [args.benchmark]
    )

    models = (
        ["qwen2.5:7b-instruct", "gpt-oss:20b"]
        if args.all_models else [args.model]
    )

    # Batch metadata
    batch_meta = BatchMetadata(
        models=models,
        benchmarks=benchmarks_to_run,
        cases_per_benchmark=n_cases,
        runs_per_case=args.n_runs,
    )

    all_results = []

    for model in models:
        for bm_key in benchmarks_to_run:
            cfg = BENCHMARKS[bm_key]
            cases = cfg["data_loader"]()[:n_cases]
            if not cases:
                print(f"Warning: No cases loaded for {bm_key}, skipping")
                continue

            try:
                result = run_experiment(
                    model, cases, bm_key,
                    num_runs=args.n_runs, provider=args.provider,
                )
                all_results.append(result)

                print(f"\n  {bm_key}: Dec={result['decision_determinism']:.1f}% "
                      f"Act={result['action_determinism']:.1f}% "
                      f"Sig={result['signature_determinism']:.1f}% "
                      f"Acc={result['accuracy']:.1f}%")
            except Exception as e:
                print(f"Error running {bm_key} with {model}: {e}")
                import traceback
                traceback.print_exc()

    # Write batch metadata
    meta_path = batch_meta.finalize()
    print(f"\nBatch metadata saved to {meta_path}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    model_slug = models[0].replace(":", "_").replace("/", "_")
    mode = "full" if args.full else f"{n_cases}cases"
    output = output_dir / f"unified_{model_slug}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {output}")

    # Summary table
    print(f"\n{'='*80}")
    print("UNIFIED BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Benchmark':<15} {'Dec%':>6} {'Act%':>6} {'Sig%':>6} {'Acc%':>6} {'Tools':>6}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['model']:<25} {r['benchmark']:<15} "
              f"{r['decision_determinism']:>6.1f} {r['action_determinism']:>6.1f} "
              f"{r['signature_determinism']:>6.1f} {r['accuracy']:>6.1f} "
              f"{r['avg_tools_per_run']:>6.1f}")


if __name__ == "__main__":
    main()
