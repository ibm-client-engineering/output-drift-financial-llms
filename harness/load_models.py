#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Load Testing for LLM Output Drift Framework
Implements closed-loop and open-loop load patterns with financial-specific traffic models.

Key Metrics:
  - queue_time_ms: Time request waits before service
  - service_time_ms: Time to complete LLM call
  - total_latency_ms: End-to-end request latency
  - p50/p95/p99_ms: Latency percentiles
  - tps: Throughput (requests per second)
  - identity_rate: Determinism under load
  - faithfulness_score: Semantic accuracy

Usage:
    # Closed-loop testing
    python harness/load_models.py --mode closed --concurrency 1 2 4 8 16 32 \\
      --models qwen-7b granite-8b --tasks rag sql summary --temps 0.0 0.2

    # Open-loop testing
    python harness/load_models.py --mode open --rate 0.5 1 2 4 8 --burst 4 \\
      --pattern market-open --models qwen-7b --temps 0.0

    # Financial pattern testing
    python harness/load_models.py --mode open --pattern eod-reconciliation \\
      --baseline-rps 2.0 --duration-min 120
"""
import os
import sys
import time
import asyncio
import argparse
import hashlib
import json
import csv
import random
import yaml
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rapidfuzz.distance import Levenshtein
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from existing codebase
from harness.deterministic_retriever import DeterministicRetriever, create_retriever_from_files
from harness.task_definitions import (
    format_rag_prompt,
    extract_citations,
    validate_citations,
    format_sql_prompt,
    format_summary_prompt,
    validate_sql_query,
    validate_summary_json
)
from providers.watsonx import WatsonxProvider

# ----------------------------- Configuration ---------------------------------

# Financial-specific load patterns
FINANCIAL_PATTERNS = {
    "market-open": {
        "baseline_rps": 1.0,
        "spike_multiplier": 3.0,
        "spike_duration_min": 30,
        "pattern": "burst",
        "description": "Market open spike - 3x baseline for 30min"
    },
    "eod-reconciliation": {
        "baseline_rps": 1.0,
        "sustained_multiplier": 2.0,
        "duration_hours": 2,
        "pattern": "sustained",
        "description": "End-of-day reconciliation - 2x sustained for 2 hours"
    },
    "regulatory-filing": {
        "baseline_rps": 0.5,
        "burst_multiplier": 10.0,
        "burst_duration_min": 5,
        "burst_frequency_min": 60,
        "pattern": "periodic_burst",
        "description": "Regulatory filing deadline - 10x bursts every hour"
    },
    "baseline": {
        "baseline_rps": 1.0,
        "pattern": "steady",
        "description": "Steady baseline load"
    }
}

MAX_TOKENS = 512
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- Data Classes ----------------------------------

@dataclass
class LoadRequest:
    """Individual request under load."""
    request_id: str
    task: str
    provider: str
    model: str
    temperature: float
    load_mode: str  # "closed" or "open"
    load_level: int  # concurrency N or arrival rate λ

    # Model resolution tracking
    requested_model: Optional[str] = None
    resolved_model: Optional[str] = None
    provider_supported: Optional[bool] = None

    # Timing
    submit_time: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Results
    output: Optional[str] = None
    output_hash: Optional[str] = None
    success: bool = False
    error: Optional[str] = None

    # Input/output for drift calculation
    prompt: Optional[str] = None
    reference_output: Optional[str] = None

    @property
    def queue_time_ms(self) -> float:
        """Time spent waiting in queue."""
        if self.start_time is None:
            return 0.0
        return (self.start_time - self.submit_time) * 1000

    @property
    def service_time_ms(self) -> float:
        """Time spent being serviced."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    @property
    def total_latency_ms(self) -> float:
        """End-to-end latency."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.submit_time) * 1000


@dataclass
class LoadMetrics:
    """Aggregate metrics for a load test."""
    task: str
    provider: str
    model: str
    temperature: float
    load_mode: str
    load_level: int

    # Model resolution tracking
    requested_model: Optional[str] = None
    resolved_model: Optional[str] = None
    provider_supported: Optional[bool] = None

    # Environment metadata (for reproducibility)
    provider_region: Optional[str] = None
    model_build_version: Optional[str] = None
    test_timestamp: Optional[str] = None

    # Request counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    skipped_requests: int = 0

    # Latency metrics (milliseconds)
    mean_queue_time_ms: float = 0.0
    mean_service_time_ms: float = 0.0
    mean_total_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Throughput
    tps: float = 0.0  # Transactions per second
    test_duration_s: float = 0.0

    # Drift metrics
    identity_rate: float = 0.0
    mean_edit_distance: float = 0.0

    # Faithfulness (if available)
    faithfulness_score: Optional[float] = None

    # SLO compliance
    latency_slo_violations: int = 0
    determinism_slo_violations: int = 0


# ----------------------------- Providers -------------------------------------

class OllamaProvider:
    """Ollama provider for load testing."""
    def __init__(self, host: str = "http://127.0.0.1:11434"):
        self.host = host
        self.name = "ollama"

    async def acomplete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_tokens: int = MAX_TOKENS
    ) -> str:
        """Complete a prompt asynchronously."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens)
            }
        }
        if seed is not None:
            payload["options"]["seed"] = int(seed)

        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{self.host}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"]
            return data.get("response", "")


# ----------------------------- Load Models -----------------------------------

class ClosedLoopLoadModel:
    """
    Closed-loop load model: Maintain exactly N in-flight requests.
    Models backpressure and queue buildup.
    """
    def __init__(self, concurrency: int):
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.pending_requests: List[LoadRequest] = []

    async def submit_request(
        self,
        request: LoadRequest,
        executor_func
    ) -> LoadRequest:
        """Submit a request with concurrency control."""
        async with self.semaphore:
            # Record when we actually start processing
            request.start_time = time.time()

            try:
                result = await executor_func(request)
                request.end_time = time.time()
                request.success = True
                return result
            except Exception as e:
                request.end_time = time.time()
                request.success = False
                request.error = str(e)
                return request


class OpenLoopLoadModel:
    """
    Open-loop load model: Schedule via Poisson arrivals at λ req/s.
    Models real-world traffic that can overload system.
    """
    def __init__(self, rate: float, burst_multiplier: float = 1.0):
        self.rate = rate  # Baseline arrival rate (requests per second)
        self.burst_multiplier = burst_multiplier
        self.current_rate = rate

    def set_burst_mode(self, enabled: bool):
        """Enable/disable burst mode."""
        if enabled:
            self.current_rate = self.rate * self.burst_multiplier
        else:
            self.current_rate = self.rate

    def get_next_arrival_delay(self) -> float:
        """Get time until next arrival (exponential distribution)."""
        return random.expovariate(self.current_rate)

    async def generate_arrivals(
        self,
        num_requests: int,
        request_factory,
        executor_func
    ) -> List[LoadRequest]:
        """Generate Poisson arrivals and execute requests."""
        requests = []
        tasks = []

        for i in range(num_requests):
            # Wait for next arrival
            delay = self.get_next_arrival_delay()
            await asyncio.sleep(delay)

            # Create and submit request
            request = request_factory(i)
            request.submit_time = time.time()
            request.start_time = time.time()  # Start immediately in open loop

            # Execute without blocking
            task = asyncio.create_task(self._execute_request(request, executor_func))
            tasks.append(task)
            requests.append(request)

        # Wait for all requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        return requests

    async def _execute_request(self, request: LoadRequest, executor_func):
        """Execute a single request."""
        try:
            await executor_func(request)
            request.end_time = time.time()
            request.success = True
        except Exception as e:
            request.end_time = time.time()
            request.success = False
            request.error = str(e)


# ----------------------------- Task Executors --------------------------------

class TaskExecutor:
    """Execute tasks under load."""
    def __init__(self, provider, corpus_docs=None, db_path=None, retriever=None):
        self.provider = provider
        self.corpus_docs = corpus_docs
        self.db_path = db_path

        # Initialize retriever if corpus available
        self.retriever = retriever
        if self.retriever is None and corpus_docs:
            self.retriever = DeterministicRetriever(corpus_docs)

        # Store reference outputs for drift calculation
        self.reference_outputs: Dict[str, str] = {}

    async def execute_rag(self, request: LoadRequest) -> LoadRequest:
        """Execute RAG task."""
        if not self.retriever:
            raise ValueError("No corpus available for RAG task")

        # Sample question
        question = "What were the key risk factors for banks in 2024?"

        # Retrieve context
        snippets = self.retriever.retrieve(question, k=5)
        available_sources = list(set(s[0].split("#")[0] for s in snippets))

        # Format prompt
        messages = format_rag_prompt(question, snippets)
        prompt_text = json.dumps(messages)
        request.prompt = prompt_text

        # Call LLM
        output = await self.provider.acomplete(
            request.model,
            messages,
            temperature=request.temperature,
            seed=42
        )

        request.output = output
        request.output_hash = hashlib.sha256(output.encode()).hexdigest()

        # Store first output as reference
        key = f"{request.task}_{request.model}_{request.temperature}"
        if key not in self.reference_outputs:
            self.reference_outputs[key] = output

        request.reference_output = self.reference_outputs[key]

        return request

    async def execute_sql(self, request: LoadRequest) -> LoadRequest:
        """Execute SQL generation task."""
        question = "What is the total amount of transactions in the West region?"

        messages = format_sql_prompt(question)
        prompt_text = json.dumps(messages)
        request.prompt = prompt_text

        output = await self.provider.acomplete(
            request.model,
            messages,
            temperature=request.temperature,
            seed=42
        )

        request.output = output
        request.output_hash = hashlib.sha256(output.encode()).hexdigest()

        # Store reference
        key = f"{request.task}_{request.model}_{request.temperature}"
        if key not in self.reference_outputs:
            self.reference_outputs[key] = output

        request.reference_output = self.reference_outputs[key]

        return request

    async def execute_summary(self, request: LoadRequest) -> LoadRequest:
        """Execute summarization task."""
        profile_text = """
Client: Acme Corp
Account Balance: $125,000.00
Last Transaction: 2024-11-01
Risk Profile: Moderate
Investment Goals: Long-term growth
"""

        messages = format_summary_prompt(profile_text)
        prompt_text = json.dumps(messages)
        request.prompt = prompt_text

        output = await self.provider.acomplete(
            request.model,
            messages,
            temperature=request.temperature,
            seed=42
        )

        request.output = output
        request.output_hash = hashlib.sha256(output.encode()).hexdigest()

        # Store reference
        key = f"{request.task}_{request.model}_{request.temperature}"
        if key not in self.reference_outputs:
            self.reference_outputs[key] = output

        request.reference_output = self.reference_outputs[key]

        return request


# ----------------------------- Analysis --------------------------------------

def calculate_metrics(requests: List[LoadRequest]) -> LoadMetrics:
    """Calculate aggregate metrics from requests."""
    if not requests:
        return None

    # Get common attributes
    first = requests[0]

    # Determine provider region/build
    provider_region = os.environ.get("WATSONX_URL", "local") if first.provider == "watsonx" else "local"
    model_build_version = "unknown"  # Can be enhanced with provider API calls
    test_timestamp = datetime.now().isoformat()

    metrics = LoadMetrics(
        task=first.task,
        provider=first.provider,
        model=first.model,
        temperature=first.temperature,
        load_mode=first.load_mode,
        load_level=first.load_level,
        requested_model=first.requested_model,
        resolved_model=first.resolved_model,
        provider_supported=first.provider_supported,
        provider_region=provider_region,
        model_build_version=model_build_version,
        test_timestamp=test_timestamp
    )

    # Count successes/failures/skips
    successful = [r for r in requests if r.success]
    skipped = [r for r in requests if r.error and "unsupported_model" in r.error]

    metrics.total_requests = len(requests)
    metrics.successful_requests = len(successful)
    metrics.skipped_requests = len(skipped)
    metrics.failed_requests = len(requests) - len(successful) - len(skipped)

    if not successful:
        return metrics

    # Calculate latency metrics
    queue_times = [r.queue_time_ms for r in successful]
    service_times = [r.service_time_ms for r in successful]
    total_latencies = [r.total_latency_ms for r in successful]

    metrics.mean_queue_time_ms = np.mean(queue_times)
    metrics.mean_service_time_ms = np.mean(service_times)
    metrics.mean_total_latency_ms = np.mean(total_latencies)

    metrics.p50_latency_ms = np.percentile(total_latencies, 50)
    metrics.p95_latency_ms = np.percentile(total_latencies, 95)
    metrics.p99_latency_ms = np.percentile(total_latencies, 99)

    # Calculate throughput
    if successful:
        start_time = min(r.submit_time for r in successful)
        end_time = max(r.end_time for r in successful if r.end_time)
        metrics.test_duration_s = end_time - start_time
        if metrics.test_duration_s > 0:
            metrics.tps = len(successful) / metrics.test_duration_s

    # Calculate drift metrics
    outputs_with_reference = [
        r for r in successful
        if r.output and r.reference_output
    ]

    if outputs_with_reference:
        # Identity rate (exact match)
        identical = sum(
            1 for r in outputs_with_reference
            if r.output == r.reference_output
        )
        metrics.identity_rate = identical / len(outputs_with_reference)

        # Mean edit distance
        edit_distances = []
        for r in outputs_with_reference:
            if r.output != r.reference_output:
                dist = Levenshtein.normalized_distance(
                    r.output, r.reference_output
                )
                edit_distances.append(dist)

        if edit_distances:
            metrics.mean_edit_distance = np.mean(edit_distances)

    # SLO violations
    # Latency SLO: p99 ≤ 5 seconds under 4x load
    if first.load_level >= 4 and metrics.p99_latency_ms > 5000:
        metrics.latency_slo_violations = 1

    # Determinism SLO: drift ≤1% at T=0.0
    if first.temperature == 0.0 and metrics.identity_rate < 0.99:
        metrics.determinism_slo_violations = 1

    return metrics


def load_model_aliases() -> Dict[str, str]:
    """Load model aliases from config file."""
    aliases_file = Path(__file__).parent.parent / "configs" / "model_aliases.yaml"
    if not aliases_file.exists():
        return {}

    try:
        with open(aliases_file, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('aliases', {})
    except Exception as e:
        print(f"Warning: Could not load model aliases: {e}")
        return {}


def resolve_model_name(model: str, provider, aliases: Dict[str, str]) -> Tuple[str, str]:
    """
    Resolve model name using aliases.

    Returns:
        Tuple of (requested_model, resolved_model)
    """
    requested = model
    resolved = aliases.get(model, model)

    # Also check provider's built-in aliases
    if hasattr(provider, '_normalize_model_id'):
        resolved = provider._normalize_model_id(resolved)

    return requested, resolved


def get_supported_models(provider) -> List[str]:
    """Get list of supported models from provider."""
    if hasattr(provider, 'list_models'):
        try:
            return provider.list_models()
        except Exception as e:
            print(f"Warning: Could not list models from provider: {e}")
            return []
    return []


def save_results(all_metrics: List[LoadMetrics], output_file: Path):
    """Save results to CSV."""
    if not all_metrics:
        print("No metrics to save")
        return

    fieldnames = [
        "task", "provider", "model", "temperature", "load_mode", "load_level",
        "requested_model", "resolved_model", "provider_supported",
        "provider_region", "model_build_version", "test_timestamp",
        "total_requests", "successful_requests", "failed_requests", "skipped_requests",
        "mean_queue_time_ms", "mean_service_time_ms", "mean_total_latency_ms",
        "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
        "tps", "test_duration_s",
        "identity_rate", "mean_edit_distance", "faithfulness_score",
        "latency_slo_violations", "determinism_slo_violations"
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(asdict(m))

    print(f"\nResults saved to: {output_file}")


# ----------------------------- Main ------------------------------------------

async def run_closed_loop_test(
    executor: TaskExecutor,
    task: str,
    model: str,
    temperature: float,
    concurrency: int,
    num_requests: int = 32,
    requested_model: Optional[str] = None,
    provider_supported: bool = True
) -> List[LoadRequest]:
    """Run closed-loop load test."""
    print(f"\n[Closed Loop] Task={task}, Model={model}, T={temperature}, N={concurrency}")

    load_model = ClosedLoopLoadModel(concurrency)

    # Create requests
    requests = []
    for i in range(num_requests):
        request = LoadRequest(
            request_id=f"req_{i}",
            task=task,
            provider=executor.provider.name,
            model=model,
            temperature=temperature,
            load_mode="closed",
            load_level=concurrency,
            requested_model=requested_model or model,
            resolved_model=model,
            provider_supported=provider_supported,
            submit_time=time.time()
        )
        requests.append(request)

    # Select executor function
    if task == "rag":
        exec_func = executor.execute_rag
    elif task == "sql":
        exec_func = executor.execute_sql
    elif task == "summary":
        exec_func = executor.execute_summary
    else:
        raise ValueError(f"Unknown task: {task}")

    # Execute all requests with concurrency control
    tasks = [
        load_model.submit_request(req, exec_func)
        for req in requests
    ]

    await asyncio.gather(*tasks, return_exceptions=True)

    return requests


async def run_open_loop_test(
    executor: TaskExecutor,
    task: str,
    model: str,
    temperature: float,
    rate: float,
    burst_multiplier: float = 1.0,
    num_requests: int = 32,
    requested_model: Optional[str] = None,
    provider_supported: bool = True
) -> List[LoadRequest]:
    """Run open-loop load test."""
    print(f"\n[Open Loop] Task={task}, Model={model}, T={temperature}, λ={rate}, Burst={burst_multiplier}x")

    load_model = OpenLoopLoadModel(rate, burst_multiplier)

    # Select executor function
    if task == "rag":
        exec_func = executor.execute_rag
    elif task == "sql":
        exec_func = executor.execute_sql
    elif task == "summary":
        exec_func = executor.execute_summary
    else:
        raise ValueError(f"Unknown task: {task}")

    # Request factory
    def request_factory(i):
        return LoadRequest(
            request_id=f"req_{i}",
            task=task,
            provider=executor.provider.name,
            model=model,
            temperature=temperature,
            load_mode="open",
            load_level=int(rate * burst_multiplier),
            requested_model=requested_model or model,
            resolved_model=model,
            provider_supported=provider_supported,
            submit_time=0.0  # Will be set when actually submitted
        )

    requests = await load_model.generate_arrivals(
        num_requests, request_factory, exec_func
    )

    return requests


async def main():
    parser = argparse.ArgumentParser(
        description="Production load testing for LLM drift framework"
    )
    parser.add_argument(
        "--mode",
        choices=["closed", "open"],
        default="closed",
        help="Load model: closed-loop or open-loop"
    )
    parser.add_argument(
        "--concurrency",
        type=str,
        default="1,4,8,16",
        help="Concurrency levels (comma-separated)"
    )
    parser.add_argument(
        "--rate",
        type=str,
        default="0.5,1,2,4",
        help="Arrival rates for open-loop (comma-separated, req/s)"
    )
    parser.add_argument(
        "--burst",
        type=float,
        default=1.0,
        help="Burst multiplier for open-loop"
    )
    parser.add_argument(
        "--pattern",
        choices=list(FINANCIAL_PATTERNS.keys()),
        help="Financial load pattern"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="qwen2.5:7b-instruct",
        help="Models to test (comma-separated)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "watsonx"],
        default="ollama",
        help="Provider to use (ollama or watsonx)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="rag,sql,summary",
        help="Tasks to test (comma-separated)"
    )
    parser.add_argument(
        "--temps",
        type=str,
        default="0.0",
        help="Temperatures to test (comma-separated)"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=32,
        help="Number of requests per test"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file (default: results/v2_load_analysis.csv)"
    )
    parser.add_argument(
        "--print-supported",
        action="store_true",
        help="Print supported models for the provider and exit"
    )
    parser.add_argument(
        "--strict-models",
        action="store_true",
        default=True,
        help="Fail fast on unsupported models (default: true)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve models and print plan without executing"
    )

    args = parser.parse_args()

    # Parse arguments
    models = [m.strip() for m in args.models.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]
    temps = [float(t.strip()) for t in args.temps.split(",")]

    # Load corpus for RAG tasks
    retriever = None
    if "rag" in tasks:
        try:
            retriever = create_retriever_from_files(
                corpus_path="data/sec", chunk_size=200, overlap=50
            )
            print(f"Loaded retriever with {len(retriever.snippets)} snippets")
        except Exception as e:
            print(f"Warning: Could not load corpus: {e}")
            print("RAG tasks will be skipped")
            tasks = [t for t in tasks if t != "rag"]

    # Initialize provider based on argument
    if args.provider == "watsonx":
        print("Using watsonx.ai provider")
        provider = WatsonxProvider()
    else:
        print("Using Ollama provider")
        provider = OllamaProvider()

    # Load model aliases
    print("\nLoading model aliases...")
    aliases = load_model_aliases()
    if aliases:
        print(f"  Loaded {len(aliases)} model aliases")

    # Get supported models from provider
    print(f"\nQuerying supported models from {args.provider}...")
    supported_models = get_supported_models(provider)

    # Handle --print-supported flag
    if args.print_supported:
        print(f"\n{'='*80}")
        print(f"SUPPORTED MODELS FOR {args.provider.upper()}")
        print(f"{'='*80}")
        if supported_models:
            for i, model in enumerate(supported_models, 1):
                print(f"{i:2d}. {model}")
            print(f"\nTotal: {len(supported_models)} models")
        else:
            print("  Model listing not supported for this provider")
            print("  (Provider may support models but doesn't implement list_models())")
        print(f"{'='*80}\n")
        return

    # Determine load levels (needed for dry-run output)
    if args.mode == "closed":
        load_levels = [int(c.strip()) for c in args.concurrency.split(",")]
    else:
        load_levels = [float(r.strip()) for r in args.rate.split(",")]

    # Resolve all requested models
    print("\nResolving requested models...")
    resolved_models = []
    model_resolutions = {}
    unsupported_models = []

    for model in models:
        requested, resolved = resolve_model_name(model, provider, aliases)
        model_resolutions[model] = (requested, resolved)

        # Check if resolved model is supported
        is_supported = True
        if supported_models:  # Only validate if we have a list
            is_supported = resolved in supported_models

        resolved_models.append({
            'requested': requested,
            'resolved': resolved,
            'supported': is_supported
        })

        status = "✓" if is_supported else "✗"
        alias_note = f" (aliased from {requested})" if requested != resolved else ""
        print(f"  {status} {resolved}{alias_note}")

        if not is_supported:
            unsupported_models.append(resolved)

    # Handle --dry-run flag
    if args.dry_run:
        print(f"\n{'='*80}")
        print("DRY RUN - MODEL RESOLUTION PLAN")
        print(f"{'='*80}")
        print(f"Provider: {args.provider}")
        print(f"Tasks: {', '.join(tasks)}")
        print(f"Temperatures: {', '.join(map(str, temps))}")
        print(f"Load levels: {load_levels}")
        print(f"\nModel Resolution:")
        for rm in resolved_models:
            status = "SUPPORTED" if rm['supported'] else "UNSUPPORTED"
            print(f"  {rm['requested']} → {rm['resolved']} [{status}]")

        if unsupported_models:
            print(f"\n⚠️  Warning: {len(unsupported_models)} unsupported model(s) detected")
            print("    These will be SKIPPED during actual execution with --strict-models")

        print(f"{'='*80}\n")
        return

    # Handle unsupported models with --strict-models
    if unsupported_models and args.strict_models:
        print(f"\n{'='*80}")
        print("ERROR: UNSUPPORTED MODELS DETECTED")
        print(f"{'='*80}")
        print(f"The following models are not supported by {args.provider}:")
        for model in unsupported_models:
            print(f"  ✗ {model}")

        print("\nOptions:")
        print(f"  1. Use --print-supported to see available models")
        print(f"  2. Check configs/model_aliases.yaml for alias mappings")
        print(f"  3. Use --strict-models=false to skip unsupported models")
        print(f"{'='*80}\n")
        sys.exit(1)

    executor = TaskExecutor(provider, retriever=retriever)

    # Apply financial pattern if specified
    burst_multiplier = args.burst
    if args.pattern:
        pattern = FINANCIAL_PATTERNS[args.pattern]
        print(f"\nUsing financial pattern: {args.pattern}")
        print(f"Description: {pattern['description']}")

        if "burst_multiplier" in pattern:
            burst_multiplier = pattern["burst_multiplier"]
        elif "spike_multiplier" in pattern:
            burst_multiplier = pattern["spike_multiplier"]

    # Run tests
    all_metrics = []
    total_skipped = 0

    for model_info in resolved_models:
        requested = model_info['requested']
        resolved = model_info['resolved']
        is_supported = model_info['supported']

        # Skip unsupported models if not in strict mode
        if not is_supported:
            print(f"\n⚠️  Skipping unsupported model: {resolved}")
            total_skipped += 1
            continue

        for task in tasks:
            for temp in temps:
                for load_level in load_levels:
                    try:
                        if args.mode == "closed":
                            requests = await run_closed_loop_test(
                                executor, task, resolved, temp,
                                int(load_level), args.num_requests,
                                requested_model=requested,
                                provider_supported=is_supported
                            )
                        else:
                            requests = await run_open_loop_test(
                                executor, task, resolved, temp,
                                load_level, burst_multiplier, args.num_requests,
                                requested_model=requested,
                                provider_supported=is_supported
                            )

                        # Calculate metrics
                        metrics = calculate_metrics(requests)
                        if metrics:
                            all_metrics.append(metrics)

                            # Print summary
                            success_pct = (metrics.successful_requests / metrics.total_requests * 100) if metrics.total_requests > 0 else 0
                            print(f"  ✓ Success: {metrics.successful_requests}/{metrics.total_requests} ({success_pct:.1f}%)")

                            if metrics.successful_requests > 0:
                                print(f"  ⏱  P50/P95/P99 latency: {metrics.p50_latency_ms:.0f}/{metrics.p95_latency_ms:.0f}/{metrics.p99_latency_ms:.0f} ms")
                                print(f"  🔄 Throughput: {metrics.tps:.2f} TPS")
                                print(f"  ✓ Identity rate: {metrics.identity_rate:.1%}")

                                if metrics.determinism_slo_violations:
                                    print(f"  ⚠️  DETERMINISM SLO VIOLATION")
                                if metrics.latency_slo_violations:
                                    print(f"  ⚠️  LATENCY SLO VIOLATION")
                            else:
                                print(f"  ✗ All requests failed")

                            if metrics.skipped_requests > 0:
                                print(f"  ⚠️  Skipped: {metrics.skipped_requests} requests (unsupported model)")

                    except Exception as e:
                        print(f"  ✗ Error: {e}")
                        continue

    # Save results
    output_file = Path(args.output) if args.output else RESULTS_DIR / "v2_load_analysis.csv"
    save_results(all_metrics, output_file)

    # Print summary
    print("\n" + "="*80)
    print("LOAD TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {len(all_metrics)}")
    print(f"Models requested: {len(models)}")
    print(f"Models tested: {len(resolved_models) - total_skipped}")
    if total_skipped > 0:
        print(f"Models skipped: {total_skipped} (unsupported)")
    print(f"Tasks tested: {len(tasks)}")
    print(f"Load levels tested: {len(load_levels)}")

    # SLO compliance summary
    determinism_violations = sum(m.determinism_slo_violations for m in all_metrics)
    latency_violations = sum(m.latency_slo_violations for m in all_metrics)

    print(f"\nSLO Compliance:")
    print(f"  Determinism violations: {determinism_violations}/{len(all_metrics)}")
    print(f"  Latency violations: {latency_violations}/{len(all_metrics)}")

    if total_skipped > 0 and args.strict_models:
        print(f"\n⚠️  {total_skipped} model(s) skipped due to lack of provider support")
        print(f"    Use --print-supported to see available models")
    elif determinism_violations == 0 and latency_violations == 0:
        print("\n✅ All SLO requirements met!")
    else:
        print("\n⚠️  Some SLO violations detected - review results")

    # Exit with non-zero if any models were skipped in strict mode
    if total_skipped > 0 and args.strict_models:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
