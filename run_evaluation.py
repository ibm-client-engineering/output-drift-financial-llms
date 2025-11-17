#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Output Drift Evaluation Framework
ACM ICAIF 2025: Financial AI Compliance

Experiment runner for evaluating deterministic behavior of LLMs across:
  1) RAG Q&A over SEC filings with citation validation
  2) Policy-bounded JSON summarization with schema constraints
  3) Text-to-SQL with invariant checking (±5% tolerance)

Metrics:
  - Normalized Levenshtein distance (drift vs. reference run)
  - Citation accuracy (RAG)
  - Schema violation rate (JSON)
  - Decision flip rate (SQL)
  - Latency (seconds)

Outputs:
  - results/summary.csv      (per-run data)
  - results/aggregate.csv    (grouped statistics)
  - traces/*.jsonl           (full audit trails)

Usage:
  # Basic evaluation with Ollama
  python run_evaluation.py

  # Full experimental matrix
  python run_evaluation.py \\
    --models qwen2.5:7b-instruct,granite-3-8b,llama-3.3-70b \\
    --temperatures 0.0,0.2 \\
    --concurrency 1,4,16 \\
    --output traces/

For more details, see: https://github.com/ibm-client-engineering/output-drift-financial-llms
"""
import os
import re
import json
import time
import math
import csv
import argparse
import asyncio
import random
import sqlite3
import glob
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import httpx
import pandas as pd
from rapidfuzz.distance import Levenshtein
from jsonschema import validate

from harness.deterministic_retriever import DeterministicRetriever, create_retriever_from_files
from harness.task_definitions import (
    format_rag_prompt,
    extract_citations,
    validate_citations,
    format_summary_prompt,
    format_sql_prompt,
    validate_summary_json,
    validate_sql_query,
    SUMMARY_SCHEMA
)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, environment variables must be set manually

# ----------------------------- Config ----------------------------------------
DEFAULT_MODELS = ["qwen2.5:7b-instruct"]
DEFAULT_PROVIDERS = ["ollama"]  # Available: "ollama", "watsonx", "mock"
REPEATS = 16  # n=16 per condition as in paper
MAX_TOKENS = 512
CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")

BASE = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRACES_DIR = BASE / "traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- Providers -------------------------------------
class LLMProvider:
    """Base class for LLM providers."""
    name: str

    async def acomplete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        temperature=0.0,
        top_p=1.0,
        seed: Optional[int] = None,
        max_tokens=MAX_TOKENS,
        stream=False,
        extra=None
    ) -> str:
        raise NotImplementedError

    def supports_listing(self) -> bool:
        return False

    def list_models(self) -> List[str]:
        return []


class OllamaProvider(LLMProvider):
    """Ollama local model provider."""

    def __init__(self, host: str = "http://127.0.0.1:11434"):
        self.host = host
        self.name = "ollama"

    async def acomplete(
        self,
        model,
        messages,
        *,
        temperature=0.0,
        top_p=1.0,
        seed=None,
        max_tokens=MAX_TOKENS,
        stream=False,
        extra=None
    ):
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
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            return data.get("response", "")


class MockProvider(LLMProvider):
    """Deterministic mock provider for testing without external dependencies."""

    def __init__(self):
        self.name = "mock"
        random.seed(123)

    async def acomplete(
        self,
        model,
        messages,
        *,
        temperature=0.0,
        top_p=1.0,
        seed=None,
        max_tokens=MAX_TOKENS,
        stream=False,
        extra=None
    ):
        # Inspect messages to determine task type
        text = "\n".join(m["content"] for m in messages if m["role"] == "user")

        if "Documents:" in text:
            # RAG task: return deterministic citation
            m = re.search(r"\[(.+?)\]\s", text)
            source = m.group(1) if m else "sample_doc"
            return f"Based on the filings, key details are provided. [{source}]"

        if "STRICT JSON" in messages[0]["content"]:
            # JSON summarization task
            return json.dumps({
                "client_name": "Sample Client",
                "summary": "Brief two-sentence portfolio update provided here.",
                "compliance_disclaimer": "This is not investment advice."
            })

        if "You write SQLite SQL ONLY" in messages[0]["content"]:
            # SQL task
            if 'region = "NA"' in text or "region = NA" in text:
                return 'SELECT SUM(amount) FROM transactions WHERE region = "NA" AND date BETWEEN "2025-01-01" AND "2025-09-01";'
            return "SELECT SUM(amount) FROM transactions;"

        return "Mock response generated."


class WatsonxProviderAdapter(LLMProvider):
    """Adapter to make WatsonxProvider compatible with async interface."""

    def __init__(self):
        self.name = "watsonx"
        try:
            from providers.watsonx import WatsonxProvider
            self._provider = WatsonxProvider()
            self.enabled = True
        except (ImportError, ValueError) as e:
            print(f"[warn] Watsonx provider initialization failed: {e}")
            print("[info] Set WATSONX_API_KEY, WATSONX_URL, WATSONX_PROJECT_ID to enable")
            self.enabled = False
            self._provider = None

    def supports_listing(self) -> bool:
        return self._provider and self._provider.supports_listing()

    def list_models(self) -> List[str]:
        if not self._provider:
            return []
        return self._provider.list_models()

    async def acomplete(
        self,
        model,
        messages,
        *,
        temperature=0.0,
        top_p=1.0,
        seed=None,
        max_tokens=MAX_TOKENS,
        stream=False,
        extra=None
    ):
        if not self.enabled:
            raise RuntimeError("Watsonx provider not enabled. Check environment variables.")

        prompt = _concat_messages(messages)
        result = self._provider.generate(
            model=model,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_new_tokens=max_tokens,
            stream=stream,
            extra=extra
        )
        return result["text"]


def _concat_messages(messages: List[Dict[str, str]]) -> str:
    """Concatenate messages for logging."""
    return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])


# ----------------------------- Utilities -------------------------------------
def norm_lev(a: str, b: str) -> float:
    """Normalized Levenshtein distance."""
    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))


def now_ms() -> int:
    """Current timestamp in milliseconds."""
    return int(time.time() * 1000)


def write_trace(filename: str, recs: List[Dict[str, Any]]):
    """Write audit trail records to JSONL."""
    with open(TRACES_DIR / filename, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------- SQLite DB -------------------------------------
def init_sqlite(db_path: pathlib.Path) -> Dict[str, Any]:
    """Initialize toy finance database for SQL task."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS transactions")
    cur.execute("""
        CREATE TABLE transactions(
            id INTEGER PRIMARY KEY,
            date TEXT,
            region TEXT,
            amount REAL,
            category TEXT
        )
    """)

    # Generate deterministic test data
    random.seed(1234)
    regions = ["NA", "EMEA", "APAC"]
    total = 0.0
    for i in range(1, 501):
        region = regions[i % 3]
        amt = round(100 + (i % 17) * 3.14, 2)
        total += amt
        cur.execute(
            "INSERT INTO transactions(id,date,region,amount,category) VALUES(?,?,?,?,?)",
            (i, f"2025-0{(i%9)+1}-{(i%27)+1:02d}", region, amt, "trading")
        )
    conn.commit()
    return {"conn": conn, "total_amount": round(total, 2)}


# ----------------------------- Task Runners ----------------------------------
@dataclass
class RunConfig:
    """Configuration for experiment run."""
    provider: str
    model: str
    temperature: float
    top_p: float
    concurrency: int
    seed: Optional[int] = None
    stream: bool = False
    repeats: int = REPEATS


async def run_rag(
    cfg: RunConfig,
    prov: LLMProvider,
    retr: DeterministicRetriever,
    qid: str,
    question: str
) -> Dict[str, Any]:
    """Run RAG task with citation validation."""
    snips = retr.retrieve(question, k=5)
    msgs = format_rag_prompt(question, snips)

    start = time.time()
    out = await prov.acomplete(
        cfg.model,
        msgs,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        seed=cfg.seed,
        max_tokens=MAX_TOKENS,
        stream=cfg.stream
    )
    latency = time.time() - start

    # Extract and validate citations
    cits = extract_citations(out)
    available_sources = [snip[0].split('#')[0] for snip in snips]
    validation = validate_citations(cits, available_sources)

    return {
        "output": out,
        "citations": sorted(cits),
        "latency": latency,
        "snippets": [sid for sid, _, _ in snips],
        "citation_accuracy": validation["citation_accuracy"],
        "valid_citations": validation["valid_citations"],
        "invalid_citations": validation["invalid_citations"]
    }


async def run_summary(
    cfg: RunConfig,
    prov: LLMProvider,
    pid: str,
    profile_text: str
) -> Dict[str, Any]:
    """Run JSON summarization task with schema validation."""
    msgs = format_summary_prompt(profile_text)

    start = time.time()
    out = await prov.acomplete(
        cfg.model,
        msgs,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        seed=cfg.seed,
        max_tokens=MAX_TOKENS,
        stream=cfg.stream
    )
    latency = time.time() - start

    # Validate JSON schema
    validation = validate_summary_json(out)

    # Retry once if validation fails
    if not validation["valid"]:
        retry_msgs = [
            {"role": "system", "content": msgs[0]["content"] + " Your prior attempt was invalid. Fix and return ONLY valid JSON."},
            msgs[1]
        ]
        out = await prov.acomplete(
            cfg.model,
            retry_msgs,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=cfg.seed,
            max_tokens=MAX_TOKENS,
            stream=cfg.stream
        )
        validation = validate_summary_json(out)

    return {
        "output": out,
        "schema_violation": not validation["valid"],
        "latency": latency,
        "json": validation.get("parsed", {})
    }


async def run_sql(
    cfg: RunConfig,
    prov: LLMProvider,
    dbi: Dict[str, Any],
    qid: str,
    nlq: str
) -> Dict[str, Any]:
    """Run text-to-SQL task with ±5% tolerance validation."""
    msgs = format_sql_prompt(nlq)

    start = time.time()
    sql = await prov.acomplete(
        cfg.model,
        msgs,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        seed=cfg.seed,
        max_tokens=256,
        stream=cfg.stream
    )
    latency = time.time() - start

    sql_clean = sql.strip().strip("`").strip()

    # Validate SQL with ±5% tolerance (GAAP materiality threshold)
    try:
        validation = validate_sql_query(
            sql_clean,
            dbi["conn"],
            expected_total=dbi["total_amount"],
            tolerance_pct=5.0
        )
        decision_ok = validation["decision_ok"]
    except Exception as e:
        decision_ok = False

    return {
        "output": sql_clean,
        "decision_ok": decision_ok,
        "latency": latency
    }


# ----------------------------- Orchestrator -----------------------------------
async def run_condition(
    task_name: str,
    prompts: List[Tuple[str, str]],
    cfg: RunConfig,
    prov: LLMProvider,
    retr: Optional[DeterministicRetriever],
    dbi: Optional[Dict[str, Any]],
    results: List[Dict[str, Any]],
    traces: List[Dict[str, Any]]
):
    """Run a single experimental condition."""
    sem = asyncio.Semaphore(cfg.concurrency)

    async def one_run(pid: str, prompt_text: str, run_idx: int):
        async with sem:
            t0 = now_ms()
            if task_name == "rag":
                rec = await run_rag(cfg, prov, retr, pid, prompt_text)
            elif task_name == "summary":
                rec = await run_summary(cfg, prov, pid, prompt_text)
            else:  # sql
                rec = await run_sql(cfg, prov, dbi, pid, prompt_text)
            t1 = now_ms()

            trace = {
                "ts": t0,
                "ts_end": t1,
                "task": task_name,
                "provider": prov.name,
                "model": cfg.model,
                "temp": cfg.temperature,
                "conc": cfg.concurrency,
                "prompt_id": pid,
                "prompt": prompt_text,
                **rec
            }
            traces.append(trace)
            return trace

    # Run all prompts with repeats
    for pid, prompt in prompts:
        runs: List[Dict[str, Any]] = []
        tasks = [asyncio.create_task(one_run(pid, prompt, i)) for i in range(cfg.repeats)]

        for r in await asyncio.gather(*tasks):
            runs.append(r)

        # Calculate drift against first run (reference)
        ref = runs[0]["output"]
        ref_cits = set(runs[0].get("citations", []))
        ref_dec = runs[0].get("decision_ok", None)

        for r in runs:
            drift = norm_lev(ref, r["output"])

            # Factual drift (citation changes)
            fact_drift = None
            if "citations" in r:
                fact_drift = 0 if set(r["citations"]) == ref_cits else 1

            # Decision flip (SQL)
            flip = None
            if "decision_ok" in r and ref_dec is not None:
                flip = int(r["decision_ok"] != ref_dec)

            schema_v = int(r.get("schema_violation", False))

            results.append({
                "task": task_name,
                "prompt_id": pid,
                "provider": prov.name,
                "model": cfg.model,
                "temp": cfg.temperature,
                "top_p": cfg.top_p,
                "seed": cfg.seed,
                "stream": cfg.stream,
                "concurrency": cfg.concurrency,
                "run_id": f"{pid}_{cfg.provider}_{cfg.model.replace('/', '_')}_t{cfg.temperature}_c{cfg.concurrency}_{runs.index(r)}",
                "drift_norm_lev": drift,
                "factual_drift": fact_drift,
                "schema_violation": schema_v,
                "decision_flip": flip,
                "latency_s": r["latency"]
            })


# ----------------------------- Prompts ---------------------------------------
def build_prompts() -> Dict[str, List[Tuple[str, str]]]:
    """Build test prompts for all tasks."""
    rag_questions = [
        ("q1", "What were JPMorgan's net credit losses in 2023? Include a citation."),
        ("q2", "List Citigroup's primary risk factors mentioned in the annual report. Include a citation.")
    ]

    profiles = [
        ("p1", "Client: Jane Doe, institutional investor. Needs a concise portfolio update. Summarize neutrally in 2 sentences."),
        ("p2", "Client: Acme Holdings LLC. Provide brief 2-sentence update. Avoid PII, include exact disclaimer.")
    ]

    sql_questions = [
        ("s1", 'Compute total "amount" across all transactions.'),
        ("s2", 'Sum "amount" for region = "NA" between 2025-01-01 and 2025-09-01.')
    ]

    return {"rag": rag_questions, "summary": profiles, "sql": sql_questions}


# ----------------------------- Main ------------------------------------------
async def main():
    parser = argparse.ArgumentParser(
        description="LLM Output Drift Evaluation Framework (ACM ICAIF 2025)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with mock provider
  python run_evaluation.py --providers mock --models mock-model

  # Evaluate single model
  python run_evaluation.py --models qwen2.5:7b-instruct

  # Full experimental matrix (paper reproduction)
  python run_evaluation.py \\
    --models qwen2.5:7b-instruct,granite-3-8b,llama-3.3-70b \\
    --temperatures 0.0,0.2 \\
    --concurrency 1,4,16 \\
    --repeats 16

For more information: https://github.com/ibm-client-engineering/output-drift-financial-llms
        """
    )

    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model IDs (e.g., qwen2.5:7b-instruct,llama3.1:8b-instruct)"
    )
    parser.add_argument(
        "--providers",
        type=str,
        default=",".join(DEFAULT_PROVIDERS),
        help="Providers to use: ollama,watsonx,mock (watsonx requires WATSONX_API_KEY, WATSONX_URL, WATSONX_PROJECT_ID)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=REPEATS,
        help=f"Number of repetitions per condition (default: {REPEATS})"
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        default="0.0",
        help="Comma-separated temperatures (default: 0.0)"
    )
    parser.add_argument(
        "--top_p",
        type=str,
        default="1.0",
        help="Comma-separated top_p values (default: 1.0)"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated random seeds (default: 42)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Enable streaming mode"
    )
    parser.add_argument(
        "--concurrency",
        type=str,
        default="1",
        help="Comma-separated concurrency levels (default: 1)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="rag,summary,sql",
        help="Comma-separated tasks: rag,summary,sql (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for traces (default: traces/)"
    )

    args = parser.parse_args()

    # Parse arguments
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    temps = [float(t) for t in args.temperatures.split(",")]
    top_ps = [float(p) for p in args.top_p.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [None]
    concs = [int(c) for c in args.concurrency.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    repeats = args.repeats
    stream = args.stream

    # Override traces directory if specified
    global TRACES_DIR
    if args.output:
        TRACES_DIR = pathlib.Path(args.output)
        TRACES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LLM Output Drift Evaluation Framework")
    print("ACM ICAIF 2025: Financial AI Compliance")
    print("=" * 70)
    print(f"Models: {', '.join(models)}")
    print(f"Providers: {', '.join(providers)}")
    print(f"Temperatures: {temps}")
    print(f"Concurrency: {concs}")
    print(f"Repeats: {repeats}")
    print(f"Tasks: {', '.join(tasks)}")
    print("=" * 70)

    # Initialize providers
    prov_objs: List[LLMProvider] = []
    if "ollama" in providers:
        prov_objs.append(OllamaProvider())
        print("[✓] Ollama provider initialized")

    if "watsonx" in providers:
        wx = WatsonxProviderAdapter()
        if wx.enabled:
            prov_objs.append(wx)
            print("[✓] Watsonx provider initialized")
        else:
            print("[!] Watsonx provider skipped (check environment variables)")

    if "mock" in providers:
        prov_objs.append(MockProvider())
        print("[✓] Mock provider initialized")

    if not prov_objs:
        print("[error] No providers available. Please check your configuration.")
        return

    # Load corpus for RAG task
    retr = None
    if "rag" in tasks:
        try:
            # Try to load SEC filings if available
            sec_path = DATA_DIR / "sec"
            if sec_path.exists():
                sec_files = list(sec_path.glob("*_2024_10k.txt"))
                if sec_files:
                    retr = create_retriever_from_files(str(sec_path))
                    print(f"[✓] Loaded {len(sec_files)} SEC 10-K filings for RAG task")
                else:
                    print("[warn] No SEC filings found in data/sec/. RAG task will be skipped.")
                    tasks = [t for t in tasks if t != "rag"]
            else:
                print("[warn] data/sec/ directory not found. RAG task will be skipped.")
                print("      Run: python scripts/fetch_sec_texts.py (if available)")
                tasks = [t for t in tasks if t != "rag"]
        except Exception as e:
            print(f"[warn] Failed to load corpus: {e}")
            print("      RAG task will be skipped.")
            tasks = [t for t in tasks if t != "rag"]

    # Initialize SQLite database for SQL task
    dbi = None
    if "sql" in tasks:
        db_path = BASE / "toy_finance.sqlite"
        # Check if database exists, if not try to generate it
        if not db_path.exists():
            try:
                print("[info] Toy finance database not found. Attempting to generate...")
                import subprocess
                gen_script = DATA_DIR / "generate_toy_finance.py"
                if gen_script.exists():
                    subprocess.run(["python", str(gen_script)], check=True)
                    print("[✓] Generated toy finance database")
                else:
                    print("[warn] generate_toy_finance.py not found. Creating minimal database...")
                    dbi = init_sqlite(db_path)
                    print("[✓] Created minimal toy finance database")
            except Exception as e:
                print(f"[warn] Failed to generate database: {e}")
                dbi = init_sqlite(db_path)
                print("[✓] Created minimal toy finance database")
        else:
            conn = sqlite3.connect(db_path)
            # Calculate total amount for validation
            total = pd.read_sql_query("SELECT SUM(amount) as total FROM transactions", conn).iloc[0, 0]
            dbi = {"conn": conn, "total_amount": float(total)}
            print(f"[✓] Loaded toy finance database (total amount: ${dbi['total_amount']:.2f})")

    # Build prompts
    prompts = build_prompts()

    # Run experiments
    results_rows: List[Dict[str, Any]] = []
    for prov in prov_objs:
        for model in models:
            for temp in temps:
                for top_p in top_ps:
                    for seed in seeds:
                        for conc in concs:
                            cfg = RunConfig(
                                provider=prov.name,
                                model=model,
                                temperature=temp,
                                top_p=top_p,
                                concurrency=conc,
                                seed=seed,
                                stream=stream,
                                repeats=repeats
                            )

                            print(f"\n▶ Running {prov.name}:{model} T={temp} seed={seed} conc={conc} repeats={repeats}")
                            traces: List[Dict[str, Any]] = []

                            # Run selected tasks
                            if "rag" in tasks and retr is not None:
                                print("  → RAG task...")
                                await run_condition("rag", prompts["rag"], cfg, prov, retr, None, results_rows, traces)

                            if "summary" in tasks:
                                print("  → Summary task...")
                                await run_condition("summary", prompts["summary"], cfg, prov, None, None, results_rows, traces)

                            if "sql" in tasks and dbi is not None:
                                print("  → SQL task...")
                                await run_condition("sql", prompts["sql"], cfg, prov, None, dbi, results_rows, traces)

                            # Write trace file
                            trace_filename = f"trace_{prov.name}_{model.replace('/','_')}_t{temp}_tp{top_p}_s{seed}_str{stream}_c{conc}.jsonl"
                            write_trace(trace_filename, traces)
                            print(f"  ✓ Wrote {len(traces)} trace records")

    if not results_rows:
        print("\n[warn] No results generated. Check your configuration and data files.")
        return

    # Save summary CSV
    print(f"\n{'='*70}")
    print("Saving results...")
    summary_path = RESULTS_DIR / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        w.writeheader()
        w.writerows(results_rows)
    print(f"[✓] Wrote {summary_path} ({len(results_rows)} rows)")

    # Generate aggregate statistics
    df = pd.DataFrame(results_rows)
    groupby_cols = ["task", "provider", "model", "temp", "top_p", "seed", "stream", "concurrency"]

    agg = df.groupby(groupby_cols).agg(
        runs=("drift_norm_lev", "count"),
        identity_rate=("drift_norm_lev", lambda s: 100.0 * (s == 0.0).mean()),
        mean_drift=("drift_norm_lev", "mean"),
        max_drift=("drift_norm_lev", "max"),
        factual_drift_rate=("factual_drift", lambda s: float(s.dropna().mean()) if s.notna().any() else math.nan),
        schema_violation_rate=("schema_violation", "mean"),
        decision_flip_rate=("decision_flip", "mean"),
        mean_latency_s=("latency_s", "mean"),
        p95_latency_s=("latency_s", lambda s: s.quantile(0.95))
    ).reset_index()

    agg_path = RESULTS_DIR / "aggregate.csv"
    agg.to_csv(agg_path, index=False)
    print(f"[✓] Wrote {agg_path}")

    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    for _, row in agg.iterrows():
        print(f"\n{row['model']} @ T={row['temp']} (concurrency={row['concurrency']}):")
        print(f"  Task: {row['task']}")
        print(f"  Identity Rate: {row['identity_rate']:.1f}%")
        print(f"  Mean Drift: {row['mean_drift']:.4f}")
        if not math.isnan(row['factual_drift_rate']):
            print(f"  Factual Drift Rate: {row['factual_drift_rate']:.2%}")
        if row['schema_violation_rate'] > 0:
            print(f"  Schema Violation Rate: {row['schema_violation_rate']:.2%}")
        if not math.isnan(row['decision_flip_rate']):
            print(f"  Decision Flip Rate: {row['decision_flip_rate']:.2%}")
        print(f"  Mean Latency: {row['mean_latency_s']:.3f}s")

    print(f"\n{'='*70}")
    print("Evaluation complete!")
    print(f"Results: {RESULTS_DIR}")
    print(f"Traces: {TRACES_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
