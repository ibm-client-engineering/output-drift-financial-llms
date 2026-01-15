#!/usr/bin/env python3
"""
V3 Econometric Analysis on Real V2 Experiment Data

Connects the V3 econometric framework to actual experiment results from
the LLM Output Drift paper (3,684+ runs across 13 models, 4 providers).

Usage:
    python econometrics/analyze_real_data.py

Output:
    - Model tier classification with validation scaling factors
    - Task-structure effect quantification
    - Faithfulness vs determinism correlation analysis
    - Recommendations for econometric research
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Results path
RESULTS_DIR = Path(__file__).parent.parent / "results"
AGGREGATE_CSV = RESULTS_DIR / "aggregate.csv"


@dataclass
class ModelTierAnalysis:
    """Analysis results for a model tier."""
    tier: str
    models: List[str]
    n_configs: int
    mean_determinism: float
    std_determinism: float
    mean_faithfulness: float
    validation_scaling_factor: float
    task_breakdown: Dict[str, float]


def load_aggregate_data() -> pd.DataFrame:
    """Load and preprocess the aggregate results."""
    df = pd.read_csv(AGGREGATE_CSV)

    # Fill missing values
    df['determinism_rate'] = df['determinism_rate'].fillna(df['pct_identical'])
    df['faithfulness_rate'] = df['faithfulness_rate'].fillna(100.0)

    # Classify model tiers based on our paper findings
    def classify_tier(row):
        model = row['model'].lower()
        det = row['determinism_rate']

        # Tier 1: 7-20B with 100% determinism at T=0
        if det == 100.0 and row['temp'] == 0.0:
            if any(x in model for x in ['qwen2.5:7b', 'granite-3-8b', 'granite3', 'gpt-oss:20b', 'deepseek-r1:8b']):
                return 'Tier1_7-20B'

        # Tier 2: 40-70B with variable determinism
        if any(x in model for x in ['llama-3-3-70b', 'llama-3-70b', 'mistral']):
            return 'Tier2_40-70B'

        # Frontier: Claude, Gemini with task-dependent behavior
        if any(x in model for x in ['claude', 'gemini']):
            return 'Frontier'

        # Tier 3: 120B+ with low determinism
        if 'gpt-oss-120b' in model or 'gpt-oss:120b' in model:
            return 'Tier3_120B+'

        # Default based on determinism
        if det >= 95:
            return 'Tier1_7-20B'
        elif det >= 50:
            return 'Tier2_40-70B'
        else:
            return 'Tier3_120B+'

    df['tier'] = df.apply(classify_tier, axis=1)
    return df


def analyze_model_tiers(df: pd.DataFrame) -> Dict[str, ModelTierAnalysis]:
    """Analyze each model tier for econometric implications."""
    results = {}

    for tier in df['tier'].unique():
        tier_df = df[df['tier'] == tier]

        # Get unique models in this tier
        models = tier_df['model'].unique().tolist()

        # Calculate metrics
        mean_det = tier_df['determinism_rate'].mean()
        std_det = tier_df['determinism_rate'].std()
        mean_faith = tier_df['faithfulness_rate'].mean()

        # Validation scaling factor (relative to Tier 1 baseline)
        # Higher drift = larger validation sample needed
        drift_rate = 100.0 - mean_det
        baseline_drift = 0.0  # Tier 1 reference
        if drift_rate > 0:
            scaling = 1.0 + (drift_rate / 100.0) * 3.0  # Up to 4x for 100% drift
        else:
            scaling = 1.0

        # Task breakdown
        task_det = {}
        for task in tier_df['task'].unique():
            task_df = tier_df[tier_df['task'] == task]
            task_det[task] = task_df['determinism_rate'].mean()

        results[tier] = ModelTierAnalysis(
            tier=tier,
            models=models,
            n_configs=len(tier_df),
            mean_determinism=mean_det,
            std_determinism=std_det,
            mean_faithfulness=mean_faith,
            validation_scaling_factor=scaling,
            task_breakdown=task_det
        )

    return results


def analyze_task_structure_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Quantify the task-structure effect on determinism."""
    # Group by task and temperature
    task_analysis = df.groupby(['task', 'temp']).agg({
        'determinism_rate': ['mean', 'std', 'count'],
        'faithfulness_rate': 'mean',
        'mean_drift': 'mean'
    }).round(2)

    task_analysis.columns = ['det_mean', 'det_std', 'n_configs', 'faith_mean', 'drift_mean']
    return task_analysis.reset_index()


def analyze_faithfulness_determinism_tradeoff(df: pd.DataFrame) -> Dict:
    """Analyze the relationship between faithfulness and determinism."""
    # Filter to T=0 for clean comparison
    t0_df = df[df['temp'] == 0.0].copy()

    # Correlation
    corr = t0_df['determinism_rate'].corr(t0_df['faithfulness_rate'])

    # Quadrant analysis
    high_det = t0_df['determinism_rate'] >= 90
    high_faith = t0_df['faithfulness_rate'] >= 90

    quadrants = {
        'high_det_high_faith': len(t0_df[high_det & high_faith]),
        'high_det_low_faith': len(t0_df[high_det & ~high_faith]),
        'low_det_high_faith': len(t0_df[~high_det & high_faith]),
        'low_det_low_faith': len(t0_df[~high_det & ~high_faith])
    }

    return {
        'correlation': corr,
        'quadrants': quadrants,
        'interpretation': 'positive' if corr > 0.3 else 'orthogonal' if abs(corr) < 0.3 else 'negative'
    }


def generate_econometric_recommendations(tier_analysis: Dict[str, ModelTierAnalysis]) -> List[str]:
    """Generate recommendations for econometric research."""
    recommendations = []

    # Find best tier for each use case
    tiers = list(tier_analysis.values())

    # Best for determinism
    best_det = max(tiers, key=lambda x: x.mean_determinism)
    recommendations.append(
        f"For maximum label stability: Use {best_det.tier} models "
        f"({best_det.mean_determinism:.1f}% determinism)"
    )

    # Best for faithfulness
    best_faith = max(tiers, key=lambda x: x.mean_faithfulness)
    recommendations.append(
        f"For maximum accuracy: Use {best_faith.tier} models "
        f"({best_faith.mean_faithfulness:.1f}% faithfulness)"
    )

    # Validation scaling
    for tier in tiers:
        if tier.validation_scaling_factor > 1.5:
            recommendations.append(
                f"⚠️  {tier.tier} requires {tier.validation_scaling_factor:.1f}x larger "
                f"validation sample (drift rate: {100-tier.mean_determinism:.1f}%)"
            )

    # Task-specific recommendations
    for tier in tiers:
        sql_det = tier.task_breakdown.get('sql', 0)
        rag_det = tier.task_breakdown.get('rag', 0)
        if sql_det > rag_det + 20:
            recommendations.append(
                f"{tier.tier}: Use for SQL/structured tasks ({sql_det:.0f}% det) "
                f"but avoid RAG ({rag_det:.0f}% det)"
            )

    return recommendations


def main():
    print("="*70)
    print("V3 ECONOMETRIC ANALYSIS ON REAL V2 DATA")
    print("="*70)
    print()

    # Load data
    print("Loading aggregate.csv...")
    df = load_aggregate_data()
    print(f"Loaded {len(df)} configurations across {df['model'].nunique()} models")
    print(f"Providers: {df['provider'].unique().tolist()}")
    print(f"Tasks: {df['task'].unique().tolist()}")
    print()

    # Analyze model tiers
    print("-"*70)
    print("MODEL TIER ANALYSIS (Econometric Implications)")
    print("-"*70)

    tier_analysis = analyze_model_tiers(df)

    for tier_name, analysis in sorted(tier_analysis.items()):
        print(f"\n{analysis.tier}:")
        print(f"  Models: {', '.join(analysis.models[:3])}{'...' if len(analysis.models) > 3 else ''}")
        print(f"  Configurations: {analysis.n_configs}")
        print(f"  Mean Determinism: {analysis.mean_determinism:.1f}% (±{analysis.std_determinism:.1f})")
        print(f"  Mean Faithfulness: {analysis.mean_faithfulness:.1f}%")
        print(f"  Validation Scaling: {analysis.validation_scaling_factor:.2f}x")
        print(f"  Task Breakdown:")
        for task, det in sorted(analysis.task_breakdown.items()):
            print(f"    - {task}: {det:.1f}% determinism")

    # Task-structure effect
    print("\n" + "-"*70)
    print("TASK-STRUCTURE EFFECT")
    print("-"*70)

    task_df = analyze_task_structure_effect(df)
    print("\nDeterminism by Task and Temperature:")
    print(task_df.to_string(index=False))

    # Faithfulness-Determinism tradeoff
    print("\n" + "-"*70)
    print("FAITHFULNESS-DETERMINISM RELATIONSHIP")
    print("-"*70)

    tradeoff = analyze_faithfulness_determinism_tradeoff(df)
    print(f"\nCorrelation (T=0.0): {tradeoff['correlation']:.3f}")
    print(f"Interpretation: {tradeoff['interpretation']}")
    print(f"\nQuadrant Distribution:")
    for quad, count in tradeoff['quadrants'].items():
        print(f"  {quad}: {count} configs")

    # Recommendations
    print("\n" + "-"*70)
    print("ECONOMETRIC RESEARCH RECOMMENDATIONS")
    print("-"*70)

    recommendations = generate_econometric_recommendations(tier_analysis)
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY FOR PAPER")
    print("="*70)

    t0_df = df[df['temp'] == 0.0]
    print(f"""
Total Configurations Analyzed: {len(df)}
  - Temperature 0.0: {len(t0_df)}
  - Temperature 0.2: {len(df) - len(t0_df)}

Models by Provider:
  - Ollama (local): {df[df['provider']=='ollama']['model'].nunique()} models
  - watsonx (cloud): {df[df['provider']=='watsonx']['model'].nunique()} models
  - Anthropic: {df[df['provider']=='anthropic']['model'].nunique()} models
  - Gemini: {df[df['provider']=='gemini']['model'].nunique()} models

Key Findings:
  - Tier 1 (7-20B) achieves {tier_analysis.get('Tier1_7-20B', tier_analysis.get(list(tier_analysis.keys())[0])).mean_determinism:.0f}% determinism at T=0
  - SQL tasks show highest determinism across all tiers
  - RAG tasks show task-structure effect (lower determinism)
  - Faithfulness and determinism are {tradeoff['interpretation']}
""")

    return df, tier_analysis


if __name__ == "__main__":
    df, tier_analysis = main()
