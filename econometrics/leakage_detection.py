"""
Training Data Leakage Detection for LLM Prediction Tasks

Implements Ludwig, Mullainathan & Rambachan (2024) "no training leakage"
requirement for prediction problems (vs. estimation problems).

References:
    Ludwig, J., Mullainathan, S., & Rambachan, A. (2024).
    Large Language Models: An Applied Econometric Framework.
    arXiv:2412.07031

Key Distinction:
    - PREDICTION tasks (e.g., "predict stock movement from earnings call"):
      Require NO OVERLAP between LLM training data and test sample
      → Leakage detection is MANDATORY

    - ESTIMATION tasks (e.g., "label sentiment for downstream regression"):
      Can tolerate training overlap if validation debiasing is used
      → Leakage detection is OPTIONAL but recommended

Financial Applications:
    - Earnings forecasting: PREDICTION (must check leakage)
    - Sentiment labeling: ESTIMATION (validation debiasing handles this)
    - Anomaly detection: PREDICTION (must check leakage)
"""

import re
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class LeakageReport:
    """Results from training data leakage detection.

    Attributes:
        total_samples: Total test samples analyzed
        exact_matches: Number of exact n-gram matches in training corpus
        fuzzy_matches: Number of fuzzy matches (>90% similarity)
        temporal_violations: Samples with dates before model cutoff
        leakage_rate: Proportion of test set with suspected leakage
        flagged_samples: Indices of samples flagged for leakage
        details: Detailed match information per sample
    """
    total_samples: int
    exact_matches: int
    fuzzy_matches: int
    temporal_violations: int
    leakage_rate: float
    flagged_samples: List[int]
    details: Dict[int, Dict]

    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
Training Data Leakage Report
{'='*60}
Total test samples:       {self.total_samples}
Exact matches:            {self.exact_matches} ({self.exact_matches/self.total_samples:.1%})
Fuzzy matches:            {self.fuzzy_matches} ({self.fuzzy_matches/self.total_samples:.1%})
Temporal violations:      {self.temporal_violations} ({self.temporal_violations/self.total_samples:.1%})
---
Overall leakage rate:     {self.leakage_rate:.1%}
Flagged samples:          {len(self.flagged_samples)}

RECOMMENDATION: {'FAIL - High leakage risk' if self.leakage_rate > 0.05 else 'PASS - Low leakage risk'}
"""


# Model training cutoff dates (for temporal leak detection)
MODEL_CUTOFF_DATES = {
    # OpenAI models
    'gpt-4-turbo': datetime(2023, 12, 1),
    'gpt-4': datetime(2023, 9, 1),
    'gpt-3.5-turbo': datetime(2023, 9, 1),

    # Anthropic models
    'claude-opus-4': datetime(2024, 8, 1),  # claimed cutoff
    'claude-sonnet-4': datetime(2024, 7, 1),
    'claude-3-opus': datetime(2023, 8, 1),

    # Google models
    'gemini-2.5-pro': datetime(2024, 11, 1),
    'gemini-1.5-pro': datetime(2024, 5, 1),

    # Meta models
    'llama-3.3-70b': datetime(2023, 12, 1),
    'llama-3.1': datetime(2023, 12, 1),

    # IBM models (typically enterprise-trained on recent data)
    'granite-3-8b': datetime(2024, 6, 1),

    # Qwen models
    'qwen2.5-7b': datetime(2024, 6, 1),
    'qwen2.5-72b': datetime(2024, 6, 1),
}


def extract_dates_from_text(text: str) -> List[datetime]:
    """Extract dates from text using regex patterns.

    Handles formats:
        - YYYY-MM-DD (ISO)
        - MM/DD/YYYY (US)
        - Month DD, YYYY (natural language)
        - Q1 2024, Q2 2023 (quarters)

    Args:
        text: Input text potentially containing dates

    Returns:
        List of datetime objects

    Example:
        >>> text = "On 2024-03-15, the company reported earnings for Q1 2024."
        >>> dates = extract_dates_from_text(text)
        >>> print(dates)  # [datetime(2024, 3, 15), datetime(2024, 1, 1)]
    """
    dates = []

    # ISO format: YYYY-MM-DD
    iso_pattern = r'\b(\d{4})-(\d{2})-(\d{2})\b'
    for match in re.finditer(iso_pattern, text):
        try:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            dates.append(datetime(year, month, day))
        except ValueError:
            pass

    # US format: MM/DD/YYYY
    us_pattern = r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'
    for match in re.finditer(us_pattern, text):
        try:
            month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
            dates.append(datetime(year, month, day))
        except ValueError:
            pass

    # Natural language: Month DD, YYYY
    nl_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})\b'
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    for match in re.finditer(nl_pattern, text):
        try:
            month_name, day, year = match.group(1), int(match.group(2)), int(match.group(3))
            month = month_map[month_name]
            dates.append(datetime(year, month, day))
        except ValueError:
            pass

    # Quarters: Q1 YYYY, Q2 YYYY, etc.
    quarter_pattern = r'\bQ([1-4])\s+(\d{4})\b'
    for match in re.finditer(quarter_pattern, text):
        try:
            quarter, year = int(match.group(1)), int(match.group(2))
            month = (quarter - 1) * 3 + 1  # Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct
            dates.append(datetime(year, month, 1))
        except ValueError:
            pass

    return dates


def detect_temporal_leakage(
    test_samples: List[str],
    model_name: str,
    cutoff_date: Optional[datetime] = None
) -> Tuple[List[int], Dict]:
    """Detect temporal leakage: test samples with dates after model cutoff.

    Args:
        test_samples: List of test texts
        model_name: Model identifier (used to lookup cutoff date)
        cutoff_date: Override default cutoff date

    Returns:
        (flagged_indices, details_dict)

    Example:
        >>> samples = [
        ...     "On 2024-01-15, Apple reported record earnings.",
        ...     "Historical data from 2020 shows..."
        ... ]
        >>> flagged, details = detect_temporal_leakage(samples, 'gpt-4')
        >>> print(flagged)  # [0] - first sample has date after cutoff
    """
    if cutoff_date is None:
        cutoff_date = MODEL_CUTOFF_DATES.get(model_name, datetime(2023, 9, 1))

    flagged = []
    details = {}

    for idx, text in enumerate(test_samples):
        dates = extract_dates_from_text(text)
        recent_dates = [d for d in dates if d > cutoff_date]

        if recent_dates:
            flagged.append(idx)
            details[idx] = {
                'reason': 'temporal_violation',
                'cutoff_date': cutoff_date.isoformat(),
                'found_dates': [d.isoformat() for d in recent_dates],
                'text_preview': text[:200]
            }

    return flagged, details


def compute_ngram_fingerprint(
    text: str,
    n: int = 8,
    num_hashes: int = 5
) -> Set[str]:
    """Compute MinHash-style fingerprint using n-grams.

    Used for efficient exact match detection in large corpora.

    Args:
        text: Input text
        n: N-gram size (default 8 words)
        num_hashes: Number of hash functions (default 5)

    Returns:
        Set of hex hash strings

    Example:
        >>> text1 = "The company reported strong earnings in Q1 2024 with revenue growth"
        >>> text2 = "The company reported strong earnings in Q1 2024 with revenue growth"
        >>> fp1 = compute_ngram_fingerprint(text1)
        >>> fp2 = compute_ngram_fingerprint(text2)
        >>> print(len(fp1 & fp2))  # High overlap indicates match
    """
    words = text.lower().split()
    fingerprints = set()

    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])

        for seed in range(num_hashes):
            h = hashlib.md5(f"{seed}:{ngram}".encode()).hexdigest()
            fingerprints.add(h)

    return fingerprints


def detect_exact_matches(
    test_samples: List[str],
    training_corpus: Optional[List[str]] = None,
    n: int = 8,
    similarity_threshold: float = 0.3
) -> Tuple[List[int], Dict]:
    """Detect exact n-gram matches between test and training corpus.

    Args:
        test_samples: List of test texts
        training_corpus: List of training texts (or None to skip)
        n: N-gram size for fingerprinting
        similarity_threshold: Jaccard similarity threshold for flagging

    Returns:
        (flagged_indices, details_dict)

    Note:
        In practice, you'd compare against a database of known training
        corpora (e.g., Common Crawl snapshots, Wikipedia dumps, etc.)
        This implementation is a simplified demonstration.
    """
    if training_corpus is None:
        # Placeholder: in real implementation, load Common Crawl hashes, etc.
        return [], {}

    # Build fingerprint index for training corpus
    training_fingerprints = {}
    for doc_id, doc in enumerate(training_corpus):
        fp = compute_ngram_fingerprint(doc, n=n)
        training_fingerprints[doc_id] = fp

    flagged = []
    details = {}

    for idx, test_text in enumerate(test_samples):
        test_fp = compute_ngram_fingerprint(test_text, n=n)

        # Find best match in training corpus
        max_similarity = 0.0
        best_match_id = None

        for doc_id, train_fp in training_fingerprints.items():
            # Jaccard similarity
            intersection = len(test_fp & train_fp)
            union = len(test_fp | train_fp)
            similarity = intersection / union if union > 0 else 0.0

            if similarity > max_similarity:
                max_similarity = similarity
                best_match_id = doc_id

        if max_similarity >= similarity_threshold:
            flagged.append(idx)
            details[idx] = {
                'reason': 'exact_match',
                'similarity': max_similarity,
                'matched_doc_id': best_match_id,
                'text_preview': test_text[:200]
            }

    return flagged, details


def detect_fuzzy_matches(
    test_samples: List[str],
    known_datasets: List[str],
    similarity_threshold: float = 0.90
) -> Tuple[List[int], Dict]:
    """Detect fuzzy matches using edit distance / semantic similarity.

    For production use, replace with:
        - Elasticsearch/OpenSearch for large-scale matching
        - Sentence embeddings (e.g., SentenceTransformers) for semantic similarity
        - Annoy/FAISS for approximate nearest neighbors

    Args:
        test_samples: List of test texts
        known_datasets: List of known training dataset names/sources
        similarity_threshold: Threshold for flagging (0-1)

    Returns:
        (flagged_indices, details_dict)
    """
    # Placeholder: in real implementation, query vector DB
    # Example pseudocode:
    #   embeddings = model.encode(test_samples)
    #   matches = faiss_index.search(embeddings, k=1)
    #   flag if similarity > threshold

    flagged = []
    details = {}

    # Simplified: just check for common dataset mentions
    dataset_keywords = ['sec 10-k', 'edgar', 'wikipedia', 'common crawl', 'finbert']

    for idx, text in enumerate(test_samples):
        text_lower = text.lower()
        for keyword in dataset_keywords:
            if keyword in text_lower:
                flagged.append(idx)
                details[idx] = {
                    'reason': 'fuzzy_match',
                    'matched_keyword': keyword,
                    'text_preview': text[:200]
                }
                break

    return flagged, details


def run_leakage_detection(
    test_samples: List[str],
    model_name: str,
    training_corpus: Optional[List[str]] = None,
    cutoff_date: Optional[datetime] = None,
    exact_match_threshold: float = 0.3,
    fuzzy_match_threshold: float = 0.90
) -> LeakageReport:
    """Comprehensive leakage detection pipeline.

    Runs three checks:
        1. Temporal leakage (dates after model cutoff)
        2. Exact n-gram matches with training corpus
        3. Fuzzy semantic matches

    Args:
        test_samples: List of test texts
        model_name: Model identifier
        training_corpus: Optional training corpus for exact matching
        cutoff_date: Override model cutoff date
        exact_match_threshold: Jaccard threshold for exact matches
        fuzzy_match_threshold: Similarity threshold for fuzzy matches

    Returns:
        LeakageReport with findings

    Example:
        >>> test_samples = [
        ...     "Apple's Q1 2024 earnings exceeded expectations.",
        ...     "Historical analysis from 2020 shows..."
        ... ]
        >>> report = run_leakage_detection(test_samples, 'gpt-4')
        >>> print(report.summary())
    """
    total = len(test_samples)
    all_flagged = set()
    all_details = {}

    # 1. Temporal leakage
    temp_flagged, temp_details = detect_temporal_leakage(test_samples, model_name, cutoff_date)
    all_flagged.update(temp_flagged)
    all_details.update(temp_details)

    # 2. Exact matches
    exact_flagged, exact_details = detect_exact_matches(
        test_samples, training_corpus, similarity_threshold=exact_match_threshold
    )
    all_flagged.update(exact_flagged)
    all_details.update(exact_details)

    # 3. Fuzzy matches
    fuzzy_flagged, fuzzy_details = detect_fuzzy_matches(
        test_samples, [], similarity_threshold=fuzzy_match_threshold
    )
    all_flagged.update(fuzzy_flagged)
    all_details.update(fuzzy_details)

    # Aggregate counts
    exact_count = len(exact_flagged)
    fuzzy_count = len(fuzzy_flagged)
    temporal_count = len(temp_flagged)

    leakage_rate = len(all_flagged) / total if total > 0 else 0.0

    return LeakageReport(
        total_samples=total,
        exact_matches=exact_count,
        fuzzy_matches=fuzzy_count,
        temporal_violations=temporal_count,
        leakage_rate=leakage_rate,
        flagged_samples=sorted(all_flagged),
        details=all_details
    )


# ==============================================================================
# Example Usage: Financial Prediction Task
# ==============================================================================

def example_earnings_prediction_leakage_check():
    """
    Demonstrates leakage detection for PREDICTION task.

    Task: Predict stock movement from earnings call transcripts
    Requirement: NO overlap with model training data
    Method: Temporal + exact match detection
    """
    print("="*60)
    print("Training Data Leakage Detection: Earnings Prediction Task")
    print("="*60)
    print()

    # Simulated test samples (earnings calls)
    test_samples = [
        # Sample 1: Contains recent date (AFTER GPT-4 cutoff)
        "On January 15, 2024, Apple Inc. reported record Q1 earnings with revenue "
        "of $119.6 billion, exceeding analyst expectations. iPhone sales drove growth.",

        # Sample 2: Historical data (BEFORE cutoff) - should be OK
        "In Q2 2022, Microsoft reported cloud revenue growth of 32%, driven by Azure "
        "adoption among enterprise customers.",

        # Sample 3: No explicit dates, but recent companies
        "Tesla's Cybertruck launch event showcased new features and pricing tiers, "
        "targeting the electric pickup truck market.",

        # Sample 4: Contains dataset mention (potential fuzzy match)
        "This analysis uses SEC 10-K filings from EDGAR database to extract financial metrics.",
    ]

    # Run leakage detection
    report = run_leakage_detection(
        test_samples=test_samples,
        model_name='gpt-4',  # cutoff: Sept 2023
        training_corpus=None,  # no training corpus provided
        cutoff_date=datetime(2023, 9, 1)
    )

    print(report.summary())
    print()

    # Detailed findings
    print("Detailed Findings:")
    print("-" * 60)
    for idx in report.flagged_samples:
        detail = report.details[idx]
        print(f"\nSample {idx}: {detail['reason'].upper()}")
        print(f"  Text: {detail['text_preview']}...")
        if 'found_dates' in detail:
            print(f"  Dates found: {detail['found_dates']}")
            print(f"  Model cutoff: {detail['cutoff_date']}")
        if 'similarity' in detail:
            print(f"  Similarity: {detail['similarity']:.2%}")

    print("\n" + "="*60)
    print("RECOMMENDATION:")
    if report.leakage_rate > 0.05:
        print("⚠️  HIGH LEAKAGE RISK - Do NOT use this model for prediction")
        print("   Options:")
        print("   1. Use pre-2023 model (e.g., GPT-3.5 base, not turbo)")
        print("   2. Filter test set to pre-cutoff dates only")
        print("   3. Switch to ESTIMATION task with validation debiasing")
    else:
        print("✅ LOW LEAKAGE RISK - Safe for prediction task")
        print(f"   Flagged {len(report.flagged_samples)}/{report.total_samples} samples")
        print("   Recommend excluding flagged samples from test set")


if __name__ == "__main__":
    example_earnings_prediction_leakage_check()
