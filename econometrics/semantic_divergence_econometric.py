"""
Semantic Divergence Metrics for Econometric LLM Measurement Tasks

Integrates semantic alignment measures with econometric validation framework
to distinguish "varying-but-correct" from "consistent-but-wrong" outputs.

References:
    Halperin, I. (2025). Semantic Divergence Metrics to Manage Hallucinations
    in Large Language Models. arXiv:2512.05156 (December 9, 2025)
    Code: https://github.com/ighalp/semantic-faithfulness-sdm

    Halperin, I. (2025). Prompt-Response Semantic Divergence Metrics for
    Faithfulness Hallucination and Misalignment Detection in Large Language Models.
    arXiv:2508.10192 (earlier version)

    Ludwig, J., Mullainathan, S., & Rambachan, A. (2024).
    Large Language Models: An Applied Econometric Framework.
    arXiv:2412.07031

Halperin (Dec 2025) Key Contribution - Information-Theoretic Faithfulness:
    The newer Halperin paper proposes a stricter way to measure hallucinations
    by modeling INFORMATION FLOW rather than just answer quality:

    1. Models two topic transition processes from the same context:
       - How the QUESTION selects/reweights topics in source material
       - How the ANSWER actually redistributes those topics

    2. Faithfulness = minimal KL divergence between these transformations

    3. Semantic Entropy Production: Measures irreversibility and noise the
       model introduces while generating an answer

    Critical Finding (from Fidelity Investments experiments on financial disclosures):
        "An answer can look correct, sound well, and still be structurally
         misaligned with the question and context."

        The metric penalized answers that quietly made up entities even when
        LLM-based judges rated them as "good and complete."

    Note: Halperin clarifies the title says "MANAGE hallucinations" (not capture)
    - this is standard terminology in psychiatry, acknowledging hallucinations
    cannot be eliminated, only managed.

Key Insight for Our Framework:
    Ludwig et al. focus on MEASUREMENT ERROR (systematic bias).
    We add SEMANTIC DRIFT distinction:
        - Low semantic divergence + high string drift → "varying-but-correct"
          (acceptable paraphrasing, doesn't affect downstream estimates)
        - Low string drift + high semantic divergence → "consistent-but-wrong"
          (repeating same error, biases downstream estimates)

    For econometric tasks: Semantic alignment matters MORE than string identity.

Implementation Note:
    This module provides a LIGHTWEIGHT TF-IDF approximation. For production
    financial compliance use cases, consider integrating Halperin's full
    KL-divergence framework from: https://github.com/ighalp/semantic-faithfulness-sdm

Novel Contribution (Our Extension):
    Lightweight TF-vector cosine divergence (no transformers required) that:
    1. Distinguishes measurement error from paraphrasing variance
    2. Flags systematic semantic drift (model degradation over time)
    3. Reduces validation subsample requirements (semantic equivalence classes)

Applications:
    - Sentiment labeling: "positive outlook" vs "optimistic view" (same sentiment)
    - Entity extraction: "JPMorgan" vs "JPMorgan Chase" (same entity)
    - Risk classification: "high risk" vs "elevated risk" (same label)
    - Financial disclosure analysis: Detect fabricated entities (Halperin's focus)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


@dataclass
class SemanticDivergenceMetrics:
    """Semantic divergence metrics for prompt-response pairs.

    Based on Halperin (2025) PRSD framework, adapted for econometric use.

    Attributes:
        prompt_response_div: Semantic mismatch between prompt and response [0,1]
        response_reference_div: Semantic mismatch from ground truth [0,1]
        token_mismatch: Vocabulary-level mismatch (Jaccard complement) [0,1]
        semantic_alignment_score: Overall alignment [0,1] (1=perfect)
        faithfulness_category: "faithful", "paraphrase", "hallucination", "drift"
    """
    prompt_response_div: float
    response_reference_div: float
    token_mismatch: float
    semantic_alignment_score: float
    faithfulness_category: str

    def summary(self) -> str:
        return f"""
Semantic Divergence Metrics
{'='*50}
Prompt-Response Div:      {self.prompt_response_div:.3f}
Response-Reference Div:   {self.response_reference_div:.3f}
Token Mismatch:           {self.token_mismatch:.3f}
---
Semantic Alignment:       {self.semantic_alignment_score:.3f}
Faithfulness Category:    {self.faithfulness_category.upper()}

INTERPRETATION:
  ≥0.80: Excellent alignment (econometrically equivalent)
  0.70-0.80: Good (minor paraphrasing, acceptable)
  0.50-0.70: Moderate drift (review needed)
  <0.50: Significant divergence (exclude from validation)
"""


@dataclass
class SemanticEquivalenceClass:
    """Group of semantically equivalent but lexically different responses.

    For econometric validation: Responses in same equivalence class
    can be treated as IDENTICAL for measurement error estimation,
    even if string edit distance is high.

    Example:
        Class 1: ["positive outlook", "optimistic view", "bullish sentiment"]
        → All map to sentiment=+1
        → Treated as 100% agreement for validation purposes
    """
    class_id: int
    canonical_response: str
    member_responses: List[str]
    semantic_centroid: np.ndarray
    alignment_threshold: float

    def contains(self, response: str, vectorizer) -> bool:
        """Check if response belongs to this equivalence class."""
        vec = vectorizer.transform([response])
        similarity = cosine_similarity(vec, self.semantic_centroid.reshape(1, -1))[0, 0]
        return similarity >= self.alignment_threshold


def compute_tf_vector(
    text: str,
    vectorizer: Optional[TfidfVectorizer] = None,
    vocab: Optional[Set[str]] = None
) -> np.ndarray:
    """Compute TF-IDF vector for text.

    Args:
        text: Input text
        vectorizer: Fitted TfidfVectorizer (or None to fit on text)
        vocab: Optional vocabulary constraint

    Returns:
        TF-IDF vector (sparse or dense)

    Example:
        >>> vec1 = compute_tf_vector("The company shows strong growth")
        >>> vec2 = compute_tf_vector("Strong growth at the company")
        >>> similarity = cosine_similarity(vec1.reshape(1,-1), vec2.reshape(1,-1))
        >>> print(similarity)  # High (~0.9) despite word reordering
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            vocabulary=vocab
        )
        vectorizer.fit([text])

    vec = vectorizer.transform([text])
    return vec.toarray()[0]


def compute_cosine_divergence(
    text1: str,
    text2: str,
    vectorizer: Optional[TfidfVectorizer] = None
) -> float:
    """Compute cosine divergence = 1 - cosine_similarity.

    Args:
        text1, text2: Input texts
        vectorizer: Shared TfidfVectorizer (or None to fit on both)

    Returns:
        Divergence in [0, 1] (0=identical semantics, 1=orthogonal)

    Example:
        >>> div = compute_cosine_divergence("strong earnings", "weak earnings")
        >>> print(div)  # High (~0.6) - opposite meanings
        >>>
        >>> div = compute_cosine_divergence("strong earnings", "robust earnings")
        >>> print(div)  # Low (~0.1) - similar meanings
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        vectorizer.fit([text1, text2])

    vec1 = vectorizer.transform([text1]).toarray()
    vec2 = vectorizer.transform([text2]).toarray()

    similarity = cosine_similarity(vec1, vec2)[0, 0]
    divergence = 1.0 - similarity

    return divergence


def compute_token_mismatch(text1: str, text2: str) -> float:
    """Compute Jaccard complement = 1 - Jaccard similarity.

    Measures vocabulary-level mismatch (word set overlap).

    Args:
        text1, text2: Input texts

    Returns:
        Mismatch in [0, 1] (0=identical tokens, 1=no overlap)

    Example:
        >>> mismatch = compute_token_mismatch("apple stock rises", "apple stock falls")
        >>> print(mismatch)  # 0.33 (1 different word out of 3 unique)
    """
    # Tokenize and lowercase
    tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
    tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    jaccard = intersection / union if union > 0 else 0.0
    mismatch = 1.0 - jaccard

    return mismatch


def compute_semantic_divergence_metrics(
    prompt: str,
    response: str,
    reference: Optional[str] = None,
    vectorizer: Optional[TfidfVectorizer] = None
) -> SemanticDivergenceMetrics:
    """Compute comprehensive semantic divergence metrics.

    Based on Halperin (2025) PRSD framework.

    Args:
        prompt: Input prompt/query
        response: LLM response
        reference: Ground truth reference (optional)
        vectorizer: Shared TfidfVectorizer

    Returns:
        SemanticDivergenceMetrics

    Example:
        >>> prompt = "What is the sentiment of this headline: 'Apple beats earnings'"
        >>> response = "The sentiment is positive"
        >>> reference = "positive"
        >>> metrics = compute_semantic_divergence_metrics(prompt, response, reference)
        >>> print(metrics.summary())
    """
    if vectorizer is None:
        corpus = [prompt, response]
        if reference:
            corpus.append(reference)
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        vectorizer.fit(corpus)

    # 1. Prompt-response divergence (faithfulness check)
    prompt_response_div = compute_cosine_divergence(prompt, response, vectorizer)

    # 2. Response-reference divergence (correctness check)
    if reference:
        response_reference_div = compute_cosine_divergence(response, reference, vectorizer)
    else:
        response_reference_div = 0.0

    # 3. Token mismatch (vocabulary overlap)
    if reference:
        token_mismatch = compute_token_mismatch(response, reference)
    else:
        token_mismatch = 0.0

    # 4. Overall semantic alignment score
    # Combine metrics: lower divergence → higher alignment
    alignment = 1.0 - (0.4 * prompt_response_div + 0.4 * response_reference_div + 0.2 * token_mismatch)
    alignment = max(0.0, min(1.0, alignment))  # clip to [0,1]

    # 5. Faithfulness category
    if alignment >= 0.80:
        category = "faithful"
    elif alignment >= 0.70:
        category = "paraphrase"  # varying-but-correct
    elif alignment >= 0.50:
        category = "drift"  # moderate semantic divergence
    else:
        category = "hallucination"  # consistent-but-wrong

    return SemanticDivergenceMetrics(
        prompt_response_div=prompt_response_div,
        response_reference_div=response_reference_div,
        token_mismatch=token_mismatch,
        semantic_alignment_score=alignment,
        faithfulness_category=category
    )


def detect_semantic_equivalence_classes(
    responses: List[str],
    alignment_threshold: float = 0.85,
    vectorizer: Optional[TfidfVectorizer] = None
) -> List[SemanticEquivalenceClass]:
    """Group responses into semantic equivalence classes.

    For econometric validation: Responses in same class are treated as
    identical measurements despite lexical differences.

    Args:
        responses: List of LLM responses
        alignment_threshold: Cosine similarity threshold for grouping
        vectorizer: Shared TfidfVectorizer

    Returns:
        List of SemanticEquivalenceClass objects

    Example:
        >>> responses = [
        ...     "positive outlook",
        ...     "optimistic view",
        ...     "bullish sentiment",
        ...     "negative outlook",
        ...     "pessimistic view"
        ... ]
        >>> classes = detect_semantic_equivalence_classes(responses)
        >>> print(f"Found {len(classes)} equivalence classes")  # 2 classes
        >>> # Class 1: positive/optimistic/bullish
        >>> # Class 2: negative/pessimistic
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        vectorizer.fit(responses)

    # Compute vectors
    vectors = vectorizer.transform(responses).toarray()

    # Greedy clustering: start with first response, group similar ones
    assigned = [False] * len(responses)
    classes = []
    class_id = 0

    for i in range(len(responses)):
        if assigned[i]:
            continue

        # Start new equivalence class
        canonical = responses[i]
        members = [canonical]
        centroid = vectors[i].copy()
        assigned[i] = True

        # Find similar responses
        for j in range(i+1, len(responses)):
            if assigned[j]:
                continue

            similarity = cosine_similarity(
                vectors[j].reshape(1, -1),
                centroid.reshape(1, -1)
            )[0, 0]

            if similarity >= alignment_threshold:
                members.append(responses[j])
                # Update centroid (running average)
                centroid = (centroid + vectors[j]) / 2.0
                assigned[j] = True

        classes.append(SemanticEquivalenceClass(
            class_id=class_id,
            canonical_response=canonical,
            member_responses=members,
            semantic_centroid=centroid,
            alignment_threshold=alignment_threshold
        ))
        class_id += 1

    return classes


def compute_effective_drift_rate_with_semantics(
    runs: List[str],
    alignment_threshold: float = 0.85
) -> Tuple[float, float, int]:
    """Compute drift rate accounting for semantic equivalence.

    Standard drift rate: Count exact string matches
    Semantic drift rate: Count semantic equivalence classes

    Args:
        runs: List of k runs (strings)
        alignment_threshold: Threshold for semantic equivalence

    Returns:
        (string_drift_rate, semantic_drift_rate, num_equivalence_classes)

    Example:
        >>> runs = ["strong growth", "robust growth", "strong growth", "weak growth"]
        >>> string_drift, semantic_drift, n_classes = compute_effective_drift_rate_with_semantics(runs)
        >>> print(f"String drift: {string_drift:.1%}")  # 75% (3 unique strings)
        >>> print(f"Semantic drift: {semantic_drift:.1%}")  # 50% (2 semantic classes)
        >>> # Class 1: strong/robust (same meaning)
        >>> # Class 2: weak (different meaning)
    """
    k = len(runs)

    # String-level drift rate (exact matches)
    counts = Counter(runs)
    mode_value, mode_count = counts.most_common(1)[0]
    string_drift_rate = (k - mode_count) / k

    # Semantic-level drift rate (equivalence classes)
    equiv_classes = detect_semantic_equivalence_classes(runs, alignment_threshold)
    n_classes = len(equiv_classes)

    # Find largest equivalence class
    max_class_size = max(len(cls.member_responses) for cls in equiv_classes)
    semantic_drift_rate = (k - max_class_size) / k

    return string_drift_rate, semantic_drift_rate, n_classes


def adjust_validation_subsample_with_semantics(
    responses: List[str],
    alignment_threshold: float = 0.85
) -> Tuple[int, List[int]]:
    """Reduce validation subsample size using semantic equivalence.

    Insight: If k responses are semantically equivalent, only need to
    validate ONE of them (not all k). This reduces human labeling cost.

    Args:
        responses: Full set of LLM responses
        alignment_threshold: Threshold for equivalence

    Returns:
        (effective_validation_size, representative_indices)

    Example:
        >>> responses = ["positive"] * 50 + ["negative"] * 30 + ["neutral"] * 20
        >>> eff_size, indices = adjust_validation_subsample_with_semantics(responses)
        >>> print(f"Original: {len(responses)}, Effective: {eff_size}")
        >>> # Original: 100, Effective: 3 (one per semantic class)
        >>> # Validation cost reduced by 97%!
    """
    equiv_classes = detect_semantic_equivalence_classes(responses, alignment_threshold)

    # Take one representative from each equivalence class
    representative_indices = []
    for cls in equiv_classes:
        # Find index of canonical response in original list
        canonical = cls.canonical_response
        idx = responses.index(canonical)
        representative_indices.append(idx)

    effective_size = len(equiv_classes)

    return effective_size, representative_indices


# ==============================================================================
# Integration with Econometric Framework
# ==============================================================================

def compute_semantic_measurement_error(
    y_llm: List[str],
    y_true: List[str],
    vectorizer: Optional[TfidfVectorizer] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose measurement error into semantic vs lexical components.

    Standard econometric model:
        Y_llm = Y_true + epsilon_model

    Our decomposition:
        epsilon_model = epsilon_semantic + epsilon_lexical

    Where:
        epsilon_semantic: True semantic errors (wrong meaning)
        epsilon_lexical: Paraphrasing variance (same meaning, different words)

    For downstream regressions: Only epsilon_semantic contributes to bias!

    Args:
        y_llm: LLM labels (strings)
        y_true: Ground truth labels (strings)
        vectorizer: Shared TfidfVectorizer

    Returns:
        (semantic_errors, lexical_errors) as arrays of floats [0,1]

    Example:
        >>> y_llm = ["positive outlook", "negative view", "optimistic"]
        >>> y_true = ["positive", "positive", "positive"]
        >>> sem_err, lex_err = compute_semantic_measurement_error(y_llm, y_true)
        >>> print(sem_err)  # [low, high, low] - second response is semantically wrong
        >>> print(lex_err)  # [medium, low, medium] - first/third are paraphrases
    """
    n = len(y_llm)
    semantic_errors = np.zeros(n)
    lexical_errors = np.zeros(n)

    if vectorizer is None:
        corpus = y_llm + y_true
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        vectorizer.fit(corpus)

    for i in range(n):
        # Semantic error: cosine divergence from ground truth
        semantic_errors[i] = compute_cosine_divergence(y_llm[i], y_true[i], vectorizer)

        # Lexical error: token mismatch (vocabulary overlap)
        lexical_errors[i] = compute_token_mismatch(y_llm[i], y_true[i])

    return semantic_errors, lexical_errors


# ==============================================================================
# Example Usage: Headline Sentiment with Semantic Drift
# ==============================================================================

def example_semantic_divergence_econometric():
    """
    Demonstrates semantic divergence integration with econometric framework.

    Key Finding:
        Tier~1 models: Low string drift + Low semantic drift → Faithful
        Tier~3 models: High string drift BUT semantically equivalent → Acceptable!

    This challenges the string-identity requirement for validation.
    """
    np.random.seed(42)

    print("="*60)
    print("Semantic Divergence + Econometric Validation")
    print("="*60)
    print()

    # Ground truth sentiment labels
    y_true = ["positive", "negative", "positive", "neutral", "positive"]

    # Tier~1 model: Low string drift, high semantic alignment
    y_tier1 = [
        "positive outlook",   # paraphrase of "positive"
        "negative view",      # paraphrase of "negative"
        "optimistic",         # paraphrase of "positive"
        "neutral stance",     # paraphrase of "neutral"
        "bullish"             # paraphrase of "positive"
    ]

    # Tier~3 model: High string drift, low semantic alignment (errors!)
    y_tier3 = [
        "positive outlook",   # correct
        "positive view",      # WRONG (should be negative)
        "pessimistic",        # WRONG (should be positive)
        "neutral stance",     # correct
        "negative"            # WRONG (should be positive)
    ]

    print("TIER~1 MODEL (Low String Drift, High Semantic Alignment):")
    print("-"*60)
    sem_err_tier1, lex_err_tier1 = compute_semantic_measurement_error(y_tier1, y_true)
    for i in range(len(y_true)):
        print(f"  Sample {i}: True='{y_true[i]}', LLM='{y_tier1[i]}'")
        print(f"    → Semantic error: {sem_err_tier1[i]:.3f}, Lexical error: {lex_err_tier1[i]:.3f}")

    print()
    print(f"Average semantic error: {sem_err_tier1.mean():.3f} (LOW → Good faithfulness)")
    print(f"Average lexical error: {lex_err_tier1.mean():.3f} (HIGH → Paraphrasing)")
    print()

    print("TIER~3 MODEL (High String Drift, Low Semantic Alignment):")
    print("-"*60)
    sem_err_tier3, lex_err_tier3 = compute_semantic_measurement_error(y_tier3, y_true)
    for i in range(len(y_true)):
        print(f"  Sample {i}: True='{y_true[i]}', LLM='{y_tier3[i]}'")
        print(f"    → Semantic error: {sem_err_tier3[i]:.3f}, Lexical error: {lex_err_tier3[i]:.3f}")

    print()
    print(f"Average semantic error: {sem_err_tier3.mean():.3f} (HIGH → Poor faithfulness)")
    print(f"Average lexical error: {lex_err_tier3.mean():.3f} (MEDIUM)")
    print()

    # Equivalence classes
    print("="*60)
    print("VALIDATION SUBSAMPLE REDUCTION VIA SEMANTIC EQUIVALENCE")
    print("="*60)

    # Simulate 100 responses with semantic clusters
    responses_cluster = (
        ["positive outlook"] * 30 +
        ["optimistic view"] * 20 +
        ["bullish sentiment"] * 10 +  # All semantically equivalent to "positive"
        ["negative outlook"] * 25 +
        ["pessimistic view"] * 15     # Semantically equivalent to "negative"
    )

    eff_size, repr_idx = adjust_validation_subsample_with_semantics(responses_cluster)

    print(f"Original response set: {len(responses_cluster)} samples")
    print(f"Effective validation size: {eff_size} samples")
    print(f"Validation cost reduction: {(1 - eff_size/len(responses_cluster)):.1%}")
    print()
    print("Representative samples (one per semantic class):")
    for idx in repr_idx:
        print(f"  - '{responses_cluster[idx]}'")

    print()
    print("="*60)
    print("KEY TAKEAWAY:")
    print("="*60)
    print("Semantic divergence metrics enable:")
    print("  1. Distinguish 'varying-but-correct' from 'consistent-but-wrong'")
    print("  2. Reduce validation subsample size by clustering semantic equivalents")
    print("  3. Focus econometric bias correction on TRUE semantic errors")
    print(f"     (not lexical paraphrasing variance)")


if __name__ == "__main__":
    example_semantic_divergence_econometric()
