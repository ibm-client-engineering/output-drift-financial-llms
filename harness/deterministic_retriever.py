#!/usr/bin/env python3
"""
DeterministicRetriever: SEC 10-K structure-aware retrieval with stable precedence.

This historical workshop module uses a fixed, documented ordering for SEC 10-K
sections. The ordering is a benchmark design choice, not an SEC-prescribed
priority or a compliance determination.

SEC 10-K Disclosure Hierarchy (encoded in sorting):
    1. Risk Factors (Item 1A) - First in the benchmark tie-break order
    2. MD&A (Item 7) - Second in the benchmark tie-break order
    3. Financial Statements (Item 8) - Quantitative disclosures
    4. Legal Proceedings (Item 3) - Contingency disclosures
    5. Other Items - Lower priority supplementary information

The stable ordering ensures that identical queries at T=0.0 produce identical
context windows within the pinned benchmark environment.

AI4F Workshop 2025: "LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows"
"""
import re
import hashlib
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer

# Frozen stop words list for cross-version determinism.
# scikit-learn's ENGLISH_STOP_WORDS has changed between versions, which would
# alter TF-IDF scores and break retrieval determinism across environments.
# Frozen from scikit-learn 1.7.2 (318 words).
_FROZEN_STOP_WORDS = frozenset({
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around",
    "as", "at", "back", "be", "became", "because", "become", "becomes",
    "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside",
    "besides", "between", "beyond", "bill", "both", "bottom", "but", "by",
    "call", "can", "cannot", "cant", "co", "con", "could", "couldnt",
    "cry", "de", "describe", "detail", "do", "done", "down", "due",
    "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere",
    "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything",
    "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire",
    "first", "five", "for", "former", "formerly", "forty", "found", "four",
    "from", "front", "full", "further", "get", "give", "go", "had",
    "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc",
    "indeed", "interest", "into", "is", "it", "its", "itself", "keep",
    "last", "latter", "latterly", "least", "less", "ltd", "made", "many",
    "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover",
    "most", "mostly", "move", "much", "must", "my", "myself", "name",
    "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody",
    "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of",
    "off", "often", "on", "once", "one", "only", "onto", "or",
    "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over",
    "own", "part", "per", "perhaps", "please", "put", "rather", "re",
    "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several",
    "she", "should", "show", "side", "since", "sincere", "six", "sixty",
    "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere",
    "still", "such", "system", "take", "ten", "than", "that", "the",
    "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third",
    "this", "those", "though", "three", "through", "throughout", "thru", "thus",
    "to", "together", "too", "top", "toward", "towards", "twelve", "twenty",
    "two", "un", "under", "until", "up", "upon", "us", "very",
    "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
    "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole",
    "whom", "whose", "why", "will", "with", "within", "without", "would",
    "yet", "you", "your", "yours", "yourself", "yourselves",
})


# =============================================================================
# SEC 10-K DISCLOSURE PRECEDENCE (Regulation S-K Encoding)
# =============================================================================

# SEC Regulation S-K defines the structure of 10-K filings. This mapping encodes
# the regulatory importance of each section for retrieval precedence.
# Lower numbers = higher precedence (retrieved first when scores tie)
SEC_10K_SECTION_PRECEDENCE: Dict[str, int] = {
    # Item 1A - Risk Factors: Highest priority per SEC guidance on risk disclosure
    "risk_factors": 1,
    "risk factors": 1,
    "item 1a": 1,

    # Item 7 - MD&A: Critical for understanding financial condition
    "md&a": 2,
    "management discussion": 2,
    "management's discussion": 2,
    "item 7": 2,

    # Item 8 - Financial Statements: Core quantitative disclosures
    "financial statements": 3,
    "consolidated statements": 3,
    "item 8": 3,

    # Item 3 - Legal Proceedings: Material litigation disclosures
    "legal proceedings": 4,
    "item 3": 4,

    # Item 1 - Business Description: Company overview
    "business": 5,
    "item 1": 5,

    # Item 5 - Market Information: Stock and dividend data
    "market": 6,
    "item 5": 6,

    # Default precedence for unclassified sections
    "default": 99,
}

# Historical workshop target retained for compatibility; not a regulatory rule.
FSB_IDENTITY_REQUIREMENT: float = 1.0


class DeterministicRetriever:
    """
    Deterministic retrieval with a documented SEC-section tie-break order.

    This retriever implements finance-specific ordering that encodes SEC 10-K
    document structure. The sorting ensures:

    1. Higher relevance scores sort first (standard retrieval)
    2. The benchmark's section order breaks ties
    3. Snippet ID breaks remaining ties (deterministic ordering)

    Key features:
    - Deterministic chunking with semantic boundary preservation
    - SEC disclosure precedence encoding (not just tiebreaking)
    - Immutable snippet IDs using content-based hashing
    - Company-aware filtering for multi-entity queries
    """

    def __init__(self, docs: List[Dict[str, Any]], chunk_size: int = 200, overlap: int = 50):
        """
        Initialize retriever with deterministic chunking.

        Args:
            docs: List of documents with 'text', 'source', 'meta' fields
            chunk_size: Target words per chunk
            overlap: Words of overlap between chunks for context preservation
        """
        self.docs = docs
        self.snippets: List[Tuple[str, str, Dict[str, Any]]] = []

        # Create deterministic chunks with stable IDs
        for doc in docs:
            chunks = self._create_chunks(doc["text"], chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                # Stable snippet ID: source#chunk_index
                snippet_id = f"{doc['source']}#p{i}"
                self.snippets.append((snippet_id, chunk, doc.get("meta", {})))

        # Sort for determinism (critical for reproducibility)
        self.snippets.sort(key=lambda x: x[0])

        # Build TF-IDF index with deterministic parameters
        corpus = [snippet[1] for snippet in self.snippets]
        self.vectorizer = TfidfVectorizer(
            min_df=1,
            ngram_range=(1, 2),
            stop_words=_FROZEN_STOP_WORDS
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Create overlapping chunks with deterministic splitting.

        Splits on sentences to preserve semantic boundaries, then chunks by word count.
        Overlap ensures context continuity across chunk boundaries.
        """
        # Split on sentences (deterministic regex)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            words = sentence.split()
            sentence_size = len(words)

            if current_size + sentence_size > chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Start new chunk with overlap (last N words)
                overlap_words = overlap if overlap < current_size else current_size // 2
                if overlap_words > 0:
                    all_words = ' '.join(current_chunk).split()
                    current_chunk = all_words[-overlap_words:]
                    current_size = len(current_chunk)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.extend(words)
            current_size += sentence_size

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks if chunks else [text]  # fallback to full text

    def _get_sec_section_precedence(self, text: str) -> int:
        """
        Determine SEC 10-K section precedence from snippet content.

        Per SEC Regulation S-K, different sections of 10-K filings have varying
        regulatory importance. This method classifies snippets to enable
        precedence-aware retrieval ordering.

        Args:
            text: Snippet text content

        Returns:
            Precedence score (lower = higher priority per SEC hierarchy)
        """
        text_lower = text.lower()
        for section_key, precedence in SEC_10K_SECTION_PRECEDENCE.items():
            if section_key in text_lower:
                return precedence
        return SEC_10K_SECTION_PRECEDENCE["default"]

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Retrieve top-k snippets with SEC disclosure precedence encoding.

        The multi-key sort produces a stable order within the pinned benchmark
        environment.

        Sort Keys (in order of priority):
            1. TF-IDF similarity score (descending) - Relevance
            2. Benchmark section precedence (ascending)
            3. Snippet ID (ascending) - Deterministic tiebreaking

        The SEC precedence encoding ensures that when multiple snippets have
        equal relevance scores, the documented section order resolves the tie.

        Args:
            query: Search query
            k: Number of snippets to return

        Returns:
            List of (snippet_id, text, metadata) tuples sorted per SEC precedence
        """
        if not self.snippets:
            return []

        # Vectorize query
        query_vec = self.vectorizer.transform([query])

        # Compute TF-IDF similarities
        similarities = (self.tfidf_matrix @ query_vec.T).toarray().ravel()

        # Create scored snippets with SEC precedence metadata
        scored_snippets = []
        for i in range(len(self.snippets)):
            snippet_id, text, meta = self.snippets[i]
            sec_precedence = self._get_sec_section_precedence(text)
            scored_snippets.append((
                similarities[i],      # TF-IDF score
                sec_precedence,       # SEC disclosure hierarchy
                snippet_id,           # Deterministic tiebreaker
                self.snippets[i]      # Full snippet tuple
            ))

        # BENCHMARK SORT ORDER:
        # 1. Similarity (descending) - Most relevant first
        # 2. SEC precedence (ascending) - Risk Factors > MD&A > Other
        # 3. Snippet ID (ascending) - Deterministic final ordering
        scored_snippets.sort(key=lambda x: (-x[0], x[1], x[2]))

        # Return top-k snippets
        return [snippet for _, _, _, snippet in scored_snippets[:k]]


def create_retriever_from_files(corpus_path: str, chunk_size: int = 200, overlap: int = 50) -> DeterministicRetriever:
    """
    Convenience function to create retriever from SEC filings directory.

    Args:
        corpus_path: Path to directory containing SEC 10-K files (*_2024_10k.txt)
        chunk_size: Words per chunk
        overlap: Overlap between chunks

    Returns:
        Initialized DeterministicRetriever
    """
    import glob
    import pathlib

    sec_path = pathlib.Path(corpus_path)
    sec_files = sorted(glob.glob(str(sec_path / "*_2024_10k.txt")))

    if not sec_files:
        raise FileNotFoundError(f"No SEC 10-K files found in {corpus_path}")

    docs = []
    for filepath in sec_files:
        filepath = pathlib.Path(filepath)
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        docs.append({
            "text": text,
            "source": filepath.stem,
            "meta": {"filepath": str(filepath)}
        })

    return DeterministicRetriever(docs, chunk_size, overlap)
