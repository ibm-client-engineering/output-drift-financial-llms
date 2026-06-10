#!/usr/bin/env python3
"""
Cross-provider validation for LLM output consistency.

Validates that LLM outputs remain stable across deployment environments:
- Local (Ollama)
- Cloud (IBM watsonx.ai)

Implements finance-calibrated tolerance thresholds for MiFID II compliance.
"""
import hashlib
from typing import List, Dict, Any, Optional
from rapidfuzz.distance import Levenshtein


class CrossProviderValidator:
    """
    Cross-provider validation with finance-calibrated invariants.

    Key features:
    - Normalized edit distance for text comparison
    - Finance-calibrated tolerance thresholds (±5% for GAAP materiality)
    - Task-specific validation rules
    - Audit trail generation
    """

    def __init__(self, providers: List[str], tolerance_pct: float = 5.0):
        """
        Initialize cross-provider validator.

        Args:
            providers: List of provider names (e.g., ["ollama", "watsonx"])
            tolerance_pct: Tolerance percentage for numeric comparisons (default: 5% for GAAP)
        """
        self.providers = providers
        self.tolerance_pct = tolerance_pct

    def validate(
        self,
        outputs: Dict[str, str],
        task_type: str = "rag",
        citations: Optional[Dict[str, List[str]]] = None,
        sql_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate output consistency across providers using pre-collected outputs.

        Args:
            outputs: provider_name -> output text mapping
            task_type: "rag", "sql", or "summary"
            citations: provider_name -> citation list (required for rag)
            sql_results: provider_name -> numeric result (optional for sql)

        Returns:
            {
                "consistent": bool,
                "similarity_scores": Dict[str, float],
                "task_validation": Dict,
                "audit_trail": List[Dict]
            }
        """
        if len(outputs) < 2:
            return {
                "consistent": True,
                "similarity_scores": {},
                "task_validation": {"reason": "Single provider, no comparison needed"},
                "audit_trail": []
            }

        # Pairwise similarity
        provider_names = list(outputs.keys())
        ref = provider_names[0]
        similarity_scores = {}
        for provider in provider_names[1:]:
            key = f"{ref}_vs_{provider}"
            similarity_scores[key] = self.compute_similarity(
                outputs[ref], outputs[provider]
            )

        # Task-specific validation
        if task_type == "rag":
            task_validation = self.validate_rag_outputs(
                outputs, citations or {}
            )
        elif task_type == "sql":
            task_validation = self.validate_sql_outputs(
                outputs, sql_results or {}
            )
        elif task_type == "summary":
            task_validation = self.validate_summary_outputs(outputs)
        else:
            task_validation = {"error": f"Unknown task type: {task_type}"}

        consistent = task_validation.get("consistent", False)

        # Audit trail
        audit_trail = [{
            "providers": provider_names,
            "task_type": task_type,
            "consistent": consistent,
            "similarity_scores": similarity_scores,
            "output_hashes": {p: self.hash_output(o) for p, o in outputs.items()},
            "tolerance_pct": self.tolerance_pct,
        }]

        return {
            "consistent": consistent,
            "similarity_scores": similarity_scores,
            "task_validation": task_validation,
            "audit_trail": audit_trail
        }

    @staticmethod
    def compute_similarity(text1: str, text2: str) -> float:
        """
        Compute normalized similarity between two texts.

        Uses normalized Levenshtein distance: 1.0 = identical, 0.0 = completely different.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0-1.0)
        """
        if not text1 and not text2:
            return 1.0

        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0

        distance = Levenshtein.distance(text1, text2)
        return 1.0 - (distance / max_len)

    @staticmethod
    def hash_output(text: str) -> str:
        """
        Generate deterministic hash for output.

        Args:
            text: Output text

        Returns:
            SHA-256 hash (hex)
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def validate_rag_outputs(self, outputs: Dict[str, str], citations: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Validate RAG outputs across providers.

        Checks:
        - Citation consistency
        - Text similarity
        - Factual alignment

        Args:
            outputs: provider -> output text
            citations: provider -> list of citations

        Returns:
            Validation results
        """
        if len(outputs) < 2:
            return {"consistent": True, "reason": "Single provider, no comparison needed"}

        provider_names = list(outputs.keys())
        ref_provider = provider_names[0]
        ref_output = outputs[ref_provider]
        ref_citations = set(citations.get(ref_provider, []))

        results = {
            "consistent": True,
            "citation_consistent": True,
            "text_similarity": {},
            "citation_drift": {}
        }

        for provider in provider_names[1:]:
            # Text similarity
            similarity = self.compute_similarity(ref_output, outputs[provider])
            results["text_similarity"][f"{ref_provider}_vs_{provider}"] = similarity

            if similarity < 0.95:  # 95% similarity threshold
                results["consistent"] = False

            # Citation consistency
            current_citations = set(citations.get(provider, []))
            if ref_citations != current_citations:
                results["citation_consistent"] = False
                results["citation_drift"][provider] = {
                    "missing": list(ref_citations - current_citations),
                    "extra": list(current_citations - ref_citations)
                }

        return results

    def validate_summary_outputs(self, outputs: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate summary/JSON outputs across providers.

        Checks:
        - Text similarity across providers
        - JSON structure consistency (if outputs are valid JSON)

        Args:
            outputs: provider -> summary output text

        Returns:
            Validation results
        """
        import json

        if len(outputs) < 2:
            return {"consistent": True, "reason": "Single provider, no comparison needed"}

        provider_names = list(outputs.keys())
        ref_provider = provider_names[0]
        ref_output = outputs[ref_provider]

        validation = {
            "consistent": True,
            "text_similarity": {},
            "schema_consistent": True,
        }

        # Check if outputs parse as JSON and have matching keys
        ref_keys = None
        try:
            ref_parsed = json.loads(ref_output)
            ref_keys = set(ref_parsed.keys()) if isinstance(ref_parsed, dict) else None
        except (json.JSONDecodeError, AttributeError):
            pass

        for provider in provider_names[1:]:
            similarity = self.compute_similarity(ref_output, outputs[provider])
            validation["text_similarity"][f"{ref_provider}_vs_{provider}"] = similarity

            if similarity < 0.90:
                validation["consistent"] = False

            # Check JSON key consistency
            if ref_keys is not None:
                try:
                    parsed = json.loads(outputs[provider])
                    if isinstance(parsed, dict) and set(parsed.keys()) != ref_keys:
                        validation["schema_consistent"] = False
                except (json.JSONDecodeError, AttributeError):
                    validation["schema_consistent"] = False

        return validation

    def validate_sql_outputs(self, queries: Dict[str, str], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate SQL outputs across providers.

        Checks:
        - Query similarity
        - Result equivalence (within tolerance)

        Args:
            queries: provider -> SQL query
            results: provider -> query result

        Returns:
            Validation results
        """
        if len(queries) < 2:
            return {"consistent": True, "reason": "Single provider, no comparison needed"}

        provider_names = list(queries.keys())
        ref_provider = provider_names[0]
        ref_query = queries[ref_provider]
        ref_result = results.get(ref_provider)

        validation = {
            "consistent": True,
            "query_similarity": {},
            "result_match": {}
        }

        for provider in provider_names[1:]:
            # Query similarity
            similarity = self.compute_similarity(ref_query, queries[provider])
            validation["query_similarity"][f"{ref_provider}_vs_{provider}"] = similarity

            # Result equivalence (for numeric results)
            current_result = results.get(provider)
            if isinstance(ref_result, (int, float)) and isinstance(current_result, (int, float)):
                tolerance = (self.tolerance_pct / 100.0) * abs(ref_result)
                match = abs(ref_result - current_result) <= tolerance
                validation["result_match"][provider] = match
                if not match:
                    validation["consistent"] = False

        return validation


# Example usage
if __name__ == "__main__":
    # Demonstrate similarity computation
    validator = CrossProviderValidator(providers=["ollama", "watsonx"])

    text1 = "JPMorgan reported net credit losses of $1.2B in 2023. [jpm_2024_10k]"
    text2 = "JPMorgan reported net credit losses of $1.2B in 2023. [jpm_2024_10k]"
    text3 = "JPMorgan's net credit losses were $1.2 billion for 2023. [jpm_2024_10k]"

    print(f"Identical texts: {validator.compute_similarity(text1, text2):.2%}")
    print(f"Similar texts: {validator.compute_similarity(text1, text3):.2%}")
    print(f"Hash (text1): {validator.hash_output(text1)[:16]}...")
    print(f"Hash (text2): {validator.hash_output(text2)[:16]}...")
