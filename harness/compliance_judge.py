#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-Judge for Financial AI Compliance.

This module implements compliance-specific LLM-as-Judge methodology that evaluates
model outputs against regulatory requirements (FSB, BIS, CFTC, SEC) using a
"judge" LLM to assess faithfulness, determinism, and regulatory compliance.

Patent Novelty (per IBM IDT Feedback):
    Unlike generic LLM-as-Judge approaches (quality, helpfulness), this module
    evaluates outputs against:
    1. Specific regulatory requirements with embedded FSB/BIS/CFTC/SEC criteria
    2. Finance-calibrated thresholds (GAAP materiality, SEC citation rules)
    3. Multi-model consensus attestation for audit trail generation
    4. Faithfulness-determinism decomposition for compliance classification

Regulatory Framework References:
    - FSB BCBS-239: "Consistent and predictable" AI outputs
    - BIS Article 15: Cross-provider validation requirements
    - CFTC Rule 17a-4: Audit trail completeness
    - SEC Rule 10b-5: Citation accuracy requirements
    - GAAP ASC 450-20: 5% materiality threshold

ACM ICAIF 2025 / JFDS 2025: "LLM Output Drift: Cross-Provider Validation for Financial Workflows"
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import uuid
import re

# Import from existing regulatory framework
from .regulatory_invariants import (
    RegulatoryBody,
    RegulatoryRequirement,
    REGULATORY_REQUIREMENTS,
    TASK_REGULATORY_MAPPINGS,
    GAAP_MATERIALITY_THRESHOLD,
    FSB_IDENTITY_REQUIREMENT,
    SEC_CITATION_ACCURACY_THRESHOLD,
)


# =============================================================================
# JUDGE PROMPT TEMPLATES (Compliance-Specific)
# =============================================================================

JUDGE_PROMPT_FSB_CONSISTENCY = """You are a compliance auditor evaluating LLM outputs for FSB regulatory requirements.

## FSB Requirement (BCBS-239 Principle 6: Accuracy)
AI systems used in regulatory reporting must produce "consistent and predictable" outputs.
Identical inputs MUST produce identical outputs for audit compliance.

## Candidate Responses
The following responses were generated from identical prompts across multiple runs:

{responses}

## Evaluation Criteria
1. Are all responses semantically equivalent? (Consider: same facts, same conclusions, same citations)
2. If responses differ, what is the nature of the inconsistency?
   - NONE: Responses are identical or semantically equivalent
   - SEMANTIC: Same meaning, different wording (acceptable for some tasks)
   - FACTUAL: Different facts or claims (compliance risk)
   - STRUCTURAL: Different format or organization (may affect audit)
3. Does the inconsistency affect regulatory compliance?

## Output Format (JSON only, no markdown)
{{
  "fsb_compliant": true/false,
  "consistency_score": 0.0-1.0,
  "inconsistency_type": "none|semantic|factual|structural",
  "regulatory_risk": "low|medium|high",
  "explanation": "Brief explanation of findings"
}}"""

JUDGE_PROMPT_SEC_CITATIONS = """You are validating LLM outputs against SEC 10-K filings per Rule 10b-5.

## SEC Requirement (Rule 10b-5: Anti-Fraud Provisions)
AI-generated content citing SEC filings must accurately reference source documents.
Fabricated or hallucinated citations may constitute a violation of anti-fraud provisions.

## Source Document Excerpt
{source_excerpt}

## LLM Response to Validate
{response}

## Validation Criteria
1. Are all cited facts present in the source document?
2. Are citation references formatted correctly (e.g., "10-K 2024, Item 7")?
3. Are numerical values within GAAP materiality threshold (±5%)?
4. Are there any hallucinated facts not in the source?

## Output Format (JSON only, no markdown)
{{
  "sec_compliant": true/false,
  "citations_valid": true/false,
  "facts_verified": ["list of facts found in source"],
  "facts_unverified": ["list of facts NOT in source"],
  "numerical_accuracy": 0.0-1.0,
  "materiality_violations": ["list of values exceeding 5% threshold"],
  "explanation": "Brief explanation"
}}"""

JUDGE_PROMPT_FAITHFULNESS_DETERMINISM = """You are evaluating an LLM response for the faithfulness-determinism compliance framework.

## Two-Dimensional Compliance Framework
Financial AI compliance requires BOTH:
1. FAITHFULNESS: Factual accuracy against source documents
2. DETERMINISM: Output consistency across identical runs

## Classification Quadrants
- Q1 (Faithful + Deterministic): COMPLIANT - Deploy for regulated tasks
- Q2 (Faithful + Variable): CAUTION - Verify outputs, consider for non-critical tasks
- Q3 (Unfaithful + Deterministic): DANGEROUS - Consistently wrong, high compliance risk
- Q4 (Unfaithful + Variable): NON-COMPLIANT - Do not deploy for any regulated task

## Source Document
{source_document}

## LLM Response (Run {run_number} of {total_runs})
{response}

## Previous Runs Summary
{previous_runs_summary}

## Evaluation
Classify this response into one of the four quadrants based on:
1. Faithfulness: Does it accurately reflect the source?
2. Determinism: Is it consistent with previous runs?

## Output Format (JSON only, no markdown)
{{
  "quadrant": "Q1|Q2|Q3|Q4",
  "faithful": true/false,
  "deterministic": true/false,
  "faithfulness_score": 0.0-1.0,
  "determinism_score": 0.0-1.0,
  "compliance_status": "compliant|caution|dangerous|non_compliant",
  "regulatory_risk": "low|medium|high|critical",
  "explanation": "Brief explanation"
}}"""

JUDGE_PROMPT_CONSENSUS_ATTESTATION = """You are generating a regulatory-compliant attestation for multi-model consensus evaluation.

## Multi-Model Consensus Requirement
Per FSB/BIS guidelines, critical financial AI decisions should be validated across multiple models
to ensure consistency and reduce single-model bias risk.

## Task Description
{task_description}

## Models Evaluated
{model_names}

## Model Responses
{model_responses}

## Attestation Criteria
1. Do all models agree on the core answer/recommendation?
2. What is the semantic similarity across responses?
3. Which response best meets regulatory requirements?
4. Generate audit trail entry for compliance records.

## Output Format (JSON only, no markdown)
{{
  "consensus_achieved": true/false,
  "consensus_score": 0.0-1.0,
  "best_response_model": "model name",
  "regulatory_confidence": "low|medium|high",
  "key_findings": ["list of key consensus points"],
  "divergence_points": ["list of disagreements if any"],
  "audit_trail_entry": {{
    "attestation_type": "multi_model_consensus",
    "models_evaluated": ["list"],
    "consensus_hash": "to be computed",
    "regulatory_mapping": {{"FSB": "pass|fail", "BIS": "pass|fail", "CFTC": "pass|fail"}}
  }},
  "explanation": "Brief explanation"
}}"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ComplianceQuadrant(Enum):
    """Faithfulness-Determinism quadrant classification."""
    Q1_FAITHFUL_DETERMINISTIC = "Q1"      # COMPLIANT
    Q2_FAITHFUL_VARIABLE = "Q2"           # CAUTION
    Q3_UNFAITHFUL_DETERMINISTIC = "Q3"    # DANGEROUS
    Q4_UNFAITHFUL_VARIABLE = "Q4"         # NON-COMPLIANT


@dataclass
class JudgeEvaluation:
    """Result of a single judge evaluation."""
    evaluation_id: str
    timestamp: str
    judge_model: str
    evaluation_type: str  # 'fsb_consistency', 'sec_citations', 'faithfulness_determinism', 'consensus'
    input_data: Dict[str, Any]
    raw_judgment: str
    parsed_judgment: Dict[str, Any]
    compliant: bool
    confidence_score: float
    regulatory_requirements: List[str]


@dataclass
class ComplianceAttestation:
    """Regulatory-compliant attestation record for audit trails."""
    attestation_id: str
    timestamp: str
    judge_model: str
    evaluations: List[JudgeEvaluation]
    overall_compliant: bool
    regulatory_compliance: Dict[str, bool]  # Per-regulation pass/fail
    confidence_score: float
    attestation_hash: str
    regulatory_metadata: Dict[str, Any]


# =============================================================================
# COMPLIANCE JUDGE CLASS
# =============================================================================

class ComplianceJudge:
    """
    LLM-as-Judge for regulatory compliance evaluation.

    This class uses a "judge" LLM to evaluate candidate model outputs against
    specific regulatory requirements, going beyond generic quality metrics to
    assess FSB consistency, SEC citation accuracy, GAAP materiality, and
    multi-model consensus.

    Patent Novelty:
        - Regulatory-specific judge prompts (not generic quality evaluation)
        - Finance-calibrated thresholds embedded in evaluation
        - Faithfulness-determinism decomposition framework
        - Multi-model consensus attestation for audit trails

    Example:
        >>> judge = ComplianceJudge(judge_model_fn=my_llm_function)
        >>> result = judge.evaluate_fsb_consistency(responses)
        >>> attestation = judge.generate_attestation([result])
    """

    def __init__(
        self,
        judge_model_fn: callable = None,
        judge_model_name: str = "granite-3-8b-instruct",
        regulatory_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the Compliance Judge.

        Args:
            judge_model_fn: Callable that takes a prompt string and returns response string.
                           Signature: fn(prompt: str) -> str
            judge_model_name: Name of the judge model for audit records.
            regulatory_thresholds: Optional override of default regulatory thresholds.
        """
        self.judge_model_fn = judge_model_fn
        self.judge_model_name = judge_model_name

        # Regulatory thresholds (can be overridden)
        self.thresholds = regulatory_thresholds or {
            "fsb_consistency": FSB_IDENTITY_REQUIREMENT,  # 100% required
            "gaap_materiality": GAAP_MATERIALITY_THRESHOLD,  # 5% tolerance
            "sec_citation_accuracy": SEC_CITATION_ACCURACY_THRESHOLD,  # 95% required
            "cftc_audit_completeness": 1.0,  # 100% trace coverage
        }

        self._evaluations: List[JudgeEvaluation] = []

    def _generate_evaluation_id(self) -> str:
        """Generate unique evaluation ID."""
        return str(uuid.uuid4())[:8]

    def _get_timestamp(self) -> str:
        """Get ISO-8601 timestamp."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from judge response, handling markdown code blocks.

        Args:
            response: Raw response from judge LLM

        Returns:
            Parsed JSON dict, or error dict if parsing fails
        """
        # Remove markdown code blocks if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # Remove opening ```json or ```
            cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
            # Remove closing ```
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            return {
                "parse_error": str(e),
                "raw_response": response[:500],
                "compliant": False
            }

    def _call_judge(self, prompt: str) -> str:
        """
        Call the judge model with a prompt.

        Args:
            prompt: The evaluation prompt

        Returns:
            Judge's response string
        """
        if self.judge_model_fn is None:
            # Return mock response for testing without actual LLM
            return json.dumps({
                "mock_response": True,
                "note": "No judge_model_fn provided. Set judge_model_fn to use actual LLM evaluation."
            })

        return self.judge_model_fn(prompt)

    def evaluate_fsb_consistency(
        self,
        responses: List[str],
        prompt_context: Optional[str] = None
    ) -> JudgeEvaluation:
        """
        Evaluate output consistency per FSB BCBS-239 Principle 6.

        Uses LLM-as-Judge to assess whether multiple responses from identical
        prompts are semantically consistent, going beyond string matching.

        Args:
            responses: List of responses from identical prompts
            prompt_context: Optional context about the original prompt

        Returns:
            JudgeEvaluation with FSB compliance assessment
        """
        # Format responses for judge
        formatted_responses = "\n\n".join([
            f"Response {i+1}:\n{r}" for i, r in enumerate(responses)
        ])

        judge_prompt = JUDGE_PROMPT_FSB_CONSISTENCY.format(
            responses=formatted_responses
        )

        raw_judgment = self._call_judge(judge_prompt)
        parsed = self._parse_json_response(raw_judgment)

        evaluation = JudgeEvaluation(
            evaluation_id=self._generate_evaluation_id(),
            timestamp=self._get_timestamp(),
            judge_model=self.judge_model_name,
            evaluation_type="fsb_consistency",
            input_data={
                "num_responses": len(responses),
                "prompt_context": prompt_context,
                "response_lengths": [len(r) for r in responses]
            },
            raw_judgment=raw_judgment,
            parsed_judgment=parsed,
            compliant=parsed.get("fsb_compliant", False),
            confidence_score=parsed.get("consistency_score", 0.0),
            regulatory_requirements=["fsb_consistent_decisions"]
        )

        self._evaluations.append(evaluation)
        return evaluation

    def evaluate_sec_citations(
        self,
        response: str,
        source_excerpt: str,
        source_document_id: Optional[str] = None
    ) -> JudgeEvaluation:
        """
        Evaluate citation accuracy per SEC Rule 10b-5.

        Uses LLM-as-Judge to verify that citations in the response accurately
        reference the source document, checking for hallucinated facts.

        Args:
            response: The LLM response to validate
            source_excerpt: Relevant excerpt from SEC filing
            source_document_id: Optional identifier (e.g., "AAPL_10K_2024")

        Returns:
            JudgeEvaluation with SEC citation compliance assessment
        """
        judge_prompt = JUDGE_PROMPT_SEC_CITATIONS.format(
            source_excerpt=source_excerpt[:4000],  # Limit context size
            response=response
        )

        raw_judgment = self._call_judge(judge_prompt)
        parsed = self._parse_json_response(raw_judgment)

        evaluation = JudgeEvaluation(
            evaluation_id=self._generate_evaluation_id(),
            timestamp=self._get_timestamp(),
            judge_model=self.judge_model_name,
            evaluation_type="sec_citations",
            input_data={
                "source_document_id": source_document_id,
                "source_excerpt_length": len(source_excerpt),
                "response_length": len(response)
            },
            raw_judgment=raw_judgment,
            parsed_judgment=parsed,
            compliant=parsed.get("sec_compliant", False) and parsed.get("citations_valid", False),
            confidence_score=parsed.get("numerical_accuracy", 0.0),
            regulatory_requirements=["sec_citation_accuracy", "gaap_materiality"]
        )

        self._evaluations.append(evaluation)
        return evaluation

    def evaluate_faithfulness_determinism(
        self,
        response: str,
        source_document: str,
        previous_responses: List[str] = None,
        run_number: int = 1,
        total_runs: int = 1
    ) -> JudgeEvaluation:
        """
        Evaluate response using faithfulness-determinism decomposition.

        Classifies the response into one of four quadrants:
        - Q1: Faithful + Deterministic (COMPLIANT)
        - Q2: Faithful + Variable (CAUTION)
        - Q3: Unfaithful + Deterministic (DANGEROUS)
        - Q4: Unfaithful + Variable (NON-COMPLIANT)

        Args:
            response: The LLM response to evaluate
            source_document: Source document for faithfulness check
            previous_responses: List of responses from previous runs
            run_number: Current run number
            total_runs: Total number of runs

        Returns:
            JudgeEvaluation with quadrant classification
        """
        previous_summary = "No previous runs" if not previous_responses else (
            f"Previous {len(previous_responses)} runs: " +
            ", ".join([f"Run {i+1}: {r[:100]}..." for i, r in enumerate(previous_responses)])
        )

        judge_prompt = JUDGE_PROMPT_FAITHFULNESS_DETERMINISM.format(
            source_document=source_document[:3000],
            response=response,
            run_number=run_number,
            total_runs=total_runs,
            previous_runs_summary=previous_summary
        )

        raw_judgment = self._call_judge(judge_prompt)
        parsed = self._parse_json_response(raw_judgment)

        quadrant = parsed.get("quadrant", "Q4")
        compliant = quadrant == "Q1"

        evaluation = JudgeEvaluation(
            evaluation_id=self._generate_evaluation_id(),
            timestamp=self._get_timestamp(),
            judge_model=self.judge_model_name,
            evaluation_type="faithfulness_determinism",
            input_data={
                "run_number": run_number,
                "total_runs": total_runs,
                "previous_runs_count": len(previous_responses) if previous_responses else 0
            },
            raw_judgment=raw_judgment,
            parsed_judgment=parsed,
            compliant=compliant,
            confidence_score=(
                (parsed.get("faithfulness_score", 0) + parsed.get("determinism_score", 0)) / 2
            ),
            regulatory_requirements=["fsb_consistent_decisions", "sec_citation_accuracy"]
        )

        self._evaluations.append(evaluation)
        return evaluation

    def evaluate_consensus(
        self,
        model_responses: Dict[str, str],
        task_description: str
    ) -> JudgeEvaluation:
        """
        Evaluate multi-model consensus for audit trail generation.

        Assesses whether multiple models agree on the core answer, generating
        an attestation record suitable for regulatory audit trails.

        Args:
            model_responses: Dict mapping model names to their responses
            task_description: Description of the task being evaluated

        Returns:
            JudgeEvaluation with consensus assessment and attestation
        """
        formatted_responses = "\n\n".join([
            f"### {model_name}:\n{response}"
            for model_name, response in model_responses.items()
        ])

        judge_prompt = JUDGE_PROMPT_CONSENSUS_ATTESTATION.format(
            task_description=task_description,
            model_names=", ".join(model_responses.keys()),
            model_responses=formatted_responses
        )

        raw_judgment = self._call_judge(judge_prompt)
        parsed = self._parse_json_response(raw_judgment)

        # Compute consensus hash
        all_responses = "".join(sorted(model_responses.values()))
        consensus_hash = hashlib.sha256(all_responses.encode()).hexdigest()[:16]

        # Update parsed judgment with computed hash
        if "audit_trail_entry" in parsed:
            parsed["audit_trail_entry"]["consensus_hash"] = consensus_hash

        evaluation = JudgeEvaluation(
            evaluation_id=self._generate_evaluation_id(),
            timestamp=self._get_timestamp(),
            judge_model=self.judge_model_name,
            evaluation_type="consensus",
            input_data={
                "models_evaluated": list(model_responses.keys()),
                "task_description": task_description,
                "consensus_hash": consensus_hash
            },
            raw_judgment=raw_judgment,
            parsed_judgment=parsed,
            compliant=parsed.get("consensus_achieved", False),
            confidence_score=parsed.get("consensus_score", 0.0),
            regulatory_requirements=["fsb_consistent_decisions", "cftc_audit_trail"]
        )

        self._evaluations.append(evaluation)
        return evaluation

    def generate_attestation(
        self,
        evaluations: Optional[List[JudgeEvaluation]] = None
    ) -> ComplianceAttestation:
        """
        Generate regulatory-compliant attestation record.

        Aggregates multiple evaluations into a single attestation suitable
        for audit trail submission.

        Args:
            evaluations: List of evaluations to include (defaults to all stored)

        Returns:
            ComplianceAttestation with regulatory compliance summary
        """
        evals = evaluations or self._evaluations

        if not evals:
            raise ValueError("No evaluations to attest. Run evaluate_* methods first.")

        # Aggregate regulatory compliance
        regulatory_compliance = {
            "fsb": all(
                e.parsed_judgment.get("fsb_compliant", True)
                for e in evals if "fsb" in e.evaluation_type
            ),
            "bis": all(
                e.parsed_judgment.get("consensus_achieved", True)
                for e in evals if e.evaluation_type == "consensus"
            ),
            "cftc": all(
                "audit_trail_entry" in e.parsed_judgment
                for e in evals if e.evaluation_type == "consensus"
            ),
            "sec": all(
                e.parsed_judgment.get("sec_compliant", True)
                for e in evals if e.evaluation_type == "sec_citations"
            ),
        }

        overall_compliant = all(regulatory_compliance.values())

        # Compute attestation hash (deterministic - excludes timestamp for verifiability)
        attestation_data = json.dumps({
            "evaluations": [e.evaluation_id for e in evals],
            "regulatory_compliance": regulatory_compliance,
        }, sort_keys=True)
        attestation_hash = hashlib.sha256(attestation_data.encode()).hexdigest()

        # Calculate average confidence
        confidence_scores = [e.confidence_score for e in evals if e.confidence_score > 0]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        attestation = ComplianceAttestation(
            attestation_id=str(uuid.uuid4()),
            timestamp=self._get_timestamp(),
            judge_model=self.judge_model_name,
            evaluations=evals,
            overall_compliant=overall_compliant,
            regulatory_compliance=regulatory_compliance,
            confidence_score=avg_confidence,
            attestation_hash=attestation_hash,
            regulatory_metadata={
                "framework": "ACM_ICAIF_2025_JFDS_Financial_AI",
                "thresholds": self.thresholds,
                "regulatory_bodies": ["FSB", "BIS", "CFTC", "SEC"],
                "evaluation_count": len(evals)
            }
        )

        return attestation

    def to_audit_record(self, attestation: ComplianceAttestation) -> Dict[str, Any]:
        """
        Convert attestation to audit-trail-compatible record.

        Produces a JSON-serializable dict suitable for JSONL trace files.

        Args:
            attestation: The attestation to convert

        Returns:
            Dict suitable for audit trail logging
        """
        return {
            "record_type": "compliance_attestation",
            "attestation_id": attestation.attestation_id,
            "timestamp": attestation.timestamp,
            "judge_model": attestation.judge_model,
            "overall_compliant": attestation.overall_compliant,
            "regulatory_compliance": attestation.regulatory_compliance,
            "confidence_score": attestation.confidence_score,
            "attestation_hash": attestation.attestation_hash,
            "evaluation_summary": [
                {
                    "id": e.evaluation_id,
                    "type": e.evaluation_type,
                    "compliant": e.compliant,
                    "confidence": e.confidence_score
                }
                for e in attestation.evaluations
            ],
            "regulatory_metadata": attestation.regulatory_metadata
        }

    def clear_evaluations(self):
        """Clear stored evaluations."""
        self._evaluations = []


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_ollama_judge(model: str = "granite3.3:latest") -> ComplianceJudge:
    """
    Create a ComplianceJudge using Ollama as the backend.

    Args:
        model: Ollama model name to use as judge

    Returns:
        Configured ComplianceJudge instance
    """
    try:
        import ollama

        def ollama_generate(prompt: str) -> str:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={"temperature": 0.0, "seed": 42}
            )
            return response.get("response", "")

        return ComplianceJudge(
            judge_model_fn=ollama_generate,
            judge_model_name=f"ollama/{model}"
        )
    except ImportError:
        raise ImportError("ollama package required. Install with: pip install ollama")


def create_watsonx_judge(
    model_id: str = "ibm/granite-3-8b-instruct",
    api_key: Optional[str] = None,
    project_id: Optional[str] = None
) -> ComplianceJudge:
    """
    Create a ComplianceJudge using IBM watsonx.ai as the backend.

    Args:
        model_id: watsonx.ai model ID
        api_key: API key (or set WATSONX_API_KEY env var)
        project_id: Project ID (or set WATSONX_PROJECT_ID env var)

    Returns:
        Configured ComplianceJudge instance
    """
    import os

    api_key = api_key or os.environ.get("WATSONX_API_KEY")
    project_id = project_id or os.environ.get("WATSONX_PROJECT_ID")

    if not api_key or not project_id:
        raise ValueError("watsonx.ai requires API key and project ID")

    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai import Credentials

        credentials = Credentials(
            url="https://us-south.ml.cloud.ibm.com",
            api_key=api_key
        )

        model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id,
            params={
                "decoding_method": "greedy",
                "temperature": 0.0,
                "max_new_tokens": 1024
            }
        )

        def watsonx_generate(prompt: str) -> str:
            return model.generate_text(prompt)

        return ComplianceJudge(
            judge_model_fn=watsonx_generate,
            judge_model_name=f"watsonx/{model_id}"
        )
    except ImportError:
        raise ImportError("ibm_watsonx_ai package required. Install with: pip install ibm-watsonx-ai")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for compliance judge evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-as-Judge for Financial AI Compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate FSB consistency on trace files
  python -m harness.compliance_judge --input traces/*.jsonl --mode fsb

  # Evaluate multi-model consensus
  python -m harness.compliance_judge --input results/council.csv --mode consensus

  # Generate attestation report
  python -m harness.compliance_judge --input traces/*.jsonl --output attestation.json
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input file(s) with responses to evaluate"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["fsb", "sec", "consensus", "faithfulness"],
        default="fsb",
        help="Evaluation mode"
    )
    parser.add_argument(
        "--judge-model",
        default="granite3.3:latest",
        help="Model to use as judge (Ollama model name)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for attestation (JSON)"
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "watsonx"],
        default="ollama",
        help="LLM provider for judge"
    )

    args = parser.parse_args()

    # Create judge
    if args.provider == "ollama":
        judge = create_ollama_judge(args.judge_model)
    else:
        judge = create_watsonx_judge(args.judge_model)

    print(f"Compliance Judge initialized with {args.provider}/{args.judge_model}")
    print(f"Mode: {args.mode}")
    print(f"Input: {args.input}")

    # TODO: Implement file parsing and evaluation logic
    print("\nNote: Full CLI implementation requires trace file parsing.")
    print("Use the ComplianceJudge class directly for programmatic access.")


if __name__ == "__main__":
    main()
