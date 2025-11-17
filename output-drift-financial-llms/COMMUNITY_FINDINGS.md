# Community Findings

> **Work in Progress** - Community members are actively experimenting with the framework. Results below are preliminary.

## Independent Validation (November 2024)

**Paul Merrison** ([paul@paulmerrison.io](mailto:paul@paulmerrison.io), [FINOS](https://www.finos.org/)) tested 6 models (3B-20B) and found that determinism is **model-specific**, not size-based:

- **Qwen2.5-7B**: 100% (confirms paper)
- **Gemma2-9B**: 100% (new Tier 1 candidate)
- **Llama3.1-8B**: 62.5% (engineering matters)
- **Mistral-7B**: Task-dependent (33% RAG, 100% SQL)

Key insight: Architecture and training approach matter more than parameter count alone for deterministic behavior in regulated applications.
