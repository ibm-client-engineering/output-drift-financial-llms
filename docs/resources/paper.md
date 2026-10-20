# Research papers and materials

The studies ask what a repeated answer leaves unmeasured: changes in tool use,
missing records, and decisions without supporting evidence.

## 2026 workshop papers

| Paper | Venue and status | Materials |
|---|---|---|
| [The Determinism–Faithfulness Assurance Harness (DFAH): Evidence Before Execution in Financial Agent Gates](https://openreview.net/forum?id=VWQqIgrKw4) | Accepted poster, AI4F at ACM ICAIF 2026 | [Banking findings and scope](../dfah/v3-study.md); [Lab 10](../lab-10/README.md) |
| [When Replay Agreement Misleads: Capture Loss and Cancellation in Local Agents](https://openreview.net/forum?id=njEKVxonV2) | Accepted poster, SLM-Agents at NeurIPS 2026 | [Original supplement](https://openreview.net/attachment?id=njEKVxonV2&name=supplementary_material); [Lab 11](../lab-11/README.md) |

The [workshop guide](../dfah/workshop-studies.md) explains the relationship
between these papers, their shared observations, and the available code and data.

## Preprint and earlier work

| Research record | Focus |
|---|---|
| [DFAH-Bench — arXiv:2607.20491](https://arxiv.org/abs/2607.20491) | Replay, capture, and evidence for financial-agent actions |
| [Replayable Financial Agents — arXiv:2601.15322](https://arxiv.org/abs/2601.15322) | Earlier tool-using agent study; ICLR 2026 FinAI Workshop |
| [LLM Output Drift — arXiv:2511.07585](https://arxiv.org/abs/2511.07585) | Earlier output consistency study; AI4F Workshop 2025 |

Use the title and version shown on the corresponding record when citing it.
The workshop titles identify specific parts of the research, not acceptance
of the entire DFAH-Bench preprint by either workshop.

## What can I run?

| Question | Material | What it does |
|---|---|---|
| Do decisions and paths repeat? | [Lab 8](../lab-8/README.md), `make reproduce-paper` | Package walkthrough or historical replay recomputation |
| Which case changed? | [Lab 9](../lab-9/README.md) | Review a saved path change with a fixed policy |
| Did a proposed call complete? | [Lab 10](../lab-10/README.md) | Inspect synthetic invocation states |
| Can omissions preserve agreement? | [Lab 11](../lab-11/README.md) | Run a small, offline capture-loss example |
| Can I recompute the SLM diagnostic? | [Supplement instructions](../dfah/workshop-studies.md#inspect-the-released-supplement) | Check the original synthetic projections with its offline verifiers |
| Can I rerun the banking study here? | [Banking guide](../dfah/v3-study.md) | Read the methods and findings; hosted study inputs are not distributed here |

The historical replay fixture and the SLM supplement support different
checks. Teaching examples are not additional empirical results. Running a
hosted model again would collect new observations rather than reconstruct
the original service.

## Cite the workshop papers

```bibtex
@misc{khatchadourian2026ai4f,
  author = {Khatchadourian, Raffi},
  title = {The Determinism--Faithfulness Assurance Harness (DFAH): Evidence Before Execution in Financial Agent Gates},
  year = {2026},
  howpublished = {AI4F Workshop, ACM ICAIF 2026},
  url = {https://openreview.net/forum?id=VWQqIgrKw4}
}

@misc{khatchadourian2026slmagents,
  author = {Khatchadourian, Raffi},
  title = {When Replay Agreement Misleads: Capture Loss and Cancellation in Local Agents},
  year = {2026},
  howpublished = {SLM-Agents Workshop, NeurIPS 2026},
  url = {https://openreview.net/forum?id=njEKVxonV2}
}
```

The repository's [citation file](https://github.com/ibm-client-engineering/output-drift-financial-llms/blob/main/CITATION.cff)
provides the software and preprint citation information.
