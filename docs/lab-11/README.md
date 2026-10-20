# Lab 11: When Matching Records Lose Calls

Three replays can agree perfectly because all three records omit the same
call. This 15-minute lab makes that failure visible, then checks each record
against a separate reference of the dispatched calls.

The exercise uses three constructed replays, Python 3.10 or later, and the
standard library. It needs no model, API key, or package installation.

## 1. Run the example

From the current repository checkout:

```bash
python3 scripts/workshop/capture_loss_demo.py
```

Each original path contains `read_balance` followed by `read_policy`, with
the same arguments and results. The example creates four views of those
records:

| View | Recorded-path agreement | Reference mismatches | Missing records |
|---|---:|---:|---:|
| Unchanged | 1.0 | 0 | 0 |
| First call deleted from every record | 1.0 | 3 | 0 |
| All calls deleted from every record | 1.0 | 3 | 0 |
| One record unavailable | `null` | 0 among the available records | 1 |

## 2. Explain the two deletion results

Agreement compares recorded replays with one another. The same deletion in
every replay leaves them identical. The reference comparison instead asks
whether each record matches the dispatched calls for that replay. It detects
both deletions.

The reference must preserve the action being checked. Two streams that omit
the same event would not expose it through this comparison. The example
does not authenticate its reference or observe effects outside these calls.

## 3. Separate missing, empty, and altered records

`null` means one replay has no usable record, so this example declines to
report agreement for the planned group. It does not turn the missing record
into an empty list.

The separate `observed_empty_reference` output compares three observed empty
records with three empty references. Agreement is 1.0 and there are no
mismatches. An empty list by itself therefore cannot tell us whether no calls
ran or calls were lost; the corresponding execution evidence matters.

Try changing an argument in every recorded path while leaving its reference
unchanged. Repeated-path agreement can still be perfect, while the reference
comparison detects the change.

## 4. Connect the exercise to the studies

The [SLM-Agents paper at NeurIPS 2026](https://openreview.net/forum?id=njEKVxonV2)
reports a separate diagnostic with 48 executions, 16 groups, and 144 record
views. Its OpenReview supplement contains the frozen plan and offline
verifiers. This teaching example explains the measurement problem; it does
not reproduce those observations.

The [AI4F paper at ACM ICAIF 2026](https://openreview.net/forum?id=VWQqIgrKw4)
adds the next question: even when a record is present, does its evidence
support the permitted action, and does the task finish?

- [Workshop studies and released materials](../dfah/workshop-studies.md)
- [Lab 9: inspect a changed path](../lab-9/README.md)
- [Lab 10: distinguish a proposal from a completed invocation](../lab-10/README.md)
