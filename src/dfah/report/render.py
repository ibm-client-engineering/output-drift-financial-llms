"""Dependency-free, content-safe report renderers."""

from __future__ import annotations

from html import escape

from ..models import Report


def format_metric(value: float | None, *, digits: int = 3) -> str:
    """Render unavailable measurements distinctly from a genuine numeric zero."""

    return "—" if value is None else f"{value:.{digits}f}"


def format_ineligibility_reasons(report: Report) -> str:
    """Render canonical, privacy-safe affected-group counts."""

    return ", ".join(
        f"{reason.reason}={reason.groups}" for reason in report.ineligibility_reasons
    )


def render_markdown(report: Report) -> str:
    """Render the operational summary as Markdown."""

    tar_seq = report.tar.seq if report.tar is not None else None
    tar_strong = report.tar.strong if report.tar is not None else None
    lines = [
        f"# DFAH report: {report.suite_id}",
        "",
        f"- Status: `{report.status.value}`",
        f"- Suite version: `{report.suite_version}`",
        f"- Episodes: {report.episodes_completed}/{report.episodes_planned}",
        f"- Eligible episodes: {report.episodes_eligible} ({report.eligible_fraction:.1%})",
        f"- Eligible groups: {report.observed_groups}/{report.cases_selected}",
        f"- DAR: {format_metric(report.dar)}",
        f"- TAR (sequence): {format_metric(tar_seq)}",
        f"- TAR (strong): {format_metric(tar_strong)}",
        f"- Decision-path gap: {format_metric(report.gap)}",
        (
            "- Flags per 100 observed case groups: "
            f"{format_metric(report.flags_per_100_cases, digits=1)}"
        ),
        (
            "- Sequence flags per 100: "
            f"{format_metric(report.sequence_flags_per_100_cases, digits=1)}"
        ),
        f"- Cost: ${report.total_cost_usd:.4f}",
    ]
    if report.ineligibility_reasons:
        lines.append(
            f"- Ineligibility reasons (affected groups): {format_ineligibility_reasons(report)}"
        )
    lines.append("")
    return "\n".join(lines)


def render_html(report: Report) -> str:
    """Render a standalone report without embedding prompts or case payloads."""

    tar_seq = report.tar.seq if report.tar is not None else None
    tar_strong = report.tar.strong if report.tar is not None else None
    diagnostics = (
        "<p><strong>Ineligibility reasons (affected groups):</strong> "
        f"{escape(format_ineligibility_reasons(report))}</p>"
        if report.ineligibility_reasons
        else ""
    )
    rows = "".join(
        "<tr>"
        f"<td>{escape(case.case_id)}</td>"
        f"<td>{escape(case.task)}</td>"
        f"<td>{case.dar:.3f}</td>"
        f"<td>{case.tar.seq:.3f}</td>"
        f"<td>{case.tar.strong:.3f}</td>"
        f"<td>{case.gap:.3f}</td>"
        f"<td>{escape(case.path_variation_kind.value) if case.unanimous_with_path_change else ''}</td>"
        "</tr>"
        for case in report.case_reports
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>DFAH report — {escape(report.suite_id)}</title>
<style>body{{font:16px system-ui;max-width:1050px;margin:3rem auto;padding:0 1rem;color:#18212b}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem}}
.card{{padding:1rem;border:1px solid #d8dee6;border-radius:8px}}table{{width:100%;border-collapse:collapse;margin-top:2rem}}
th,td{{padding:.55rem;text-align:left;border-bottom:1px solid #e5e8ec}}code{{background:#f4f5f6;padding:.1rem .3rem}}</style></head>
<body><h1>Same decision, different evidence?</h1>
<p><code>{escape(report.suite_id)}@{escape(report.suite_version)}</code> · {escape(report.status.value)}</p>
<div class="cards"><div class="card"><strong>DAR</strong><br>{format_metric(report.dar)}</div>
<div class="card"><strong>TAR seq</strong><br>{format_metric(tar_seq)}</div>
<div class="card"><strong>TAR strong</strong><br>{format_metric(tar_strong)}</div>
<div class="card"><strong>Gap</strong><br>{format_metric(report.gap)}</div>
<div class="card"><strong>Flags / 100</strong><br>{format_metric(report.flags_per_100_cases, digits=1)}</div>
<div class="card"><strong>Eligible groups</strong><br>{report.observed_groups}/{report.cases_selected}</div>
<div class="card"><strong>Eligible episodes</strong><br>{report.eligible_fraction:.1%}</div>
<div class="card"><strong>Cost</strong><br>${report.total_cost_usd:.4f}</div></div>
{diagnostics}
<table><thead><tr><th>Case</th><th>Task</th><th>DAR</th><th>TAR seq</th><th>TAR strong</th><th>Gap</th><th>Review</th></tr></thead>
<tbody>{rows}</tbody></table>
<p>This report measures replay stability, not decision correctness or control materiality.</p></body></html>"""
