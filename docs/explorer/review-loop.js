"use strict";

// Display recorded package results. Running the example itself is a local lab step.
(() => {
  const example = DFAH_REVIEW_EXAMPLE;
  const byId = id => document.getElementById(id);
  const labels = {
    tar_seq: "ordered-path agreement",
    gap: "decision–path gap",
    flags_per_100_cases: "flag rate"
  };
  const buttons = document.querySelectorAll("[data-review-candidate]");
  byId("review-policy").textContent = JSON.stringify(example.policy, null, 2);

  function renderCandidate(index) {
    const candidate = example.candidates[index];
    buttons.forEach(button => button.setAttribute("aria-pressed", String(Number(button.dataset.reviewCandidate) === index)));
    byId("review-dar").textContent = candidate.dar.toFixed(2);
    byId("review-tar").textContent = candidate.tar_seq.toFixed(2);
    byId("review-flags").textContent = `${candidate.flagged_groups} / ${candidate.groups}`;
    const verdict = byId("review-verdict");
    verdict.className = `verdict ${candidate.passed ? "good" : "bad"}`;
    verdict.textContent = candidate.passed
      ? "PASS — all checks meet the unchanged policy."
      : `FAIL — ${candidate.failed_checks.map(check => labels[check] || check).join(", ")} miss the policy.`;
    byId("review-explanation").textContent = candidate.passed
      ? "Each case now preserves both its decision and its ordered tool path. The supplied correction changes the adapter implementation; the cases, tools, and gate stay fixed."
      : "Both replays reach the same decision for each case. Reversing the two tool calls creates a review signal.";
    byId("review-caption").textContent = `${candidate.candidate} · ${candidate.episodes} episodes`;
    const rows = candidate.replays.map(replay => {
      const row = document.createElement("tr");
      [`${replay.case} / ${replay.replay}`, replay.decision, replay.tools.join(" → ")].forEach(value => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.appendChild(cell);
      });
      return row;
    });
    byId("review-replays").replaceChildren(...rows);
    byId("review-commitments").textContent = [
      `Package: dfah-bench ${example.package_version}`,
      `Source commit: ${example.source_commit}`,
      `Adapter: ${candidate.adapter_version}`,
      `Implementation SHA-256: ${candidate.implementation_sha256}`,
      `Manifest SHA-256: ${candidate.manifest_sha256}`,
      `Episode root SHA-256: ${candidate.episode_root_sha256}`
    ].join("\n\n");
  }

  buttons.forEach(button => button.addEventListener("click", () => renderCandidate(Number(button.dataset.reviewCandidate))));
  renderCandidate(0);
})();
