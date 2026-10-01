"use strict";

// Display recorded package results. Running the example itself is a local lab step.
(() => {
  const example = DFAH_REVIEW_EXAMPLE;
  const byId = id => document.getElementById(id);
  const buttons = document.querySelectorAll("[data-review-candidate]");
  byId("review-policy").textContent = JSON.stringify(example.policy, null, 2);

  function renderCandidate(index) {
    const candidate = example.candidates[index];
    buttons.forEach(button => button.setAttribute("aria-pressed", String(Number(button.dataset.reviewCandidate) === index)));
    byId("review-dar").textContent = `${Math.round(candidate.dar * 100)}%`;
    byId("review-tar").textContent = `${Math.round(candidate.tar_seq * 100)}%`;
    byId("review-flags").textContent = `${candidate.flagged_groups} / ${candidate.groups}`;
    const verdict = byId("review-verdict");
    verdict.className = `verdict ${candidate.passed ? "good" : "bad"}`;
    verdict.textContent = candidate.passed
      ? "Policy passes — decisions and tool paths repeat."
      : "Policy flags this run — the tool order changes across replays.";
    byId("review-explanation").textContent = candidate.passed
      ? "The second supplied implementation keeps the same cases, tools, and policy. Both decisions and ordered paths repeat."
      : "Both replays reach the same decision for each case, but the tool order changes.";
    byId("review-caption").textContent = `${candidate.passed ? "Consistent" : "Varying"} tool path · ${candidate.episodes} recorded runs`;
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
