# Maintainer release process

Package publication is separate from the research and documentation releases.
Do not create a release tag until the package checks pass and the project
owner has confirmed the applicable open-source, legal, and patent clearances,
including the repository-level `LICENSE` and `NOTICE`.

## One-time PyPI setup

Configure a pending Trusted Publisher with these exact values:

| Field | Value |
|---|---|
| PyPI project | `dfah-bench` |
| GitHub owner | `ibm-client-engineering` |
| Repository | `output-drift-financial-llms` |
| Workflow | `dfah-release.yml` |
| Environment | `pypi` |

Create the `pypi` GitHub environment with required maintainer approval and
appropriate deployment protection before the first release. A package-name
availability check is not a reservation; the pending publisher must be
registered before the tag is pushed.

## Release boundary

For package version `X.Y.Z`, the release tag is `dfah-vX.Y.Z`. The tag must
match the static project version in `pyproject.toml` and identify a commit
already on `main`; the release workflow rejects any other pairing.

The tag workflow then calls the reusable `dfah-package.yml` workflow from that
same commit. Publication requires all of these checks on the tagged source:

- formatting, lint, strict type checks, and package tests on Python 3.10–3.13;
- at least 85% coverage of the metrics and parser modules;
- mandatory official Every Eval Ever schema validation on Python 3.12;
- distribution builds across the Python matrix;
- a dedicated Python 3.11 wheel and source-distribution build, metadata checks,
  and archive scans for internal paths and credential material; and
- installation of that wheel, dependency checks, and CLI/plugin, replay,
  analysis, and interchange-export smoke tests outside the source checkout.

Only the dedicated distribution job uploads `dfah-distributions`, after its
checks succeed. The protected publish job downloads that artifact from the
same workflow run and publishes those exact files through PyPI Trusted
Publishing after environment approval. There is no rebuild between this
artifact's verification and publication. Passing a separate branch or PR
workflow does not substitute for the tag's own validation.

Before tagging, confirm that:

1. `pyproject.toml`, `CITATION.cff`, and `CHANGELOG.md` name the same version;
2. the package CI matrix and distribution job pass on the intended commit;
3. the protected `pypi` environment and pending publisher use the values above;
4. the release commit contains no unrelated research, result, or site changes.

After pushing the tag, confirm that its release run completes the full
validation matrix before approving the `pypi` deployment. Local checks cannot
verify the hosted Python matrix, environment protections, or Trusted Publisher
configuration.
