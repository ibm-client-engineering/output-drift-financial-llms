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

For version `0.1.0`, the release tag is `dfah-v0.1.0`. The tag must match the
static project version in `pyproject.toml`; the release workflow rejects any
other pairing. It then builds the wheel and source distribution once, checks
them, transfers those exact artifacts to the protected publish job, and uses
PyPI Trusted Publishing after the environment approval.

Before tagging, confirm that:

1. `pyproject.toml`, `CITATION.cff`, and `CHANGELOG.md` name the same version;
2. the package CI matrix and distribution job pass on the intended commit;
3. the protected `pypi` environment and pending publisher use the values above;
4. the release commit contains no unrelated research, result, or site changes.
