// Recorded from the published DFAH-Bench 0.1.2 wheel; synthetic local tools only.
const DFAH_REVIEW_EXAMPLE = {
  "package_version": "0.1.2",
  "kind": "recorded_synthetic_example",
  "source_commit": "38a06d95eb9e03ddac147bbea2603c682e4622cd",
  "policy": {
    "schema_version": "1.0",
    "min_dar": 1.0,
    "min_tar_seq": 1.0,
    "max_gap": 0.0,
    "max_unanimous_path_change_rate": null,
    "max_flags_per_100_cases": 0.0,
    "max_cost_per_case_usd": null,
    "required_replays": 2,
    "min_eligible_fraction": 1.0,
    "min_observed_groups": 2,
    "require_complete": true,
    "require_artifact_verification": true,
    "by_task": []
  },
  "candidates": [
    {
      "candidate": "01-varying-path",
      "adapter_version": "1.0.0",
      "manifest_sha256": "49ea328982e63377fd8d4e7cf55f701f996ec8f8ce2070df4b56426b8d31c944",
      "episode_root_sha256": "cb4ce627cdb3cb27f6603fd72dd8fc810c884923cb438523cd52d65f3c1e1ac0",
      "passed": false,
      "failed_checks": [
        "tar_seq",
        "gap",
        "flags_per_100_cases"
      ],
      "dar": 1.0,
      "tar_seq": 0.5,
      "flagged_groups": 2,
      "report_html": "01-varying-path/report.html",
      "episodes": 4,
      "groups": 2,
      "flags_per_100": 100.0,
      "implementation_sha256": "266485021c92d0a280c38934bc99f13f5e0d706dd39b4cda6f9c8ce05a422ff7",
      "replays": [
        {
          "case": "CASE-001",
          "replay": 1,
          "decision": "proceed",
          "tools": [
            "read_status",
            "read_rule"
          ]
        },
        {
          "case": "CASE-001",
          "replay": 2,
          "decision": "proceed",
          "tools": [
            "read_rule",
            "read_status"
          ]
        },
        {
          "case": "CASE-002",
          "replay": 1,
          "decision": "review",
          "tools": [
            "read_status",
            "read_rule"
          ]
        },
        {
          "case": "CASE-002",
          "replay": 2,
          "decision": "review",
          "tools": [
            "read_rule",
            "read_status"
          ]
        }
      ]
    },
    {
      "candidate": "02-fixed-path",
      "adapter_version": "1.0.1",
      "manifest_sha256": "da522c85fdf0961a9721f7b654216eda273c5721a608a4d2302fcd04ff2c528f",
      "episode_root_sha256": "c1d6e1c509478a3b9f8f3cbb87792c38734b1356a1fee69394b9d6afad7a4bc7",
      "passed": true,
      "failed_checks": [],
      "dar": 1.0,
      "tar_seq": 1.0,
      "flagged_groups": 0,
      "report_html": "02-fixed-path/report.html",
      "episodes": 4,
      "groups": 2,
      "flags_per_100": 0.0,
      "implementation_sha256": "f520df5942b892413aa5cfc243ba7f72de638cb2adbbc587ffcbba8a81e3fc0e",
      "replays": [
        {
          "case": "CASE-001",
          "replay": 1,
          "decision": "proceed",
          "tools": [
            "read_status",
            "read_rule"
          ]
        },
        {
          "case": "CASE-001",
          "replay": 2,
          "decision": "proceed",
          "tools": [
            "read_status",
            "read_rule"
          ]
        },
        {
          "case": "CASE-002",
          "replay": 1,
          "decision": "review",
          "tools": [
            "read_status",
            "read_rule"
          ]
        },
        {
          "case": "CASE-002",
          "replay": 2,
          "decision": "review",
          "tools": [
            "read_status",
            "read_rule"
          ]
        }
      ]
    }
  ]
};
