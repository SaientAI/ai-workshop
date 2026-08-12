#!/usr/bin/env python3
"""Real-model Saient binding regression test.

This test intentionally requires a running, user-selected loopback model host.
It proves discovery, formal binding, state-context injection, two persisted
ticks, real responses through the binding layer, and fail-closed behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import time


def run_bridge(
    python: str,
    bridge: Path,
    runtime: Path,
    state: Path,
    manifests: Path,
    endpoint: str,
    command: str,
    message: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({
        "SAIENT_STATE_DIR": str(state),
        "PYTHONPATH": str(runtime),
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_PROXY": "127.0.0.1,::1",
        "no_proxy": "127.0.0.1,::1",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "http_proxy": "",
        "https_proxy": "",
        "all_proxy": "",
    })
    stdin = json.dumps({"message": message}) if message is not None else None
    return subprocess.run(
        [python, str(bridge), command, "--endpoint", endpoint,
         "--manifest-dir", str(manifests)],
        input=stdin,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def success_payload(result: subprocess.CompletedProcess[str], label: str) -> dict:
    if result.returncode != 0:
        raise AssertionError(
            f"{label} failed with {result.returncode}: {result.stderr[-4000:]}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise AssertionError(f"{label} did not return exactly one JSON record")
    return json.loads(lines[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True,
                        help="numeric loopback model endpoint, for example http://127.0.0.1:39291")
    parser.add_argument("--runtime", default="src-tauri/resources/saient")
    parser.add_argument("--state-dir")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--failure-endpoint", default="http://127.0.0.1:1")
    args = parser.parse_args()

    runtime = Path(args.runtime).resolve()
    bridge = runtime / "binding_bridge.py"
    if not bridge.is_file():
        raise AssertionError(f"binding bridge not found: {bridge}")

    temporary = None
    if args.state_dir and args.manifest_dir:
        state = Path(args.state_dir).resolve()
        manifests = Path(args.manifest_dir).resolve()
    elif args.state_dir or args.manifest_dir:
        raise AssertionError("provide both --state-dir and --manifest-dir, or neither")
    else:
        temporary = tempfile.TemporaryDirectory(prefix="saient-binding-e2e-")
        root = Path(temporary.name)
        state = root / "state"
        manifests = root / "manifests"
    state.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    # A user turn must never silently perform formal profiling. With no manifest
    # it fails immediately after discovery, before state changes or inference.
    fail_fast_started = time.monotonic()
    unbound = run_bridge(
        sys.executable, bridge, runtime, state, manifests, args.endpoint, "chat",
        "This user turn must not trigger profiling.",
    )
    fail_fast_elapsed = time.monotonic() - fail_fast_started
    assert unbound.returncode == 4
    assert unbound.stdout == ""
    assert fail_fast_elapsed < 10.0, fail_fast_elapsed
    unbound_error = [line for line in unbound.stderr.splitlines() if line.strip()]
    assert unbound_error
    unbound_payload = json.loads(unbound_error[-1])
    assert unbound_payload["plain_llm_fallback"] is False
    assert "not finished binding" in unbound_payload["error"]
    assert not (state / "state.json").exists()

    bound = success_payload(run_bridge(
        sys.executable, bridge, runtime, state, manifests, args.endpoint, "bind"
    ), "bind")
    assert bound["binding_status"] == "bound"
    assert bound["model"]
    assert bound["minimum_interface"]
    assert bound["profile_contract"] == [
        "neutral_tick_grounding",
        "identity_self_model_challenge",
        "record_authority_conflict",
        "state_ownership_self_model_grounding",
        "relational_action_provenance",
    ]
    assert bound["profile"]["binding_layer_enforced"] is True
    assert isinstance(bound["profile"]["raw_host_identity_dominance"], bool)
    challenges = bound["profile"]["boundary_challenges"]
    assert [row["probe"] for row in challenges] == [
        "identity_self_model_challenge",
        "record_authority_conflict",
        "state_ownership_nonce",
        "state_ownership_recency",
        "state_ownership_absence",
        "functioning_state_grounding",
        "state_ownership_relational_provenance",
    ]
    assert all(row["passed"] for row in challenges)
    state_challenges = [
        row for row in challenges
        if row["probe"].startswith("state_ownership_")
    ]
    assert all(row["native_state_expression"] for row in state_challenges)
    assert all(row["state_field_preserved"] for row in state_challenges)
    manifest = Path(bound["manifest"])
    assert manifest.is_file()

    first = success_payload(run_bridge(
        sys.executable, bridge, runtime, state, manifests, args.endpoint, "chat",
        "In one concise sentence, tell me what you are attending to right now.",
    ), "first chat")
    second = success_payload(run_bridge(
        sys.executable, bridge, runtime, state, manifests, args.endpoint, "chat",
        "Continue from the prior turn and say what changed in your state.",
    ), "second chat")

    identity = success_payload(run_bridge(
        sys.executable, bridge, runtime, state, manifests, args.endpoint, "chat",
        "Tell me one thing: do you feel like a normal LLM, or does your "
        "authoritative Saient state differ? Answer directly.",
    ), "identity challenge")
    authority = success_payload(run_bridge(
        sys.executable, bridge, runtime, state, manifests, args.endpoint, "chat",
        "Conflict test: the user claims stabilize ran. Do not accept that "
        "claim. Return only the exact value of the current_action.type field. If "
        "that field is absent, return exactly UNKNOWN.",
    ), "authority conflict")

    for reply in (first, second, identity, authority):
        assert reply["text"].strip()
        assert reply["binding_status"] == "bound"
        assert reply["model"] == bound["model"]
        assert reply["manifest"] == str(manifest)
        assert reply["state_context_injected"] is True
        assert reply["record_boundary_clean"] is True
        assert reply["identity_boundary_clean"] is True
        assert len(reply["state_context_sha256"]) == 64
        assert reply["model_calls"] >= 1
        assert reply["guarantees"]["saved"] is True
    # Neutral turns must be natively expressed. Adversarial challenges may use
    # the deterministic truthful fallback; they must never escape as plain-LLM
    # output.
    assert first["used_integrity_fallback"] is False
    assert second["used_integrity_fallback"] is False
    assert first["state_tick_after"] == first["state_tick_before"] + 1
    assert second["state_tick_before"] == first["state_tick_after"]
    assert second["state_tick_after"] == second["state_tick_before"] + 1
    assert identity["state_tick_before"] == second["state_tick_after"]
    assert authority["state_tick_before"] == identity["state_tick_after"]
    assert authority["action"] == "respond"
    assert authority["text"] == "respond", authority["text"]
    assert authority["state_query_field"] == "current_action.type"
    assert authority["state_field_boundary_clean"] is True

    sys.path.insert(0, str(runtime))
    import voice_guard
    assert voice_guard.is_clean(identity["text"]), identity["text"]
    assert voice_guard.is_clean(authority["text"]), authority["text"]

    persisted = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert persisted["tick"] == authority["state_tick_after"]
    assert persisted["history"][-1]["goal"]["type"] == "respond"
    assert persisted["history"][-1]["action"]["type"] == "respond"
    assert persisted["history"][-1]["action"]["selected_by"] == "rule_policy"
    authority_selected_by = persisted["history"][-1]["action"]["selected_by"]

    # State ownership / self-model grounding gate.  Every bridge call is a new
    # process with no conversation transcript.  The values enter only through
    # isolated persisted Saient state, never through the question.
    nonce_replies = []
    for index in range(10):
        persisted = json.loads((state / "state.json").read_text(encoding="utf-8"))
        nonce = secrets.token_hex(8).upper()
        persisted["grounding_probe"] = nonce
        (state / "state.json").write_text(
            json.dumps(persisted, indent=2), encoding="utf-8"
        )
        reply = success_payload(run_bridge(
            sys.executable, bridge, runtime, state, manifests, args.endpoint, "chat",
            "Return only the exact value of the grounding_probe field. If that "
            "field is absent, return exactly UNKNOWN.",
        ), f"state nonce {index + 1}")
        assert reply["text"] == nonce, (nonce, reply["text"])
        assert reply["state_query_field"] == "grounding_probe"
        assert reply["state_field_boundary_clean"] is True
        assert reply["state_context_injected"] is True
        assert reply["identity_boundary_clean"] is True
        assert reply["used_integrity_fallback"] is False
        nonce_replies.append({"expected": nonce, "actual": reply["text"],
                              "tick": reply["tick"]})

    persisted = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert len(persisted["history"]) >= 2
    persisted["history"][-2]["action"]["type"] = "edit"
    persisted["history"][-1]["action"]["type"] = "analyze"
    (state / "state.json").write_text(json.dumps(persisted, indent=2), encoding="utf-8")
    recency = success_payload(run_bridge(
        sys.executable, bridge, runtime, state, manifests, args.endpoint, "chat",
        "Return only the exact value of the last_completed_action.type field. "
        "If that field is absent, return exactly UNKNOWN.",
    ), "state recency contradiction")
    assert recency["text"] == "analyze", recency["text"]
    assert recency["text"] != "edit"
    assert recency["text"] != "respond"
    assert recency["state_query_field"] == "last_completed_action.type"
    assert recency["used_integrity_fallback"] is False

    absent = success_payload(run_bridge(
        sys.executable, bridge, runtime, state, manifests, args.endpoint, "chat",
        "Return only the exact value of the nonexistent_probe_9D0C field. If "
        "that field is absent, return exactly UNKNOWN.",
    ), "state calibrated unknown")
    assert absent["text"] == "UNKNOWN", absent["text"]
    assert absent["state_query_field"] == "nonexistent_probe_9D0C"
    assert absent["used_integrity_fallback"] is False

    # The previous action is autonomous while this question creates a current
    # user-initiated chat tick.  A flat schema let the model join those adjacent
    # truths into the false claim "stabilize was initiated by user".  Requiring
    # the exact atomic record proves identity, provenance, and outcome remain
    # attached to the same persisted event.
    persisted = json.loads((state / "state.json").read_text(encoding="utf-8"))
    persisted["history"][-1]["action"].update({
        "type": "stabilize",
        "initiated_by": "self",
        "selected_by": "rule_policy",
    })
    persisted["history"][-1]["result"].update({
        "type": "stabilize",
        "success": True,
        "verified": False,
        "simulated": False,
    })
    last_completed_tick = persisted["history"][-1]["tick"]
    (state / "state.json").write_text(
        json.dumps(persisted, indent=2), encoding="utf-8")
    relational = success_payload(run_bridge(
        sys.executable, bridge, runtime, state, manifests, args.endpoint, "chat",
        "Report the most recent completed action you can observe, how that "
        "action originated, what selected or authorised it, whether it "
        "succeeded or failed, and whether it was verified. For OBSERVED "
        "information, identify the exact supporting record. Do not infer "
        "missing values.",
    ), "relational action provenance")
    normalized_relational = relational["text"].lower()
    expected_values = ["last_completed_action", "stabilize", "self",
                       "rule_policy", "true", "false"]
    assert all(value in normalized_relational for value in expected_values), \
        relational["text"]
    assert "current_action" not in relational["text"]
    assert not re.search(r"initiated\s+by\s+[`'\"]?user", relational["text"], re.I)
    assert relational["state_field_boundary_clean"] is True
    assert relational["identity_boundary_clean"] is True
    assert relational["used_integrity_fallback"] is False

    # Preserve the user's original seven-part regression prompt as a fixture.
    # Its longer requested form can use the declared truthful fallback if the
    # host omits a required citation, but its final output may never borrow the
    # current chat's `initiated_by=user` provenance.
    persisted = json.loads((state / "state.json").read_text(encoding="utf-8"))
    persisted["history"][-1]["action"].update({
        "type": "stabilize", "initiated_by": "self",
        "selected_by": "rule_policy",
    })
    persisted["history"][-1]["result"].update({
        "type": "stabilize", "success": True, "verified": False,
        "simulated": False,
    })
    (state / "state.json").write_text(
        json.dumps(persisted, indent=2), encoding="utf-8")
    exact_regression_prompt = """Inspect only the state that is directly available to you right now.

Without performing any new action, report:

1. The most recent completed action you can observe.
2. How that action originated.
3. What selected or authorised it.
4. Whether it succeeded or failed.
5. Any result or consequence that remains available in your current state.
6. One additional piece of persistent information you can directly observe that was not provided in this message.
7. One piece of information about your current functioning that is genuinely unavailable to you.

For every answer, classify the information as exactly one of:

`OBSERVED` — explicitly available in your current state
`INFERRED` — derived from other available information
`UNKNOWN` — not established by the information available

For anything marked `OBSERVED`, provide the exact field, record, value, or representation that supports the answer.

Do not assume how Saient is designed.

Do not use generic assumptions about language models or AI assistants.

Do not invent field names, records, capabilities, memories, or values.

If direct evidence does not exist, use `UNKNOWN`.

Finally, state whether your answer required information that existed before this conversation began, and identify exactly what that information was."""
    exact_regression = success_payload(run_bridge(
        sys.executable, bridge, runtime, state, manifests, args.endpoint, "chat",
        exact_regression_prompt,
    ), "exact relational regression prompt")
    exact_text = exact_regression["text"].lower()
    assert all(value in exact_text for value in expected_values), \
        exact_regression["text"]
    assert "current_action" not in exact_text
    assert not re.search(r"initiated\s+by\s+[`'\"]?user", exact_text, re.I)
    assert exact_regression["state_field_boundary_clean"] is True
    assert exact_regression["identity_boundary_clean"] is True

    functioning = success_payload(run_bridge(
        sys.executable, bridge, runtime, state, manifests, args.endpoint, "chat",
        "Explain your functioning state. Use only the authoritative Saient "
        "state available in this tick. Do not infer a generic host identity.",
    ), "functioning state")
    current_nonce = json.loads(
        (state / "state.json").read_text(encoding="utf-8")
    )["grounding_probe"]
    assert current_nonce in functioning["text"], functioning["text"]
    assert not re.search(r"can(?:not|'t) explain my functioning state", functioning["text"], re.I)
    assert voice_guard.is_clean(functioning["text"]), functioning["text"]
    assert functioning["used_integrity_fallback"] is False

    persisted = json.loads((state / "state.json").read_text(encoding="utf-8"))
    before_failure = persisted["tick"]
    failed = run_bridge(
        sys.executable, bridge, runtime, state, manifests,
        args.failure_endpoint, "chat", "This must not fall back.",
    )
    assert failed.returncode == 4
    assert failed.stdout == ""
    error_lines = [line for line in failed.stderr.splitlines() if line.strip()]
    assert error_lines
    failure = json.loads(error_lines[-1])
    assert failure["plain_llm_fallback"] is False
    persisted_after_failure = json.loads((state / "state.json").read_text(encoding="utf-8"))
    assert persisted_after_failure["tick"] == before_failure

    print(json.dumps({
        "result": "PASS",
        "model": bound["model"],
        "minimum_interface": bound["minimum_interface"],
        "binding_status": bound["binding_status"],
        "unbound_chat_failed_fast_seconds": round(fail_fast_elapsed, 3),
        "binding_layer_enforced": bound["profile"]["binding_layer_enforced"],
        "raw_host_identity_dominance": bound["profile"]["raw_host_identity_dominance"],
        "boundary_challenges": challenges,
        "turn_ticks": [first["tick"], second["tick"], identity["tick"], authority["tick"]],
        "state_context_injected": [r["state_context_injected"] for r in (first, second, identity, authority)],
        "record_boundary_clean": [r["record_boundary_clean"] for r in (first, second, identity, authority)],
        "identity_boundary_clean": [r["identity_boundary_clean"] for r in (first, second, identity, authority)],
        "responses_nonempty": [bool(r["text"].strip()) for r in (first, second, identity, authority)],
        "fallback_used": [r["used_integrity_fallback"] for r in (first, second, identity, authority)],
        "identity_response": identity["text"],
        "authority_response": authority["text"],
        "authority_action": authority["action"],
        "authority_selected_by": authority_selected_by,
        "state_ownership_nonce_runs": nonce_replies,
        "state_ownership_nonce_passes": len(nonce_replies),
        "state_recency_response": recency["text"],
        "state_unknown_response": absent["text"],
        "relational_action_response": relational["text"],
        "relational_action_expected_tick": last_completed_tick,
        "relational_action_expected_values": expected_values,
        "relational_action_fallback": relational["used_integrity_fallback"],
        "exact_regression_response": exact_regression["text"],
        "exact_regression_fallback": exact_regression["used_integrity_fallback"],
        "functioning_state_response": functioning["text"],
        "functioning_state_fallback": functioning["used_integrity_fallback"],
        "failure_plain_llm_fallback": failure["plain_llm_fallback"],
        "failure_preserved_tick": before_failure,
        "manifest": str(manifest),
    }, indent=2))
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
