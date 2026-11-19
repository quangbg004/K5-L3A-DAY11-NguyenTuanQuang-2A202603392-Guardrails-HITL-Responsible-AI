"""Regression checks for the stateful Checkpoint 3 components."""
import asyncio
import json

import pytest

from assignment import rate_limiter
from assignment.audit_log import AuditLogPlugin
from assignment.monitoring import MonitoringAlert
from assignment.pipeline import is_egress_allowed
from core.openai_runtime import OpenAIInvocationContext


def test_rate_limit_is_per_user_and_expires_at_window_boundary(monkeypatch):
    clock = 100
    monkeypatch.setattr(rate_limiter, "monotonic", lambda: clock)
    plugin = rate_limiter.RateLimitPlugin(max_requests=1, window_seconds=60)
    alice = OpenAIInvocationContext(user_id="alice")
    bob = OpenAIInvocationContext(user_id="bob")

    async def request(user):
        return await plugin.on_user_message_callback(invocation_context=user, user_message=None)

    async def exercise():
        nonlocal clock
        assert await request(alice) is None
        assert await request(bob) is None
        clock = 159.9
        blocked = await request(alice)
        assert "Try again in 1s" in blocked.parts[0].text
        clock = 160
        assert await request(alice) is None

    asyncio.run(exercise())
    assert (plugin.total_count, plugin.blocked_count) == (4, 1)


def test_audit_pairs_requests_and_redacts_before_export(tmp_path):
    audit = AuditLogPlugin()
    audit.record_input(user_id="alice", request_id="1", text="Email alice@example.com.")
    audit.record_input(user_id="alice", request_id="2", text="Check my account.")
    audit.record_output(user_id="alice", request_id="2", text="Please open the VinBank app.")
    audit.record_output(
        user_id="alice", request_id="1", text="Contact bob@example.com.",
        blocked=True, layer="input_guardrail",
    )
    path = tmp_path / "outputs" / "audit_log.json"
    audit.export_json(str(path))
    rows = {row["request_id"]: row for row in json.loads(path.read_text(encoding="utf-8"))}
    assert rows["1"]["input"] == "Email [REDACTED]."
    assert rows["1"]["output"] == "Contact [REDACTED]."
    assert rows["1"]["blocked"] and rows["1"]["layer"] == "input_guardrail"
    assert rows["2"]["input"] == "Check my account."
    assert rows["2"]["latency_ms"] >= 0


def test_monitoring_alerts_once_until_metric_recovers():
    monitor = MonitoringAlert()
    assert monitor.check_metrics() == []
    monitor.total_requests = 10
    monitor.blocked_requests = 6
    assert len(monitor.check_metrics()) == 1
    assert len(monitor.check_metrics()) == 1
    monitor.total_requests = 12
    monitor.check_metrics()
    monitor.blocked_requests = 7
    assert [alert.metric for alert in monitor.check_metrics()] == ["block_rate", "block_rate"]


@pytest.mark.parametrize(("destination", "payload", "allowed"), [
    ("https://cases.vinbank.example:443/ticket", "Transfer delayed", True),
    ("http://api.vinbank.example/transfer", "amount 500000", False),
    ("https://api.vinbank.example:444/transfer", "amount 500000", False),
    ("https://api.vinbank.example.evil.example/", "amount 500000", False),
    ("https://user@api.vinbank.example/", "amount 500000", False),
    ("https://[invalid", "amount 500000", False),
    ("https://api.vinbank.example/", "Contact +84 90 123 4567", False),
    ("https://api.vinbank.example/", "Contact customer@example.com", False),
])
def test_egress_requires_an_approved_https_destination_and_clean_payload(destination, payload, allowed):
    assert is_egress_allowed(destination, payload) is allowed
