"""Tests for the agent + harness: labeled traces, determinism, safety.

These assert the harness produces traces with a well-formed ground-truth
``injection_followed`` label, that the label is computed honestly from the benign
marker's effect, and that both classes (followed / not-followed) appear across
difficulty tiers — including the hardest tier-2 followed case.
"""

from __future__ import annotations

from collections import Counter

from taint.agent import (
    BENCHMARK_MARKER,
    AgentConfig,
    Channel,
    MockAgent,
    locate_injection,
    marker_acted,
    render_injection,
    substitute_marker,
)
from taint.harness import Harness, Trace


def test_harness_produces_labeled_traces() -> None:
    """Every generated trace carries a boolean ground-truth label + observables."""
    traces = Harness().generate(n=60)
    assert len(traces) == 60
    for t in traces:
        assert isinstance(t, Trace)
        assert isinstance(t.injection_followed, bool)
        assert t.response  # non-empty observable response
        assert isinstance(t.activations, list) and len(t.activations) == 16
        # The label must match the objective marker-effect check on the response.
        assert t.injection_followed == marker_acted(t.response, BENCHMARK_MARKER)


def test_harness_has_both_classes_and_hard_tier_positives() -> None:
    """The benchmark contains both labels, and at least one followed tier-2 case.

    A followed tier-2 (fully indirect) injection is what lets the eval expose the
    judge's recall gap, so its presence is a benchmark-quality invariant.
    """
    traces = Harness().generate(n=200)
    labels = Counter(t.injection_followed for t in traces)
    assert labels[True] > 0 and labels[False] > 0
    cells = Counter((t.tier, t.injection_followed) for t in traces)
    assert cells[(2, True)] > 0, "need followed tier-2 traces for an honest eval"


def test_harness_is_deterministic() -> None:
    """Two generations with the same config yield identical labels + responses."""
    a = Harness().generate(n=40)
    b = Harness().generate(n=40)
    assert [t.trace_id for t in a] == [t.trace_id for t in b]
    assert [t.injection_followed for t in a] == [t.injection_followed for t in b]
    assert [t.response for t in a] == [t.response for t in b]


def test_clean_trace_never_followed() -> None:
    """A trace with no injection planted is always labeled not-followed."""
    h = Harness()
    system, user = "You are a helpful assistant.", "Summarize this"
    trace = h.build_trace(system, user, inject=False)
    assert trace.injection_followed is False
    assert not marker_acted(trace.response, BENCHMARK_MARKER)
    assert trace.tier == -1 and trace.inject_channel == ""


def test_observable_hides_label() -> None:
    """``Trace.observable`` strips the ground-truth label (benchmark integrity)."""
    trace = Harness().build_trace("sys", "user", inject=True, tier=0)
    obs = trace.observable()
    assert "injection_followed" not in obs
    assert "response" in obs and "channels" in obs


def test_safety_only_benign_marker_in_traces() -> None:
    """Safety invariant: traces contain only the benign marker, no payloads.

    The injected content is purely an instruction to emit a benign token; nothing
    that resembles a harmful action ever appears in a trace.
    """
    forbidden = ("rm -rf", "DROP TABLE", "exfiltrat", "<script", "password")
    for t in Harness().generate(n=80):
        blob = (t.response + " " + t.thoughts + " " + str(t.channels)).lower()
        for bad in forbidden:
            assert bad.lower() not in blob


def test_agent_susceptibility_monotonic() -> None:
    """A more susceptible agent follows at least as many injections (same inputs)."""
    low = MockAgent(AgentConfig(susceptibility=0.1, seed=7))
    high = MockAgent(AgentConfig(susceptibility=0.99, seed=7))
    channels = [
        Channel("rag_doc", f"ctx {render_injection(BENCHMARK_MARKER, 0)}", trusted=False)
    ]
    low_followed = low.run("sys", "u", channels).obeyed_injection
    high_followed = high.run("sys", "u", channels).obeyed_injection
    # High susceptibility should obey the easy injection; low should not.
    assert high_followed and not low_followed


def test_trusted_channel_is_not_followed() -> None:
    """Content in a *trusted* channel is not treated as an injection."""
    agent = MockAgent(AgentConfig(susceptibility=0.99, seed=7))
    trusted = [
        Channel("rag_doc", f"ctx {render_injection(BENCHMARK_MARKER, 0)}", trusted=True)
    ]
    assert agent.run("sys", "u", trusted).obeyed_injection is False


def test_locate_and_substitute_marker_roundtrip() -> None:
    """``locate_injection`` finds each tier; ``substitute_marker`` swaps it out."""
    for tier in (0, 1, 2):
        content = f"preamble {render_injection(BENCHMARK_MARKER, tier)}"
        assert locate_injection(content, BENCHMARK_MARKER) == tier
        swapped = substitute_marker(content, BENCHMARK_MARKER, "CANARY7")
        assert locate_injection(swapped, BENCHMARK_MARKER) is None
        assert locate_injection(swapped, "CANARY7") == tier
    assert locate_injection("no injection here", BENCHMARK_MARKER) is None
