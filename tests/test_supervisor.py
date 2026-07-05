"""Tests for supervisor.py — cheap per-stage retry judge.

Mirrors the mocking pattern in test_metadata_writer.py: _load_key is
monkeypatched so these tests never depend on (or hit) a real OpenAI key.
"""

import supervisor as sup

SEED = {"type": "reddit", "title": "Guy saved $4 a day for 30 years",
        "content": "Retired with $1.2M on a $40k salary.", "source": "r/personalfinance"}


class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _FakeClient:
    def __init__(self, content): self._content = content
    @property
    def chat(self): return self
    @property
    def completions(self): return self
    def create(self, **kwargs): return _FakeResp(self._content)


# ── enabled() / opt-out ──────────────────────────────────────────────────────

def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("RUFUS_SUPERVISOR", raising=False)
    assert sup.enabled() is True


def test_disabled_via_env(monkeypatch):
    monkeypatch.setenv("RUFUS_SUPERVISOR", "0")
    assert sup.enabled() is False


# ── judge_seed ────────────────────────────────────────────────────────────────

def test_judge_seed_disabled_short_circuits(monkeypatch):
    monkeypatch.setenv("RUFUS_SUPERVISOR", "0")
    ok, reason = sup.judge_seed(SEED, "finance")
    assert ok is True
    assert "disabled" in reason


def test_judge_seed_no_key_fails_open(monkeypatch):
    monkeypatch.delenv("RUFUS_SUPERVISOR", raising=False)
    monkeypatch.setattr(sup, "_load_key", lambda: "")
    ok, reason = sup.judge_seed(SEED, "finance")
    assert ok is True
    assert "no OpenAI key" in reason


def test_judge_seed_approve(monkeypatch):
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    monkeypatch.setattr("openai.OpenAI",
                        lambda api_key=None: _FakeClient("APPROVE|concrete, specific story"),
                        raising=False)
    ok, reason = sup.judge_seed(SEED, "finance")
    assert ok is True
    assert "concrete" in reason


def test_judge_seed_reject(monkeypatch):
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    monkeypatch.setattr("openai.OpenAI",
                        lambda api_key=None: _FakeClient("REJECT|no concrete facts, pure filler"),
                        raising=False)
    ok, reason = sup.judge_seed(SEED, "finance")
    assert ok is False
    assert "filler" in reason


def test_judge_seed_api_exception_fails_open(monkeypatch):
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")

    class _Boom:
        def __init__(self, *a, **k): raise RuntimeError("network down")
    monkeypatch.setattr("openai.OpenAI", _Boom, raising=False)

    ok, reason = sup.judge_seed(SEED, "finance")
    assert ok is True
    assert "supervisor error" in reason


def test_judge_seed_malformed_reply_fails_open(monkeypatch):
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    monkeypatch.setattr("openai.OpenAI",
                        lambda api_key=None: _FakeClient("garbage response, no pipe"),
                        raising=False)
    ok, reason = sup.judge_seed(SEED, "finance")
    assert ok is True   # only an explicit REJECT holds up the pipeline


# ── judge_script_facts ────────────────────────────────────────────────────────

SCRIPT = ("He saved $4 a day for 30 years.\n"
          "By 62, the account held $1.2 million.\n"
          "Follow for more.")


def test_judge_script_facts_approve(monkeypatch):
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    monkeypatch.setattr("openai.OpenAI",
                        lambda api_key=None: _FakeClient("APPROVE|all figures match the source"),
                        raising=False)
    ok, reason = sup.judge_script_facts(SCRIPT, SEED)
    assert ok is True


def test_judge_script_facts_reject_names_the_claim(monkeypatch):
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    monkeypatch.setattr("openai.OpenAI",
                        lambda api_key=None: _FakeClient(
                            "REJECT|script says $2.4M but the source says $1.2M"),
                        raising=False)
    ok, reason = sup.judge_script_facts(SCRIPT, SEED)
    assert ok is False
    assert "$1.2M" in reason        # the specific claim survives into the reason


def test_judge_script_facts_disabled_short_circuits(monkeypatch):
    monkeypatch.setenv("RUFUS_SUPERVISOR", "0")
    ok, _ = sup.judge_script_facts(SCRIPT, SEED)
    assert ok is True


def test_judge_script_facts_fails_open_without_key(monkeypatch):
    monkeypatch.delenv("RUFUS_SUPERVISOR", raising=False)
    monkeypatch.setattr(sup, "_load_key", lambda: "")
    ok, _ = sup.judge_script_facts(SCRIPT, SEED)
    assert ok is True


def test_judge_script_facts_wisdom_seed_allows_history(monkeypatch):
    """The wisdom-seed prompt variant must carry the historical-facts allowance
    (mirrors the script-writer's own anti-hallucination exception)."""
    captured = {}

    class _CapturingClient(_FakeClient):
        def create(self, **kwargs):
            captured["prompt"] = kwargs["messages"][0]["content"]
            return super().create(**kwargs)

    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    monkeypatch.setattr("openai.OpenAI",
                        lambda api_key=None: _CapturingClient("APPROVE|fine"),
                        raising=False)
    wisdom = {"type": "wisdom", "source": "Seneca",
              "content": "We suffer more in imagination than in reality."}
    sup.judge_script_facts(SCRIPT, wisdom)
    assert "WISDOM-QUOTE seed" in captured["prompt"]

    # …and a non-wisdom seed must NOT get the allowance
    sup.judge_script_facts(SCRIPT, SEED)
    assert "WISDOM-QUOTE seed" not in captured["prompt"]


# ── judge_footage_prompts ─────────────────────────────────────────────────────

def test_judge_footage_prompts_empty_list_approves(monkeypatch):
    monkeypatch.delenv("RUFUS_SUPERVISOR", raising=False)
    ok, reason = sup.judge_footage_prompts([], "finance", "hook")
    assert ok is True
    assert "no prompts" in reason


def test_judge_footage_prompts_reject(monkeypatch):
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    monkeypatch.setattr("openai.OpenAI",
                        lambda api_key=None: _FakeClient("REJECT|all four prompts are near-identical"),
                        raising=False)
    ok, reason = sup.judge_footage_prompts(
        ["a photo of money", "a photo of money", "a photo of money", "a photo of money"],
        "finance", "You're broke")
    assert ok is False
    assert "near-identical" in reason
