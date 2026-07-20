"""Tests for supervisor.py — cheap per-stage retry judge.

Mirrors the mocking pattern in test_metadata_writer.py: _load_key is
monkeypatched so these tests never depend on (or hit) a real OpenAI key.
"""

import pytest

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


# ── Criteria alignment with script_writer._fact_gate ────────────────────────────

def test_judge_script_facts_covers_conspiracy_framing():
    """The two fact-check layers (this judge + script_writer._fact_gate) must
    enforce the same standards — including the conspiracy/misinformation clause
    added after a live incident (Freedman-speech seed → 8/10 conspiracy script).
    This judge drives the corrective rewrite, so its objection must cover it."""
    import inspect
    import supervisor
    src = inspect.getsource(supervisor.judge_script_facts)
    assert "conspiracy" in src.lower()
    assert "misinformation" in src.lower()


# ── judge_seed: knowledge-gap requirement (not just thin/off-topic) ───────────

class _CapturingClient:
    """Like _FakeClient but records the prompt actually sent, so tests can
    assert on what the model was asked, not just how a canned reply parses."""
    def __init__(self, content):
        self._content = content
        self.last_kwargs = None
    @property
    def chat(self): return self
    @property
    def completions(self): return self
    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResp(self._content)


def test_judge_seed_prompt_includes_knowledge_gap_requirement(monkeypatch):
    """Accuracy alone must not be enough to pass — the prompt has to demand
    a genuinely counter-intuitive angle, not just concrete/on-topic facts."""
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    client = _CapturingClient("APPROVE|has a real surprise")
    monkeypatch.setattr("openai.OpenAI", lambda api_key=None: client, raising=False)

    sup.judge_seed(SEED, "finance")

    prompt = client.last_kwargs["messages"][0]["content"].lower()
    assert "knowledge gap" in prompt
    assert "counter-intuitive" in prompt


def test_judge_seed_rejects_flat_accurate_seed(monkeypatch):
    """A seed can be concrete and on-topic and still fail the knowledge-gap
    half of the check — this must still come through as a real REJECT."""
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    monkeypatch.setattr(
        "openai.OpenAI",
        lambda api_key=None: _FakeClient(
            "REJECT|accurate and on-topic but no counter-intuitive angle, knowledge gap test fails"),
        raising=False)
    ok, reason = sup.judge_seed(SEED, "finance")
    assert ok is False
    assert "knowledge gap" in reason.lower()


# ── Verdict logging (so the dashboard can compare gate bottlenecks) ───────────
# Real gap this closes: none of the three supervisor gates wrote anything to
# script_attempts — only script_writer's hook_gen/body_gen phases did. That
# means the dashboard's bottleneck breakdown could never actually answer
# "is Hook Scorer or Fact-check the real bottleneck" — the fact-check verdict
# simply didn't exist anywhere queryable.

@pytest.fixture(autouse=False)
def _isolated_attempts_db(tmp_path, monkeypatch):
    monkeypatch.setattr(sup.db_manager, "DB_FILE", tmp_path / "test.db")
    sup.db_manager.init_db()
    return sup.db_manager


def _attempt_rows(dbm):
    with dbm._conn() as c:
        return c.execute(
            "SELECT phase, niche, accepted, rejected_reason FROM script_attempts"
        ).fetchall()


def test_judge_seed_logs_approval(monkeypatch, _isolated_attempts_db):
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    monkeypatch.setattr("openai.OpenAI",
                        lambda api_key=None: _FakeClient("APPROVE|has a real surprise"),
                        raising=False)
    sup.judge_seed(SEED, "finance", run_id="run1")
    rows = _attempt_rows(_isolated_attempts_db)
    assert len(rows) == 1
    assert rows[0][0] == "seed_gate"
    assert rows[0][1] == "finance"
    assert rows[0][2] == 1
    assert rows[0][3] is None


def test_judge_seed_logs_rejection_with_reason(monkeypatch, _isolated_attempts_db):
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    monkeypatch.setattr("openai.OpenAI",
                        lambda api_key=None: _FakeClient("REJECT|no counter-intuitive angle"),
                        raising=False)
    sup.judge_seed(SEED, "finance")
    rows = _attempt_rows(_isolated_attempts_db)
    assert rows[0][0] == "seed_gate"
    assert rows[0][2] == 0
    assert "counter-intuitive" in rows[0][3]


def test_judge_script_facts_logs_as_fact_check_phase(monkeypatch, _isolated_attempts_db):
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    monkeypatch.setattr("openai.OpenAI",
                        lambda api_key=None: _FakeClient("REJECT|invented figure"),
                        raising=False)
    sup.judge_script_facts("some script", SEED, niche_name="finance", run_id="run2")
    rows = _attempt_rows(_isolated_attempts_db)
    assert rows[0][0] == "fact_check"
    assert rows[0][2] == 0
    assert "invented" in rows[0][3]


def test_judge_footage_prompts_logs_as_footage_gate_phase(monkeypatch, _isolated_attempts_db):
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    monkeypatch.setattr("openai.OpenAI",
                        lambda api_key=None: _FakeClient("APPROVE|on topic"),
                        raising=False)
    sup.judge_footage_prompts(["a prompt"], "finance", "some hook", run_id="run3")
    rows = _attempt_rows(_isolated_attempts_db)
    assert rows[0][0] == "footage_gate"
    assert rows[0][2] == 1


def test_disabled_supervisor_does_not_log(monkeypatch, _isolated_attempts_db):
    monkeypatch.setenv("RUFUS_SUPERVISOR", "0")
    sup.judge_seed(SEED, "finance")
    assert _attempt_rows(_isolated_attempts_db) == []


def test_footage_prompts_empty_list_does_not_log(monkeypatch, _isolated_attempts_db):
    """The early-return for an empty prompt list isn't a real gate
    evaluation — it must not pollute the bottleneck stats."""
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    sup.judge_footage_prompts([], "finance", "hook")
    assert _attempt_rows(_isolated_attempts_db) == []


def test_logging_failure_never_breaks_the_actual_verdict(monkeypatch, _isolated_attempts_db):
    """A DB write failure in the logging path must not affect the real
    APPROVE/REJECT the caller depends on."""
    monkeypatch.setattr(sup, "_load_key", lambda: "sk-test")
    monkeypatch.setattr("openai.OpenAI",
                        lambda api_key=None: _FakeClient("APPROVE|fine"),
                        raising=False)
    monkeypatch.setattr(sup.db_manager, "save_attempt",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("db locked")))
    ok, reason = sup.judge_seed(SEED, "finance")
    assert ok is True
    assert "fine" in reason
