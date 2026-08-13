"""The doctor answers "I downloaded the models" — did they land where ComfyUI looks?

Every ComfyUI-backed engine here is inert until its API export exists, which is
right (comfy_template.py: the graph is never hand-wired). The gap it leaves is
that "off" reads identically for four different causes — server down, nodes
missing, weights in the wrong folder, export not done — and telling them apart
used to mean starting a full run and reading the log.

Tests run with no ComfyUI and no network, per AGENTS.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import comfy_doctor  # noqa: E402


def test_an_unreachable_server_stops_before_guessing(monkeypatch, capsys):
    """Every later answer is derived from /object_info. Without it the report
    would be confident and wrong, which is worse than no report."""
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: False)
    assert comfy_doctor.main([]) == 1
    out = capsys.readouterr().out
    assert "not reachable" in out
    assert "Start ComfyUI first" in out


def test_the_wrong_folder_is_named_as_the_cause(monkeypatch, capsys):
    """The single most common reason a downloaded model "doesn't work" is that
    it is not under ComfyUI's own models/ tree. Say that, don't say "missing"."""
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files",
                        lambda host: {"UNETLoader": {"flux1-dev.safetensors"}})
    comfy_doctor.main(["wan_t2v"])
    out = capsys.readouterr().out
    assert "wrong folder" in out
    assert "loader dropdown" in out


def test_a_present_model_is_listed_with_its_loader(monkeypatch, capsys):
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files", lambda host: {
        "UNETLoader": {"wan2.2_t2v_high_noise_14B_fp8.safetensors"},
        "CLIPLoader": {"umt5_xxl_fp8.safetensors"},
    })
    comfy_doctor.main(["wan_t2v"])
    out = capsys.readouterr().out
    assert "wan2.2_t2v_high_noise_14B_fp8.safetensors" in out
    assert "[UNETLoader]" in out
    assert "umt5_xxl_fp8.safetensors" in out


def test_a_missing_export_gets_the_exact_click_path(monkeypatch, capsys):
    """The export is the one step that is never automatic, so the instruction
    has to be complete enough to follow without leaving the terminal."""
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files",
                        lambda host: {"UNETLoader": {"wan2.2_t2v.safetensors"}})
    monkeypatch.setattr(comfy_doctor.Path, "exists", lambda self: False)
    comfy_doctor.main(["wan_t2v"])
    out = capsys.readouterr().out
    assert "no export at config/wan_t2v_api.json" in out
    assert "RUFUS_PROMPT" in out
    assert "Export (API)" in out


def test_an_unknown_engine_name_is_reported_not_ignored(monkeypatch, capsys):
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files", lambda host: {})
    comfy_doctor.main(["wan_t2v", "nonsense"])
    out = capsys.readouterr().out
    assert "unknown engine(s): nonsense" in out


def test_no_argument_reports_every_engine(monkeypatch, capsys):
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files", lambda host: {})
    comfy_doctor.main([])
    out = capsys.readouterr().out
    for name in comfy_doctor.ENGINES:
        assert name in out


def test_every_template_path_it_names_matches_the_engine_that_reads_it():
    """A doctor that points at the wrong filename sends the owner to export a
    file nothing will ever load."""
    import wan_t2v_client
    assert comfy_doctor.ENGINES["wan_t2v"][1] in str(wan_t2v_client._template_path())


def test_it_is_read_only():
    """This runs when something is already broken. It must not write, delete or
    start anything."""
    src = Path(comfy_doctor.__file__).read_text(encoding="utf-8")
    for forbidden in ("write_text(", "unlink(", "mkdir(", "requests.post",
                      "subprocess"):
        assert forbidden not in src, forbidden


# ── the Wan inventory: which variant, and what it costs per clip ─────────────

def test_ti2v_5b_is_not_filed_as_an_i2v_model():
    """"ti2v" CONTAINS "i2v". Without an exclusion the one variant that solves
    the 16GB-RAM problem gets reported as a 14B image-to-video model and the
    advice inverts — it would say "you have i2v but no t2v, go download more"
    at someone who already has the right file."""
    found = comfy_doctor._classify_wan(
        {"UNETLoader": {"wan2.2_ti2v_5B_fp16.safetensors"}})
    assert found["ti2v_5b"] == ["wan2.2_ti2v_5B_fp16.safetensors"]
    assert found["i2v_model"] == []


def test_t2v_and_i2v_models_are_told_apart():
    found = comfy_doctor._classify_wan({"UNETLoader": {
        "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"}})
    assert found["t2v_model"] == ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"]
    assert found["i2v_model"] == ["wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"]


def test_a_lora_is_not_counted_as_a_model():
    """Otherwise "you have T2V models" is true while nothing can generate."""
    found = comfy_doctor._classify_wan({"LoraLoaderModelOnly": {
        "wan2.2_t2v_lightx2v_4steps_lora_v1_high_noise.safetensors"}})
    assert found["t2v_model"] == []
    assert found["lora_t2v"]


def test_having_i2v_but_not_t2v_says_they_are_separate_downloads():
    found = comfy_doctor._classify_wan(
        {"UNETLoader": {"wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"}})
    tips = " ".join(comfy_doctor._wan_advice(found, have_export=False))
    assert "different downloads" in tips


def test_the_missing_4_step_lora_leads_the_advice():
    """It is a 5x saving against maybe one minute from fixing the expert swap,
    so it must come before any RAM talk."""
    found = comfy_doctor._classify_wan({
        "UNETLoader": {"wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"},
        "CLIPLoader": {"umt5_xxl_fp8.safetensors"}})
    tips = comfy_doctor._wan_advice(found, have_export=False)
    assert "BIGGEST WIN" in tips[0]
    assert "4-step" in tips[0] or "4 steps" in tips[0]


def test_a_present_lora_is_pointed_at_export_time_not_an_env_var():
    """prepare() substitutes prompt/image/seed/dims only — steps and the LoRA
    toggle are frozen into the export. Telling someone to set an env var for
    them would send them chasing a variable that does not exist."""
    found = comfy_doctor._classify_wan({
        "UNETLoader": {"wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"},
        "LoraLoaderModelOnly": {"wan2.2_t2v_lightx2v_4steps_lora_v1_high.safetensors"},
        "CLIPLoader": {"umt5_xxl_fp8.safetensors"}})
    tips = " ".join(comfy_doctor._wan_advice(found, have_export=False))
    assert "Export (API)" in tips
    assert "no env var" in tips.lower()


def test_ti2v_without_its_own_vae_is_flagged_before_the_run_not_after():
    """TI2V-5B does not use wan_2.1_vae. The wrong VAE is only rejected at
    submit time — after the whole stills phase has already run."""
    found = comfy_doctor._classify_wan({
        "UNETLoader": {"wan2.2_ti2v_5B_fp16.safetensors"},
        "VAELoader": {"wan_2.1_vae.safetensors"}})
    tips = " ".join(comfy_doctor._wan_advice(found, have_export=False))
    assert "Wan 2.2 VAE" in tips
    assert "value_not_in_list" in tips


def test_a_missing_text_encoder_is_reported(monkeypatch):
    found = comfy_doctor._classify_wan(
        {"UNETLoader": {"wan2.2_t2v_high_noise_14B.safetensors"}})
    tips = " ".join(comfy_doctor._wan_advice(found, have_export=False))
    assert "umt5" in tips


def test_an_empty_comfyui_does_not_raise():
    found = comfy_doctor._classify_wan({})
    assert all(v == [] for v in found.values())
    assert comfy_doctor._wan_advice(found, have_export=False)


def test_the_advice_does_not_claim_a_lora_the_inventory_does_not_show():
    """VERBATIM FROM THE FIRST REAL RUN of this report. It printed

        ✗ 4-step LoRA (i2v)          none visible
        …
        1. BIGGEST WIN … You already have the i2v version of this LoRA

    two lines apart. The claim came from wan_client.py's header, which lists
    what the I2V template WOULD install — not what is on this disk. A doctor
    that asserts something the reader can see is false stops being trusted for
    the things it gets right."""
    found = comfy_doctor._classify_wan({
        "UNETLoader": {"wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"},
        "CLIPLoader": {"umt5_xxl_fp8_e4m3fn_scaled.safetensors"},
        "VAELoader": {"wan_2.1_vae.safetensors"}})
    assert found["lora_i2v"] == []
    tips = " ".join(comfy_doctor._wan_advice(found, have_export=False))
    assert "already have the i2v version" not in tips


def test_it_does_mention_the_i2v_lora_when_that_one_is_present():
    found = comfy_doctor._classify_wan({
        "UNETLoader": {"wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"},
        "LoraLoaderModelOnly": {
            "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"},
        "CLIPLoader": {"umt5_xxl_fp8.safetensors"}})
    tips = " ".join(comfy_doctor._wan_advice(found, have_export=False))
    assert "already have the i2v version" in tips


def test_a_no_download_fallback_is_offered_while_the_lora_is_missing():
    """Cutting KSampler steps by hand is most of the win and needs nothing
    fetched — someone blocked on a download should not be fully blocked."""
    found = comfy_doctor._classify_wan({
        "UNETLoader": {"wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"},
        "CLIPLoader": {"umt5_xxl_fp8.safetensors"}})
    tips = " ".join(comfy_doctor._wan_advice(found, have_export=False))
    assert "no download" in tips


def test_a_specific_engine_that_cannot_run_exits_nonzero(monkeypatch, capsys):
    """A preflight that always exits 0 cannot gate a launcher, and the failure
    it would have stopped is expensive: a bad template is rejected at SUBMIT
    time, after the whole stills phase has run."""
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files",
                        lambda host: {"UNETLoader": {"wan2.2_t2v.safetensors"}})
    monkeypatch.setattr(comfy_doctor.Path, "exists", lambda self: False)
    assert comfy_doctor.main(["wan_t2v"]) == 2
    assert "not runnable yet" in capsys.readouterr().out


def test_a_bare_survey_never_exits_nonzero(monkeypatch, capsys):
    """Most engines being un-exported is this repo's normal resting state, not
    an error — only an engine the caller ASKED about is a gate."""
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files", lambda host: {})
    monkeypatch.setattr(comfy_doctor.Path, "exists", lambda self: False)
    assert comfy_doctor.main([]) == 0


# ── the false negative that sent the owner to re-download what he had ────────

def test_lora_loaders_are_queried_at_all():
    """VERBATIM: the first real run printed

        ✗ 4-step LoRA (t2v)          none visible

    on a box where wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise and its
    low_noise partner were in models/loras AND already wired into the open
    workflow. Nothing here had ever asked ComfyUI for the LoRA enum, so the
    answer was always "none". A doctor that reports absent for something
    present makes every other line it prints suspect."""
    assert "LoraLoaderModelOnly" in comfy_doctor._LOADERS
    assert "LoraLoader" in comfy_doctor._LOADERS


def test_a_lora_visible_only_through_its_own_loader_is_found(monkeypatch, capsys):
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files", lambda host: {
        "UNETLoader": {"wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"},
        "LoraLoaderModelOnly": {
            "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
            "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"},
    })
    comfy_doctor.main(["wan_t2v"])
    out = capsys.readouterr().out
    assert "✗ 4-step LoRA (t2v)" not in out
    assert "BIGGEST WIN, and it is missing" not in out


# ── settings frozen into an export ───────────────────────────────────────────

def test_turbo_mode_off_in_an_export_is_reported():
    """ComfyUI's packaged Wan node folds the whole 4-step LoRA path behind one
    boolean. Exported false, the LoRA files sit in the graph contributing
    nothing, and the only symptom is that clips are slow."""
    notes = comfy_doctor._speed_notes(
        {"1": {"class_type": "WanT2V", "inputs": {"enable_turbo_mode": False}}})
    assert notes and "enable_turbo_mode is FALSE" in notes[0]


def test_turbo_mode_on_is_silent():
    assert comfy_doctor._speed_notes(
        {"1": {"class_type": "WanT2V", "inputs": {"enable_turbo_mode": True}}}) == []


def test_a_high_step_count_is_reported_with_what_it_costs():
    notes = comfy_doctor._speed_notes(
        {"1": {"class_type": "KSamplerAdvanced", "inputs": {"steps": 20}}})
    assert notes and "20 steps" in notes[0]


def test_a_fast_step_count_is_silent():
    assert comfy_doctor._speed_notes(
        {"1": {"class_type": "KSamplerAdvanced", "inputs": {"steps": 4}}}) == []


def test_speed_notes_tolerate_a_junk_graph():
    assert comfy_doctor._speed_notes({"1": {"inputs": None}, "2": {}}) == []


# ── values the export will silently ignore ───────────────────────────────────
#
# prepare() writes prompt, image, seed and dims into inputs THAT EXIST. A
# missing input means the write is skipped and nothing reports it: the env var
# the owner typed did nothing, the run succeeded, and there is no line anywhere
# to read. Packaged all-in-one nodes collapse a graph into one node and expose
# only some knobs, which is exactly where this bites.

def test_a_graph_with_seed_and_dims_is_silent():
    tpl = {"1": {"class_type": "K", "inputs": {
        "seed": 1, "width": 480, "height": 832, "length": 49}}}
    assert comfy_doctor._substitution_gaps(tpl) == []


def test_a_node_with_no_seed_input_is_flagged():
    """wan_t2v_client's whole seed-lineage mechanism is inert without one, and
    nothing in a run would ever say so."""
    tpl = {"1": {"class_type": "WanT2V", "inputs": {
        "width": 480, "height": 832, "duration": 3}}}
    gaps = " ".join(comfy_doctor._substitution_gaps(tpl))
    assert "no seed input" in gaps
    assert "seed lineage" in gaps


def test_a_node_with_no_dimension_inputs_is_flagged():
    """RUFUS_T2V_W/H/FRAMES silently doing nothing means the export's own
    resolution wins — a landscape export into a vertical pipeline still
    'succeeds', pillarboxed."""
    tpl = {"1": {"class_type": "X", "inputs": {"seed": 1}}}
    gaps = " ".join(comfy_doctor._substitution_gaps(tpl))
    assert "RUFUS_T2V_W/H/FRAMES will do nothing" in gaps


def test_duration_counts_as_a_dimension_input():
    """Seconds-based nodes are sized by prepare()'s duration branch, so they
    are NOT a gap."""
    tpl = {"1": {"class_type": "X", "inputs": {
        "noise_seed": 7, "width": 1, "height": 1, "duration": 5.0}}}
    assert comfy_doctor._substitution_gaps(tpl) == []


def test_width_and_height_without_a_length_is_still_a_gap():
    """prepare() requires the trio; two out of three means the branch never
    fires and none of the three is written."""
    tpl = {"1": {"class_type": "X", "inputs": {
        "seed": 1, "width": 480, "height": 832}}}
    assert comfy_doctor._substitution_gaps(tpl)


def test_gaps_tolerate_a_junk_graph():
    assert len(comfy_doctor._substitution_gaps({"1": {"inputs": None}})) == 2


def test_a_gap_is_a_warning_and_never_blocks_the_run():
    """These degrade quality, they do not break generation — and this repo
    reserves hard gates for correctness (AGENTS.md, the rejection ladder)."""
    import inspect
    src = inspect.getsource(comfy_doctor._report_engine)
    assert "_substitution_gaps" in src
    # usable is not touched by the gap check
    after = src.split("_substitution_gaps")[1].split("return usable")[0]
    assert "usable = False" not in after


# ── say what the export contains, not only what is wrong with it ─────────────

def test_the_export_facts_report_the_numbers():
    """Silence from _speed_notes reads identically whether the settings are
    good or whether nothing recognisable was found, so "did the 4-step toggle
    make it into the file?" was being answered by inferring from an absence."""
    facts = comfy_doctor._export_facts({"1": {
        "class_type": "KSamplerAdvanced",
        "inputs": {"steps": 4, "cfg": 1.0, "sampler_name": "euler"}}})
    assert facts == ["KSamplerAdvanced: steps=4, cfg=1.0, sampler_name=euler"]


def test_size_is_reported_with_its_length_field():
    facts = comfy_doctor._export_facts({"1": {
        "class_type": "WanT2V",
        "inputs": {"width": 640, "height": 640, "duration": 5.0}}})
    assert facts == ["WanT2V: 640x640 duration=5.0"]


def test_a_linked_input_is_not_reported_as_a_value():
    """[node_id, slot] is a WIRE, not a setting. Printing it as one would read
    as a nonsense step count."""
    assert comfy_doctor._export_facts({"1": {
        "class_type": "K", "inputs": {"steps": ["7", 0]}}}) == []


def test_an_export_that_states_nothing_says_so(monkeypatch, capsys):
    """An empty fact list is itself informative — it means every guess about
    this export's speed is a guess, so say that instead of printing nothing."""
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files", lambda host: {})
    monkeypatch.setattr(comfy_doctor.Path, "exists", lambda self: True)
    monkeypatch.setattr(comfy_doctor.comfy_template, "load_template",
                        lambda p: {"1": {"class_type": "X",
                                         "inputs": {"text": "RUFUS_PROMPT"}}})
    monkeypatch.setattr(comfy_doctor.comfy_template, "missing_nodes",
                        lambda t, h: [])
    monkeypatch.setattr(comfy_doctor.comfy_template, "missing_models",
                        lambda t, h: [])
    comfy_doctor.main(["wan_t2v"])
    assert "judge it by the clip time" in capsys.readouterr().out


# ── --dry-run: will MY settings land on MY export? ───────────────────────────
#
# Different question from "is the export loadable", and the expensive one:
# prepare() writes into inputs that EXIST and skips the rest, so a valid export
# can silently ignore every environment variable that was set.

def test_dimensions_landing_on_the_node_are_confirmed(monkeypatch):
    monkeypatch.setenv("RUFUS_T2V_W", "480")
    monkeypatch.setenv("RUFUS_T2V_H", "832")
    monkeypatch.setenv("RUFUS_T2V_FRAMES", "49")
    tpl = {"1": {"class_type": "WanT2V", "inputs": {
        "width": 640, "height": 640, "duration": 5.0,
        "positive_prompt": "RUFUS_PROMPT"}}}
    lines = comfy_doctor._dry_run("wan_t2v", tpl)
    assert any("asking for 480x832, 49 frames" in l for l in lines)
    assert any("received 480x832" in l for l in lines)
    assert not any("NOTHING received" in l for l in lines)


def test_a_frozen_export_is_called_out_on_every_axis():
    """An export whose node exposes none of the substitutable inputs runs its
    saved size and its saved prompt on every beat, forever, silently."""
    tpl = {"1": {"class_type": "Frozen", "inputs": {"text": "a cat"}}}
    lines = " | ".join(comfy_doctor._dry_run("wan_t2v", tpl))
    assert "NOTHING received the dimensions" in lines
    assert "nothing received a seed" in lines
    assert "prompt placeholder did not substitute" in lines


def test_a_missing_seed_input_is_reported_even_when_dims_land():
    """ComfyUI's packaged Wan node takes width/height/duration but exposes no
    seed, so the dims succeed while wan_t2v_client's whole seed lineage does
    nothing — exactly the kind of half-success that reads as working."""
    tpl = {"1": {"class_type": "WanT2V", "inputs": {
        "width": 1, "height": 1, "duration": 5.0, "prompt": "RUFUS_PROMPT"}}}
    lines = " | ".join(comfy_doctor._dry_run("wan_t2v", tpl))
    assert "nothing received a seed" in lines
    assert "NOTHING received the dimensions" not in lines


def test_a_seed_bearing_graph_reports_it(monkeypatch):
    monkeypatch.setenv("RUFUS_T2V_W", "480")
    monkeypatch.setenv("RUFUS_T2V_H", "832")
    tpl = {"1": {"class_type": "KSampler", "inputs": {
        "width": 1, "height": 1, "length": 1, "seed": 0,
        "text": "RUFUS_PROMPT"}}}
    lines = " | ".join(comfy_doctor._dry_run("wan_t2v", tpl))
    assert "received the seed" in lines


def test_the_dry_run_uses_wans_framerate_for_seconds_based_nodes(monkeypatch):
    monkeypatch.setenv("RUFUS_T2V_FRAMES", "49")
    tpl = {"1": {"class_type": "WanT2V", "inputs": {
        "width": 1, "height": 1, "duration": 9.0, "p": "RUFUS_PROMPT"}}}
    lines = comfy_doctor._dry_run("wan_t2v", tpl)
    assert any("16fps" in l for l in lines)
    assert any("duration=3" in l for l in lines)      # 49/16, not 49/25


def test_the_dry_run_is_off_unless_asked(monkeypatch, capsys):
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files", lambda host: {})
    monkeypatch.setattr(comfy_doctor.Path, "exists", lambda self: False)
    comfy_doctor.main(["wan_t2v"])
    assert "dry run" not in capsys.readouterr().out


def test_a_broken_engine_module_does_not_crash_the_report():
    assert comfy_doctor._dry_run("stills_i2i", {"1": {}}) == []


# ── a toggle is "off" in more spellings than `is False` ──────────────────────

def test_every_spelling_of_off_is_caught():
    """The owner's ComfyUI showed enable_turbo_mode false in the open workflow
    while this check stayed silent on the export made from it. Both cannot be
    right, and an under-reporting check is the worse failure: silence here
    reads as "your settings are fine"."""
    for off in (False, "false", "False", " FALSE ", 0, "0", "off", "no",
                "disabled"):
        assert comfy_doctor._is_off(off), repr(off)


def test_on_and_absent_are_not_off():
    """Absent is unknown, not off — warning about a key the export never had
    would send someone to fix a toggle that does not exist."""
    for on in (True, "true", 1, "yes", None, "enabled"):
        assert not comfy_doctor._is_off(on), repr(on)


def test_a_string_step_count_is_still_a_step_count():
    notes = comfy_doctor._speed_notes(
        {"1": {"class_type": "K", "inputs": {"steps": "20"}}})
    assert notes and "20 steps" in notes[0]


def test_a_wire_is_never_read_as_a_step_count():
    """["7", 0] is a link to node 7 output 0. Reading it as 7 would report a
    fast sampler on a graph that has none."""
    assert comfy_doctor._as_int(["7", 0]) is None
    assert comfy_doctor._speed_notes(
        {"1": {"class_type": "K", "inputs": {"steps": ["7", 0]}}}) == []


def test_a_bool_is_not_a_step_count():
    """True == 1 in Python; without the guard a boolean input would read as a
    1-step sampler."""
    assert comfy_doctor._as_int(True) is None


def test_turbo_off_as_a_string_reaches_the_warning():
    notes = comfy_doctor._speed_notes(
        {"1": {"class_type": "WanT2V", "inputs": {"enable_turbo_mode": "false"}}})
    assert notes and "enable_turbo_mode is FALSE" in notes[0]


# ── a mistyped flag must not look like success ───────────────────────────────

def test_a_mistyped_dry_run_flag_stops_instead_of_reporting(monkeypatch, capsys):
    """VERBATIM FROM A REAL SESSION: two commands arrived pasted onto one line
    as `--dry-rungit pull origin <branch>`. Every token fell into the "unknown
    engine" list, the ordinary report printed in full, and the dry run never
    happened — in a tool written specifically to catch degraded paths nobody
    can see."""
    def _boom(*a, **k):
        raise AssertionError("must not reach ComfyUI after a bad flag")
    monkeypatch.setattr(comfy_doctor, "_reachable", _boom)
    assert comfy_doctor.main(["wan_t2v", "--dry-rungit", "pull"]) == 2
    out = capsys.readouterr().out
    assert "unknown option(s): --dry-rungit" in out
    assert "did you mean --dry-run?" in out
    assert "pasted onto one line" in out


def test_an_unrelated_bad_flag_is_also_refused(monkeypatch, capsys):
    monkeypatch.setattr(comfy_doctor, "_reachable",
                        lambda h: (_ for _ in ()).throw(AssertionError()))
    assert comfy_doctor.main(["--verbose"]) == 2
    out = capsys.readouterr().out
    assert "unknown option(s): --verbose" in out
    assert "did you mean" not in out          # only suggested when plausible


def test_the_real_flag_still_works(monkeypatch, capsys):
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files", lambda host: {})
    monkeypatch.setattr(comfy_doctor.Path, "exists", lambda self: False)
    comfy_doctor.main(["wan_t2v", "--dry-run"])
    assert comfy_doctor.DRY_RUN is True


def test_a_bare_engine_name_is_unaffected(monkeypatch, capsys):
    monkeypatch.setattr(comfy_doctor, "_reachable", lambda host: True)
    monkeypatch.setattr(comfy_doctor, "_visible_files", lambda host: {})
    monkeypatch.setattr(comfy_doctor.Path, "exists", lambda self: False)
    assert comfy_doctor.main(["wan_t2v"]) == 2      # not runnable, not a usage error
    assert "unknown option" not in capsys.readouterr().out
