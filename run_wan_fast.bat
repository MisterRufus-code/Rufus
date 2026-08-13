@echo off
REM run_wan_fast.bat — one motion beat via Wan 2.2 text-to-video, tuned for speed.
REM
REM WHY THIS FILE EXISTS. The fast T2V path needs eight environment variables
REM set together and one STALE variable cleared, and getting any of them wrong
REM fails in a way that looks like something else. Typing them by hand into a
REM PowerShell session that already has leftovers from the last run is how
REM "motion engines bypassed" happened.
REM
REM It also refuses to start when the template is not exported. That refusal is
REM the point: a missing or wrong export is only rejected by ComfyUI at SUBMIT
REM time, which is after the whole stills phase has run and the GPU time is
REM already spent.

REM UTF-8 in both places it matters — see run.bat for why these are two
REM different bugs and why chcp alone does not fix the second one.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo(
    echo ERROR: %CD%\%PY% not found.
    echo The virtualenv is missing or was moved - venvs are not relocatable on
    echo Windows. Recreate it:  python -m venv .venv
    echo Then:                  .venv\Scripts\pip install -r requirements.txt
    echo Refusing to fall back to the system Python, which lacks this project packages.
    exit /b 9009
)

REM --- engine selection -------------------------------------------------------
set RUFUS_GPU=1
REM Kokoro directly, not via the ElevenLabs attempt that precedes it. This file
REM did not set RUFUS_TTS, so every run opened with
REM     [tts] backend: ElevenLabs (eleven_turbo_v2_5)
REM     [tts] ElevenLabs failed (… is a library (premade) voice — free accounts
REM           cannot use those via the API …) — falling back to Kokoro
REM The fallback works and the audio is fine, but it spends a round trip and
REM prints a failure on a run where nothing is wrong. Kokoro is the voice this
REM channel actually ships. Delete this line once an ElevenLabs voice the
REM account can use is set in RUFUS_ELEVEN_VOICE.
set RUFUS_TTS=kokoro
set RUFUS_VIDEO_SOURCE=comfy
set RUFUS_RENDERER=remotion
set RUFUS_EDIT_DIRECTOR=1
set RUFUS_CHARACTER_MODE=1

REM Motion on, but on exactly ONE beat: the one the story architect identified
REM as a filmable moment. Wan 2.2 14B is a mixture-of-experts, so every clip
REM costs two expert loads on a 24GB card; ten of them is an hour of GPU for a
REM 42-second video.
set RUFUS_STILLS_ONLY=0
set RUFUS_BEAT_MOTION=hero
set RUFUS_T2V=1

REM CLEAR the stale one. RUFUS_FRAMES_PER_BEAT>1 means "cut between stills",
REM which BYPASSES every motion engine — a live run printed exactly that
REM ("motion engines bypassed") while every other setting here said motion was
REM wanted. A leftover from a previous shell must not silently win.
set RUFUS_FRAMES_PER_BEAT=

REM --- speed ------------------------------------------------------------------
REM Frame counts must be 4n+1: 49 and 33 are valid, 50 is not.
REM 480x832 is roughly a third of 832x1472's pixels; the client upscales the
REM finished clip to 1080x1920 either way.
REM NOTE: steps, cfg and the 4-step LoRA toggle are NOT settable here.
REM comfy_template.prepare() substitutes prompt, image, seed and dims only, so
REM those three are frozen into whatever you exported from ComfyUI. Export with
REM the Lightning/lightx2v LoRA ON at 4 steps — that is the single biggest win,
REM worth about 5x on this card.
REM Stills are the OTHER half of the runtime once only one beat moves: the
REM owner's ComfyUI queue shows 12-14s per still, so 9 beats x 3 = 27 stills is
REM about six minutes before Wan even starts. 1 makes it two. The cost is the
REM hard cut inside each narration line on the eight beats that stay still.
REM Set to 3 for more visual interest at ~4 minutes more per run.
set RUFUS_HERO_OTHER_FRAMES=1

set RUFUS_T2V_FRAMES=49
set RUFUS_T2V_W=480
set RUFUS_T2V_H=832
set RUFUS_T2V_TIMEOUT=600

REM --- am I running the code I think I am? -------------------------------------
REM Observed three times in one session: a report was read, acted on, and
REM discussed while the box was several commits behind, so the fixes being
REM described were not the code being run. Git says so in one cheap call and
REM nothing else does — a stale checkout produces output that looks completely
REM normal, which is the whole problem.
REM
REM Warn, never pull. Pulling someone's repository out from under them mid-run
REM is not this file's business; knowing is.
git fetch --quiet origin 2>nul
for /f %%b in ('git rev-list --count HEAD..@{u} 2^>nul') do set BEHIND=%%b
if not "%BEHIND%"=="" if not "%BEHIND%"=="0" (
    echo(
    echo NOTE: this checkout is %BEHIND% commit^(s^) behind origin.
    echo Anything fixed upstream since then is NOT in the code about to run.
    echo   git pull origin claude/automation-work-ohlqyt
    echo(
)

REM --- preflight --------------------------------------------------------------
echo(
echo Checking ComfyUI and the Wan text-to-video template...
"%PY%" scripts\comfy_doctor.py wan_t2v
if errorlevel 1 (
    echo(
    echo Refusing to start: the text-to-video engine is not runnable yet.
    echo Fix what is marked above, then run this file again. Nothing has been
    echo generated, so no GPU time has been spent.
    echo(
    echo To render stills-only in the meantime, use run.bat instead.
    pause
    exit /b 2
)

REM --- run --------------------------------------------------------------------
echo(
echo Preflight passed. Starting the run - the hero beat gets the T2V clip.
echo Per-clip seconds are printed as each clip finishes.
echo(
"%PY%" scripts\main.py --skip-upload %*

pause
