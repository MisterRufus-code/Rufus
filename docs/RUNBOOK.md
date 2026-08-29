# Runbook

What to do when something goes wrong, written from the failures that actually
happened on this machine rather than from a list of things that could.

Every entry has the same three parts: what you see, what is actually happening,
and what to do. Where a check exists that would have told you sooner, it is
named — the point of a runbook is to get shorter over time as the software
learns to say these things itself.

**Start here, always:**

```bash
python scripts/smoke.py          # does this installation work at all?
python scripts/preflight.py      # can this configuration finish a run?
python scripts/health_check.py   # the broad survey
```

Between them they cover most of what follows. If one of them already told you
the answer, the rest of this page is history.

---

## The pictures stop partway through

**You see** — a gallery frozen at, say, 9 of 38, and a page that says it is
finished. Or an estimate like "~12h 10m left" that never moves.

**What is happening** — ComfyUI is not answering. Every draw comes back empty
and the loop keeps asking, so it burns through every remaining prompt in
seconds and reports itself complete. This is the failure that produced the
9-of-38 set.

**What to do** — open `/drawing/<set>`. It now says *stalled* rather than
quoting an estimate at a dead run, and shows whether the GPU is up. Start
ComfyUI, wait for it to finish loading, and press **Draw it again**.

**Guarded since** — the loop gives up after four consecutive empty draws, and
the preflight refuses to start a `comfy` run against a renderer that is not
answering. There is no longer an ETA computed from the set row's creation
time; that number once read 12 hours because it was measuring three voice
takes and a storyboard call as if they had been drawing pictures.

---

## The dashboard will not start

**You see** — `WinError 10048`, or a page that will not load on `:8765`.

**What is happening** — either a dashboard is already running, or something
else holds the port, or a previous one died without releasing it. These are
three different situations and they used to wear one message.

**What to do** — `run_dashboard.bat` prints which it is and exits with a
different code for each. If it says a dashboard is already running, open it.
If it says the port is held by something else, the process is named.

---

## Everything says "running" and nothing is

**You see** — `/system` shows a channel as running, indefinitely.

**What is happening** — a run died without releasing its lock file. The lock is
per channel and is what stops two renders touching the same output at once.

**What to do** — check `/failures` first: a crashed run leaves a record, and
that record tells you whether it got far enough to have produced anything worth
keeping. Only then remove the `.lock` file by hand. Deleting one while a run is
genuinely in progress is how you get two processes writing the same video.

---

## The voice sounds flat, and nothing in the log says why

**You see** — narration that reads like a screen reader, on a machine
configured for Kokoro or XTTS.

**What is happening** — the TTS backend failed and fell through to Edge, which
is by design so a render never breaks. The usual cause is numpy: Kokoro's
dependency chain predates numpy 2's dtype interop and fails at *synthesis*
time, not install time, so `pip install` succeeds and every run is silently
downgraded. Three consecutive videos went out this way.

**What to do** — `pip install "numpy<2"`. The pin is in both requirements files
with the reason written beside it; something upgraded past it.

---

## The database is locked

**You see** — `database is locked` while the dashboard is open and a run is
going.

**What is happening** — the write-ahead log is not in force. `_conn` asks for
WAL on every connection, and some filesystems — network shares especially —
refuse it without saying so. Without WAL, a reader and a writer cannot coexist.

**What to do** — `python scripts/smoke.py` reports the journal mode and fails
if it is not WAL. Move the database to a local disk.

---

## Analytics show videos performing identically, or not at all

**You see** — several videos with the same view count and a watch percentage of
zero. Or `feedback_analyzer` learning from far fewer videos than you have
published.

**What is happening** — one YouTube id recorded against several videos. Six
rows once carried the same id: one link pasted six times from a phone. Metrics
join on that column, so five videos that were never published looked published
and performed like a seventh video they had nothing to do with. That is worse
than having no data — it is data that teaches the wrong lesson.

**What to do** — the audit on `/measure` finds duplicates and names which row
owns each id. Clear the wrong ones, then re-run `feedback_analyzer`.

**Guarded since** — `mark_published` refuses a duplicate and names the row that
already holds it.

---

## A video finished and nobody was told

**You see** — nothing. That is the problem: a video rendered hours ago and is
sitting on disk.

**What is happening** — a render can end six ways, and only one of them used to
make a sound. Held by QC, held on a factual flag, held on score — all silent.

**What to do** — tick **Send it to me** on the render page. Every ending is now
announced: the video to Discord, a push with a deep link to your phone.

---

## The queues are full of things you do not recognise

**You see** — scripts, galleries and reads from sittings you finished days ago.

**What is happening** — an older database from before one-video-at-a-time. The
queues used to list every pending row regardless of which project it belonged
to.

**What to do** — `/create` lists them under **Left over from before**, each with
an Abandon button. Nothing is deleted; they stop being offered as decisions,
because they are not.

---

## Something leaked

**You see** — nothing, until you look.

**What is happening** — `auth.py add` prints a sign-in link containing that
user's token, and Werkzeug logs the full request target. Opening one recorded an
owner credential in `logs/dashboard.log`, which a backup copies and a bug report
gets pasted into.

**What to do**

```bash
python scripts/logscrub.py         # scan; changes nothing
python scripts/logscrub.py --fix   # replace the values in place
python scripts/auth.py rotate <name>
```

**Rotate even after a clean scrub.** A secret that sat in a file may already
have been copied — into a backup, a screenshot, an old disk. Rewriting the file
revokes nothing.

**Guarded since** — the sign-in redirects to a clean URL and the access log is
filtered before it is written. Neither touches a line written last month, which
is what the scrub is for. `health_check` fails while any remain.

---

## You need to go back

**You see** — a database that is wrong, or a change you regret.

**What to do**

```bash
python scripts/backup.py list             # every snapshot, each verified
python scripts/backup.py restore latest
```

One snapshot is taken automatically the first time the dashboard or a run opens
the database each day. Restoring never deletes: the database being replaced is
moved aside with a timestamp, along with its journal, so it still opens. The
same list and buttons are on `/system`.

---

## You want to leave

```bash
python scripts/export_data.py
```

CSV and plain text — the videos, the scripts as files, what each month cost, and
every choice you made paired with what you chose it over. Nothing in it needs
Rufus to read it, and no credential goes with it.

---

## Reporting something not on this page

Say which build: `python scripts/version.py`, or read the line at the foot of
any dashboard page. A bug report and a fix that are about different code waste
everybody's afternoon.
