#!/usr/bin/env python3
"""Live terminal dashboard of running Claude Code sessions.

Reads ~/.claude/sessions/*.json for every active interactive session
(pid, name, status, cwd), then incrementally tails each session's
transcript in ~/.claude/projects/*/ to show model, context size, tool
usage, skills, cache efficiency, tool errors, and loop/stall flags —
aimed at spotting sessions with bloated context or thrashing behavior.

Usage: python3 ~/.claude/scripts/claude-live-dashboard.py
Keys: up/down to select a row, Enter/g to jump to it in cmux,
      c to send /clear, k to send /compact (both confirm with y/N),
      ? for a glossary of statuses/columns/keys, q to quit.

Thresholds below are heuristics, not exact model limits — tune to taste.
"""
import curses
import json
import locale
import os
import re
import shutil
import subprocess
import textwrap
import time
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path

CLAUDE_DIR = Path.home() / ".claude"
SESSIONS_DIR = CLAUDE_DIR / "sessions"
PROJECTS = CLAUDE_DIR / "projects"
REFRESH_SECONDS = 1.5

# Native context window per model (Anthropic API). Source: code.claude.com/docs/en/model-config
# Sonnet 5 / Opus 5 / Fable 5 run 1M natively with no special config; everything else defaults to
# 200K unless it's running with a "[1m]" suffix, which we can't see from the transcript so we don't
# try to detect it. DEFAULT_CTX_LIMIT covers any model name not in this table (older/unknown models).
MODEL_CONTEXT_WINDOWS = {
    "claude-sonnet-5": 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-fable-5": 1_000_000,
}
DEFAULT_CTX_LIMIT = 200_000

CTX_WARN_RATIO = 0.60  # fraction of the model's context window
CTX_CRIT_RATIO = 0.85  # Claude Code itself auto-compacts around ~0.967 of the window
LOWCACHE_MIN_CTX = 20_000
LOWCACHE_RATIO = 0.4
LOOP_RUN_LEN = 4
STALL_BUSY_SECS = 300

CMUX_BIN = "/Applications/cmux.app/Contents/Resources/bin/cmux"
if not Path(CMUX_BIN).exists():
    CMUX_BIN = shutil.which("cmux")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _msg_text(msg: dict) -> str:
    """Flatten a message's content (string or list of blocks) into plain text."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return ""


def duration_str(secs: float) -> str:
    """Compact duration like '42s', '5m', '3h 20m', '2d 4h'."""
    secs = max(0, secs)
    if secs < 60:
        return f"{int(secs)}s"
    mins = secs / 60
    if mins < 60:
        return f"{int(mins)}m"
    hours = mins / 60
    if hours < 24:
        return f"{int(hours)}h {int(mins % 60)}m"
    days = hours / 24
    return f"{int(days)}d {int(hours % 24)}h"


def ago_str(ts_ms: float) -> str:
    return duration_str(time.time() - ts_ms / 1000)


def abs_str(ts_ms: float) -> str:
    return datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M")


class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.name = session_id[:8]
        self.status = "-"
        self.cwd = "?"
        self.pid = None
        self.path = None
        self.offset = 0
        self.model = None
        self.tools = OrderedDict()
        self.skills = OrderedDict()
        self.total_output = 0
        self.last_input_tokens = 0
        self.last_cache_read = 0
        self.last_cache_creation = 0
        self.has_usage = False  # True once we've seen a real assistant usage entry
        self.compact_pending = False  # True from a /compact marker until the next real usage entry
        self.last_msg_ts_ms = None
        self.created_ms = None
        self.last_tool = None
        self.tool_errors = 0
        self.recent_tools = deque(maxlen=8)

    def _find_path(self):
        if self.path and self.path.exists():
            return
        matches = list(PROJECTS.glob(f"*/{self.session_id}.jsonl"))
        if matches:
            self.path = matches[0]

    def _reset_for_clear(self):
        self.total_output = 0
        self.last_input_tokens = 0
        self.last_cache_read = 0
        self.last_cache_creation = 0
        self.has_usage = False
        self.last_tool = None
        self.tool_errors = 0
        self.tools.clear()
        self.skills.clear()
        self.recent_tools.clear()

    def update(self):
        self._find_path()
        if not self.path or not self.path.exists():
            return
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
        if size == self.offset:
            return
        with open(self.path, "r", errors="ignore") as f:
            f.seek(self.offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                # Only "user"/"assistant" entries are real conversation turns — other
                # event types (queue-operation, file-history-snapshot, attachment, ...)
                # have timestamps too but don't represent an actual message.
                if d.get("type") in ("user", "assistant"):
                    ts = d.get("timestamp")
                    if ts:
                        try:
                            self.last_msg_ts_ms = (
                                datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                                * 1000
                            )
                        except ValueError:
                            pass
                msg = d.get("message")
                if not isinstance(msg, dict):
                    continue
                # A real slash-command invocation is the ENTIRE message content, nothing else —
                # e.g. exactly "<command-name>/clear</command-name>\n  <command-message>...".
                # Checking startswith (not "in") avoids false-triggering on text that merely
                # mentions the marker, e.g. a compaction summary describing this very feature.
                msg_text_stripped = _msg_text(msg).strip() if d.get("type") == "user" else ""
                if msg_text_stripped.startswith("<command-name>/clear</command-name>"):
                    # /clear resets the model's context immediately, but the dashboard only learns
                    # the new (near-zero) size from the NEXT turn's usage stats — same lag as the
                    # real context itself. Reset our own counters now so the display isn't stuck
                    # showing the pre-clear numbers in the meantime.
                    self._reset_for_clear()
                if msg_text_stripped.startswith("<command-name>/compact</command-name>"):
                    # /compact rewrites history into a summary, but (unlike /clear) we have no way
                    # to know the new context size until the next real usage entry arrives — mark
                    # the current numbers as stale/pending instead of guessing or leaving them look
                    # accurate.
                    self.compact_pending = True
                if msg.get("model"):
                    self.model = msg["model"]
                u = msg.get("usage")
                if u:
                    self.has_usage = True
                    self.compact_pending = False
                    self.total_output += u.get("output_tokens", 0) or 0
                    # context_tokens is a snapshot of the LAST turn only — cache_read/creation/input
                    # together represent that turn's full context. Do NOT accumulate across turns,
                    # each turn's usage already covers the whole conversation up to that point.
                    self.last_input_tokens = u.get("input_tokens", 0) or 0
                    self.last_cache_read = u.get("cache_read_input_tokens", 0) or 0
                    self.last_cache_creation = u.get("cache_creation_input_tokens", 0) or 0
                content = msg.get("content")
                if isinstance(content, list):
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        ctype = c.get("type")
                        if ctype == "tool_use":
                            name = c.get("name", "?")
                            inp = c.get("input") or {}
                            if name == "Skill":
                                sname = inp.get("skill") or inp.get("name") or "?"
                                self.skills[sname] = self.skills.get(sname, 0) + 1
                            self.tools[name] = self.tools.get(name, 0) + 1
                            self.last_tool = name
                            self.recent_tools.append((name, json.dumps(inp, sort_keys=True, default=str)))
                        elif ctype == "tool_result" and c.get("is_error"):
                            self.tool_errors += 1
            self.offset = f.tell()

    @property
    def context_tokens(self) -> int:
        return self.last_cache_read + self.last_cache_creation + self.last_input_tokens

    @property
    def ctx_limit(self) -> int:
        """Native context window for this session's model, or a conservative default if unknown."""
        return MODEL_CONTEXT_WINDOWS.get(self.model, DEFAULT_CTX_LIMIT)

    @property
    def ctx_pct(self):
        """Percentage of the model's context window used by the last turn, or None with no data yet."""
        if not self.has_usage:
            return None
        return self.context_tokens / self.ctx_limit * 100

    @property
    def cache_ratio(self):
        """Fraction of the last turn's context served from cache, or None if no turn has happened yet."""
        if not self.has_usage:
            return None
        denom = self.last_cache_read + self.last_cache_creation
        return (self.last_cache_read / denom) if denom else 1.0

    @property
    def is_looping(self) -> bool:
        """True when the last LOOP_RUN_LEN tool calls are identical (same tool + same input) —
        a real stuck-retry signal, unlike merely calling the same tool repeatedly with different args."""
        if len(self.recent_tools) < LOOP_RUN_LEN:
            return False
        last = list(self.recent_tools)[-LOOP_RUN_LEN:]
        return len(set(last)) == 1

    def warnings(self):
        """Plain-language list of (severity, message) — severity is 'crit' or 'warn'."""
        out = []
        if self.compact_pending:
            out.append(("warn", "/compact ran — CTX/%LIM below are stale until the next real turn"))
        if self.ctx_pct is not None and self.ctx_pct >= CTX_CRIT_RATIO * 100:
            out.append(("crit", f"Context is very large ({fmt_k(self.context_tokens)} tokens, {self.ctx_pct:.0f}% of {fmt_k(self.ctx_limit)}) — consider a fresh session"))
        elif self.ctx_pct is not None and self.ctx_pct >= CTX_WARN_RATIO * 100:
            out.append(("warn", f"Context is getting big ({fmt_k(self.context_tokens)} tokens, {self.ctx_pct:.0f}% of {fmt_k(self.ctx_limit)})"))
        if self.is_looping:
            out.append(("crit", f"Repeating the exact same {self.recent_tools[-1][0]} call — looks stuck in a loop"))
        if self.tool_errors:
            out.append(("warn", f"{self.tool_errors} tool call(s) failed this session"))
        if self.cache_ratio is not None and self.context_tokens >= LOWCACHE_MIN_CTX and self.cache_ratio < LOWCACHE_RATIO:
            out.append(("warn", f"Only {self.cache_ratio * 100:.0f}% of the last turn's context was cached — paying full price for a lot of it"))
        if self.status == "busy" and self.last_msg_ts_ms:
            idle_secs = time.time() - self.last_msg_ts_ms / 1000
            if idle_secs >= STALL_BUSY_SECS:
                out.append(("crit", f"Marked busy but no message for {ago_str(self.last_msg_ts_ms)} — may be stuck"))
        return out

    def top_tools(self, n=8):
        return sorted(self.tools.items(), key=lambda kv: -kv[1])[:n]

    def all_skills(self) -> str:
        return ", ".join(self.skills.keys()) if self.skills else "none"


def load_sessions():
    """Return {pid_str: session_dict} for every *.json in ~/.claude/sessions/."""
    out = {}
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if d.get("sessionId"):
            out[f.stem] = d
    return out


def fmt_k(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def safe_addstr(stdscr, y, x, text, attr=curses.A_NORMAL):
    try:
        stdscr.addstr(y, x, text, attr)
    except curses.error:
        pass


def cmux_run(*args, timeout=5):
    """Run a cmux CLI subcommand. Returns (ok, stdout_or_error)."""
    if not CMUX_BIN:
        return False, "cmux not found"
    try:
        r = subprocess.run([CMUX_BIN, *args], capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout).strip()[:200]
    return True, r.stdout


CMUX_WS_RE = re.compile(r"workspace:\d+")
CMUX_SURFACE_BOTH_RE = re.compile(r"(surface:\d+)\s+([0-9A-Fa-f-]{36})")
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def cmux_find_surface(session_id: str):
    """Find the cmux pane running this Claude session.

    cmux's own --resume marker (shown by `surface resume get`) embeds the same
    session UUID Claude Code uses, so we match on that. Two CLI quirks drove this
    implementation: `cmux rpc <method> '{...}'` silently ignores workspace/surface
    scoping in its JSON body and always operates on the CALLING shell's own pane
    (so we use the purpose-built subcommands, which honor --workspace/--surface
    flags correctly) — and the `surface.focus` RPC needs a real UUID, not the
    short "surface:N" ref, hence the --id-format both lookup.
    Returns {workspace_ref, surface_ref, surface_uuid} or None.
    """
    if not CMUX_BIN:
        return None
    ok, out = cmux_run("list-workspaces")
    if not ok:
        return None
    for ws_ref in sorted(set(CMUX_WS_RE.findall(out))):
        ok, out = cmux_run("list-panels", "--workspace", ws_ref, "--id-format", "both")
        if not ok:
            continue
        for surface_ref, surface_uuid in CMUX_SURFACE_BOTH_RE.findall(out):
            ok, resume_cmd = cmux_run("surface", "resume", "get", "--workspace", ws_ref, "--surface", surface_ref)
            if not ok or "--resume" not in resume_cmd:
                continue
            m = UUID_RE.search(resume_cmd.split("--resume", 1)[1])
            if m and m.group(0) == session_id:
                return {"workspace_ref": ws_ref, "surface_ref": surface_ref, "surface_uuid": surface_uuid}
    return None


def cmux_navigate(session_id: str):
    loc = cmux_find_surface(session_id)
    if not loc:
        return False, "Not found in cmux (not running in a cmux pane, or already closed)"
    # `rpc surface.focus` looks like the right tool (it's what a JSON body targeting a specific
    # surface should be for) but it's silently broken: it always operates on the CALLING shell's
    # own workspace/surface (same root bug as the rpc scoping quirk elsewhere in this file),
    # so every navigation landed back on wherever the dashboard itself happened to be running —
    # confirmed by testing it against several different real targets and watching it no-op every
    # time. `select-workspace` + `focus-panel`, the purpose-built subcommands, reliably switch
    # both the workspace AND the exact tab — verified against multiple real cross-workspace targets.
    ok, msg = cmux_run("select-workspace", "--workspace", loc["workspace_ref"])
    if not ok:
        return False, f"select-workspace failed: {msg}"
    ok, msg = cmux_run("focus-panel", "--panel", loc["surface_ref"], "--workspace", loc["workspace_ref"])
    if not ok:
        return False, f"focus-panel failed: {msg}"
    subprocess.run(["open", "-a", "cmux"], capture_output=True)  # best-effort: bring app to front
    return True, f"Switched to {loc['workspace_ref']} / {loc['surface_ref']} in cmux"


def cmux_send_command(session_id: str, text: str):
    loc = cmux_find_surface(session_id)
    if not loc:
        return False, "Not found in cmux (not running in a cmux pane, or already closed)"
    ws, sf = loc["workspace_ref"], loc["surface_ref"]
    # `send` just types at the current cursor position — it does NOT clear existing input.
    # If the box already had leftover text (e.g. an untyped/aborted command), our text would
    # just get appended onto it (observed: sending "/clear" produced "/clear/clear" because
    # the box already had "/clear" sitting in it). Clear the line both directions first.
    ok, msg = cmux_run("send-key", "--workspace", ws, "--surface", sf, "ctrl+u")
    if not ok:
        return False, f"clear-line failed: {msg}"
    ok, msg = cmux_run("send-key", "--workspace", ws, "--surface", sf, "ctrl+k")
    if not ok:
        return False, f"clear-line failed: {msg}"
    ok, msg = cmux_run("send", "--workspace", ws, "--surface", sf, text)
    if not ok:
        return False, f"send failed: {msg}"
    ok, msg = cmux_run("send-key", "--workspace", ws, "--surface", sf, "Enter")
    if not ok:
        return False, f"enter failed: {msg}"
    return True, f"Sent {text}"


def draw(stdscr):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_RED, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(4, curses.COLOR_RED, curses.COLOR_WHITE)
    RED, YELLOW, SELECTED, SELECTED_CRIT = (curses.color_pair(i) for i in (1, 2, 3, 4))
    DIM = curses.A_DIM

    curses.init_pair(5, curses.COLOR_GREEN, -1)
    GREEN = curses.color_pair(5)

    states = {}
    selected_sid = None
    tool_scroll = 0
    last_refresh = 0.0
    status_msg = ""
    status_ok = True
    status_expiry = 0.0
    pending_confirm = None  # {"action": "clear"|"compact", "sid": ...}
    show_help = False
    prev_show_help = False
    stdscr.timeout(100)  # getch blocks up to 100ms — keeps nav responsive between data refreshes

    col_headers = ["NAME", "STATUS", "CTX", "%LIM", "!", "LAST MSG", "CREATED"]
    widths = [16, 8, 8, 7, 3, 9, 9]

    HELP_LINES = [
        ("STATUS values", curses.A_BOLD),
        ("  idle       Sitting at the prompt, nothing running.", 0),
        ("  busy       Actively generating (model streaming a response).", 0),
        ("  shell      Currently running a shell/Bash command.", 0),
        ("  waiting    Blocked on a permission/approval dialog — needs your input.", 0),
        ("", 0),
        ("Columns", curses.A_BOLD),
        ("  CTX        Context size of the LAST turn (cache_read + cache_creation + input).", 0),
        ("  %LIM       CTX as a % of this session's model's context window (1M for Sonnet/Opus/", 0),
        ("             Fable 5, 200K default otherwise). THIS is the number to watch for compacting —", 0),
        ("             not cache hit rate. '-' = no turn yet. Row turns yellow at 60%, red at 85%.", 0),
        ("             A trailing '?' on CTX, or '?' in %LIM, means /compact ran but the session", 0),
        ("             hasn't had a real turn since — the numbers are stale until it does.", 0),
        ("  !          Number of active warnings (see the detail panel below for what they are).", 0),
        ("  LAST MSG   Time since the last real conversation turn (not process uptime).", 0),
        ("  CREATED    Time since this terminal/process started (can predate LAST MSG on a", 0),
        ("             --resume'd conversation, or postdate it right after a /clear).", 0),
        ("", 0),
        ("Keys", curses.A_BOLD),
        ("  up/down       select a row", 0),
        ("  Enter / g     jump to that session's tab in cmux", 0),
        ("  c             send /clear (asks y/N; queues if the session is busy)", 0),
        ("  k             send /compact (asks y/N; queues if the session is busy)", 0),
        ("  PgUp/PgDn     scroll the tool list in the detail panel", 0),
        ("  ?             toggle this help", 0),
        ("  q             quit", 0),
        ("", 0),
        ("Press any key to close.", curses.A_DIM),
    ]

    def set_status(msg, ok=True):
        nonlocal status_msg, status_ok, status_expiry
        status_msg = msg
        status_ok = ok
        status_expiry = time.time() + 4

    while True:
        now = time.time()
        if now - last_refresh >= REFRESH_SECONDS:
            sessions = load_sessions()
            live_ids = set()
            for pid_str, w in sessions.items():
                pid = w.get("pid")
                if not pid or not pid_alive(pid):
                    continue
                sid = w.get("sessionId")
                if not sid:
                    continue
                live_ids.add(sid)
                if sid not in states:
                    states[sid] = SessionState(sid)
                st = states[sid]
                st.name = w.get("name", pid_str)
                st.status = w.get("status", "-")
                st.cwd = w.get("cwd", "?")
                st.pid = pid
                if w.get("startedAt"):
                    st.created_ms = w["startedAt"]
                st.update()

            for sid in list(states.keys()):
                if sid not in live_ids:
                    del states[sid]
            last_refresh = now

        rows = sorted(states.items(), key=lambda kv: -(kv[1].last_msg_ts_ms or 0))
        if selected_sid not in states and rows:
            selected_sid = rows[0][0]

        if show_help != prev_show_help:
            stdscr.clear()  # full repaint on transition — erase() alone can leave stale cells behind
        else:
            stdscr.erase()
        prev_show_help = show_help
        h, w_ = stdscr.getmaxyx()

        if show_help:
            safe_addstr(stdscr, 0, 0, "Claude Code dashboard — help"[: w_ - 1], curses.A_BOLD)
            safe_addstr(stdscr, 1, 0, "-" * (w_ - 1))
            for i, (line, attr) in enumerate(HELP_LINES):
                if i + 2 >= h:
                    break
                safe_addstr(stdscr, i + 2, 0, line[: w_ - 1], attr)
            stdscr.refresh()
            c = stdscr.getch()
            if c != -1:
                show_help = False
            time.sleep(0.05)
            continue

        # Give the table only the rows it needs; the detail panel gets whatever's left,
        # so the tool list has as much room as possible (still scrollable if it overflows).
        table_needed = len(rows) + 4
        detail_min = 10
        table_h = table_needed if h - table_needed >= detail_min else max(5, h - detail_min)

        safe_addstr(
            stdscr, 0, 0,
            f"Claude Code — {len(states)} running, sorted by last message  [↑/↓ select, Enter/g cmux, c clear, k compact, ? help, q quit]"[: w_ - 1],
        )

        if pending_confirm:
            sel = states.get(pending_confirm["sid"])
            sel_name = sel.name if sel else "?"
            action_word = "/clear" if pending_confirm["action"] == "clear" else "/compact"
            queued_note = " (session is busy — it'll queue and run once its current turn finishes)" if sel and sel.status == "busy" else ""
            safe_addstr(
                stdscr, 1, 0,
                f"Send {action_word} to {sel_name}?{queued_note} This changes its live conversation. [y/N]"[: w_ - 1],
                curses.A_BOLD | YELLOW,
            )
        elif status_msg and time.time() < status_expiry:
            safe_addstr(stdscr, 1, 0, status_msg[: w_ - 1], GREEN if status_ok else RED)
        else:
            safe_addstr(stdscr, 1, 0, "-" * (w_ - 1))

        row = 2
        x = 0
        for hname, cw in zip(col_headers, widths):
            safe_addstr(stdscr, row, x, hname[:cw].ljust(cw), curses.A_BOLD)
            x += cw + 1
        row += 1
        safe_addstr(stdscr, row, 0, "-" * (w_ - 1))
        row += 1

        for sid, st in rows:
            if row >= table_h - 1:
                break
            warn = st.warnings()
            has_crit = any(sev == "crit" for sev, _ in warn)
            is_selected = sid == selected_sid

            if is_selected:
                attr = SELECTED_CRIT if has_crit else SELECTED
            elif has_crit:
                attr = RED
            elif warn:
                attr = YELLOW
            else:
                attr = curses.A_NORMAL

            vals = [
                st.name,
                st.status,
                f"{fmt_k(st.context_tokens)}?" if st.compact_pending else fmt_k(st.context_tokens),
                "?" if st.compact_pending else (
                    f"{st.ctx_pct:.0f}%" if st.ctx_pct is not None else "-"
                ),
                str(len(warn)) if warn else "-",
                ago_str(st.last_msg_ts_ms) if st.last_msg_ts_ms else "-",
                ago_str(st.created_ms) if st.created_ms else "-",
            ]
            x = 0
            for val, cw in zip(vals, widths):
                safe_addstr(stdscr, row, x, str(val)[:cw].ljust(cw), attr)
                x += cw + 1
            row += 1

        # ---- Detail panel for the selected session ----
        safe_addstr(stdscr, table_h, 0, "=" * (w_ - 1))
        sel = states.get(selected_sid)
        drow = table_h + 1
        if sel:
            safe_addstr(stdscr, drow, 0, f"{sel.name}  ({sel.status})"[: w_ - 1], curses.A_BOLD)
            drow += 1
            cache_str = f"{sel.cache_ratio * 100:.0f}%" if sel.cache_ratio is not None else "-"
            safe_addstr(
                stdscr, drow, 0,
                f"model: {(sel.model or '?').replace('claude-', '')}   pid: {sel.pid}   "
                f"cache hit: {cache_str}   cwd: {sel.cwd}"[: w_ - 1],
                DIM,
            )
            drow += 1
            safe_addstr(
                stdscr, drow, 0,
                f"lifetime output: {fmt_k(sel.total_output)} tokens "
                "(not context — includes anything discarded by past /compact runs)"[: w_ - 1],
                DIM,
            )
            drow += 1
            created = f"{abs_str(sel.created_ms)} ({ago_str(sel.created_ms)} ago)" if sel.created_ms else "?"
            last_msg = f"{abs_str(sel.last_msg_ts_ms)} ({ago_str(sel.last_msg_ts_ms)} ago)" if sel.last_msg_ts_ms else "no messages yet"
            safe_addstr(stdscr, drow, 0, f"created: {created}   last message: {last_msg}"[: w_ - 1], DIM)
            drow += 1

            warn = sel.warnings()
            if warn:
                for sev, msg in warn:
                    if drow >= h - 4:
                        break
                    attr = RED if sev == "crit" else YELLOW
                    safe_addstr(stdscr, drow, 0, f"! {msg}"[: w_ - 1], attr)
                    drow += 1
            else:
                safe_addstr(stdscr, drow, 0, "No issues detected.", DIM)
                drow += 1

            drow += 1
            all_tools = sorted(sel.tools.items(), key=lambda kv: -kv[1])
            skills_text = f"Skills used: {sel.all_skills()}"
            skills_lines = textwrap.wrap(
                skills_text, width=max(20, w_ - 1), subsequent_indent="  "
            ) or [skills_text]
            skills_line_reserved = len(skills_lines) + 1  # +1 for the blank line before it
            if all_tools:
                header_row = drow
                drow += 1
                list_start = drow
                list_end = h - skills_line_reserved  # leave room for the skills line(s)
                capacity = max(1, list_end - list_start)

                tool_scroll = max(0, min(tool_scroll, max(0, len(all_tools) - capacity)))
                visible = all_tools[tool_scroll : tool_scroll + capacity]

                scrolled = len(all_tools) > capacity
                label = "Tools used:"
                if scrolled:
                    lo, hi = tool_scroll + 1, tool_scroll + len(visible)
                    label = f"Tools used ({lo}-{hi} of {len(all_tools)}, PgUp/PgDn to scroll):"
                safe_addstr(stdscr, header_row, 0, label[: w_ - 1], curses.A_BOLD)

                max_count = max(c for _, c in all_tools)
                bar_budget = max(10, w_ - 34)
                for name, count in visible:
                    bar_len = max(1, int(count / max_count * bar_budget))
                    line = f"  {name[:24]:24} {count:>5}  " + "#" * bar_len
                    safe_addstr(stdscr, drow, 0, line[: w_ - 1])
                    drow += 1
                drow = max(drow, list_end)
            drow += 1
            for line in skills_lines:
                if drow >= h:
                    break
                safe_addstr(stdscr, drow, 0, line[: w_ - 1])
                drow += 1

        stdscr.refresh()

        c = stdscr.getch()

        if pending_confirm is not None:
            if c in (ord("y"), ord("Y")):
                sid = pending_confirm["sid"]
                cmd = "/clear" if pending_confirm["action"] == "clear" else "/compact"
                if sid in states:
                    ok, msg = cmux_send_command(sid, cmd)
                    set_status(f"{cmd} -> {states[sid].name}: {msg}" if ok else f"Failed: {msg}", ok)
                pending_confirm = None
            elif c != -1:
                set_status("Cancelled.", True)
                pending_confirm = None
            continue

        if c in (ord("q"), ord("Q")):
            break
        elif c == ord("?"):
            show_help = True
        elif c in (curses.KEY_UP, curses.KEY_DOWN) and rows:
            ids = [sid for sid, _ in rows]
            idx = ids.index(selected_sid) if selected_sid in ids else 0
            idx = (idx - 1) if c == curses.KEY_UP else (idx + 1)
            selected_sid = ids[idx % len(ids)]
            tool_scroll = 0
        elif c == curses.KEY_NPAGE:
            tool_scroll += 5
        elif c == curses.KEY_PPAGE:
            tool_scroll = max(0, tool_scroll - 5)
        elif c in (10, 13, curses.KEY_ENTER, ord("g"), ord("G")) and selected_sid:
            ok, msg = cmux_navigate(selected_sid)
            set_status(msg, ok)
        elif c in (ord("c"), ord("C")) and selected_sid in states:
            pending_confirm = {"action": "clear", "sid": selected_sid}
        elif c in (ord("k"), ord("K")) and selected_sid in states:
            pending_confirm = {"action": "compact", "sid": selected_sid}


if __name__ == "__main__":
    locale.setlocale(locale.LC_ALL, "")
    curses.wrapper(draw)
