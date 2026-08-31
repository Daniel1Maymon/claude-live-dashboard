# claude-live-dashboard

A live terminal dashboard for monitoring all your locally-running [Claude Code](https://claude.com/claude-code) CLI sessions at once — built to make it easy to spot a session that's burning context, looping on the same tool call, or stalled while busy.

![status](https://img.shields.io/badge/status-personal_tool-blue)

## What it does

Claude Code writes a small registry file per running session (`~/.claude/sessions/*.json`) and a full JSONL transcript per session (`~/.claude/projects/*/`). This tool reads both, incrementally tails every transcript, and renders a live-updating table in your terminal — no browser, no server, just `curses`.

For each session you get:

- **Context size** of the last turn, with warning/critical color thresholds
- **Cache hit ratio** for that turn (low cache = expensive turn)
- **Cumulative output tokens** for the session
- **Tool usage breakdown** (which tools, how often) and any Skills invoked
- **Tool error count**
- **Loop detection** — flags a session repeating the same tool call back-to-back
- **Stall detection** — flags a session stuck `busy` for an unusually long time
- Last-message time and session-created time, live-sorted by most recent activity

## Why

If you run several Claude Code sessions in parallel (e.g. across `cmux` workspaces), it's easy to lose track of which one has quietly ballooned to a huge context window or gotten stuck retrying the same failing tool call. This dashboard surfaces that at a glance instead of tabbing through every session individually.

## Requirements

- Python 3 (standard library only — `curses`, no third-party dependencies)
- macOS or Linux terminal
- Optional: [cmux](https://github.com) terminal multiplexer, for the jump-to-session feature

## Usage

```bash
python3 claude_live_dashboard.py
```

### Keybindings

| Key | Action |
|---|---|
| `↑` / `↓` | Select a session row |
| `Enter` / `g` | Jump to that session's tab (requires cmux) |
| `c` | Send `/clear` to the selected session (asks `y/N`; queues if the session is busy) |
| `k` | Send `/compact` to the selected session (asks `y/N`; queues if the session is busy) |
| `PgUp` / `PgDn` | Scroll the tool list in the detail panel |
| `?` | Toggle a glossary explaining statuses, columns, and keybindings |
| `q` | Quit |

### Columns

| Column | Meaning |
|---|---|
| `STATUS` | `idle`, `busy` (generating), `shell` (running a command), or `waiting` (blocked on a permission dialog) |
| `CTX` | Context size of the *last* turn only (cache read + cache creation + input tokens) |
| `CACHE%` | Share of that turn's context served from prompt cache (`-` if no turn has happened yet) |
| `OUT` | Total output tokens generated so far this session |
| `!` | Number of active warnings for that session |
| `LAST MSG` | Time since the last real conversation turn |
| `CREATED` | Time since the session's process started |

## How it works

- Session discovery: polls `~/.claude/sessions/*.json` for live PIDs.
- Transcript parsing: tails each session's `.jsonl` transcript from a saved byte offset, so re-reads stay cheap even for very long sessions.
- `/clear` detection: the tool recognizes an immediate `/clear` command in the transcript and resets its own counters right away, rather than waiting for the next turn's usage stats (which lag behind the model's actual reset).
- cmux integration: resolves a session's `sessionId` to its exact cmux workspace/surface and focuses it directly, rather than just switching workspace-level focus.

## Notes

This was built as a personal tool against the current on-disk layout of Claude Code's session/transcript files. That layout is an implementation detail, not a stable public API, so future Claude Code versions may change it in ways that break this script.
