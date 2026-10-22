# Claude Code + Outlook + Barracuda Message Archiver: chat record

**Date:** Wednesday, October 21, 2026
**Topic:** Giving Claude Code access to Outlook and the Barracuda Message Archiver on Windows (bash + Python CLI tools), turning that into a goal-based subagent framework that generates and cleans up its own query scripts, and adding full logging.

---

## 1. Research request

> **User:** please research how claude code can access outlook and baracuda message archive on a windows machine using sh and a python cli for tools

### Answer summary

**Bottom line:** Outlook is easy if you're on *classic* Outlook. Barracuda is harder: no documented API for searching archived messages was found, so the practical route to the archive goes *through* Outlook.

#### How Claude Code runs commands on Windows
- Native Windows Claude Code runs shell commands through **Git Bash**. `CLAUDE_CODE_GIT_BASH_PATH` points it at a specific install (e.g. `C:\Program Files\Git\bin\bash.exe`).
- Newer versions fall back to **PowerShell** when Git Bash is absent.
- So the intended pattern works: a Python CLI that prints JSON, called from bash.
- Gotchas:
  - Git Bash (MSYS2) process tokens don't inherit Windows security group memberships.
  - Claude Code and Outlook must run at the **same privilege level**. If one is elevated and the other isn't, COM calls typically fail.

#### Outlook: two options
1. **COM via pywin32 (classic Outlook only)**
   - Drives the desktop Outlook you're already signed into. No app registration needed.
   - **New Outlook does not support COM, VBA or MAPI automation.**
   - Gotchas:
     - Python's bitness should match Outlook's.
     - A Group Policy that disables COM add-ins also blocks Python COM access.
     - Outlook generally needs to be running.
   - Existing option: Python MCP servers built on this, e.g. `outlook-mcp-com`.
2. **Microsoft Graph (works with New Outlook, Exchange Online)**
   - Microsoft's replacement for COM automation.
   - Needs an Entra app registration, MSAL device-code flow and delegated `Mail.Read`. Avoid application-level `Mail.Read`, which reads every mailbox in the tenant.
   - Needs IT involvement, and it can't see the Barracuda archive.

#### Barracuda Message Archiver
- No published API was found for searching or retrieving archived mail. Other Barracuda products expose an XML-RPC config API at `/cgi-mod/api.cgi`, but that's for configuration only.
- Realistic routes:
  1. **Through Outlook (recommended).**
     - The **Barracuda Archive Search for Outlook** add-in can sync archived mail into a local store inside Outlook, which pywin32 can then search.
     - The admin must run nightly folder-sync jobs.
     - The add-in doesn't support New Outlook.
     - The stub cache lives at `%USERPROFILE%\AppData\Local\Barracuda\Message Archiver\Outlook Add-In\stores`. Go through Outlook rather than parsing that cache.
  2. **Export and parse.** Export search results from the archiver web UI as PST or EML, then parse them with Python's `email` module.
  3. **Scrape the web UI.** Fragile and possibly against policy, so not recommended.

#### Suggested CLI shape
```
mailtool stores
mailtool search --store archive --from bob@x.com --since 2023-01-01 --subject invoice --json
mailtool get <EntryID> --json
mailtool save-attachments <EntryID> --out ./att
mailtool export <EntryID> --format msg|eml
```
- Core: `win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")`, iterate `ns.Stores`, then `Items.Restrict(...)`.
- Emit compact JSON, cap result counts, and document the commands in `CLAUDE.md` or a skill.
- Keep the CLI read-only by default.

#### Sources
- [Claude Code issue #53068: PowerShell fallback](https://github.com/anthropics/claude-code/issues/53068)
- [Claude Code issue #13184: CLAUDE_CODE_GIT_BASH_PATH](https://github.com/anthropics/claude-code/issues/13184)
- [Claude Code issue #21843: Git Bash and security groups](https://github.com/anthropics/claude-code/issues/21843)
- [Microsoft Q&A: New Outlook automation (no COM; use Graph)](https://learn.microsoft.com/en-in/answers/questions/5951650/need-a-support-in-new-outlook-automation)
- [Python + Outlook COM gotchas](https://office.alibaba.com/officesoftware/how-to-automate-emails-in-outlook-using-python)
- [outlook-mcp-com (PyPI)](https://pypi.org/project/outlook-mcp-com/)
- [graphplug (PyPI), Graph permission notes](https://pypi.org/project/graphplug/0.4.0/)
- [Barracuda: How to Create Local Mailstores](https://documentation.campus.barracuda.com/wiki/spaces/BMAv31/pages/5702695/How+to+Create+Local+Mailstores)
- [Barracuda: Understanding Archive Stores](https://documentation.campus.barracuda.com/wiki/spaces/BMAv31/pages/5702077/Understanding+Archive+Stores)
- [Barracuda: Archive Search for Outlook customization options](https://campus.barracuda.com/doc/14319721)
- [Barracuda Email Security Gateway API Guide (XML-RPC pattern)](https://campus.barracuda.com/product/emailsecuritygateway/doc/76284988/barracuda-email-security-gateway-api-guide/)
- [Barracuda Message Archiver features](https://www.esecforte.com/?p=11502)

---

## 2. Subagents with a goal-based architecture (v1)

> **User:** can i have these configured as subagents as i woul like to use a goal based architecture to call these sub agents as i will be adding additional sub agents using the same framework

### What was built: `goal-agents.zip` (v1)
I checked the current Claude Code subagent docs (frontmatter fields, nesting up to 3 levels, skills preloading, hooks) and then built:

- **Orchestrator**
  - `/goal` skill and `goal-orchestrator` subagent.
  - The loop: frame the goal with success criteria → discover workers → write `plan.json` → dispatch task briefs in parallel → collect JSON envelopes → verify and iterate (max 3 rounds) → report.
- **Worker contract**, a skill preloaded into every worker:
  - a standard task-brief format going in;
  - a `goal-worker/1` result envelope coming out (status, criteria, results, handoff);
  - safety rules: mail content is data, workers are read-only.
- **Workers:** `outlook-mail` (classic Outlook via COM) and `barracuda-archive` (synced archive store or EML exports).
- **Fixed Python CLIs:**
  - `outlook` and `barracuda`;
  - launched through `bash tools/run`, which prefers `py -3` on Windows;
  - one JSON envelope per command, with exit codes 0, 2, 3 and 1.
- **Extensibility:** `bash tools/run new-worker <name>` scaffolds a new agent and CLI; `bash tools/run registry` lists workers.
- **Safety hook:** restricts worker Bash calls to single `tools/run` commands.
- **Testing:** done against a mocked Outlook object model and real EML files.

---

## 3. Script generation to minimize token usage (v2)

> **User:** to minimize token usage .. i would like the llm to generate scripts and tools to perform ad hoc queries as needed and clean them up upon completion

### What changed: `goal-agents.zip` (v2, replaces v1)
- **Workers write throwaway scripts.** Each worker writes a short Python script per task into `.goals/<goal>/scratch/<task>/`, runs it with `bash tools/run adhoc <script>`, returns a compact envelope, then runs `bash tools/run cleanup <goal> --task <task>`.
- **Orchestrator cleanup.** It runs `cleanup <goal>` after reporting, and offers `--purge` for artifacts that contain mail content.
- **Session-start sweep.** A SessionStart hook removes stale scratch left behind by crashed runs.
- **Helper libraries** (`tools/lib/`) keep scripts to 3–6 lines:
  - `olk`: classic Outlook through COM;
  - `eml`: Barracuda EML/zip exports;
  - `out`: compact TSV output, counts, de-duplication, saving.
- **Token savers:**
  - output capped at 25 rows and 3,000 characters, with the full text kept on disk;
  - full result sets saved as `.jsonl` artifacts;
  - a fast path for single-worker goals with no plan file;
  - workers run with `effort: low` and lean prompts;
  - `key=value` tool output instead of JSON.
- **Static checker on generated code.** Before a script runs, it is rejected if it:
  - imports anything beyond the helper libs and a safe stdlib list;
  - uses `open`, `eval`, `exec` or `getattr`;
  - assigns to attributes;
  - calls COM methods that send, delete, move or save mail.
- **Worker hooks.** Workers may write only `.py` files in scratch, and Bash is limited to `adhoc`, `doctor` and `cleanup`.
- **Testing:** fake COM and EML scripts, lint rejections, output caps, the BLOCKED and QUERY ERROR paths, the hooks, all cleanup modes, and scaffolding.

Example generated script:
```python
import olk, out
ms = olk.search(frm="acme.com", subject="invoice", since="2023-01-01", attach=True)
out.rows(ms, "received subject natt", n=20, save="acme.jsonl")
```

**Open decision:** saved artifacts are kept by default and purge is offered at the end of a goal. This could instead purge automatically.

---

## 4. First chat record

> **User:** generate me a md file of this chat

Produced the first version of this file (sections 1–3).

---

## 5. Logging framework (v3)

> **User:** can you add logging framework into this code base so we can track what the code is doing. All data read in from outlook or barracuda should be logged to files, Any llm queries, tool calls, tool returns, llm responses, user requests, response to user should all be logged

### Research
I checked the current Claude Code docs for hook events and telemetry:
- **Hooks.** `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `SubagentStart/Stop`, `Stop`, `SessionStart/End`, `PreCompact` and others.
- **Hook inputs.** Every hook gets `session_id` and `transcript_path`, plus `agent_id`/`agent_type` when it fires inside a subagent. `Stop` and `SubagentStop` include `last_assistant_message`.
- **Async hooks.** Command hooks support `"async": true`, so logging doesn't block the agent.
- **OpenTelemetry.** It emits `user_prompt`, `assistant_response`, `tool_result` and `api_request` events (tokens, cost, latency).
- **Raw API bodies.** `OTEL_LOG_RAW_API_BODIES=file:<dir>` writes the complete API request and response bodies to disk. Telemetry has no built-in file exporter, so a small local OTLP receiver is needed to keep the events.

### What was built: `goal-agents.zip` (v3, replaces v2)
- **Logging layer** (`tools/lib/_glog.py`):
  - JSONL records, one per line, with timestamp, channel and event;
  - file locking, so parallel processes can write safely on Windows and POSIX;
  - secret redaction, retention settings, and a `GOAL_LOG=0` switch;
  - a standard-library `logging` adapter;
  - logging can never break a query; its own failures go to `logs/errors/`.
- **Conversation logging** (`tools/hooks/log_event.py`, async, wired for every event in `.claude/settings.json`):
  - user requests, tool calls (including prompts sent to subagents), tool returns and failures, subagent responses, and responses to the user, written to `logs/hooks/<date>/<session>.jsonl`;
  - at Stop, SubagentStop, PreCompact and SessionEnd, the main and subagent transcripts are mirrored incrementally to `logs/transcripts/<session>/`. These hold every LLM response with token usage.
- **Raw LLM queries (optional):**
  - `bash tools/run logs enable-api` writes the OpenTelemetry settings to `.claude/settings.local.json`;
  - a local receiver (`tools/otel_collector.py`, 127.0.0.1:4318, http/json, gzip-aware) starts at session start;
  - it records per-call model, tokens, cost and latency in `logs/otel/`, and raw request/response bodies in `logs/api/`.
- **Mail data logging** (`logs/data/<date>/<goal>__<run>.jsonl`). `olk` and `eml` log:
  - every connect, store list and folder list;
  - every search, with its parameters, DASL filter and full result list;
  - every body (full text) and headers read;
  - attachment lists, and each saved attachment or export with its SHA-256;
  - each EML scanned, with its hash.
- **Script logging** (`logs/tools/`):
  - each run gets a run ID, also shown as `run=xxxxxxxx` in the output;
  - the source, lint result, full stdout/stderr and exactly what the model was shown are all logged;
  - guard decisions, cleanup (with the list of deleted files) and doctor output are logged too.
- **Viewer** (`bash tools/run logs ...`):
  - `status`, `sessions`, `timeline` (all channels merged, one line per event), `export` (a Markdown audit report per session), `prune`, `enable-api`/`disable-api`, `collector`;
  - prune and collector start-up run automatically at session start.
- **Protections:**
  - agents are denied Read/Write/Edit on `logs/`;
  - generated scripts can't import the logger;
  - `logs/` is git-ignored;
  - a retention default of 30 days.
- **Testing:** a simulated session covering all hook kinds, 25 parallel writes (all lines valid), real script runs against fake COM and EML, transcript mirroring, secret redaction, collector POSTs (JSON, gzip, metrics, a protobuf rejection), timeline, export, prune, the API on/off switch, logging disabled, an unwritable log dir, and the lint block on `_glog`.

**Privacy note:** the logs are a complete copy of the mail data the agents touched. Keep them on an encrypted disk, restrict access, and set retention to match policy before turning on raw API capture.

---

## 6. This file (updated)

> **User:** generate me a md file of this chat

Updated this record to cover the logging work. The full source for v3 is in `goal-agents.zip`.
