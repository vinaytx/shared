"""
PDF Section Comparator
======================
Reads the extracted section data saved by pdf_section_extractor.py and
produces a side-by-side comparison with AI-generated difference summaries.

Workflow (modular — run each step independently):
─────────────────────────────────────────────────
  Step 1 — extract PDF A:
      python pdf_section_extractor.py doc_a.pdf extracted_a/

  Step 2 — extract PDF B:
      python pdf_section_extractor.py doc_b.pdf extracted_b/

  Step 3 — compare:
      python pdf_compare.py extracted_a/ extracted_b/

Each step reads from and writes to files, so any step can be re-run
independently without repeating the others.

File I/O
────────
  Inputs  (produced by pdf_section_extractor.py):
      <dir_a>/_sections.json
      <dir_b>/_sections.json

  Outputs (written to output_dir/):
      _COMPARISON_REPORT.html   Full side-by-side diff report (default)
      _COMPARISON_REPORT.txt    Plain-text report (--report-format text)
      _MATCHED.json             Machine-readable match + summary data
      matched/                  One .txt diff file per matched section pair
      unmatched_a/              Sections only found in A
      unmatched_b/              Sections only found in B

Options:
    --text-model MODEL      Ollama text model for AI summaries (default: llama3.2)
    --ollama-url URL        Ollama server URL (default: http://localhost:11434)
    --skip-summary          Skip AI summaries (diff only, much faster)
    --fuzzy-threshold N     Min title similarity 0–100 to count as a match (default: 75)
    --report-format FORMAT  text | html (default: html)

Requirements:
    pip install requests
    (pdf_section_extractor.py must be in the same directory)
"""

import os
import sys
import re
import json
import argparse
import difflib
import textwrap
from pathlib import Path
from datetime import datetime

# ── Import only the loader + data classes from the extractor ─────────────────
_HERE = Path(globals().get("__file__", ".")).parent
sys.path.insert(0, str(_HERE))

try:
    from pdf_section_extractor import (
        load_sections_json,
        DEFAULT_OLLAMA_URL,
    )
except ImportError:
    print("✗  Could not import pdf_section_extractor.py")
    print("   Make sure it is in the same directory as this script.")
    sys.exit(1)

# Text LLM used for summarising differences (separate from the vision model)
DEFAULT_TEXT_MODEL = "llama3.2"   # any Ollama text model works: llama3, mistral, phi3, etc.

# Prompt sent to the text LLM for each section pair
SUMMARY_SYSTEM_PROMPT = (
    "You are a precise technical document analyst. "
    "You will be given two versions of the same document section. "
    "Your job is to write a clear, concise summary of the differences between them. "
    "Focus on: added content, removed content, changed facts or figures, "
    "reworded claims, structural changes, and tone shifts. "
    "Be specific — mention actual words, numbers, or phrases that changed when relevant. "
    "If the sections are identical, say so in one sentence. "
    "Use plain prose, no bullet points, no markdown."
)

SUMMARY_USER_TEMPLATE = """\
Section title: {title}

=== VERSION A ({name_a}) ===
{text_a}

=== VERSION B ({name_b}) ===
{text_b}

Write a concise summary of the differences between Version A and Version B of this section.\
"""

# Maximum characters of each section sent to the LLM (keeps prompts manageable)
SUMMARY_MAX_CHARS = 4000


# ══════════════════════════════════════════════════════════════════
# OLLAMA TEXT SUMMARISATION
# ══════════════════════════════════════════════════════════════════

def check_text_model(base_url: str, model: str) -> bool:
    """Return True if Ollama is running and the text model is available."""
    try:
        import requests as _req
        resp = _req.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        available = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        if model.split(":")[0] not in available:
            print(f"  ⚠  Text model '{model}' not found.")
            print(f"     Available: {', '.join(available) or 'none'}")
            print(f"     Run: ollama pull {model}")
            return False
        return True
    except Exception as exc:
        print(f"  ⚠  Ollama check failed: {exc}")
        return False


def summarise_diff_with_ollama(
    text_a:   str,
    text_b:   str,
    title:    str,
    name_a:   str,
    name_b:   str,
    base_url: str,
    model:    str,
    retries:  int = 2,
) -> str:
    """
    Ask the Ollama text LLM to write a plain-English summary of the differences
    between text_a and text_b.  Returns the summary string, or an error message.
    """
    import requests as _req

    # Truncate to keep prompts manageable
    ta = text_a[:SUMMARY_MAX_CHARS]
    tb = text_b[:SUMMARY_MAX_CHARS]
    truncated = len(text_a) > SUMMARY_MAX_CHARS or len(text_b) > SUMMARY_MAX_CHARS
    note = "\n[Note: sections were truncated to fit context window.]" if truncated else ""

    prompt = SUMMARY_USER_TEMPLATE.format(
        title=title, name_a=name_a, name_b=name_b,
        text_a=ta + note, text_b=tb + note,
    )

    payload = {
        "model":  model,
        "system": SUMMARY_SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,     # slight creativity, mostly factual
            "num_predict": 512,
        },
    }

    for attempt in range(retries + 1):
        try:
            resp = _req.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except _req.exceptions.Timeout:
            if attempt < retries:
                print(f"    ⏱  Timeout — retrying ({attempt + 1}/{retries})...")
                import time; time.sleep(2)
            else:
                return "[Summary timed out]"
        except Exception as exc:
            if attempt < retries:
                import time; time.sleep(1)
            else:
                return f"[Summary error: {exc}]"
    return "[Summary unavailable]"


def summarise_all_matches(
    matched:      list,
    name_a:       str,
    name_b:       str,
    base_url:     str,
    model:        str,
    skip_summary: bool = False,
) -> list:
    """
    For each (sec_a, sec_b, score) tuple in matched, generate an LLM summary
    and return an extended list of (sec_a, sec_b, score, summary) tuples.

    Identical sections get a canned one-liner; changed sections are sent to Ollama.
    """
    if skip_summary:
        return [(sa, sb, sc, "") for sa, sb, sc in matched]

    ollama_ok = check_text_model(base_url, model)
    if not ollama_ok:
        print("  ⚠  LLM summaries disabled.\n")
        return [(sa, sb, sc, "[LLM unavailable — run: ollama pull " + model + "]")
                for sa, sb, sc in matched]

    print(f"  Using model '{model}' to summarise {len(matched)} section(s)...\n")
    results = []

    for i, (sa, sb, score) in enumerate(matched, 1):
        stats = text_diff_stats(sa.content(), sb.content())
        sim   = stats["line_similarity_pct"]
        print(f"  [{i}/{len(matched)}] {sa.title}  ({sim}% similar)", end=" ... ", flush=True)

        if sim == 100.0:
            summary = "The two versions of this section are identical — no differences found."
            print("identical, skipped")
        else:
            summary = summarise_diff_with_ollama(
                sa.content(), sb.content(),
                title=sa.title, name_a=name_a, name_b=name_b,
                base_url=base_url, model=model,
            )
            preview = summary[:80].replace("\n", " ")
            suffix  = "..." if len(summary) > 80 else ""
            print(f'✓ "{preview}{suffix}"')

        results.append((sa, sb, score, summary))

    return results


# ══════════════════════════════════════════════════════════════════
# TITLE NORMALISATION & MATCHING
# ══════════════════════════════════════════════════════════════════

def normalise_title(title: str) -> str:
    """
    Strip numbering, punctuation, and whitespace so that titles like
    "1.2 Background" and "1.3 Background" still match on the word 'Background'.
    """
    t = title.lower().strip()
    t = re.sub(r"^[\d\.\:\-\s]+", "", t)          # leading numbers / dots
    t = re.sub(r"[^\w\s]", " ", t)                 # punctuation → space
    t = re.sub(r"\s+", " ", t).strip()
    return t


def title_similarity(a: str, b: str) -> float:
    """
    Return a 0–100 similarity score between two normalised titles using
    SequenceMatcher (same algorithm as difflib.get_close_matches).
    """
    na, nb = normalise_title(a), normalise_title(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio() * 100


def match_sections(sections_a: list, sections_b: list, threshold: float = 75.0):
    """
    Greedily match sections from A to sections in B by title similarity.

    Returns:
        matched   : list of (section_a, section_b, score)
        only_in_a : list of section_a with no match in B
        only_in_b : list of section_b with no match in A
    """
    used_b  = set()
    matched = []

    for sec_a in sections_a:
        best_score = 0.0
        best_b     = None
        best_idx   = -1

        for idx, sec_b in enumerate(sections_b):
            if idx in used_b:
                continue
            # Only match sections of the same structural level
            if sec_a.level != sec_b.level:
                continue
            score = title_similarity(sec_a.title, sec_b.title)
            if score > best_score:
                best_score = score
                best_b     = sec_b
                best_idx   = idx

        if best_b is not None and best_score >= threshold:
            matched.append((sec_a, best_b, best_score))
            used_b.add(best_idx)

    matched_a   = {id(m[0]) for m in matched}
    matched_b   = {id(m[1]) for m in matched}
    only_in_a   = [s for s in sections_a if id(s) not in matched_a]
    only_in_b   = [s for s in sections_b if id(s) not in matched_b]

    return matched, only_in_a, only_in_b


# ══════════════════════════════════════════════════════════════════
# DIFF HELPERS
# ══════════════════════════════════════════════════════════════════

def text_diff_stats(text_a: str, text_b: str) -> dict:
    """Compute word-level and line-level change statistics."""
    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()
    words_a = text_a.split()
    words_b = text_b.split()

    line_ratio = difflib.SequenceMatcher(None, lines_a, lines_b).ratio()
    word_ratio = difflib.SequenceMatcher(None, words_a, words_b).ratio()

    opcodes    = difflib.SequenceMatcher(None, lines_a, lines_b).get_opcodes()
    added      = sum(j2 - j1 for tag, i1, i2, j1, j2 in opcodes if tag in ("insert", "replace"))
    removed    = sum(i2 - i1 for tag, i1, i2, j1, j2 in opcodes if tag in ("delete", "replace"))
    unchanged  = sum(i2 - i1 for tag, i1, i2, j1, j2 in opcodes if tag == "equal")

    return {
        "line_similarity_pct": round(line_ratio * 100, 1),
        "word_similarity_pct": round(word_ratio * 100, 1),
        "lines_added":   added,
        "lines_removed": removed,
        "lines_unchanged": unchanged,
        "total_lines_a": len(lines_a),
        "total_lines_b": len(lines_b),
        "total_words_a": len(words_a),
        "total_words_b": len(words_b),
    }


def unified_diff(text_a: str, text_b: str, label_a: str, label_b: str) -> str:
    """Return a unified diff string between two texts."""
    lines_a = text_a.splitlines(keepends=True)
    lines_b = text_b.splitlines(keepends=True)
    diff    = difflib.unified_diff(
        lines_a, lines_b,
        fromfile=label_a, tofile=label_b,
        lineterm=""
    )
    return "\n".join(diff)


def side_by_side_diff(text_a: str, text_b: str, width: int = 60) -> list:
    """
    Return a list of (left_line, right_line, change_type) tuples.
    change_type: 'equal' | 'replace' | 'insert' | 'delete'
    """
    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()
    sm      = difflib.SequenceMatcher(None, lines_a, lines_b)
    rows    = []

    def pad(s, w):
        return s[:w].ljust(w) if len(s) <= w else s[:w - 1] + "…"

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for la, lb in zip(lines_a[i1:i2], lines_b[j1:j2]):
                rows.append((pad(la, width), pad(lb, width), "equal"))
        elif tag == "replace":
            block_a = lines_a[i1:i2]
            block_b = lines_b[j1:j2]
            for la, lb in zip(block_a, block_b):
                rows.append((pad(la, width), pad(lb, width), "replace"))
            for la in block_a[len(block_b):]:
                rows.append((pad(la, width), " " * width, "delete"))
            for lb in block_b[len(block_a):]:
                rows.append((" " * width, pad(lb, width), "insert"))
        elif tag == "delete":
            for la in lines_a[i1:i2]:
                rows.append((pad(la, width), " " * width, "delete"))
        elif tag == "insert":
            for lb in lines_b[j1:j2]:
                rows.append((" " * width, pad(lb, width), "insert"))

    return rows


# ══════════════════════════════════════════════════════════════════
# TEXT REPORT
# ══════════════════════════════════════════════════════════════════

def write_text_report(
    matched, only_a, only_b,
    name_a, name_b,
    out_dir: Path,
):
    """Write plain-text comparison files for each matched pair."""
    matched_dir  = out_dir / "matched"
    unmatched_a  = out_dir / "unmatched_a"
    unmatched_b  = out_dir / "unmatched_b"
    for d in (matched_dir, unmatched_a, unmatched_b):
        d.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "PDF COMPARISON REPORT",
        "=" * 70,
        f"  File A : {name_a}",
        f"  File B : {name_b}",
        f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Matched: {len(matched)} sections",
        f"  Only A : {len(only_a)} sections",
        f"  Only B : {len(only_b)} sections",
        "=" * 70,
        "",
    ]

    # ── Matched sections ──────────────────────────────────────────
    report_lines.append("MATCHED SECTIONS")
    report_lines.append("-" * 70)

    for i, item in enumerate(matched, 1):
        sa, sb, score, summary = item if len(item) == 4 else (*item, "")
        stats = text_diff_stats(sa.content(), sb.content())
        sim   = stats["line_similarity_pct"]
        label = "IDENTICAL" if sim == 100 else f"{sim}% similar"

        report_lines += [
            f"\n[{i}] {sa.level.upper()} — {sa.title}",
            f"     Title match score : {score:.1f}%",
            f"     Content similarity: {label}",
            f"     Lines A / B       : {stats['total_lines_a']} / {stats['total_lines_b']}",
            f"     Words A / B       : {stats['total_words_a']} / {stats['total_words_b']}",
            f"     Lines added       : +{stats['lines_added']}",
            f"     Lines removed     : -{stats['lines_removed']}",
        ]

        if summary:
            wrapped = "\n".join(
                "     " + ln for ln in textwrap.wrap(summary, width=65)
            )
            report_lines += ["", "     ── AI SUMMARY ──", wrapped, ""]

        # Write per-pair diff file
        safe   = re.sub(r"[^\w\-]", "_", sa.title)[:50]
        fname  = f"{i:03d}_{sa.level[:3].upper()}_{safe}.txt"
        diff   = unified_diff(sa.content(), sb.content(), f"A: {sa.title}", f"B: {sb.title}")

        with open(matched_dir / fname, "w", encoding="utf-8") as f:
            f.write(f"SECTION  : {sa.title}\n")
            f.write(f"LEVEL    : {sa.level}\n")
            f.write(f"MATCH    : {score:.1f}% title similarity\n")
            f.write(f"CONTENT  : {sim}% line similarity\n")
            f.write("=" * 70 + "\n\n")
            if summary:
                f.write("AI SUMMARY\n")
                f.write("─" * 70 + "\n")
                f.write(textwrap.fill(summary, width=70))
                f.write("\n\n")
            f.write(f"{'─── FILE A: ' + name_a:─<70}\n")
            f.write(sa.content() or "(empty)")
            f.write(f"\n\n{'─── FILE B: ' + name_b:─<70}\n")
            f.write(sb.content() or "(empty)")
            f.write("\n\n" + "─" * 70 + "\n")
            f.write("UNIFIED DIFF (A → B)\n")
            f.write("─" * 70 + "\n")
            f.write(diff or "(no differences)")
            f.write("\n")

        print(f"  ✓  Matched [{sim}%] {sa.title}  →  matched/{fname}")

    # ── Unmatched A ───────────────────────────────────────────────
    report_lines += ["", "", "ONLY IN FILE A (no match in B)", "-" * 70]
    for sec in only_a:
        report_lines.append(f"  [{sec.level.upper()}] {sec.title}")
        safe  = re.sub(r"[^\w\-]", "_", sec.title)[:55]
        fname = f"A_{safe}.txt"
        (unmatched_a / fname).write_text(
            f"LEVEL : {sec.level.upper()}\nTITLE : {sec.title}\n"
            f"{'='*60}\n\n{sec.content()}\n",
            encoding="utf-8"
        )

    # ── Unmatched B ───────────────────────────────────────────────
    report_lines += ["", "", "ONLY IN FILE B (no match in A)", "-" * 70]
    for sec in only_b:
        report_lines.append(f"  [{sec.level.upper()}] {sec.title}")
        safe  = re.sub(r"[^\w\-]", "_", sec.title)[:55]
        fname = f"B_{safe}.txt"
        (unmatched_b / fname).write_text(
            f"LEVEL : {sec.level.upper()}\nTITLE : {sec.title}\n"
            f"{'='*60}\n\n{sec.content()}\n",
            encoding="utf-8"
        )

    report_path = out_dir / "_COMPARISON_REPORT.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n  ✓  _COMPARISON_REPORT.txt written")
    return report_path


# ══════════════════════════════════════════════════════════════════
# HTML REPORT
# ══════════════════════════════════════════════════════════════════

def _sim_colour(pct: float) -> str:
    """Red → yellow → green based on similarity percentage."""
    if pct >= 90:   return "#22c55e"    # green
    if pct >= 70:   return "#eab308"    # yellow
    if pct >= 40:   return "#f97316"    # orange
    return "#ef4444"                    # red


def write_html_report(
    matched, only_a, only_b,
    name_a, name_b,
    out_dir: Path,
):
    """Write a self-contained HTML comparison report with side-by-side diffs."""

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace('"', "&quot;"))

    def diff_html(text_a: str, text_b: str) -> str:
        """Render a side-by-side diff as an HTML table body."""
        rows = side_by_side_diff(text_a, text_b, width=80)
        colours = {
            "equal":   ("", ""),
            "replace": ("#fef3c7", "#dbeafe"),
            "delete":  ("#fee2e2", "#f0fdf4"),
            "insert":  ("#f0fdf4", "#dbeafe"),
        }
        lines = []
        for left, right, tag in rows:
            cl, cr = colours[tag]
            style_l = f' style="background:{cl}"' if cl else ""
            style_r = f' style="background:{cr}"' if cr else ""
            marker  = {"equal": " ", "replace": "~", "delete": "-", "insert": "+"}[tag]
            lines.append(
                f'<tr><td class="ln">{marker}</td>'
                f'<td{style_l}><pre>{esc(left)}</pre></td>'
                f'<td{style_r}><pre>{esc(right)}</pre></td></tr>'
            )
        return "\n".join(lines)

    # Build matched-section cards
    cards_html = []
    for i, item in enumerate(matched, 1):
        sa, sb, score, summary = item if len(item) == 4 else (*item, "")
        stats   = text_diff_stats(sa.content(), sb.content())
        sim     = stats["line_similarity_pct"]
        colour  = _sim_colour(sim)
        diff_tb = diff_html(sa.content(), sb.content())
        label   = "IDENTICAL" if sim == 100.0 else f"{sim}% similar"
        anchor  = f"sec{i}"

        # AI summary block
        if summary:
            summary_html = f"""
          <div class="ai-summary">
            <div class="ai-summary-label">
              <span class="ai-icon">✦</span> AI Summary of Differences
            </div>
            <p>{esc(summary)}</p>
          </div>"""
        else:
            summary_html = ""

        cards_html.append(f"""
        <div class="card" id="{anchor}">
          <div class="card-header">
            <div class="card-meta">
              <span class="badge badge-{sa.level}">{sa.level.upper()}</span>
              <h3>{esc(sa.title)}</h3>
            </div>
            <div class="card-scores">
              <span class="score" style="color:{colour}">{label}</span>
              <span class="score-small">title match {score:.0f}%</span>
            </div>
          </div>
          <div class="stats-row">
            <span>Lines: <b>{stats['total_lines_a']}</b> → <b>{stats['total_lines_b']}</b></span>
            <span>Words: <b>{stats['total_words_a']}</b> → <b>{stats['total_words_b']}</b></span>
            <span class="added">+{stats['lines_added']} added</span>
            <span class="removed">-{stats['lines_removed']} removed</span>
            <span class="unchanged">{stats['lines_unchanged']} unchanged</span>
          </div>
          {summary_html}
          <details {'open' if sim < 100 else ''}>
            <summary>Show diff</summary>
            <div class="diff-wrap">
              <table class="diff-table">
                <thead>
                  <tr>
                    <th class="ln"></th>
                    <th>A &mdash; {esc(name_a)}</th>
                    <th>B &mdash; {esc(name_b)}</th>
                  </tr>
                </thead>
                <tbody>
                  {diff_tb}
                </tbody>
              </table>
            </div>
          </details>
        </div>""")

    # Unmatched lists
    def unmatched_list(sections, label):
        if not sections:
            return f'<p class="empty">All {label} sections were matched.</p>'
        items = "".join(
            f'<li><span class="badge badge-{s.level}">{s.level.upper()}</span> {esc(s.title)}</li>'
            for s in sections
        )
        return f"<ul class='unmatched-list'>{items}</ul>"

    # TOC
    toc_items = "".join(
        f'<li><a href="#sec{i}">{esc(sa.title)}</a>'
        f' <span style="color:{_sim_colour(text_diff_stats(sa.content(), sb.content())["line_similarity_pct"])}">'
        f'({text_diff_stats(sa.content(), sb.content())["line_similarity_pct"]}%)</span></li>'
        for i, (sa, sb, _) in enumerate(matched, 1)
    )

    overall_sim = (
        round(sum(text_diff_stats(sa.content(), sb.content())["line_similarity_pct"]
                  for sa, sb, _ in matched) / len(matched), 1)
        if matched else 0
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PDF Comparison Report</title>
<style>
  :root {{
    --bg:       #0f172a;
    --surface:  #1e293b;
    --border:   #334155;
    --text:     #e2e8f0;
    --muted:    #94a3b8;
    --accent:   #38bdf8;
    --green:    #22c55e;
    --yellow:   #eab308;
    --red:      #ef4444;
    --orange:   #f97316;
    --add-bg:   #052e16;
    --del-bg:   #450a0a;
    --chg-bg:   #1e1b4b;
    --eq-bg:    transparent;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Courier New', monospace;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  /* ── Layout ── */
  .layout {{ display: flex; min-height: 100vh; }}
  .sidebar {{
    width: 280px; min-width: 220px; max-width: 320px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 24px 16px;
    position: sticky; top: 0; height: 100vh;
    overflow-y: auto; flex-shrink: 0;
  }}
  .main {{ flex: 1; padding: 32px; max-width: 1400px; overflow: hidden; }}

  /* ── Sidebar ── */
  .sidebar h2 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 2px;
                  color: var(--muted); margin-bottom: 16px; }}
  .sidebar ul {{ list-style: none; }}
  .sidebar li {{ margin-bottom: 6px; font-size: 12px; }}
  .sidebar li a {{ color: var(--text); }}
  .sidebar li a:hover {{ color: var(--accent); }}

  /* ── Header ── */
  .page-header {{
    margin-bottom: 40px;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border);
  }}
  .page-header h1 {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
  .meta-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px; margin-top: 20px;
  }}
  .meta-box {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 18px;
  }}
  .meta-box .label {{ font-size: 10px; text-transform: uppercase;
                       letter-spacing: 1.5px; color: var(--muted); }}
  .meta-box .value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
  .meta-box .sub   {{ font-size: 11px; color: var(--muted); margin-top: 2px;
                       word-break: break-all; }}

  /* ── Section cards ── */
  .section-title {{ font-size: 18px; font-weight: 700; margin-bottom: 24px;
                     color: var(--muted); text-transform: uppercase;
                     letter-spacing: 1px; }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 24px;
    overflow: hidden;
  }}
  .card-header {{
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 18px 20px; border-bottom: 1px solid var(--border);
    gap: 12px;
  }}
  .card-meta {{ display: flex; align-items: center; gap: 10px; flex: 1; }}
  .card-meta h3 {{ font-size: 15px; font-weight: 600; }}
  .card-scores {{ text-align: right; flex-shrink: 0; }}
  .score {{ font-size: 15px; font-weight: 700; display: block; }}
  .score-small {{ font-size: 11px; color: var(--muted); }}

  .stats-row {{
    display: flex; gap: 16px; padding: 10px 20px;
    font-size: 12px; color: var(--muted);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }}
  .added   {{ color: var(--green); }}
  .removed {{ color: var(--red); }}
  .unchanged {{ color: var(--muted); }}

  /* ── Badge ── */
  .badge {{
    font-size: 10px; font-weight: 700; padding: 2px 7px;
    border-radius: 4px; text-transform: uppercase; letter-spacing: 1px;
    flex-shrink: 0;
  }}
  .badge-chapter    {{ background: #1d4ed8; color: #bfdbfe; }}
  .badge-section    {{ background: #065f46; color: #a7f3d0; }}
  .badge-subsection {{ background: #6b21a8; color: #e9d5ff; }}
  .badge-preamble   {{ background: #374151; color: #d1d5db; }}

  /* ── Details / diff ── */
  details summary {{
    cursor: pointer; padding: 10px 20px; font-size: 12px;
    color: var(--accent); user-select: none;
  }}
  details summary:hover {{ background: rgba(56, 189, 248, 0.05); }}
  .diff-wrap {{ overflow-x: auto; }}
  .diff-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .diff-table th {{
    padding: 8px 12px; text-align: left; font-size: 11px;
    text-transform: uppercase; letter-spacing: 1px;
    background: var(--bg); color: var(--muted);
    border-bottom: 1px solid var(--border); position: sticky; top: 0;
  }}
  .diff-table td {{ padding: 1px 12px; vertical-align: top; white-space: pre; }}
  .diff-table td.ln {{
    width: 20px; text-align: center; color: var(--muted);
    font-size: 10px; user-select: none; border-right: 1px solid var(--border);
  }}
  .diff-table pre {{ font-family: inherit; font-size: 12px; white-space: pre-wrap; word-break: break-all; }}

  /* ── AI Summary ── */
  .ai-summary {{
    margin: 0;
    padding: 14px 20px;
    background: linear-gradient(135deg, #0c1a2e 0%, #0f2240 100%);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
  }}
  .ai-summary-label {{
    font-size: 10px; text-transform: uppercase; letter-spacing: 2px;
    color: var(--accent); margin-bottom: 8px; display: flex;
    align-items: center; gap: 6px;
  }}
  .ai-icon {{ font-size: 13px; }}
  .ai-summary p {{
    font-size: 13px; line-height: 1.7; color: #cbd5e1;
    font-family: Georgia, serif;
  }}

  /* ── Unmatched ── */
  .unmatched-section {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px; margin-bottom: 20px;
  }}
  .unmatched-section h3 {{ font-size: 14px; color: var(--muted); margin-bottom: 12px; }}
  .unmatched-list {{ list-style: none; }}
  .unmatched-list li {{ padding: 6px 0; font-size: 13px;
                         border-bottom: 1px solid var(--border); display: flex;
                         align-items: center; gap: 8px; }}
  .empty {{ color: var(--muted); font-size: 13px; font-style: italic; }}
</style>
</head>
<body>
<div class="layout">

  <!-- Sidebar TOC -->
  <nav class="sidebar">
    <h2>Matched Sections</h2>
    <ul>{toc_items}</ul>
    <br>
    <h2>Jump to</h2>
    <ul>
      <li><a href="#unmatched">Unmatched Sections</a></li>
    </ul>
  </nav>

  <!-- Main content -->
  <main class="main">

    <div class="page-header">
      <h1>PDF Comparison Report</h1>
      <div class="meta-grid">
        <div class="meta-box">
          <div class="label">File A</div>
          <div class="value" style="font-size:14px">{esc(name_a)}</div>
        </div>
        <div class="meta-box">
          <div class="label">File B</div>
          <div class="value" style="font-size:14px">{esc(name_b)}</div>
        </div>
        <div class="meta-box">
          <div class="label">Overall Similarity</div>
          <div class="value" style="color:{_sim_colour(overall_sim)}">{overall_sim}%</div>
          <div class="sub">avg across {len(matched)} matched sections</div>
        </div>
        <div class="meta-box">
          <div class="label">Sections Matched</div>
          <div class="value">{len(matched)}</div>
          <div class="sub">+{len(only_a)} only in A &nbsp;·&nbsp; +{len(only_b)} only in B</div>
        </div>
        <div class="meta-box">
          <div class="label">Generated</div>
          <div class="value" style="font-size:14px">{datetime.now().strftime('%Y-%m-%d')}</div>
          <div class="sub">{datetime.now().strftime('%H:%M:%S')}</div>
        </div>
      </div>
    </div>

    <!-- Matched sections -->
    <div class="section-title">Matched Sections ({len(matched)})</div>
    {"".join(cards_html) or '<p class="empty">No sections were matched.</p>'}

    <!-- Unmatched -->
    <div id="unmatched" class="section-title" style="margin-top:48px">
      Unmatched Sections
    </div>
    <div class="unmatched-section">
      <h3>Only in A — {esc(name_a)} ({len(only_a)})</h3>
      {unmatched_list(only_a, "A")}
    </div>
    <div class="unmatched-section">
      <h3>Only in B — {esc(name_b)} ({len(only_b)})</h3>
      {unmatched_list(only_b, "B")}
    </div>

  </main>
</div>
</body>
</html>"""

    report_path = out_dir / "_COMPARISON_REPORT.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\n  ✓  _COMPARISON_REPORT.html written")
    return report_path


# ══════════════════════════════════════════════════════════════════
# JSON METADATA
# ══════════════════════════════════════════════════════════════════

def write_json_metadata(matched, only_a, only_b, name_a, name_b, out_dir: Path):
    data = {
        "file_a": name_a,
        "file_b": name_b,
        "generated": datetime.now().isoformat(),
        "summary": {
            "matched":  len(matched),
            "only_a":   len(only_a),
            "only_b":   len(only_b),
        },
        "matched": [
            {
                "level":       sa.level,
                "title_a":     sa.title,
                "title_b":     sb.title,
                "title_score": round(score, 1),
                "ai_summary":  summary,
                **text_diff_stats(sa.content(), sb.content()),
            }
            for sa, sb, score, summary in (
                item if len(item) == 4 else (*item, "")
                for item in matched
            )
        ],
        "only_in_a": [{"level": s.level, "title": s.title} for s in only_a],
        "only_in_b": [{"level": s.level, "title": s.title} for s in only_b],
    }
    path = out_dir / "_MATCHED.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  ✓  _MATCHED.json written")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two extracted PDF section sets (file-driven, modular)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  # Step 1 — extract each PDF first (run separately, once per PDF)
  python pdf_section_extractor.py doc_a.pdf extracted_a/
  python pdf_section_extractor.py doc_b.pdf extracted_b/

  # Step 2 — compare the saved extractions
  python pdf_compare.py extracted_a/ extracted_b/

  # Re-run comparison with different settings — no re-extraction needed:
  python pdf_compare.py extracted_a/ extracted_b/ --fuzzy-threshold 60
  python pdf_compare.py extracted_a/ extracted_b/ --skip-summary
  python pdf_compare.py extracted_a/ extracted_b/ --report-format text
  python pdf_compare.py extracted_a/ extracted_b/ --text-model mistral

Popular Ollama text models for summaries:
  llama3.2   Fast and capable         ollama pull llama3.2
  llama3     Meta Llama 3             ollama pull llama3
  mistral    Mistral 7B               ollama pull mistral
  phi3       Microsoft Phi-3 (small)  ollama pull phi3
        """
    )
    parser.add_argument(
        "dir_a",
        help="Directory containing _sections.json for the first PDF "
             "(output of pdf_section_extractor.py)",
    )
    parser.add_argument(
        "dir_b",
        help="Directory containing _sections.json for the second PDF",
    )
    parser.add_argument(
        "output_dir", nargs="?", default="pdf_comparison",
        help="Where to write comparison outputs (default: pdf_comparison/)",
    )
    parser.add_argument(
        "--text-model", default=DEFAULT_TEXT_MODEL,
        help=f"Ollama text model for AI summaries (default: {DEFAULT_TEXT_MODEL})",
    )
    parser.add_argument(
        "--ollama-url", default=DEFAULT_OLLAMA_URL,
        help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--skip-summary", action="store_true",
        help="Skip AI difference summaries (diff stats only, much faster)",
    )
    parser.add_argument(
        "--fuzzy-threshold", type=float, default=75.0,
        help="Min title similarity %% to count as a match (default: 75)",
    )
    parser.add_argument(
        "--report-format", choices=["text", "html"], default="html",
        help="Output report format (default: html)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate inputs ───────────────────────────────────────────
    for label, d in [("A", args.dir_a), ("B", args.dir_b)]:
        p = Path(d)
        if not p.exists():
            print(f"✗  Directory not found: {d}")
            sys.exit(1)
        json_path = p / "_sections.json"
        if not json_path.exists():
            print(f"✗  No _sections.json in '{d}'")
            print(f"   Run pdf_section_extractor.py on your PDF first:")
            print(f"   python pdf_section_extractor.py your_file.pdf {d}/")
            sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  PDF Comparator  (file-driven)")
    print(f"{'═'*60}")
    print(f"  Source A    : {args.dir_a}/_sections.json")
    print(f"  Source B    : {args.dir_b}/_sections.json")
    print(f"  Output      : {args.output_dir}/")
    print(f"  Threshold   : {args.fuzzy_threshold}% title similarity")
    print(f"  Format      : {args.report_format}")
    print(f"  Text model  : {args.text_model}  (AI summaries)")
    if args.skip_summary:
        print(f"  AI summaries: disabled (--skip-summary)")
    print(f"{'═'*60}\n")

    # ── Step 1/3: Load saved extractions ─────────────────────────
    print("Step 1/3  Loading saved extractions...\n")

    try:
        sections_a, _, meta_a = load_sections_json(args.dir_a)
        name_a = meta_a.get("source_pdf") or Path(args.dir_a).name
        print(f"  ✓  A: {len(sections_a)} sections  ←  {name_a}")
        if meta_a.get("extracted_at"):
            print(f"     Extracted : {meta_a['extracted_at']}")
        if meta_a.get("model"):
            print(f"     Model used: {meta_a['model']}")
    except FileNotFoundError as e:
        print(f"✗  {e}")
        sys.exit(1)

    print()

    try:
        sections_b, _, meta_b = load_sections_json(args.dir_b)
        name_b = meta_b.get("source_pdf") or Path(args.dir_b).name
        print(f"  ✓  B: {len(sections_b)} sections  ←  {name_b}")
        if meta_b.get("extracted_at"):
            print(f"     Extracted : {meta_b['extracted_at']}")
        if meta_b.get("model"):
            print(f"     Model used: {meta_b['model']}")
    except FileNotFoundError as e:
        print(f"✗  {e}")
        sys.exit(1)

    if not sections_a or not sections_b:
        print("\n✗  One or both extractions are empty. Re-run pdf_section_extractor.py.")
        sys.exit(1)

    # ── Step 2/3: Match + summarise ───────────────────────────────
    print(f"\nStep 2/3  Matching sections (threshold: {args.fuzzy_threshold}%)...\n")
    matched_raw, only_a, only_b = match_sections(
        sections_a, sections_b, threshold=args.fuzzy_threshold
    )
    print(f"  Matched  : {len(matched_raw)}")
    print(f"  Only A   : {len(only_a)}")
    print(f"  Only B   : {len(only_b)}")

    print(f"\n  Generating AI summaries with '{args.text_model}'...\n")
    matched = summarise_all_matches(
        matched_raw,
        name_a       = name_a,
        name_b       = name_b,
        base_url     = args.ollama_url,
        model        = args.text_model,
        skip_summary = args.skip_summary,
    )

    # ── Step 3/3: Write reports ───────────────────────────────────
    print(f"\nStep 3/3  Writing reports to {args.output_dir}/\n")
    if args.report_format == "html":
        write_html_report(matched, only_a, only_b, name_a, name_b, out_dir)
    else:
        write_text_report(matched, only_a, only_b, name_a, name_b, out_dir)

    write_json_metadata(matched, only_a, only_b, name_a, name_b, out_dir)

    # ── Summary ───────────────────────────────────────────────────
    if matched:
        sims = [
            text_diff_stats(sa.content(), sb.content())["line_similarity_pct"]
            for sa, sb, *_ in matched
        ]
        avg       = round(sum(sims) / len(sims), 1)
        identical = sum(1 for s in sims if s == 100.0)
        print(f"\n{'═'*60}")
        print(f"  ✅  Comparison complete!")
        print(f"      Matched sections  : {len(matched)}")
        print(f"      Identical         : {identical}")
        print(f"      Average similarity: {avg}%")
        print(f"      Only in A         : {len(only_a)}")
        print(f"      Only in B         : {len(only_b)}")
        print(f"      Output            : {args.output_dir}/")
        print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
