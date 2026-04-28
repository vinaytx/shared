"""
PDF Section Extractor — with Ollama Vision OCR
===============================================
Breaks a PDF into chapters, sections, and subsections, extracts text from
embedded images using a local Ollama vision model, and saves each section
to its own text file.

Features:
  - Font-size based heading detection (most reliable)
  - Regex fallback heading detection
  - Multi-line heading support (headings split across lines are merged)
  - Ollama vision OCR for images embedded in the PDF
  - One output .txt file per section + _INDEX.txt + _metadata.json

Usage:
    python pdf_section_extractor.py input.pdf [output_dir] [options]

Options:
    --model MODEL           Ollama vision model (default: llava)
    --ollama-url URL        Ollama server URL (default: http://localhost:11434)
    --skip-images           Skip image OCR, text extraction only
    --min-image-size N      Ignore images smaller than N px on either side (default: 50)
    --no-font-size          Use regex heading detection instead of font-size detection

Requirements:
    pip install pdfplumber pypdf pymupdf requests

Ollama setup:
    1. Install Ollama: https://ollama.com
    2. Start server:   ollama serve
    3. Pull a model:   ollama pull llava
"""

import re
import os
import sys
import json
import base64
import time
import argparse
import pdfplumber
import fitz                 # PyMuPDF — image extraction
import requests
from pathlib import Path
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL      = "llava"        # Also works: llama3.2-vision, moondream, bakllava
MIN_IMAGE_SIZE     = 50             # px — skip decorative icons / rule lines
MAX_HEADING_LEN    = 120            # chars — longest plausible merged heading

IMAGE_OCR_PROMPT = (
    "Extract ALL text visible in this image exactly as it appears. "
    "Include captions, labels, table contents, annotations, and any other text. "
    "If there is no text in the image, reply with exactly: [NO TEXT]. "
    "Return only the extracted text with no explanation or commentary."
)


# ══════════════════════════════════════════════════════════════════
# HEADING PATTERNS
# ══════════════════════════════════════════════════════════════════
#
# COMPLETE patterns — the entire heading is on one line.
# PARTIAL patterns  — only the number/keyword is on this line;
#                     the title text is on the next line.

CHAPTER_PATTERNS = [
    r"^(chapter\s+\d+[\.\:]?\s*.+)$",      # "Chapter 1: Foo"
    r"^(\d+\s{2,}.{3,})$",                  # "1  Long Title"
    r"^(CHAPTER\s+[A-Z0-9]+.*)$",           # "CHAPTER ONE ..."
]
SECTION_PATTERNS = [
    r"^(\d+\.\d+\s+.{3,})$",               # "1.1 Title"
    r"^(section\s+\d+[\.\d]*\s*.+)$",      # "Section 1.2 Title"
]
SUBSECTION_PATTERNS = [
    r"^(\d+\.\d+\.\d+\s+.{2,})$",          # "1.1.1 Title"
    r"^([A-Z]\.\d+\.\d+\s+.{2,})$",        # "A.1.2 Title"
]

PARTIAL_CHAPTER_PATTERNS = [
    r"^chapter\s+\d+[\.\:]?\s*$",           # "Chapter 3" / "Chapter 3:"
    r"^\d+[\.\:]?\s*$",                      # "1" / "1."
    r"^CHAPTER\s+[A-Z0-9]+[\.\:]?\s*$",     # "CHAPTER ONE"
]
PARTIAL_SECTION_PATTERNS = [
    r"^\d+\.\d+[\.\:]?\s*$",                # "1.1" / "1.1:"
    r"^section\s+\d+[\.\d]*[\.\:]?\s*$",   # "Section 1.2"
]
PARTIAL_SUBSECTION_PATTERNS = [
    r"^\d+\.\d+\.\d+[\.\:]?\s*$",           # "1.1.1"
    r"^[A-Z]\.\d+\.\d+[\.\:]?\s*$",         # "A.1.2"
]

# Body-text tell: line starts with a common article / preposition
_BODY_START_RE   = re.compile(
    r"^(the|a|an|in|on|at|to|for|of|is|are|was|were|it|this|that|these|those)\s",
    re.IGNORECASE
)
_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")
_CONTINUATION_END_RE = re.compile(
    r"\b(and|or|of|the|a|an|to|in|on|for|with|at|by|from|into|about|"
    r"between|through|within|without|during|including|versus|vs|"
    r"their|its|our|your|this|that)\s*$",
    re.IGNORECASE
)


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class ImageOCRResult:
    page_num:       int
    image_index:    int
    width:          int
    height:         int
    extracted_text: str
    model_used:     str
    skipped:        bool = False
    error:          str  = ""


@dataclass
class Section:
    level:             str           # "chapter" | "section" | "subsection" | "preamble"
    title:             str
    content_lines:     list = field(default_factory=list)
    image_ocr_results: list = field(default_factory=list)

    def content(self) -> str:
        """Return full section text, including OCR results appended at the end."""
        lines = list(self.content_lines)
        for r in self.image_ocr_results:
            if r.skipped:
                lines.append(
                    f"\n[IMAGE p{r.page_num}#{r.image_index} — skipped "
                    f"(too small: {r.width}x{r.height}px)]"
                )
            elif r.error:
                lines.append(
                    f"\n[IMAGE p{r.page_num}#{r.image_index} — OCR error: {r.error}]"
                )
            elif r.extracted_text and r.extracted_text.strip() != "[NO TEXT]":
                lines.append(
                    f"\n[IMAGE TEXT — page {r.page_num}, image {r.image_index} "
                    f"({r.width}x{r.height}px)]\n"
                    f"{r.extracted_text.strip()}\n"
                    f"[/IMAGE TEXT]"
                )
        return "\n".join(lines).strip()


# ══════════════════════════════════════════════════════════════════
# HEADING DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_heading(line: str):
    """Return (level, line) if line is a complete heading, else None."""
    line = line.strip()
    if not line or len(line) < 3:
        return None
    for pat in SUBSECTION_PATTERNS:
        if re.match(pat, line, re.IGNORECASE):
            return ("subsection", line)
    for pat in SECTION_PATTERNS:
        if re.match(pat, line, re.IGNORECASE):
            return ("section", line)
    for pat in CHAPTER_PATTERNS:
        if re.match(pat, line, re.IGNORECASE):
            return ("chapter", line)
    return None


def detect_partial_heading(line: str):
    """Return (level, line) if the line is just the number/keyword of a heading."""
    line = line.strip()
    if not line:
        return None
    for pat in PARTIAL_SUBSECTION_PATTERNS:
        if re.match(pat, line, re.IGNORECASE):
            return ("subsection", line)
    for pat in PARTIAL_SECTION_PATTERNS:
        if re.match(pat, line, re.IGNORECASE):
            return ("section", line)
    for pat in PARTIAL_CHAPTER_PATTERNS:
        if re.match(pat, line, re.IGNORECASE):
            return ("chapter", line)
    return None


def _looks_incomplete(text: str) -> bool:
    """
    Return True if a heading line seems cut short and may continue on the next line.
    Short lines with no sentence-ending punctuation, or lines ending mid-phrase.
    """
    text = text.strip()
    if len(text) > 80:
        return False                        # Already long — treat as complete
    if _SENTENCE_END_RE.search(text):
        return False                        # Ends a sentence
    if len(text) <= 60:
        return True                         # Short — likely continues
    if _CONTINUATION_END_RE.search(text):
        return True                         # Ends mid-phrase
    return False


def _is_continuation_candidate(line: str) -> bool:
    """
    Return True if this line could be the tail of a multi-line heading.
    It must NOT look like body text, a new heading, or a sentence opener.
    """
    line = line.strip()
    if not line or len(line) >= MAX_HEADING_LEN:
        return False
    if detect_heading(line) or detect_partial_heading(line):
        return False                        # It's its own heading
    if _BODY_START_RE.match(line):
        return False                        # Starts like body text
    if re.match(r"^\d+\.", line):
        return False                        # Numbered list item
    if _SENTENCE_END_RE.search(line):
        return False                        # Ends a sentence
    return True


# ══════════════════════════════════════════════════════════════════
# MULTI-LINE HEADING MERGING
# ══════════════════════════════════════════════════════════════════

def merge_multiline_headings_regex(raw_lines: list) -> list:
    """
    Pre-process text lines from a page to join headings that span multiple lines.

    Three cases handled:
      1. Complete heading split across lines:
             "Chapter 3: A Very Long Title That"
             "Continues on the Next Line"
         → "Chapter 3: A Very Long Title That Continues on the Next Line"

      2. Number / keyword alone, title on next line:
             "1.2"
             "Background and Motivation"
         → "1.2 Background and Motivation"

      3. Chapter keyword + colon, title on next line:
             "Chapter 3:"
             "Introduction to Databases"
         → "Chapter 3: Introduction to Databases"
    """
    result = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        if not line:
            i += 1
            continue

        full    = detect_heading(line)
        partial = detect_partial_heading(line) if not full else None

        if full:
            # Complete heading — check if it looks cut short
            merged = line
            j = i + 1
            while j < len(raw_lines) and _looks_incomplete(merged):
                nxt = raw_lines[j].strip()
                if not nxt:
                    break
                if _is_continuation_candidate(nxt):
                    merged = merged + " " + nxt
                    j += 1
                else:
                    break
            result.append(merged)
            i = j

        elif partial:
            # Number / keyword alone — pull in the very next non-blank line as the title
            merged = line
            j = i + 1
            while j < len(raw_lines):           # skip blanks
                nxt = raw_lines[j].strip()
                j += 1
                if nxt:
                    merged = merged + " " + nxt
                    break
            # Then optionally more continuation lines
            while j < len(raw_lines) and _looks_incomplete(merged):
                nxt = raw_lines[j].strip()
                if not nxt:
                    break
                if _is_continuation_candidate(nxt):
                    merged = merged + " " + nxt
                    j += 1
                else:
                    break
            result.append(merged)
            i = j

        else:
            result.append(line)
            i += 1

    return result


def merge_multiline_headings_fontsize(lines: list, size_map: dict) -> list:
    """
    Merge consecutive font-size-tagged lines that belong to the same heading.

    A line is merged into the previous heading only when:
      - Both share the same heading-level font size
      - The accumulated heading text still looks incomplete
      - The next line does NOT itself start a new heading
      - The merged result would not exceed MAX_HEADING_LEN

    Args:
        lines    : list of (text, font_size) tuples
        size_map : {font_size: level} mapping from build_size_to_level()

    Returns a new list of (text, font_size) tuples.
    """
    if not lines:
        return lines

    merged = []
    i = 0
    while i < len(lines):
        text, sz = lines[i]
        text = text.strip()

        if not size_map.get(sz):
            # Body text — pass through unchanged
            merged.append((text, sz))
            i += 1
            continue

        # Heading-level font — look ahead for continuation lines
        combined = text
        j = i + 1
        while j < len(lines):
            nxt_text, nxt_sz = lines[j]
            nxt_text = nxt_text.strip()

            if not nxt_text:
                break

            same_level   = (nxt_sz == sz)
            not_new_head = (not detect_heading(nxt_text) and
                            not detect_partial_heading(nxt_text))
            fits          = len(combined + " " + nxt_text) <= MAX_HEADING_LEN

            if same_level and _looks_incomplete(combined) and not_new_head and fits:
                combined = combined + " " + nxt_text
                j += 1
            else:
                break

        merged.append((combined, sz))
        i = j

    return merged


# ══════════════════════════════════════════════════════════════════
# FONT STATISTICS
# ══════════════════════════════════════════════════════════════════

def get_font_stats(pdf_path: str) -> list:
    """Collect every font size used in the document, sorted largest first."""
    sizes = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for w in page.extract_words(extra_attrs=["size"]):
                sz = round(float(w.get("size", 0)))
                sizes[sz] = sizes.get(sz, 0) + 1
    return sorted(sizes.keys(), reverse=True)


def build_size_to_level(sizes: list) -> dict:
    """Map the 3 largest font sizes to chapter / section / subsection."""
    levels = ["chapter", "section", "subsection"]
    return {sz: levels[i] for i, sz in enumerate(sizes[:3])}


# ══════════════════════════════════════════════════════════════════
# OLLAMA INTEGRATION
# ══════════════════════════════════════════════════════════════════

def check_ollama(base_url: str, model: str) -> bool:
    """Return True if Ollama is running and the requested model is available."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        available = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        if model.split(":")[0] not in available:
            print(f"  ⚠  Model '{model}' not found in Ollama.")
            print(f"     Available: {', '.join(available) or 'none'}")
            print(f"     Run: ollama pull {model}")
            return False
        return True
    except requests.exceptions.ConnectionError:
        print(f"  ✗  Cannot connect to Ollama at {base_url}")
        print(f"     Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"  ✗  Ollama check failed: {e}")
        return False


def ocr_image_with_ollama(
    image_bytes: bytes,
    base_url: str,
    model: str,
    retries: int = 2,
) -> str:
    """Send image bytes to the Ollama vision model and return extracted text."""
    payload = {
        "model":  model,
        "prompt": IMAGE_OCR_PROMPT,
        "images": [base64.b64encode(image_bytes).decode("utf-8")],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 1024},
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            if attempt < retries:
                print(f"    ⏱  Timeout — retrying ({attempt + 1}/{retries})...")
                time.sleep(2)
            else:
                raise
        except Exception:
            if attempt < retries:
                time.sleep(1)
            else:
                raise
    return ""


# ══════════════════════════════════════════════════════════════════
# IMAGE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_page_images(doc, page_num: int, min_size: int) -> list:
    """
    Extract all raster images from the given PDF page (1-indexed).

    Returns a list of dicts:
        bytes    : raw image bytes (None if skipped/error)
        mime     : MIME type string
        width    : image width in pixels
        height   : image height in pixels
        index    : position index on the page
        skipped  : True if below min_size threshold
        error    : error message string (empty if OK)
    """
    page    = doc[page_num - 1]         # fitz is 0-indexed
    results = []

    for idx, img in enumerate(page.get_images(full=True)):
        xref  = img[0]
        entry = {
            "bytes":   None,
            "mime":    "image/png",
            "width":   0,
            "height":  0,
            "index":   idx,
            "skipped": False,
            "error":   "",
        }
        try:
            raw   = doc.extract_image(xref)
            w, h  = raw["width"], raw["height"]
            entry["width"], entry["height"] = w, h

            if w < min_size or h < min_size:
                entry["skipped"] = True
                results.append(entry)
                continue

            ext      = raw["ext"].lower()
            mime_map = {
                "jpeg": "image/jpeg", "jpg": "image/jpeg",
                "png":  "image/png",  "gif": "image/gif",
                "webp": "image/webp",
            }
            if ext in mime_map:
                entry["bytes"] = raw["image"]
                entry["mime"]  = mime_map[ext]
            else:
                # Convert BMP / TIFF / CMYK etc. to PNG via fitz
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                entry["bytes"] = pix.tobytes("png")
                entry["mime"]  = "image/png"

        except Exception as exc:
            entry["error"] = str(exc)

        results.append(entry)

    return results


# ══════════════════════════════════════════════════════════════════
# SECTION HELPERS
# ══════════════════════════════════════════════════════════════════

def _ensure_preamble(sections: list, current: dict) -> None:
    """Insert a Preamble section at index 0 if no heading has been seen yet."""
    if not sections or sections[0].level != "preamble":
        pre = Section(level="preamble", title="Preamble")
        sections.insert(0, pre)
        current["chapter"] = pre


def _register_heading(level: str, title: str, sections: list, current: dict) -> None:
    """Create a new Section and update the current-section pointers."""
    sec = Section(level=level, title=title)
    sections.append(sec)
    current[level] = sec
    if level == "chapter":
        current["section"] = current["subsection"] = None
    elif level == "section":
        current["subsection"] = None


def _active_section(current: dict, sections: list) -> Section:
    """Return the deepest active section, creating a preamble if needed."""
    target = current["subsection"] or current["section"] or current["chapter"]
    if target is None:
        _ensure_preamble(sections, current)
        target = sections[0]
    return target


# ══════════════════════════════════════════════════════════════════
# CORE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_structure(
    pdf_path:       str,
    use_font_size:  bool = True,
    ollama_url:     str  = DEFAULT_OLLAMA_URL,
    model:          str  = DEFAULT_MODEL,
    skip_images:    bool = False,
    min_image_size: int  = MIN_IMAGE_SIZE,
) -> tuple:
    """
    Parse the PDF into Section objects and optionally OCR embedded images.

    Returns:
        (sections list, ocr_stats dict)
    """
    sections  = []
    current   = {"chapter": None, "section": None, "subsection": None}
    ocr_stats = {"total": 0, "success": 0, "skipped": 0, "errors": 0, "no_text": 0}

    # ── Font-size map (built once for the whole document) ──────────
    size_map = {}
    if use_font_size:
        sizes    = get_font_stats(pdf_path)
        size_map = build_size_to_level(sizes)
        print(f"  Font size → level map: {size_map}")

    # ── Ollama pre-flight check ────────────────────────────────────
    ollama_ok = False
    if not skip_images:
        print(f"\n  Checking Ollama at {ollama_url} (model: '{model}')...")
        ollama_ok = check_ollama(ollama_url, model)
        if ollama_ok:
            print(f"  ✓  Ollama ready\n")
        else:
            print(f"  ⚠  Image OCR disabled — continuing text-only\n")

    fitz_doc = fitz.open(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, 1):
            print(f"  Page {page_num}/{total}", end="", flush=True)

            # ── TEXT EXTRACTION ────────────────────────────────────
            if use_font_size:
                # Step 1: group words into visual lines by y-position
                raw_lines, cur_y, cur_words, cur_sz = [], None, [], None
                for w in page.extract_words(extra_attrs=["size"]):
                    y  = round(float(w["top"]))
                    sz = round(float(w.get("size", 0)))
                    if cur_y is None or abs(y - cur_y) < 3:
                        cur_words.append(w["text"])
                        cur_y = y
                        if cur_sz is None:
                            cur_sz = sz
                    else:
                        raw_lines.append((" ".join(cur_words), cur_sz))
                        cur_words, cur_y, cur_sz = [w["text"]], y, sz
                if cur_words:
                    raw_lines.append((" ".join(cur_words), cur_sz))

                # Step 2: merge consecutive heading lines
                lines = merge_multiline_headings_fontsize(raw_lines, size_map)

                # Step 3: classify each line
                for text, sz in lines:
                    text = text.strip()
                    if not text:
                        continue
                    level = size_map.get(sz)
                    if level:
                        _register_heading(level, text, sections, current)
                    else:
                        _active_section(current, sections).content_lines.append(text)

            else:
                # Regex mode
                raw = [ln.strip() for ln in (page.extract_text() or "").splitlines()]
                for line in merge_multiline_headings_regex(raw):
                    if not line:
                        continue
                    result = detect_heading(line)
                    if result:
                        level, title = result
                        _register_heading(level, title, sections, current)
                    else:
                        _active_section(current, sections).content_lines.append(line)

            # ── IMAGE OCR ──────────────────────────────────────────
            page_images = extract_page_images(fitz_doc, page_num, min_image_size)
            if page_images:
                print(f" — {len(page_images)} image(s)", end="", flush=True)

            target = _active_section(current, sections)

            for entry in page_images:
                ocr_stats["total"] += 1
                idx = entry["index"]

                # Extraction error
                if entry["error"] and not entry["bytes"]:
                    ocr_stats["errors"] += 1
                    target.image_ocr_results.append(ImageOCRResult(
                        page_num=page_num, image_index=idx,
                        width=entry["width"], height=entry["height"],
                        extracted_text="", model_used=model, error=entry["error"],
                    ))
                    continue

                # Too small — skipped
                if entry["skipped"]:
                    ocr_stats["skipped"] += 1
                    target.image_ocr_results.append(ImageOCRResult(
                        page_num=page_num, image_index=idx,
                        width=entry["width"], height=entry["height"],
                        extracted_text="", model_used=model, skipped=True,
                    ))
                    continue

                # OCR disabled
                if skip_images or not ollama_ok:
                    target.image_ocr_results.append(ImageOCRResult(
                        page_num=page_num, image_index=idx,
                        width=entry["width"], height=entry["height"],
                        extracted_text="", model_used=model, error="OCR skipped",
                    ))
                    continue

                # ── Send to Ollama ─────────────────────────────────
                try:
                    print(
                        f"\n    🔍 OCR image {idx} "
                        f"({entry['width']}x{entry['height']}px)...",
                        end=" ", flush=True,
                    )
                    text = ocr_image_with_ollama(entry["bytes"], ollama_url, model)

                    if not text.strip() or text.strip() == "[NO TEXT]":
                        ocr_stats["no_text"] += 1
                        text = "[NO TEXT]"
                        print("(no text)")
                    else:
                        ocr_stats["success"] += 1
                        preview = text[:70].replace("\n", " ")
                        suffix  = "..." if len(text) > 70 else ""
                        print(f'✓ "{preview}{suffix}"')

                    target.image_ocr_results.append(ImageOCRResult(
                        page_num=page_num, image_index=idx,
                        width=entry["width"], height=entry["height"],
                        extracted_text=text, model_used=model,
                    ))

                except Exception as exc:
                    ocr_stats["errors"] += 1
                    print(f"✗ {exc}")
                    target.image_ocr_results.append(ImageOCRResult(
                        page_num=page_num, image_index=idx,
                        width=entry["width"], height=entry["height"],
                        extracted_text="", model_used=model, error=str(exc),
                    ))

            print()     # newline after each page's progress line

    fitz_doc.close()

    if not skip_images and ocr_stats["total"] > 0:
        print(f"\n  📷 Image OCR summary:")
        print(f"     Total     : {ocr_stats['total']}")
        print(f"     With text : {ocr_stats['success']}")
        print(f"     No text   : {ocr_stats['no_text']}")
        print(f"     Too small : {ocr_stats['skipped']}")
        print(f"     Errors    : {ocr_stats['errors']}")

    return sections, ocr_stats


# ══════════════════════════════════════════════════════════════════
# SERIALISATION — save / load Section objects as JSON
# ══════════════════════════════════════════════════════════════════

SECTIONS_JSON_FILE = "_sections.json"   # canonical filename inside an output dir


def save_sections_json(
    sections:   list,
    output_dir: str,
    ocr_stats:  dict,
    model:      str,
    source_pdf: str = "",
) -> Path:
    """
    Serialise every Section (including ImageOCRResult records) to
    <output_dir>/_sections.json so that pdf_compare.py can load the
    extraction results without re-running the extractor.

    Schema
    ------
    {
      "source_pdf":   "report.pdf",
      "extracted_at": "2026-04-24T12:00:00",
      "model":        "llava",
      "ocr_stats":    {...},
      "sections": [
        {
          "level":         "chapter",
          "title":         "1  Introduction",
          "content_lines": ["line 1", "line 2", ...],
          "image_ocr_results": [
            {
              "page_num":       1,
              "image_index":    0,
              "width":          800,
              "height":         600,
              "extracted_text": "text from image",
              "model_used":     "llava",
              "skipped":        false,
              "error":          ""
            }
          ]
        },
        ...
      ]
    }
    """
    import datetime as _dt
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / SECTIONS_JSON_FILE

    data = {
        "source_pdf":   source_pdf,
        "extracted_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "model":        model,
        "ocr_stats":    ocr_stats,
        "sections": [
            {
                "level":         sec.level,
                "title":         sec.title,
                "content_lines": sec.content_lines,
                "image_ocr_results": [
                    {
                        "page_num":       r.page_num,
                        "image_index":    r.image_index,
                        "width":          r.width,
                        "height":         r.height,
                        "extracted_text": r.extracted_text,
                        "model_used":     r.model_used,
                        "skipped":        r.skipped,
                        "error":          r.error,
                    }
                    for r in sec.image_ocr_results
                ],
            }
            for sec in sections
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_sections_json(source: str) -> tuple:
    """
    Load a previously saved _sections.json and reconstruct Section +
    ImageOCRResult objects.

    ``source`` can be:
      - a directory path  →  looks for <source>/_sections.json
      - a direct file path to any .json file

    Returns:
        (sections, ocr_stats, meta)
        where meta = {"source_pdf", "extracted_at", "model"}
    """
    src = Path(source)
    if src.is_dir():
        path = src / SECTIONS_JSON_FILE
    else:
        path = src

    if not path.exists():
        raise FileNotFoundError(
            f"No {SECTIONS_JSON_FILE} found at '{path}'.\n"
            f"Run pdf_section_extractor.py first to generate it."
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    sections = []
    for sd in data.get("sections", []):
        sec = Section(level=sd["level"], title=sd["title"])
        sec.content_lines = sd.get("content_lines", [])
        for rd in sd.get("image_ocr_results", []):
            sec.image_ocr_results.append(ImageOCRResult(
                page_num       = rd["page_num"],
                image_index    = rd["image_index"],
                width          = rd["width"],
                height         = rd["height"],
                extracted_text = rd["extracted_text"],
                model_used     = rd["model_used"],
                skipped        = rd.get("skipped", False),
                error          = rd.get("error", ""),
            ))
        sections.append(sec)

    ocr_stats = data.get("ocr_stats", {})
    meta      = {
        "source_pdf":   data.get("source_pdf", ""),
        "extracted_at": data.get("extracted_at", ""),
        "model":        data.get("model", ""),
    }
    return sections, ocr_stats, meta


# ══════════════════════════════════════════════════════════════════
# OUTPUT — text files + index + metadata
# ══════════════════════════════════════════════════════════════════

def safe_filename(title: str, max_len: int = 60) -> str:
    name = re.sub(r"[^\w\s\-]", "", title)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:max_len]


def write_sections(
    sections:   list,
    output_dir: str,
    ocr_stats:  dict,
    model:      str,
    source_pdf: str = "",
) -> None:
    """
    Write one .txt file per section plus _INDEX.txt, _metadata.json,
    and _sections.json (the machine-readable snapshot used by pdf_compare.py).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prefix_map = {"chapter": "CH", "section": "SEC", "subsection": "SUB", "preamble": "PRE"}
    indent_map  = {"chapter": "", "section": "  ", "subsection": "    ", "preamble": ""}
    counts      = {}
    index_lines = ["PDF SECTION INDEX", "=" * 50, ""]

    for sec in sections:
        counts[sec.level] = counts.get(sec.level, 0) + 1
        n        = counts[sec.level]
        lp       = prefix_map.get(sec.level, "SEC")
        filename = f"{lp}{n:03d}_{safe_filename(sec.title)}.txt"

        imgs_with_text = [
            r for r in sec.image_ocr_results
            if r.extracted_text and r.extracted_text.strip() not in ("[NO TEXT]", "")
        ]

        with open(out / filename, "w", encoding="utf-8") as f:
            f.write(f"LEVEL    : {sec.level.upper()}\n")
            f.write(f"TITLE    : {sec.title}\n")
            f.write(f"FILE     : {filename}\n")
            if sec.image_ocr_results:
                f.write(
                    f"IMAGES   : {len(sec.image_ocr_results)} found, "
                    f"{len(imgs_with_text)} with text\n"
                )
            f.write("=" * 60 + "\n\n")
            f.write(sec.content())
            f.write("\n")

        indent   = indent_map.get(sec.level, "")
        img_note = f" [{len(imgs_with_text)} image(s) with text]" if imgs_with_text else ""
        index_lines.append(
            f"{indent}[{sec.level.upper()}] {sec.title}  →  {filename}{img_note}"
        )
        print(
            f"  ✓  {filename}"
            f"  ({len(sec.content_lines)} lines, {len(sec.image_ocr_results)} images)"
        )

    # _INDEX.txt
    (out / "_INDEX.txt").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"\n  ✓  _INDEX.txt written")

    # _metadata.json  (human-friendly summary — no full content)
    meta = {
        "source_pdf":  source_pdf,
        "model_used":  model,
        "ocr_summary": ocr_stats,
        "sections": [
            {
                "level":      s.level,
                "title":      s.title,
                "char_count": len(s.content()),
                "line_count": len(s.content_lines),
                "images": [
                    {
                        "page":     r.page_num,
                        "index":    r.image_index,
                        "size":     f"{r.width}x{r.height}",
                        "has_text": bool(
                            r.extracted_text
                            and r.extracted_text.strip() not in ("[NO TEXT]", "")
                        ),
                        "skipped":  r.skipped,
                        "error":    r.error,
                    }
                    for r in s.image_ocr_results
                ],
            }
            for s in sections
        ],
    }
    (out / "_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"  ✓  _metadata.json written")

    # _sections.json  (full content — consumed by pdf_compare.py)
    json_path = save_sections_json(sections, output_dir, ocr_stats, model, source_pdf)
    print(f"  ✓  {SECTIONS_JSON_FILE} written  ← used by pdf_compare.py")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDF Section Extractor with Ollama Vision OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_section_extractor.py report.pdf
  python pdf_section_extractor.py report.pdf output/ --model llama3.2-vision
  python pdf_section_extractor.py report.pdf --ollama-url http://192.168.1.10:11434
  python pdf_section_extractor.py report.pdf --skip-images
  python pdf_section_extractor.py report.pdf --min-image-size 100
  python pdf_section_extractor.py report.pdf --no-font-size

Popular Ollama vision models:
  llava              General purpose (recommended)   ollama pull llava
  llama3.2-vision    Meta Llama 3.2 (very capable)   ollama pull llama3.2-vision
  moondream          Lightweight and fast             ollama pull moondream
  bakllava           Alternative llava variant        ollama pull bakllava
        """,
    )
    parser.add_argument("pdf",            help="Path to input PDF")
    parser.add_argument("output_dir",     nargs="?", default="pdf_sections",
                        help="Output directory (default: pdf_sections/)")
    parser.add_argument("--model",        default=DEFAULT_MODEL,
                        help=f"Ollama vision model (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url",   default=DEFAULT_OLLAMA_URL,
                        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("--skip-images",  action="store_true",
                        help="Skip image OCR entirely")
    parser.add_argument("--min-image-size", type=int, default=MIN_IMAGE_SIZE,
                        help=f"Minimum image side in pixels (default: {MIN_IMAGE_SIZE})")
    parser.add_argument("--no-font-size", action="store_true",
                        help="Use regex heading detection instead of font-size detection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: file not found — {args.pdf}")
        sys.exit(1)

    print(f"\n📄  PDF    : {args.pdf}")
    print(f"📁  Output : {args.output_dir}")
    print(f"🤖  Model  : {args.model}")
    print(f"🌐  Ollama : {args.ollama_url}")
    print(f"🖼   Images : {'skipped' if args.skip_images else 'OCR enabled'}\n")

    print("Step 1: Extracting structure and running image OCR...\n")
    sections, ocr_stats = extract_structure(
        pdf_path       = args.pdf,
        use_font_size  = not args.no_font_size,
        ollama_url     = args.ollama_url,
        model          = args.model,
        skip_images    = args.skip_images,
        min_image_size = args.min_image_size,
    )

    # If font-size detection found nothing, retry with regex
    if not sections:
        print("\n⚠  No sections found via font-size. Retrying with regex detection...\n")
        sections, ocr_stats = extract_structure(
            pdf_path       = args.pdf,
            use_font_size  = False,
            ollama_url     = args.ollama_url,
            model          = args.model,
            skip_images    = True,          # don't re-run OCR
            min_image_size = args.min_image_size,
        )

    if not sections:
        print("✗  No structure detected. PDF may be scanned or have unusual formatting.")
        sys.exit(1)

    print(f"\nStep 2: Writing {len(sections)} sections to {args.output_dir}/\n")
    write_sections(sections, args.output_dir, ocr_stats, args.model, source_pdf=args.pdf)

    chapters = sum(1 for s in sections if s.level == "chapter")
    secs     = sum(1 for s in sections if s.level == "section")
    subsecs  = sum(1 for s in sections if s.level == "subsection")
    total_i  = sum(len(s.image_ocr_results) for s in sections)

    print(f"\n✅  Done!")
    print(f"    {chapters} chapters | {secs} sections | {subsecs} subsections")
    if not args.skip_images:
        print(f"    {total_i} images processed, {ocr_stats.get('success', 0)} contained text")
    print(f"    Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
