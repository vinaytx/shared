"""
PDF Section Extractor
=====================
Breaks a PDF into chapters, sections, and subsections.  For each embedded
image it:
  1. Saves the image file to  <output_dir>/images/
  2. Runs OCR (Ollama vision model) to extract any text in the image
  3. Writes a machine-readable catalog  <output_dir>/_image_catalog.json
  4. Saves all section data to          <output_dir>/_sections.json
     (consumed by pdf_compare.py and pdf_image_describer.py)

Workflow (modular):
  Step 1  Extract:    python pdf_section_extractor.py report.pdf extracted/
  Step 2  Describe:   python pdf_image_describer.py   extracted/
  Step 3  Compare:    python pdf_compare.py            extracted_a/ extracted_b/

Usage:
    python pdf_section_extractor.py input.pdf [output_dir] [options]

Options:
    --model MODEL           Ollama vision model for OCR (default: llava)
    --ollama-url URL        Ollama server URL (default: http://localhost:11434)
    --skip-images           Skip image OCR (images are still saved to disk)
    --min-image-size N      Ignore images smaller than N px on either side (default: 50)
    --no-font-size          Use regex heading detection instead of font-size detection

Requirements:
    pip install pdfplumber pymupdf requests

Ollama:
    ollama serve
    ollama pull llava
"""

import re
import os
import sys
import json
import base64
import time
import argparse
import pdfplumber
import fitz
import requests
from pathlib import Path
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL      = "gemma4:31b-cloud"
MIN_IMAGE_SIZE     = 50
MAX_HEADING_LEN    = 120
IMAGES_SUBDIR      = "images"           # sub-folder inside output_dir for saved images
IMAGE_CATALOG_FILE = "_image_catalog.json"
SECTIONS_JSON_FILE = "_sections.json"
PIPELINE_STATE_FILE = "_pipeline_state.json"

# Pipeline step names — used as keys in _pipeline_state.json
STEP_TEXT   = "extract_text"
STEP_IMAGES = "save_images"
STEP_OCR    = "run_ocr"
STEP_WRITE  = "write_files"

ALL_STEPS = [STEP_TEXT, STEP_IMAGES, STEP_OCR, STEP_WRITE]

IMAGE_OCR_PROMPT = (
    "Extract ALL text visible in this image exactly as it appears. "
    "Include captions, labels, table contents, annotations, and any other text. "
    "If there is no text in the image, reply with exactly: [NO TEXT]. "
    "Return only the extracted text with no explanation or commentary."
)


# ══════════════════════════════════════════════════════════════════
# HEADING PATTERNS
# ══════════════════════════════════════════════════════════════════

CHAPTER_PATTERNS = [
    r"^(chapter\s+\d+[\.\:]?\s*.+)$",
    r"^(\d+\s{2,}.{3,})$",
    r"^(CHAPTER\s+[A-Z0-9]+.*)$",
]
SECTION_PATTERNS = [
    r"^(\d+\.\d+\s+.{3,})$",
    r"^(section\s+\d+[\.\d]*\s*.+)$",
]
SUBSECTION_PATTERNS = [
    r"^(\d+\.\d+\.\d+\s+.{2,})$",
    r"^([A-Z]\.\d+\.\d+\s+.{2,})$",
]

PARTIAL_CHAPTER_PATTERNS = [
    r"^chapter\s+\d+[\.\:]?\s*$",
    r"^\d+[\.\:]?\s*$",
    r"^CHAPTER\s+[A-Z0-9]+[\.\:]?\s*$",
]
PARTIAL_SECTION_PATTERNS = [
    r"^\d+\.\d+[\.\:]?\s*$",
    r"^section\s+\d+[\.\d]*[\.\:]?\s*$",
]
PARTIAL_SUBSECTION_PATTERNS = [
    r"^\d+\.\d+\.\d+[\.\:]?\s*$",
    r"^[A-Z]\.\d+\.\d+[\.\:]?\s*$",
]

_BODY_START_RE = re.compile(
    r"^(the|a|an|in|on|at|to|for|of|is|are|was|were|it|this|that|these|those)\s",
    re.IGNORECASE,
)
_SENTENCE_END_RE    = re.compile(r"[.!?]\s*$")
_CONTINUATION_END_RE = re.compile(
    r"\b(and|or|of|the|a|an|to|in|on|for|with|at|by|from|into|about|"
    r"between|through|within|without|during|including|versus|vs|"
    r"their|its|our|your|this|that)\s*$",
    re.IGNORECASE,
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
    # Set by extractor when the image is saved to disk
    image_file:     str  = ""   # relative path, e.g. "images/p001_i00.png"
    # Set by pdf_image_describer.py in a later step
    description:    str  = ""   # AI-generated visual description


@dataclass
class Section:
    level:             str      # "chapter" | "section" | "subsection" | "preamble"
    title:             str
    content_lines:     list = field(default_factory=list)
    image_ocr_results: list = field(default_factory=list)

    def content(self) -> str:
        """Return full section text including OCR text and AI descriptions."""
        lines = list(self.content_lines)
        for r in self.image_ocr_results:
            if r.skipped:
                lines.append(
                    f"\n[IMAGE p{r.page_num}#{r.image_index} — skipped "
                    f"(too small: {r.width}x{r.height}px)]"
                )
            elif r.error:
                lines.append(
                    f"\n[IMAGE p{r.page_num}#{r.image_index} — error: {r.error}]"
                )
            else:
                header = (
                    f"\n[IMAGE — page {r.page_num}, image {r.image_index} "
                    f"({r.width}x{r.height}px)"
                )
                if r.image_file:
                    header += f" | saved: {r.image_file}"
                header += "]"
                lines.append(header)
                if r.extracted_text and r.extracted_text.strip() != "[NO TEXT]":
                    lines.append(
                        f"[OCR TEXT]\n{r.extracted_text.strip()}\n[/OCR TEXT]"
                    )
                if r.description:
                    lines.append(
                        f"[AI DESCRIPTION]\n{r.description.strip()}\n[/AI DESCRIPTION]"
                    )
        return "\n".join(lines).strip()


# ══════════════════════════════════════════════════════════════════
# HEADING DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_heading(line: str):
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
    text = text.strip()
    if len(text) > 80:
        return False
    if _SENTENCE_END_RE.search(text):
        return False
    if len(text) <= 60:
        return True
    if _CONTINUATION_END_RE.search(text):
        return True
    return False


def _is_continuation_candidate(line: str) -> bool:
    line = line.strip()
    if not line or len(line) >= MAX_HEADING_LEN:
        return False
    if detect_heading(line) or detect_partial_heading(line):
        return False
    if _BODY_START_RE.match(line):
        return False
    if re.match(r"^\d+\.", line):
        return False
    if _SENTENCE_END_RE.search(line):
        return False
    return True


# ══════════════════════════════════════════════════════════════════
# MULTI-LINE HEADING MERGING
# ══════════════════════════════════════════════════════════════════

def merge_multiline_headings_regex(raw_lines: list) -> list:
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
            merged = line
            j = i + 1
            while j < len(raw_lines):
                nxt = raw_lines[j].strip()
                j += 1
                if nxt:
                    merged = merged + " " + nxt
                    break
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
    if not lines:
        return lines
    merged = []
    i = 0
    while i < len(lines):
        text, sz = lines[i]
        text = text.strip()
        if not size_map.get(sz):
            merged.append((text, sz))
            i += 1
            continue
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
    sizes = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for w in page.extract_words(extra_attrs=["size"]):
                sz = round(float(w.get("size", 0)))
                sizes[sz] = sizes.get(sz, 0) + 1
    return sorted(sizes.keys(), reverse=True)


def build_size_to_level(sizes: list) -> dict:
    levels = ["chapter", "section", "subsection"]
    return {sz: levels[i] for i, sz in enumerate(sizes[:3])}


# ══════════════════════════════════════════════════════════════════
# OLLAMA — OCR
# ══════════════════════════════════════════════════════════════════

def check_ollama(base_url: str, model: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        available = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        if model.split(":")[0] not in available:
            print(f"  ⚠  Model '{model}' not found.  Available: {', '.join(available) or 'none'}")
            print(f"     Run: ollama pull {model}")
            return False
        return True
    except requests.exceptions.ConnectionError:
        print(f"  ✗  Cannot connect to Ollama at {base_url}  (run: ollama serve)")
        return False
    except Exception as e:
        print(f"  ✗  Ollama check failed: {e}")
        return False


def ocr_image_with_ollama(image_bytes: bytes, base_url: str, model: str, retries: int = 2) -> str:
    payload = {
        "model":   model,
        "prompt":  IMAGE_OCR_PROMPT,
        "images":  [base64.b64encode(image_bytes).decode("utf-8")],
        "stream":  False,
        "options": {"temperature": 0.0, "num_predict": 1024},
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
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
# IMAGE EXTRACTION & SAVING
# ══════════════════════════════════════════════════════════════════

def image_filename(page_num: int, img_index: int, ext: str = "png") -> str:
    """Canonical filename for a saved image: p001_i00.png"""
    return f"p{page_num:03d}_i{img_index:02d}.{ext}"


def extract_and_save_page_images(
    doc,
    page_num:   int,
    min_size:   int,
    images_dir: Path,
) -> list:
    """
    Extract all raster images from a PDF page, save each one to images_dir,
    and return a list of entry dicts:

        bytes      : raw image bytes
        mime       : MIME type
        width      : px
        height     : px
        index      : position index on page
        skipped    : True if below min_size
        error      : error string (empty if OK)
        saved_file : relative path string, e.g. "images/p001_i00.png"
                     (empty if skipped or error)
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    page    = doc[page_num - 1]
    results = []

    for idx, img in enumerate(page.get_images(full=True)):
        xref  = img[0]
        entry = {
            "bytes": None, "mime": "image/png",
            "width": 0, "height": 0,
            "index": idx, "skipped": False,
            "error": "", "saved_file": "",
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
                img_bytes = raw["image"]
                save_ext  = ext if ext != "jpg" else "jpeg"
                # Normalise jpg → jpeg for filename clarity
                save_ext  = "jpg" if ext in ("jpeg", "jpg") else ext
                entry["mime"] = mime_map[ext]
            else:
                # Convert BMP/TIFF/CMYK → PNG
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
                save_ext  = "png"
                entry["mime"] = "image/png"

            entry["bytes"] = img_bytes

            # Save to disk
            fname      = image_filename(page_num, idx, save_ext)
            fpath      = images_dir / fname
            fpath.write_bytes(img_bytes)
            # Store as relative path (images/p001_i00.png)
            entry["saved_file"] = f"{IMAGES_SUBDIR}/{fname}"

        except Exception as exc:
            entry["error"] = str(exc)

        results.append(entry)

    return results


# ══════════════════════════════════════════════════════════════════
# IMAGE CATALOG
# ══════════════════════════════════════════════════════════════════

def save_image_catalog(catalog: list, output_dir: Path) -> Path:
    """
    Write _image_catalog.json — one entry per extracted image.

    Each entry:
    {
      "image_id":      "p001_i00",
      "image_file":    "images/p001_i00.png",
      "page_num":      1,
      "image_index":   0,
      "width":         800,
      "height":        600,
      "section_title": "1.1 Background",
      "section_level": "section",
      "ocr_text":      "Figure 1: Results",
      "description":   "",          ← filled in by pdf_image_describer.py
      "skipped":       false,
      "error":         ""
    }
    """
    import datetime as _dt
    path = Path(output_dir) / IMAGE_CATALOG_FILE
    data = {
        "generated":  _dt.datetime.now().isoformat(timespec="seconds"),
        "total":      len(catalog),
        "images":     catalog,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_image_catalog(source) -> dict:
    """
    Load _image_catalog.json from a directory or direct file path.
    Returns the parsed dict (keys: generated, total, images).
    """
    src = Path(source)
    path = src / IMAGE_CATALOG_FILE if src.is_dir() else src
    if not path.exists():
        raise FileNotFoundError(
            f"No {IMAGE_CATALOG_FILE} found at '{path}'.\n"
            f"Run pdf_section_extractor.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════
# PIPELINE CHECKPOINT — save / load / query step completion state
# ══════════════════════════════════════════════════════════════════

def save_checkpoint(
    output_dir: str,
    step:       str,
    status:     str  = "completed",
    details:    dict = None,
) -> Path:
    """
    Record that ``step`` has finished in _pipeline_state.json.

    Schema
    ──────
    {
      "source_pdf":  "report.pdf",
      "created_at":  "2026-01-01T12:00:00",
      "steps": {
        "extract_text": {
          "status":       "completed",
          "completed_at": "2026-01-01T12:00:05",
          "details":      { ... }
        },
        ...
      }
    }

    Parameters
    ──────────
    step    : one of STEP_TEXT, STEP_IMAGES, STEP_OCR, STEP_WRITE
    status  : "completed" | "skipped" | "failed"
    details : optional dict of step-specific metadata to persist
    """
    import datetime as _dt
    out  = Path(output_dir)
    path = out / PIPELINE_STATE_FILE

    # Load existing state or start fresh
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {
            "source_pdf": "",
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "steps":      {},
        }

    state["steps"][step] = {
        "status":       status,
        "completed_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "details":      details or {},
    }

    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_checkpoint(output_dir: str) -> dict:
    """
    Load the pipeline state from _pipeline_state.json.

    Returns the full state dict, or an empty dict if the file does not exist.
    Always safe to call — never raises.
    """
    path = Path(output_dir) / PIPELINE_STATE_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def step_is_done(output_dir: str, step: str) -> bool:
    """
    Return True if ``step`` has status "completed" or "skipped" in the
    pipeline state file.  Used by --resume to skip already-finished work.
    """
    state = load_checkpoint(output_dir)
    entry = state.get("steps", {}).get(step, {})
    return entry.get("status") in ("completed", "skipped")


def set_source_pdf(output_dir: str, source_pdf: str) -> None:
    """Write the source_pdf field into the pipeline state."""
    path = Path(output_dir) / PIPELINE_STATE_FILE
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        import datetime as _dt
        state = {"created_at": _dt.datetime.now().isoformat(timespec="seconds"),
                 "steps": {}}
    state["source_pdf"] = source_pdf
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def pipeline_status(output_dir: str) -> None:
    """
    Print a human-readable summary of pipeline progress for an output directory.
    Call this with --status to see what has been done before re-running.
    """
    state = load_checkpoint(output_dir)
    if not state:
        print(f"  No pipeline state found in '{output_dir}'.")
        print(f"  Run: python pdf_section_extractor.py <pdf> {output_dir}/")
        return

    print(f"\n  Pipeline state: {output_dir}/")
    print(f"  Source PDF    : {state.get('source_pdf', '(unknown)')}")
    print(f"  Created       : {state.get('created_at', '(unknown)')}")
    print()

    labels = {
        STEP_TEXT:   "1. Extract text",
        STEP_IMAGES: "2. Save images",
        STEP_OCR:    "3. Run OCR",
        STEP_WRITE:  "4. Write files",
    }
    for step in ALL_STEPS:
        entry = state.get("steps", {}).get(step, {})
        if not entry:
            status_str = "⬜  pending"
        elif entry["status"] == "completed":
            status_str = f"✅  completed  ({entry['completed_at']})"
        elif entry["status"] == "skipped":
            status_str = f"⏭   skipped   ({entry['completed_at']})"
        else:
            status_str = f"❌  {entry['status']}  ({entry['completed_at']})"
        print(f"  {labels.get(step, step):<22} {status_str}")

    # Check which downstream files exist
    out = Path(output_dir)
    print()
    file_checks = [
        (SECTIONS_JSON_FILE,   "Sections data"),
        (IMAGE_CATALOG_FILE,   "Image catalog"),
        (PIPELINE_STATE_FILE,  "Pipeline state"),
        ("_INDEX.txt",         "Section index"),
        ("_metadata.json",     "Metadata"),
    ]
    print("  Output files:")
    for fname, label in file_checks:
        exists = (out / fname).exists()
        mark   = "✓" if exists else "✗"
        print(f"    {mark}  {label:<20} {fname}")
    img_dir = out / IMAGES_SUBDIR
    if img_dir.exists():
        img_count = len(list(img_dir.glob("*")))
        print(f"    ✓  Images saved          {IMAGES_SUBDIR}/  ({img_count} files)")
    else:
        print(f"    ✗  Images saved          {IMAGES_SUBDIR}/")
    print()


# ══════════════════════════════════════════════════════════════════
# SERIALISATION — _sections.json
# ══════════════════════════════════════════════════════════════════

def save_sections_json(
    sections:   list,
    output_dir: str,
    ocr_stats:  dict,
    model:      str,
    source_pdf: str = "",
) -> Path:
    """
    Serialise every Section (including ImageOCRResult records) to
    <output_dir>/_sections.json.

    image_file and description fields are included so that
    pdf_image_describer.py can update them in place.
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
                        "image_file":     r.image_file,
                        "description":    r.description,
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
    Load _sections.json and reconstruct Section + ImageOCRResult objects.

    source: directory path  →  looks for <source>/_sections.json
            file path       →  loads that file directly

    Returns: (sections, ocr_stats, meta)
    """
    src  = Path(source)
    path = src / SECTIONS_JSON_FILE if src.is_dir() else src
    if not path.exists():
        raise FileNotFoundError(
            f"No {SECTIONS_JSON_FILE} found at '{path}'.\n"
            f"Run pdf_section_extractor.py first to generate it."
        )

    data     = json.loads(path.read_text(encoding="utf-8"))
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
                skipped        = rd.get("skipped",     False),
                error          = rd.get("error",       ""),
                image_file     = rd.get("image_file",  ""),
                description    = rd.get("description", ""),
            ))
        sections.append(sec)

    ocr_stats = data.get("ocr_stats", {})
    meta      = {
        "source_pdf":   data.get("source_pdf",   ""),
        "extracted_at": data.get("extracted_at", ""),
        "model":        data.get("model",         ""),
    }
    return sections, ocr_stats, meta


# ══════════════════════════════════════════════════════════════════
# SECTION TREE HELPERS
# ══════════════════════════════════════════════════════════════════

def _ensure_preamble(sections: list, current: dict) -> None:
    """Insert a Preamble section at index 0 if no heading has been seen yet."""
    if not sections or sections[0].level != "preamble":
        pre = Section(level="preamble", title="Preamble")
        sections.insert(0, pre)
        current["chapter"] = pre


def _register_heading(level: str, title: str, sections: list, current: dict) -> None:
    """Create a new Section, append it to the list, and update current pointers."""
    sec = Section(level=level, title=title)
    sections.append(sec)
    current[level] = sec
    if level == "chapter":
        current["section"] = current["subsection"] = None
    elif level == "section":
        current["subsection"] = None


def _active_section(current: dict, sections: list) -> Section:
    """Return the deepest active section, creating a preamble if none exists yet."""
    target = current["subsection"] or current["section"] or current["chapter"]
    if target is None:
        _ensure_preamble(sections, current)
        target = sections[0]
    return target


# ══════════════════════════════════════════════════════════════════
# STEP 1 — EXTRACT TEXT
# ══════════════════════════════════════════════════════════════════

def step_extract_text(
    pdf_path:      str,
    output_dir:    str,
    use_font_size: bool = True,
) -> list:
    """
    Parse all text from the PDF into Section objects and save to
    _sections.json (images and OCR left empty at this stage).

    Returns the sections list.
    Saves checkpoint STEP_TEXT on success.
    """
    sections = []
    current  = {"chapter": None, "section": None, "subsection": None}

    size_map = {}
    if use_font_size:
        sizes    = get_font_stats(pdf_path)
        size_map = build_size_to_level(sizes)
        print(f"  Font size → level map: {size_map}")

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, 1):
            print(f"  Page {page_num}/{total} (text)", end="", flush=True)

            if use_font_size:
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

                lines = merge_multiline_headings_fontsize(raw_lines, size_map)
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

            print()

    # Persist text-only snapshot
    ocr_stats_empty = {"total": 0, "success": 0, "skipped": 0, "errors": 0, "no_text": 0}
    save_sections_json(sections, output_dir, ocr_stats_empty, "", Path(pdf_path).name)
    save_checkpoint(output_dir, STEP_TEXT, details={
        "sections": len(sections),
        "chapters": sum(1 for s in sections if s.level == "chapter"),
    })
    print(f"\n  ✓  Text extracted: {len(sections)} sections → {SECTIONS_JSON_FILE}")
    return sections


# ══════════════════════════════════════════════════════════════════
# STEP 2 — SAVE IMAGES
# ══════════════════════════════════════════════════════════════════

def step_save_images(
    pdf_path:      str,
    output_dir:    str,
    sections:      list,
    min_image_size: int = MIN_IMAGE_SIZE,
) -> tuple:
    """
    Extract every image from the PDF, save to images/, build the catalog,
    and attach ImageOCRResult skeletons (no OCR yet) to the right sections.

    Returns (sections, image_catalog).
    Saves checkpoint STEP_IMAGES on success.
    """
    images_dir    = Path(output_dir) / IMAGES_SUBDIR
    image_catalog = []
    current       = _rebuild_current(sections)

    fitz_doc = fitz.open(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for page_num, _ in enumerate(pdf.pages, 1):
            print(f"  Page {page_num}/{total} (images)", end="", flush=True)

            page_images = extract_and_save_page_images(
                fitz_doc, page_num, min_image_size, images_dir
            )
            if page_images:
                print(f" — {len(page_images)} image(s)", end="", flush=True)

            target    = _active_section(current, sections)
            sec_title = target.title
            sec_level = target.level

            for entry in page_images:
                idx      = entry["index"]
                image_id = f"p{page_num:03d}_i{idx:02d}"

                catalog_entry = {
                    "image_id":      image_id,
                    "image_file":    entry["saved_file"],
                    "page_num":      page_num,
                    "image_index":   idx,
                    "width":         entry["width"],
                    "height":        entry["height"],
                    "section_title": sec_title,
                    "section_level": sec_level,
                    "ocr_text":      "",
                    "description":   "",
                    "skipped":       entry["skipped"],
                    "error":         entry["error"],
                }

                if entry["error"] and not entry["bytes"]:
                    target.image_ocr_results.append(ImageOCRResult(
                        page_num=page_num, image_index=idx,
                        width=entry["width"], height=entry["height"],
                        extracted_text="", model_used="",
                        error=entry["error"], image_file=entry["saved_file"],
                    ))
                elif entry["skipped"]:
                    target.image_ocr_results.append(ImageOCRResult(
                        page_num=page_num, image_index=idx,
                        width=entry["width"], height=entry["height"],
                        extracted_text="", model_used="", skipped=True,
                    ))
                else:
                    target.image_ocr_results.append(ImageOCRResult(
                        page_num=page_num, image_index=idx,
                        width=entry["width"], height=entry["height"],
                        extracted_text="", model_used="",
                        image_file=entry["saved_file"],
                    ))
                image_catalog.append(catalog_entry)

            print()

    fitz_doc.close()

    # Persist updated sections + catalog
    save_sections_json(sections, output_dir, {}, "", Path(pdf_path).name)
    save_image_catalog(image_catalog, output_dir)
    save_checkpoint(output_dir, STEP_IMAGES, details={"images_saved": len(image_catalog)})
    print(f"\n  ✓  Images saved: {len(image_catalog)} → {IMAGES_SUBDIR}/  |  {IMAGE_CATALOG_FILE}")
    return sections, image_catalog


# ══════════════════════════════════════════════════════════════════
# STEP 3 — RUN OCR
# ══════════════════════════════════════════════════════════════════

def step_run_ocr(
    output_dir: str,
    sections:   list,
    catalog:    list,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model:      str = DEFAULT_MODEL,
) -> tuple:
    """
    Run Ollama vision OCR on every saved image that doesn't yet have
    extracted text.  Updates sections and catalog in-place.

    Returns (sections, catalog, ocr_stats).
    Saves checkpoint STEP_OCR on success.
    """
    ocr_stats = {"total": 0, "success": 0, "skipped": 0, "errors": 0, "no_text": 0}

    print(f"\n  Checking Ollama at {ollama_url} (model: '{model}')...")
    if not check_ollama(ollama_url, model):
        print("  ⚠  OCR skipped — Ollama unavailable.")
        save_checkpoint(output_dir, STEP_OCR, status="skipped",
                        details={"reason": "Ollama unavailable"})
        return sections, catalog, ocr_stats

    # Build a lookup: image_file → catalog entry
    catalog_by_file = {e["image_file"]: e for e in catalog}

    # Build a lookup: (page_num, image_index) → ImageOCRResult
    ocr_lookup = {}
    for sec in sections:
        for r in sec.image_ocr_results:
            ocr_lookup[(r.page_num, r.image_index)] = r

    out_dir   = Path(output_dir)
    img_total = len([e for e in catalog if not e["skipped"] and not e["error"]])
    processed = 0

    for entry in catalog:
        if entry["skipped"] or entry["error"]:
            ocr_stats["skipped"] += 1
            continue
        if entry.get("ocr_text") and entry["ocr_text"] not in ("", "OCR skipped"):
            # Already has OCR text from a previous run
            ocr_stats["success"] += 1
            continue

        img_path = out_dir / entry["image_file"]
        if not img_path.exists():
            ocr_stats["errors"] += 1
            continue

        ocr_stats["total"] += 1
        processed          += 1
        print(
            f"  [{processed}/{img_total}] OCR {entry['image_id']} "
            f"({entry['width']}x{entry['height']}px)...",
            end=" ", flush=True,
        )

        try:
            text = ocr_image_with_ollama(img_path.read_bytes(), ollama_url, model)

            if not text.strip() or text.strip() == "[NO TEXT]":
                ocr_stats["no_text"] += 1
                text = "[NO TEXT]"
                print("(no text)")
            else:
                ocr_stats["success"] += 1
                preview = text[:70].replace("\n", " ")
                print(f'✓ "{preview}{"..." if len(text) > 70 else ""}"')

            # Update catalog entry
            entry["ocr_text"] = text

            # Update matching ImageOCRResult
            key = (entry["page_num"], entry["image_index"])
            if key in ocr_lookup:
                ocr_lookup[key].extracted_text = text
                ocr_lookup[key].model_used     = model

        except Exception as exc:
            ocr_stats["errors"] += 1
            print(f"✗ {exc}")
            entry["error"] = str(exc)

    # Persist updated data
    save_sections_json(sections, output_dir, ocr_stats, model, Path(output_dir).name)
    save_image_catalog(catalog, output_dir)
    save_checkpoint(output_dir, STEP_OCR, details=ocr_stats)

    print(f"\n  ✓  OCR complete: {ocr_stats['success']} with text, "
          f"{ocr_stats['no_text']} no text, {ocr_stats['errors']} errors")
    return sections, catalog, ocr_stats


# ══════════════════════════════════════════════════════════════════
# STEP 4 — WRITE TEXT FILES
# ══════════════════════════════════════════════════════════════════

def step_write_files(
    sections:      list,
    output_dir:    str,
    ocr_stats:     dict,
    model:         str,
    image_catalog: list,
    source_pdf:    str = "",
) -> None:
    """
    Write all human-readable outputs:
      ├── CH001_Introduction.txt   one .txt per section
      ├── _INDEX.txt
      ├── _metadata.json
      ├── _image_catalog.json      (already written, refreshed here)
      └── _sections.json           (already written, refreshed here)

    Saves checkpoint STEP_WRITE on success.
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
                saved = [r for r in sec.image_ocr_results if r.image_file]
                f.write(
                    f"IMAGES   : {len(sec.image_ocr_results)} found, "
                    f"{len(saved)} saved, {len(imgs_with_text)} with OCR text\n"
                )
            f.write("=" * 60 + "\n\n")
            f.write(sec.content())
            f.write("\n")

        indent   = indent_map.get(sec.level, "")
        img_note = f" [{len(imgs_with_text)} img(s) with text]" if imgs_with_text else ""
        index_lines.append(
            f"{indent}[{sec.level.upper()}] {sec.title}  →  {filename}{img_note}"
        )
        print(f"  ✓  {filename}  ({len(sec.content_lines)} lines, "
              f"{len(sec.image_ocr_results)} images)")

    (out / "_INDEX.txt").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"\n  ✓  _INDEX.txt")

    meta = {
        "source_pdf":  source_pdf,
        "model_used":  model,
        "ocr_summary": ocr_stats,
        "image_count": len(image_catalog),
        "sections": [
            {
                "level":       s.level,
                "title":       s.title,
                "char_count":  len(s.content()),
                "line_count":  len(s.content_lines),
                "image_count": len(s.image_ocr_results),
            }
            for s in sections
        ],
    }
    (out / "_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  ✓  _metadata.json")

    save_image_catalog(image_catalog, out)
    print(f"  ✓  {IMAGE_CATALOG_FILE}  ({len(image_catalog)} images)")

    save_sections_json(sections, output_dir, ocr_stats, model, source_pdf)
    print(f"  ✓  {SECTIONS_JSON_FILE}")

    save_checkpoint(output_dir, STEP_WRITE, details={"sections_written": len(sections)})


# ══════════════════════════════════════════════════════════════════
# HELPER — rebuild 'current' pointers from a loaded sections list
# ══════════════════════════════════════════════════════════════════

def _rebuild_current(sections: list) -> dict:
    """
    Reconstruct the {chapter, section, subsection} pointer dict from an
    existing sections list (used when resuming after a checkpoint load).
    """
    current = {"chapter": None, "section": None, "subsection": None}
    for sec in sections:
        if sec.level in current:
            current[sec.level] = sec
    return current


# ══════════════════════════════════════════════════════════════════
# LEGACY WRAPPER — kept so pdf_compare.py can still call it
# ══════════════════════════════════════════════════════════════════

def extract_structure(
    pdf_path:       str,
    output_dir:     str,
    use_font_size:  bool = True,
    ollama_url:     str  = DEFAULT_OLLAMA_URL,
    model:          str  = DEFAULT_MODEL,
    skip_images:    bool = False,
    min_image_size: int  = MIN_IMAGE_SIZE,
) -> tuple:
    """
    Convenience wrapper that runs all three extraction steps in sequence.
    Prefer calling the individual step_* functions for resumable pipelines.
    Returns (sections, ocr_stats, image_catalog).
    """
    sections = step_extract_text(pdf_path, output_dir, use_font_size)
    sections, catalog = step_save_images(pdf_path, output_dir, sections, min_image_size)
    if not skip_images:
        sections, catalog, ocr_stats = step_run_ocr(
            output_dir, sections, catalog, ollama_url, model
        )
    else:
        ocr_stats = {"total": 0, "success": 0, "skipped": 0, "errors": 0, "no_text": 0}
        save_checkpoint(output_dir, STEP_OCR, status="skipped",
                        details={"reason": "--skip-images"})
    return sections, ocr_stats, catalog


# ══════════════════════════════════════════════════════════════════
# OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════════

def safe_filename(title: str, max_len: int = 60) -> str:
    name = re.sub(r"[^\w\s\-]", "", title)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:max_len]


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDF Section Extractor — modular, checkpoint-based pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modular workflow (each step saves to disk — resume at any point):
  Step 1  python pdf_section_extractor.py report.pdf extracted/
  Step 2  python pdf_image_describer.py   extracted/
  Step 3  python pdf_compare.py           extracted_a/ extracted_b/

Resume after a failure (skips completed steps automatically):
  python pdf_section_extractor.py report.pdf extracted/ --resume

Check what has been done:
  python pdf_section_extractor.py extracted/ --status

Run only specific steps:
  python pdf_section_extractor.py report.pdf extracted/ --steps text
  python pdf_section_extractor.py report.pdf extracted/ --steps text,images
  python pdf_section_extractor.py report.pdf extracted/ --steps ocr,write

Popular Ollama vision models:
  llava             ollama pull llava
  llama3.2-vision   ollama pull llama3.2-vision
  moondream         ollama pull moondream
        """,
    )
    parser.add_argument("pdf",
        nargs="?",
        help="Path to input PDF (omit when using --status on an existing output dir)")
    parser.add_argument("output_dir",
        nargs="?", default="pdf_sections",
        help="Output directory (default: pdf_sections/)")
    parser.add_argument("--model",      default=DEFAULT_MODEL,
        help=f"Ollama vision model for OCR (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL,
        help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("--skip-images", action="store_true",
        help="Save images but skip OCR")
    parser.add_argument("--min-image-size", type=int, default=MIN_IMAGE_SIZE,
        help=f"Min image side in px to save (default: {MIN_IMAGE_SIZE})")
    parser.add_argument("--no-font-size", action="store_true",
        help="Use regex heading detection instead of font-size detection")
    parser.add_argument("--resume", action="store_true",
        help="Skip steps already marked completed in _pipeline_state.json")
    parser.add_argument("--steps", default="",
        help="Comma-separated subset of steps to run: text,images,ocr,write "
             "(default: all steps)")
    parser.add_argument("--status", action="store_true",
        help="Print pipeline progress for output_dir and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── --status: just print progress and exit ────────────────────
    if args.status:
        pipeline_status(args.output_dir)
        return

    if not args.pdf:
        print("✗  Please provide a PDF file path.")
        print("   Usage: python pdf_section_extractor.py <pdf> [output_dir]")
        sys.exit(1)

    if not os.path.exists(args.pdf):
        print(f"✗  File not found: {args.pdf}")
        sys.exit(1)

    out_dir       = args.output_dir
    use_font_size = not args.no_font_size

    # Determine which steps to run
    step_filter = {s.strip().lower() for s in args.steps.split(",") if s.strip()}
    def should_run(step_name: str, step_key: str) -> bool:
        if step_filter and step_name not in step_filter:
            return False
        if args.resume and step_is_done(out_dir, step_key):
            print(f"  ⏭   {step_name} already complete — skipping (--resume)")
            return False
        return True

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    set_source_pdf(out_dir, args.pdf)

    print(f"\n📄  PDF    : {args.pdf}")
    print(f"📁  Output : {out_dir}")
    print(f"🤖  Model  : {args.model}")
    print(f"🌐  Ollama : {args.ollama_url}")
    print(f"🔁  Resume : {args.resume}")
    if step_filter:
        print(f"🎯  Steps  : {', '.join(sorted(step_filter))}")
    print()

    # ── Step 1: Extract text ──────────────────────────────────────
    sections = None
    if should_run("text", STEP_TEXT):
        print("── Step 1/4  Extract text ──────────────────────────────")
        sections = step_extract_text(args.pdf, out_dir, use_font_size)
        if not sections:
            print("  ⚠  No sections via font-size. Retrying with regex...")
            sections = step_extract_text(args.pdf, out_dir, False)
        if not sections:
            print("✗  No structure detected.")
            sys.exit(1)

    # ── Step 2: Save images ───────────────────────────────────────
    catalog = []
    if should_run("images", STEP_IMAGES):
        print("\n── Step 2/4  Save images ───────────────────────────────")
        if sections is None:
            sections, _, _ = load_sections_json(out_dir)
        sections, catalog = step_save_images(
            args.pdf, out_dir, sections, args.min_image_size
        )

    # ── Step 3: Run OCR ───────────────────────────────────────────
    ocr_stats = {}
    if should_run("ocr", STEP_OCR) and not args.skip_images:
        print("\n── Step 3/4  Run OCR ───────────────────────────────────")
        if sections is None:
            sections, _, _ = load_sections_json(out_dir)
        if not catalog:
            catalog = load_image_catalog(out_dir).get("images", [])
        sections, catalog, ocr_stats = step_run_ocr(
            out_dir, sections, catalog, args.ollama_url, args.model
        )
    elif args.skip_images:
        save_checkpoint(out_dir, STEP_OCR, status="skipped",
                        details={"reason": "--skip-images"})

    # ── Step 4: Write files ───────────────────────────────────────
    if should_run("write", STEP_WRITE):
        print("\n── Step 4/4  Write files ───────────────────────────────")
        if sections is None:
            sections, _, _ = load_sections_json(out_dir)
        if not catalog:
            raw = load_image_catalog(out_dir)
            catalog = raw.get("images", []) if isinstance(raw, dict) else raw
        step_write_files(
            sections, out_dir, ocr_stats or {}, args.model, catalog, args.pdf
        )

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  ✅  Pipeline complete!")
    pipeline_status(out_dir)
    if not args.skip_images and catalog:
        print(f"  Next: python pdf_image_describer.py {out_dir}/")
    print(f"{'═'*58}\n")


if __name__ == "__main__":
    main()
