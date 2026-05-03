"""
PDF Image Describer
===================
Reads the image catalog produced by pdf_section_extractor.py, sends each
saved image to a local Ollama vision model, and generates a rich natural-
language description of what is depicted.  Descriptions are written back
into both _image_catalog.json and _sections.json so every downstream tool
(pdf_compare.py, etc.) can see them.

Modular workflow
────────────────
  Step 1   python pdf_section_extractor.py  report.pdf   extracted/
  Step 2   python pdf_image_describer.py    extracted/           ← this script
  Step 3   python pdf_compare.py            extracted_a/ extracted_b/

File I/O
────────
  Reads   (from pdf_section_extractor.py output):
      <dir>/_image_catalog.json   catalog of every saved image
      <dir>/images/*.png|jpg|...  the actual image files
      <dir>/_sections.json        full section data

  Writes (in-place updates):
      <dir>/_image_catalog.json   adds "description" to every entry
      <dir>/_sections.json        adds "description" to every ImageOCRResult
      <dir>/_descriptions.txt     human-readable report of all descriptions

Usage:
    python pdf_image_describer.py <extracted_dir> [options]

Options:
    --model MODEL       Ollama vision model (default: llava)
    --ollama-url URL    Ollama server URL (default: http://localhost:11434)
    --force             Re-describe images that already have a description
    --skip LIST         Comma-separated image_ids to skip, e.g. p001_i00,p002_i01
    --min-size N        Skip images smaller than NxN pixels (default: 50)

Requirements:
    pip install requests
    (pdf_section_extractor.py must be in the same directory)
"""

import os
import sys
import re
import json
import base64
import time
import argparse
from pathlib import Path
from datetime import datetime

# ── Import shared constants + loaders from the extractor ─────────────────────
_HERE = Path(globals().get("__file__", ".")).parent
sys.path.insert(0, str(_HERE))

try:
    from pdf_section_extractor import (
        load_image_catalog,
        load_sections_json,
        save_sections_json,
        DEFAULT_OLLAMA_URL,
        DEFAULT_MODEL,
        MIN_IMAGE_SIZE,
        IMAGE_CATALOG_FILE,
        SECTIONS_JSON_FILE,
    )
except ImportError:
    print("✗  Could not import pdf_section_extractor.py")
    print("   Make sure it is in the same directory as this script.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

DESCRIPTIONS_REPORT = "_descriptions.txt"

DESCRIPTION_PROMPT = """\
Provide a detailed, precise description of this image. Include:
- What type of image it is (photograph, diagram, chart, table, illustration, screenshot, etc.)
- The main subject or content depicted
- Any visible text, labels, axis titles, legends, or captions
- For charts/graphs: the data trend, axes, and key values if visible
- For tables: the column headers and general content
- For diagrams: the components, relationships, and flow
- Any colours, shapes, or visual patterns that carry meaning
- The overall context or purpose this image appears to serve

Be specific and thorough. Do not start with "This image shows" — begin directly with what it depicts.\
"""


# ══════════════════════════════════════════════════════════════════
# OLLAMA
# ══════════════════════════════════════════════════════════════════

def check_ollama(base_url: str, model: str) -> bool:
    """Return True if Ollama is running and the model is available."""
    import requests as _req
    try:
        resp = _req.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        available = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        if model.split(":")[0] not in available:
            print(f"  ⚠  Model '{model}' not found.")
            print(f"     Available: {', '.join(available) or 'none'}")
            print(f"     Run: ollama pull {model}")
            return False
        return True
    except _req.exceptions.ConnectionError:
        print(f"  ✗  Cannot connect to Ollama at {base_url}")
        print(f"     Make sure Ollama is running: ollama serve")
        return False
    except Exception as exc:
        print(f"  ✗  Ollama check failed: {exc}")
        return False


def describe_image_with_ollama(
    image_bytes: bytes,
    base_url:    str,
    model:       str,
    retries:     int = 2,
) -> str:
    """
    Send image bytes to Ollama and return a natural-language description.
    """
    import requests as _req

    payload = {
        "model":  model,
        "prompt": DESCRIPTION_PROMPT,
        "images": [base64.b64encode(image_bytes).decode("utf-8")],
        "stream": False,
        "options": {
            "temperature": 0.3,     # slight creativity for natural prose
            "num_predict": 800,
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
                time.sleep(2)
            else:
                return "[Description timed out]"
        except Exception as exc:
            if attempt < retries:
                time.sleep(1)
            else:
                return f"[Description error: {exc}]"

    return "[Description unavailable]"


# ══════════════════════════════════════════════════════════════════
# CATALOG UPDATE
# ══════════════════════════════════════════════════════════════════

def update_catalog_file(catalog_data: dict, output_dir: Path) -> None:
    """Write the updated catalog dict back to _image_catalog.json."""
    catalog_data["descriptions_updated"] = datetime.now().isoformat(timespec="seconds")
    path = output_dir / IMAGE_CATALOG_FILE
    path.write_text(
        json.dumps(catalog_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def update_sections_file(descriptions: dict, output_dir: Path) -> None:
    """
    Patch _sections.json in-place: write the description string into every
    ImageOCRResult whose image_file matches a key in ``descriptions``.

    descriptions: { "images/p001_i00.png": "A bar chart showing ..." }
    """
    sections_path = output_dir / SECTIONS_JSON_FILE
    if not sections_path.exists():
        print(f"  ⚠  {SECTIONS_JSON_FILE} not found — skipping sections update.")
        return

    data = json.loads(sections_path.read_text(encoding="utf-8"))

    updated = 0
    for sec in data.get("sections", []):
        for img in sec.get("image_ocr_results", []):
            key = img.get("image_file", "")
            if key in descriptions:
                img["description"] = descriptions[key]
                updated += 1

    sections_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  ✓  Updated {updated} image record(s) in {SECTIONS_JSON_FILE}")


# ══════════════════════════════════════════════════════════════════
# TEXT REPORT
# ══════════════════════════════════════════════════════════════════

def write_descriptions_report(
    results:    list,
    output_dir: Path,
    model:      str,
    source_pdf: str,
) -> None:
    """
    Write a human-readable _descriptions.txt with every image's
    metadata, OCR text, and AI description.

    results: list of dicts — one per processed image, with keys:
        image_id, image_file, page_num, image_index, width, height,
        section_title, section_level, ocr_text, description, error
    """
    lines = [
        "IMAGE DESCRIPTIONS REPORT",
        "=" * 70,
        f"  Source PDF : {source_pdf}",
        f"  Model      : {model}",
        f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Images     : {len(results)}",
        "=" * 70,
        "",
    ]

    for i, r in enumerate(results, 1):
        lines += [
            f"[{i}] Image ID   : {r['image_id']}",
            f"    File       : {r['image_file']}",
            f"    Page       : {r['page_num']}",
            f"    Size       : {r['width']}x{r['height']} px",
            f"    Section    : [{r['section_level'].upper()}] {r['section_title']}",
        ]

        if r.get("ocr_text") and r["ocr_text"].strip() not in ("", "[NO TEXT]", "OCR skipped"):
            lines += ["    OCR Text   :", ""]
            for ln in r["ocr_text"].strip().splitlines():
                lines.append(f"      {ln}")
            lines.append("")

        if r.get("error"):
            lines.append(f"    ⚠  Error   : {r['error']}")
        elif r.get("description"):
            lines += ["    Description:", ""]
            for ln in r["description"].strip().splitlines():
                lines.append(f"      {ln}")
        else:
            lines.append("    Description: (not generated)")

        lines += ["", "-" * 70, ""]

    path = output_dir / DESCRIPTIONS_REPORT
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓  {DESCRIPTIONS_REPORT}  ({len(results)} entries)")


# ══════════════════════════════════════════════════════════════════
# MAIN DESCRIBE LOOP
# ══════════════════════════════════════════════════════════════════

def describe_all_images(
    extracted_dir:  str,
    model:          str  = DEFAULT_MODEL,
    ollama_url:     str  = DEFAULT_OLLAMA_URL,
    force:          bool = False,
    skip_ids:       set  = None,
    min_size:       int  = MIN_IMAGE_SIZE,
) -> None:
    """
    Main pipeline:
      1. Load _image_catalog.json
      2. For each image: load bytes → send to Ollama → store description
      3. Update _image_catalog.json and _sections.json in-place
      4. Write _descriptions.txt
    """
    out_dir = Path(extracted_dir)
    skip_ids = skip_ids or set()

    # ── Load catalog ──────────────────────────────────────────────
    try:
        catalog_data = load_image_catalog(out_dir)
    except FileNotFoundError as exc:
        print(f"✗  {exc}")
        sys.exit(1)

    images = catalog_data.get("images", [])
    total  = len(images)

    # Try to get source_pdf from _sections.json for the report header
    source_pdf = ""
    try:
        _, _, meta = load_sections_json(out_dir)
        source_pdf = meta.get("source_pdf", "")
    except Exception:
        pass

    print(f"\n  Catalog loaded: {total} image(s) from '{source_pdf or extracted_dir}'")

    # ── Ollama pre-flight ─────────────────────────────────────────
    if not check_ollama(ollama_url, model):
        sys.exit(1)
    print(f"  ✓  Ollama ready — model: {model}\n")

    # ── Describe loop ─────────────────────────────────────────────
    # descriptions maps relative image_file → description string
    descriptions = {}
    processed    = []
    stats        = {"described": 0, "skipped_size": 0, "skipped_existing": 0,
                    "skipped_user": 0, "errors": 0, "no_file": 0}

    for i, entry in enumerate(images, 1):
        image_id   = entry.get("image_id",      f"img_{i:04d}")
        image_file = entry.get("image_file",     "")
        w          = entry.get("width",          0)
        h          = entry.get("height",         0)
        skipped    = entry.get("skipped",        False)
        error_prev = entry.get("error",          "")
        existing   = entry.get("description",    "").strip()

        print(f"  [{i}/{total}] {image_id}  ({w}x{h}px)", end="")

        # Skip: too small or previously marked skipped
        if skipped or w < min_size or h < min_size:
            print("  → skipped (too small)")
            stats["skipped_size"] += 1
            entry["description"] = entry.get("description", "")
            processed.append(entry)
            continue

        # Skip: user-specified list
        if image_id in skip_ids:
            print("  → skipped (--skip list)")
            stats["skipped_user"] += 1
            processed.append(entry)
            continue

        # Skip: already has description and --force not set
        if existing and not force:
            print("  → already described (use --force to redo)")
            stats["skipped_existing"] += 1
            descriptions[image_file] = existing
            processed.append(entry)
            continue

        # Resolve absolute path to the image file
        img_path = out_dir / image_file
        if not img_path.exists():
            print(f"  → ✗ file not found: {img_path}")
            stats["no_file"] += 1
            entry["description"] = "[Image file not found]"
            processed.append(entry)
            continue

        # ── Send to Ollama ────────────────────────────────────────
        print("  → describing...", end=" ", flush=True)
        try:
            img_bytes   = img_path.read_bytes()
            description = describe_image_with_ollama(img_bytes, ollama_url, model)

            if description.startswith("["):
                # Error or timeout marker
                stats["errors"] += 1
                print(f"⚠  {description}")
            else:
                stats["described"] += 1
                preview = description[:80].replace("\n", " ")
                suffix  = "..." if len(description) > 80 else ""
                print(f'✓ "{preview}{suffix}"')

            entry["description"] = description
            descriptions[image_file] = description

        except Exception as exc:
            stats["errors"] += 1
            print(f"✗ {exc}")
            entry["description"] = f"[Error: {exc}]"

        processed.append(entry)

    # ── Persist updates ───────────────────────────────────────────
    print(f"\n  Saving updates...\n")

    catalog_data["images"] = processed
    update_catalog_file(catalog_data, out_dir)
    print(f"  ✓  {IMAGE_CATALOG_FILE} updated")

    update_sections_file(descriptions, out_dir)

    write_descriptions_report(processed, out_dir, model, source_pdf)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  ✅  Descriptions complete!")
    print(f"      Described        : {stats['described']}")
    print(f"      Already had desc : {stats['skipped_existing']}")
    print(f"      Too small        : {stats['skipped_size']}")
    print(f"      Skipped (user)   : {stats['skipped_user']}")
    print(f"      File not found   : {stats['no_file']}")
    print(f"      Errors           : {stats['errors']}")
    print(f"      Output dir       : {extracted_dir}/")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AI descriptions for images extracted by pdf_section_extractor.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  # Step 1 — extract (images saved automatically)
  python pdf_section_extractor.py report.pdf extracted/

  # Step 2 — describe (reads catalog, writes descriptions back)
  python pdf_image_describer.py extracted/

  # Step 3 — compare (reads _sections.json which now has descriptions)
  python pdf_compare.py extracted_a/ extracted_b/

Options examples:
  # Use a different vision model
  python pdf_image_describer.py extracted/ --model llama3.2-vision

  # Re-describe everything even if descriptions already exist
  python pdf_image_describer.py extracted/ --force

  # Skip specific images by ID
  python pdf_image_describer.py extracted/ --skip p001_i00,p003_i01

  # Ignore images smaller than 100px
  python pdf_image_describer.py extracted/ --min-size 100

Popular Ollama vision models:
  llava             General purpose        ollama pull llava
  llama3.2-vision   Meta Llama 3.2         ollama pull llama3.2-vision
  moondream         Lightweight/fast       ollama pull moondream
  bakllava          Alternative llava      ollama pull bakllava
        """,
    )
    parser.add_argument(
        "extracted_dir",
        help="Directory produced by pdf_section_extractor.py (contains _image_catalog.json)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama vision model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ollama-url", default=DEFAULT_OLLAMA_URL,
        help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-describe images that already have a description",
    )
    parser.add_argument(
        "--skip", default="",
        help="Comma-separated image_ids to skip, e.g. p001_i00,p002_i01",
    )
    parser.add_argument(
        "--min-size", type=int, default=MIN_IMAGE_SIZE,
        help=f"Skip images smaller than N px on either side (default: {MIN_IMAGE_SIZE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.extracted_dir).exists():
        print(f"✗  Directory not found: {args.extracted_dir}")
        sys.exit(1)

    skip_ids = {s.strip() for s in args.skip.split(",") if s.strip()}

    print(f"\n{'═'*60}")
    print(f"  PDF Image Describer")
    print(f"{'═'*60}")
    print(f"  Source dir  : {args.extracted_dir}")
    print(f"  Model       : {args.model}")
    print(f"  Ollama URL  : {args.ollama_url}")
    print(f"  Force redo  : {args.force}")
    print(f"  Min size    : {args.min_size}px")
    if skip_ids:
        print(f"  Skip IDs    : {', '.join(sorted(skip_ids))}")
    print(f"{'═'*60}\n")

    describe_all_images(
        extracted_dir = args.extracted_dir,
        model         = args.model,
        ollama_url    = args.ollama_url,
        force         = args.force,
        skip_ids      = skip_ids,
        min_size      = args.min_size,
    )


if __name__ == "__main__":
    main()
