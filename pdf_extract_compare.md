Here's the complete modular workflow:
Step 1 — Extract each PDF independently (run once per PDF):

bash
python pdf_section_extractor.py doc_a.pdf extracted_a/
python pdf_section_extractor.py doc_b.pdf extracted_b/

Step 2 — Compare the saved extractions:

bash
python pdf_compare.py extracted_a/ extracted_b/

What gets passed between steps via files:
FileWritten byRead byContainsextracted_a/_sections.jsonextractorcomparatorAll sections, content, image OCR textextracted_a/_INDEX.txtextractorhumanTable of contentsextracted_a/_metadata.jsonextractorhumanStats summarypdf_comparison/_COMPARISON_REPORT.htmlcomparatorhumanSide-by-side diff + AI summariespdf_comparison/_MATCHED.jsoncomparatordownstream toolsMachine-readable results
Key benefit of the modular design: extraction is slow (it runs Ollama OCR on every image). Now you only run it once per PDF. After that you can re-run comparisons instantly with different settings — lower match threshold, different AI model, text vs HTML report — without touching the PDFs again:

bash# Tweak threshold — no re-extraction needed
python pdf_compare.py extracted_a/ extracted_b/ --fuzzy-threshold 60

# Try a different summary model
python pdf_compare.py extracted_a/ extracted_b/ --text-model mistral

# Fast diff-only (no AI summaries)
python pdf_compare.py extracted_a/ extracted_b/ --skip-summary