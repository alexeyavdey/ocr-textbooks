"""
Rebuild <stem>.json and <stem>.md for documents whose pdfplumber payload
was generated with the buggy stripe-carving logic (see commit 78d4729).

Full-OCR docs are untouched — their JSON came from sotaocr.
Pure-pdfplumber docs are regenerated from the source PDF.
Hybrid docs reuse the existing OCR pages and rebuild only the pdfplumber half.

No API calls. Run locally or on the server after `git pull`.

Usage:
    python3 rebuild_pdfplumber.py                    # walk docs/, fix everything
    python3 rebuild_pdfplumber.py docs/<subfolder>   # restrict to a subtree
    python3 rebuild_pdfplumber.py --dry-run          # only print what would change
"""
import argparse
import json
import sys
from pathlib import Path

from ocr_sotaocr import (
    DOCS_DIR,
    extract_pages,
    pdfplumber_to_payload,
    write_atomic,
)
from sotaocr_json_to_md import render_document


def classify(job_meta: dict) -> str:
    if job_meta.get("mode") == "hybrid":
        return "hybrid"
    job_id = (job_meta.get("id") or "")
    if job_id.startswith("pdfplumber-local"):
        return "pdfplumber"
    if job_id.startswith("job_"):
        return "ocr"
    return "unknown"


def find_docs(root: Path):
    """Yield (folder, pdf_path, job_path, json_path, md_path)."""
    for job_path in sorted(root.glob("**/*.job.json")):
        # ocr-job.json is the inner hybrid job; skip it as a doc marker.
        if job_path.name.endswith(".ocr-job.json"):
            continue
        stem = job_path.name[: -len(".job.json")]
        folder = job_path.parent
        pdf = folder / f"{stem}.pdf"
        if not pdf.exists():
            continue
        yield folder, pdf, job_path, folder / f"{stem}.json", folder / f"{stem}.md"


def rebuild_pdfplumber(pdf: Path, json_path: Path, md_path: Path) -> int:
    images_dir = pdf.parent / "images"
    payload = pdfplumber_to_payload(pdf, images_dir)
    write_atomic(json_path, json.dumps(payload, ensure_ascii=False, indent=2))
    pages = extract_pages(payload)
    write_atomic(md_path, render_document(pages))
    return len(pages)


def rebuild_hybrid(pdf: Path, json_path: Path, md_path: Path,
                   ocr_pages: list[int]) -> tuple[int, int]:
    # 1. Pull OCR-derived pages out of the existing merged JSON.
    if not json_path.exists():
        # No prior JSON to salvage from; fall back to full pdfplumber
        # (the OCR sub-job result is in <stem>.ocr-job.json but reading
        # it here would re-import too much — just regenerate cleanly).
        n = rebuild_pdfplumber(pdf, json_path, md_path)
        return n, 0
    existing = json.loads(json_path.read_text(encoding="utf-8"))
    existing_pages = extract_pages(existing)
    ocr_set = set(ocr_pages or [])
    kept: dict[int, dict] = {
        p["page_number"]: p for p in existing_pages if p.get("page_number") in ocr_set
    }

    # 2. Regenerate pdfplumber side from PDF.
    images_dir = pdf.parent / "images"
    fresh = pdfplumber_to_payload(pdf, images_dir)
    by_num: dict[int, dict] = {p["page_number"]: p for p in extract_pages(fresh)}

    # 3. Overlay OCR pages on top of fresh pdfplumber pages.
    for n, page in kept.items():
        by_num[n] = page

    merged_pages = sorted(by_num.values(), key=lambda p: p.get("page_number") or 0)
    stem = pdf.stem
    inner = {
        "job_id": f"hybrid-{stem}",
        "page_count": len(merged_pages),
        "pages": merged_pages,
    }
    merged_payload = {
        "job_id": f"hybrid-{stem}",
        "format": "json",
        "page_count": len(merged_pages),
        "content": json.dumps(inner, ensure_ascii=False),
    }
    write_atomic(json_path, json.dumps(merged_payload, ensure_ascii=False, indent=2))
    write_atomic(md_path, render_document(merged_pages))
    return len(merged_pages), len(kept)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default=str(DOCS_DIR),
                   help=f"Folder to walk (default: {DOCS_DIR}).")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't write anything; only list what would be rebuilt.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Not found: {root}", file=sys.stderr)
        return 1

    counts = {"pdfplumber": 0, "hybrid": 0, "ocr": 0, "unknown": 0, "skipped": 0}
    for folder, pdf, job_path, json_path, md_path in find_docs(root):
        try:
            meta = json.loads(job_path.read_text(encoding="utf-8"))
        except Exception as err:
            print(f"  SKIP {folder}: bad job.json ({err})", file=sys.stderr)
            counts["skipped"] += 1
            continue
        mode = classify(meta)
        rel = folder.relative_to(root) if folder != root else folder.name
        if mode == "ocr":
            counts["ocr"] += 1
            continue
        if mode == "unknown":
            print(f"  SKIP {rel}: unknown job id {meta.get('id')!r}")
            counts["unknown"] += 1
            continue
        if args.dry_run:
            extra = (f" (ocr_pages={len(meta.get('ocr_pages') or [])})"
                     if mode == "hybrid" else "")
            print(f"  WOULD rebuild [{mode}] {rel}{extra}")
            counts[mode] += 1
            continue
        print(f"  rebuilding [{mode}] {rel}…")
        try:
            if mode == "pdfplumber":
                n = rebuild_pdfplumber(pdf, json_path, md_path)
                print(f"    saved {n} pages")
            else:  # hybrid
                n, kept = rebuild_hybrid(pdf, json_path, md_path,
                                         meta.get("ocr_pages") or [])
                print(f"    saved {n} pages (kept {kept} OCR pages)")
            counts[mode] += 1
        except Exception as err:
            print(f"    FAILED: {err}", file=sys.stderr)
            counts["skipped"] += 1

    print()
    print(f"Summary: pdfplumber={counts['pdfplumber']}, "
          f"hybrid={counts['hybrid']}, ocr-skipped={counts['ocr']}, "
          f"unknown={counts['unknown']}, errors={counts['skipped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
