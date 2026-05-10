"""
Profile pdfplumber_to_payload memory + time on a real textbook PDF.

Usage:
    python3 profile_pdfplumber.py <pdf-path>

Reports per-page peak RSS, total elapsed time, and the largest allocations
via tracemalloc snapshot.
"""
import argparse
import gc
import resource
import sys
import time
import tracemalloc
from pathlib import Path

import pdfplumber

from ocr_sotaocr import (
    PDFPLUMBER_RENDER_DPI,
    PDFPLUMBER_SCALE,
    _validated_tables,
    table_to_markdown,
)


def rss_mb() -> float:
    """Resident Set Size of this process, in MB. Linux returns kB, macOS bytes."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return r / 1024 / 1024
    return r / 1024


def profile_run(pdf_path: Path, render_previews: bool, flush_cache: bool) -> dict:
    """Re-implement pdfplumber_to_payload inline so we can sample memory mid-run."""
    images_dir = Path("/tmp/profile_previews")
    images_dir.mkdir(exist_ok=True)
    for f in images_dir.glob("*.png"):
        f.unlink()

    gc.collect()
    start_rss = rss_mb()
    t0 = time.time()
    peak_rss = start_rss
    pages_with_images = 0
    pages_total = 0
    text_chars = 0

    pages_out: list[dict] = []
    scale = PDFPLUMBER_SCALE
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_num = page.page_number
            page_w = page.width
            page_h = page.height
            full_text = (page.extract_text() or "").strip()
            text_chars += len(full_text)
            tables = _validated_tables(page, full_text)
            blocks: list[dict] = []
            if full_text:
                blocks.append({
                    "id": "0", "type": "text",
                    "bbox": [0, 0, int(page_w * scale), int(page_h * scale)],
                    "content": full_text, "order": 1,
                })
            for rows, bbox_pdf in tables:
                md = table_to_markdown(rows)
                if md:
                    blocks.append({
                        "id": str(len(blocks)), "type": "table",
                        "bbox": [int(c * scale) for c in bbox_pdf],
                        "content": md, "order": len(blocks) + 1,
                    })
            for img in page.images:
                bbox_pdf = (img["x0"], img["top"], img["x1"], img["bottom"])
                blocks.append({
                    "id": str(len(blocks)), "type": "image",
                    "bbox": [int(c * scale) for c in bbox_pdf],
                    "content": "", "order": len(blocks) + 1,
                })
            has_images = any(b["type"] == "image" for b in blocks)
            if has_images:
                pages_with_images += 1
                if render_previews:
                    preview_path = images_dir / f"p{page_num}_preview.png"
                    page.to_image(resolution=PDFPLUMBER_RENDER_DPI).original.save(
                        preview_path, format="PNG"
                    )
            pages_out.append({
                "page_number": page_num, "status": "completed",
                "text": full_text,
                "page_preview": {
                    "content_type": "image/png",
                    "coordinate_space": "local_render",
                    "width": int(page_w * scale),
                    "height": int(page_h * scale),
                },
                "layout_blocks": blocks,
            })
            pages_total += 1
            if flush_cache:
                page.flush_cache()
                try:
                    page.get_textmap.cache_clear()
                except AttributeError:
                    pass
            cur = rss_mb()
            if cur > peak_rss:
                peak_rss = cur
            if pages_total % 25 == 0:
                print(f"    page {pages_total:>3}  rss={cur:>6.1f} MB  "
                      f"peak={peak_rss:>6.1f} MB", flush=True)

    elapsed = time.time() - t0
    end_rss = rss_mb()
    return {
        "pages": pages_total,
        "pages_with_images": pages_with_images,
        "text_chars": text_chars,
        "start_rss": start_rss,
        "end_rss": end_rss,
        "peak_rss": peak_rss,
        "elapsed_s": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--no-previews", action="store_true",
                    help="Skip page rendering (isolates extract-text/objects cost)")
    ap.add_argument("--flush", action="store_true",
                    help="Call page.flush_cache() after each page")
    ap.add_argument("--tracemalloc", action="store_true",
                    help="Show top-10 allocators (slower)")
    args = ap.parse_args()

    if args.tracemalloc:
        tracemalloc.start(25)

    print(f"PDF: {args.pdf} ({args.pdf.stat().st_size/1024/1024:.1f} MB)")
    print(f"Mode: previews={not args.no_previews}  flush_cache={args.flush}")
    print()

    res = profile_run(args.pdf, render_previews=not args.no_previews,
                      flush_cache=args.flush)

    print()
    print(f"pages:             {res['pages']}")
    print(f"pages_with_images: {res['pages_with_images']}")
    print(f"text chars:        {res['text_chars']:,}")
    print(f"elapsed:           {res['elapsed_s']:.1f} s")
    print(f"rss start:         {res['start_rss']:.1f} MB")
    print(f"rss end:           {res['end_rss']:.1f} MB")
    print(f"rss peak:          {res['peak_rss']:.1f} MB")

    if args.tracemalloc:
        snap = tracemalloc.take_snapshot()
        stats = snap.statistics("filename")
        print()
        print("Top 10 allocators by file:")
        for s in stats[:10]:
            print(f"  {s.size/1024/1024:>7.1f} MB  {s.count:>7} blocks  {s.traceback}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
