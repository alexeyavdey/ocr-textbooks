"""
Walk a Google Drive folder and print a summary: total files, breakdown by mime
type, PDFs per folder, top-N largest PDFs. Useful before running
download_drive.py to estimate volume and OCR cost.

Auth setup: см. шапку download_drive.py (нужны credentials.json и token.json).

Usage:
    python3 inventory_drive.py <folder-url-or-id>
    python3 inventory_drive.py <url> --top 20         # top-20 крупнейших PDF
    python3 inventory_drive.py <url> --tree           # полный список PDF
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pdfplumber

from download_drive import (
    FOLDER_MIME,
    PDF_MIME,
    download_file,
    folder_id_from,
    get_service,
    target_for,
    walk,
)


def fmt_bytes(n: int) -> str:
    if n <= 0:
        return "—"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.1f} {u}" if u != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folder", help="Drive folder URL or ID.")
    p.add_argument("--top", type=int, default=10,
                   help="Top N largest PDFs (default 10; 0 to disable).")
    p.add_argument("--tree", action="store_true",
                   help="Print every PDF grouped by folder.")
    p.add_argument("--count-pages", action="store_true",
                   help="Download each PDF (or reuse cached copy in docs/) and "
                   "count pages with pdfplumber. Adds page totals to summary.")
    p.add_argument("--exclude", action="append", default=[], metavar="NAME",
                   help="Skip a Drive subfolder by exact name (case-insensitive). "
                   "Repeatable: --exclude 'Начальная школа'.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    folder_id = folder_id_from(args.folder)
    print(f"Drive folder: {folder_id}\n", flush=True)

    service = get_service()

    exclude_set = {n.lower() for n in args.exclude}
    if exclude_set:
        print(f"Excluding folders: {sorted(args.exclude)}")

    items: list[tuple[list[str], dict]] = []
    for parts, item in walk(service, folder_id, [], exclude=exclude_set):
        items.append((parts, item))

    by_mime: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [count, bytes]
    pdfs: list[tuple[list[str], dict]] = []
    for parts, item in items:
        mime = item.get("mimeType") or "?"
        size = int(item.get("size") or 0)
        by_mime[mime][0] += 1
        by_mime[mime][1] += size
        if mime == PDF_MIME:
            pdfs.append((parts, item))

    pdf_count = by_mime[PDF_MIME][0]
    pdf_bytes = by_mime[PDF_MIME][1]
    print(f"Total files: {len(items)}")
    print(f"PDFs:        {pdf_count}  ({fmt_bytes(pdf_bytes)})\n")

    print("By mime type (sorted by count):")
    for mime, (cnt, sz) in sorted(by_mime.items(), key=lambda x: -x[1][0]):
        print(f"  {cnt:4}  {fmt_bytes(sz):>10}  {mime}")

    by_folder: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for parts, item in pdfs:
        key = "/".join(parts) if parts else "(root)"
        sz = int(item.get("size") or 0)
        by_folder[key][0] += 1
        by_folder[key][1] += sz

    if by_folder:
        print(f"\nPDFs by folder ({len(by_folder)} folder(s)):")
        for folder in sorted(by_folder.keys()):
            cnt, sz = by_folder[folder]
            print(f"  {cnt:4}  {fmt_bytes(sz):>10}  {folder}/")

    if args.top and pdfs:
        sorted_pdfs = sorted(
            pdfs, key=lambda x: int(x[1].get("size") or 0), reverse=True
        )
        print(f"\nTop {min(args.top, len(sorted_pdfs))} largest PDFs:")
        for parts, item in sorted_pdfs[: args.top]:
            sz = int(item.get("size") or 0)
            full = "/".join(parts + [item["name"]])
            print(f"  {fmt_bytes(sz):>10}  {full}")

    if args.tree and pdfs:
        tree: dict[str, list[str]] = defaultdict(list)
        for parts, item in pdfs:
            tree["/".join(parts) if parts else "(root)"].append(item["name"])
        print("\nTree (PDFs only):")
        for folder in sorted(tree.keys()):
            print(f"\n{folder}/")
            for name in sorted(tree[folder]):
                print(f"  - {name}")

    if args.count_pages and pdfs:
        print(f"\nCounting pages in {len(pdfs)} PDF(s) "
              f"(cached files reused, missing ones downloaded)…")
        per_pdf: list[tuple[list[str], dict, int | None, Path]] = []
        downloaded = 0
        for i, (parts, item) in enumerate(pdfs, 1):
            target = target_for(parts, item["name"])
            if not target.exists():
                try:
                    download_file(service, item["id"], target)
                    downloaded += 1
                except Exception as err:
                    print(f"  [{i}/{len(pdfs)}] DOWNLOAD FAILED {item['name']}: {err}",
                          file=sys.stderr)
                    per_pdf.append((parts, item, None, target))
                    continue
            try:
                with pdfplumber.open(target) as pdf:
                    n = len(pdf.pages)
            except Exception as err:
                print(f"  [{i}/{len(pdfs)}] PARSE FAILED {item['name']}: {err}",
                      file=sys.stderr)
                n = None
            per_pdf.append((parts, item, n, target))
            tag = "↓" if not target.exists() else "•"  # cached marker
            print(f"  [{i}/{len(pdfs)}] {n if n is not None else '?':>5}p  "
                  f"{'/'.join(parts + [item['name']])}")

        valid = [(p, it, n, t) for p, it, n, t in per_pdf if n is not None]
        total_pages = sum(n for _, _, n, _ in valid)
        unparsed = len(per_pdf) - len(valid)

        print(f"\nTotal pages: {total_pages}  "
              f"(parsed {len(valid)}/{len(per_pdf)}; "
              f"downloaded {downloaded}; failed {unparsed})")

        by_folder_pages: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for parts, _, n, _ in valid:
            key = "/".join(parts) if parts else "(root)"
            by_folder_pages[key][0] += 1
            by_folder_pages[key][1] += n
        if by_folder_pages:
            print("Pages per folder:")
            for folder in sorted(by_folder_pages.keys()):
                cnt, pages = by_folder_pages[folder]
                avg = pages / cnt if cnt else 0
                print(f"  {pages:6}p  ({cnt:3} pdf, avg {avg:5.1f}/pdf)  {folder}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
