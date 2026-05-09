"""
Walk a Google Drive folder, download every PDF mirroring the folder structure
into docs/, and optionally feed each PDF into the OCR pipeline.

First-time setup
----------------
1. https://console.cloud.google.com/ → создать или выбрать проект.
2. APIs & Services → Library → найти "Google Drive API" → Enable.
3. APIs & Services → OAuth consent screen → External → заполнить.
4. APIs & Services → Credentials → Create OAuth client ID → Application type
   "Desktop app" → Download JSON → положить как credentials.json в корень
   проекта.
5. python3 download_drive.py <folder-url>
   → откроется браузер, нажмите Allow → токен сохранится в token.json.

Usage
-----
    python3 download_drive.py <folder-url-or-id>
    python3 download_drive.py <folder-url> --no-process     # только скачать
    python3 download_drive.py <folder-url> --dry-run         # ничего не скачивать
"""
import argparse
import re
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

import ocr_sotaocr

PROJECT_DIR = Path(__file__).parent
DOCS_DIR = PROJECT_DIR / "docs"
CREDENTIALS_PATH = PROJECT_DIR / "credentials.json"
TOKEN_PATH = PROJECT_DIR / "token.json"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
PDF_MIME = "application/pdf"
FOLDER_MIME = "application/vnd.google-apps.folder"

# Filesystem-unsafe characters (Linux/macOS/Windows union).
_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(name: str) -> str:
    s = _UNSAFE_RE.sub("_", name).strip(". ")
    # Reject path-traversal markers so a maliciously-named Drive folder can't
    # write outside docs/.
    if not s or s in (".", ".."):
        return "unnamed"
    return s


def folder_id_from(arg: str) -> str:
    if "/" not in arg and "?" not in arg:
        return arg
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", arg)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract folder id from: {arg}")


def get_service():
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                raise RuntimeError(
                    f"OAuth client file is missing: {CREDENTIALS_PATH}\n"
                    "Создайте Desktop OAuth client в Google Cloud Console и "
                    "положите его JSON как credentials.json (см. шапку файла)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
        print(f"  saved token → {TOKEN_PATH}")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_children(service, folder_id: str) -> list[dict]:
    items: list[dict] = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType, size, md5Checksum)",
            pageToken=page_token,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def walk(service, folder_id: str, parts: list[str],
         exclude: set[str] | None = None):
    """Yield (path_parts, file_meta) for every non-folder descendant.

    `exclude` is a set of lowercased folder names to skip entirely (the
    folder itself and everything under it).
    """
    excl = exclude or set()
    for item in list_children(service, folder_id):
        name = item.get("name", "")
        mt = item.get("mimeType", "")
        if mt == FOLDER_MIME:
            if name.lower() in excl:
                continue
            yield from walk(service, item["id"], parts + [name], exclude=excl)
        else:
            yield (parts, item)


def download_file(service, file_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    request = service.files().get_media(fileId=file_id)
    with tmp.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    tmp.rename(out_path)


def target_for(parts: list[str], filename: str) -> Path:
    safe_parts = [safe_name(p) for p in parts]
    fname = safe_name(filename)
    stem = Path(fname).stem
    base = DOCS_DIR.joinpath(*safe_parts) if safe_parts else DOCS_DIR
    return base / stem / fname


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folder", help="Drive folder URL or ID.")
    p.add_argument("--no-process", action="store_true",
                   help="Only download PDFs; do not run OCR pipeline.")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be downloaded; touch nothing.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N PDFs (useful for first run).")
    p.add_argument("--exclude", action="append", default=[], metavar="NAME",
                   help="Skip a Drive subfolder by exact name (case-insensitive). "
                   "Repeatable: --exclude 'Начальная школа' --exclude 'Литература'.")
    # Pipeline flags shared with ocr_sotaocr (only the relevant subset).
    p.add_argument("--no-formulas", action="store_true",
                   help="Pass through to OCR pipeline.")
    p.add_argument("--has-formulas", action="store_true",
                   help="Pass through to OCR pipeline.")
    p.add_argument("--force-ocr", action="store_true",
                   help="Pass through to OCR pipeline.")
    p.add_argument("--force", action="store_true",
                   help="Pass through to OCR pipeline.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    folder_id = folder_id_from(args.folder)
    print(f"Drive folder: {folder_id}")

    # Auth always needed (even for --dry-run we still list).
    try:
        service = get_service()
    except Exception as err:
        print(f"Auth failed: {err}", file=sys.stderr)
        return 1

    # Pre-build md5 index of already-processed docs so we can recognise
    # duplicates that come from Drive under different names without
    # downloading them again.
    md5_index = ocr_sotaocr.find_md5_index()
    seen_md5: dict[str, Path] = {}  # md5 → first target_folder this run

    exclude_set = {n.lower() for n in args.exclude}
    if exclude_set:
        print(f"Excluding folders: {sorted(args.exclude)}")

    pdfs: list[Path] = []
    skipped: list[str] = []
    duplicates: list[str] = []
    seen = 0
    for parts, item in walk(service, folder_id, [], exclude=exclude_set):
        full_path = "/".join(parts + [item["name"]])
        seen += 1
        if item.get("mimeType") != PDF_MIME:
            skipped.append(full_path)
            print(f"  skip non-pdf: {full_path}")
            continue

        target = target_for(parts, item["name"])
        target_folder = target.parent
        rel = target.relative_to(PROJECT_DIR)

        # Drive-side dedup: skip if a content-identical PDF lives at a
        # different docs/<...>/<stem>/ already, or if we just downloaded
        # one earlier in this same walk.
        drive_md5 = (item.get("md5Checksum") or "").lower()
        if drive_md5 and len(drive_md5) == 32:
            first_run = seen_md5.get(drive_md5)
            if first_run is not None and first_run != target_folder:
                rel_first = first_run.relative_to(PROJECT_DIR)
                print(f"  dup-in-drive: {full_path} == {rel_first} (md5={drive_md5[:8]}…)")
                duplicates.append(f"{full_path} == {rel_first}")
                continue
            existing = md5_index.get(drive_md5)
            if existing is not None and existing != target_folder:
                rel_existing = existing.relative_to(PROJECT_DIR)
                print(f"  dup-in-docs: {full_path} == {rel_existing} (md5={drive_md5[:8]}…)")
                duplicates.append(f"{full_path} == {rel_existing}")
                continue
            seen_md5[drive_md5] = target_folder

        if target.exists():
            print(f"  cached: {rel}")
        elif args.dry_run:
            print(f"  would download: {full_path} → {rel}")
            pdfs.append(target)
            continue
        else:
            size = item.get("size") or "?"
            print(f"  downloading ({size}b): {full_path} → {rel}")
            try:
                download_file(service, item["id"], target)
            except HttpError as err:
                print(f"    FAILED: {err}", file=sys.stderr)
                continue
        pdfs.append(target)
        if args.limit and len(pdfs) >= args.limit:
            print(f"  limit reached ({args.limit})")
            break

    print(
        f"\nSummary: {seen} items inspected, {len(pdfs)} PDF(s) ready, "
        f"{len(skipped)} non-PDF skipped, {len(duplicates)} duplicate(s) skipped"
    )
    if duplicates:
        print("Duplicates (same content, different names):")
        for d in duplicates:
            print(f"  - {d}")

    if args.dry_run or args.no_process:
        return 0

    if not pdfs:
        return 0

    # Run OCR pipeline on each PDF, in-place. We reuse process_pdf directly.
    pipeline_args = ocr_sotaocr.make_pipeline_args(
        force_ocr=args.force_ocr,
        force=args.force,
        no_formulas=args.no_formulas,
        has_formulas=args.has_formulas,
    )
    try:
        balance = ocr_sotaocr.check_balance()
        print(
            f"\nBalance: remaining_pages={balance.get('remaining_pages')}, "
            f"total_affordable_pages={balance.get('total_affordable_pages')}"
        )
    except Exception as err:
        print(f"Balance check failed: {err}", file=sys.stderr)
        return 1

    failures = []
    for pdf in pdfs:
        try:
            ocr_sotaocr.process_pdf(pdf, pipeline_args)
        except Exception as err:
            print(f"  FAILED {pdf.name}: {err}", file=sys.stderr)
            failures.append((pdf, err))

    if failures:
        print(f"\n{len(failures)} file(s) failed:", file=sys.stderr)
        for pdf, err in failures:
            print(f"  - {pdf}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
