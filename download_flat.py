"""
Download every PDF from a Google Drive folder (recursively) into the current
working directory, saving each as <md5>.pdf. Same Drive file under different
names is downloaded only once. Self-contained — does not import other project
modules.

Auth setup
----------
Needs credentials.json (OAuth Desktop client from Google Cloud Console) in the
CWD or alongside the script. First run opens a browser; token.json is saved
next to credentials.json.

Usage (from inside the target download folder):
    python3 /path/to/download_flat.py <drive-url-or-id>
    python3 /path/to/download_flat.py <url> --dry-run
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
PDF_MIME = "application/pdf"
FOLDER_MIME = "application/vnd.google-apps.folder"
SCRIPT_DIR = Path(__file__).resolve().parent


def find_credentials_file() -> Path:
    """Look for credentials.json in CWD first, then next to the script."""
    for d in (Path.cwd(), SCRIPT_DIR):
        p = d / "credentials.json"
        if p.exists():
            return p
    raise RuntimeError(
        "credentials.json not found in CWD or alongside the script.\n"
        "Create a Desktop OAuth client in Google Cloud Console and save it as "
        "credentials.json."
    )


def get_service():
    creds_path = find_credentials_file()
    token_path = creds_path.parent / "token.json"
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())
        print(f"  saved token → {token_path}")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def folder_id_from(arg: str) -> str:
    if "/" not in arg and "?" not in arg:
        return arg
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", arg)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract folder id from: {arg}")


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


def walk_pdfs(service, folder_id: str):
    """Yield (drive_path, file_meta) for every PDF descendant."""
    def _walk(fid: str, parts: list[str]):
        for item in list_children(service, fid):
            mt = item.get("mimeType", "")
            if mt == FOLDER_MIME:
                yield from _walk(item["id"], parts + [item["name"]])
            elif mt == PDF_MIME:
                yield ("/".join(parts + [item["name"]]), item)
    yield from _walk(folder_id, [])


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


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("folder", help="Drive folder URL or ID.")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be downloaded; touch nothing.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    folder_id = folder_id_from(args.folder)
    cwd = Path.cwd()
    print(f"Drive folder: {folder_id}")
    print(f"Output dir:   {cwd}")

    try:
        service = get_service()
    except Exception as err:
        print(f"Auth failed: {err}", file=sys.stderr)
        return 1

    seen_md5: set[str] = set()
    new = 0
    cached = 0
    duplicates = 0
    no_md5 = 0
    failed = 0

    for drive_path, item in walk_pdfs(service, folder_id):
        md5 = (item.get("md5Checksum") or "").lower()
        if md5 and len(md5) == 32:
            target = cwd / f"{md5}.pdf"
            if md5 in seen_md5:
                print(f"  dup  (run): {drive_path}")
                duplicates += 1
                continue
            seen_md5.add(md5)
            if target.exists():
                print(f"  have:       {drive_path} → {target.name}")
                cached += 1
                continue
            if args.dry_run:
                print(f"  would dl:   {drive_path} → {target.name}")
                new += 1
                continue
            print(f"  downloading {drive_path} → {target.name}")
            try:
                download_file(service, item["id"], target)
                new += 1
            except HttpError as err:
                print(f"    FAILED: {err}", file=sys.stderr)
                failed += 1
        else:
            # No Drive-side md5 — compute after download.
            tmp = cwd / f".incoming-{item['id']}.pdf"
            if args.dry_run:
                print(f"  would dl:   {drive_path} (no drive md5)")
                no_md5 += 1
                continue
            try:
                download_file(service, item["id"], tmp)
            except HttpError as err:
                print(f"  FAILED {drive_path}: {err}", file=sys.stderr)
                failed += 1
                continue
            md5 = md5_of_file(tmp)
            target = cwd / f"{md5}.pdf"
            if target.exists():
                tmp.unlink()
                print(f"  dup (post): {drive_path} → {target.name}")
                duplicates += 1
            else:
                tmp.rename(target)
                print(f"  saved:      {drive_path} → {target.name}")
                new += 1
            seen_md5.add(md5)
            no_md5 += 1

    print()
    print(f"Summary: new={new}, cached={cached}, duplicates={duplicates}, "
          f"no-drive-md5={no_md5}, failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
