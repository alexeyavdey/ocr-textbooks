import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import pdfplumber
import requests
from dotenv import load_dotenv

from sotaocr_json_to_md import render_document

load_dotenv()

PROJECT_DIR = Path(__file__).parent
DOCS_DIR = PROJECT_DIR / "docs"
LOG_PATH = DOCS_DIR / "api.log"

API_BASE = "https://sotaocr.com/v1"
POLL_INTERVAL_SECONDS = 3.0
POLL_TIMEOUT_SECONDS = 60 * 60
UPLOAD_TIMEOUT_SECONDS = 300

# HTTP retry settings (transient 5xx + network errors).
RETRY_STATUSES = {502, 503, 504}
RETRY_MAX_ATTEMPTS = 4
RETRY_BASE_DELAY = 2.0

# Pre-OCR text check.
# A PDF counts as "text-based" only if both thresholds are met:
#   1. average chars/page >= MIN_TEXT_CHARS_PER_PAGE  (vs empty scans)
#   2. alpha_ratio >= MIN_ALPHA_RATIO                 (vs garbled cmaps that
#                                                       decode as ASCII punct.)
MIN_TEXT_CHARS_PER_PAGE = 50
MIN_ALPHA_RATIO = 0.5

# Early-exit during scan: if the first N pages have an alpha_ratio clearly
# below the threshold, this is a scan or garbled cmap — we'll route to full
# OCR regardless, no need to walk the remaining pages.
SCAN_EARLY_EXIT_PAGES = 5
SCAN_EARLY_EXIT_RATIO = 0.3

# Per-page brokenness threshold — broken_ratio >= this → full OCR
# (skip the hybrid merge step since it would be mostly OCR anyway).
HYBRID_FULL_OCR_THRESHOLD = 0.5

# pdfplumber rendering scale.
PDFPLUMBER_RENDER_DPI = 200
PDFPLUMBER_SCALE = PDFPLUMBER_RENDER_DPI / 72  # PDF points → render pixels

# Real-table heuristic: pdfplumber's find_tables over-detects on decorative
# borders / columnar text. A candidate counts as a real table only if every
# check passes. The page must also reference it ("Таблица N" / "табл. N" /
# "Table N") — labelled tables are nearly universal in school textbooks.
TABLE_MIN_DATA_ROWS = 3      # rows with >=2 short filled cells, after trim
TABLE_MIN_COLS = 2
TABLE_BBOX_MARGIN = 5        # pt — bbox must fit within the page (with slack)
TABLE_MAX_CELL_LEN = 200     # cell content longer than this is a paragraph
_TABLE_REF_RE = re.compile(r"\b(?:табл|table)\w*\.?\s*\d", re.IGNORECASE)

# Formula detection signals (used by is_page_broken / page_formula_score).
_SUBSCRIPT_CHARS = set("₀₁₂₃₄₅₆₇₈₉")
_SUPERSCRIPT_CHARS = set("⁰¹²³⁴⁵⁶⁷⁸⁹")
_GREEK_CHARS = set("αβγδεζηθικλμνξοπρσςτυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")
_MATH_OP_CHARS = set("√∫∑∏∂∞±×÷≈≠≤≥∈∉∪∩⊂⊃")
_FRACTION_RE = re.compile(r"\b\d+/\d+\b")
_DIGIT_ONLY_NO_PUNCT_RE = re.compile(r"^[\d\s]{1,6}$")
_MATH_OPS_FOR_BROKEN = ("=", "+", "−", "×", "÷")


class SotaOCRError(RuntimeError):
    pass


# --- IO helpers --------------------------------------------------------------

def log_event(direction: str, message: str) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"{ts} {direction} {message}\n")


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.rename(path)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def auth_headers() -> dict[str, str]:
    key = os.environ.get("SOTAOCR_API_KEY")
    if not key:
        raise SotaOCRError("SOTAOCR_API_KEY is not set in environment / .env")
    return {"Authorization": f"Bearer {key}"}


# --- HTTP with retry ---------------------------------------------------------

def _http(method: str, url: str, **kwargs) -> requests.Response:
    """HTTP request with exponential backoff on 5xx and network errors.
    Caller still validates the final response status code."""
    last_err: Exception | None = None
    last_resp: requests.Response | None = None
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            resp = requests.request(method, url, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as err:
            last_err = err
            last_resp = None
        else:
            if resp.status_code not in RETRY_STATUSES:
                return resp
            last_err = None
            last_resp = resp
        if attempt < RETRY_MAX_ATTEMPTS:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            status = last_resp.status_code if last_resp is not None else "net-err"
            log_event(
                "##",
                f"retry {method} {url} attempt={attempt}/{RETRY_MAX_ATTEMPTS} "
                f"delay={delay:.0f}s status={status}",
            )
            time.sleep(delay)
    if last_resp is not None:
        return last_resp
    assert last_err is not None
    raise last_err


# --- API calls ---------------------------------------------------------------

def check_balance() -> dict:
    log_event("->", "GET /v1/balance")
    resp = _http("GET", f"{API_BASE}/balance", headers=auth_headers(), timeout=30)
    if not resp.ok:
        log_event("<-", f"{resp.status_code} balance ERROR body={resp.text[:200]}")
        resp.raise_for_status()
    data = resp.json()
    monthly = data.get("monthly_pages", {})
    log_event(
        "<-",
        f"{resp.status_code} balance remaining={data.get('remaining_pages')} "
        f"monthly_total={monthly.get('total')} monthly_allocated={monthly.get('allocated')}",
    )
    return data


def upload_document(pdf_path: Path, page_ranges: list[dict] | None = None) -> dict:
    size = pdf_path.stat().st_size
    tag = "extract"
    extra = ""
    data: dict[str, str] = {}
    if page_ranges:
        tag = "extract subset"
        extra = f" pages={[(r['start'], r['end']) for r in page_ranges]}"
        data["page_ranges"] = json.dumps(page_ranges)
    log_event("->", f"POST /v1/extract file={pdf_path.name} size={size}{extra}")
    with pdf_path.open("rb") as fh:
        resp = _http(
            "POST", f"{API_BASE}/extract",
            headers=auth_headers(),
            files={"file": (pdf_path.name, fh, "application/pdf")},
            data=data or None,
            timeout=UPLOAD_TIMEOUT_SECONDS,
        )
    if resp.status_code not in (200, 202):
        log_event(
            "<-",
            f"{resp.status_code} {tag} ERROR file={pdf_path.name} body={resp.text[:300]}",
        )
        raise SotaOCRError(f"Upload failed [{resp.status_code}]: {resp.text}")
    payload = resp.json()
    log_event(
        "<-",
        f"{resp.status_code} {tag} job={payload.get('id')} "
        f"pages={payload.get('page_count')} profile={payload.get('model_profile')} "
        f"upstream={payload.get('upstream_job_id')} status={payload.get('status')}",
    )
    return payload


def get_job(job_id: str) -> dict:
    log_event("->", f"GET /v1/jobs/{job_id}")
    resp = _http("GET", f"{API_BASE}/jobs/{job_id}", headers=auth_headers(), timeout=30)
    if not resp.ok:
        log_event("<-", f"{resp.status_code} job={job_id} ERROR body={resp.text[:200]}")
        resp.raise_for_status()
    data = resp.json()
    log_event(
        "<-",
        f"{resp.status_code} job={data.get('id')} status={data.get('status')} "
        f"pages={data.get('pages_completed')}/{data.get('page_count')}",
    )
    return data


def fetch_result(job_id: str, fmt: str) -> dict:
    log_event("->", f"GET /v1/jobs/{job_id}/result?format={fmt}")
    resp = _http(
        "GET", f"{API_BASE}/jobs/{job_id}/result",
        headers=auth_headers(),
        params={"format": fmt},
        timeout=120,
    )
    if resp.status_code == 202:
        log_event("<-", f"202 job={job_id} result_not_ready body={resp.text[:200]}")
        raise SotaOCRError(f"Result not ready: {resp.text}")
    if not resp.ok:
        log_event(
            "<-", f"{resp.status_code} job={job_id} result ERROR body={resp.text[:200]}"
        )
        resp.raise_for_status()
    payload = resp.json()
    content = payload.get("content")
    log_event(
        "<-",
        f"{resp.status_code} job={payload.get('job_id', job_id)} "
        f"format={payload.get('format')} pages={payload.get('page_count')} "
        f"content_len={len(content) if isinstance(content, str) else '-'}",
    )
    return payload


def extract_pages(payload: dict) -> list[dict]:
    """Unwrap pages[] from a sotaocr-shaped JSON. Robust to nested 'content'
    string variants. Raises SotaOCRError on malformed payloads."""
    inner = payload
    raw_content = payload.get("content")
    if isinstance(raw_content, str):
        try:
            inner = json.loads(raw_content)
        except json.JSONDecodeError as err:
            raise SotaOCRError(
                f"Could not parse content as JSON: {raw_content[:200]}"
            ) from err
    elif isinstance(payload.get("json"), dict):
        inner = payload["json"]
    pages = inner.get("pages")
    if not isinstance(pages, list):
        raise SotaOCRError("Could not locate pages[] in result payload")
    return pages


def wait_for_completion(job_id: str, page_count: int) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    last_done = -1
    while True:
        job = get_job(job_id)
        status = job.get("status")
        done = job.get("pages_completed", 0)
        total = job.get("page_count", page_count)
        if done != last_done:
            print(f"    {status}: {done}/{total} pages", flush=True)
            last_done = done
        if status == "completed":
            return job
        if status in {"failed", "error"}:
            raise SotaOCRError(f"Job {job_id} failed: {job}")
        if time.monotonic() > deadline:
            raise SotaOCRError(f"Job {job_id} timed out (last status: {status})")
        time.sleep(POLL_INTERVAL_SECONDS)


# --- pdfplumber: scan + brokenness -------------------------------------------

def page_formula_score(page_text: str) -> int:
    """Score for likely-formula content in pdfplumber output. Higher = more
    math signals pdfplumber struggles to represent (subscripts, Greek letters,
    math operators, broken-subscript line patterns)."""
    if not page_text:
        return 0
    score = 0
    for c in page_text:
        if c in _SUBSCRIPT_CHARS or c in _SUPERSCRIPT_CHARS:
            score += 5
        elif c in _GREEK_CHARS:
            score += 3
        elif c in _MATH_OP_CHARS:
            score += 2
    score += 3 * len(_FRACTION_RE.findall(page_text))
    # Broken-subscript: line with explicit math operator (=, +, ×, ÷)
    # immediately followed by a line that is just digits (no punctuation).
    lines = page_text.split("\n")
    for i in range(len(lines) - 1):
        cur = lines[i].strip()
        nxt = lines[i + 1].strip()
        if not (cur and nxt):
            continue
        if not _DIGIT_ONLY_NO_PUNCT_RE.match(nxt):
            continue
        if any(op in cur for op in _MATH_OPS_FOR_BROKEN):
            score += 5
    return score


def is_page_broken(page_text: str) -> tuple[bool, str]:
    """Decide if a single page's pdfplumber text is unusable (needs OCR).

    Returns (broken, reason) where reason is one of:
      - 'empty'       — not broken, just empty/image-only
      - 'garbled(…)'  — alpha_ratio is low (broken cmap)
      - 'formula(…)'  — formula signals present
      - 'ok'          — page is fine
    """
    text = page_text or ""
    stripped = text.strip()
    chars = len(stripped)
    if chars < 30:
        return False, "empty"
    alpha = sum(1 for c in stripped if c.isalpha())
    ratio = alpha / chars if chars else 0
    if ratio < MIN_ALPHA_RATIO:
        return True, f"garbled(alpha_ratio={ratio:.2f})"
    score = page_formula_score(text)
    if score >= 5:
        return True, f"formula(score={score})"
    return False, "ok"


def scan_pdf_text(pdf_path: Path) -> dict:
    """One pass over the PDF via pdfplumber. Combines:
      - aggregate text-density / readability stats
      - per-page brokenness flag

    Early-exit: after SCAN_EARLY_EXIT_PAGES pages, if alpha_ratio is well
    below the text threshold, stops the scan — the doc is clearly a scan or
    has a garbled cmap and will be routed to full OCR anyway.

    Returns a dict with: pages (total), scanned (pages actually examined),
    chars, alpha, avg_per_page, alpha_ratio, text_based, broken,
    early_exit (bool).
    """
    total = 0
    alpha_total = 0
    scanned = 0
    broken: list[tuple[int, str]] = []
    early_exit = False
    with pdfplumber.open(pdf_path) as pdf:
        all_pages = list(pdf.pages)
        page_total = len(all_pages)
        for page in all_pages:
            scanned += 1
            text = page.extract_text() or ""
            total += len(text)
            alpha_total += sum(1 for c in text if c.isalpha())
            is_broken, reason = is_page_broken(text)
            if is_broken:
                broken.append((page.page_number, reason))
            if scanned == SCAN_EARLY_EXIT_PAGES and total > 0:
                ratio_so_far = alpha_total / total
                if ratio_so_far < SCAN_EARLY_EXIT_RATIO:
                    early_exit = True
                    break
    avg = total / scanned if scanned else 0
    ratio = alpha_total / total if total else 0
    text_based = (
        not early_exit
        and avg >= MIN_TEXT_CHARS_PER_PAGE
        and ratio >= MIN_ALPHA_RATIO
    )
    return {
        "pages": page_total,
        "scanned": scanned,
        "chars": total,
        "alpha": alpha_total,
        "avg_per_page": avg,
        "alpha_ratio": ratio,
        "text_based": text_based,
        "broken": broken,
        "early_exit": early_exit,
    }


# --- pdfplumber: extraction --------------------------------------------------

def _md_cell(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|").strip()


def table_to_markdown(rows: list[list]) -> str:
    if not rows:
        return ""
    rows = [r for r in rows if any(str(c or "").strip() for c in r)]
    if not rows:
        return ""
    cols = max(len(r) for r in rows)
    rows = [list(r) + [None] * (cols - len(r)) for r in rows]
    header = "| " + " | ".join(_md_cell(c) for c in rows[0]) + " |"
    sep = "| " + " | ".join("---" for _ in range(cols)) + " |"
    body = ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join([header, sep] + body)


def _is_data_row(row: list) -> bool:
    """A real table row has >= 2 short, filled cells. Long cells (paragraphs
    that pdfplumber accidentally chunked into the grid when the bbox swallowed
    a decorative border / surrounding text) disqualify the row."""
    filled = [c for c in row if c and c.strip()]
    if len(filled) < 2:
        return False
    return all(len(c) <= TABLE_MAX_CELL_LEN for c in filled)


def _trim_table_rows(rows: list[list]) -> list[list]:
    """Trim leading and trailing rows that aren't data rows. Interior rows are
    preserved even if they fail the heuristic (long cells inside a real table
    are legitimate; over-extended bboxes only pollute the edges)."""
    first = next((i for i, r in enumerate(rows) if _is_data_row(r)), None)
    if first is None:
        return []
    last = len(rows) - next(
        i for i, r in enumerate(reversed(rows)) if _is_data_row(r)
    )
    return rows[first:last]


def _validated_tables(page, page_text: str) -> list[tuple[list, tuple]]:
    """Return only real tables on the page, sorted top-to-bottom.

    Filters out pdfplumber's frequent false positives (decorative borders,
    columnar layouts) by requiring all of:
      - the page references "Таблица N" / "табл. N" / "Table N";
      - bbox fits within the page (with small margin);
      - >= TABLE_MIN_COLS cols;
      - after trim, >= TABLE_MIN_DATA_ROWS data rows remain.
    Empty decorative columns (frequent in textbook tables) don't affect the
    decision because we count rows, not overall fill ratio.
    """
    if not _TABLE_REF_RE.search(page_text or ""):
        return []
    page_w, page_h = page.width, page.height
    out: list[tuple[float, list, tuple]] = []
    for tbl in page.find_tables():
        rows = tbl.extract()
        if not rows:
            continue
        cols = max(len(r) for r in rows)
        if cols < TABLE_MIN_COLS:
            continue
        x0, y0, x1, y1 = tbl.bbox
        m = TABLE_BBOX_MARGIN
        if x0 < -m or y0 < -m or x1 > page_w + m or y1 > page_h + m:
            continue
        rows = _trim_table_rows(rows)
        data_rows = sum(1 for r in rows if _is_data_row(r))
        if data_rows < TABLE_MIN_DATA_ROWS:
            continue
        out.append((y0, rows, tbl.bbox))
    out.sort(key=lambda x: x[0])
    return [(rows, bbox) for _, rows, bbox in out]


def pdfplumber_to_payload(pdf_path: Path, images_dir: Path) -> dict:
    """Build a sotaocr-shaped JSON payload from a text-based PDF using
    pdfplumber. Same structure as the API result, so render_document and
    enhance_images work uniformly for both paths.

    Side effect: pre-renders page-preview PNGs into images_dir for any page
    that contains image blocks.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    pages_out: list[dict] = []
    scale = PDFPLUMBER_SCALE
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_num = page.page_number
            page_w = page.width
            page_h = page.height
            full_text = (page.extract_text() or "").strip()
            tables = _validated_tables(page, full_text)
            blocks: list[dict] = []
            # Always emit the full page text. Carving stripes around table
            # bboxes is fragile because pdfplumber routinely extends a table's
            # bbox to include decorative ruled borders, swallowing surrounding
            # paragraphs. Real tables are added as extra blocks; cell content
            # appears twice (linearised in text + md-formatted in the table)
            # but no paragraph is ever lost.
            if full_text:
                blocks.append({
                    "id": "0",
                    "type": "text",
                    "bbox": [0, 0, int(page_w * scale), int(page_h * scale)],
                    "content": full_text,
                    "order": 1,
                })
            for rows, bbox_pdf in tables:
                md = table_to_markdown(rows)
                if md:
                    blocks.append({
                        "id": str(len(blocks)),
                        "type": "table",
                        "bbox": [int(c * scale) for c in bbox_pdf],
                        "content": md,
                        "order": len(blocks) + 1,
                    })
            for img in page.images:
                bbox_pdf = (img["x0"], img["top"], img["x1"], img["bottom"])
                blocks.append({
                    "id": str(len(blocks)),
                    "type": "image",
                    "bbox": [int(c * scale) for c in bbox_pdf],
                    "content": "",
                    "order": len(blocks) + 1,
                })
            has_images = any(b["type"] == "image" for b in blocks)
            if has_images:
                preview_path = images_dir / f"p{page_num}_preview.png"
                if not preview_path.exists():
                    page.to_image(resolution=PDFPLUMBER_RENDER_DPI).original.save(
                        preview_path, format="PNG"
                    )
            pages_out.append({
                "page_number": page_num,
                "status": "completed",
                "text": (page.extract_text() or "").strip(),
                "page_preview": {
                    "content_type": "image/png",
                    "coordinate_space": "local_render",
                    "width": int(page_w * scale),
                    "height": int(page_h * scale),
                },
                "layout_blocks": blocks,
            })
    inner = {
        "job_id": "pdfplumber-local",
        "page_count": len(pages_out),
        "pages": pages_out,
    }
    return {
        "job_id": "pdfplumber-local",
        "format": "json",
        "page_count": len(pages_out),
        "content": json.dumps(inner, ensure_ascii=False),
    }


# --- Document folder & info.txt ---------------------------------------------

def doc_folder_for(input_path: Path) -> Path:
    """Locate the working folder for a PDF.

    If the PDF is already inside a folder named after its stem (e.g. placed
    there by an external downloader that mirrors a Drive tree), respect that
    location. Otherwise default to docs/<stem>/.
    """
    if input_path.parent.name == input_path.stem:
        return input_path.parent
    return DOCS_DIR / input_path.stem


def ensure_pdf_in_folder(input_path: Path) -> Path:
    """Move the source PDF into its working folder. Idempotent."""
    folder = doc_folder_for(input_path)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / input_path.name
    if target.exists():
        if input_path.exists() and input_path.resolve() != target.resolve():
            log_event("##", f"pdf duplicate input={input_path} target={target}")
        return target
    if input_path.resolve() == target.resolve():
        return target
    shutil.move(str(input_path), str(target))
    log_event("##", f"move {input_path} -> {target}")
    return target


def find_md5_index() -> dict[str, Path]:
    """Scan docs/**/info.txt and return {md5_lower: doc_folder}.

    md5 is read from line 4 of info.txt (the convention written by
    write_info_txt). Folders without a valid info.txt are skipped.
    """
    index: dict[str, Path] = {}
    if not DOCS_DIR.is_dir():
        return index
    for info_path in DOCS_DIR.glob("**/info.txt"):
        try:
            lines = info_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        if len(lines) < 4:
            continue
        md5 = lines[3].strip().lower()
        if len(md5) == 32 and all(c in "0123456789abcdef" for c in md5):
            index[md5] = info_path.parent
    return index


def write_info_txt(folder: Path, pdf_path: Path,
                   authors: str | None, title: str | None, grade: str | None,
                   md5: str | None = None) -> Path:
    info_path = folder / "info.txt"
    if md5 is None:
        md5 = md5_of(pdf_path)
    if info_path.exists():
        existing = info_path.read_text(encoding="utf-8").splitlines()
        existing = (existing + ["", "", "", ""])[:4]
        a = authors if authors is not None else existing[0]
        t = title if title is not None else existing[1]
        g = grade if grade is not None else existing[2]
    else:
        a = authors or ""
        t = title or ""
        g = grade or ""
    write_atomic(info_path, f"{a}\n{t}\n{g}\n{md5}\n")
    return info_path


def render_md_from_json(json_path: Path, md_path: Path) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    pages = extract_pages(payload)
    md = render_document(pages)
    write_atomic(md_path, md)
    log_event(
        "##",
        f"render md from json src={json_path.name} dst={md_path.name} pages={len(pages)}",
    )


# --- Pipeline helpers --------------------------------------------------------

def run_ocr_job(pdf_path: Path, job_path: Path,
                page_ranges: list[dict] | None = None) -> dict:
    """Upload (or resume) and wait until the OCR job is completed.
    Returns the result payload."""
    if job_path.exists():
        job_meta = json.loads(job_path.read_text(encoding="utf-8"))
        print(f"  resuming job {job_meta['id']}")
        log_event("##", f"resume file={pdf_path.name} job={job_meta['id']}")
    else:
        if page_ranges:
            print(f"  uploading subset (pages {[r['start'] for r in page_ranges]})…")
        else:
            print("  uploading…")
        job_meta = upload_document(pdf_path, page_ranges)
        write_atomic(job_path, json.dumps(job_meta, indent=2))
        print(
            f"  job {job_meta['id']} accepted "
            f"({job_meta.get('page_count', '?')} pages, "
            f"profile={job_meta.get('model_profile', '?')})"
        )
    job = wait_for_completion(job_meta["id"], job_meta.get("page_count", 0))
    print(f"  fetching json for {job['id']}…")
    return fetch_result(job["id"], "json")


def _save_full_ocr(pdf_path: Path, json_path: Path,
                   md_path: Path, job_path: Path) -> None:
    payload = run_ocr_job(pdf_path, job_path)
    write_atomic(json_path, json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"  saved → {json_path}")
    pages = extract_pages(payload)
    md = render_document(pages)
    write_atomic(md_path, md)
    print(f"  saved → {md_path}")
    log_event("##", f"done file={pdf_path.name} mode=ocr pages={len(pages)}")


def _save_pdfplumber(pdf_path: Path, md_path: Path) -> None:
    folder = pdf_path.parent
    stem = pdf_path.stem
    json_path = folder / f"{stem}.json"
    job_path = folder / f"{stem}.job.json"
    images_dir = folder / "images"
    payload = pdfplumber_to_payload(pdf_path, images_dir)
    write_atomic(json_path, json.dumps(payload, ensure_ascii=False, indent=2))
    write_atomic(job_path, json.dumps(
        {"id": f"pdfplumber-local-{stem}", "page_count": payload["page_count"]},
        indent=2,
    ))
    pages = extract_pages(payload)
    md = render_document(pages)
    write_atomic(md_path, md)
    print(f"  saved → {json_path}")
    print(f"  saved → {md_path}")
    log_event(
        "##",
        f"done file={pdf_path.name} mode=pdfplumber pages={len(pages)} path={md_path}",
    )


def _save_hybrid(pdf_path: Path, json_path: Path, md_path: Path,
                 job_path: Path, broken_pages: list[int]) -> None:
    """OCR only the broken pages and merge with pdfplumber output for the rest."""
    folder = pdf_path.parent
    stem = pdf_path.stem
    images_dir = folder / "images"
    ocr_job_path = folder / f"{stem}.ocr-job.json"

    print(f"  building pdfplumber payload for all pages…")
    pdfplumber_payload = pdfplumber_to_payload(pdf_path, images_dir)
    by_num: dict[int, dict] = {
        p["page_number"]: p for p in extract_pages(pdfplumber_payload)
    }

    print(f"  OCR-ing {len(broken_pages)} broken page(s): {broken_pages}")
    page_ranges = [{"start": p, "end": p} for p in sorted(broken_pages)]
    ocr_payload = run_ocr_job(pdf_path, ocr_job_path, page_ranges=page_ranges)
    ocr_pages = extract_pages(ocr_payload)
    for ocr_page in ocr_pages:
        n = ocr_page.get("page_number")
        if n in by_num:
            by_num[n] = ocr_page

    # OCR bboxes are in a different coord space than pdfplumber renders.
    # Drop pdfplumber-rendered previews for OCR'd pages — enhance_images
    # will fetch the API preview when needed.
    for n in broken_pages:
        cached_preview = images_dir / f"p{n}_preview.png"
        if cached_preview.exists():
            cached_preview.unlink()

    merged_pages = sorted(by_num.values(), key=lambda p: p.get("page_number") or 0)
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
    write_atomic(job_path, json.dumps({
        "id": f"hybrid-{stem}",
        "page_count": len(merged_pages),
        "mode": "hybrid",
        "ocr_job_id": ocr_payload.get("job_id"),
        "ocr_pages": sorted(broken_pages),
    }, indent=2))
    md = render_document(merged_pages)
    write_atomic(md_path, md)
    print(f"  saved → {json_path}")
    print(f"  saved → {md_path}")
    log_event(
        "##",
        f"done file={pdf_path.name} mode=hybrid ocr={len(broken_pages)} "
        f"pdfplumber={len(merged_pages) - len(broken_pages)}",
    )


# --- Main per-file pipeline --------------------------------------------------

def process_pdf(input_path: Path, args: argparse.Namespace) -> Path | None:
    """Route a single PDF through the appropriate extraction path.

    Decision tree:
      .md exists           → skip
      .json exists         → re-render md from cached json
      --force-ocr / --has-formulas → full OCR
      not text-based       → full OCR
      --no-formulas        → pdfplumber
      no broken pages      → pdfplumber
      ≥50% broken          → full OCR
      <50% broken          → hybrid (OCR broken pages, pdfplumber rest)
    """
    print(f"\n=== {input_path.name} ===")
    log_event("##", f"start input={input_path}")
    pdf_path = ensure_pdf_in_folder(input_path)
    folder = pdf_path.parent
    stem = pdf_path.stem
    md_path = folder / f"{stem}.md"
    json_path = folder / f"{stem}.json"
    job_path = folder / f"{stem}.job.json"

    # Content-based dedup: if the same PDF (by md5) is already processed in
    # another folder, we don't want to spend OCR credits twice. Drop a marker
    # file pointing to the canonical copy.
    pdf_md5 = md5_of(pdf_path).lower()
    duplicate_marker = folder / "duplicate.txt"
    if not args.force:
        index = find_md5_index()
        canonical = index.get(pdf_md5)
        if canonical is not None and canonical.resolve() != folder.resolve():
            rel_canonical = canonical.relative_to(DOCS_DIR)
            print(f"  duplicate of {rel_canonical} (md5={pdf_md5[:8]}…)")
            log_event(
                "##",
                f"duplicate file={pdf_path.name} canonical={canonical} md5={pdf_md5}",
            )
            write_atomic(
                duplicate_marker,
                f"This PDF has the same content as: {rel_canonical}\n"
                f"md5: {pdf_md5}\n"
                f"To process anyway, delete this file and re-run with --force.\n",
            )
            return None
    if duplicate_marker.exists() and args.force:
        duplicate_marker.unlink()

    write_info_txt(folder, pdf_path, args.authors, args.title, args.grade, md5=pdf_md5)

    if md_path.exists() and not args.force:
        print(f"  already done → {md_path}")
        log_event("##", f"skip file={pdf_path.name} reason=already_done")
        return md_path
    if json_path.exists():
        print(f"  rendering md from cached json {json_path.name}")
        render_md_from_json(json_path, md_path)
        print(f"  saved → {md_path}")
        log_event("##", f"done file={pdf_path.name} mode=local_render")
        return md_path

    # 1. Forced overrides.
    if args.force_ocr or args.has_formulas:
        reason = "force_ocr" if args.force_ocr else "has_formulas"
        log_event("##", f"route file={pdf_path.name} mode=ocr reason={reason}")
        _save_full_ocr(pdf_path, json_path, md_path, job_path)
        return md_path

    # 2. Single pdfplumber pass: aggregate stats + per-page brokenness.
    print("  scanning pdf with pdfplumber…")
    s = scan_pdf_text(pdf_path)
    log_event(
        "##",
        f"scan file={pdf_path.name} pages={s['pages']} scanned={s['scanned']} "
        f"chars={s['chars']} avg={s['avg_per_page']:.1f} "
        f"alpha_ratio={s['alpha_ratio']:.3f} text_based={s['text_based']} "
        f"broken={len(s['broken'])} early_exit={s['early_exit']}",
    )
    early_note = " (early-exit: clearly scan)" if s["early_exit"] else ""
    print(
        f"  scan: {s['scanned']}/{s['pages']} pages, {s['chars']} chars, "
        f"avg={s['avg_per_page']:.1f}/page, alpha_ratio={s['alpha_ratio']:.2f} → "
        f"{'text-based' if s['text_based'] else 'image-based'}{early_note}"
    )

    if not s["text_based"]:
        log_event("##", f"route file={pdf_path.name} mode=ocr reason=image_based")
        _save_full_ocr(pdf_path, json_path, md_path, job_path)
        return md_path

    if args.no_formulas:
        log_event("##", f"route file={pdf_path.name} mode=pdfplumber reason=no_formulas_flag")
        _save_pdfplumber(pdf_path, md_path)
        return md_path

    # 3. Per-page brokenness routing.
    broken = s["broken"]
    total = s["pages"]
    broken_count = len(broken)
    broken_ratio = broken_count / total if total else 0
    sample_reasons = [r for _, r in broken[:5]]
    print(
        f"  brokenness: {broken_count}/{total} pages need OCR "
        f"({broken_ratio:.0%})"
        + (f"; e.g. {', '.join(sample_reasons)}" if sample_reasons else "")
    )

    if broken_count == 0:
        log_event("##", f"route file={pdf_path.name} mode=pdfplumber reason=nothing_broken")
        _save_pdfplumber(pdf_path, md_path)
    elif broken_ratio >= HYBRID_FULL_OCR_THRESHOLD:
        log_event(
            "##",
            f"route file={pdf_path.name} mode=ocr reason=mostly_broken({broken_ratio:.2f})",
        )
        print(f"  → full OCR (broken ratio ≥ {HYBRID_FULL_OCR_THRESHOLD:.0%})")
        _save_full_ocr(pdf_path, json_path, md_path, job_path)
    else:
        log_event("##", f"route file={pdf_path.name} mode=hybrid broken={broken_count}")
        print(f"  → hybrid (OCR {broken_count} pages, pdfplumber rest)")
        _save_hybrid(pdf_path, json_path, md_path, job_path, [n for n, _ in broken])
    return md_path


# --- CLI ---------------------------------------------------------------------

def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard pipeline flags to a parser. Used by both ocr_sotaocr
    and download_drive so flags don't drift between CLIs."""
    parser.add_argument("--no-balance-check", action="store_true",
                        help="Skip the GET /v1/balance pre-flight check.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Re-render .md from existing .json (no API).")
    parser.add_argument("--force-ocr", action="store_true",
                        help="Send to OCR even if PDF has selectable text.")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if .md already exists.")
    parser.add_argument("--no-formulas", action="store_true",
                        help="Subject has no formulas (history, languages). Use pdfplumber.")
    parser.add_argument("--has-formulas", action="store_true",
                        help="Document has formulas. Skip checks; use OCR directly.")
    parser.add_argument("--authors", default=None, help="Authors line for info.txt.")
    parser.add_argument("--title", default=None, help="Title line for info.txt.")
    parser.add_argument("--grade", default=None, help="Grade/class line for info.txt.")


def make_pipeline_args(**overrides) -> argparse.Namespace:
    """Build a default Namespace for process_pdf with optional overrides.

    Used by helper scripts (e.g. download_drive.py) so that adding a new
    pipeline flag in one place doesn't break callers that pass synthetic args.
    """
    defaults = dict(
        no_balance_check=False, rebuild=False,
        force_ocr=False, force=False,
        no_formulas=False, has_formulas=False,
        authors=None, title=None, grade=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract markdown from PDFs. Routes per-page between "
        "pdfplumber (text-based pages) and sotaocr.com OCR (formulas/scans). "
        "Each document gets a docs/<stem>/ folder with pdf, json, md, info.txt.",
    )
    parser.add_argument(
        "paths", nargs="+",
        help="One or more PDF files. Path may be either external or already "
        "moved into docs/<stem>/.",
    )
    add_pipeline_args(parser)
    return parser.parse_args()


def resolve_input(raw: str) -> Path | None:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if p.exists():
        return p
    moved = doc_folder_for(p) / p.name
    if moved.exists():
        return moved
    print(f"Skipping missing file: {p}", file=sys.stderr)
    return None


def main() -> int:
    args = parse_args()

    pdf_paths: list[Path] = []
    for raw in args.paths:
        resolved = resolve_input(raw)
        if resolved is not None:
            pdf_paths.append(resolved)
    if not pdf_paths:
        print("No valid input files.", file=sys.stderr)
        return 2

    log_event(
        "##",
        f"run start files={[p.name for p in pdf_paths]} "
        f"rebuild={args.rebuild} force_ocr={args.force_ocr} force={args.force} "
        f"no_formulas={args.no_formulas} has_formulas={args.has_formulas}",
    )

    if args.rebuild:
        failures: list[tuple[Path, Exception]] = []
        for pdf in pdf_paths:
            folder = doc_folder_for(pdf)
            json_path = folder / f"{pdf.stem}.json"
            md_path = folder / f"{pdf.stem}.md"
            if not json_path.exists():
                print(f"  {pdf.name}: no cached json at {json_path}", file=sys.stderr)
                failures.append((pdf, FileNotFoundError(json_path)))
                continue
            print(f"\n=== {pdf.name} (rebuild) ===")
            try:
                render_md_from_json(json_path, md_path)
                print(f"  saved → {md_path}")
            except (SotaOCRError, ValueError) as err:
                print(f"  FAILED: {err}", file=sys.stderr)
                log_event("##", f"fail file={pdf.name} err={err}")
                failures.append((pdf, err))
        log_event(
            "##",
            f"run end mode=rebuild ok={len(pdf_paths) - len(failures)} fail={len(failures)}",
        )
        return 1 if failures else 0

    if not args.no_balance_check:
        try:
            balance = check_balance()
            print(
                f"Balance: remaining_pages={balance.get('remaining_pages')}, "
                f"total_affordable_pages={balance.get('total_affordable_pages')}"
            )
        except (SotaOCRError, requests.RequestException) as err:
            print(f"Balance check failed: {err}", file=sys.stderr)
            log_event("##", f"run abort reason=balance_check_failed err={err}")
            return 1

    failures: list[tuple[Path, Exception]] = []
    for pdf in pdf_paths:
        try:
            process_pdf(pdf, args)
        except (SotaOCRError, requests.RequestException, OSError) as err:
            print(f"  FAILED: {err}", file=sys.stderr)
            log_event("##", f"fail file={pdf.name} err={err}")
            failures.append((pdf, err))
    log_event("##", f"run end ok={len(pdf_paths) - len(failures)} fail={len(failures)}")

    if failures:
        print(f"\n{len(failures)} file(s) failed:", file=sys.stderr)
        for path, err in failures:
            print(f"  - {path.name}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
