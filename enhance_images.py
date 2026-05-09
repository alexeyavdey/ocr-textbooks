"""
Caption image regions in OCR'd documents using a vision LLM (OpenRouter), and
re-render markdown with image references + descriptions.

Pipeline (per document folder docs/<stem>/):
  1. Read <stem>.json (full OCR result) and <stem>.job.json (for job_id).
  2. For each page that has image blocks:
       a. Fetch /v1/jobs/<id>/pages/<n>/preview from sotaocr.com (cached).
       b. Crop each image block by its bbox → save as images/p<N>_b<id>.png.
       c. Send crop to qwen3-vl via OpenRouter, get a Russian transcription.
       d. Cache caption by md5(crop) so re-runs reuse results.
  3. Re-render <stem>.md, replacing `<!-- figure -->` with markdown image
     references and "**Содержимое рисунка:** …" captions.

Usage:
    python3 enhance_images.py docs/<stem>/                         # whole doc
    python3 enhance_images.py docs/<stem>/<stem>.pdf               # same
    python3 enhance_images.py docs/<stem>/ --pages 105,106         # subset
    python3 enhance_images.py docs/<stem>/ --rebuild               # only re-render md
"""
import argparse
import base64
import hashlib
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

import ocr_sotaocr
from sotaocr_json_to_md import render_document

load_dotenv()

PROJECT_DIR = Path(__file__).parent
DOCS_DIR = PROJECT_DIR / "docs"

# First model is tried first; subsequent ones act as fallbacks if the call
# fails (timeout, server error, model unavailable, etc.).
VISION_MODELS = (
    "qwen/qwen3.6-35b-a3b",
    "qwen/qwen3-vl-235b-a22b-instruct",
)

TRIAGE_MODEL = "openrouter/auto"

TRIAGE_PROMPT_TEMPLATE = """Ниже текст одной страницы учебника. На странице есть несколько картинок (известны их id, размер и положение в координатах страницы).

Для каждой картинки реши, нужна ли её детальная визуальная транскрипция для понимания содержания страницы.

Ответ YES, если картинка несёт информацию, которую нельзя восстановить из текста:
- схема, диаграмма, чертёж устройства, механизм, лабораторная установка
- график, гистограмма, диаграмма потоков
- векторная диаграмма, геометрическая фигура с обозначениями
- формула или таблица, набранная как изображение
- анатомическая/биологическая схема, карта

Ответ NO, если картинка декоративна, иллюстрирует общеизвестный объект или прямо описанный в тексте предмет:
- фотография или портрет известной личности
- фото здания, памятника, природного объекта, города
- художественная иллюстрация, обложка
- декоративная виньетка, эмблема, орнамент

Текст страницы:
---
{page_text}
---

Картинки на странице:
{image_list}

Label — на русском языке, 2-5 слов, кратко описывает что вероятно изображено.

Ответь СТРОГО валидным JSON, без markdown-обвёртки и без пояснений:
{{"<id>": {{"needed": "YES" | "NO", "label": "<2-5 русских слов>"}}, ...}}"""

PROMPT = """Опиши подробно содержимое этого фрагмента из учебника.
Транскрибируй точно все видимые элементы:
- Числа, единицы измерения, подписи и метки
- Стрелки, векторы, их направления и подписи (F₁, R, и т.п.)
- Буквенные обозначения (точки, оси, переменные)
- Геометрические фигуры (углы, отрезки, окружности, треугольники)
- Графики и диаграммы — описать характер кривых, оси, шкалы
- Любой текст внутри изображения

Выдай чистое описание без вступлений ("На рисунке…") и заключений.
Если несколько объектов — перечисли каждый с пояснением.
Используй LaTeX для математических обозначений: $F_1$, $\\vec{R}$, $\\frac{m}{V}$."""

IMAGE_LIKE_TYPES = {"image", "figure", "chart", "diagram", "picture"}

# Bump when the cache schema changes incompatibly.
CACHE_VERSION = 1


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _render_local_preview(dest: Path, page_num: int) -> bytes:
    """Render a page preview from the local PDF via pdfplumber."""
    doc_folder = dest.parent.parent
    stem = doc_folder.name
    pdf_path = doc_folder / f"{stem}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Cannot render preview: PDF missing at {pdf_path}")
    import pdfplumber
    ocr_sotaocr.log_event(
        "##", f"local-render preview pdf={pdf_path.name} page={page_num}"
    )
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        page.to_image(
            resolution=ocr_sotaocr.PDFPLUMBER_RENDER_DPI
        ).original.save(dest, format="PNG")
    return dest.read_bytes()


def _api_fetch_preview(job_id: str, page_num: int, dest: Path) -> bytes:
    url = f"{ocr_sotaocr.API_BASE}/jobs/{job_id}/pages/{page_num}/preview"
    ocr_sotaocr.log_event(
        "->", f"GET /v1/jobs/{job_id}/pages/{page_num}/preview"
    )
    r = requests.get(url, headers=ocr_sotaocr.auth_headers(), timeout=120)
    r.raise_for_status()
    ocr_sotaocr.log_event(
        "<-",
        f"{r.status_code} preview job={job_id} page={page_num} bytes={len(r.content)}",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return r.content


def fetch_preview(job_meta, page_num: int, dest: Path) -> bytes:
    """Get a page preview. Accepts a job_meta dict or a plain job_id string.

    Routing:
      - cached file on disk → use it
      - mode=hybrid → API for ocr_pages, local render for pdfplumber pages
      - id starts with 'pdfplumber-local' or 'hybrid' → local render
      - else → API
    """
    if dest.exists():
        return dest.read_bytes()
    meta = job_meta if isinstance(job_meta, dict) else {"id": str(job_meta)}
    if meta.get("mode") == "hybrid":
        ocr_pages = set(meta.get("ocr_pages") or [])
        if page_num in ocr_pages:
            ocr_job_id = meta.get("ocr_job_id") or meta.get("id", "")
            return _api_fetch_preview(ocr_job_id, page_num, dest)
        return _render_local_preview(dest, page_num)
    job_id = str(meta.get("id", ""))
    if job_id.startswith(("pdfplumber-local", "hybrid")):
        return _render_local_preview(dest, page_num)
    return _api_fetch_preview(job_id, page_num, dest)


def page_text_for_triage(page: dict) -> str:
    """Concatenate textual blocks of a page in reading order for triage."""
    blocks = page.get("layout_blocks") or page.get("chunks") or []
    keep_types = {
        "text", "paragraph", "header", "title", "doc_title", "section_title",
        "paragraph_title", "subtitle", "subheader", "caption",
        "figure_title", "table_caption", "list", "bullet_list",
        "ordered_list", "table", "formula", "equation", "math",
    }
    items = [
        b for b in blocks
        if (b.get("type") or b.get("label") or "").lower() in keep_types
    ]
    items.sort(key=lambda b: b.get("order") or 0)
    parts = []
    for b in items:
        txt = (b.get("content") or b.get("text") or "").strip()
        if txt:
            parts.append(txt)
    return "\n".join(parts)


def triage_images(
    client: OpenAI, page_text: str, image_blocks: list[dict]
) -> dict:
    """Returns {block_id: {"needed": "YES"|"NO", "label": "..."}} via openrouter/auto."""
    if not image_blocks:
        return {}
    img_lines = []
    for b in image_blocks:
        bid = str(b.get("id"))
        bbox = b.get("bbox") or [0, 0, 0, 0]
        w = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
        h = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
        img_lines.append(
            f"  - id={bid}, размер {w}x{h}, bbox=[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"
        )
    prompt = TRIAGE_PROMPT_TEMPLATE.format(
        page_text=page_text or "(текстовых блоков не распознано)",
        image_list="\n".join(img_lines),
    )
    resp = client.chat.completions.create(
        model=TRIAGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    if not resp.choices:
        raise RuntimeError(f"triage returned no choices: {resp.model_dump()}")
    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"triage returned non-JSON: {raw[:200]}") from err
    out = {}
    for bid, val in (parsed.items() if isinstance(parsed, dict) else []):
        if not isinstance(val, dict):
            continue
        out[str(bid)] = {
            "needed": str(val.get("needed", "YES")).upper(),
            "label": str(val.get("label", "")).strip(),
        }
    return out


def caption_image(client: OpenAI, png_bytes: bytes) -> tuple[str, str]:
    """Try each VISION_MODELS in order; return (caption, model_id_used)."""
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ],
        }
    ]
    last_err: Exception | None = None
    for model in VISION_MODELS:
        try:
            resp = client.chat.completions.create(model=model, messages=messages)
        except Exception as err:
            last_err = err
            print(f"      model {model} failed: {err}", file=sys.stderr)
            continue
        if not resp.choices:
            last_err = RuntimeError(f"no choices: {resp.model_dump()}")
            print(f"      model {model} returned no choices", file=sys.stderr)
            continue
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            last_err = RuntimeError("empty caption")
            print(f"      model {model} returned empty caption", file=sys.stderr)
            continue
        return text, model
    raise RuntimeError(f"all vision models failed; last error: {last_err}")


def enhance_doc(doc_folder: Path, args: argparse.Namespace) -> None:
    stem = doc_folder.name
    json_path = doc_folder / f"{stem}.json"
    job_path = doc_folder / f"{stem}.job.json"
    md_path = doc_folder / f"{stem}.md"
    images_dir = doc_folder / "images"
    cache_path = doc_folder / "images.cache.json"

    if not json_path.exists():
        raise FileNotFoundError(f"OCR JSON not found: {json_path}")
    if not job_path.exists():
        raise FileNotFoundError(f"Job metadata not found: {job_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    job_meta = json.loads(job_path.read_text(encoding="utf-8"))
    pages = ocr_sotaocr.extract_pages(payload)

    cache: dict = {}
    if cache_path.exists():
        try:
            raw_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            print(f"  cache unreadable ({err}), starting fresh", file=sys.stderr)
            raw_cache = {}
        # Legacy caches (pre-versioning) are treated as compatible — schema
        # was extended additively. Newer/unknown versions are discarded.
        cache_ver = raw_cache.get("_version", CACHE_VERSION)
        if cache_ver == CACHE_VERSION:
            cache = raw_cache
        else:
            print(
                f"  cache version mismatch ({cache_ver} != {CACHE_VERSION}), discarding",
                file=sys.stderr,
            )
    items_by_block: dict = cache.get("by_block", {})
    captions_by_hash: dict = cache.get("by_hash", {})

    client: OpenAI | None = None
    if not args.rebuild:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in env / .env")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    target_pages: set[int] | None = None
    if args.pages:
        target_pages = {int(p) for p in args.pages.split(",")}

    images_dir.mkdir(parents=True, exist_ok=True)

    for page in pages:
        page_num = page.get("page_number")
        if target_pages and page_num not in target_pages:
            continue
        blocks = page.get("layout_blocks") or page.get("chunks") or []
        image_blocks = [
            b
            for b in blocks
            if (b.get("type") or b.get("label") or "").lower() in IMAGE_LIKE_TYPES
        ]
        if not image_blocks:
            continue
        print(f"  page {page_num}: {len(image_blocks)} image block(s)")

        # Step 1: triage decisions (cached per block in items_by_block).
        # If any block on this page has no cached decision → call triage once.
        triage_decisions: dict = {}
        if not args.rebuild:
            need_triage = any(
                items_by_block.get(f"{page_num}:{b.get('id')}", {}).get("needed") is None
                for b in image_blocks
            )
            if need_triage:
                page_text = page_text_for_triage(page)
                try:
                    triage_decisions = triage_images(client, page_text, image_blocks)
                    yes = sum(1 for v in triage_decisions.values() if v.get("needed") == "YES")
                    print(
                        f"    triage: {yes}/{len(triage_decisions)} need vision "
                        f"({TRIAGE_MODEL})"
                    )
                except Exception as err:
                    print(f"    triage FAILED: {err}", file=sys.stderr)
                    triage_decisions = {
                        str(b.get("id")): {"needed": "YES", "label": ""}
                        for b in image_blocks
                    }

        preview_path = images_dir / f"p{page_num}_preview.png"
        try:
            preview_bytes = fetch_preview(job_meta, page_num, preview_path)
        except Exception as err:
            print(f"    preview FAILED: {err}", file=sys.stderr)
            continue

        try:
            preview = Image.open(io.BytesIO(preview_bytes)).convert("RGB")
        except Exception as err:
            print(f"    preview parse FAILED: {err}", file=sys.stderr)
            continue

        for block in image_blocks:
            bid = str(block.get("id"))
            bbox = block.get("bbox") or []
            if len(bbox) < 4:
                continue
            x0, y0, x1, y1 = (int(v) for v in bbox[:4])
            crop_path = images_dir / f"p{page_num}_b{bid}.png"

            try:
                crop = preview.crop((x0, y0, x1, y1))
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                crop_bytes = buf.getvalue()
                crop_path.write_bytes(crop_bytes)
            except Exception as err:
                print(f"    crop FAILED p{page_num}/b{bid}: {err}", file=sys.stderr)
                continue

            digest = md5_bytes(crop_bytes)
            key = f"{page_num}:{bid}"
            existing = items_by_block.get(key, {})

            decision = triage_decisions.get(bid) or {
                "needed": existing.get("needed", "YES"),
                "label": existing.get("label", ""),
            }
            needed = decision.get("needed", "YES")
            label = decision.get("label", "")

            if args.rebuild:
                caption = existing.get("caption", "")
                src = "rebuild"
            elif needed == "NO":
                caption = ""
                src = f"skip/{label or '?'}"
            elif digest in captions_by_hash:
                caption = captions_by_hash[digest]["caption"]
                src = "cache"
            else:
                try:
                    caption, model_used = caption_image(client, crop_bytes)
                except Exception as err:
                    print(
                        f"    caption FAILED p{page_num}/b{bid}: {err}",
                        file=sys.stderr,
                    )
                    continue
                captions_by_hash[digest] = {
                    "caption": caption,
                    "model": model_used,
                    "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
                src = f"new/{model_used.split('/')[-1]}"

            items_by_block[key] = {
                "file": f"images/p{page_num}_b{bid}.png",
                "hash": digest,
                "bbox": bbox,
                "needed": needed,
                "label": label,
                "caption": caption,
            }
            preview_text = (caption or label).replace("\n", " ")[:90]
            print(f"    p{page_num}/b{bid} [{src}] needed={needed}: {preview_text}")

    ocr_sotaocr.write_atomic(
        cache_path,
        json.dumps(
            {
                "_version": CACHE_VERSION,
                "by_block": items_by_block,
                "by_hash": captions_by_hash,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    print(f"  cache → {cache_path}")

    def lookup(p, bid):
        return items_by_block.get(f"{p}:{bid}")

    md = render_document(pages, image_lookup=lookup)
    ocr_sotaocr.write_atomic(md_path, md)
    print(f"  md   → {md_path}")


def resolve_folder(arg: str) -> Path:
    p = Path(arg).expanduser().resolve()
    if p.is_dir():
        return p
    if p.suffix.lower() == ".pdf":
        if p.parent.name == p.stem:
            return p.parent
        candidate = DOCS_DIR / p.stem
        if candidate.is_dir():
            return candidate
    raise SystemExit(f"Cannot resolve doc folder from: {arg}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("path", help="docs/<stem>/ folder, or path to <stem>.pdf")
    p.add_argument("--pages", default=None,
                   help="Comma-separated page numbers to process (default: all).")
    p.add_argument("--rebuild", action="store_true",
                   help="Re-render md from cache only; no API/LLM calls.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    folder = resolve_folder(args.path)
    print(f"=== {folder.name} ===")
    enhance_doc(folder, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
