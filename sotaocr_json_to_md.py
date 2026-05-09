"""
Convert sotaocr.com JSON result into a structured markdown file.

The /v1/jobs/{id}/result?format=markdown endpoint emits a flat dump.
The JSON form carries layout_blocks with type/order/bbox, which lets us
render headers, paragraphs, lists and tables properly.

Usage:
    python3 sotaocr_json_to_md.py sotaocr/test-h.json [-o sotaocr/test-h.improved.md]
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

# Block types we explicitly handle. Anything else is rendered as a paragraph
# with a hint comment so we can extend the converter when we hit new types.
SKIP_TYPES = {"number"}  # page-number gutter blocks
HEADER_TYPES = {"header", "title", "doc_title", "section_title"}
SUBHEADER_TYPES = {"paragraph_title", "subtitle", "subheader"}
PARAGRAPH_TYPES = {"text", "paragraph", "plain text"}
LIST_TYPES = {"list", "bullet_list", "ordered_list"}
TABLE_TYPES = {"table"}
FIGURE_TYPES = {"figure", "image"}
CAPTION_TYPES = {"caption", "figure_caption", "table_caption"}
FORMULA_TYPES = {"formula", "equation", "math"}


def block_text(block: dict) -> str:
    return (block.get("content") or block.get("text") or "").strip()


def render_table(block: dict) -> str:
    # Some pipelines put HTML / markdown table directly into `content`.
    # If we ever see a richer structure (rows/cells), branch here.
    raw = block_text(block)
    if not raw:
        return ""
    if raw.lstrip().startswith("|") or "<table" in raw.lower():
        return raw
    return f"```\n{raw}\n```"


def render_block(block: dict, image_lookup=None, page_number=None) -> str | None:
    btype = (block.get("type") or block.get("label") or "").lower()
    text = block_text(block)
    if btype in SKIP_TYPES:
        return None
    if btype in FIGURE_TYPES:
        # An image block may have empty `text`; we still want to emit it when
        # an image_lookup provides a saved crop and/or caption.
        entry = None
        if image_lookup is not None:
            entry = image_lookup(page_number, block.get("id"))
        if entry:
            file = entry.get("file") or ""
            caption = (entry.get("caption") or "").strip()
            label = (entry.get("label") or "").strip()
            bits: list[str] = []
            if file:
                bits.append(f"![]({file})")
            if caption:
                if bits:
                    bits.append("")
                bits.append(f"**Содержимое рисунка:** {caption}")
            elif label:
                if bits:
                    bits.append("")
                bits.append(f"_{label}_")
            if bits:
                return "\n".join(bits)
        return f"<!-- figure (bbox={block.get('bbox')}) -->"
    if not text:
        return None
    if btype in HEADER_TYPES:
        return f"# {text}"
    if btype in SUBHEADER_TYPES:
        return f"## {text}"
    if btype in PARAGRAPH_TYPES:
        return text
    if btype in LIST_TYPES:
        return text  # already line-formatted in most pipelines
    if btype in TABLE_TYPES:
        return render_table(block)
    if btype in CAPTION_TYPES:
        return f"_{text}_"
    if btype in FORMULA_TYPES:
        if "\n" in text or len(text) > 60:
            return f"$$\n{text}\n$$"
        return f"${text}$"
    return f"<!-- block type='{btype}' -->\n{text}"


def order_key(block: dict, fallback: int) -> tuple:
    order = block.get("order")
    if order is not None:
        return (0, int(order), fallback)
    bbox = block.get("bbox") or [0, 0, 0, 0]
    # left-to-right, top-to-bottom for blocks without explicit order
    return (1, int(bbox[1]), int(bbox[0]), fallback)


def render_page(page: dict, image_lookup=None) -> str:
    page_number = page.get("page_number")
    blocks = page.get("layout_blocks") or page.get("chunks") or []
    indexed = list(enumerate(blocks))
    indexed.sort(key=lambda iv: order_key(iv[1], iv[0]))
    parts: list[str] = []
    for _, block in indexed:
        rendered = render_block(block, image_lookup=image_lookup, page_number=page_number)
        if rendered:
            parts.append(rendered)
    if not parts:
        plain = (page.get("text") or "").strip()
        if plain:
            parts.append(plain)
    return "\n\n".join(parts)


def render_document(pages: Iterable[dict], image_lookup=None) -> str:
    """Render markdown.

    image_lookup: optional callable (page_number, block_id) -> dict | None
        with keys 'file' (relative path) and 'caption' (description).
        When provided, image blocks are rendered with a Markdown image link
        and a "Содержимое рисунка:" caption. When None, image blocks become
        a hidden HTML comment (current behaviour).
    """
    sections: list[str] = []
    for page in pages:
        n = page.get("page_number")
        body = render_page(page, image_lookup=image_lookup)
        header = f"<!-- page {n} -->" if n is not None else "<!-- page -->"
        sections.append(f"{header}\n\n{body}".rstrip())
    return "\n\n---\n\n".join(sections) + "\n"


def load_pages(json_path: Path) -> list[dict]:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    inner = raw
    if isinstance(raw.get("content"), str):
        inner = json.loads(raw["content"])
    elif isinstance(raw.get("json"), dict):
        inner = raw["json"]
    pages = inner.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"Could not locate pages[] in {json_path}")
    return pages


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json_path", help="Path to sotaocr JSON result.")
    p.add_argument(
        "-o",
        "--output",
        help="Output .md path. Defaults to <input>.improved.md alongside the json.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.json_path).expanduser().resolve()
    if not src.exists():
        print(f"Not found: {src}", file=sys.stderr)
        return 2
    if args.output:
        dst = Path(args.output).expanduser().resolve()
    else:
        dst = src.with_suffix("").with_suffix(".improved.md")
    pages = load_pages(src)
    md = render_document(pages)
    dst.write_text(md, encoding="utf-8")
    print(f"saved {dst} ({dst.stat().st_size} bytes, {len(pages)} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
