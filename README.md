# ocr-textbooks

Pipeline для извлечения markdown из PDF-учебников. Для каждой страницы решает, идти через **pdfplumber** (бесплатно, нативное извлечение текста) или через **sotaocr.com OCR** (платно, для формул и сканов). Опционально описывает картинки через vision LLM на OpenRouter.

Per-page routing экономит баланс: учебник физики на 250 страниц с 30 формульными страницами обойдётся в ~30 OCR-кредитов вместо 250.

## Скрипты

| Скрипт | Назначение |
|---|---|
| `ocr_sotaocr.py` | Главный pipeline: pdfplumber → per-page check → pdfplumber / OCR / hybrid |
| `sotaocr_json_to_md.py` | JSON → Markdown с поддержкой image-блоков |
| `enhance_images.py` | Triage картинок + caption через OpenRouter (qwen3.6 → qwen3-vl fallback) |
| `download_drive.py` | Скачивает PDF из Google Drive, зеркалит структуру в `docs/` |
| `inventory_drive.py` | Inventory Drive-папки до скачивания (опционально с подсчётом страниц) |

## Routing

```
PDF
 │
 ▼  scan_pdf_text  (pdfplumber, бесплатно)
 │
 ├─ alpha_ratio < 0.5 (скан / битый cmap) ──────► full OCR
 │
 └─ text_based ──┬─ --has-formulas / --force-ocr ► full OCR
                 ├─ --no-formulas ──────────────► pure pdfplumber
                 │
                 └─ default: per-page brokenness
                              │
                              ├─ 0 broken ────► pure pdfplumber
                              ├─ ≥50% broken ─► full OCR
                              └─ <50% broken ─► HYBRID
                                                (OCR только сломанных,
                                                 pdfplumber для остальных)
```

**Страница «сломана», если:**
- `alpha_ratio < 0.5` — мало букв относительно общего объёма (битый cmap)
- `formula_score >= 5` — есть subscripts/superscripts/греческие/мат-операторы или паттерн «строка с `=` → строка только из цифр» (разломанный subscript)

## Setup

```bash
git clone https://github.com/alexeyavdey/ocr-textbooks.git
cd ocr-textbooks
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Секреты (`.env`)

```bash
cp .env.example .env
```
Заполнить:
```
SOTAOCR_API_KEY=sk-pdl-...        # https://sotaocr.com
OPENROUTER_API_KEY=sk-or-v1-...   # https://openrouter.ai/keys (нужен только для enhance_images)
```

### Google Drive OAuth (для download_drive / inventory_drive)

1. https://console.cloud.google.com/ → создать/выбрать проект
2. APIs & Services → Library → Google Drive API → **Enable**
3. OAuth consent screen → External → добавить свой email в **Test users** (или Publish app)
4. Credentials → Create OAuth client ID → **Desktop app** → Download JSON
5. Положить как `credentials.json` в корень проекта

Первый запуск откроет браузер → Allow → токен сохранится в `token.json`. Дальше — без браузера.

## Usage

```bash
# Один PDF
python3 ocr_sotaocr.py path/to/file.pdf

# С метаданными для info.txt
python3 ocr_sotaocr.py --authors "Перышкин А.В." --title "Физика 7" --grade "7" file.pdf

# Гуманитарные предметы — пропустить per-page check
python3 ocr_sotaocr.py --no-formulas history.pdf

# Заведомо формульные — сразу full OCR
python3 ocr_sotaocr.py --has-formulas algebra.pdf

# Скачать из Drive + прогон
python3 download_drive.py "https://drive.google.com/drive/u/0/folders/<ID>"

# Inventory перед скачиванием (без затрат)
python3 inventory_drive.py "<URL>" --top 20

# Inventory с подсчётом страниц (скачивает PDF для подсчёта)
python3 inventory_drive.py "<URL>" --count-pages

# Описать картинки в уже распознанном учебнике
python3 enhance_images.py docs/<stem>/
```

### Полезные флаги

| Флаг | Эффект |
|---|---|
| `--force` | Переобработать даже если `.md` уже существует |
| `--rebuild` | Пересобрать `.md` из кэшированного `.json` (без API) |
| `--no-balance-check` | Пропустить pre-flight `/balance` запрос |
| `--limit N` | (только download_drive) Обработать первые N PDF |
| `--dry-run` | (только download_drive) Только показать, что было бы скачано |

## Output

```
docs/<stem>/
├── <stem>.pdf            # перенесённый источник
├── <stem>.md             # финальный markdown
├── <stem>.json           # sotaocr-shaped result (используется enhance_images)
├── <stem>.job.json       # job metadata (resumability)
├── <stem>.ocr-job.json   # (только в hybrid) sub-job для broken страниц
├── info.txt              # 4 строки: authors, title, grade, md5(pdf)
├── images/               # (после enhance_images) crops + page previews
└── images.cache.json     # (после enhance_images) кэш описаний по md5
```

В `docs/api.log` пишутся все API-вызовы:
```
2026-05-09T07:14:24+03:00 -> POST /v1/extract file=test.pdf size=21075
2026-05-09T07:14:25+03:00 <- 202 extract job=job_xxx pages=1 profile=fast
2026-05-09T07:14:28+03:00 ## route file=test.pdf mode=hybrid broken=15
```

## Resumability

Прерывание по ^C безопасно — все записи через atomic rename. При повторном запуске:
- `.md` существует → skip
- `.json` существует → md рендерится локально (без API)
- `.job.json` существует → polling возобновляется по тому же `job_id`
- `images.cache.json` → дедупликация описаний по md5 крошек, версионируется

## Стоимость

Для одного учебника ~150 стр.:

| Сценарий | Стоимость |
|---|---|
| Гуманитарный, текстовый | ~0 (pdfplumber, бесплатно) |
| Физика/математика, hybrid (15-30% broken) | 20-50 OCR-кредитов |
| Скан без OCR-слоя | ~150 OCR-кредитов (full OCR) |
| Hybrid + enhance_images (~$0.001 за картинку) | + $0.05-0.2 на учебник |

## Зависимости

- Python 3.10+
- `pdfplumber>=0.11`
- `requests>=2.31`
- `openai>=1.40` (для enhance_images / probe judge)
- `python-dotenv>=1.0`
- `google-api-python-client>=2.130`, `google-auth-oauthlib>=1.2` (для Drive)
