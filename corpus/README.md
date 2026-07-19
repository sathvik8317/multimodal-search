# corpus/

Demo corpus for `mmsearch ingest corpus/`. Four subdirectories, one per
modality the ingester dispatches on:

| Directory | Modality | File types | Ingester |
|---|---|---|---|
| `specs/` | `pdf_page` | `.pdf` | `ingest.documents.ingest_pdf` |
| `docs/` | `diagram` | `.png`/`.jpg`/`.jpeg`/`.gif`/`.bmp`/`.webp` | `ingest.documents.ingest_diagram` |
| `data/` | `table` | `.csv` | `ingest.tables.ingest_table` (truncated to the first `MAX_TABLE_ROWS` rows, default 200, see `config.py`) |
| `src/` | `code` | `.py` | `ingest.code.ingest_code_file` |

## What's committed vs. gitignored

Three files are gitignored and **must be sourced separately** before a full
ingest reproduces the real demo index. Everything else in this directory
is committed as-is.

| File | Status | Why |
|---|---|---|
| `specs/2106.09685v2.pdf` (LoRA) | **gitignored** | arXiv's default "arXiv.org perpetual, non-exclusive license": grants arXiv the right to host it, not third parties the right to redistribute the file. Source: <https://arxiv.org/abs/2106.09685> |
| `specs/NIPS-2017-attention-is-all-you-need-Paper.pdf` (Attention Is All You Need) | **gitignored** | Same license situation. Source: <https://arxiv.org/abs/1706.03762> |
| `data/Bank_Marketing_Dataset.csv` | **gitignored** | Not a copyright issue: 3.5 MB / 45,211 rows is disproportionate to ship for a demo that only ever embeds the first 200 rows (`MAX_TABLE_ROWS`) once truncated. Original dataset source not confirmed (column headers in this copy are generic `V1`..`V16` and `Class`, not the named columns of the commonly-cited UCI Bank Marketing dataset, so don't assume it's that one without checking). Re-source your own copy of a bank-marketing-style dataset, or regenerate/trim this one, before ingesting. |
| `specs/2407.01449v6.pdf` (ColPali) | committed | CC0 1.0 Universal (public domain): unambiguously safe to redistribute. Source: <https://arxiv.org/abs/2407.01449> |
| `specs/Fine-Tune_Your_Own_LLM_for_Free_on_a_Kaggle_GPU_in_30_Minutes.pdf` | committed | Own content, no third-party copyright issue. |
| `docs/*.png` (all 3 diagrams) | committed | |
| `src/*.py` (all code files) | committed | |
| `data/car_prediction_data.csv`, `data/automobile_dataset.csv`, `data/ecommerce_sales_analytics_5000.csv` | committed | Small enough (≤540 KB) to ship as-is. |

## Reproducing the full demo ingest

1. Source the three gitignored files above and place them at the exact
   paths listed (`corpus/specs/2106.09685v2.pdf`,
   `corpus/specs/NIPS-2017-attention-is-all-you-need-Paper.pdf`,
   `corpus/data/Bank_Marketing_Dataset.csv`).
2. Ensure `COHERE_API_KEY` is set in `.env` at the repo root.
3. Run:

   ```
   python -m mmsearch.ingest.cli ingest corpus/
   ```

Ingesting without the gitignored files still works. The walker skips
missing paths gracefully, it just won't reproduce the exact 113-row index
the eval labels (`src/mmsearch/eval/labels.yaml`) were written against for
those three files' rows.
