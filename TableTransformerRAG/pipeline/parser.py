"""
pipeline/parser.py
==================
Table Transformer (TATR) based PDF parser for TableTransformerRAG.

Replaces HierFinRAG's pdfplumber heuristic table detection with Microsoft's
Table Transformer — a DETR-based vision model that detects tables and
recognises their row/column structure from page images.

Pipeline per page
-----------------
1. Render PDF page to a PIL Image at 300 DPI using PyMuPDF (fitz).
2. Run TATR detection model → table bounding boxes in image coordinates.
3. For each detected table:
   a. Crop the table region (+padding) and run TATR structure model →
      row, column, and column-header bounding boxes.
   b. Map pdfplumber word coordinates into the detected cells using
      bounding-box overlap (no OCR required for native PDFs).
   c. Identify the first-row group as column headers.
   d. Produce the same chunk schema as HierFinRAG:
      table chunk → cell chunks (one per row×column intersection).
4. Extract non-table text blocks as text/section chunks.

Chunk schema (identical to HierFinRAG for fair downstream comparison)
-----------------------------------------------------------------------
{
    type           : 'section' | 'text' | 'table' | 'cell'
    content        : str
    page           : int
    source         : str       (PDF filename)
    chunk_id       : str       (globally unique)
    parent_id      : str | None
    children_ids   : list[str]
    parent_section : str | None
    # Table chunks only:
    col_headers    : list[str]
    row_headers    : list[str]
    rows           : list[list[str]]
    # Cell chunks only:
    row_idx        : int
    col_idx        : int
    row_header     : str
    col_header     : str
    value          : str
}

Run standalone
--------------
    python -m pipeline.parser
    python -m pipeline.parser --pdf data/pdfs/apple_2023.pdf --out data/extracted/chunks.json
"""

import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import Optional, Tuple

import fitz           # PyMuPDF
import pdfplumber
import torch
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

import config


# ---------------------------------------------------------------------------
# TATR model loader (cached as module-level singletons)
# ---------------------------------------------------------------------------

_det_processor: Optional[AutoImageProcessor]          = None
_det_model:     Optional[TableTransformerForObjectDetection] = None
_str_processor: Optional[AutoImageProcessor]          = None
_str_model:     Optional[TableTransformerForObjectDetection] = None

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_model_path(local_path: str, hf_id: str) -> str:
    """Return the local model path if it exists, else fall back to HF model ID."""
    if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
        return local_path
    return hf_id


def _load_models() -> None:
    """Load TATR detection and structure models (once, cached globally)."""
    global _det_processor, _det_model, _str_processor, _str_model
    if _det_model is not None:
        return  # already loaded

    det_path = _get_model_path(config.TATR_DETECTION_LOCAL, config.TATR_DETECTION_MODEL)
    str_path = _get_model_path(config.TATR_STRUCTURE_LOCAL, config.TATR_STRUCTURE_MODEL)

    print(f"Loading TATR detection model ({det_path}) on {_DEVICE}...")
    _det_processor = AutoImageProcessor.from_pretrained(det_path)
    _det_model     = TableTransformerForObjectDetection.from_pretrained(
        det_path
    ).to(_DEVICE).eval()

    print(f"Loading TATR structure model ({str_path}) on {_DEVICE}...")
    _str_processor = AutoImageProcessor.from_pretrained(str_path)
    _str_model     = TableTransformerForObjectDetection.from_pretrained(
        str_path
    ).to(_DEVICE).eval()

    print("TATR models loaded.\n")


# ---------------------------------------------------------------------------
# Section-heading heuristic (same as HierFinRAG)
# ---------------------------------------------------------------------------

# Regex for 10-K financial section headings like "ITEM 1", "Note 3", "Part II"
_SECTION_RE = re.compile(
    r"^(?:"
    r"item\s+\d+[a-z]?"
    r"|note\s+\d+"
    r"|part\s+[ivxIVX]+"
    r"|consolidated\s+"
    r"|selected\s+financial"
    r"|management.?s\s+discussion"
    r"|quantitative\s+and\s+qualitative"
    r"|financial\s+statements"
    r"|report\s+of"
    r")",
    re.IGNORECASE,
)


def _is_header(text: str) -> bool:
    """Return True if the text block looks like a section heading.

    Criteria (any one sufficient):
    - Short (< 12 words), no trailing period, all-title-case or ALL-CAPS
    - Matches a known 10-K financial section pattern
    """
    stripped = text.strip()
    words = stripped.split()
    if 0 < len(words) < 12 and not stripped.endswith("."):
        if stripped.istitle() or stripped.isupper():
            return True
    if _SECTION_RE.match(stripped):
        return True
    return False


def _has_numeric_content(rows: list) -> bool:
    """Return True if at least one cell in the table contains a digit.

    Used to filter out non-financial tables (cover pages, ToC, form headers)
    that TATR mis-detects.  Real financial tables always contain numbers.
    """
    _digit_re = re.compile(r"\d")
    for row in rows:
        for cell in row:
            if _digit_re.search(str(cell)):
                return True
    return False


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _img_to_pdf(bbox_img: list, scale: float) -> tuple:
    """Convert image-space bbox [x1,y1,x2,y2] (pixels) to PDF-space (points)."""
    x1, y1, x2, y2 = bbox_img
    return x1 / scale, y1 / scale, x2 / scale, y2 / scale


def _overlap_ratio(a: tuple, b: tuple) -> float:
    """Intersection-over-min-area between two (x1,y1,x2,y2) rects."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = min(area_a, area_b)
    return inter / denom if denom > 0 else 0.0


def _words_in_cell(words: list, cell_bbox: tuple, min_overlap: float = 0.3) -> str:
    """Return text of pdfplumber words whose centre falls in the cell bbox."""
    x1, y1, x2, y2 = cell_bbox
    matched = []
    for w in words:
        # word bbox in pdfplumber: x0, top, x1, bottom
        cx = (w["x0"] + w["x1"]) / 2
        cy = (w["top"] + w["bottom"]) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            matched.append(w["text"])
    return " ".join(matched).strip()


# ---------------------------------------------------------------------------
# TATR inference helpers
# ---------------------------------------------------------------------------

def _detect_tables(image: Image.Image, threshold: float) -> list:
    """Run TATR detection model; return list of (score, [x1,y1,x2,y2]) in image pixels."""
    inputs = _det_processor(images=image, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        outputs = _det_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=_DEVICE)
    results = _det_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    tables = []
    id2label = _det_model.config.id2label
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        lbl = id2label[label.item()]
        if "table" in lbl.lower():
            tables.append((float(score), box.tolist()))
    return tables


def _recognise_structure(table_img: Image.Image, threshold: float) -> dict:
    """Run TATR structure model on a cropped table image.

    Returns:
        {
          'rows': list of (y1, y2) sorted by y1,
          'cols': list of (x1, x2) sorted by x1,
          'header_rows': int  (number of rows that are column headers)
        }
    """
    # Explicitly set size — the v1.1-All preprocessor_config.json only has
    # longest_edge=800 which transformers 5.x rejects (needs shortest_edge too)
    inputs = _str_processor(
        images=table_img, return_tensors="pt",
        size={"shortest_edge": 800, "longest_edge": 1333},
    ).to(_DEVICE)
    with torch.no_grad():
        outputs = _str_model(**inputs)
    target_sizes = torch.tensor([table_img.size[::-1]], device=_DEVICE)
    results = _str_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    id2label  = _str_model.config.id2label
    rows      = []
    cols      = []
    hdr_rows  = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        lbl = id2label[label.item()].lower()
        b   = box.tolist()  # [x1,y1,x2,y2] in table-crop pixels
        if "column header" in lbl:
            hdr_rows.append((b[1], b[3]))  # (y1, y2)
        elif "row" in lbl and "header" not in lbl:
            rows.append((b[1], b[3]))
        elif "column" in lbl:
            cols.append((b[0], b[2]))

    # Include header rows as the first rows
    all_rows = sorted(hdr_rows + rows, key=lambda r: r[0])
    all_cols = sorted(cols, key=lambda c: c[0])
    n_header = len(hdr_rows)

    return {"rows": all_rows, "cols": all_cols, "header_rows": n_header}


# ---------------------------------------------------------------------------
# Table chunk builder
# ---------------------------------------------------------------------------

def _build_table_chunks(
    structure:      dict,
    table_img_bbox: tuple,        # (x1,y1,x2,y2) of the table in page PDF coords
    crop_size:      Tuple[int, int],  # (crop_width_px, crop_height_px) of the table PIL crop
    words:          list,         # pdfplumber words from the page
    page_num:       int,
    source:         str,
    parent_section: Optional[str],
    chunk_prefix:   str,
) -> list:
    """Convert TATR structure results into HierFinRAG-compatible chunk dicts."""

    rows_ys = structure["rows"]    # [(y1,y2), ...]
    cols_xs = structure["cols"]    # [(x1,x2), ...]
    n_hdr   = structure["header_rows"]

    if not rows_ys or not cols_xs:
        return []

    tx1, ty1, tx2, ty2 = table_img_bbox  # table bbox in PDF points (page-level origin)
    crop_w_px, crop_h_px = crop_size      # actual pixel dimensions of the table PIL crop

    # Use the real crop pixel dimensions to map TATR structure coordinates
    # (which are in crop-image pixel space) back to page PDF point space.
    w_pdf = tx2 - tx1
    h_pdf = ty2 - ty1

    crop_w_px = max(crop_w_px, 1)
    crop_h_px = max(crop_h_px, 1)

    all_x = [x for (x1, x2) in cols_xs for x in (x1, x2)]
    all_y = [y for (y1, y2) in rows_ys for y in (y1, y2)]
    if not all_x or not all_y:
        return []

    def px_col_to_pdf(x):
        return tx1 + (x / crop_w_px) * w_pdf

    def px_row_to_pdf(y):
        return ty1 + (y / crop_h_px) * h_pdf

    def _extract_row_text(ry1, ry2):
        """Extract text for each column in a row."""
        row = []
        for ci, (cx1, cx2) in enumerate(cols_xs):
            cell_bbox = (
                px_col_to_pdf(cx1), px_row_to_pdf(ry1),
                px_col_to_pdf(cx2), px_row_to_pdf(ry2),
            )
            row.append(_words_in_cell(words, cell_bbox))
        return row

    # Build column header text (first n_hdr rows)
    col_headers = []
    for ci, (cx1, cx2) in enumerate(cols_xs):
        hdr_text_parts = []
        for ri in range(n_hdr):
            ry1, ry2 = rows_ys[ri]
            cell_bbox = (
                px_col_to_pdf(cx1), px_row_to_pdf(ry1),
                px_col_to_pdf(cx2), px_row_to_pdf(ry2),
            )
            txt = _words_in_cell(words, cell_bbox)
            if txt:
                hdr_text_parts.append(txt)
        col_headers.append(" ".join(hdr_text_parts) if hdr_text_parts else f"Col{ci}")

    # Data rows (skip header rows)
    data_rows_ys = rows_ys[n_hdr:] if n_hdr < len(rows_ys) else rows_ys
    if not data_rows_ys:
        data_rows_ys = rows_ys

    # Extract all data row text
    raw_rows = []
    for ry1, ry2 in data_rows_ys:
        raw_rows.append(_extract_row_text(ry1, ry2))

    # --- FIX Change 3: Promote first data row to headers if headers are placeholders ---
    # When TATR misses header detection, col_headers will be ['Col0', 'Col1', ...].
    # If the first data row looks like headers (contains year-like or text-only values),
    # use it as the column headers instead.
    placeholder_count = sum(1 for h in col_headers if re.match(r"^Col\d+$", h))
    if placeholder_count > len(col_headers) // 2 and raw_rows:
        first_row = raw_rows[0]
        # Heuristic: a header row has mostly non-empty text without $ signs
        non_empty = sum(1 for cell in first_row if cell.strip())
        has_dollar = any("$" in cell for cell in first_row)
        if non_empty >= len(first_row) // 2 and not has_dollar:
            col_headers = [cell.strip() if cell.strip() else f"Col{i}"
                          for i, cell in enumerate(first_row)]
            raw_rows = raw_rows[1:]  # Remove the promoted header row from data

    # --- FIX Change 2: Deduplicate rows ---
    # TATR sometimes produces duplicate rows where:
    # (a) a data row is identical to the column headers (header repeated as data)
    # (b) two consecutive data rows have identical text (overlapping row bboxes)
    def _row_signature(row):
        return "|".join(cell.strip().lower() for cell in row)

    hdr_sig = _row_signature(col_headers)
    rows_data = []
    seen_sigs = {hdr_sig}  # pre-seed with header signature to skip header duplicates
    for row in raw_rows:
        sig = _row_signature(row)
        # Skip completely empty rows
        if all(not cell.strip() for cell in row):
            continue
        # Skip rows identical to headers or previously seen
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        rows_data.append(row)

    if not rows_data:
        # If dedup removed everything, fall back to raw rows
        rows_data = [r for r in raw_rows if any(cell.strip() for cell in r)]

    # Row headers = first-column text for each data row
    row_headers = [row[0] if row else f"Row{ri}" for ri, row in enumerate(rows_data)]

    # Flatten table to a text representation for embedding
    header_line = " | ".join(col_headers)
    row_lines   = [" | ".join(r) for r in rows_data]
    table_text  = header_line + "\n" + "\n".join(row_lines)

    table_id = f"{chunk_prefix}_table"
    cell_ids = []
    cell_chunks = []

    for ri, row in enumerate(rows_data):
        for ci, value in enumerate(row):
            cell_id = f"{table_id}_r{ri}_c{ci}"
            cell_ids.append(cell_id)
            cell_content = (
                f"{row_headers[ri]} | {col_headers[ci]}: {value}"
            ).strip()
            cell_chunks.append({
                "type":           "cell",
                "content":        cell_content,
                "page":           page_num,
                "source":         source,
                "chunk_id":       cell_id,
                "parent_id":      table_id,
                "children_ids":   [],
                "parent_section": parent_section,
                "col_headers":    None,
                "row_headers":    None,
                "rows":           None,
                "row_idx":        ri,
                "col_idx":        ci,
                "row_header":     row_headers[ri],
                "col_header":     col_headers[ci],
                "value":          value,
            })

    table_chunk = {
        "type":           "table",
        "content":        table_text,
        "page":           page_num,
        "source":         source,
        "chunk_id":       table_id,
        "parent_id":      None,
        "children_ids":   cell_ids,
        "parent_section": parent_section,
        "col_headers":    col_headers,
        "row_headers":    row_headers,
        "rows":           rows_data,
        "row_idx":        None,
        "col_idx":        None,
        "row_header":     None,
        "col_header":     None,
        "value":          None,
    }

    return [table_chunk] + cell_chunks


# ---------------------------------------------------------------------------
# Per-page parser
# ---------------------------------------------------------------------------

def _parse_page(
    fitz_page,       # fitz.Page
    plumb_page,      # pdfplumber.Page
    page_num:  int,
    source:    str,
    chunk_prefix: str,
    current_section: dict,
) -> tuple:
    """Parse one PDF page into chunks using TATR + pdfplumber.

    Returns (chunks, current_section) where current_section is updated if a
    new heading was detected on this page.
    """
    chunks = []

    # Render page to PIL image at configured DPI
    dpi    = config.TATR_RENDER_DPI
    scale  = dpi / 72.0           # PDF points → image pixels
    mat    = fitz.Matrix(scale, scale)
    pix    = fitz_page.get_pixmap(matrix=mat, alpha=False)
    img    = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Extract all words from pdfplumber (PDF-space coords in points)
    all_words = plumb_page.extract_words() or []

    # ── Step 1: detect tables ────────────────────────────────────────────────
    detected = _detect_tables(img, config.TATR_DET_THRESHOLD)

    # Track which PDF-coord regions are covered by tables (to exclude their
    # words from text-block extraction below)
    table_bboxes_pdf = []   # [(x1,y1,x2,y2) in PDF points]
    table_chunk_idx  = 0

    for t_score, t_box_img in detected:
        # Convert table bbox: image pixels → PDF points
        x1p, y1p, x2p, y2p = _img_to_pdf(t_box_img, scale)
        table_bboxes_pdf.append((x1p, y1p, x2p, y2p))

        # Crop the table image for structure recognition
        x1i, y1i, x2i, y2i = t_box_img
        # Add a small pixel padding to avoid clipping row/col lines
        pad = 5
        table_img_crop = img.crop((
            max(0, x1i - pad), max(0, y1i - pad),
            min(img.width,  x2i + pad),
            min(img.height, y2i + pad),
        ))
        crop_size = table_img_crop.size  # (width_px, height_px) — passed to _build_table_chunks

        # Run TATR structure recognition on the crop
        structure = _recognise_structure(table_img_crop, config.TATR_STR_THRESHOLD)

        # Get words that overlap with this table's PDF bbox
        table_words = [
            w for w in all_words
            if (w["x0"] >= x1p - 5 and w["x1"] <= x2p + 5
                and w["top"] >= y1p - 5 and w["bottom"] <= y2p + 5)
        ]

        prefix = f"{chunk_prefix}_tbl{page_num}_{table_chunk_idx}"
        table_chunk_idx += 1

        tbl_chunks = _build_table_chunks(
            structure      = structure,
            table_img_bbox = (x1p, y1p, x2p, y2p),
            crop_size      = crop_size,
            words          = table_words,
            page_num       = page_num,
            source         = source,
            parent_section = current_section.get("content"),
            chunk_prefix   = prefix,
        )

        # FIX: skip tables with no numeric content (cover pages, ToC, form headers)
        if not tbl_chunks or not _has_numeric_content(
            [row for row in (tbl_chunks[0].get("rows") or [])]
        ):
            continue

        # Link table chunk to its parent section
        if current_section:
            current_section["children_ids"].append(tbl_chunks[0]["chunk_id"])
            tbl_chunks[0]["parent_id"] = current_section["chunk_id"]

        chunks.extend(tbl_chunks)

    # ── Step 2: extract non-table text ───────────────────────────────────────
    # Words not covered by any detected table → group into text blocks
    def _in_table(word):
        cx = (word["x0"] + word["x1"]) / 2
        cy = (word["top"] + word["bottom"]) / 2
        for (tx1, ty1, tx2, ty2) in table_bboxes_pdf:
            if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                return True
        return False

    non_table_words = [w for w in all_words if not _in_table(w)]

    # Group words into lines (same top coordinate), then into paragraphs
    lines: dict = {}
    for w in non_table_words:
        key = round(w["top"])  # group by y-position
        lines.setdefault(key, []).append(w["text"])

    para_lines = []
    for key in sorted(lines.keys()):
        para_lines.append(" ".join(lines[key]))

    # Merge lines into paragraphs (split on blank-ish gaps)
    paragraphs = []
    current_para = []
    for i, line in enumerate(para_lines):
        if line.strip():
            current_para.append(line)
        else:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
    if current_para:
        paragraphs.append(" ".join(current_para))

    # Split long paragraphs at PARSER_CHUNK_SIZE words
    final_paras = []
    for para in paragraphs:
        words_list = para.split()
        chunk_sz   = config.PARSER_CHUNK_SIZE
        if len(words_list) <= chunk_sz:
            final_paras.append(para)
        else:
            for start in range(0, len(words_list), chunk_sz):
                final_paras.append(" ".join(words_list[start:start + chunk_sz]))

    text_chunk_idx = 0
    for para in final_paras:
        if len(para) < config.PARSER_MIN_TEXT_LEN:
            continue

        if _is_header(para):
            # New section
            sec_id = f"{chunk_prefix}_sec{page_num}_{text_chunk_idx}"
            sec_chunk = {
                "type":           "section",
                "content":        para,
                "page":           page_num,
                "source":         source,
                "chunk_id":       sec_id,
                "parent_id":      None,
                "children_ids":   [],
                "parent_section": None,
                "col_headers":    None,
                "row_headers":    None,
                "rows":           None,
                "row_idx":        None,
                "col_idx":        None,
                "row_header":     None,
                "col_header":     None,
                "value":          None,
            }
            chunks.append(sec_chunk)
            current_section = sec_chunk
        else:
            txt_id = f"{chunk_prefix}_txt{page_num}_{text_chunk_idx}"
            txt_chunk = {
                "type":           "text",
                "content":        para,
                "page":           page_num,
                "source":         source,
                "chunk_id":       txt_id,
                "parent_id":      current_section.get("chunk_id") if current_section else None,
                "children_ids":   [],
                "parent_section": current_section.get("content") if current_section else None,
                "col_headers":    None,
                "row_headers":    None,
                "rows":           None,
                "row_idx":        None,
                "col_idx":        None,
                "row_header":     None,
                "col_header":     None,
                "value":          None,
            }
            if current_section:
                current_section["children_ids"].append(txt_id)
            chunks.append(txt_chunk)

        text_chunk_idx += 1

    return chunks, current_section


# ---------------------------------------------------------------------------
# Main PDF parser
# ---------------------------------------------------------------------------

def parse_pdf(
    pdf_path: str,
    progress: bool = True,
) -> list:
    """Parse a single PDF into a flat list of hierarchical chunks using TATR.

    Args:
        pdf_path  : Path to the PDF file.
        progress  : Print per-page progress.

    Returns:
        Flat list of chunk dicts with identical schema to HierFinRAG.
    """
    _load_models()

    source   = Path(pdf_path).name
    prefix   = Path(pdf_path).stem.replace(" ", "_").replace("-", "_")
    all_chunks = []
    current_section: dict = {}

    fitz_doc  = fitz.open(pdf_path)
    plumb_pdf = pdfplumber.open(pdf_path)

    try:
        n_pages = len(fitz_doc)
        for page_num in range(1, n_pages + 1):
            if progress:
                print(f"  Page {page_num:3d}/{n_pages} ...", end="\r", flush=True)

            fitz_page  = fitz_doc[page_num - 1]
            plumb_page = plumb_pdf.pages[page_num - 1]

            page_chunks, current_section = _parse_page(
                fitz_page      = fitz_page,
                plumb_page     = plumb_page,
                page_num       = page_num,
                source         = source,
                chunk_prefix   = prefix,
                current_section= current_section,
            )
            all_chunks.extend(page_chunks)

        if progress:
            print(f"  Done — {len(all_chunks)} chunks from {n_pages} pages.    ")
    finally:
        plumb_pdf.close()
        fitz_doc.close()

    return all_chunks


def parse_all_pdfs(
    pdf_dir:     str = "data/pdfs",
    output_path: str = "data/extracted/chunks.json",
) -> int:
    """Parse all PDFs in pdf_dir and write combined chunks to output_path.

    Returns:
        Total number of chunks written.
    """
    pdf_dir    = Path(pdf_dir)
    pdfs       = sorted(pdf_dir.glob("*.pdf"))
    all_chunks = []

    print(f"Found {len(pdfs)} PDF(s) in {pdf_dir}\n")
    for pdf in pdfs:
        print(f"Parsing: {pdf.name}")
        chunks = parse_pdf(str(pdf))

        # --- FIX Change 7: Promote text chunks to pseudo-sections if no sections found ---
        # Some PDFs (e.g. Alphabet 10-K) have headings in formats that _is_header
        # doesn't catch. If parsing produced zero section chunks for this PDF,
        # promote each text chunk to a section and link nearby tables as children.
        section_count = sum(1 for c in chunks if c["type"] == "section")
        if section_count == 0:
            # Build page -> [table chunk_ids] mapping
            page_tables: dict = {}
            for c in chunks:
                if c["type"] == "table":
                    pg = c["page"]
                    page_tables.setdefault(pg, []).append(c["chunk_id"])

            for c in chunks:
                if c["type"] == "text":
                    c["type"] = "section"
                    # Link tables from the same page and adjacent pages as children
                    pg = c["page"]
                    child_ids = []
                    for p in (pg - 1, pg, pg + 1):
                        child_ids.extend(page_tables.get(p, []))
                    c["children_ids"] = child_ids

        all_chunks.extend(chunks)
        # Count by type for summary
        by_type: dict = {}
        for c in chunks:
            by_type[c["type"]] = by_type.get(c["type"], 0) + 1
        print(f"  -> {by_type}")
        print()

    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_chunks)} total chunks → {output_path}")
    return len(all_chunks)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse PDFs using Table Transformer for TableTransformerRAG."
    )
    parser.add_argument("--pdf",   default=None,
                        help="Parse a single PDF (default: parse all in data/pdfs/)")
    parser.add_argument("--out",   default="data/extracted/chunks.json",
                        help="Output chunks JSON path")
    args = parser.parse_args()

    if args.pdf:
        chunks = parse_pdf(args.pdf)
        os.makedirs(Path(args.out).parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(chunks)} chunks → {args.out}")
    else:
        parse_all_pdfs(output_path=args.out)
