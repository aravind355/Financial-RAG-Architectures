"""
pipeline/parser.py
==================
Hierarchical PDF document parser.

Produces a four-level chunk hierarchy:

    Document
    └── Section  (level 1 — detected by heading heuristic)
        ├── Paragraph  (level 2 — body text)
        └── Table      (level 2 — structured data)
            └── Cell   (level 3 — individual table cells)

Each chunk schema:
    type           : 'section' | 'text' | 'table' | 'cell' | 'image'
    content        : str
    page           : int
    source         : str   (PDF filename)
    chunk_id       : str   (globally unique)
    parent_id      : str | None
    children_ids   : list[str]
    parent_section : str | None

Table chunks additionally carry row_headers, col_headers, rows.
Cell chunks additionally carry row_idx, col_idx, row_header, col_header, value.
"""

import pdfplumber
import fitz
import json
import os
from pathlib import Path
from PIL import Image  # noqa: F401


def _is_header(text: str) -> bool:
    """Return True if the text block looks like a section heading.
    Heuristics: short line, title-cased or ALL-CAPS, no trailing period."""
    words = text.split()
    if 0 < len(words) < 10 and not text.endswith("."):
        if text.istitle() or text.isupper():
            return True
    return False


def parse_pdf(
    pdf_path: str,
    min_text_len: int = 40,
    chunk_size: int = 500,
    image_dir: str = "data/images",
    dpi: int = 150,
) -> list[dict]:
    """Parse a single PDF into a flat list of hierarchical chunks.

    Args:
        pdf_path     : Path to the PDF file.
        min_text_len : Minimum character count for a text chunk to be kept.
        chunk_size   : Target word count for text splitting fallback.
        image_dir    : Directory to save page screenshots for chart-heavy pages.
        dpi          : Screenshot resolution.

    Returns:
        List of chunk dicts.
    """
    stem        = Path(pdf_path).stem
    source_name = Path(pdf_path).name
    os.makedirs(image_dir, exist_ok=True)
    print(f"\nParsing: {source_name}")

    chunks: list[dict] = []

    root_sec_id           = f"{stem}_sec_root"
    current_section       = "Document"
    current_section_id    = root_sec_id
    current_section_title = "Document"

    chunks.append({
        "type":          "section",
        "content":       current_section,
        "page":          1,
        "source":        source_name,
        "chunk_id":      root_sec_id,
        "parent_id":     None,
        "children_ids":  [],
        "parent_section": None,
    })

    fitz_doc = fitz.open(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):

            raw_text = page.extract_text()
            if raw_text:
                raw_text   = raw_text.strip()
                paragraphs = [p.strip() for p in raw_text.split("\n\n")
                              if len(p.strip()) > min_text_len]
                if not paragraphs:
                    words      = raw_text.split()
                    window     = chunk_size // 6
                    paragraphs = [
                        " ".join(words[i: i + window])
                        for i in range(0, len(words), window)
                        if len(" ".join(words[i: i + window])) > min_text_len
                    ]

                for idx, para in enumerate(paragraphs):
                    if _is_header(para):
                        current_section_title = para
                        safe = "".join(
                            c for c in para if c.isalnum() or c == " "
                        ).replace(" ", "_")[:20]
                        current_section_id = f"{stem}_sec_{safe}_{page_num}_{idx}"

                        chunks.append({
                            "type":          "section",
                            "content":       para,
                            "page":          page_num + 1,
                            "source":        source_name,
                            "chunk_id":      current_section_id,
                            "parent_id":     None,
                            "children_ids":  [],
                            "parent_section": None,
                        })

                    else:
                        para_id    = f"{stem}_text_p{page_num+1}_{idx}"
                        para_chunk = {
                            "type":          "text",
                            "content":       para,
                            "page":          page_num + 1,
                            "source":        source_name,
                            "chunk_id":      para_id,
                            "parent_id":     current_section_id,
                            "children_ids":  [],
                            "parent_section": current_section_title,
                        }
                        chunks.append(para_chunk)
                        _add_child(chunks, current_section_id, para_id)

            tables = page.extract_tables()
            for t_idx, raw_table in enumerate(tables):
                if not raw_table or len(raw_table) < 2:
                    continue

                col_headers = [str(h) if h else "" for h in raw_table[0]]
                data_rows   = raw_table[1:]
                row_headers = [str(row[0]) if row[0] else "" for row in data_rows]

                serialized  = "Table:\n"
                serialized += " | ".join(col_headers) + "\n"
                serialized += "-" * 60 + "\n"
                for row in data_rows:
                    serialized += " | ".join(str(c) if c is not None else "" for c in row) + "\n"

                table_id    = f"{stem}_table_p{page_num+1}_{t_idx}"
                table_chunk = {
                    "type":          "table",
                    "content":       serialized,
                    "page":          page_num + 1,
                    "source":        source_name,
                    "chunk_id":      table_id,
                    "parent_id":     current_section_id,
                    "children_ids":  [],
                    "parent_section": current_section_title,
                    "col_headers":   col_headers,
                    "row_headers":   row_headers,
                    "rows":          [[str(c) if c is not None else "" for c in row]
                                      for row in data_rows],
                }
                chunks.append(table_chunk)
                _add_child(chunks, current_section_id, table_id)

                for r_idx, row in enumerate(data_rows):
                    for c_idx, cell_val in enumerate(row):
                        cell_val_str = str(cell_val) if cell_val is not None else ""
                        if not cell_val_str.strip():
                            continue

                        cell_id    = f"{table_id}_r{r_idx}_c{c_idx}"
                        row_hdr    = row_headers[r_idx] if r_idx < len(row_headers) else ""
                        col_hdr    = col_headers[c_idx] if c_idx < len(col_headers) else ""
                        cell_chunk = {
                            "type":          "cell",
                            "content":       f"{row_hdr} | {col_hdr} | {cell_val_str}",
                            "page":          page_num + 1,
                            "source":        source_name,
                            "chunk_id":      cell_id,
                            "parent_id":     table_id,
                            "children_ids":  [],
                            "parent_section": current_section_title,
                            "row_idx":       r_idx,
                            "col_idx":       c_idx,
                            "row_header":    row_hdr,
                            "col_header":    col_hdr,
                            "value":         cell_val_str,
                        }
                        chunks.append(cell_chunk)
                        _add_child(chunks, table_id, cell_id)

            word_count = len(raw_text.split()) if raw_text else 0
            if word_count < 120:
                mat       = fitz.Matrix(dpi / 72, dpi / 72)
                fitz_page = fitz_doc[page_num]
                pix       = fitz_page.get_pixmap(matrix=mat)
                img_name  = f"{stem}_screenshot_p{page_num+1}.png"
                img_path  = os.path.join(image_dir, img_name)
                pix.save(img_path)

                img_chunk = {
                    "type":          "image",
                    "content":       "",
                    "image_path":    img_path,
                    "page":          page_num + 1,
                    "source":        source_name,
                    "chunk_id":      f"{stem}_screenshot_p{page_num+1}",
                    "parent_id":     current_section_id,
                    "children_ids":  [],
                    "parent_section": current_section_title,
                }
                chunks.append(img_chunk)
                _add_child(chunks, current_section_id, img_chunk["chunk_id"])

    fitz_doc.close()

    counts = {"text": 0, "table": 0, "cell": 0, "image": 0, "section": 0}
    for c in chunks:
        counts[c["type"]] = counts.get(c["type"], 0) + 1
    print(
        f"  sections={counts['section']}  text={counts['text']}"
        f"  tables={counts['table']}  cells={counts['cell']}"
        f"  images={counts['image']}"
    )
    return chunks


def _add_child(chunks: list[dict], parent_id: str, child_id: str) -> None:
    """Append child_id to the parent chunk's children_ids list.
    Uses reverse scan since parents are always inserted before children."""
    for chunk in reversed(chunks):
        if chunk["chunk_id"] == parent_id:
            chunk["children_ids"].append(child_id)
            return


def parse_all_pdfs(
    pdf_dir: str = "data/pdfs",
    output_path: str = "data/extracted/chunks.json",
) -> list[dict]:
    """Parse every PDF in pdf_dir and write all chunks to a single JSON file."""
    os.makedirs("data/extracted", exist_ok=True)
    all_chunks: list[dict] = []

    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in data/pdfs/ — place your financial PDFs there.")
        return []

    for pdf_file in pdf_files:
        all_chunks.extend(parse_pdf(str(pdf_file)))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Total chunks: {len(all_chunks)}  →  saved to {output_path}")
    return all_chunks


if __name__ == "__main__":
    parse_all_pdfs()