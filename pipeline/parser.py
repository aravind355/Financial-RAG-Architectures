import pdfplumber
import fitz  # PyMuPDF
import json
import os
from pathlib import Path
from PIL import Image
import io

def extract_text_chunks(pdf_path, min_length=80, chunk_size=500):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue

            # Clean up the text
            text = text.strip()

            # Split by double newline first, fall back to fixed-size windows
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > min_length]

            # If no paragraphs found (single-newline PDF), use sliding window
            if not paragraphs:
                words = text.split()
                for i in range(0, len(words), chunk_size // 6):  # ~6 chars per word avg
                    chunk_words = words[i:i + chunk_size // 6]
                    chunk_text = " ".join(chunk_words)
                    if len(chunk_text) > min_length:
                        paragraphs.append(chunk_text)

            for idx, para in enumerate(paragraphs):
                chunks.append({
                    "type": "text",
                    "content": para,
                    "page": page_num + 1,
                    "source": Path(pdf_path).name,
                    "chunk_id": f"{Path(pdf_path).stem}_text_p{page_num+1}_{idx}"
                })
    return chunks

def extract_table_chunks(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue
                header = table[0]
                rows = table[1:]
                serialized = "Table:\n"
                serialized += " | ".join([str(h) for h in header if h]) + "\n"
                serialized += "-" * 60 + "\n"
                for row in rows:
                    serialized += " | ".join([str(c) for c in row if c is not None]) + "\n"
                chunks.append({
                    "type": "table",
                    "content": serialized,
                    "page": page_num + 1,
                    "source": Path(pdf_path).name,
                    "chunk_id": f"{Path(pdf_path).stem}_table_p{page_num+1}_{t_idx}"
                })
    return chunks

def extract_images(pdf_path, output_dir="data/images"):
    os.makedirs(output_dir, exist_ok=True)
    image_records = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                pil_img = Image.open(io.BytesIO(image_bytes))
                # skip tiny decorative images
                if pil_img.width < 150 or pil_img.height < 150:
                    continue
                img_filename = f"{Path(pdf_path).stem}_p{page_num+1}_{img_idx}.{image_ext}"
                img_path = os.path.join(output_dir, img_filename)
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                image_records.append({
                    "type": "image",
                    "image_path": img_path,
                    "content": "",  # filled later by captioner
                    "page": page_num + 1,
                    "source": Path(pdf_path).name,
                    "chunk_id": f"{Path(pdf_path).stem}_img_p{page_num+1}_{img_idx}"
                })
            except Exception as e:
                print(f"  Skipping image on page {page_num+1}: {e}")
    doc.close()
    return image_records

def extract_page_screenshots(pdf_path, output_dir="data/images", dpi=150):
    """Render pages that have charts as PNG screenshots."""
    os.makedirs(output_dir, exist_ok=True)
    image_records = []
    doc = fitz.open(pdf_path)

    # Pages likely to have charts (we'll detect by low text density)
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Check text density — low text + not a table-heavy page = likely a chart page
        text = page.get_text()
        word_count = len(text.split())
        
        # Render if page looks visual (few words, likely a chart)
        if word_count < 120:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_filename = f"{Path(pdf_path).stem}_screenshot_p{page_num+1}.png"
            img_path = os.path.join(output_dir, img_filename)
            pix.save(img_path)

            image_records.append({
                "type": "image",
                "image_path": img_path,
                "content": "",
                "page": page_num + 1,
                "source": Path(pdf_path).name,
                "chunk_id": f"{Path(pdf_path).stem}_screenshot_p{page_num+1}"
            })

    doc.close()
    print(f"  Rendered {len(image_records)} chart pages as screenshots")
    return image_records

def parse_pdf(pdf_path):
    print(f"\nParsing: {Path(pdf_path).name}")
    text_chunks   = extract_text_chunks(pdf_path)
    table_chunks  = extract_table_chunks(pdf_path)
    image_records = extract_page_screenshots(pdf_path)  # ← changed this line
    print(f"  text={len(text_chunks)}  tables={len(table_chunks)}  images={len(image_records)}")
    return text_chunks + table_chunks + image_records

def parse_all_pdfs(pdf_dir="data/pdfs", output_path="data/extracted/chunks.json"):
    os.makedirs("data/extracted", exist_ok=True)
    all_chunks = []
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in data/pdfs/ — check the folder.")
        return []
    for pdf_file in pdf_files:
        all_chunks.extend(parse_pdf(str(pdf_file)))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Total chunks: {len(all_chunks)}  →  saved to {output_path}")
    return all_chunks

if __name__ == "__main__":
    parse_all_pdfs()