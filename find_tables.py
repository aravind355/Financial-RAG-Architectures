# save as find_tables.py
import pdfplumber

pdf_path = "data/pdfs/apple_2023.pdf"

with pdfplumber.open(pdf_path) as pdf:
    print(f"Total pages: {len(pdf.pages)}\n")
    print("Pages with tables:")
    for i, page in enumerate(pdf.pages):
        tables = page.extract_tables()
        if tables:
            print(f"  Page {i+1}: {len(tables)} table(s), first table has {len(tables[0])} rows")