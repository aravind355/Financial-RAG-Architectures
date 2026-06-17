import pdfplumber
import sys

pdf_path = "data/pdfs/apple_2023.pdf"  # adjust filename if different

with pdfplumber.open(pdf_path) as pdf:
    print(f"Total pages: {len(pdf.pages)}")
    
    # Test page 5
    page = pdf.pages[4]
    text = page.extract_text()
    tables = page.extract_tables()
    
    print(f"\n--- Page 5 text (first 500 chars) ---")
    print(text[:500] if text else "No text found")
    
    print(f"\n--- Tables on page 5: {len(tables)} found ---")
    if tables:
        print("First table preview:")
        for row in tables[0][:3]:  # first 3 rows
            print(row)