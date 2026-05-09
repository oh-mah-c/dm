import pdfplumber
import os

pdf_path = r"E:\Github Projects\dm\docs\N-list.pdf"
output_path = r"E:\Github Projects\dm\scratch\nlist_paper_text.txt"

if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
else:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Total pages: {len(pdf.pages)}\n\n")
                for i, page in enumerate(pdf.pages):
                    f.write(f"--- Page {i+1} ---\n")
                    f.write(page.extract_text() or "[No text]")
                    f.write("\n\n")
        print(f"Successfully extracted to {output_path}")
    except Exception as e:
        print(f"Error: {e}")
