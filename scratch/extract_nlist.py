import os
import PyPDF2

pdf_path = r"E:\Github Projects\dm\docs\N-list.pdf"
output_path = r"E:\Github Projects\dm\scratch\nlist_paper_text.txt"

if os.path.exists(pdf_path):
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            with open(output_path, 'w', encoding='utf-8') as out:
                out.write(f"Number of pages: {len(reader.pages)}\n\n")
                for i, page in enumerate(reader.pages):
                    out.write(f"--- Page {i+1} ---\n")
                    text = page.extract_text()
                    if text:
                        out.write(text)
                    out.write("\n\n")
        print(f"Text extracted to {output_path}")
    except Exception as e:
        print(f"Error reading PDF: {e}")
else:
    print(f"File not found: {pdf_path}")
