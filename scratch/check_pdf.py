import os
import PyPDF2

pdf_path = r"E:\Github Projects\dm\docs\2008-An Efficient Algorithm for Frequent Closed Itemsets Mining.pdf"

print(f"Checking path: {pdf_path}")
if os.path.exists(pdf_path):
    print("File exists!")
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            print(f"Number of pages: {len(reader.pages)}")
            for i, page in enumerate(reader.pages):
                print(f"Page {i+1} text:")
                print(page.extract_text()[:500]) # Print first 500 chars
    except Exception as e:
        print(f"Error reading PDF: {e}")
else:
    print("File does NOT exist at this path.")
