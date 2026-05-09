import PyPDF2
import os

pdf_path = r"e:\Github Projects\dm\docs\2008-An Efficient Algorithm for Frequent Closed Itemsets Mining.pdf"

if __name__ == "__main__":
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
    else:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                print(f"Total pages: {num_pages}")
                for i in range(num_pages):
                    print(f"--- Page {i+1} ---")
                    text = reader.pages[i].extract_text()
                    if text:
                        print(text)
                    else:
                        print("[No text extracted from this page]")
        except Exception as e:
            print(f"Error: {str(e)}")
