#!/usr/bin/env python3
import sys
import subprocess
import os

def extract_text_from_pdf(pdf_path):
    if not os.path.isfile(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        sys.exit(1)
    
    try:
        # Run pdftotext and capture the output to stdout
        result = subprocess.run(['pdftotext', pdf_path, '-'], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE,
                                text=True,
                                check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error extracting text from PDF: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'pdftotext' command not found. Please install poppler-utils.")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    text = extract_text_from_pdf(pdf_path)
    print(text)
