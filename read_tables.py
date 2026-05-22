import pypdf

reader = pypdf.PdfReader("docs/2404.10518v2.pdf")
print("Total pages:", len(reader.pages))

# Search for pages containing Table 11
for idx, page in enumerate(reader.pages):
    text = page.extract_text()
    if "Table 11" in text or "MNv4-Conv-S" in text:
        print(f"--- Page {idx + 1} ---")
        print(text)
