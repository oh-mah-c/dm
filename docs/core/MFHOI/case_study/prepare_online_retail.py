"""Convert the UCI Online Retail dataset (Chen, 2015; doi:10.24432/C5BW33) into a
transaction database for the case study.

Each invoice is one transaction and each distinct StockCode is one item.
Removed: cancelled invoices (InvoiceNo starting with 'C'), lines with a
non-positive quantity, and non-product codes (postage, carriage, discounts,
manual adjustments, bank charges, samples, gift vouchers).

Usage (standard library only):
  python3 prepare_online_retail.py "Online Retail.xlsx" out_dir
Writes out_dir/online_retail.txt (one transaction per line, item ids) and
out_dir/items.tsv (item id, StockCode, most frequent description).
"""
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NON_PRODUCT = {"POST", "DOT", "M", "D", "C2", "CRUK", "BANK CHARGES", "AMAZONFEE", "S", "PADS", "B", "m"}


def col_index(ref):
    letters = "".join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def main(xlsx, out_dir):
    z = zipfile.ZipFile(xlsx)
    shared = []
    for _, el in ET.iterparse(z.open("xl/sharedStrings.xml")):
        if el.tag.endswith("}si"):
            shared.append("".join(t.text or "" for t in el.iter() if t.tag.endswith("}t")))
            el.clear()
    invoices, names, codes = {}, {}, {}
    header = True
    for _, el in ET.iterparse(z.open("xl/worksheets/sheet1.xml")):
        if not el.tag.endswith("}row"):
            continue
        vals = {}
        for c in el:
            if not c.tag.endswith("}c"):
                continue
            v = next((x.text for x in c if x.tag.endswith("}v")), None)
            if v is not None and c.get("t") == "s":
                v = shared[int(v)]
            vals[col_index(c.get("r"))] = v
        el.clear()
        if header:
            header = False
            continue
        inv, code, desc, qty = (str(vals.get(i, "") or "").strip() for i in range(4))
        if not inv or inv.startswith("C") or not code or code.upper() in NON_PRODUCT or code.startswith("gift"):
            continue
        try:
            if float(qty) <= 0:
                continue
        except ValueError:
            continue
        if code not in codes:
            codes[code] = len(codes) + 1
            names[codes[code]] = {}
        # description: the most frequent one of the code (some lines carry notes such as "mailout")
        if desc:
            d = names[codes[code]]
            d[desc] = d.get(desc, 0) + 1
        invoices.setdefault(inv, set()).add(codes[code])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "online_retail.txt", "w") as f:
        for items in invoices.values():
            f.write(" ".join(map(str, sorted(items))) + "\n")
    inv_codes = {v: k for k, v in codes.items()}
    with open(out / "items.tsv", "w") as f:
        for i in sorted(names):
            d = names[i]
            best = max(d, key=lambda k: d[k]) if d else ""   # ties: first seen (dicts keep insertion order)
            f.write(f"{i}\t{inv_codes[i]}\t{best}\n")
    lens = [len(v) for v in invoices.values()]
    print(f"transactions={len(invoices)} items={len(codes)} avg_len={sum(lens)/len(lens):.2f} "
          f"min_len={min(lens)} max_len={max(lens)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
