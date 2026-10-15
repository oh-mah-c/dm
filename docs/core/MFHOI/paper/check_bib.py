"""Check every DOI entry of a BibTeX file against its registry record.

For each entry with a DOI, fetch CSL-JSON metadata from doi.org (works for
Crossref and DataCite DOIs) and compare title, year, volume, pages and the
authors (family and given names). Entries with a URL are fetched and must
answer HTTP 200. Exit status 1 if anything differs.

  python3 check_bib.py mfhoi.bib
"""
import json
import re
import sys
import time
import unicodedata
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) bib-check/1.0"


def norm(s):
    s = re.sub(r"\\[a-zA-Z]+\s*|[{}\\'\"`^~]", "", s or "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def latex_to_text(s):
    rep = {r"{\'n}": "n", r"{\'e}": "e", r"{\'i}": "i", r"{\'a}": "a", r'{\"o}': "o", r"{\i}": "i", r"{\.I}": "I"}
    for a, b in rep.items():
        s = s.replace(a, b)
    return s


def fetch(doi):
    for attempt in range(6):
        try:
            req = urllib.request.Request("https://doi.org/" + doi,
                                         headers={"Accept": "application/vnd.citationstyles.csl+json", "User-Agent": UA})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("rate limited")


def url_ok(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        return urllib.request.urlopen(req, timeout=30).status == 200
    except Exception:
        return False


def parse_bib(path):
    text = open(path).read()
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        fields = dict((k.lower(), v.strip()) for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\},?\s*\n", m.group(3) + "\n", re.S))
        entries.append((m.group(2), fields))
    return entries


def main(path):
    bad = 0
    for key, f in parse_bib(path):
        problems = []
        if "doi" in f:
            try:
                m = fetch(f["doi"])
            except Exception as e:
                print(f"FAIL {key}: DOI {f['doi']} not resolvable ({e})")
                bad += 1
                continue
            title = m.get("title") or ""
            if norm(latex_to_text(f.get("title", ""))) != norm(title):
                problems.append(f"title: bib='{f.get('title')}' registry='{title}'")
            year = None
            for k in ("issued", "published-print", "published-online"):
                if m.get(k, {}).get("date-parts"):
                    year = m[k]["date-parts"][0][0]
                    break
            if "year" in f and year and int(f["year"]) != int(year):
                problems.append(f"year: bib={f['year']} registry={year}")
            if "volume" in f and m.get("volume") and norm(f["volume"]) != norm(str(m["volume"])):
                problems.append(f"volume: bib={f['volume']} registry={m.get('volume')}")
            pages = m.get("page") or m.get("article-number")
            if "pages" in f and pages and norm(f["pages"].replace("--", "-")) != norm(str(pages).replace("\u2013", "-")):
                problems.append(f"pages: bib={f['pages']} registry={pages}")
            reg_auth = [(a.get("family", a.get("literal", "")), a.get("given", "")) for a in m.get("author", [])]
            bib_field = f.get("author") or f.get("editor") or ""
            bib_auth = []
            for a in re.split(r"\s+and\s+", bib_field):
                a = latex_to_text(a.strip())
                if not a:
                    continue
                fam, _, giv = a.partition(",")
                bib_auth.append((fam.strip(), giv.strip()))
            if reg_auth and "author" in f:
                if [norm(x[0]) for x in bib_auth] != [norm(x[0]) for x in reg_auth]:
                    problems.append(f"authors (family): bib={[x[0] for x in bib_auth]} registry={[x[0] for x in reg_auth]}")
                else:
                    for (bf, bg), (rf, rg) in zip(bib_auth, reg_auth):
                        if rg and bg and norm(bg).split()[0][:1] != norm(rg).split()[0][:1]:
                            problems.append(f"given name of {bf}: bib={bg} registry={rg}")
                        elif rg and bg and len(norm(bg)) > 2 and len(norm(rg)) > 2 and norm(bg) != norm(rg):
                            problems.append(f"given name of {bf}: bib={bg} registry={rg}")
            time.sleep(1)
        elif "url" in f or "howpublished" in f:
            url = f.get("url") or re.sub(r"\\url\{(.*)\}", r"\1", f.get("howpublished", ""))
            if url.startswith("http") and not url_ok(url):
                problems.append(f"url not reachable: {url}")
        status = "OK  " if not problems else "DIFF"
        bad += bool(problems)
        print(f"{status} {key}" + ("" if not problems else "\n      " + "\n      ".join(problems)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1])
