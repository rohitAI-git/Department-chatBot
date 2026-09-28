"""
EXTRACTOR ENGINE -  raw crawled html/pdf to clean structured text (docs) ready for chunking
+ BOILERPLATE FOR BEAUTIFULSOUP4 NOT 5
"""
import json
import re
from datetime import datetime
from pathlib import Path
import trafilatura
from bs4 import BeautifulSoup
from pypdf import PdfReader
from dateutil import parser as dateparser
rawdir = Path(__file__).resolve().parent.parent / "data" / "raw"
docspath = Path(__file__).resolve().parent.parent / "data" / "extracteddocs.jsonl"
datepattern = re.compile(
    r"\b(\d{1,2}[-/ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/ ]\d{2,4}"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
    re.IGNORECASE,
)
def guessdates(text: str) -> list[str]:
    found = []
    for m in datepattern.findall(text[:5000]):#dates usually near top of notices
        try:
            found.append(dateparser.parse(m, dayfirst=True, fuzzy=True).date().isoformat())
        except Exception:
            continue
    return sorted(set(found), reverse=True)
def extracthtml(rawbytes: bytes, url: str) -> dict:
    html = rawbytes.decode("utf-8", errors="ignore")
    maintext = trafilatura.extract(html, include_tables=True, include_links=False) or ""
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string if soup.title else "").strip()
    # notice-board and course pages on this template put the real payload
    # in tables / <li> blocks. grab those explicitly as a fallback /
    # supplement so nothing "hidden in a sidebar" is lost.
    listitems = [li.get_text(" ", strip=True) for li in soup.find_all("li")]
    tablerows = []
    for row in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        if any(cells):
            tablerows.append(" | ".join(cells))
    combined = "\n".join(filter(None, [maintext, *listitems, *tablerows]))
    return {"title": title, "text": combined}
def extractpdf(rawpath: Path) -> dict:
    try:
        reader = PdfReader(str(rawpath))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return {"title": rawpath.stem, "text": text}
    except Exception as e:
        return {"title": rawpath.stem, "text": "", "error": str(e)}





def infersection(url: str) -> str:
    from bkend.categories import sections  # local import, avoid circular load
    urllower = url.lower()
    for s in sections:
        if any(hint in urllower for hint in s["urlhints"]):
            return s["id"]
    return "general"
def extractall() -> list[dict]:
    manifestpath = rawdir / "manifest.json"
    manifest = json.loads(manifestpath.read_text())
    out = []
    for url, meta in manifest.items():
        rawpath = Path(meta["rawpath"])
        if not rawpath.exists():
            continue
        parsed = extractpdf(rawpath) if meta["ctype"] == "pdf" else extracthtml(rawpath.read_bytes(), url)
        text = re.sub(r"\n{3,}", "\n\n", parsed["text"]).strip()
        if len(text) < 30:
            continue  #empty/boilerplate-onlypagetoskip
        doc = {
            "url": url,
            "title": parsed["title"],
            "text": text,
            "ctype": meta["ctype"],
            "chash": meta["chash"],
            "fetchedat": meta["fetchedat"],
            "fetchedatiso": datetime.fromtimestamp(meta["fetchedat"]).isoformat(),
            "datesfound": guessdates(text),
            "section": infersection(url),
        }
        out.append(doc)
    docspath.write_text("\n".join(json.dumps(d) for d in out))
    print(f"extracted {len(out)} clean docs -> {docspath}")
    return out
if __name__ == "__main__":
    extractall()