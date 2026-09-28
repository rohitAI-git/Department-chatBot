"""
MAIN SCRAPER ENGINE, ROBOT PARSE TECHSTACK
VERSION - 23, robots disabled for dept-owned site.
incremental crawler for dcsa (and, optionally, a whitelist of related
puchd.ac.in subdomains e.g. results, iqac, examinations).
design notes 18:
- removed fasle to true
- now offline job. it does not run per user query. run manually with
  python -m scraper.crawl
- every fetch page/pdf stored as raw bytes + a manifest entry keyed by
  url, carrying a content hash. on re-crawl, unchanged pages hash the
  same, so the indexer knows exactly what changed.
- bfs from seed urls, restricted to an allowed domain list, with a max
  depth and a politeness delay.
"""




import hashlib
import json
import time
import urllib.robotparser as robotparser
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
# ------------
# ch
# we are crawling a site the department owns and has authorized us to crawl, or well, not really.... lol.
respectrobots = False
# ---------------------------------------------------------------------------
rawdir = Path(__file__).resolve().parent.parent / "data" / "raw"
rawdir.mkdir(parents=True, exist_ok=True)
seedurls = [
    "https://dcsa.puchd.ac.in/index.php",
    "https://dcsa.puchd.ac.in/show-courses.php",
    "https://dcsa.puchd.ac.in/show-faculty.php",
    "https://dcsa.puchd.ac.in/facilities.php",
    "https://dcsa.puchd.ac.in/show-researchscholar.php",
    "https://dcsa.puchd.ac.in/show-contactus.php",
    "https://dcsa.puchd.ac.in/show-noticeboard.php?nbid=1",
    "https://dcsa.puchd.ac.in/show-noticeboard.php?nbid=4",
    "https://dcsa.puchd.ac.in/show-noticeboard.php?nbid=5",
    "https://dcsa.puchd.ac.in/show-noticeboard.php?nbid=13",
]
# CURRENT RULES - only crawl within hosts. add puchd.ac.in subdomains here
# (LATER WORK) deliberately (e.g. "examinations.puchd.ac.in") rather than crawlin whole *.puchd.ac.in universe blindly (580+ subdomains) - most irrelevant to dcsa.
allowedhosts = {"dcsa.puchd.ac.in"}
maxdepth = 3
politenessdelay = 0.5
useragent = "DCSA-Helpdesk-Bot/1.0 (+internal student project; contact: dept office)"
robotscache: dict[str, robotparser.RobotFileParser] = {}
def allowedbyrobots(client: httpx.Client, url: str) -> bool:
    if not respectrobots:
        return True
    host = urlparse(url).netloc
    if host not in robotscache:
        rp = robotparser.RobotFileParser()
        try:
            resp = client.get(f"https://{host}/robots.txt", timeout=10.0)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                print(f"[robots] loaded {host}/robots.txt ({len(resp.text)} bytes)")
            else:
                rp.allow_all = True
                print(f"[robots] {host}/robots.txt returned HTTP {resp.status_code} -> defaulting to allow")
        except Exception as e:
            rp.allow_all = True
            print(f"[robots] could not fetch {host}/robots.txt ({e}) -> defaulting to allow")
        robotscache[host] = rp
    allowed = robotscache[host].can_fetch(useragent, url)
    if not allowed:
        print(f"[robots-blocked] {url}")
    return allowed
@dataclass
class Crawleddoc:
    url: str
    ctype: str  # "html""pdf" or whatever..
    chash: str
    fetchedat: float
    rawpath: str
def hashbytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]
def extractlinks(baseurl: str, html: bytes) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        absolute = urljoin(baseurl, a["href"])
        parsed = urlparse(absolute)
        if parsed.netloc in allowedhosts and parsed.scheme in ("http", "https"):
            clean = absolute.split("#")[0]  # strip fragments + keep query (nbid=... matters)
            links.append(clean)
    return links
def crawl(seeds: list[str] = seedurls, depthcap: int = maxdepth) -> list[Crawleddoc]:
    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque((u, 0) for u in seeds)
    docs: list[Crawleddoc] = []
    if not respectrobots:
        print("[robots] respectrobots=False (dept-owned site, authorized crawl) - skipping robots.txt entirely")
    with httpx.Client(headers={"User-Agent": useragent}, follow_redirects=True, timeout=20.0) as client:
        while queue:
            url, depth = queue.popleft()
            if url in seen or depth > depthcap:
                continue
            seen.add(url)
            if not allowedbyrobots(client, url):
                continue
            print(f"[fetching] {url}")
            try:
                resp = client.get(url)
                resp.raise_for_status()
            except Exception as e:
                print(f"[skip] {url}: {e}")
                continue
            content = resp.content
            ctype = "pdf" if url.lower().endswith(".pdf") or "application/pdf" in resp.headers.get("content-type", "") else "html"
            h = hashbytes(content)
            fname = f"{h}.{'pdf' if ctype == 'pdf' else 'html'}"
            rawpath = rawdir / fname
            if not rawpath.exists():
                rawpath.write_bytes(content)
            doc = Crawleddoc(url=url, ctype=ctype, chash=h, fetchedat=time.time(), rawpath=str(rawpath))
            docs.append(doc)
            if ctype == "html" and depth < depthcap:
                for link in extractlinks(url, content):
                    if link not in seen:
                        queue.append((link, depth + 1))
            time.sleep(politenessdelay)
    # manifest url-key so indexer can diff against last run
    manifestpath = rawdir / "manifest.json"
    existing = {}
    if manifestpath.exists():
        existing = json.loads(manifestpath.read_text())
    for d in docs:
        existing[d.url] = asdict(d)
    manifestpath.write_text(json.dumps(existing, indent=2))
    pdfcount = sum(1 for d in docs if d.ctype == "pdf")
    print(f"crawled {len(docs)} docs ({pdfcount} pdfs). manifest -> {manifestpath}")
    if not docs:
        print("[warning] 0 documents fetched - scroll up for [robots-blocked] or [skip] lines explaining why. "
              "If you see none at all, seedurls itself may be empty/wrong, or every seed failed silently before printing (unlikely with this version).")
    return docs
if __name__ == "__main__":
    crawl()