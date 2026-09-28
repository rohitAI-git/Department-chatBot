# V4  -  just run applaunchr.py, no need of unicorn cmd now


#CHANGE 44 - REMOVED AS OF NOW LANGCHAIN AND GRAPH, TRY SPLIT DIRECT TO LAMA MODEL, REMOVED PINECONE SWITCH TO FASISS, FREE TIER ISSUE. 
#44 - edtable settings inside now, with a simple ui
#43 - LANGCHAIN and langraph TEST FAILED, NOT WORTH, RAM SPIKE 8GB+ alone
#42 - REMOVED THE BASIC RETRIEV
#45 - REMOVED THE 3-SECTION CLARIFY BLOCK (was firing "which of these" on every real question)
#86 - GENERAL RETRIEVAL VIA MULTI-QUERY + HyDE (bkend/queryexpand.py +
#      Retriever.retrieve_multi). No intents, no categories. Single-query
#      retrieve() kept as fallback toggleable from cfg["retrieval"]["multiquery"].
#DCSA helpdesk FastAPI app - chat UI + settings + JSON API.
import re as _re
import time
from collections import OrderedDict
from typing import Optional
import httpx
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
)
import json as jsonlib
from pydantic import BaseModel
from bkend.categories import sections, sectionbyid
from bkend.retrieveonceall import Retriever
from bkend import llmsystem as llm
from bkend import cfgstore
from bkend import queryexpand
from scraper import appcrawler, extrct
from ingest import chunkidx
from contextlib import asynccontextmanager
#
# basic greeting/vague fast paths
_GREET = _re.compile(
    r'^\s*((hi+|hello+|hey+|hii+|yo|sup|namaste|'
    r'good\s*(morning|afternoon|evening)|greetings)\s+)*'
    r'(hi+|hello+|hey+|hii+|yo|sup|namaste|'
    r'good\s*(morning|afternoon|evening)|greetings)'
    r'\s*[!.,?]*\s*$', _re.I)
_VAGUE = _re.compile(
    r'^\s*(help|i\s*need\s*help|i\s*have\s*a\s*problem|need\s*help|'
    r'how\s*to\s*apply|fees|exam|result|need\s*certificate|'
    r"i\s*can'?t\s*login|i\s*cant\s*login|payment\s*issue|"
    r'my\s*attendance\s*is\s*wrong|who\s*can\s*help\s*me|'
    r'what\s*do\s*i\s*do\s*now|can\s*you\s*connect\s*me\s*to\s*someone|'
    r'i\s*need\s*urgent\s*help|who\s*do\s*i\s*contact|'
    r'who\s*should\s*i\s*contact|what\s*do\s*i\s*do)\s*[!.,?]*\s*$', _re.I)
GREET_REPLY = (
    "Hello, I'm the DCSA Helpdesk AI. Shoot about any query of DCSA courses, faculty, fees, "
    "exams, notices, contacts — I'll answer from official sources and show you where I got it."
)
VAGUE_REPLY = (
    "Sure — which of these is closest to what you need? "
    "Admissions, fees, exams, courses, faculty/contacts, notices, hostel, placements, or a specific department? "
    "Add a detail (your programme/semester) if you can."
)





#mainapp
app = FastAPI(title="dcsa helpdeskapi")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
retriever: Optional[Retriever] = None
@app.on_event("startup")
def loadidx():
    global retriever
    try:
        retriever = Retriever()
    except Exception as e:
        print(f"[startup] could not load index yet ({e}). run the offline pipeline, or use /settings.")
        retriever = None
def rerunpipeline():
    global retriever
    print("[pipeline] refresh triggered - crawling (this is slow, expect a minute or more)")
    appcrawler.crawl()
    extrct.extractall()
    chunkidx.buildidx()
    retriever = Retriever()
    print("[pipeline] refresh complete, index reloaded")
def rebuildindexonly():
    global retriever
    print("[pipeline] rebuilding index from existing raw files (skip crawl)...")
    extrct.extractall()
    chunkidx.buildidx()
    retriever = Retriever()
    print("[pipeline] index rebuilt, retriever reloaded")
# LRU response cache - built latest on 10th/9/26 - 
class Lrucache:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.store: OrderedDict[tuple, dict] = OrderedDict()
    def get(self, key):
        if key not in self.store:
            return None
        self.store.move_to_end(key)
        return self.store[key]
    def put(self, key, value):
        self.store[key] = value
        self.store.move_to_end(key)
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)
respcache = Lrucache()
def normq(q: str) -> str:
    return " ".join(q.lower().strip().split())
# schemas
class Askreq(BaseModel):
    query: str
    section: Optional[str] = None
    sessionid: Optional[str] = None
class Src(BaseModel):
    title: str
    url: str
    lastchecked: str
class Askresp(BaseModel):
    answer: str
    status: str
    sources: list[Src] = []
    sectionused: Optional[str] = None
    latencyms: int
# pipeline
def runpipeline(req: Askreq):
    cfg = cfgstore.loadcfg()
    maxllmchunks = cfg["retrieval"]["max_llm_chunks"]
    conf_floor   = cfg["retrieval"].get("conf_floor", 0.005)
    if cfg.get("enablelatestfetch"):
        print("[ask] 'enable latest fetch' is ON - refreshing before answering (slow by design)")
        rerunpipeline()
    q = req.query
    #fastpathgreetin
    if _GREET.match(q):
        yield {"stage": "final",
               "answer": GREET_REPLY,
               "status": "greeting", "sources": [], "sectionused": None}
        return
    #fastpathvagueoneshot
    if _VAGUE.match(q):
        yield {"stage": "final",
               "answer": VAGUE_REPLY,
               "status": "clarify", "sources": [], "sectionused": None}
        return
    if retriever is None:
        yield {"stage": "final",
               "answer": "The search index hasn't been built yet. Go to /settings and click 'Refresh data now', or run the pipeline commands manually.",
               "status": "no_match", "sources": [], "sectionused": req.section}
        return
    nq = normq(q)
    cachekey = (nq, req.section)
    cached = respcache.get(cachekey)
    if cached:
        yield {"stage": "cached", **cached}
        return
    if len(nq) < 4:
        result = {
            "answer": "Could you tell me a bit more about what you need help with? "
                      "(e.g. admissions, fees, exams, courses, hostel, placements, or a specific department contact)",
            "status": "clarify", "sources": [], "sectionused": req.section,
        }
        yield {"stage": "final", **result}
        return
    # --- tweak 86 --- general multi-query + HyDE retrieval.
    # expand() rewrites the question into 3 search phrases + one short
    # hypothetical answer. retrieve_multi() fuses FAISS + BM25 RRF
    # across all variants, applies date decay + section boost, then
    # reranks the pool against the ORIGINAL query. If multiquery is
    # turned off in cfg, we fall back to the single-query path.
    if cfg["retrieval"].get("multiquery", True):
        try:
            exp = queryexpand.expand(q)
            print(f"[ask] expanded -> {exp['queries']}")
            chunks = retriever.retrieve_multi(
                exp["queries"], hyde=exp["hyde"], section=req.section,
            )
        except Exception as e:
            print(f"[ask] multiquery failed ({e}) -> single-query fallback")
            chunks = retriever.retrieve(q, section=req.section)
    else:
        chunks = retriever.retrieve(q, section=req.section)
    topscore = retriever.topscore(chunks)
    llmchunks = chunks[:maxllmchunks]
    yield {"stage": "searching",
           "sources": [{"title": c.title or c.url, "url": c.url, "lastchecked": c.fetchedatiso}
                       for c in llmchunks[:5]]}
    #NOTE - REMOVED THE 3-SECTION CLARIFY BLOCK. IT WAS FIRING "WHICH OF THESE" ON
    #       EVERY REAL QUESTION AND THE USER HATES IT. JUST ANSWER.
    if topscore < conf_floor or not chunks:
        result = {"answer": llm.nomatchmsg, "status": "no_match",
                  "sources": [], "sectionused": req.section}
    else:
        answer = llm.genanswer(q, llmchunks)
        seen, srcs = set(), []
        for c in llmchunks:
            u = (c.url or "").split("?")[0]
            if u in seen:
                continue
            seen.add(u)
            srcs.append({"title": c.title or c.url, "url": c.url, "lastchecked": c.fetchedatiso})
            if len(srcs) >= 4:
                break
        result = {"answer": answer, "status": "answered",
                  "sources": srcs, "sectionused": req.section}
    respcache.put(cachekey, result)
    yield {"stage": "final", **result}
#allfastendpointz
@app.get("/sections")
def getsections():
    return [{"id": s["id"], "label": s["label"]} for s in sections]
@app.get("/health")
def health():
    ollamaok = True
    try:
        httpx.get("http://localhost:11434/api/tags", timeout=3.0)
    except Exception:
        ollamaok = False
    return {
        "indexloaded": retriever is not None,
        "chunksindexed": len(retriever.chunksmeta) if retriever else 0,
        "ollamareachable": ollamaok,
        "enablelatestfetch": cfgstore.loadcfg().get("enablelatestfetch"),
    }
@app.post("/ask", response_model=Askresp)
def ask(req: Askreq):
    start = time.time()
    final = None
    for event in runpipeline(req):
        if event["stage"] in ("final", "cached"):
            final = event
    final.pop("stage", None)
    return Askresp(**final, latencyms=int((time.time() - start) * 1000))
@app.get("/asklive")
def asklive(query: str, section: Optional[str] = None):
    req = Askreq(query=query, section=section)
    def gen():
        for event in runpipeline(req):
            yield f"data: {jsonlib.dumps(event)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
@app.get("/api/settings")
def api_get_settings():
    return cfgstore.loadcfg()
@app.post("/api/settings")
async def api_set_settings(request: Request):
    body = await request.json()
    cfg = cfgstore.loadcfg()
    def merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v
    merge(cfg, body)
    cfgstore.savecfg(cfg)
    return {"ok": True, "cfg": cfg}
@app.post("/api/reset")
def api_reset():
    return {"ok": True}
# FRONTEND ROUTES
from pathlib import Path
from fastapi.responses import FileResponse
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent
# MAIN CHAT PAGE
@app.get("/")
async def home():
    return FileResponse(
        FRONTEND_DIR / "index.html"
    )
# SETTINGS PAGE
@app.get("/settings")
async def settingspage():
    return FileResponse(
        FRONTEND_DIR / "settings.html"
    )
# SHARED CSS
@app.get("/style.css")
async def stylesheet():
    return FileResponse(
        FRONTEND_DIR / "style.css",
        media_type="text/css"
    )
# LEGACY SETTINGS SAVE
@app.post("/settings")
async def savesettings_legacy(
    enablelatestfetch: str = Form(...)
):
    try:
        cfg = cfgstore.loadcfg()
        cfg["enablelatestfetch"] = (
            enablelatestfetch == "yes"
        )
        cfgstore.savecfg(cfg)
        return RedirectResponse(
            url="/settings",
            status_code=303
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Failed to save settings",
                "detail": str(e)
            }
        )

# REFRESH DATA
@app.post("/settings/refresh")
async def refreshnow():
    try:
        rerunpipeline()
        return RedirectResponse(
            url="/settings",
            status_code=303
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Failed to refresh pipeline",
                "detail": str(e)
            }
        )
# REBUILD INDEX
@app.post("/settings/rebuild")
async def rebuildonly():
    try:
        rebuildindexonly()
        return RedirectResponse(
            url="/settings",
            status_code=303
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Failed to rebuild index",
                "detail": str(e)
            }
        )
# 404
@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Not Found",
            "message": "The requested page or resource was not found.",
            "path": request.url.path
        }
    )
# 403
@app.exception_handler(403)
async def forbidden(request: Request, exc):
    return JSONResponse(
        status_code=403,
        content={
            "success": False,
            "error": "Forbidden",
            "message": "You do not have permission to access this resource.",
            "path": request.url.path
        }
    )
# 500
@app.exception_handler(500)
async def internal_server_error(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal Server Error",
            "message": "An unexpected error occurred on the server.",
            "path": request.url.path
        }
    )