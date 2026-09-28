"""
Multi-query rewrite + HyDE.

The user asks one question. We rewrite it into 3 shorter search phrases
and one hypothetical answer paragraph, then feed all of them to the
retriever. This is how a general RAG answers "who teaches at dcsa",
"wifi form kaha hai", "library timings", "TPC contact" and "how many
courses does DCSA offer" through the SAME code path - no intents, no
categories, no hardcoded lists.

Every failure falls back to the raw query. Retrieval never blocks on
this file.
"""
import json
import httpx
from bkend import cfgstore
OLLAMA_CHAT = "http://localhost:11434/api/chat"

SYS = """You help a student search an official university website and its PDFs.
Given a student question, produce:
  "queries": 3 short search phrases (2-6 words each) that would likely appear
             on the official page or document that answers this question
  "hyde":    one plausible 2-3 sentence answer written in the same tone as an
             official departmental page would write it.
Output ONLY a JSON object with keys "queries" and "hyde". No markdown, no
explanation, no code fences.
"""
USER = "Question: {q}\n\nReturn only the JSON object."
def _call(model, question, timeout=25.0):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": USER.format(q=question)},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 300},
    }
    with httpx.Client(timeout=timeout) as c:
        r = c.post(OLLAMA_CHAT, json=payload)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()
def _parse(s):
    s = (s or "").strip()
    if s.startswith("```"):
        parts = s.split("```", 2)
        if len(parts) >= 2:
            s = parts[1]
        if s.startswith("json"):
            s = s[4:]
    a, b = s.find("{"), s.rfind("}")
    if a >= 0 and b > a:
        s = s[a:b + 1]
    try:
        return json.loads(s)
    except Exception:
        return None
def expand(query: str) -> dict:
    """Return {"queries": [...incl original...], "hyde": "..."}"""
    cfg = cfgstore.loadcfg()
    model = cfg["model"].get("llm_expand") or cfg["model"]["llm"]
    try:
        data = _parse(_call(model, query)) or {}
    except Exception as e:
        print(f"[expand] {e}")
        data = {}
    qs = [query]
    for q in data.get("queries", []) or []:
        if isinstance(q, str) and q.strip():
            qs.append(q.strip())
    seen, uniq = set(), []
    for q in qs:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(q)
    return {
        "queries": uniq[:4],
        "hyde": (data.get("hyde") or "").strip(),
    }