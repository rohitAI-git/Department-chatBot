"""Config storage - extended with persona, tone, all tunables."""
import json
from pathlib import Path


cfgpath = Path(__file__).resolve().parent.parent / "data" / "cfg.json"
defaultcfg = {
    "enablelatestfetch": False,

    "model": {
        "llm": "qwen2.5:7b-instruct-q4_K_M",
        
        #"llm": "qwen2.5:3b-instruct",
        "llm_fallback": "qwen2.5:1.5b-instruct",
        # tweak 86: query rewriter. Leave as None to reuse "llm" above.
        "llm_expand": None,
        #"embedding": "BAAI/bge-small-en-v1.5",
        "embedding": "BAAI/bge-base-en-v1.5",
        # tweak 86: bigger reranker (278M) - real quality jump vs MiniLM-L6.
        # Downloads ~1GB on first use, then cached in HF hub.
        "reranker": "BAAI/bge-reranker-base",
        "faiss_mode": "gpu",
        "device": "cuda",
    },

    "retrieval": {
        "topk_per_method": 10,
        # tweak 86: feed the LLM more context. Small models handle 15 fine.
        "final_topk": 15,
        "max_llm_chunks": 15,
        "rrf_k": 40,
        "date_half_life_days": 120,
        "conf_floor": 0.005,
        "section_boost": 1.4,
        "date_decay_floor": 0.1,
        # tweak 86: turn the general multi-query + HyDE path on/off.
        "multiquery": True,
    },

    "generation": {
        "temperature": 0.2,
        "response_mode": "balanced",   # NEW ADDS - short/balanced/detailed
        "persona": "friendly",          # friendly/professional/concise/chill
        "max_tokens": 700,
    },
}
def _merge(base, override):
    out = json.loads(json.dumps(base))
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
def loadcfg() -> dict:
    if not cfgpath.exists():
        savecfg(defaultcfg)
        return json.loads(json.dumps(defaultcfg))
    try:
        raw = json.loads(cfgpath.read_text())
    except Exception:
        return json.loads(json.dumps(defaultcfg))
    return _merge(defaultcfg, raw)
def savecfg(cfg: dict) -> None:
    cfgpath.parent.mkdir(parents=True, exist_ok=True)
    cfgpath.write_text(json.dumps(cfg, indent=2))