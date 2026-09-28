
"""
Hybrid retrieval: FAISS + BM25 + RRF + date decay + optional reranker.

#85 - DATE DECAY RUNS ON EVERY QUERY.
#86 - retrieve_multi() fuses RRF across multiple query variants + optional
      HyDE text, then reranks against the ORIGINAL query.
"""

import json
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from bkend import cfgstore


idxdir = Path(__file__).resolve().parent.parent / "data" / "index"

# Backwards compatibility for any older imports.
conffloor = 0.005

_URL_YMD = re.compile(r"/(20\d{2})(\d{2})(\d{2})")
_URL_YM = re.compile(r"/(20\d{2})/(\d{2})/")
_URL_Y = re.compile(r"/(20\d{2})/")


def url_date(url: str) -> datetime:
    m = _URL_YMD.search(url or "")
    if m:
        y, mo, d = m.groups()
        try:
            return datetime(int(y), int(mo), int(d), tzinfo=timezone.utc)
        except ValueError:
            pass

    m = _URL_YM.search(url or "")
    if m:
        y, mo = m.groups()
        try:
            return datetime(int(y), int(mo), 15, tzinfo=timezone.utc)
        except ValueError:
            pass

    m = _URL_Y.search(url or "")
    if m:
        return datetime(int(m.group(1)), 6, 15, tzinfo=timezone.utc)

    return datetime(2005, 1, 1, tzinfo=timezone.utc)


@dataclass
class Retrievedchunk:
    text: str
    url: str
    title: str
    section: str
    fetchedatiso: str
    datesfound: list
    score: float
    rrf_score: float = 0.0
    rerank_score: float = 0.0


_EMBED_CACHE = {}
_RERANK_CACHE = {}


def get_embed(name: str, device: str = "cpu"):
    key = (name, device)

    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]

    try:
        model = SentenceTransformer(name, device=device)
        print(f"[embed] loaded {name} on {device}")
    except Exception as e:
        print(f"[embed] failed to load on {device}: {e} -> CPU")
        model = SentenceTransformer(name, device="cpu")

    _EMBED_CACHE[key] = model
    return model


def get_rerank(name: str, device: str = "cpu"):
    if not name:
        return None

    key = (name, device)

    if key in _RERANK_CACHE:
        return _RERANK_CACHE[key]

    try:
        from sentence_transformers import CrossEncoder
        import torch

        requested_device = device

        if requested_device == "cuda" and not torch.cuda.is_available():
            print("[rerank] CUDA unavailable -> using CPU")
            requested_device = "cpu"

        print(f"[rerank] requested device: {requested_device}")

        reranker = CrossEncoder(
            name,
            device=requested_device
        )

        if requested_device == "cuda":
            reranker.model.to("cuda")

        actual_device = next(
            reranker.model.parameters()
        ).device

        print(f"[rerank] actual device: {actual_device}")

        if requested_device == "cuda" and actual_device.type != "cuda":
            raise RuntimeError(
                f"CUDA was requested but CrossEncoder is on {actual_device}"
            )

        _RERANK_CACHE[(name, requested_device)] = reranker

        print(f"[rerank] successfully loaded {name} on {actual_device}")

        return reranker

    except Exception as e:
        print(f"[rerank] failed to load {name} on {device}: {e}")

        if device != "cpu":
            try:
                from sentence_transformers import CrossEncoder

                print("[rerank] falling back to CPU")

                reranker = CrossEncoder(
                    name,
                    device="cpu"
                )

                _RERANK_CACHE[(name, "cpu")] = reranker

                print("[rerank] CPU fallback successful")

                return reranker

            except Exception as e2:
                print(
                    f"[rerank] CPU fallback failed: {e2}"
                )

        return None

def load_faiss(mode: str):
    index_path = idxdir / "faiss.index"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    base = faiss.read_index(str(index_path))
    print(f"[faiss] loaded {base.ntotal} vectors")

    if mode != "gpu":
        return base, None

    try:
        if not hasattr(faiss, "get_num_gpus"):
            raise RuntimeError("FAISS GPU support unavailable")

        if faiss.get_num_gpus() <= 0:
            raise RuntimeError("No FAISS GPU detected")

        res = faiss.StandardGpuResources()
        gpu = faiss.index_cpu_to_gpu(res, 0, base)

        print("[faiss] index promoted to GPU")
        return gpu, res

    except Exception as e:
        print(f"[faiss] GPU promotion failed: {e} -> CPU")
        return base, None


class Retriever:
    def __init__(self):
        self._loaded_key = None
        self.faiss_resources = None
        self._maybe_reload()

    def _maybe_reload(self):
        cfg = cfgstore.loadcfg()

        key = (
            cfg["model"]["embedding"],
            cfg["model"]["device"],
            cfg["model"]["faiss_mode"],
        )

        if key == self._loaded_key:
            return

        self._loaded_key = key

        embed_name = cfg["model"]["embedding"]
        device = cfg["model"]["device"]
        faiss_mode = cfg["model"]["faiss_mode"]

        self.model = get_embed(embed_name, device)
        self.faissidx, self.faiss_resources = load_faiss(faiss_mode)

        with open(idxdir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        with open(idxdir / "chunksmeta.json", encoding="utf-8") as f:
            self.chunksmeta = json.load(f)

        if self.faissidx.ntotal != len(self.chunksmeta):
            print("[WARNING] FAISS vector count does not match metadata count")

        print(f"[retriever] loaded {len(self.chunksmeta)} chunks")

    def tok(self, text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def vecsearch(self, query, k):
        qemb = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")

        scores, idxs = self.faissidx.search(qemb, k)

        return [
            (int(i), float(s))
            for i, s in zip(idxs[0], scores[0])
            if i != -1
        ]

    def bm25search(self, query, k):
        scores = self.bm25.get_scores(self.tok(query))
        topidx = np.argsort(scores)[::-1][:k]

        return [
            (int(i), float(scores[i]))
            for i in topidx
            if scores[i] > 0
        ]

    def is_freshness_query(self, query):
        # Kept for backwards compatibility.
        # Date decay is now applied to every query.
        q = query.lower()

        freshness_terms = [
            "latest",
            "current",
            "currently",
            "recent",
            "recently",
            "newest",
            "new",
            "ongoing",
            "still open",
            "still ongoing",
            "deadline",
            "last date",
            "closing date",
            "today",
            "this year",
            "this semester",
            "this session",
            "2025",
            "2026",
            "2027",
            "admission",
            "application",
            "registration",
            "notification",
            "notice",
        ]

        return any(term in q for term in freshness_terms)

    @staticmethod
    def _rerank_scores(reranker, pairs):
        scores = reranker.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        scores = np.asarray(scores).reshape(-1)

        if len(scores) != len(pairs):
            raise RuntimeError(
                f"Reranker returned {len(scores)} scores for "
                f"{len(pairs)} pairs"
            )

        return scores

    def _rerank_candidates(self, query, ranked_rrf, model_cfg):
        """
        Rerank candidates against the supplied query.
        Falls back cleanly to RRF ordering if reranking fails.
        """
        if not ranked_rrf:
            return []

        reranker_name = model_cfg.get("reranker", "")
        device = model_cfg.get("device", "cpu")

        reranker = get_rerank(reranker_name, device)

        if reranker is None:
            return [
                (idx, float(score), float(score))
                for idx, score in ranked_rrf
            ]

        try:
            pairs = [
                (query, self.chunksmeta[idx].get("text", "")[:1500])
                for idx, _ in ranked_rrf
            ]

            rerank_scores = self._rerank_scores(reranker, pairs)

            final_ranked = [
                (
                    idx,
                    float(rerank_score),
                    float(rrf_score),
                )
                for (idx, rrf_score), rerank_score
                in zip(ranked_rrf, rerank_scores)
            ]

            final_ranked.sort(key=lambda x: x[1], reverse=True)
            return final_ranked

        except Exception as e:
            print(f"[rerank] prediction failed: {e}")
            return [
                (idx, float(score), float(score))
                for idx, score in ranked_rrf
            ]

    def _apply_date_decay_and_section_boost(
        self,
        rrf,
        section,
        section_boost,
        date_half_life,
        date_decay_floor,
    ):
        now = datetime.now(timezone.utc)

        for idx in list(rrf.keys()):
            d = url_date(self.chunksmeta[idx].get("url", ""))
            age_days = max(0, (now - d).days)

            weight = 0.5 ** (
                age_days / max(1, date_half_life)
            )

            freshness_weight = (
                date_decay_floor
                + (1.0 - date_decay_floor) * weight
            )

            rrf[idx] *= freshness_weight

        if section and section_boost != 1.0:
            for idx in list(rrf.keys()):
                if self.chunksmeta[idx].get("section") == section:
                    rrf[idx] *= section_boost

    def _build_output(self, final_ranked):
        out = []

        for idx, rerank_score, rrf_score in final_ranked:
            m = self.chunksmeta[idx]

            out.append(
                Retrievedchunk(
                    text=m.get("text", ""),
                    url=m.get("url", ""),
                    title=m.get("title", ""),
                    section=m.get("section", ""),
                    fetchedatiso=m.get("fetchedatiso", ""),
                    datesfound=m.get("datesfound", []),
                    score=float(rerank_score),
                    rrf_score=float(rrf_score),
                    rerank_score=float(rerank_score),
                )
            )

        return out

    def retrieve(
        self,
        query,
        section: Optional[str] = None,
        topk: Optional[int] = None,
    ):
        self._maybe_reload()

        cfg = cfgstore.loadcfg()
        retrieval_cfg = cfg["retrieval"]
        model_cfg = cfg["model"]

        k_per = retrieval_cfg["topk_per_method"]
        rrfk = retrieval_cfg["rrf_k"]
        final_k = topk or retrieval_cfg["final_topk"]

        section_boost = retrieval_cfg.get("section_boost", 1.0)
        date_half_life = retrieval_cfg.get("date_half_life_days", 365)
        date_decay_floor = retrieval_cfg.get("date_decay_floor", 0.5)

        vec_hits = self.vecsearch(query, k_per)
        bm25_hits = self.bm25search(query, k_per)

        rrf = defaultdict(float)

        for rank, (idx, _) in enumerate(
            sorted(vec_hits, key=lambda x: -x[1])
        ):
            rrf[idx] += 1.0 / (rrfk + rank + 1)

        for rank, (idx, _) in enumerate(
            sorted(bm25_hits, key=lambda x: -x[1])
        ):
            rrf[idx] += 1.0 / (rrfk + rank + 1)

        self._apply_date_decay_and_section_boost(
            rrf,
            section,
            section_boost,
            date_half_life,
            date_decay_floor,
        )

        candidate_count = max(final_k * 10, k_per)

        ranked_rrf = sorted(
            rrf.items(),
            key=lambda x: -x[1],
        )[:candidate_count]

        final_ranked = self._rerank_candidates(
            query,
            ranked_rrf,
            model_cfg,
        )

        final_ranked = final_ranked[:final_k]

        return self._build_output(final_ranked)

    def retrieve_multi(
        self,
        queries,
        hyde="",
        section=None,
        topk=None,
    ):
        """
        Multi-query retrieval.

        Fuses FAISS + BM25 results from all query variants and optional
        HyDE text using RRF, applies date decay + section boost, then
        reranks the candidates against the ORIGINAL user query.
        """
        self._maybe_reload()

        cfg = cfgstore.loadcfg()
        rcfg = cfg["retrieval"]
        mcfg = cfg["model"]

        k_per = rcfg["topk_per_method"]
        rrfk = rcfg["rrf_k"]
        final_k = topk or rcfg["final_topk"]

        section_boost = rcfg.get("section_boost", 1.0)
        date_half_life = rcfg.get("date_half_life_days", 365)
        date_floor = rcfg.get("date_decay_floor", 0.5)

        if isinstance(queries, str):
            queries = [queries]

        queries = queries or []

        all_texts = [q.strip() for q in queries if q and q.strip()]

        if hyde and hyde.strip():
            all_texts.append(hyde.strip())

        if not all_texts:
            return []

        # Original user query is what the reranker should judge against.
        original_query = (
            queries[0].strip()
            if queries and queries[0] and queries[0].strip()
            else all_texts[0]
        )

        rrf = defaultdict(float)

        for q in all_texts:
            vec_hits = self.vecsearch(q, k_per)
            bm25_hits = self.bm25search(q, k_per)

            for rank, (idx, _) in enumerate(
                sorted(vec_hits, key=lambda x: -x[1])
            ):
                rrf[idx] += 1.0 / (rrfk + rank + 1)

            for rank, (idx, _) in enumerate(
                sorted(bm25_hits, key=lambda x: -x[1])
            ):
                rrf[idx] += 1.0 / (rrfk + rank + 1)

        self._apply_date_decay_and_section_boost(
            rrf,
            section,
            section_boost,
            date_half_life,
            date_floor,
        )

        candidate_count = max(final_k * 10, k_per)

        ranked_rrf = sorted(
            rrf.items(),
            key=lambda x: -x[1],
        )[:candidate_count]

        final_ranked = self._rerank_candidates(
            original_query,
            ranked_rrf,
            mcfg,
        )

        final_ranked = final_ranked[:final_k]

        # IMPORTANT:
        # This return was missing in your current retrieve_multi().
        # Without it Python returns None, causing:
        # TypeError: 'NoneType' object is not subscriptable
        return self._build_output(final_ranked)

    def topscore(self, chunks):
        if not chunks:
            return 0.0

        return float(chunks[0].score)
 