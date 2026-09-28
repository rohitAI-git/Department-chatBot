"""
#LAST CHANGE - REMOVED PINE + MedBM
chunks extracted docs embeds & builds two indexes:
#REMOVED PINECONE BS, FREE TIER SHIT
  - V3.2 - NEW IN - faiss (dense/vector): good semantic/paraphrased matches, NO NEED OF BERT/SBERT , TILL NOW
    ("how do i get my degree paper" / "provisional certificate")
  - bm25 (sparse/keyword): good for exact tokens (vector search is bad)
    at - phone numbers, course codes, notification numbers, names.
both queried at retrieval time & fused (retriever)
# why? - standard fix for the classic rag failure mode where 
semantic only index confidently retrieves the wrong phone number
because its topic wize similar
"""
"""Chunk + embed + build FAISS & BM25 indexes. GPU-aware, title-aware chunks.
#86 - FIXED TAIL BUG: bm25.pkl + chunksmeta.json writes were nested inside
      the CPU else-branch, so when faiss_mode="gpu" they were NEVER written.
      Moved them out so they run in both branches.
"""
import json
import os
import pickle
import re
from pathlib import Path
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from bkend import cfgstore
datadir  = Path(__file__).resolve().parent.parent / "data"
docspath = datadir / "extracteddocs.jsonl"
idxdir   = datadir / "index"
idxdir.mkdir(parents=True, exist_ok=True)
chunksize    = 260
chunkoverlap = 40
def chunktext(text, size=chunksize, overlap=chunkoverlap):
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks
def tokbm25(text):
    return re.findall(r"[a-z0-9]+", text.lower())
def _load_embedder(name, device):
    m = SentenceTransformer(name)
    if device == "cuda":
        try:
            m = m.to("cuda")
            print(f"[embed] using CUDA for {name}")
        except Exception as e:
            print(f"[embed] CUDA requested but unavailable ({e}) -> falling back to CPU")
    return m
def _maybe_gpu(index, mode):
    if mode != "gpu":
        return index
    try:
        if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() <= 0:
            print("[faiss] gpu mode requested but no CUDA device found -> using CPU")
            return index
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        print("[faiss] index moved to GPU")
        return gpu_index
    except Exception as e:
        print(f"[faiss] gpu build failed ({e}) -> using CPU")
        return index
def buildidx():
    cfg = cfgstore.loadcfg()
    embedmodel = cfg["model"]["embedding"]
    device     = cfg["model"]["device"]
    faiss_mode = cfg["model"]["faiss_mode"]
    docs = [json.loads(l) for l in docspath.read_text().splitlines() if l.strip()]
    chunks = []
    for doc in docs:
        title = (doc.get("title") or "").strip()
        for i, ctext in enumerate(chunktext(doc["text"])):
            ctx_text = f"{title}\n\n{ctext}" if title else ctext
            chunks.append({
                "id": f"{doc['chash']}_{i}",
                "url": doc["url"],
                "title": doc["title"],
                "section": doc["section"],
                "text": ctx_text,
                "datesfound": doc.get("datesfound", []),
                "fetchedat": doc["fetchedat"],
                "fetchedatiso": doc["fetchedatiso"],
            })
    print(f"{len(docs)} docs -> {len(chunks)} chunks")
    model = _load_embedder(embedmodel, device)
    texts = [c["text"] for c in chunks]
    embeds = model.encode(texts, batch_size=32, show_progress_bar=True,
                          normalize_embeddings=True).astype("float32")
    dim = embeds.shape[1]

    if faiss_mode == "gpu":
        try:
            if faiss.get_num_gpus() <= 0:
                raise RuntimeError("No FAISS GPU available")

            res = faiss.StandardGpuResources()

            gpu_index = faiss.GpuIndexFlatIP(res, dim)
            gpu_index.add(embeds)

            print("[faiss] embeddings indexed on GPU")

        # Convert GPU index back to CPU for permanent storage
            cpu_index = faiss.index_gpu_to_cpu(gpu_index)

            faiss.write_index(
                cpu_index,
                str(idxdir / "faiss.index")
            )

            print("[faiss] CPU copy saved to disk")

        except Exception as e:
            print(f"[faiss] GPU indexing failed: {e}")
            print("[faiss] Falling back to CPU")

            cpu_index = faiss.IndexFlatIP(dim)
            cpu_index.add(embeds)

            faiss.write_index(
                cpu_index,
                str(idxdir / "faiss.index")
            )

    else:
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(embeds)

        faiss.write_index(
            cpu_index,
            str(idxdir / "faiss.index")
        )

    # ---- tweak 86: these MUST run regardless of which branch wrote faiss ----
    bm25 = BM25Okapi([tokbm25(c["text"]) for c in chunks])
    with open(idxdir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(idxdir / "chunksmeta.json", "w") as f:
        json.dump(chunks, f)
    print(f"indexes written to {idxdir}")
if __name__ == "__main__":
    buildidx()