"""6회차: 문서 분할(Chunking)·인덱스·검색·검색 품질 측정 공통 모듈.

index_builder.py(인덱스 만들기), rag.py(검색 기반 생성), benchmark.py --mode rag가 함께 쓴다.

인덱스 저장 위치: index/<도메인>_<분할전략>/   (Network Volume에 있으므로 Pod를 꺼도 남는다)
백엔드: config.VECTOR_DB = "chroma" | "faiss" | "numpy"
  chroma  벡터DB(SQLite 기반 파일). 실무형 도구 체험용
  faiss   파일 하나짜리 인덱스. chroma가 네트워크 볼륨에서 문제를 일으키면 이것으로 바꾼다
  numpy   의존성 없는 확인용(전수 내적)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import config
import utils


# ---------------------------------------------------------------- 코퍼스·분할
def load_corpus(domain: str | None = None, with_updates: bool = False) -> list[dict]:
    """with_updates=True면 corpus_updates/(기준일 뒤에 나온 문서)도 넣음. 6회차 '새 문서가 나왔다' 사건"""
    ddir = config.data_dir(domain)
    manifest = utils.read_json(ddir / "corpus_manifest.json")
    upd = ddir / "corpus_updates" / "updates_manifest.json"
    if with_updates and upd.exists():
        manifest = manifest + utils.read_json(upd)
    docs = []
    for m in manifest:
        text = (ddir / m["file"]).read_text(encoding="utf-8")
        docs.append({"doc_id": m["doc_id"], "title": m["title"], "text": text})
    return docs


def chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    """글자 수 기준으로 자른다. 문장 중간이 끊길 수 있어 overlap으로 앞 조각 끝을 겹친다."""
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step) if text[i:i + size].strip()]


_SENT = re.compile(r"(?<=[.!?])\s+")


def chunk_sentence(text: str, max_chars: int) -> list[str]:
    """문장·소제목 경계로 나눈 뒤 max_chars까지 이어 붙인다.
    조각마다 문서 제목을 앞에 붙여, 조각만 떼어 봐도 무엇에 관한 내용인지 알 수 있게 한다."""
    lines = [ln.strip() for ln in text.splitlines()]
    title = next((ln.lstrip("# ").strip() for ln in lines if ln.startswith("# ")), "")
    units: list[tuple[str, str]] = []
    for ln in lines:
        if not ln or ln.startswith("# "):
            continue
        if ln.startswith("#"):
            units.append(("H", f"({ln.lstrip('# ').strip()})"))
            continue
        units += [("S", u.strip()) for u in _SENT.split(ln) if u.strip()]
    chunks, cur = [], ""
    for kind, u in units:
        if kind == "H" and len(cur) >= max_chars * 0.4:      # 소제목에서 끊되, 너무 작은 조각은 만들지 않음
            chunks.append(cur)
            cur = ""
        if cur and len(cur) + 1 + len(u) > max_chars:
            chunks.append(cur)
            cur = ""
        cur = f"{cur} {u}".strip()
    if cur:
        chunks.append(cur)
    return [f"[{title}] {c}" if title else c for c in chunks]


def make_chunks(docs: list[dict], strategy: str) -> list[dict]:
    p = config.CHUNK[strategy]
    out = []
    for d in docs:
        pieces = chunk_fixed(d["text"], p["size"], p["overlap"]) if strategy == "fixed" else chunk_sentence(d["text"], p["max_chars"])
        for i, t in enumerate(pieces):
            out.append({"chunk_id": f"{d['doc_id']}#{i}", "doc_id": d["doc_id"], "title": d["title"], "text": t})
    return out


def chunk_stats(chunks: list[dict]) -> dict:
    lens = sorted(len(c["text"]) for c in chunks)
    return {"n_chunks": len(lens), "min": lens[0], "median": lens[len(lens) // 2], "max": lens[-1],
            "mean": round(sum(lens) / len(lens), 1), "n_docs": len({c["doc_id"] for c in chunks})}


# ---------------------------------------------------------------- 인덱스
def index_dir(strategy: str, domain: str | None = None) -> Path:
    return config.PATHS["index"] / f"{domain or config.DOMAIN}_{strategy}"


def build_index(chunks: list[dict], embedder, strategy: str, backend: str | None = None,
                domain: str | None = None) -> Path:
    backend = backend or config.VECTOR_DB
    out = index_dir(strategy, domain)
    out.mkdir(parents=True, exist_ok=True)
    texts = [c["text"] for c in chunks]
    with utils.Timer() as t:
        emb = embedder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")
    meta = {"backend": backend, "strategy": strategy, "embed_model": config.EMBED_MODEL, "n": len(chunks),
            "dim": int(emb.shape[1]), "embed_sec": round(t.sec, 2), "built_at": utils.now()}
    utils.write_jsonl(out / "chunks.jsonl", chunks)
    if backend == "chroma":
        import chromadb
        client = chromadb.PersistentClient(path=str(out / "chroma"))
        try:
            client.delete_collection("chunks")
        except Exception:
            pass
        col = client.create_collection("chunks", metadata={"hnsw:space": "cosine"})
        for i in range(0, len(chunks), 500):
            part = chunks[i:i + 500]
            col.add(ids=[c["chunk_id"] for c in part], embeddings=emb[i:i + 500].tolist(),
                    documents=[c["text"] for c in part],
                    metadatas=[{"doc_id": c["doc_id"], "title": c["title"]} for c in part])
    elif backend == "faiss":
        import faiss
        idx = faiss.IndexFlatIP(emb.shape[1])
        idx.add(emb)
        faiss.write_index(idx, str(out / "faiss.index"))
    elif backend == "numpy":
        np.save(out / "emb.npy", emb)
    else:
        raise ValueError(f"알 수 없는 VECTOR_DB: {backend}")
    utils.write_json(out / "meta.json", meta)
    return out


class Retriever:
    """index/<도메인>_<전략>/을 열어 질의와 가까운 조각을 찾는다."""

    def __init__(self, strategy: str | None = None, embedder=None, domain: str | None = None):
        self.strategy = strategy or config.CHUNK_STRATEGY
        self.dir = index_dir(self.strategy, domain)
        if not (self.dir / "meta.json").exists():
            raise FileNotFoundError(f"인덱스 없음: {self.dir} → python src/index_builder.py --strategy {self.strategy}")
        self.meta = utils.read_json(self.dir / "meta.json")
        self.chunks = utils.read_jsonl(self.dir / "chunks.jsonl")
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        if embedder is None:
            import model_utils
            embedder = model_utils.load_embedder(self.meta["embed_model"])
        self.embedder = embedder
        b = self.meta["backend"]
        if b == "chroma":
            import chromadb
            self.col = chromadb.PersistentClient(path=str(self.dir / "chroma")).get_collection("chunks")
        elif b == "faiss":
            import faiss
            self.index = faiss.read_index(str(self.dir / "faiss.index"))
        else:
            self.emb = np.load(self.dir / "emb.npy")

    def encode(self, query: str) -> np.ndarray:
        return np.asarray(self.embedder.encode([query], normalize_embeddings=True), dtype="float32")

    def search(self, query: str, k: int | None = None) -> list[dict]:
        k = k or config.RAG_TOP_K
        q = self.encode(query)
        b = self.meta["backend"]
        if b == "chroma":
            r = self.col.query(query_embeddings=q.tolist(), n_results=k)
            pairs = [(cid, 1 - d) for cid, d in zip(r["ids"][0], r["distances"][0])]
        elif b == "faiss":
            D, I = self.index.search(q, k)
            pairs = [(self.chunks[i]["chunk_id"], float(s)) for s, i in zip(D[0], I[0]) if i >= 0]
        else:
            s = self.emb @ q[0]
            top = np.argsort(-s)[:k]
            pairs = [(self.chunks[i]["chunk_id"], float(s[i])) for i in top]
        return [{**self.by_id[cid], "score": round(sc, 4), "rank": r + 1} for r, (cid, sc) in enumerate(pairs)]


# ---------------------------------------------------------------- 컨텍스트 조립
def format_context(hits: list[dict], tok=None, max_tokens: int | None = None) -> str:
    """검색 결과를 프롬프트에 넣을 문자열로 만든다. 토큰 상한을 넘으면 뒤쪽 조각부터 버린다."""
    max_tokens = max_tokens or config.CONTEXT_MAX_TOKENS
    parts, used = [], 0
    for h in hits:
        block = f"<{h['doc_id']}> {h['title']}\n{h['text']}"
        n = len(tok(block, add_special_tokens=False)["input_ids"]) if tok is not None else len(block) // 2
        if parts and used + n > max_tokens:
            break
        parts.append(block)
        used += n
    return "\n\n".join(parts)


# ---------------------------------------------------------------- 검색 품질
def doc_ranking(hits: list[dict]) -> list[str]:
    """조각 순위를 문서 순위로 바꾼다(같은 문서의 두 번째 조각부터는 건너뜀)."""
    seen, out = set(), []
    for h in hits:
        if h["doc_id"] not in seen:
            seen.add(h["doc_id"])
            out.append(h["doc_id"])
    return out


def eval_retrieval(retriever: Retriever, items: list[dict], ks=(1, 3, 5), depth: int = 20) -> dict:
    """gold_doc_ids가 있는 문항으로 Recall@k·MRR 계산. 정답 문서 중 하나라도 상위 k 안에 있으면 적중."""
    rows = []
    for it in items:
        gold = set(it.get("gold_doc_ids") or [])
        if not gold:
            continue
        ranking = doc_ranking(retriever.search(it["input"], depth))
        first = next((i + 1 for i, d in enumerate(ranking) if d in gold), None)
        rows.append({"id": it["id"], "type": it.get("type"), "first_hit": first, "top3": ranking[:3]})
    n = len(rows)
    if not n:
        return {"n": 0}
    res = {"n": n, "strategy": retriever.strategy}
    for k in ks:
        res[f"recall@{k}"] = round(sum(1 for r in rows if r["first_hit"] and r["first_hit"] <= k) / n, 4)
    res["mrr"] = round(sum(1 / r["first_hit"] for r in rows if r["first_hit"]) / n, 4)
    types = sorted({r["type"] for r in rows})
    res["recall@3_by_type"] = {t: round(sum(1 for r in rows if r["type"] == t and r["first_hit"] and r["first_hit"] <= 3)
                                        / sum(1 for r in rows if r["type"] == t), 4) for t in types}
    res["misses"] = [r for r in rows if not r["first_hit"] or r["first_hit"] > 3][:10]
    return res


if __name__ == "__main__":
    docs = load_corpus()
    for s in ("fixed", "sentence"):
        print(s, json.dumps(chunk_stats(make_chunks(docs, s)), ensure_ascii=False))
