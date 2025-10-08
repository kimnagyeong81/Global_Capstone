# pipelines/slack/ask_slack.py
import os, sys
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# --- 설정 (env 우선) ---
ROOT       = Path(__file__).resolve().parents[2]
DB_DIR     = Path(os.getenv("SLACK_DB_DIR", ROOT / "vectorstores" / "slack_bge_m3"))
COLLECTION = os.getenv("SLACK_COLLECTION", "slack")
EMB_MODEL  = os.getenv("SLACK_EMB_MODEL", "BAAI/bge-m3")
TOP_K      = int(os.getenv("SLACK_TOP_K", "6"))

# 전역 임베더 1회 로드
_EMB = SentenceTransformer(EMB_MODEL)

def _connect_collection():
    client = PersistentClient(path=str(DB_DIR))
    try:
        return client.get_collection(COLLECTION)
    except Exception as e:
        raise RuntimeError(f"Chroma collection '{COLLECTION}' not found at {DB_DIR}. "
                           f"인덱스가 생성됐는지 확인하세요.") from e

def _mmr_lite(docs: List[str], dists: List[float], keep: int = 6, overlap=0.6):
    """간단 중복 억제: 토큰 bag 겹침이 큰 문서 제거."""
    pairs = list(zip(docs, dists))
    pairs.sort(key=lambda x: x[1])  # 거리↑(유사도↓)이므로 오름차순=가까운 것 먼저
    chosen: List[Tuple[str, float]] = []
    for doc, dist in pairs:
        ws = set(doc.split())
        if all(len(ws & set(x.split())) / max(1, len(ws)) < overlap for x, _ in chosen):
            chosen.append((doc, dist))
        if len(chosen) >= keep:
            break
    return [c[0] for c in chosen], [c[1] for c in chosen]

def search(query: str, top_k: int = TOP_K, mmr: bool = True):
    col = _connect_collection()
    qvec = _EMB.encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=qvec, n_results=top_k,
                    include=["documents", "metadatas", "distances"])
    if not res or not res.get("documents") or not res["documents"][0]:
        return []

    docs  = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    if mmr:
        docs2, dists2 = _mmr_lite(docs, dists, keep=min(6, top_k))
        # mmr로 추린 문서의 메타 재매핑
        keep_set = set(docs2)
        docs, metas, dists = [], [], []
        for d, m, s in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if d in keep_set:
                docs.append(d); metas.append(m); dists.append(s)
        # 순서 보장
        order = {d:i for i,d in enumerate(docs2)}
        zipped = sorted(zip(docs, metas, dists), key=lambda x: order.get(x[0], 1e9))
        return zipped
    else:
        return list(zip(docs, metas, dists))

def format_hit(doc, meta, dist):
    ws   = meta.get("workspace") or "-"
    ch   = meta.get("channel") or "-"
    user = meta.get("user") or "-"
    ts   = meta.get("ts") or "-"
    head = f"[{ws}#{ch} | @{user} | {ts}] (score={1-float(dist):.3f})"
    body = doc.replace("\n", " ").strip()
    if len(body) > 300:
        body = body[:300] + " ..."
    return head + "\n" + body + "\n"

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "virtualenv와 conda 같이 쓰는 법"
    try:
        hits = search(query, top_k=TOP_K, mmr=True)
        if not hits:
            print("결과가 없습니다. 인덱스 또는 컬렉션 설정을 확인하세요.")
        else:
            print("\n=== Top Results ===")
            for doc, meta, dist in hits:
                print(format_hit(doc, meta, dist))
    except Exception as e:
        print("[ERROR]", e)
