# pipelines/slack/ask_slack.py
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

ROOT = Path(__file__).resolve().parents[2]
DB_DIR = ROOT / "vectorstores" / "slack_bge_m3"

EMB_MODEL = "BAAI/bge-m3"
TOP_K = 6

def search(query: str):
    emb = SentenceTransformer(EMB_MODEL)
    qvec = emb.encode([query], normalize_embeddings=True).tolist()
    col = PersistentClient(path=str(DB_DIR)).get_collection("slack")
    res = col.query(query_embeddings=qvec, n_results=TOP_K,
                    include=["documents", "metadatas", "distances"])
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return list(zip(docs, metas, dists))

def format_hit(doc, meta, dist):
    ws = meta.get("workspace")
    ch = meta.get("channel")
    user = meta.get("user")
    ts  = meta.get("ts")
    head = f"[{ws or '-'}#{ch or '-'} | @{user or '-'} | {ts or '-'}] (score={1-dist:.3f})"
    body = doc.replace("\n", " ")[:300]
    return head + "\n" + body + ("\n" if len(doc) <= 300 else " ...\n")

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "virtualenv와 conda 같이 쓰는 법"
    hits = search(query)
    print("\n=== Top Results ===")
    for doc, meta, dist in hits:
        print(format_hit(doc, meta, dist))

