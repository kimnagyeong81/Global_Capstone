# build_index_from_parquet.py
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

PARQUET_PATH = "data/train-00000-of-00001.parquet"  # 네 파일 경로
TEXT_COL = "sentences"   # 메시지 본문 컬럼명
META_COLS = ["workspace", "channel", "user", "ts"]  # 있으면 메타데이터로

# 1) 읽기
df = pd.read_parquet(PARQUET_PATH)  # pyarrow 백엔드
df = df.dropna(subset=[TEXT_COL])
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df = df[df[TEXT_COL] != ""]

# 2) 임베딩 준비
embedder = SentenceTransformer("BAAI/bge-m3")  # 로컬 캐시 사용
client = PersistentClient(path="vectorstores/slack_bge_m3")
try:
    client.delete_collection("slack")
except Exception:
    pass
col = client.create_collection(name="slack", metadata={"hnsw:space": "cosine"})

# 3) 배치 업서트
BATCH = 128
ids, docs, metas, embs = [], [], [], []
for i, row in df.iterrows():
    ids.append(f"row-{i}")
    docs.append(row[TEXT_COL])
    metas.append({k: str(row[k]) for k in META_COLS if k in df.columns})

    if len(ids) == BATCH:
        vecs = embedder.encode(docs, normalize_embeddings=True).tolist()
        col.upsert(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)
        ids, docs, metas = [], [], []

# 남은 배치
if ids:
    vecs = embedder.encode(docs, normalize_embeddings=True).tolist()
    col.upsert(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)

print("✅ Parquet → Chroma 인덱스 완료: vectorstores/slack_bge_m3")
