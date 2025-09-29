# pipelines/slack/ingest_slack_parquet.py
import os
from pathlib import Path 
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# === 경로 설정 ===
ROOT = Path(__file__).resolve().parents[2]   # project/
DATA_DIR = ROOT / "data" / "slack"
PARQUET_GLOB = "train-*.parquet"             # 예: train-00000-of-00001.parquet
DB_DIR = ROOT / "vectorstores" / "slack_bge_m3"

TEXT_COL = "sentences"                        # 메시지 본문 컬럼명
META_COLS = ["workspace", "channel", "user", "ts"]  # 있으면 자동 메타데이터로

EMB_MODEL = os.getenv("SLACK_EMB_MODEL", "BAAI/bge-m3")
BATCH = int(os.getenv("SLACK_INGEST_BATCH", "128"))

def _load_df() -> pd.DataFrame:
    files: List[Path] = sorted(DATA_DIR.glob(PARQUET_GLOB))
    if not files:
        raise FileNotFoundError(f"No parquet files under {DATA_DIR} (pattern={PARQUET_GLOB})")
    dfs = [pd.read_parquet(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)
    if TEXT_COL not in df.columns:
        raise KeyError(f"Column '{TEXT_COL}' not found. Columns={list(df.columns)[:10]}...")
    # 클렌징
    df = df.dropna(subset=[TEXT_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL] != ""]
    # 메타 컬럼 존재만 남기기
    keep_meta = [c for c in META_COLS if c in df.columns]
    return df[[TEXT_COL] + keep_meta]

def main():
    print(f"[Ingest] loading parquet from {DATA_DIR} ...")
    df = _load_df()
    print(f"[Ingest] rows={len(df):,}")

    print(f"[Ingest] embedding model: {EMB_MODEL}")
    embedder = SentenceTransformer(EMB_MODEL)

    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(DB_DIR))
    # 기존 컬렉션 정리
    try:
        client.delete_collection("slack")
    except Exception:
        pass
    col = client.create_collection(name="slack", metadata={"hnsw:space": "cosine"})

    # 업서트
    ids, docs, metas = [], [], []
    for i, row in df.iterrows():
        ids.append(f"slack-{i}")
        docs.append(row[TEXT_COL])
        metas.append({k: (None if pd.isna(row[k]) else str(row[k]))
                      for k in df.columns if k != TEXT_COL})

        if len(ids) >= BATCH:
            vecs = embedder.encode(docs, normalize_embeddings=True).tolist()
            col.upsert(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)
            ids, docs, metas = [], [], []

    if ids:
        vecs = embedder.encode(docs, normalize_embeddings=True).tolist()
        col.upsert(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)

    print(f"✅ Done. Index at {DB_DIR}")

if __name__ == "__main__":
    main()

