# ingest_faster.py
import os, glob
import concurrent.futures as cf
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from ollama import Client as OllamaClient

# ==== 설정 ====
DATA_DIR   = "data/emails/raw"
PERSIST_DIR= "vectorstores/emails_bge_m3"
COLLECTION = "emails_bge_m3"

EMBED_MODEL= "bge-m3:latest"      # 더 가볍게: "nomic-embed-text:latest"
BASE_URL   = "http://127.0.0.1:11434"

CHUNK_SIZE     = 1500
CHUNK_OVERLAP  = 100
BATCH          = 256              # add() 배치 크기
EMB_WORKERS    = 8                # 임베딩 병렬 스레드 수

def read_text(p: Path) -> str:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try: return p.read_text(encoding=enc, errors="ignore")
        except Exception: pass
    return p.read_text(errors="ignore")

def basic_clean(t: str) -> str:
    t = t.replace("\r", "")
    lines = [ln for ln in t.split("\n") if not ln.strip().startswith("> ")]
    t = "\n".join(lines)
    t = t.split("-----BEGIN PGP")[0]
    return t.strip()

def split_text(t: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    # 아주 단순한 문자 기준 슬라이싱 (langchain splitter보다 오버헤드↓)
    out, n = [], len(t)
    i = 0
    while i < n:
        j = min(n, i + size)
        out.append(t[i:j])
        if j == n: break
        i = j - overlap
        if i < 0: i = 0
    return out

def load_chunks() -> List[Dict]:
    files = sorted([Path(p) for p in glob.glob(os.path.join(DATA_DIR, "*.txt"))])
    rows = []
    for fp in files:
        raw = read_text(fp)
        cleaned = basic_clean(raw)
        chunks = split_text(cleaned)
        for idx, ch in enumerate(chunks):
            rows.append({
                "id": f"{fp.stem}_{idx}",
                "source": str(fp),
                "text": ch
            })
    return rows

def embed_batch_texts(client: OllamaClient, texts: List[str]) -> List[List[float]]:
    # Ollama Python 클라이언트는 단건 API라서 멀티스레드로 병렬 호출
    def one(t): return client.embeddings(model=EMBED_MODEL, prompt=t)["embedding"]
    vecs = []
    with cf.ThreadPoolExecutor(max_workers=EMB_WORKERS) as ex:
        for v in ex.map(one, texts):
            vecs.append(v)
    return vecs

def main():
    os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

    # 1) 데이터 로드 & 청크
    rows = load_chunks()
    print(f"[INFO] 총 청크 수: {len(rows)}")

    # 2) Chroma 준비
    client = chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    col = client.get_or_create_collection(COLLECTION)

    # 3) Ollama 클라이언트
    ollama = OllamaClient(host=BASE_URL)

    # 4) 배치 루프
    total = len(rows)
    for i in range(0, total, BATCH):
        batch = rows[i:i+BATCH]
        docs   = [r["text"] for r in batch]
        ids    = [r["id"]   for r in batch]
        metas  = [{"source": r["source"]} for r in batch]

        # 4-1) 병렬 임베딩
        vecs = embed_batch_texts(ollama, docs)

        # 4-2) 새 DB면 add()가 upsert()보다 빠름
        col.add(ids=ids, metadatas=metas, documents=docs, embeddings=vecs)

        if (i//BATCH) % 1 == 0:  # 매 배치마다 출력
            print(f" - add {min(i+BATCH, total)}/{total}")

    print("[DONE] 인덱싱 종료")

if __name__ == "__main__":
    main()
