# pipelines/slack/ingest_slack_parquet.py
from pathlib import Path
import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "pipelines" / "slack" / "slack_messages.json"
DB_DIR = ROOT / "vectorstores" / "slack_bge_m3"
COLLECTION = "slack"

def main():
    print("[준비] Slack 메시지 로딩 중...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    emb = SentenceTransformer("BAAI/bge-m3")
    client = chromadb.Client(chromadb.config.Settings(
        persist_directory=str(DB_DIR),
        is_persistent=True
    ))
    collection = client.get_or_create_collection(COLLECTION)

    print(f"[정보] 메시지 수: {len(data)}")
    texts, metadatas, ids = [], [], []
    for i, msg in enumerate(tqdm(data)):
        text = msg.get("text", "").strip()
        if not text:
            continue
        meta = {
            "workspace": "Global_Capstone",
            "channel": msg.get("channel_name", "-"),
            "user": msg.get("user_name", "-"),
            "ts": msg.get("ts", "-"),
        }
        texts.append(text)
        metadatas.append(meta)
        ids.append(str(i))

    embeddings = emb.encode(texts, normalize_embeddings=True).tolist()
    collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
    print(f"[저장 완료] {DB_DIR} | collection='{COLLECTION}'")

if __name__ == "__main__":
    main()
