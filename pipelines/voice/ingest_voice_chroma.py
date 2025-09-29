# voice/ingest_voice_chroma.py
from pathlib import Path
import json
from typing import List
from langchain_community.embeddings import OllamaEmbeddings
# ⬇️ 권고대로 최신 패키지 사용
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

BASE_URL   = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"
DATA_FILE   = Path("data/voice_transcripts/transcripts.jsonl")
PERSIST_DIR = "vectorstores/voices_bge_m3"
COLLECTION  = "voices_bge_m3"

def load_docs() -> List[Document]:
    docs: List[Document] = []
    if not DATA_FILE.exists():
        print(f"❗ 파일이 없습니다: {DATA_FILE}")
        return docs
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = (r.get("text") or "").strip()
            if not text:
                continue  # ⬅️ 빈 텍스트 스킵
            meta = {
                "doc_id": r.get("doc_id"),
                "source": r.get("source", "voice"),
                "title": r.get("title"),
                "created_at": r.get("created_at"),
            }
            docs.append(Document(page_content=text, metadata=meta))
    return docs

def main():
    print("🔧 load voice docs...")
    raw_docs = load_docs()
    # 빈 경우 안전 종료
    if not raw_docs:
        print("❗ 유효한 보이스 문서가 없습니다. STT 결과를 확인하세요.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=128, separators=["\n\n", "\n", " ", ""]
    )
    chunks = [d for d in splitter.split_documents(raw_docs) if d.page_content.strip()]
    if not chunks:
        print("❗ 보이스 청크가 생성되지 않았습니다. 텍스트가 너무 짧거나 비어 있을 수 있어요.")
        return

    embeddings = OllamaEmbeddings(base_url=BASE_URL, model=EMBED_MODEL)
    vs = Chroma(collection_name=COLLECTION, embedding_function=embeddings, persist_directory=PERSIST_DIR)
    vs.add_documents(chunks)
    print(f"✅ indexed {len(chunks)} chunks → {PERSIST_DIR} ({COLLECTION})")

if __name__ == "__main__":
    main()
