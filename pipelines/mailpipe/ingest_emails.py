# ingest_emails.py
from pathlib import Path
from typing import List
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ===== 설정 =====
BASE_URL    = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"                      # 1024차원
DATA_DIR    = Path("data/emails/raw")       # .txt 파일들
PERSIST_DIR = "vectorstores/emails_bge_m3"  # 새 폴더
COLLECTION  = "emails_bge_m3"               # 새 컬렉션
# ===============

def clean_text(t: str) -> str:
    t = t.replace("\r", "")
    # 인용/서명/PGP 같은 잡음 가볍게 제거
    lines = [ln for ln in t.split("\n") if not ln.strip().startswith("> ")]
    t = "\n".join(lines)
    t = t.split("-----BEGIN PGP")[0]
    return t.strip()

def read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")

def load_docs() -> List[Document]:
    docs: List[Document] = []
    for p in DATA_DIR.rglob("*.txt"):
        txt = clean_text(read_txt(p))
        if txt.strip():
            docs.append(Document(page_content=txt, metadata={"source": str(p)}))
    return docs

def main():
    print("[준비] 로딩 중...")
    raw_docs = load_docs()
    print(f"[정보] 원문 문서 수: {len(raw_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[정보] 청크 수: {len(chunks)}")

    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=emb,                 # (구버전) embedding 파라미터
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
    )
    vectordb.persist()
    print(f"[저장 완료] Chroma → {PERSIST_DIR} | collection='{COLLECTION}'")

if __name__ == "__main__":
    main()
