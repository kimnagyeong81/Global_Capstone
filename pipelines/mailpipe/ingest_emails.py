# ingest_emails.py
from pathlib import Path
from typing import List
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma                      # ✅ 여기로
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pipelines.common.preprocess import clean_email


# ===== 설정 =====
BASE_URL    = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"                      # 1024차원
DATA_DIR    = Path("Data/Emails/raw")       # .txt 파일들
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
        raw = read_txt(p)
        txt = clean_email(raw, pii=True, remove_html=True, drop_quotes=True, drop_signature=True)
        # 필요하면 기존 간단 클린도 이어서 적용
        # txt = clean_text(txt)
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
    # ✅ 빈 청크 제거
    chunks = [d for d in splitter.split_documents(raw_docs) if d.page_content.strip()]
    print(f"[정보] 청크 수: {len(chunks)}")

    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)

    # ✅ 임베딩 사전 점검 (서버/모델 확인)
    try:
        test_vec = emb.embed_query("ping")
        assert isinstance(test_vec, list) and len(test_vec) > 0, "임베딩 결과가 비어 있음"
    except Exception as e:
        raise RuntimeError(
            "Ollama 임베딩 호출 실패. 다음을 확인하세요:\n"
            " - ollama 서버가 실행 중인지 (ollama serve)\n"
            " - 임베딩 모델이 설치되었는지 (ollama pull bge-m3)\n"
            f" - 에러: {e}"
        )

    # ✅ from_documents 대신 명시적으로 생성 후 add_documents
    vectordb = Chroma(
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=emb,
    )
    vectordb.add_documents(chunks)   # 내부에서 임베딩 호출
    print(f"[저장 완료] Chroma → {PERSIST_DIR} | collection='{COLLECTION}'")

if __name__ == "__main__":
    main()
