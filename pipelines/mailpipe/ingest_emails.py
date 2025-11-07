# ingest_emails.py
from pathlib import Path
from typing import List
import json
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pipelines.common.preprocess import clean_email


# ===== 설정 =====
JSON_PATH = Path(r"C:\\Users\\jeong\\Desktop\\Global_Capstone-Main\\Global_Capstone-Main\\pipelines\\mailpipe\\fake_emails_2025.json")
BASE_URL    = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"                      # 1024차원
PERSIST_DIR = "vectorstores/emails_bge_m3"  # 새 폴더
COLLECTION  = "emails_bge_m3"               # 새 컬렉션
# ===============


def load_docs() -> List[Document]:
    """JSON 이메일 데이터 로드 및 Document 변환"""
    docs: List[Document] = []

    print(f"[DEBUG] JSON 파일 경로: {JSON_PATH}")
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"❌ JSON 파일을 찾을 수 없습니다: {JSON_PATH}")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        emails = json.load(f)

    for e in emails:
        body = clean_email(e["body"], pii=True, remove_html=True, drop_quotes=True, drop_signature=True)
        content = f"Subject: {e['subject']}\n\n{body}"
        meta = {
            "source": f"email_{e['id']}",
            "from": e["from_email"],
            "to": e["to_email"],
            "category": e["category"],
            "date": e["date"],
        }
        docs.append(Document(page_content=content, metadata=meta))

    print(f"[DEBUG] JSON 로드 완료 — 문서 {len(docs)}개 변환됨")
    return docs


def main():
    print("[준비] JSON 로딩 중...")
    raw_docs = load_docs()
    print(f"[정보] 원문 문서 수: {len(raw_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = [d for d in splitter.split_documents(raw_docs) if d.page_content.strip()]
    print(f"[정보] 청크 수: {len(chunks)}")

    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)

    # ✅ 임베딩 사전 점검
    try:
        test_vec = emb.embed_query("ping")
        assert isinstance(test_vec, list) and len(test_vec) > 0, "임베딩 결과가 비어 있음"
    except Exception as e:
        raise RuntimeError(
            "Ollama 임베딩 호출 실패. 다음을 확인하세요:\n"
            " - ollama 서버 실행 여부 (ollama serve)\n"
            " - 모델 설치 여부 (ollama pull bge-m3)\n"
            f" - 에러: {e}"
        )

    # ✅ 벡터스토어 저장
    vectordb = Chroma(
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=emb,
    )
    vectordb.add_documents(chunks)
    print(f"[저장 완료] Chroma → {PERSIST_DIR} | collection='{COLLECTION}'")


if __name__ == "__main__":
    main()
