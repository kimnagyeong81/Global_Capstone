import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document

BASE_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:7b-instruct"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "Emails" / "fake_emails_2025.json"
PERSIST_DIR = PROJECT_ROOT / "vectorstores" / "emails_bge_m3"
COLLECTION = "emails_bge_m3"

TOP_K = 5
MAX_CTX_CHARS = 15000


def load_emails():
    if not DATA_PATH.exists():
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_email_index():
    emails = load_emails()
    if not emails:
        print("❌ 이메일 데이터 없음")
        return

    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db = Chroma(collection_name=COLLECTION,
                persist_directory=str(PERSIST_DIR),
                embedding_function=emb)

    docs = []
    for e in emails:
        text = (
            f"From: {e['from_name']} → To: {e['to_name']}\n"
            f"Date: {e['date']}\n"
            f"Subject: {e['subject']}\n\n{e['body']}"
        )
        docs.append(Document(
            page_content=text,
            metadata={"source": f"email_{e['id']}", "category": e["category"]}
        ))

    db.add_documents(docs)
    print(f"✅ Email 인덱싱 완료 ({len(docs)}개)")


def ask_email(query: str, mode="ask") -> str:
    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db = Chroma(collection_name=COLLECTION,
                persist_directory=str(PERSIST_DIR),
                embedding_function=emb)

    results = db.similarity_search(query, k=TOP_K)
    ctx = "\n\n---\n\n".join(r.page_content for r in results)[:MAX_CTX_CHARS]

    prompt = f"""
당신은 이메일 분석 도우미입니다.
아래 이메일 내용만 근거로 답하세요.

[질문]
{query}

[이메일]
{ctx}
"""

    llm = OllamaLLM(model=LLM_MODEL, base_url=BASE_URL, temperature=0.2)
    return llm.invoke(prompt)
