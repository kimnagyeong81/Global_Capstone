import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document

BASE_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:7b-instruct"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "Slack" / "internal_slack_logs.json"
PERSIST_DIR = PROJECT_ROOT / "vectorstores" / "slack_bge_m3"
COLLECTION = "slack_bge_m3"

TOP_K = 10
MAX_CTX_CHARS = 15000


def load_slack():
    if not DATA_PATH.exists():
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_slack_index():
    msgs = load_slack()
    if not msgs:
        print("❌ Slack 데이터 없음")
        return

    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db = Chroma(collection_name=COLLECTION,
                persist_directory=str(PERSIST_DIR),
                embedding_function=emb)

    docs = []
    for m in msgs:
        text = (
            f"[채널] {m['channel_name']}\n"
            f"[사용자] {m['user_name']}\n"
            f"[시간] {m['ts']}\n"
            f"[내용] {m['text']}"
        )
        docs.append(Document(
            page_content=text,
            metadata={"source": f"slack_{m['id']}"}
        ))

    db.add_documents(docs)
    print(f"✅ Slack 인덱싱 완료 ({len(docs)}개)")


def ask_slack(query: str) -> str:
    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db = Chroma(collection_name=COLLECTION,
                persist_directory=str(PERSIST_DIR),
                embedding_function=emb)

    results = db.similarity_search(query, k=TOP_K)
    ctx = "\n\n---\n\n".join(r.page_content for r in results)[:MAX_CTX_CHARS]

    prompt = f"""
당신은 Slack 로그 분석 어시스턴트입니다.
아래 메시지 내용만 근거로 답하세요.

[질문]
{query}

[Slack]
{ctx}
"""

    llm = OllamaLLM(model=LLM_MODEL, base_url=BASE_URL, temperature=0.2)
    return llm.invoke(prompt)
