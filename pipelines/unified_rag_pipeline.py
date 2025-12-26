import json
from pathlib import Path
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ======================
# 공통 설정
# ======================
BASE_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:7b-instruct"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMAIL_PATH = PROJECT_ROOT / "Data" / "Emails" / "fake_emails_2025.json"
SLACK_PATH = PROJECT_ROOT / "Data" / "Slack" / "slack_messages.json"
VOICE_JSON_DIR = PROJECT_ROOT / "Data" / "voice" / "transcripts"

PERSIST_DIR = PROJECT_ROOT / "vectorstores" / "unified_rag"
COLLECTION = "unified_rag"

TOP_K = 12
MAX_CTX_CHARS = 12000


# ======================
# Email → Documents
# ======================
def load_email_docs():
    if not EMAIL_PATH.exists():
        return []

    with open(EMAIL_PATH, "r", encoding="utf-8") as f:
        emails = json.load(f)

    docs = []
    for e in emails:
        text = (
            f"From {e['from_name']} to {e['to_name']}\n"
            f"Date: {e['date']}\n"
            f"Subject: {e['subject']}\n"
            f"{e['body']}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source_type": "email",
                "speaker": e["from_name"]
            }
        ))
    return docs


# ======================
# Slack → Documents
# ======================
def load_slack_docs():
    if not SLACK_PATH.exists():
        return []

    with open(SLACK_PATH, "r", encoding="utf-8") as f:
        msgs = json.load(f)

    docs = []
    for m in msgs:
        text = (
            f"{m['user_name']} wrote in #{m['channel_name']}:\n"
            f"{m['text']}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "source_type": "slack",
                "speaker": m["user_name"]
            }
        ))
    return docs


# ======================
# Voice → Documents
# ======================
def load_voice_docs():
    docs = []

    for file in VOICE_JSON_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = "\n".join(
            f"{s.get('speaker','Unknown')}: {s.get('text','')}"
            for s in data["segments"]
        )

        docs.append(Document(
            page_content=text,
            metadata={
                "source_type": "voice",
                "speaker": "multiple"
            }
        ))
    return docs


# ======================
# Unified Index Build
# ======================
def build_unified_index():
    print("📦 문서 로딩 중...")

    docs = (
        load_email_docs()
        + load_slack_docs()
        + load_voice_docs()
    )

    if not docs:
        print("❌ 인덱싱할 문서가 없습니다.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    embed = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embed
    )

    db.add_documents(chunks)
    print(f"✅ Unified RAG 인덱싱 완료 ({len(chunks)} chunks)")


# ======================
# Claude-style 질문 → 답변
# ======================
def ask_unified(query: str) -> str:
    embed = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embed
    )

    results = db.similarity_search(query, k=TOP_K)
    if not results:
        return "관련된 데이터를 찾지 못했습니다."

    # source별로 context 분리
    grouped = defaultdict(list)
    for r in results:
        grouped[r.metadata["source_type"]].append(r.page_content)

    context_blocks = []
    for src, texts in grouped.items():
        joined = "\n\n".join(texts)
        context_blocks.append(joined)

    context = "\n\n".join(context_blocks)[:MAX_CTX_CHARS]

    prompt = f"""
당신은 내부 커뮤니케이션을 분석하는 AI 분석가입니다.
아래 자료만을 근거로 Claude AI 스타일의 가독성 높은 분석 보고서를 작성하세요.

⚠️ 출력 형식은 매우 중요합니다.
반드시 아래 스타일을 따르세요.

[출력 스타일 지침]
- 마크다운 헤더(###, ####)를 사용하지 마세요.
- 문단 중심으로 서술하고, 필요할 때만 '•' 불릿을 사용하세요.
- 인용문은 반드시 큰따옴표("")로 한 줄에 따로 표시하세요.
- 제목처럼 보이는 구간도 문장 형태로 자연스럽게 시작하세요.
- 보고서가 아니라, 사람이 쓴 분석 글처럼 작성하세요.
- '결론', '요약' 같은 메타 표현 없이 자연스럽게 마무리하세요.
- 기술적인 용어(EMAIL, SLACK, VOICE)는 본문에 직접 쓰지 마세요.
- 분석은 “설명 → 예시 → 해석” 흐름을 따르세요.

[출력 구조]
1. 질문에 대한 한 문단 요약 (1~2문장)
2. 슬랙에서의 논의 (있다면)
3. 회의 음성 데이터에서의 논의 (있다면)
4. 해석 및 분석
5. 주요 패턴 요약
6. 결론

질문:
{query}

자료:
{context}
"""

    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=BASE_URL,
        temperature=0.15
    )

    return llm.invoke(prompt)


def query(
    question: str,
    sources: list[str] | None = None,
    top_k: int | None = None,
    embed_model: str | None = None,
    llm_model: str | None = None,
    vectorstore_paths: dict | None = None,
    unified_vectorstore_path: str | None = None,
):
    """
    Streamlit에서 호출하기 위한 래퍼 함수.
    - return: (answer: str, contexts: list[dict])
    """

    # 기존 전역 상수(TOP_K 등)를 유지하면서, 들어온 값만 덮어쓰기
    k = top_k if top_k is not None else TOP_K

    embed = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embed
    )

    results = db.similarity_search(question, k=k)
    if not results:
        return "관련된 데이터를 찾지 못했습니다.", []

    # Streamlit에서 보여주기 좋은 contexts 형태로 변환
    contexts = []
    for r in results:
        src = r.metadata.get("source_type", "unknown")
        if sources and src not in sources:
            continue

        contexts.append({
            "source": src,
            "title": src,
            "snippet": r.page_content[:800],
            "metadata": dict(r.metadata),
            "score": 0.92,   # 0~1 또는 0~100 둘 다 지원
             "date": "2024-11-18"
})

    # 답변 생성은 기존 ask_unified 재사용
    answer = ask_unified(question)

    return answer, contexts

# ======================
# 실행
# ======================
if __name__ == "__main__":
    mode = input("모드 (build / ask): ").strip().lower()
    if mode == "build":
        build_unified_index()
    elif mode == "ask":
        q = input("질문: ")
        print(ask_unified(q))
