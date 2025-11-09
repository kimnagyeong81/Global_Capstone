import sys, os, re, unicodedata, json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document

# ===== 설정 =====
BASE_URL    = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"
LLM_MODEL   = "qwen2.5:7b-instruct"

DATA_PATH   = Path("C:\\Users\\jeong\\Desktop\\Global_Capstone\\Data\\Emails\\fake_emails_2025.json")
PERSIST_DIR = "vectorstores/emails_bge_m3"
COLLECTION  = "emails_bge_m3"

TOP_K       = 5
MAX_CTX_CHARS = 15000


# ===== 벡터DB 빌드 =====
def load_emails():
    if not DATA_PATH.exists():
        print(f"❌ 파일 없음: {DATA_PATH}")
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_vectorstore():
    emails = load_emails()
    if not emails:
        print("❗ 이메일 데이터가 없습니다.")
        return

    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db  = Chroma(collection_name=COLLECTION, persist_directory=PERSIST_DIR, embedding_function=emb)

    docs = []
    for e in emails:
        body = (
            f"From: {e['from_name']} → To: {e['to_name']} | Date: {e['date']} | "
            f"Subject: {e['subject']}\n\n{e['body']}"
        )
        docs.append(Document(page_content=body, metadata={"source": f"email_{e['id']}", "category": e["category"]}))
    db.add_documents(docs)
    print(f"✅ {len(docs)}개의 이메일을 인덱싱했습니다 → {PERSIST_DIR} ({COLLECTION})")


# ===== 질의응답 =====
def ask(query: str, mode="ask"):
    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db  = Chroma(collection_name=COLLECTION, persist_directory=PERSIST_DIR, embedding_function=emb)

    results = db.similarity_search(query, k=TOP_K)

    # 컨텍스트 구성 및 출처 수집
    context_parts = []
    sources = []
    for r in results:
        context_parts.append(r.page_content)
        src = r.metadata.get("source", "unknown")
        sources.append(src)
    context = "\n\n---\n\n".join(context_parts)[:MAX_CTX_CHARS]
    sources_str = ", ".join(sorted(set(sources)))

    # ===== 모드별 프롬프트 =====
    system_prompts = {
        "ask": """
당신은 이메일 분석 도우미입니다.
아래 이메일 데이터를 기반으로 사용자의 질문에 정확하고 간결하게 답하세요.
필요하면 이메일 본문 내용을 요약하고, 누가 보냈는지·언제 보냈는지도 함께 제시하세요.
""",
        "cs": """
당신은 고객센터 상담원입니다.
아래 이메일 데이터를 기반으로 고객 문의에 부드럽고 친절한 어조로 대답하세요.
가능하다면 고객이 바로 이해할 수 있도록 구체적인 문장으로 답하세요.
""",
        "audit": """
당신은 감사 담당자입니다.
아래 이메일 데이터를 분석하여 포함된 항목(날짜, 금액, 개인 정보 등)을 점검하고,
규정 준수 여부를 검토하듯이 객관적으로 요약하세요.
""",
        "dev": """
당신은 개발/QA 담당자입니다.
아래 이메일 내용을 기반으로 템플릿 구조, 문구 패턴, 자동발송 여부를 분석하세요.
중복, 일관성, 문구 오류 등도 함께 지적해 주세요.
"""
    }

    system = system_prompts.get(mode, system_prompts["ask"])

    llm = OllamaLLM(model=LLM_MODEL, base_url=BASE_URL, temperature=0.2)

    prompt = f"""{system}

[이메일 데이터]
{context}

[질문]
{query}
"""

    answer = llm.invoke(prompt)
    print("\n=== 답변 ===\n", answer)
    print(f"\n📎 관련 이메일: {sources_str}\n")


# ===== 실행 =====
if __name__ == "__main__":
    mode = input("모드 선택 (build / ask / cs / audit / dev): ").strip().lower()
    if mode == "build":
        build_vectorstore()
    else:
        query = input("질문을 입력하세요: ").strip()
        ask(query, mode)
