import sys, os, json, re
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document

# ===== 설정 =====
BASE_URL    = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"
LLM_MODEL   = "qwen2.5:7b-instruct"

DATA_PATH   = Path(r"C:\\Users\\jeong\\Desktop\\Global_Capstone\\Data\\Slack\\slack_messages.json")
PERSIST_DIR = "vectorstores/slack_bge_m3"
COLLECTION  = "slack_bge_m3"

TOP_K = 16
MAX_CTX_CHARS = 15000


# ===== Slack 데이터 불러오기 =====
def load_slack_messages():
    if not DATA_PATH.exists():
        print(f"❌ 파일 없음: {DATA_PATH}")
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ===== Vector DB 빌드 =====
def build_vectorstore():
    msgs = load_slack_messages()
    if not msgs:
        print("❗ 슬랙 메시지가 없습니다.")
        return

    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db  = Chroma(collection_name=COLLECTION, persist_directory=PERSIST_DIR, embedding_function=emb)

    docs = []
    for m in msgs:
        content = (
            f"[채널] {m['channel_name']} ({m['channel_type']})\n"
            f"[사용자] {m['user_name']} ({m['role']})\n"
            f"[시간] {m['ts']}\n"
            f"[내용] {m['text']}"
        )
        docs.append(Document(page_content=content, metadata={
            "source": f"slack_{m['id']}",
            "channel": m["channel_name"],
            "user": m["user_name"]
        }))
    
    db.add_documents(docs)
    print(f"✅ {len(docs)}개의 Slack 메시지를 인덱싱했습니다 → {PERSIST_DIR} ({COLLECTION})")


# ===== 질의응답 =====
def ask(query: str, mode="ask"):
    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db  = Chroma(collection_name=COLLECTION, persist_directory=PERSIST_DIR, embedding_function=emb)

    results = db.similarity_search(query, k=TOP_K)

    if not results:
        print("❌ 관련 메시지를 찾지 못했습니다.")
        return

    # 컨텍스트 구성 + 출처 수집
    context_parts = []
    sources = []
    for r in results:
        context_parts.append(r.page_content)
        src = r.metadata.get("source", "unknown")
        sources.append(src)
    context = "\n\n---\n\n".join(context_parts)[:MAX_CTX_CHARS]
    sources_str = ", ".join(sorted(set(sources)))

    # ===== 프롬프트 =====
    system_prompts = {
    "ask": """
당신은 슬랙 로그 기반 분석 어시스턴트입니다.
질문에 답할 때는 반드시 **아래 Slack 메시지 내용만 근거로** 작성하세요.
허구로 이름이나 정보를 만들어내면 안 됩니다.

질문이 '누구', '참여', '관련', '업데이트' 등의 단어를 포함할 경우, 
하나의 메시지뿐 아니라 **연관된 모든 사용자**를 함께 분석하여 나열하세요.

출력 형식은 반드시 아래와 같아야 합니다:
=== 답변 ===
 [요약된 답변 내용 (실제 사용자 이름, 역할, 메시지 기반으로 작성)]

📎 관련 메시지: slack_1, slack_2, ...
""",

        "audit": """
당신은 보안 감사 담당자입니다.
Slack 로그를 분석해 보안 위험이나 내부 위반을 탐지하세요.
출력은 반드시 한국어로 작성하세요.
""",
        "dev": """
당신은 개발 QA 분석가입니다.
Slack 메시지에서 반복 문장, 시스템 알림 템플릿을 찾아 요약하세요.
"""
    }

    system = system_prompts.get(mode, system_prompts["ask"])

    llm = OllamaLLM(model=LLM_MODEL, base_url=BASE_URL, temperature=0.2)

    prompt = f"""{system}

[질문]
{query}

[Slack 데이터]
{context}
"""

    answer = llm.invoke(prompt)
    print(answer)


# ===== 실행 =====
if __name__ == "__main__":
    mode = input("모드 선택 (build / ask / audit / dev): ").strip().lower()
    if mode == "build":
        build_vectorstore()
    else:
        query = input("질문을 입력하세요: ").strip()
        ask(query, mode)
