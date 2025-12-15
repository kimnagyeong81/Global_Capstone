## voice_reply_gen.py

# ======================
# 필요한 라이브러리 import
# ======================

import json                          # WhisperX가 만든 JSON 회의록을 읽기 위해 사용
from pathlib import Path             # 파일/디렉토리 경로를 OS 독립적으로 다루기 위함

from langchain_chroma import Chroma  # 벡터 DB (Chroma) 사용
from langchain_ollama import OllamaEmbeddings, OllamaLLM
                                     # Ollama 기반 임베딩 모델과 LLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
                                     # 긴 텍스트를 chunk로 나누는 도구
from langchain_core.documents import Document
                                     # LangChain에서 사용하는 문서 객체
# ======================
# 설정
# ======================
BASE_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:7b-instruct"

VOICE_DIR = Path("./data/voice")
PERSIST_DIR = Path("vectorstores/voices_bge_m3")
COLLECTION = "voices_bge_m3"

TOP_K = 10
THRESHOLD = 0.85
MAX_KEEP = 10


# ======================
# diarization JSON 읽기
# ======================
def load_diarization_docs():
    docs = []

    for file in VOICE_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments_text = []
        for seg in data["segments"]:
            line = f"[{seg['speaker']}] {seg['text']}"
            segments_text.append(line)

        full_text = "\n".join(segments_text)

        docs.append(
            Document(
                page_content=full_text,
                metadata={
                    "doc_id": data["doc_id"],
                    "source": "voice",
                    "created_at": data["created_at"]
                }
            )
        )

    print(f"📄 로드된 회의록 수: {len(docs)}")
    return docs


# ======================
# 인덱싱
# ======================
def build_index():
    docs = load_diarization_docs()
    if not docs:
        print("❌ 회의록이 없습니다.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = splitter.split_documents(docs)

    embed = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    vs = Chroma(collection_name=COLLECTION, embedding_function=embed, persist_directory=str(PERSIST_DIR))
    vs.add_documents(chunks)

    print("✅ VectorStore 인덱싱 완료")


# ======================
# 질문 → 답변
# ======================
def ask():
    query = input("질문: ")

    embed = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db = Chroma(collection_name=COLLECTION, embedding_function=embed, persist_directory=str(PERSIST_DIR))

    raw = db.similarity_search_with_score(query, k=TOP_K)
    filtered = [r for r in raw if r[1] < THRESHOLD][:MAX_KEEP]

    ctx = "\n\n---\n\n".join([d.page_content for d, _ in filtered])

    prompt = f"""
당신은 회의 내용을 분석하는 비서입니다.
다음 회의록을 기반으로 사실 그대로 답변하세요.

[질문]
{query}

[회의 내용]
{ctx}
"""

    llm = OllamaLLM(model=LLM_MODEL, base_url=BASE_URL)
    answer = llm.invoke(prompt)
    print(answer)


# ======================
# 메인
# ======================
def main():
    mode = input("모드 (build / ask): ").strip().lower()
    if mode == "build":
        build_index()
    elif mode == "ask":
        ask()


if __name__ == "__main__":
    main()
# ======================
# FastAPI 전용 함수
# ======================
def ask_question(question: str) -> str:
    """
    FastAPI에서 사용하기 위한 RAG 질의 함수.
    input()을 사용하지 않고, question 문자열을 직접 받아서 답변을 반환한다.
    """
    embed = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db = Chroma(
        collection_name=COLLECTION,
        embedding_function=embed,
        persist_directory=str(PERSIST_DIR)
    )

    # 검색
    raw = db.similarity_search_with_score(question, k=TOP_K)
    filtered = [r for r in raw if r[1] < THRESHOLD][:MAX_KEEP]

    # context
    ctx = "\n\n---\n\n".join([d.page_content for d, _ in filtered])

    # LLM 프롬프트 생성
    prompt = f"""
당신은 회의 내용을 분석하는 비서입니다.
다음 회의록을 기반으로 사실 그대로 답변하세요.

[질문]
{question}

[회의 내용]
{ctx}
"""

    llm = OllamaLLM(model=LLM_MODEL, base_url=BASE_URL)
    answer = llm.invoke(prompt)

    return answer
