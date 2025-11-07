# email_qa.py
import sys
import re, unicodedata
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pipelines.common.preprocess import clean_email

# ===== 설정 =====
BASE_URL    = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"                  # ollama list 에 있어야 함
LLM_MODEL   = "qwen2.5:7b-instruct"     # 8B 쓰려면 "llama3.1:8b"
PERSIST_DIR = "vectorstores/emails_bge_m3"
COLLECTION  = "emails_bge_m3"

TOP_K       = 20      # 넉넉히 가져오고
MAX_KEEP    = 20      # 혹은 50 정도로 넉넉히
THRESHOLD   = 0.85    # 약간 완화
MAX_CTX_CHARS = 10000 # 컨텍스트 길이 제한
# ====================================

def clean(t: str) -> str:
    """텍스트 정리 함수"""
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\r", "")
    t = "\n".join(ln for ln in t.split("\n") if not ln.strip().startswith(">"))
    t = t.split("-----BEGIN PGP")[0]
    t = t.replace("_", "").replace("*", "•")
    t = re.sub(r"(\w+)-\s*\n(\w+)", r"\1\2\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def main():
    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db  = Chroma(collection_name=COLLECTION, persist_directory=PERSIST_DIR, embedding_function=emb)

    # 질문 입력
    query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else input("질문을 입력하세요: ").strip()

    # ---------- 1) 검색 ----------
    raw = db.similarity_search_with_score(query, k=TOP_K)

    print("[DEBUG] candidates (score, source):")
    for d, s in raw[:8]:
        print("  ", round(s, 3), Path(d.metadata.get("source", "")).name)

    # ---------- 2) 중복 필터 완전 제거 ----------
    ordered = sorted(raw, key=lambda x: x[1])[:TOP_K]
    pairs = ordered[:MAX_KEEP] 

    # (선택) 임계값으로 정리
    pruned = [p for p in pairs if p[1] < THRESHOLD]
    if pruned:
        pairs = pruned[:MAX_KEEP]

    # ---------- 3) 컨텍스트 구성 ----------
    ctx_items = []
    for i, (d, s) in enumerate(pairs, start=1):
        src = Path(d.metadata.get("source", "")).name
        snippet = clean_email(d.page_content, pii=True, remove_html=True, drop_quotes=True, drop_signature=True)
        snippet = clean(snippet)

        # ✅ 날짜 정규화 (metadata에서 직접 추출)
        date_val = d.metadata.get("date", "날짜 없음")
        date_val = re.sub(r"T\d{2}:\d{2}:\d{2}", "", date_val)  # ISO에서 T이후 제거

        ctx_items.append(f"[{i}] source={src}\ndate={date_val}\n{snippet}")

    context = ("\n\n---\n\n".join(ctx_items))[:MAX_CTX_CHARS]

    # ---------- 4) 시스템 프롬프트 ----------
    system = """당신은 이메일 데이터 분석 도우미입니다.
사용자의 질문을 먼저 분석해서 어떤 유형인지 판단하세요:
- 사람(보낸 사람, 받은 사람)
- 날짜/시간
- 이메일 내용 요약
- 카테고리/주제
그에 맞춰 적절한 정보를 아래 컨텍스트에서 찾아 간결히 한국어로 답하세요.

규칙:
- 이메일이 여러 개일 경우 관련된 모든 사람·제목·날짜를 **절대 빠짐없이 모두 나열하세요.**
- 여러 명이면 쉼표로 구분하지 말고, **각 이메일을 줄바꿈으로 구분해 각각 한 줄로 작성하세요.**
- 질문이 단수형이라도 관련된 모든 이메일을 모두 나열해야 합니다.
- 날짜는 반드시 원문에서 찾아서 'YYYY-MM-DD' 형식으로 출력하세요.
- 만약 ISO 형식(예: 2025-09-05T15:42:57)이라면 앞의 날짜만 사용하세요.
- 날짜를 찾지 못하면 '날짜 없음'이라고 적으세요.
- 같은 제목이 반복되어도 모두 각각 출력해야 하며, 절대 생략하거나 요약하지 마세요.
- 각 이메일은 아래 형식으로 출력하세요:
  • 보낸 사람 → 받는 사람 | 날짜 | 제목
- 답변은 반드시 **한국어로만 작성**하세요 (영어, 중국어, 기타 언어 사용 금지).
"""

    prompt = f"{system}\n[질문]\n{query}\n\n[검색 컨텍스트]\n{context}"

    # ---------- 5) LLM 호출 ----------
    llm = Ollama(model=LLM_MODEL, base_url=BASE_URL, temperature=0.2)
    answer = llm.invoke(prompt)
    print("\n=== 답변 ===\n", answer)

if __name__ == "__main__":
    main()
