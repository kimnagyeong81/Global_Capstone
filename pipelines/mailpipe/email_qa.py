# email_qa.py
import sys
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# ===== 설정 =====
BASE_URL    = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"                      # 인덱싱 때와 동일
LLM_MODEL   = "llama3.1:8b"                 # 답변용 LLM
PERSIST_DIR = "vectorstores/emails_bge_m3"
COLLECTION  = "emails_bge_m3"
TOP_K       = 12                            # 우선 넉넉히 가져오고
THRESHOLD   = 0.35                          # 거리(score) 필터(낮을수록 유사)
MAX_KEEP    = 6                             # 최종 컨텍스트에 넣을 개수
MAX_CTX_CHARS = 8000                        # 컨텍스트 길이 제한
# ===============

def clean(t: str) -> str:
    t = t.replace("\r", "")
    lines = [ln for ln in t.split("\n") if not ln.strip().startswith("> ")]
    t = "\n".join(lines)
    t = t.split("-----BEGIN PGP")[0]
    return t.strip()

def main():
    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db  = Chroma(collection_name=COLLECTION, persist_directory=PERSIST_DIR, embedding_function=emb)

    # 질문 입력(인자 또는 프롬프트)
    query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else input("질문을 입력하세요: ").strip()

    # 1) 검색 (거리 기반 점수: 낮을수록 유사)
    pairs = db.similarity_search_with_score(query, k=TOP_K)

    # 디버그: 어떤 파일에서 왔는지 확인
    print("[DEBUG] candidates (score, source):")
    for d, s in pairs[:8]:
        print("  ", round(s, 3), Path(d.metadata.get("source", "")).name)

    # 2) 필터링 & 상위 선택
    pairs = [p for p in pairs if p[1] < THRESHOLD][:MAX_KEEP]
    if not pairs:                             # 너무 엄격해서 0개면 상위 4개 채택
        pairs = db.similarity_search_with_score(query, k=4)

    # 3) 컨텍스트 구성(출처 레이블 포함)
    ctx_items = []
    for i, (d, s) in enumerate(pairs, start=1):
        src = Path(d.metadata.get("source", "")).name
        snippet = clean(d.page_content)
        ctx_items.append(f"[{i}] source={src}\n{snippet}")

    context = ("\n\n---\n\n".join(ctx_items))[:MAX_CTX_CHARS]

    # 4) 지시문(출처·인용 강제)
    system = """아래 검색 컨텍스트(여러 이메일의 발췌)만 사용해 한국어로 간결히 답해줘.
형식:
- 찬성/반대/중립 등 입장 구분이 필요하면 명시
- 핵심 주장 3가지 (불릿)
- 결론: 한 문장
- 근거: 컨텍스트에서 짧은 인용 1–2개와 출처 번호([1], [2] 형식)
- 각 항목에 출처 번호([n]) 또는 파일명(source)을 포함
컨텍스트에 없으면 '자료 없음'이라고 말해.
"""

    prompt = f"{system}\n[질문]\n{query}\n\n[검색 컨텍스트]\n{context}"

    # 5) LLM 호출
    llm = Ollama(model=LLM_MODEL, base_url=BASE_URL, temperature=0.2)
    answer = llm.invoke(prompt)
    print("\n=== 답변 ===\n", answer)

if __name__ == "__main__":
    main()
