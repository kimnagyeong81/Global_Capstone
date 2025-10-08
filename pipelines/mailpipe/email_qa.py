# email_qa.py
import sys
import re, unicodedata
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from pipelines.common.preprocess import clean_email

# ===== 설정 =====
BASE_URL    = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"                  # ollama list 에 있어야 함
LLM_MODEL   = "qwen2.5:7b-instruct"     # 8B 쓰려면 "llama3.1:8b"
PERSIST_DIR = "vectorstores/emails_bge_m3"
COLLECTION  = "emails_bge_m3"

TOP_K       = 20     # 넉넉히 가져오고
MAX_KEEP    = 8      # 컨텍스트에 더 많이 남기기
THRESHOLD   = 0.45   # 너무 엄격하면 0개가 되므로 완화
MAX_CTX_CHARS = 10000  # 컨텍스트 길이 제한
# ===============  # 

def clean(t: str) -> str:
    # 유니코드 정규화
    t = unicodedata.normalize("NFKC", t)
    # CR 제거
    t = t.replace("\r", "")
    # 인용부(> 로 시작하는 줄) 제거
    t = "\n".join(ln for ln in t.split("\n") if not ln.strip().startswith(">"))
    # PGP 블록 앞에서 자르기
    t = t.split("-----BEGIN PGP")[0]
    # 밑줄/별 강조 정리
    t = t.replace("_", "").replace("*", "•")
    # 줄바꿈 하이픈 단어 재결합: care-\nful -> careful
    t = re.sub(r"(\w+)-\s*\n(\w+)", r"\1\2\n", t)
    # 과도한 공백/개행 정리
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def main():
    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db  = Chroma(collection_name=COLLECTION, persist_directory=PERSIST_DIR, embedding_function=emb)

    # 질문 입력(인자 또는 프롬프트)
    query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else input("질문을 입력하세요: ").strip()

    # ---------- 1) 검색 + 후보 디버그 ----------
    raw = db.similarity_search_with_score(query, k=TOP_K)

    print("[DEBUG] candidates (score, source):")
    for d, s in raw[:8]:
        print("  ", round(s, 3), Path(d.metadata.get("source", "")).name)

    # ---------- 2) 간단 MMR 흉내로 재선택 ----------
    # 기본: 점수 낮은 것부터 TOP_K개 정렬
    ordered = sorted(raw, key=lambda x: x[1])[:TOP_K]

    selected = []
    for d, s in ordered:
        # 문장 bag-of-words 겹침 비율로 유사한 것은 걸러냄
        dw = set(d.page_content.split())
        keep = True
        for x, _ in selected:
            overlap = len(dw & set(x.page_content.split())) / max(1, len(dw))
            if overlap >= 0.6:  # 너무 비슷하면 버림
                keep = False
                break
        if keep:
            selected.append((d, s))
        if len(selected) >= MAX_KEEP:
            break

    pairs = selected or ordered[:MAX_KEEP]

    # (선택) 임계값으로 한 번 더 정리 (남은 게 0개면 원본 유지)
    pruned = [p for p in pairs if p[1] < THRESHOLD]
    if pruned:
        pairs = pruned[:MAX_KEEP]

    # ---------- 3) 컨텍스트 구성(출처 포함) ----------
    ctx_items = []
    for i, (d, s) in enumerate(pairs, start=1):
        src = Path(d.metadata.get("source", "")).name
        snippet = clean_email(d.page_content, pii=True, remove_html=True, drop_quotes=True, drop_signature=True)
        snippet = clean(snippet)
        ctx_items.append(f"[{i}] source={src}\n{snippet}")

    context = ("\n\n---\n\n".join(ctx_items))[:MAX_CTX_CHARS]

    #  지시문 
    system = """아래 검색 컨텍스트(여러 이메일의 발췌)만 사용해 한국어로 간결히 답해줘.
규칙:
- 영어와 한국어를 섞지 마. 모든 영어 단어는 자연스러운 한국어로 번역해.
- 번역이 애매하면 한국어 관용 표현으로 자연스럽게 풀어써.
- 인용은 한국어로 옮기되, 필요한 경우 원문 키워드는 괄호 안에 한 번만 제시해.
- 과도한 외래어나 직역체를 피하고, 매끄러운 문장으로 교정해.

형식:
- (필요시) 찬성/반대/중립 구분
- 핵심 주장 3가지 (불릿)
- 결론: 한 문장
- 근거: 컨텍스트에서 짧은 인용 1–2개와 출처 번호([1], [2]) 또는 파일명(source)
컨텍스트에 없으면 '자료 없음'이라고 말해.
"""

    prompt = f"{system}\n[질문]\n{query}\n\n[검색 컨텍스트]\n{context}"

    # ---------- 5) LLM 호출 ----------
    llm = Ollama(model=LLM_MODEL, base_url=BASE_URL, temperature=0.2)
    answer = llm.invoke(prompt)
    print("\n=== 답변 ===\n", answer)

if __name__ == "__main__":
    main()
