# pipelines/voice/voice_reply_gen.py
import sys, json
from pathlib import Path
from datetime import datetime
from typing import List
from faster_whisper import WhisperModel
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chardet

# ====== 설정 ======
BASE_URL     = "http://127.0.0.1:11434"
EMBED_MODEL  = "bge-m3"
LLM_MODEL    = "qwen2.5:7b-instruct"

AUDIO_PATH   = Path("C:/Users/jeong/Desktop/Global_Capstone/Data/audio/Systemic_Failure_or_Intelligent_Governance__The_Project_Repeat_.mp3")
TRANSCRIPT_PATH = Path("data/voice_transcripts/transcripts.jsonl")
PERSIST_DIR  = Path("vectorstores/voices_bge_m3")
COLLECTION   = "voices_bge_m3"

TOP_K        = 10
THRESHOLD    = 0.85
MAX_KEEP     = 10
MAX_CTX_CHARS = 12000
# ====================================


# ① STT 단계 ------------------------------------------------
def transcribe_audio(audio_path: Path, out_path: Path):
    """오디오 파일을 STT로 변환하여 JSONL 저장"""
    print(f"🎙️ STT 변환 시작: {audio_path.name}")
    model = WhisperModel("medium", device="cuda", compute_type="float16")

    segments, info = model.transcribe(str(audio_path), language="en")
    text = "".join(seg.text for seg in segments).strip()

    if not text:
        print("❌ 음성 인식 결과가 비어 있습니다.")
        return False

    doc = {
        "doc_id": audio_path.stem,
        "source": "voice",
        "title": audio_path.stem,
        "created_at": datetime.now().isoformat(),
        "text": text,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"✅ STT 완료 → {out_path}")
    return True


# ② STT 결과 로드 -------------------------------------------
def load_voice_docs():
    """transcripts.jsonl 파일에서 문서 불러오기"""
    docs = []
    if not TRANSCRIPT_PATH.exists():
        print(f"❗ STT 결과 파일이 없습니다: {TRANSCRIPT_PATH}")
        return docs

    with open(TRANSCRIPT_PATH, "rb") as fb:
        enc = chardet.detect(fb.read()).get("encoding", "utf-8")

    with open(TRANSCRIPT_PATH, "r", encoding=enc, errors="ignore") as f:
        for line in f:
            try:
                r = json.loads(line)
                if not isinstance(r, dict):
                    continue
                text = (r.get("text") or "").strip()
                if not text:
                    continue
                meta = {
                    "doc_id": r.get("doc_id"),
                    "source": r.get("source", "voice"),
                    "title": r.get("title"),
                    "created_at": r.get("created_at"),
                }
                docs.append(Document(page_content=text, metadata=meta))
            except json.JSONDecodeError:
                continue

    print(f"📦 유효한 보이스 문서 수: {len(docs)}")
    return docs


# ③ 인덱싱 단계 --------------------------------------------
def build_index():
    """STT → 인덱싱 전체 파이프라인"""
    if not AUDIO_PATH.exists():
        print(f"❌ 오디오 파일이 없습니다: {AUDIO_PATH}")
        return

    # STT 먼저 실행
    if not transcribe_audio(AUDIO_PATH, TRANSCRIPT_PATH):
        print("❌ STT 실패로 인덱싱 중단")
        return

    # STT 결과 로드
    docs = load_voice_docs()
    if not docs:
        print("❗ 유효한 텍스트 문서가 없습니다.")
        return

    # 문서 청크 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = splitter.split_documents(docs)
    print(f"🔍 {len(chunks)}개 청크로 분할 완료")

    # 임베딩 & 인덱싱
    embeddings = OllamaEmbeddings(base_url=BASE_URL, model=EMBED_MODEL)
    vs = Chroma(collection_name=COLLECTION, embedding_function=embeddings, persist_directory=str(PERSIST_DIR))
    vs.add_documents(chunks)
    print(f"✅ {len(chunks)}개 청크 인덱싱 완료 → {PERSIST_DIR}")


# ④ 질의응답 단계 ------------------------------------------
def ask_question():
    """LLM 기반 질의응답"""
    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db  = Chroma(collection_name=COLLECTION, persist_directory=str(PERSIST_DIR), embedding_function=emb)

    query = input("질문을 입력하세요: ").strip()
    raw = db.similarity_search_with_score(query, k=TOP_K)
    if not raw:
        print("❌ 관련 회의 발화가 없습니다.")
        return

    ordered = sorted(raw, key=lambda x: x[1])
    pairs = [p for p in ordered if p[1] < THRESHOLD][:MAX_KEEP]

    ctx_items = []
    for i, (d, s) in enumerate(pairs, start=1):
        meta = d.metadata
        src = meta.get("doc_id", "-")
        date = meta.get("created_at", "날짜 없음")[:10]
        snippet = d.page_content.strip().replace("\n", " ")
        ctx_items.append(f"[{i}] ({date}) {snippet[:400]}")

    context = "\n\n---\n\n".join(ctx_items)
    context = context[:MAX_CTX_CHARS]

    system_prompt = """당신은 회의 음성 전사 데이터를 분석하는 비서입니다.
사용자의 질문이 '누가', '언제', '무엇을', '논의했는가', '결정했는가' 등을 묻는다면
아래 회의 전사 내용을 기반으로 **단 한 번만**, 구체적이고 사실적으로 답하세요.
허구의 내용을 추가하지 말고, 실제 발화된 내용만 사용하세요.

출력 형식 (단 한 번만 출력):
=== 답변 ===
[회의에서 실제로 논의된 핵심 내용 한 줄로 요약]
📎 관련 회의: [doc_id1, doc_id2, ...]
"""

    prompt = f"{system_prompt}\n\n[질문]\n{query}\n\n[회의 컨텍스트]\n{context}"
    llm = OllamaLLM(model=LLM_MODEL, base_url=BASE_URL, temperature=0.3)
    answer = llm.invoke(prompt)
    print(answer)


# ⑤ 인덱스 점검 --------------------------------------------
def audit_index():
    if not PERSIST_DIR.exists():
        print("❌ 인덱스 폴더가 없습니다.")
        return
    files = list(PERSIST_DIR.glob("**/*"))
    print(f"📦 인덱스 파일 {len(files)}개 존재")
    for f in files[:10]:
        print(" -", f)


# ⑥ 메인 ---------------------------------------------------
def main():
    mode = input("모드 선택 (build / ask / audit / dev): ").strip().lower()
    if mode == "build":
        build_index()
    elif mode == "ask":
        ask_question()
    elif mode == "audit":
        audit_index()
    elif mode == "dev":
        ask_question()
    else:
        print("❌ 잘못된 모드입니다. build / ask / audit 중 선택하세요.")


if __name__ == "__main__":
    main()
