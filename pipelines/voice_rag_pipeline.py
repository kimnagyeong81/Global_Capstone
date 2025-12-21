import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

BASE_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:7b-instruct"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOICE_JSON_DIR = PROJECT_ROOT / "Data" / "voice" / "transcripts"
PERSIST_DIR = PROJECT_ROOT / "vectorstores" / "voices_bge_m3"
COLLECTION = "voices_bge_m3"

TOP_K = 8


def build_voice_index():
    docs = []

    for file in VOICE_JSON_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = "\n".join(
            f"[{s.get('speaker','UNK')}] {s.get('text','')}"
            for s in data["segments"]
        )

        docs.append(Document(
            page_content=text,
            metadata={"source": data["doc_id"]}
        ))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = splitter.split_documents(docs)

    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db = Chroma(collection_name=COLLECTION,
                persist_directory=str(PERSIST_DIR),
                embedding_function=emb)
    db.add_documents(chunks)

    print(f"✅ Voice RAG 인덱싱 완료 ({len(chunks)} chunks)")


def ask_voice(query: str) -> str:
    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    db = Chroma(collection_name=COLLECTION,
                persist_directory=str(PERSIST_DIR),
                embedding_function=emb)

    results = db.similarity_search(query, k=TOP_K)
    ctx = "\n\n---\n\n".join(r.page_content for r in results)

    prompt = f"""
당신은 회의록 분석 비서입니다.
다음 회의 내용만 근거로 답하세요.

[질문]
{query}

[회의]
{ctx}
"""

    llm = OllamaLLM(model=LLM_MODEL, base_url=BASE_URL)
    return llm.invoke(prompt)
