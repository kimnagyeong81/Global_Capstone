# voice/ask_voice.py
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

BASE_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"
PERSIST_DIR = "vectorstores/voices_bge_m3"
COLLECTION = "voices_bge_m3"

def main():
    q = input("질문: ").strip()
    embeddings = OllamaEmbeddings(base_url=BASE_URL, model=EMBED_MODEL)
    vs = Chroma(collection_name=COLLECTION, embedding_function=embeddings, persist_directory=PERSIST_DIR)
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(q)
    print("\n--- 결과 ---")
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.metadata.get('title')}  ({d.metadata.get('created_at')})")
        print(d.page_content[:300].replace("\n"," ") + ("..." if len(d.page_content)>300 else ""))
        print(f"meta: {d.metadata}\n")

if __name__ == "__main__":
    main()
 