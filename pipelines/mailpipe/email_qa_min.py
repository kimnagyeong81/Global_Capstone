# email_qa_min.py
BASE_URL    = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"
LLM_MODEL   = "llama3.1:8b"

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

print("[1] start")
emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
print("[2] emb ok")

db = Chroma(collection_name="emails",
            persist_directory="vectorstores/emails",
            embedding_function=emb)
print("[3] db ok")

docs = db.similarity_search("테스트", k=3)
print("[4] retrieved:", len(docs))

llm = Ollama(model=LLM_MODEL, base_url=BASE_URL, temperature=0.2)
print("[5] llm ok")
print("[6] llm says:", llm.invoke("한 줄로 인사해 줘."))
