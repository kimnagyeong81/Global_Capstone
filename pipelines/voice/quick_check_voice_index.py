# quick_check_voice_index.py
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

emb = OllamaEmbeddings(base_url="http://127.0.0.1:11434", model="bge-m3")
vs = Chroma(collection_name="voices_bge_m3",
            persist_directory="vectorstores/voices_bge_m3",
            embedding_function=emb)
print("docs:", vs._collection.count())
