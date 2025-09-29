# check_db.py
from chromadb import PersistentClient
import os

DIR = "vectorstores/emails_bge_m3"
NAME = "emails_bge_m3"

print("db file exists:", os.path.exists(os.path.join(DIR, "chroma.sqlite3")))
client = PersistentClient(path=DIR)
col = client.get_or_create_collection(NAME)
print("docs in collection:", col.count())
