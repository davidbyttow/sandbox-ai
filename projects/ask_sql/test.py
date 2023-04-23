import pathlib
import json
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

dir = pathlib.Path(__file__).parent
index = faiss.read_index(str(dir / "localdata/docs.index"))

with open(dir / "localdata/faiss_store.pkl", "rb") as f:
    store: FAISS = pickle.load(f)

store.index = index

docs_and_scores = store.similarity_search_with_score(
    "how much revenue did we make?", k=10
)

print(docs_and_scores)

# sort docs_and_scores by score
docs_and_scores.sort(key=lambda x: x[1], reverse=True)

for doc, score in docs_and_scores:
    print(doc.page_content)
    # print(doc.metadata["source"], score)
