import pathlib
import faiss
import pickle
import argparse

from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain

parser = argparse.ArgumentParser(description="Ask Notion docs")
parser.add_argument("question", type=str, help="Question to ask")
args = parser.parse_args()

index = faiss.read_index("output-data/docs.index")

with open("output-data/faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index

chain = RetrievalQAWithSourcesChain.from_llm(
    llm=OpenAI(temperature=0), retriever=store.as_retriever()
)

result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sourced from: {result['sources']}")
