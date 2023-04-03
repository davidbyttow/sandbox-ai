import os
import pickle
import faiss
import pathlib
import argparse

from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

parser = argparse.ArgumentParser(description="Ingest Notion docs")
parser.add_argument("dir", type=str, help="Root directory to load documents from")
args = parser.parse_args()

ps = list(Path(args.dir).glob("**/*.md"))

data = []
sources = []
for p in ps:
    with open(p) as f:
        data.append(f.read())
    sources.append(p)

splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "output-data/docs.index")
store.index = None

with open("output-data/faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
