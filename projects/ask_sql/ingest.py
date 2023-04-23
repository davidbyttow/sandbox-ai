import pathlib
import pickle
import json

import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

dir = pathlib.Path(__file__).parent
f = open(dir / "localdata/ddls.json", "r")
ddls = json.loads(f.read())


def to_doc(ddl):
    db = ddl["database"]
    schema = ddl["schema"]
    name = ddl["name"]
    lines = []
    lines.append(f"Table definition for {db}.{schema}.{name}:")
    for col in ddl["cols"]:
        name = col["name"]
        dt = col["data_type"]
        comment = col["comment"]
        line = f"  - column:{name} type:{dt}"
        if comment:
            line += f" comment:{comment}"
        lines.append(line)
    return "\n".join(lines)


docs = []
metadatas = []
for i, ddl in enumerate(ddls):
    db = ddl["database"]
    schema = ddl["schema"]
    name = ddl["name"]
    fqn = f"{db}.{schema}.{name}"
    docs.append(to_doc(ddl))
    metadatas.append({"source": fqn})

store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, str(dir / "localdata/docs.index"))
store.index = None
with open(dir / "localdata/faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
