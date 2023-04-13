import sys

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

embedding = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0.3, max_tokens=400, model="gpt-4")
docsearch = Chroma(
    embedding_function=embedding,
    collection_name="gdoc1",
    persist_directory=".cache",
)

query = sys.argv[1]
docs = docsearch.similarity_search(query, k=10)

# [print(d.metadata["source"]) for d in docs]

chain = load_qa_chain(llm, chain_type="stuff")
out = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

print(out.get("output_text").split("\n"))
