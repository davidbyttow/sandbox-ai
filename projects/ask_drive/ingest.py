from dateutil.parser import isoparse
from typing import Any, Dict, List

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

SCOPES = [
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]
SERVICE_ACCOUNT_FILE = "./.secrets/sandbox-ai-383600-0b44db8715ca.json"


credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
drive_service = build("drive", "v3", credentials=credentials)
docs_service = build("docs", "v1", credentials=credentials)
embedding = OpenAIEmbeddings()


class DocRef:
    def __init__(self, id, name, modified_at):
        self.id = id
        self.name = name
        self.modified_at = modified_at


def gather_docs():
    results = (
        drive_service.files()
        .list(
            q="trashed = false",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
        )
        .execute()
    )
    items = results.get("files", [])

    docs = []
    for item in items:
        if item["mimeType"] == "application/vnd.google-apps.document":
            modified_time_str = item["modifiedTime"]
            modified_time = isoparse(modified_time_str)
            docs.append(DocRef(item["id"], item["name"], modified_time))
    return docs


def fetch_doc(ref: DocRef) -> Document:
    try:
        doc = docs_service.documents().get(documentId=ref.id).execute()
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

    if not doc:
        return None

    page_content = ""
    for elem in doc.get("body").get("content"):
        if "paragraph" in elem:
            for elem2 in elem.get("paragraph").get("elements"):
                if "textRun" in elem2:
                    page_content += elem2.get("textRun").get("content")
    return Document(
        page_content=page_content,
        metadata={
            "id": ref.id,
            "source": ref.name,
            "title": ref.name,
            "modified_at": ref.modified_at.isoformat(),
        },
    )


def ingest():
    gdocs = gather_docs()
    docs = [fetch_doc(gdoc) for gdoc in gdocs]
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")

    texts = []
    metadatas = []
    ids = []
    for doc in docs:
        chunks = splitter.split_text(doc.page_content)
        for i in range(len(chunks)):
            texts.append(chunks[i])
            metadatas.append(doc.metadata)
            ids.append(f'{doc.metadata["id"]}_{i+1}')
    print(f"ingesting {len(texts)} chunks")
    Chroma.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        ids=ids,
        collection_name="gdoc1",
        persist_directory=".cache",
    )


if __name__ == "__main__":
    ingest()
