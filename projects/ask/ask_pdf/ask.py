from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("principles.pdf")
pages = loader.load_and_split()

print(pages[0])

# TODO(d)
