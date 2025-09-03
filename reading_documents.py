# https://python.langchain.com/docs/how_to/#document-loaders
#pip install langchain
#pip install langchain-community
#pip install pypdf

from langchain_community.document_loaders import TextLoader, PyPDFLoader

file_path = "onboarding.txt"
loader = TextLoader(file_path)
docs = loader.load()
print(docs)

print("===========\n\n\n\n\n============")

file_path = "spgi-annual-report-2023.pdf"
loader = PyPDFLoader(file_path)
docs2 = loader.load()
print(docs2)

print("Done")

