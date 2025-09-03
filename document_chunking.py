
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "spgi-annual-report-2023.pdf"
loader = PyPDFLoader(file_path)
docs1 = loader.load()
print("Done")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(docs1)
print(len(docs))
print("Chunk1")
print(docs[0].page_content)
print("Chunk2")
print(docs[1].page_content)
print("Chunk3")
print(docs[2].page_content)
print("Chunk4")
print(docs[3].page_content)
print("Done")
