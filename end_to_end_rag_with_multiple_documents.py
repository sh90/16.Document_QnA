# pip install unstructured
# pip install python-docx pypdf

import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --------- Config ---------
DATA_DIR = "data"  # Directory containing your files
CHROMA_PERSIST_DIR = "chroma_store_multiple_document_openai"

from dotenv import load_dotenv
load_dotenv()

LLM_MODEL = "gpt-4o-mini"  # or "gpt-4"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Replace this with your open ai key "SK-"

# --------- Helper function to load documents ---------
def load_documents_from_directory(files):
    documents = []
    for filename in files:
        print(filename)
        if filename.endswith(".txt"):
            loader = TextLoader(filename)
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filename)
        else:
            print(f"[WARN] Skipping unsupported file type: {filename}")
            continue

        docs = loader.load()
        documents.extend(docs)
    return documents

# --------- Load and Split Documents ---------
print("[INFO] Loading and splitting documents...")
raw_docs = load_documents_from_directory(["onboarding.txt","spgi-annual-report-2023.pdf"])

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)

# --------- Initialize Embeddings ---------
EMBED_MODEL = "text-embedding-3-small"
print(f"[INFO] Using OpenAI Embeddings: {EMBED_MODEL}")
embedding_model = OpenAIEmbeddings(model=EMBED_MODEL,openai_api_key=OPENAI_API_KEY)

# --------- Create / Load Vector Store ---------
if os.path.exists(CHROMA_PERSIST_DIR):
    print("[INFO] Loading existing Chroma vector store...")
    vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_model)
else:
    print("[INFO] Creating Chroma vector store and embedding documents...")
    vectordb = Chroma.from_documents(docs, embedding_model, persist_directory=CHROMA_PERSIST_DIR)

# --------- Load LLM ---------
# --------- Initialize LLM ---------
print(f"[INFO] Using OpenAI LLM: {LLM_MODEL}")
llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0,openai_api_key=OPENAI_API_KEY)

# --------- Setup RetrievalQA ---------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # or "map_reduce"
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

# --------- Q&A Loop ---------
print("\n[READY] Ask me anything about the content of your files. Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa_chain.invoke(query)
    print(result)
    print("\nAI:", result["result"], "\n")
