#pip install tiktoken openai
#pip install chromadb

import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

# --------- Config ---------
from dotenv import load_dotenv
load_dotenv()

TEXT_FILE_PATH = "onboarding.txt"
CHROMA_PERSIST_DIR = "chroma_store_openai"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"  # or "gpt-4"

# --------- Load and Split Text ---------
print("[INFO] Loading and splitting text...")
with open(TEXT_FILE_PATH, "r", encoding="utf-8") as f:
    text = f.read()

raw_docs = [Document(page_content=text)]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)

# --------- Initialize Embeddings ---------
EMBED_MODEL = "text-embedding-3-small"
print(f"[INFO] Using OpenAI Embeddings: {EMBED_MODEL}")
embedding_model = OpenAIEmbeddings(model=EMBED_MODEL,openai_api_key=OPENAI_API_KEY)

# --------- Create or Load ChromaDB ---------
if os.path.exists(CHROMA_PERSIST_DIR):
    print("[INFO] Loading existing Chroma vector store...")
    vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_model)
else:
    print("[INFO] Creating Chroma vector store and embedding documents...")
    vectordb = Chroma.from_documents(docs, embedding_model, persist_directory=CHROMA_PERSIST_DIR)

# --------- Initialize LLM ---------
print(f"[INFO] Using OpenAI LLM: {LLM_MODEL}")
llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0,openai_api_key=OPENAI_API_KEY)

# --------- Create RetrievalQA Chain ---------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

# --------- Q&A Loop ---------
print("\n[READY] Ask me anything about the content in your file. Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    #query = "What is the customer onboarding process"
    # query = "Using only the given context answer following question:\n" + query
    result = qa_chain.invoke(query)
    print("\nAI:", result["result"], "\n")
