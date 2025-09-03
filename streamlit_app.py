# app.py
# Streamlit UI for RAG over TXT and PDF (OpenAI + Chroma)
# -------------------------------------------------------

import os
import io
import time
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

from pypdf import PdfReader  # <-- NEW

# -------------------- Config & Helpers --------------------
load_dotenv()

DEFAULT_TEXT_FILE_PATH = "onboarding.txt"  # can also be a .pdf now
DEFAULT_CHROMA_DIR = "chroma_store_openai"
DEFAULT_LLM = "gpt-4o-mini"
DEFAULT_EMBED = "text-embedding-3-small"

SESSION_KEYS = ["vectordb", "qa_chain", "docs_ready", "messages", "last_build_info"]

def init_session_state():
    for k in SESSION_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None
    if "messages" not in st.session_state or st.session_state["messages"] is None:
        st.session_state["messages"] = []

def ensure_api_key():
    api_key = os.getenv("OPENAI_API_KEY") or ""
    return api_key

def read_text_from_upload(uploaded_file) -> Tuple[str, str]:
    """
    Returns (text, source_label) from an uploaded file (.txt or .pdf).
    """
    if uploaded_file is None:
        return "", ""
    name = uploaded_file.name
    data = uploaded_file.read()
    ext = os.path.splitext(name.lower())[1]
    if ext == ".pdf":
        text = extract_text_from_pdf_bytes(data)
    else:
        # treat as text
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="ignore")
    return text, f"Uploaded: {name}"

def load_text_from_path(path: str) -> Tuple[str, str]:
    """
    Returns (text, source_label) from a path to .txt / .pdf.
    """
    if not path or not os.path.exists(path):
        return "", ""
    ext = os.path.splitext(path.lower())[1]
    if ext == ".pdf":
        text = extract_text_from_pdf_path(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    return text, f"Path: {os.path.basename(path)}"

def extract_text_from_pdf_path(path: str) -> str:
    try:
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n".join(pages)
    except Exception as e:
        return f""

def extract_text_from_pdf_bytes(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n".join(pages)
    except Exception:
        return ""

def split_into_docs_with_meta(raw_text: str, source: str, per_page_chunks: bool,
                              chunk_size=500, chunk_overlap=50) -> List[Document]:
    """
    Splits text into chunks, annotating with lightweight metadata (filename/page=None).
    If you later want true per-PDF-page chunks, pass per_page_chunks=True with pre-split text by page.
    For simplicity we keep a single text stream here; metadata keeps source label.
    """
    if not raw_text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_text(raw_text)
    return [
        Document(page_content=chunk, metadata={"source": source})
        for chunk in docs
    ]

def split_pdf_into_docs_per_page(uploaded_file=None, path: str = "", chunk_size=500, chunk_overlap=50) -> List[Document]:
    """
    Optional: Split PDFs page-by-page, then chunk each page, preserving page_number in metadata.
    Works for a single PDF (uploaded or path).
    """
    try:
        if uploaded_file is not None:
            name = uploaded_file.name
            data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
            reader = PdfReader(io.BytesIO(data))
            source = f"Uploaded: {name}"
        else:
            reader = PdfReader(path)
            source = f"Path: {os.path.basename(path)}"

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs: List[Document] = []
        for i, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
            chunks = splitter.split_text(page_text)
            for ch in chunks:
                docs.append(Document(
                    page_content=ch,
                    metadata={"source": source, "page": i}
                ))
        return docs
    except Exception:
        return []

def build_or_load_chroma(docs: List[Document], persist_dir: str, embedding_model: OpenAIEmbeddings, force_rebuild=False):
    if os.path.exists(persist_dir) and not force_rebuild:
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
        return vectordb, False
    if not docs:
        return None, False
    # Pass metadatas so source/page info is stored for retrieval display
    vectordb = Chroma.from_documents(docs, embedding_model, persist_directory=persist_dir)
    return vectordb, True

def make_qa_chain(vectordb: Chroma, llm_model: str, temperature: float, api_key: str):
    llm = ChatOpenAI(model_name=llm_model, temperature=temperature, openai_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# -------------------- UI --------------------
st.set_page_config(page_title="RAG Chat (TXT + PDF)", page_icon="📄", layout="wide")
init_session_state()

st.title("📄💬 RAG Chat over TXT & PDF (OpenAI + Chroma)")
st.caption("Upload .txt/.pdf or provide a file path. Build embeddings with OpenAI, chat, and view retrieved source chunks with filename and page numbers.")

with st.sidebar:
    st.header("⚙️ Settings")
    openai_key = st.text_input("OPENAI_API_KEY (optional if in .env)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    api_key = ensure_api_key()
    if not api_key:
        st.warning("Set OPENAI_API_KEY here or in your .env file.", icon="⚠️")

    llm_model = st.selectbox("LLM model", options=[DEFAULT_LLM, "gpt-4o", "gpt-4o-mini-2024-07-18"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    embed_model_name = st.selectbox("Embedding model", options=[DEFAULT_EMBED, "text-embedding-3-large"], index=0)

    st.divider()
    st.subheader("📄 Corpus")
    mode = st.radio("Choose input mode", ["Upload files", "Use a single file path", "Manual text"], index=0)

    uploaded_files = None
    path_input = ""
    manual_text = ""

    if mode == "Upload files":
        uploaded_files = st.file_uploader(
            "Upload .txt or .pdf (you can upload multiple)", type=["txt", "pdf"], accept_multiple_files=True
        )
    elif mode == "Use a single file path":
        path_input = st.text_input("File path (.txt or .pdf)", value=DEFAULT_TEXT_FILE_PATH)
    else:
        manual_text = st.text_area("Paste text here", height=200, placeholder="Paste the content you want to chat over…")

    st.divider()
    persist_dir = st.text_input("Chroma persist dir", value=DEFAULT_CHROMA_DIR)
    force_rebuild = st.checkbox("Force rebuild vector store", value=False)

    per_page_pdf = st.checkbox("PDF: split per page (keeps page numbers in metadata)", value=True)
    chunk_size = st.number_input("Chunk size", min_value=100, max_value=4000, value=500, step=50)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=50, step=10)

    build_btn = st.button("🔨 Build / Load Vector Store", type="primary")

    st.divider()
    if st.button("🧹 Clear chat history"):
        st.session_state["messages"] = []
        st.toast("Chat history cleared.", icon="🧽")
        st.rerun()

# -------------------- Build / Load Vector Store --------------------
if build_btn:
    if not api_key:
        st.error("OPENAI_API_KEY is required to build embeddings.")
    else:
        docs: List[Document] = []
        sources_desc = []
        total_chunks = 0

        try:
            if mode == "Upload files":
                if not uploaded_files:
                    st.error("No files uploaded.")
                else:
                    for uf in uploaded_files:
                        name = uf.name
                        ext = os.path.splitext(name.lower())[1]
                        if ext == ".pdf":
                            if per_page_pdf:
                                page_docs = split_pdf_into_docs_per_page(
                                    uploaded_file=uf, path="", chunk_size=chunk_size, chunk_overlap=chunk_overlap
                                )
                                docs.extend(page_docs)
                            else:
                                # Single stream from PDF
                                text = extract_text_from_pdf_bytes(uf.getvalue() if hasattr(uf, "getvalue") else uf.read())
                                docs.extend(split_into_docs_with_meta(text, f"Uploaded: {name}", per_page_chunks=False,
                                                                      chunk_size=chunk_size, chunk_overlap=chunk_overlap))
                        else:
                            # .txt
                            uf.seek(0)
                            content = uf.read()
                            try:
                                text = content.decode("utf-8")
                            except UnicodeDecodeError:
                                text = content.decode("latin-1", errors="ignore")
                            docs.extend(split_into_docs_with_meta(text, f"Uploaded: {name}", per_page_chunks=False,
                                                                  chunk_size=chunk_size, chunk_overlap=chunk_overlap))
                        sources_desc.append(name)

            elif mode == "Use a single file path":
                if not path_input or not os.path.exists(path_input):
                    st.error("Invalid file path.")
                else:
                    name = os.path.basename(path_input)
                    ext = os.path.splitext(name.lower())[1]
                    if ext == ".pdf":
                        if per_page_pdf:
                            page_docs = split_pdf_into_docs_per_page(
                                uploaded_file=None, path=path_input, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                            )
                            docs.extend(page_docs)
                        else:
                            text = extract_text_from_pdf_path(path_input)
                            docs.extend(split_into_docs_with_meta(text, f"Path: {name}", per_page_chunks=False,
                                                                  chunk_size=chunk_size, chunk_overlap=chunk_overlap))
                    else:
                        with open(path_input, "r", encoding="utf-8") as f:
                            text = f.read()
                        docs.extend(split_into_docs_with_meta(text, f"Path: {name}", per_page_chunks=False,
                                                              chunk_size=chunk_size, chunk_overlap=chunk_overlap))
                    sources_desc.append(name)

            else:  # Manual text
                if not manual_text.strip():
                    st.error("No text provided.")
                else:
                    docs.extend(split_into_docs_with_meta(manual_text, "Manual text", per_page_chunks=False,
                                                          chunk_size=chunk_size, chunk_overlap=chunk_overlap))
                    sources_desc.append("Manual text")

            if docs:
                total_chunks = len(docs)
                with st.spinner("Initializing embeddings and (re)building/loading Chroma…"):
                    embedding_model = OpenAIEmbeddings(model=embed_model_name, openai_api_key=api_key)
                    vectordb, rebuilt = build_or_load_chroma(docs, persist_dir, embedding_model, force_rebuild=force_rebuild)

                if vectordb is None:
                    st.error("Failed to initialize Chroma.")
                else:
                    st.session_state["vectordb"] = vectordb
                    st.session_state["qa_chain"] = make_qa_chain(vectordb, llm_model, temperature, api_key)
                    st.session_state["docs_ready"] = True
                    st.session_state["last_build_info"] = {
                        "rebuilt": rebuilt,
                        "chunks": total_chunks,
                        "persist_dir": persist_dir,
                        "sources": sources_desc,
                        "embed_model": embed_model_name,
                        "pdf_per_page": per_page_pdf
                    }
                    msg = "Vector store rebuilt." if rebuilt else "Loaded existing vector store."
                    st.success(f"✅ {msg}")
                    with st.expander("Build details"):
                        st.json(st.session_state["last_build_info"])
                    st.toast("Vector store ready!", icon="✅")
            else:
                st.error("No chunks were created. Check your inputs or adjust chunk size/overlap.")
        except Exception as e:
            st.error(f"Build failed: {e}")

# If vectordb exists, allow rewire
if st.session_state.get("vectordb") and ensure_api_key():
    with st.sidebar:
        if st.button("==== Apply LLM settings (no rebuild)"):
            st.session_state["qa_chain"] = make_qa_chain(st.session_state["vectordb"], llm_model, temperature, ensure_api_key())
            st.toast("LLM settings applied.", icon="🔁")

# -------------------- Chat Area --------------------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("💬 Chat")
    if not st.session_state.get("qa_chain"):
        st.info("Build or load the vector store from the sidebar to start chatting.")
    else:
        for m in st.session_state["messages"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_input = st.chat_input("Ask something about your document…")
        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                try:
                    result = st.session_state["qa_chain"].invoke({"query": user_input})
                    answer = (result.get("result") or "").strip()
                    sources = result.get("source_documents", []) or []

                    streamed = ""
                    for token in answer.split():
                        streamed += token + " "
                        placeholder.markdown(streamed)
                        time.sleep(0.005)
                    placeholder.markdown(answer)

                    st.session_state["messages"].append({"role": "assistant", "content": answer})

                    if sources:
                        with st.expander(" Retrieved context (sources)"):
                            for i, d in enumerate(sources, start=1):
                                meta = d.metadata or {}
                                src = meta.get("source", "unknown")
                                page = meta.get("page")
                                header = f"**Chunk {i}** — *{src}*"
                                if page:
                                    header += f", page {page}"
                                st.markdown(header)
                                snippet = (d.page_content or "")[:700].strip().replace("\n", " ")
                                st.write(snippet + ("…" if len(d.page_content or "") > 700 else ""))

                except Exception as e:
                    st.error(f"Error during retrieval/QA: {e}")

with col_right:
    st.subheader("==== Status ===")
    if st.session_state.get("last_build_info"):
        info = st.session_state["last_build_info"]
        st.markdown(
            f"- **Chroma dir:** `{info['persist_dir']}`\n"
            f"- **Chunks:** {info['chunks']}\n"
            f"- **Embedding:** `{info['embed_model']}`\n"
            f"- **Per-page PDF:** `{info['pdf_per_page']}`\n"
            f"- **Sources:** {', '.join(info['sources']) if info['sources'] else '—'}"
        )
    else:
        st.markdown("- Vector store not built/loaded yet.")

    st.divider()
    st.subheader(" Recommendations/Tips")
    st.markdown(
        "- If a PDF returns little/no text, it might be a scanned PDF (images). Add OCR (e.g., `pytesseract`) for those.\n"
        "- Use **Per-page PDF** to preserve page numbers in citations.\n"
        "- Tune chunk size/overlap if answers feel shallow or fragmented."
    )
