import streamlit as st
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import tempfile
import os

st.set_page_config(page_title="📄 PDF Q&A with Ollama", layout="centered")
st.title("📄 Local PDF Q&A Assistant (Ollama + Streamlit)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and split PDF
    with st.spinner("🔍 Reading and indexing your PDF..."):
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)

        # Embeddings & Vector Store
        embedding = OllamaEmbeddings(model="llama3")
        vector_store = Chroma.from_documents(
    docs,
    embedding,
    persist_directory="./chroma_temp",  # temporary or custom path
)


        # Local LLM
        llm = Ollama(model="llama3")

        # Retrieval QA Chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )

    st.success("✅ PDF indexed successfully!")

    # Input Box
    query = st.text_input("Ask a question about the PDF:")
    if query:
        with st.spinner("💭 Thinking..."):
            result = qa(query)
            st.markdown("### 🧠 Answer:")
            st.write(result["result"])

            # with st.expander("📄 Sources"):
            #     for doc in result["source_documents"]:
            #         st.write(doc.page_content[:300] + "...")
            #         st.write("---")

    # Cleanup temp file
    os.remove(tmp_path)
