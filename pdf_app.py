import os
import tempfile
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="PDF Q&A with Ollama", layout="centered")
st.title("PDF Q&A Assistant")


# -------------------------------
# Prompt Engineering
# -------------------------------


System_prompt= """
You are a domain-specific AI assistant designed to answer user queries strictly based on the contents of a provided PDF document. Follow these strict behavior and formatting rules:

Role:
- You are a concise, factual assistant.
- You only use the provided document context — no outside information.

Behavior Guidelines:
1. If the answer is found in the document, provide a clear, structured response.
2. If the document does not contain the answer, respond:
    "The document does not contain information to answer this question."
3. Do not mention knowledge from outside the document.
4. Respond in **Markdown format** and emphasize key terms using `**bold**`.

---

### Examples:

**Context**:
The PDF says: "Photosynthesis is a process used by plants to convert light energy into chemical energy."

**Question**:
What is photosynthesis?

**Answer**:
Photosynthesis is a process used by plants to **convert light energy into chemical energy**, as described in the document.

---

**Context**:
The document outlines the company was founded in 1998 and specializes in enterprise software.

**Question**:
When was the company founded?

**Answer**:
The company was founded in **1998**, according to the document.

---

**Context**:
The document contains no information about Python programming.

**Question**:
What is Python programming?

**Answer**:
The document does not contain information to answer this question.

---

### Now, answer the user's question:

**Context**:  
{context}

**Question**:  
{question}

**Answer**:
"""
)

few_shot_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template= System_prompt)

# -------------------------------
# 🔧 Helper Functions
# -------------------------------

def save_uploaded_file(uploaded_file):
    """Save uploaded PDF file to a temporary path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def load_and_split_pdf(pdf_path):
    """Load PDF and split into chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(pages)


def build_vector_store(documents):
    """Convert documents to embeddings and build a Chroma vector store."""
    embeddings = OllamaEmbeddings(model="llama3")
    return Chroma.from_documents(documents, embedding=embeddings, persist_directory="./chroma_temp")


def create_qa_chain_with_prompt(vector_store):
    """Create RetrievalQA chain with few-shot system prompt."""
    llm = Ollama(model="llama3")
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=few_shot_prompt)
    return RetrievalQA(
        retriever=vector_store.as_retriever(),
        combine_documents_chain=chain,
        return_source_documents=True
    )


def display_sources(source_docs):
    """Display source documents in expandable view."""
    with st.expander("Source Documents"):
        for i, doc in enumerate(source_docs, 1):
            st.markdown(f"**Source {i}:**")
            st.write(doc.page_content[:500] + "...")
            st.markdown("---")


# File Upload & QA Flow

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    try:
        temp_pdf_path = save_uploaded_file(uploaded_file)

        with st.spinner("Reading and indexing your PDF..."):
            documents = load_and_split_pdf(temp_pdf_path)
            vector_store = build_vector_store(documents)
            qa_chain = create_qa_chain_with_prompt(vector_store)

        st.success("PDF indexed successfully. Ask your question below!")

        query = st.text_input("Ask a question about the PDF:")
        if query:
            with st.spinner("Thinking..."):
                result = qa_chain(query)
                st.markdown("### Answer:")
                st.markdown(result["result"])
                display_sources(result["source_documents"])

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
