import streamlit as st
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import tempfile
import os
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

st.set_page_config(page_title="PDF Q&A with Ollama", layout="centered")
st.title("PDF Q&A Assistant ")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

from langchain.prompts import PromptTemplate

prompt_with_few_shot = PromptTemplate(
    input_variables=["context", "question"],
    template="""
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



if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and split PDF
    with st.spinner(" Reading and indexing your PDF..."):
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

    st.success(" PDF indexed successfully!")

    # Input Box
    query = st.text_input("Ask a question about the PDF:")
    if query:
        with st.spinner(" Thinking..."):
            result = qa(query)
            st.markdown("###  Answer:")
            st.write(result["result"])

            # with st.expander(" Sources"):
            #     for doc in result["source_documents"]:
            #         st.write(doc.page_content[:300] + "...")
            #         st.write("---")

    # Cleanup temp file
    os.remove(tmp_path)
