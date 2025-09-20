# app_rag.py
import os
from pathlib import Path
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import tempfile
import sys

print(">>> Running file:", sys.argv[0])

# Streamlit page config
st.set_page_config(page_title="RAG PDF Bot (Summarize + Q&A)")

st.title("📚 RAG AI Bot — Upload PDFs/DOCs (Summarize + Q&A)")

# File uploader
uploaded = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

# Button to process, summarize & index
if st.button("Process, Summarize & Index"):
    if not uploaded:
        st.error("❌ Please upload at least one file.")
    else:
        docs = []
        for f in uploaded:
            suffix = Path(f.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(f.read())
                tmp_path = tmp_file.name

            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = UnstructuredWordDocumentLoader(tmp_path)

            docs.extend(loader.load())

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # ✅ HuggingFace embeddings (local)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create Chroma DB (in-memory)
        vectordb = Chroma.from_documents(chunks, embeddings)
        st.session_state.vectordb = vectordb

        # ✅ Local LLM pipeline for summarization
        pipe = pipeline(
            task="text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
            do_sample=True,
            temperature=0.1
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        # Summarize PDF chunks
        summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
        with st.spinner("📝 Generating summary..."):
            summary = summary_chain.run(chunks)
            st.session_state.pdf_summary = summary

        st.subheader("📄 PDF Summary:")
        st.write(summary)

        st.success("✅ Documents indexed and summarized successfully.")

# Query input
query = st.text_input("🔍 Ask a question about your uploaded docs:")
if query:
    if "vectordb" not in st.session_state:
        st.error("❌ Please process and index documents first.")
    else:
        vectordb = st.session_state.vectordb
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})

        # Local LLM pipeline for Q&A
        pipe = pipeline(
            task="text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
            do_sample=True,
            temperature=0.1
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        with st.spinner("🤔 Thinking..."):
            ans = qa.run(query)

        st.subheader("Answer:")
        st.write(ans)
