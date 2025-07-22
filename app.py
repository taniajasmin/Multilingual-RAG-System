import os
import PyPDF2
import re
import nltk
import numpy as np
import pdfplumber

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

# Load .env
load_dotenv()

# Download NLTK data
nltk.download("punkt")

# FastAPI app
app = FastAPI()

# Initialize LLM & Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Memory & vector store
short_term_memory: List[Dict[str, str]] = []
vector_store = None

# def preprocess_text(text: str, language: str = "bn") -> str:
#     text = re.sub(r"\s+", " ", text.strip())
#     if language == "bn":
#         text = re.sub(r"[।!?]+", "।", text)
#         text = re.sub(r"[^\u0980-\u09FF\s।]", "", text)
#     else:
#         text = re.sub(r"[^\w\s.]", "", text)
#     return text

def preprocess_text(text: str, language: str = "bn") -> str:
    # Remove control chars, normalize spaces
    text = re.sub(r"\s+", " ", text)
    # Remove non-Bangla Unicode chars except punctuation
    if language == "bn":
        text = re.sub(r"[^\u0980-\u09FF\s।]", "", text)
    else:
        text = re.sub(r"[^\w\s.]", "", text)
    return text.strip()


def load_pdf_corpus(pdf_path: str) -> str:
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    combined_text = " ".join(full_text)
    return preprocess_text(combined_text)

# def load_pdf_corpus(pdf_path: str) -> str:
#     try:
#         with open(pdf_path, "rb") as file:
#             reader = PyPDF2.PdfReader(file)
#             return preprocess_text(" ".join(page.extract_text() or "" for page in reader.pages))
#     except Exception as e:
#         raise Exception(f"Error loading PDF: {str(e)}")


def create_vector_store(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings)


def initialize_rag(pdf_path: str) -> RetrievalQA:
    global vector_store
    corpus_text = load_pdf_corpus(pdf_path)
    vector_store = create_vector_store(corpus_text)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )


rag_chain = initialize_rag("hsc26_bangla_1st_paper.pdf")


def evaluate_groundedness(query: str, answer: str, docs: List[str]) -> float:
    query_emb = embeddings.embed_query(query)
    doc_embs = [embeddings.embed_query(doc.page_content) for doc in docs]
    sims = cosine_similarity([query_emb], doc_embs)[0]
    return float(np.mean(sims))


def evaluate_relevance(query: str, docs: List[str]) -> float:
    query_emb = embeddings.embed_query(query)
    doc_embs = [embeddings.embed_query(doc.page_content) for doc in docs]
    sims = cosine_similarity([query_emb], doc_embs)[0]
    return float(np.max(sims))


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head><title>Bangla-English RAG</title></head>
    <body style="font-family: Arial; padding: 30px;">
        <h2>Ask a question (Bangla or English)</h2>
        <form action="/query" method="post">
            <input type="text" name="query" style="width: 400px;" placeholder="Enter your question..." required />
            <br><br>
            <label>Language:
                <select name="language">
                    <option value="bn" selected>Bengali</option>
                    <option value="en">English</option>
                </select>
            </label><br><br>
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """


@app.post("/query", response_class=HTMLResponse)
async def process_query(request: Request, query: str = Form(...), language: str = Form("bn")):
    try:
        result = rag_chain({"query": query})
        answer = result["result"]
        source_docs = result["source_documents"]

        groundedness = evaluate_groundedness(query, answer, source_docs)
        relevance = evaluate_relevance(query, source_docs)

        short_term_memory.append({
            "query": query,
            "answer": answer,
            "language": language,
            "groundedness": groundedness,
            "relevance": relevance
        })

        if len(short_term_memory) > 5:
            short_term_memory.pop(0)

        doc_preview = "".join(f"<li>{doc.page_content[:200]}...</li>" for doc in source_docs)

        return f"""
        <html>
        <head><title>Result</title></head>
        <body style="font-family: Arial; padding: 30px;">
            <h2>Query:</h2>
            <p>{query}</p>
            <h2>Answer:</h2>
            <p>{answer}</p>
            <h3>Groundedness Score: {groundedness:.2f}</h3>
            <h3>Relevance Score: {relevance:.2f}</h3>
            <h4>Source Snippets:</h4>
            <ul>{doc_preview}</ul>
            <br><a href="/">Ask another question</a>
        </body>
        </html>
        """

    except Exception as e:
        return f"<h2>Error:</h2><pre>{str(e)}</pre><br><a href='/'>Back</a>"

