import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Download NLTK data
nltk.download('punkt')

# Initialize FastAPI
app = FastAPI()

# Pydantic model for API input
class QueryInput(BaseModel):
    query: str
    language: str = "bn"  # Default to Bengali

# Initialize LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Global variables for memory
short_term_memory: List[Dict[str, str]] = []  # Store recent queries and responses
vector_store = None

def preprocess_text(text: str, language: str = "bn") -> str:
    """Preprocess text for better chunking and accuracy."""
    # Remove unwanted characters and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    if language == "bn":
        # Bengali-specific cleaning (remove extra punctuation, normalize spaces)
        text = re.sub(r'[।!?]+', '।', text)
        text = re.sub(r'[^\u0980-\u09FF\s।]', '', text)  # Keep only Bengali characters
    else:
        # English cleaning
        text = re.sub(r'[^\w\s.]', '', text)
    
    return text

def load_pdf_corpus(pdf_path: str) -> str:
    """Load and extract text from PDF."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return preprocess_text(text)
    except Exception as e:
        raise Exception(f"Error loading PDF: {str(e)}")

def create_vector_store(text: str) -> FAISS:
    """Create FAISS vector store from text."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings)

def initialize_rag(pdf_path: str) -> RetrievalQA:
    """Initialize the RAG pipeline."""
    global vector_store
    corpus_text = load_pdf_corpus(pdf_path)
    vector_store = create_vector_store(corpus_text)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# Initialize RAG with the HSC26 Bangla 1st paper PDF
rag_chain = initialize_rag("hsc26_bangla_1st_paper.pdf") 

def evaluate_groundedness(query: str, answer: str, source_docs: List[str]) -> float:
    """Evaluate if answer is grounded in retrieved documents using cosine similarity."""
    query_embedding = embeddings.embed_query(query)
    doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in source_docs]
    
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    return float(np.mean(similarities))

def evaluate_relevance(query: str, source_docs: List[str]) -> float:
    """Evaluate relevance of retrieved documents."""
    query_embedding = embeddings.embed_query(query)
    doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in source_docs]
    
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    return float(np.max(similarities))

@app.post("/query")
async def process_query(input: QueryInput):
    """API endpoint to process user queries."""
    try:
        # Process query
        result = rag_chain({"query": input.query})
        answer = result["result"]
        source_docs = result["source_documents"]

        # Evaluate response
        groundedness_score = evaluate_groundedness(input.query, answer, source_docs)
        relevance_score = evaluate_relevance(input.query, source_docs)

        # Update short-term memory
        short_term_memory.append({
            "query": input.query,
            "answer": answer,
            "language": input.language,
            "groundedness": groundedness_score,
            "relevance": relevance_score
        })
        
        # Keep only last 5 interactions in short-term memory
        if len(short_term_memory) > 5:
            short_term_memory.pop(0)

        return {
            "answer": answer,
            "groundedness_score": groundedness_score,
            "relevance_score": relevance_score,
            "source_documents": [doc.page_content for doc in source_docs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)