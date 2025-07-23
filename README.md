# Multilingual-RAG-System

# Bangla-English RAG System

A Retrieval-Augmented Generation (RAG) system to answer Bangla and English questions from a PDF-based text corpus.

## Setup Guide

- **Prerequisites**:
  - Python 3.11+
  - Git
  - Poppler (install via: https://github.com/oschwartz10612/poppler-windows)

- **Installation**:
  1. Clone the repo:
     ```bash
     git clone https://github.com/your-username/bangla-rag.git
     cd bangla-rag
     ```
  2. Create and activate a virtual environment:
     ```bash
     python -m venv ragcon
     .\ragcon\Scripts\activate  # Windows
     ```
  3. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
  4.Set up OpenAI API key
    - Get a key from OpenAI.
    - Add to .env:
      ```bash
      GOOGLE_API_KEY = key
      OPENAI_API_KEY = key```
  5. Extract text from PDF (run OCR):
     ```bash
     python ocr_pdf.py
  6. Run the app:
     ```bash
     uvicorn app_v4:app --reload --port 8001
     ```
  7. Visit http://localhost:8001/ in your browser.


## Used Tools and Libraries
- OCR: pytesseract and pdf2image (in ocr_pdf.py) with Poppler.
- FastAPI: Web app framework.
- LangChain: RAG implementation (langchain, langchain-openai, etc.).
- OpenAI: ChatOpenAI (e.g., gpt-4o).
- HuggingFace: sentence-transformers/paraphrase-multilingual-mpnet-base-v2 for embeddings.
- FAISS: Vector store.
- pdfplumber: PDF text extraction.
- NLTK: Text preprocessing.
- NumPy & scikit-learn: Similarity calculations.
- Pillow: Image handling for OCR.
- uvicorn: ASGI server.


## Sample Queries and Outputs
### Bangla Queries
Query: "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"
- Answer: "অনুপমের ভাগ্যদেবতা হিসেবে তার মামা এবং দেবী লক্ষ্মীকে উল্লেখ করা হয়েছে।"
- Groundedness Score: 0.65
- Relevance Score: 0.78

### English Queries
Query: "What is Anupam's age?"
- Answer: "অনুপমের বয়স সম্পর্কে নির্দিষ্ট তথ্য নেই, তবে উল্লেখিত বিকল্পগুলো হতে পারে: পঁচিশ, ছাবিবিশ, সাতাশ, আটাশ।"
- Groundedness Score: ~0.70
- Relevance Score: ~0.80


## API Documentation
/ (GET): Shows a query form.
/query (POST):
Params: query (string), language (e.g., "bn", "en").
Response: HTML with answer, scores, and snippets.
/reinit (GET): Re-initializes RAG chain, returns JSON {"message": "RAG chain re-initialized"}.
Evaluation Matrix
Groundedness Score: Support from documents (0-1, cosine similarity mean).
Relevance Score: Best query-chunk match (0-1, cosine similarity max).
Must Answer Questions
Text Extraction Method:
Used pytesseract and pdf2image in ocr_pdf.py with Poppler for scanned PDFs.
Faced formatting issues (e.g., broken text), fixed with regex preprocessing.
Chunking Strategy:
RecursiveCharacterTextSplitter (size 1500, overlap 300).
Works well for semantic retrieval with fragmented text, though larger chunks could help.
Embedding Model:
sentence-transformers/paraphrase-multilingual-mpnet-base-v2.
Chosen for Bangla support and semantic accuracy, captures meaning via contextual vectors.
Query Comparison:
Cosine similarity with FAISS storage.
Chosen for robustness and speed in high-dimensional spaces.
Meaningful Comparison:
Embeddings map query and chunks to a shared space; custom prompt aids inference.
Vague queries return "দুঃখিত, এই প্রশ্নের উত্তর নির্ধারণ করা সম্ভব নয়।"
Relevance of Results:
Relevant for factual queries, not interpretive ones.
Improvements: Larger chunks, xlm-roberta-base embedding, or more document context.
