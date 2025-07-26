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
- / (GET): Shows a query form.
-  **/query (POST)**:
  - Params: query (string), language (e.g., "bn", "en").
  - Response: HTML with answer, scores, and snippets.
- /reinit (GET): Re-initializes RAG chain, returns JSON {"message": "RAG chain re-initialized"}.


## Evaluation Matrix
- **Groundedness Score**: Measures how well the answer is supported by source documents (0 to 1, based on the mean of cosine similarity).
- **Relevance Score**: Indicates the best match between the query and document chunks (0 to 1, based on the maximum cosine similarity).

## Must Answer Questions

- **Text Extraction Method**:
  - Used `pytesseract` and `pdf2image` in `ocr_pdf.py` with Poppler to process scanned PDFs.
  - Faced formatting issues (e.g., broken text, mixed numerals), mitigated with regex preprocessing in the RAG app.

- **Chunking Strategy**:
  - Employed `RecursiveCharacterTextSplitter` with a chunk size of 1500 and an overlap of 300.
  - Works well for semantic retrieval with fragmented text, though larger chunks might improve narrative coherence.

- **Embedding Model**:
  - Utilized `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
  - Chosen for its multilingual support (including Bangla) and strong semantic accuracy, capturing meaning through contextual word relationships.

- **Query Comparison**:
  - Applied cosine similarity with FAISS for storage and retrieval.
  - Selected for its robustness and speed in handling high-dimensional vector spaces.

- **Meaningful Comparison**:
  - Embeddings map both query and chunks to a shared semantic space; a custom prompt enhances inference when explicit answers are lacking.
  - Vague or context-missing queries result in "দুঃখিত, এই প্রশ্নের উত্তর নির্ধারণ করা সম্ভব নয়।"

- **Relevance of Results**:
  - Results are relevant for factual queries (e.g., "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?") but not for interpretive ones.
  - Potential improvements include larger chunk sizes, switching to `xlm-roberta-base` for better Bangla support, or adding more contextual document content.
