# Construction Project Delay AI

A simple Streamlit RAG application for construction project delay-risk analysis.

## Architecture

PDF
-> PyMuPDF text extraction
-> overlapping document chunks
-> open-source Sentence Transformer embeddings
-> FAISS vector search
-> focused retrieval for construction risk categories
-> Groq LLM
-> structured problems + recommendations + evidence

## Files

All Python files are intentionally in the repository root for easy GitHub upload.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `GROQ_API_KEY` as an environment variable or Streamlit secret.

## Streamlit Cloud

1. Upload all files to GitHub.
2. Create a Streamlit Cloud app from the repository.
3. Main file: `app.py`.
4. In App Settings -> Secrets, add:

```toml
GROQ_API_KEY = "your_actual_key"
```

Do not commit the actual API key.

## Important

This MVP uses an evidence-based risk screening score, not a trained statistical delay-probability model. The RAG pipeline retrieves document evidence and Groq generates evidence-based construction findings and recommendations.

A future production version can add historical project data, a trained delay prediction model, OCR for scanned PDFs, Primavera/MS Project integration, weather APIs, and real-time project feeds.
