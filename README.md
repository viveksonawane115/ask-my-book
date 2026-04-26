# 📖 Ask My Book — RAG-Powered Book Q&A

> Ask any question about **Altered Traits** by Daniel Goleman & Richard Davidson.  
> Powered by Dual Embedding RAG, Multi-Query Retrieval, Cohere Reranking, and GPT-4o-mini.

---

## 🔗 Live Demo

**[http://54.84.40.25:8501](http://54.84.40.25:8501)**

---

## 🧠 What It Does

A production-aware RAG (Retrieval Augmented Generation) pipeline that lets you have a chat-style conversation with a book. Instead of the LLM answering from its training data, it retrieves relevant passages directly from the book and answers only from those.

---

## 🏗️ Architecture

```
User Question
      ↓
MultiQueryRetriever (LLM generates 3 query variants)
      ↓
Dual ChromaDB Search
├── OpenAI text-embedding-ada-002 (1536 dimensions)
└── HuggingFace all-MiniLM-L6-v2 (384 dimensions)
      ↓
Cohere Reranker (rerank-english-v3.0, top 3)
      ↓
GPT-4o-mini → Answer (Groq/LLaMA-3.3 fallback)
      ↓
Guardrails (input + output safety check)
      ↓
RAGAS Evaluation (silent, logged to Excel)
```

---

## ✨ Features

- **Dual Embedding Comparison** — OpenAI and HuggingFace embeddings run in parallel
- **Multi-Query Retrieval** — LLM generates 3 query variants per question to improve recall
- **Cohere Reranking** — Cross-encoder reranks retrieved chunks for precision
- **LLM Fallback** — GPT-4o-mini → Groq/LLaMA-3.3 if OpenAI fails
- **Guardrails** — Pattern + LLM-judge injection detection, PII check, output safety
- **RAGAS Evaluation** — Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **Chat UI** — Conversation history with session state
- **Rate Limiting** — 10 questions per session
- **Excel Audit Trail** — Every evaluation result logged with timestamp
- **Idempotent Embeddings** — ChromaDB skips re-embedding if already populated

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | GPT-4o-mini (OpenAI) |
| Fallback LLM | LLaMA-3.3-70b (Groq) |
| Embeddings | text-embedding-ada-002 + all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| Reranker | Cohere rerank-english-v3.0 |
| Retriever | LangChain MultiQueryRetriever |
| Evaluation | RAGAS |
| UI | Streamlit |
| Deployment | Docker + AWS EC2 (t3.small) |

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/viveksonawane115/ask-my-book.git
cd ask-my-book
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 5. Run the app
```bash
streamlit run main.py
```

---

## 🔑 Environment Variables

```
OPENAI_API_KEY=your-openai-key
COHERE_API_KEY=your-cohere-key
GROQ_API_KEY=your-groq-key
```

Get your keys from:
- OpenAI → [platform.openai.com](https://platform.openai.com)
- Cohere → [dashboard.cohere.com](https://dashboard.cohere.com)
- Groq → [console.groq.com](https://console.groq.com)

---

## 🐳 Run With Docker

```bash
# Build and run
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 📊 RAGAS Evaluation Results

Evaluated on 5 curated questions with ground truth answers.

| Metric | Score |
|--------|-------|
| Faithfulness | — |
| Answer Relevancy | — |
| Context Precision | — |
| Context Recall | — |

> Run the app and ask the 5 eval questions to populate scores.

---

## 📁 Project Structure

```
ask-my-book/
├── main.py              # Streamlit app + RAG pipeline
├── safety.py            # Guardrails (injection, PII, output safety)
├── book.txt             # Altered Traits full text
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
├── docker-compose.yml   # Container orchestration
├── .env.example         # API key template
└── README.md            # This file
```

---

## 🔒 Safety & Guardrails

Three-layer input protection:
1. **PII Detection** — regex patterns for email, phone, Aadhar, PAN
2. **Pattern Injection** — blocklist of known prompt injection phrases
3. **LLM Judge** — GPT-4o-mini classifies question as SAFE/UNSAFE

Output protection:
- Scans response for unsafe content before displaying to user

---

## 📝 Resume Bullets

- Built end-to-end RAG pipeline with dual embedding comparison (OpenAI + HuggingFace), MultiQuery retrieval, and Cohere reranking on a 600-page book
- Implemented three-layer guardrails system with PII detection, pattern matching, and LLM-as-judge for prompt injection prevention
- Integrated RAGAS evaluation framework measuring Faithfulness, Answer Relevancy, Context Precision, and Context Recall with Excel audit logging
- Deployed Dockerized Streamlit application on AWS EC2 with LLM fallback chain (GPT-4o-mini → Groq/LLaMA-3.3) for production resiliency

---

## 🗺️ Roadmap

- [ ] Persistent conversation memory across sessions
- [ ] Support multiple books
- [ ] Better chunking strategy (semantic chunking)
- [ ] Authentication system
- [ ] CI/CD pipeline

---

## 👤 Author

**Vivek Sonawane**  
[GitHub](https://github.com/viveksonawane115)