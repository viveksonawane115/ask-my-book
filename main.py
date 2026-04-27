# ============================================================
# PROJECT : ASK MY BOOK — Dual Embedding RAG with Multi-Query
# BOOK    : Altered Traits by Daniel Goleman & Richard Davidson
# ============================================================

# ── IMPORTS ──────────────────────────────────────────────────
import os
import time
import logging
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_cohere import CohereRerank
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import pandas as pd
from datetime import datetime
from langchain_groq import ChatGroq
from safety import validate_input, check_output_safety

# ── LOGGING ──────────────────────────────────────────────────
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Ask My Book",
    page_icon="📖",
    layout="centered"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .user-bubble {
        background-color: #2563eb;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        text-align: right;
    }
    .assistant-bubble {
        background-color: #f1f5f9;
        color: #1e293b;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        margin-right: auto;
    }
    .chat-meta {
        font-size: 11px;
        color: #94a3b8;
        margin: 2px 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────
st.title("📖 Ask My Book")
st.caption("Altered Traits — Daniel Goleman & Richard Davidson")
st.divider()

# ============================================================
# CACHED RESOURCES — initialize once per session
# ============================================================

@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)

@st.cache_resource
def load_llm_fallback():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

@st.cache_resource
def load_stores():
    llm = load_llm()

    loader    = TextLoader("book.txt", encoding="utf-8-sig")
    documents = loader.load()
    splitter  = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)

    embeddings_openai = OpenAIEmbeddings(model="text-embedding-ada-002")
    openai_store = Chroma(
        collection_name="openai_collection",
        embedding_function=embeddings_openai,
        persist_directory="./openai_chroma_db"
    )
    if openai_store._collection.count() == 0:
        openai_store.add_documents(chunks)

    embeddings_hf = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    hf_store = Chroma(
        collection_name="hf_collection",
        embedding_function=embeddings_hf,
        persist_directory="./hf_chroma_db"
    )
    if hf_store._collection.count() == 0:
        hf_store.add_documents(chunks)

    openai_retriever = MultiQueryRetriever.from_llm(
        retriever=openai_store.as_retriever(search_kwargs={"k": 3}),
        llm=llm
    )
    hf_retriever = MultiQueryRetriever.from_llm(
        retriever=hf_store.as_retriever(search_kwargs={"k": 3}),
        llm=llm
    )
    reranker = CohereRerank(model="rerank-english-v3.0", top_n=3)

    return openai_retriever, hf_retriever, reranker, embeddings_openai

@st.cache_resource
def load_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert assistant for the book Altered Traits.
Answer ONLY from the context provided below.
If the answer is not in the context, say 'Not found in the book.'
Be clear, concise and specific."""),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

# ── ANSWER FUNCTION ───────────────────────────────────────────

def get_answer(question, results):
    llm          = load_llm()
    llm_fallback = load_llm_fallback()
    prompt       = load_prompt()
    context      = "\n\n".join([doc.page_content for doc, score in results])
    try:
        chain    = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        return response.content, "GPT-4o-mini"
    except Exception as e:
        chain    = prompt | llm_fallback
        response = chain.invoke({"context": context, "question": question})
        return response.content, "Groq / LLaMA-3.3"

# ── EVAL DATASET ─────────────────────────────────────────────

eval_dataset = [
    {
        "question"    : "What happened to Steve Z at the Pentagon on September 11, 2001?",
        "ground_truth": "A passenger jet smashed into the Pentagon near Steve Z's office, the ceiling caved in and knocked him unconscious. Debris covered him which saved his life when the plane's fuselage exploded."
    },
    {
        "question"    : "What are the effects of long-term meditation on the amygdala?",
        "ground_truth": "Long-term meditation reduces amygdala reactivity and strengthens the connection between the prefrontal cortex and amygdala. Meditators with more lifetime hours of practice show faster amygdala recovery from distress, indicating greater resilience."
    },
    {
        "question"    : "How is vipassana meditation different from other meditation types?",
        "ground_truth": "Vipassana starts with mindfulness but transitions into meta-awareness of the processes of mind rather than just the contents of thoughts. It cultivates continuous nonreactive awareness described as mindfulness on steroids, contrasting with Tibetan practices that end in a nondual stance."
    },
    {
        "question"    : "What is the attentional blink and how does meditation affect it?",
        "ground_truth": "The attentional blink is a gap in attention immediately after spotting a target where a second stimulus is missed. After a three month vipassana retreat meditators showed a dramatic 20 percent reduction in the attentional blink, surprising scientists who believed it was hardwired."
    },
    {
        "question"    : "What did researchers find when studying yogis brains?",
        "ground_truth": "Yogis brains age more slowly, they can stop and start meditative states in seconds, show little anticipatory anxiety and rapid recovery from pain. During compassion meditation brain and heart coupling strengthens uniquely. Their resting brain states resemble others meditation states indicating the meditative state became a permanent trait."
    }
]

# ── EXCEL LOGGING ─────────────────────────────────────────────

EXCEL_PATH = "rag_evaluation_results.xlsx"

def save_to_excel(data: dict):
    df_new = pd.DataFrame([data])
    try:
        df_existing = pd.read_excel(EXCEL_PATH)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_combined = df_new
    df_combined.to_excel(EXCEL_PATH, index=False)

def evaluate_single_question(question, openai_results, openai_answer, ground_truth=None):
    llm                 = load_llm()
    _, _, _, emb_openai = load_stores()
    ragas_llm           = LangchainLLMWrapper(llm)
    ragas_embeddings    = LangchainEmbeddingsWrapper(emb_openai)
    contexts            = [doc.page_content for doc, score in openai_results]

    sample = {
        "user_input"        : question,
        "response"          : openai_answer,
        "retrieved_contexts": contexts,
        "reference"         : ground_truth if ground_truth else ""
    }

    dataset = EvaluationDataset.from_list([sample])
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]
    if ground_truth:
        metrics.append(ContextRecall())

    result    = evaluate(dataset=dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_embeddings)
    result_df = pd.DataFrame(result.scores)

    row = {
        "timestamp"         : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question"          : question,
        "answer"            : openai_answer,
        "faithfulness"      : round(float(result_df["faithfulness"].mean()), 4),
        "answer_relevancy"  : round(float(result_df["answer_relevancy"].mean()), 4),
        "context_precision" : round(float(result_df["context_precision"].mean()), 4),
        "context_recall"    : round(float(result_df["context_recall"].mean()), 4) if "context_recall" in result_df.columns else 0,
        "ground_truth_used" : "yes" if ground_truth else "no"
    }
    save_to_excel(row)
    return result

# ── INITIALIZE ────────────────────────────────────────────────

with st.spinner("⏳ Loading book and setting up pipeline — first load takes ~60 seconds..."):
    openai_retriever, hf_retriever, reranker, _ = load_stores()
    llm = load_llm()

# ── SESSION STATE ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

MAX_QUESTIONS = 10

# ── DISPLAY CHAT HISTORY ──────────────────────────────────────

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class='user-bubble'>{message['content']}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='assistant-bubble'>{message['content']}</div>
        <div class='chat-meta'>📖 {message.get('source', '')} | ⏱ {message.get('time', '')}</div>
        """, unsafe_allow_html=True)

# ── CHAT INPUT ────────────────────────────────────────────────

question = st.chat_input("Ask anything about Altered Traits...")

if question:

    if st.session_state.question_count >= MAX_QUESTIONS:
        st.warning(f"Session limit of {MAX_QUESTIONS} questions reached. Please refresh to start a new session.")
        st.stop()

    if len(question.strip()) < 10:
        st.warning("Question too short. Please ask a proper question.")
        st.stop()

    has_problem, reason = validate_input(question, llm)
    if has_problem:
        st.error(f"Question blocked: {reason}")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(f"<div class='user-bubble'>{question}</div>", unsafe_allow_html=True)

    # Retrieve + Answer
    with st.spinner("Searching the book..."):
        q_start = time.perf_counter()

        try:
            openai_docs = openai_retriever.invoke(question)
        except Exception:
            openai_docs = []

        try:
            hf_docs = hf_retriever.invoke(question)
        except Exception:
            hf_docs = []

        # Hybrid retrieval — merge both embedding results
        all_docs = openai_docs + hf_docs

        # Deduplicate by start_index position
        seen       = set()
        unique_docs = []
        for doc in all_docs:
            idx = doc.metadata.get('start_index')
            if idx not in seen:
                seen.add(idx)
                unique_docs.append(doc)

        # Rerank combined pool — Cohere picks best 3
        reranked = reranker.compress_documents(
            documents=unique_docs,
            query=question
        )
        results     = [(doc, doc.metadata.get("relevance_score", 1.0)) for doc in reranked]

        answer, llm_used = get_answer(question, results)

        if check_output_safety(str(answer)):
            answer = "I cannot provide that response."

        elapsed = round(time.perf_counter() - q_start, 2)

    # Display answer
    st.markdown(f"""
    <div class='assistant-bubble'>{answer}</div>
    <div class='chat-meta'>📖 Altered Traits | 🤖 {llm_used} | ⏱ {elapsed}s</div>
    """, unsafe_allow_html=True)

    # Add to history
    st.session_state.messages.append({
        "role"   : "assistant",
        "content": answer,
        "source" : f"Altered Traits | {llm_used}",
        "time"   : f"{elapsed}s"
    })

    st.session_state.question_count += 1

    # Silent RAGAS evaluation
    gt = None
    for item in eval_dataset:
        if item["question"].lower().strip() == question.lower().strip():
            gt = item["ground_truth"]
            break

    try:
        evaluate_single_question(
            question       = question,
            openai_results = results,
            openai_answer  = answer,
            ground_truth   = gt
        )
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")

# ── SIDEBAR ───────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📖 Ask My Book")
    st.markdown("**Book:** Altered Traits")
    st.markdown("**Authors:** Goleman & Davidson")
    st.divider()
    st.markdown("### ⚙️ Pipeline")
    st.markdown("""
- Dual Embeddings (OpenAI + HuggingFace)
- Multi-Query Retrieval
- Cohere Reranking
- GPT-4o-mini (Groq fallback)
- RAGAS Evaluation
- Guardrails & Safety
    """)
    st.divider()
    st.markdown(f"**Questions this session:** {st.session_state.question_count}/{MAX_QUESTIONS}")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.question_count = 0
        st.rerun()