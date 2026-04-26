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

# ── HEADER ───────────────────────────────────────────────────
st.title("📖 Ask My Book")
st.caption("Altered Traits — Daniel Goleman & Richard Davidson")
st.markdown("Ask any question about the book. Powered by **Dual Embedding RAG** + **Multi-Query Retrieval** + **Cohere Reranking**.")
st.divider()

# ============================================================
# BLOCK 1 — INITIALIZE LLM
# Wrapped in cache_resource — runs only ONCE per session
# Without this, Streamlit reinitializes LLM on every click
# ============================================================

@st.cache_resource
def load_llm():
    """GPT-4o-mini: cost efficient, temperature=0 for factual answers"""
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
    return llm

@st.cache_resource
def load_llm_fallback():
    """Groq LLaMA-3.3: fallback if OpenAI fails"""
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ============================================================
# BLOCK 2+3 — LOAD DOCUMENT, CHUNK, EMBED, STORE IN CHROMADB
# Wrapped in cache_resource — runs only ONCE per session
# Idempotent: skips re-embedding if ChromaDB already populated
# ============================================================

@st.cache_resource
def load_stores():
    """
    Loads book.txt, splits into chunks, creates both ChromaDB stores.
    Skips re-embedding if stores already populated (saves time + money).
    Returns both retrievers and reranker.
    """
    llm = load_llm()

    # ── Load + Split ─────────────────────────────────────────
    loader    = TextLoader("book.txt", encoding="utf-8-sig")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)

    # ── OpenAI Embeddings → ChromaDB ─────────────────────────
    embeddings_openai = OpenAIEmbeddings(model="text-embedding-ada-002")
    openai_store = Chroma(
        collection_name="openai_collection",
        embedding_function=embeddings_openai,
        persist_directory="./openai_chroma_db"
    )
    if openai_store._collection.count() == 0:
        openai_store.add_documents(chunks)

    # ── HuggingFace Embeddings → ChromaDB ────────────────────
    embeddings_hf = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    hf_store = Chroma(
        collection_name="hf_collection",
        embedding_function=embeddings_hf,
        persist_directory="./hf_chroma_db"
    )
    if hf_store._collection.count() == 0:
        hf_store.add_documents(chunks)

    # ── MultiQuery Retrievers ─────────────────────────────────
    # Created ONCE — reused for every question
    # MultiQuery: LLM generates 3 variations → searches all → deduplicates
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

# ============================================================
# BLOCK 4 — PROMPT TEMPLATE
# Strict instruction — answer only from context
# Prevents LLM from using its own training knowledge
# ============================================================

@st.cache_resource
def load_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert assistant.
Answer ONLY from the context provided below.
If the answer is not in the context, say 'Not found in the book.'
Be clear, concise and specific."""),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

# ============================================================
# BLOCK 5 — ANSWER GENERATOR
# Takes question + retrieved (doc, score) tuples
# Joins chunks into context string → sends to LLM
# Falls back to Groq if OpenAI fails
# ============================================================

def get_answer(question, results):
    """
    Generate answer from LLM using retrieved chunks as context.
    Args:
        question : user's original question
        results  : list of (Document, score) tuples
    Returns:
        (answer string, llm name used)
    """
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

# ============================================================
# EVAL DATASET — Ground truth for RAGAS evaluation
# ============================================================

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

# ============================================================
# EXCEL LOGGING
# ============================================================

EXCEL_PATH = "rag_evaluation_results.xlsx"

def save_to_excel(data: dict):
    """Saves one evaluation result row to Excel. Appends if file exists."""
    df_new = pd.DataFrame([data])
    try:
        df_existing = pd.read_excel(EXCEL_PATH)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_combined = df_new
    df_combined.to_excel(EXCEL_PATH, index=False)

def evaluate_single_question(question, openai_results, hf_results, openai_answer, hf_answer, ground_truth=None):
    """Runs RAGAS on a single question and saves to Excel."""
    llm               = load_llm()
    _, _, _, emb_openai = load_stores()
    ragas_llm         = LangchainLLMWrapper(llm)
    ragas_embeddings  = LangchainEmbeddingsWrapper(emb_openai)

    contexts = [doc.page_content for doc, score in openai_results]
    sample   = {
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
        "openai_answer"     : openai_answer,
        "hf_answer"         : hf_answer,
        "faithfulness"      : round(float(result_df["faithfulness"].mean()), 4),
        "answer_relevancy"  : round(float(result_df["answer_relevancy"].mean()), 4),
        "context_precision" : round(float(result_df["context_precision"].mean()), 4),
        "context_recall"    : round(float(result_df["context_recall"].mean()), 4) if "context_recall" in result_df.columns else 0,
        "openai_chunks"     : str([doc.metadata.get('start_index') for doc, score in openai_results]),
        "hf_chunks"         : str([doc.metadata.get('start_index') for doc, score in hf_results]),
        "ground_truth_used" : "yes" if ground_truth else "no"
    }
    save_to_excel(row)
    return result

# ============================================================
# INITIALIZE — runs once, cached for the session
# ============================================================

with st.spinner("⏳ Loading book and setting up retrievers — first load takes ~30 seconds..."):
    openai_retriever, hf_retriever, reranker, _ = load_stores()
    llm = load_llm()

# ============================================================
# STREAMLIT UI — replaces the while True CLI loop
# ============================================================

question   = st.text_input(
    "Your Question",
    placeholder="e.g. What are the effects of long-term meditation on the amygdala?"
)
ask_button = st.button("Ask", type="primary")

# ── MAIN LOGIC ───────────────────────────────────────────────

if ask_button and question:

    # ── Input validation ─────────────────────────────────────
    if len(question.strip()) < 10:
        st.warning("Question too short. Please ask a proper question.")
        st.stop()

    has_problem, reason = validate_input(question, llm)
    if has_problem:
        st.error(f"Question blocked: {reason}")
        st.stop()

    # ── Retrieval ────────────────────────────────────────────
    with st.spinner("Running Multi-Query Retrieval..."):
        try:
            openai_docs = openai_retriever.invoke(question)
        except Exception as e:
            logger.warning(f"OpenAI retriever failed: {e}. Falling back.")
            openai_docs = []

        try:
            hf_docs = hf_retriever.invoke(question)
        except Exception as e:
            logger.warning(f"HF retriever failed: {e}. Falling back.")
            hf_docs = []

        reranked_openai = reranker.compress_documents(documents=openai_docs, query=question)
        reranked_hf     = reranker.compress_documents(documents=hf_docs,     query=question)

        openai_results  = [(doc, doc.metadata.get("relevance_score", 1.0)) for doc in reranked_openai]
        hf_results      = [(doc, doc.metadata.get("relevance_score", 1.0)) for doc in reranked_hf]

    # ── Answer generation ────────────────────────────────────
    with st.spinner("Generating answers..."):
        openai_answer, openai_llm_used = get_answer(question, openai_results)
        hf_answer,     hf_llm_used     = get_answer(question, hf_results)

        if check_output_safety(str(openai_answer)):
            openai_answer = "Response blocked by safety filter."
        if check_output_safety(str(hf_answer)):
            hf_answer = "Response blocked by safety filter."

    # ── Display answers ──────────────────────────────────────
    st.subheader("Answers")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**OpenAI Embeddings**")
        st.info(openai_answer)
        st.caption(f"LLM: {openai_llm_used}")

    with col2:
        st.markdown("**HuggingFace Embeddings**")
        st.info(hf_answer)
        st.caption(f"LLM: {hf_llm_used}")

    # ── Retrieval comparison ──────────────────────────────────
    with st.expander("🔍 Retrieval Comparison"):
        openai_positions = set([doc.metadata.get('start_index') for doc, _ in openai_results])
        hf_positions     = set([doc.metadata.get('start_index') for doc, _ in hf_results])
        common           = openai_positions & hf_positions
        only_openai      = openai_positions - hf_positions
        only_hf          = hf_positions - openai_positions

        c1, c2, c3 = st.columns(3)
        c1.metric("Both agreed", len(common))
        c2.metric("Only OpenAI", len(only_openai))
        c3.metric("Only HuggingFace", len(only_hf))

        if len(common) == len(openai_positions) == len(hf_positions):
            st.success("Both models agreed completely — same chunks retrieved")
        elif len(common) == 0:
            st.warning("Models disagreed completely — retrieved different chunks")
        else:
            st.info(f"Partial agreement — {len(common)} common, {len(only_openai)+len(only_hf)} different")

    # ── Chunk previews ────────────────────────────────────────
    with st.expander("📄 Retrieved Chunks — OpenAI"):
        for i, (doc, score) in enumerate(openai_results):
            st.markdown(f"**Chunk {i+1}** | Position: `{doc.metadata.get('start_index', 'N/A')}`")
            st.text(doc.page_content[:300] + "...")
            st.divider()

    with st.expander("📄 Retrieved Chunks — HuggingFace"):
        for i, (doc, score) in enumerate(hf_results):
            st.markdown(f"**Chunk {i+1}** | Position: `{doc.metadata.get('start_index', 'N/A')}`")
            st.text(doc.page_content[:300] + "...")
            st.divider()

    # ── RAGAS evaluation ──────────────────────────────────────
    gt = None
    for item in eval_dataset:
        if item["question"].lower().strip() == question.lower().strip():
            gt = item["ground_truth"]
            break

    with st.spinner("Running RAGAS evaluation..."):
        eval_result = evaluate_single_question(
            question      = question,
            openai_results= openai_results,
            hf_results    = hf_results,
            openai_answer = openai_answer,
            hf_answer     = hf_answer,
            ground_truth  = gt
        )

    with st.expander("📊 RAGAS Evaluation Scores"):
        result_df = pd.DataFrame(eval_result.scores)
        r1, r2, r3 = st.columns(3)
        r1.metric("Faithfulness",       round(float(result_df["faithfulness"].mean()), 4))
        r2.metric("Answer Relevancy",   round(float(result_df["answer_relevancy"].mean()), 4))
        r3.metric("Context Precision",  round(float(result_df["context_precision"].mean()), 4))
        if gt:
            st.metric("Context Recall", round(float(result_df["context_recall"].mean()), 4))
        st.caption("Results also saved to rag_evaluation_results.xlsx")