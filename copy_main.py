# ============================================================
# PROJECT : ASK MY BOOK — Dual Embedding RAG with Multi-Query
# BOOK    : Altered Traits by Daniel Goleman & Richard Davidson
# ============================================================

# ── IMPORTS ──────────────────────────────────────────────────
import os
import time
import logging
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import MultiQueryRetriever
#from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_cohere import CohereRerank
from ragas import evaluate,EvaluationDataset 
#from ragas.metrics.collections import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import pandas as pd
from datetime import datetime
import numpy as np
from langchain_groq import ChatGroq
from safety import validate_input, check_output_safety
# ── LOGGING — shows generated queries from MultiQueryRetriever
logging.basicConfig()
#logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
logger = logging.getLogger(__name__)
# ============================================================
# BLOCK 1 — INITIALIZE LLM
# GPT-4o-mini: cost efficient, temperature=0 for factual answers
# ============================================================
print("\n" + "="*60)
print("BLOCK 1 — Initializing LLM")
print("="*60)

llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
print("✅ LLM initialized: gpt-4o-mini | temperature=0")

llm_fallback=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)
# ============================================================
# BLOCK 2 — LOAD DOCUMENT AND SPLIT INTO CHUNKS
# TextLoader reads the book as a single document
# RecursiveCharacterTextSplitter cuts it into overlapping chunks
# chunk_size=1000   : each chunk is max 1000 characters
# chunk_overlap=250 : 250 char overlap prevents answer loss at boundaries
# add_start_index   : each chunk remembers its position in the book
# ============================================================
print("\n" + "="*60)
print("BLOCK 2 — Loading and Chunking Document")
print("="*60)

start = time.perf_counter()

loader = TextLoader("book.txt", encoding="utf-8-sig")
documents = loader.load()
print(f"✅ Document loaded")
print(f"   Total characters in book : {len(documents[0].page_content)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=250,
    add_start_index=True
)
chunks = splitter.split_documents(documents)

print(f"✅ Splitting complete")
print(f"   Total chunks created     : {len(chunks)}")
print(f"   First chunk position     : {chunks[0].metadata.get('start_index')}")
print(f"   First chunk length       : {len(chunks[0].page_content)} characters")
print(f"   First chunk preview      : {chunks[0].page_content[:150]}...")

print(f"\n  Load + Split time: {time.perf_counter() - start:.2f}s")

# ============================================================
# BLOCK 3 — CREATE EMBEDDINGS AND STORE IN CHROMADB
# Two embedding models used for comparison:
#
# OpenAI text-embedding-ada-002
#   → 1536 dimensions, paid, cloud-based
#   → better semantic understanding of nuanced text
#
# HuggingFace all-MiniLM-L6-v2
#   → 384 dimensions, free, runs locally
#   → faster, good for keyword-heavy queries
#
# Idempotent check: if ChromaDB already has chunks → skip
# This avoids re-embedding on every run (saves time and money)
# ============================================================
print("\n" + "="*60)
print("BLOCK 3 — Creating Embeddings and Storing in ChromaDB")
print("="*60)

start = time.perf_counter()

# OpenAI Embeddings
embeddings_openai = OpenAIEmbeddings(model="text-embedding-ada-002")
openai_store = Chroma(
    collection_name="openai_collection",
    embedding_function=embeddings_openai,
    persist_directory="./openai_chroma_db"
)
if openai_store._collection.count() == 0:
    print(" OpenAI — embedding 772 chunks (first run only)...")
    openai_store.add_documents(chunks)
    print(f" OpenAI store created — {openai_store._collection.count()} chunks stored")
else:
    print(f" OpenAI store ready — {openai_store._collection.count()} chunks (skipping re-embedding)")

# HuggingFace Embeddings
embeddings_hf = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
hf_store = Chroma(
    collection_name="hf_collection",
    embedding_function=embeddings_hf,
    persist_directory="./hf_chroma_db"
)
if hf_store._collection.count() == 0:
    print("⏳ HuggingFace — embedding 772 chunks (first run only)...")
    hf_store.add_documents(chunks)
    print(f" HuggingFace store created — {hf_store._collection.count()} chunks stored")
else:
    print(f" HuggingFace store ready — {hf_store._collection.count()} chunks (skipping re-embedding)")

print(f"\n  Embedding setup time: {time.perf_counter() - start:.2f}s")

# ============================================================
# BLOCK 4 — INITIALIZE MULTI-QUERY RETRIEVERS
# MultiQueryRetriever solves the single interpretation problem
# It uses the LLM to generate 3 variations of each question
# Searches ChromaDB with all 3 versions
# Returns all unique chunks after deduplication
#
# Created ONCE outside the loop — reused for every question
# Creating inside loop would waste resources
# ============================================================
print("\n" + "="*60)
print("BLOCK 4 — Initializing Multi-Query Retrievers")
print("="*60)

openai_retriever = MultiQueryRetriever.from_llm(
    retriever=openai_store.as_retriever(search_kwargs={"k": 3}),
    llm=llm
)
print(" OpenAI MultiQueryRetriever ready")

hf_retriever = MultiQueryRetriever.from_llm(
    retriever=hf_store.as_retriever(search_kwargs={"k": 3}),
    llm=llm
)

print("\n" + "="*60)
print("\n ASK MY BOOK — EMBEDDING COMPARISON")
print("\n Powered by Multi-Query Retrieval + GPT-4o-mini")
print("="*60)


reranker=CohereRerank(model="rerank-english-v3.0", top_n=3)
print(" HuggingFace MultiQueryRetriever ready")

# ============================================================
# BLOCK 5 — PROMPT TEMPLATE
# system: strict instruction — answer only from context
#         prevents LLM from using its own training knowledge
#         forces grounded, faithful answers
# human:  passes the retrieved chunks + user question
#
# Defined ONCE outside loop — same prompt used for every question
# ============================================================
print("\n" + "="*60)
print("BLOCK 5 — Building Prompt Template")
print("="*60)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant.
Answer ONLY from the context provided below.
If the answer is not in the context, say 'Not found in the book.'
Be clear, concise and specific."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])
print(" Prompt template ready")

# ============================================================
# BLOCK 6 — ANSWER GENERATOR FUNCTION
# Takes question + retrieved results
# Joins all chunk texts into single context string
# Sends context + question to LLM via chain
# Returns LLM response as string
#
# Called twice per question:
#   once with OpenAI chunks → OpenAI answer
#   once with HF chunks     → HF answer
# ============================================================

def get_answer(question, results):
    """
    Generate answer from LLM using retrieved chunks as context.
    
    Args:
        question: user's original question
        results:  list of (Document, score) tuples
    Returns:
        LLM generated answer string
    """
    context = "\n\n".join([doc.page_content for doc, score in results])
    try:
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        return response.content, "OpenAI"
    except Exception as e:
        print(f"OpenAI failed: {e}. Switching to Groq...")
        chain = prompt | llm_fallback
        response = chain.invoke({"context": context, "question": question})
        return response.content, "Groq"
    

print(" Answer generator function ready")

# ============================================================
# MAIN LOOP — QUESTION → RETRIEVE → COMPARE → ANSWER
# Runs until user types 'exit'
# Each iteration:
#   1. Takes user question
#   2. MultiQuery generates 3 variations per store
#   3. Retrieves unique chunks from both stores
#   4. Compares which chunks each model found
#   5. Generates answer using both sets of chunks
#   6. Displays both answers for comparison
# ============================================================


eval_dataset = [
    {
        "question": "What happened to Steve Z at the Pentagon on September 11, 2001?",
        "ground_truth": "A passenger jet smashed into the Pentagon near Steve Z's office, the ceiling caved in and knocked him unconscious. Debris covered him which saved his life when the plane's fuselage exploded."
    },
    {
        "question": "What are the effects of long-term meditation on the amygdala?",
        "ground_truth": "Long-term meditation reduces amygdala reactivity and strengthens the connection between the prefrontal cortex and amygdala. Meditators with more lifetime hours of practice show faster amygdala recovery from distress, indicating greater resilience."
    },
    {
        "question": "How is vipassana meditation different from other meditation types?",
        "ground_truth": "Vipassana starts with mindfulness but transitions into meta-awareness of the processes of mind rather than just the contents of thoughts. It cultivates continuous nonreactive awareness described as mindfulness on steroids, contrasting with Tibetan practices that end in a nondual stance."
    },
    {
        "question": "What is the attentional blink and how does meditation affect it?",
        "ground_truth": "The attentional blink is a gap in attention immediately after spotting a target where a second stimulus is missed. After a three month vipassana retreat meditators showed a dramatic 20 percent reduction in the attentional blink, surprising scientists who believed it was hardwired."
    },
    {
        "question": "What did researchers find when studying yogis brains?",
        "ground_truth": "Yogis brains age more slowly, they can stop and start meditative states in seconds, show little anticipatory anxiety and rapid recovery from pain. During compassion meditation brain and heart coupling strengthens uniquely. Their resting brain states resemble others meditation states indicating the meditative state became a permanent trait."
    }
]

ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings_openai)
EXCEL_PATH = "rag_evaluation_results.xlsx"
def save_to_excel(data: dict):
    """
    Saves one evaluation result row to Excel.
    Creates file if not exists.
    Appends if file already exists.
    """
    df_new = pd.DataFrame([data])
    
    try:
        # File exists — append
        df_existing = pd.read_excel(EXCEL_PATH)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        # File does not exist — create
        df_combined = df_new
    
    df_combined.to_excel(EXCEL_PATH, index=False)
    print(f"Results saved to {EXCEL_PATH}")
    

def evaluate_single_question(question, openai_results, hf_results, openai_answer, hf_answer, ground_truth=None):
    """
    Runs RAGAS on a single question and saves to Excel.
    ground_truth is optional — used for context_recall if available.
    """
    contexts = [doc.page_content for doc, score in openai_results]
    
    sample = {
        "user_input": question,
        "response": openai_answer,
        "retrieved_contexts": contexts,
        "reference": ground_truth if ground_truth else ""
    }
    
    dataset = EvaluationDataset.from_list([sample])
    
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]
    if ground_truth:
        metrics.append(ContextRecall())
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )
    
    result_df = result.to_pandas()
    
    # Build Excel row
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

def run_ragas_evaluation():
    print("\n" + "="*60)
    print("📊 RUNNING RAGAS EVALUATION")
    print("="*60)
    
    samples = []
    
    for item in eval_dataset:
        q = item["question"]
        gt = item["ground_truth"]
        
        # Run your pipeline for this question
        openai_docs      = openai_retriever.invoke(q)
        reranked_openai  = reranker.compress_documents(documents=openai_docs, query=q)
        openai_results   = [(doc, doc.metadata.get("relevance_score", 1.0)) for doc in reranked_openai]
        
        # Get answer from LLM
        answer,_ = get_answer(q, openai_results)
        
        # Get contexts — plain text list
        contexts = [doc.page_content for doc, score in openai_results]
        
        # Build sample
        samples.append({
            "user_input": q,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": gt
        })
        
        print(f"✅ Evaluated: {q[:50]}...")
    
    # Run RAGAS
    dataset = EvaluationDataset.from_list(samples)
    
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    
    print("\n" + "="*60)
    print("📊 RAGAS SCORES")
    print("="*60)
    print(f"Faithfulness      : {result['Faithfulness']:.4f}")
    print(f"Answer Relevancy  : {result['AnswerRelevancy']:.4f}")
    print(f"Context Recall    : {result['ContextRecall']:.4f}")
    print(f"Context Precision : {result['ContextPrecision']:.4f}")
    
    return result

exit_words = ["exit", "quit", "bye", "stop", "close", "q"]

while True:

    question = input("\n Your Question (or type 'exit' to quit): ")

    if question.strip().lower() in exit_words:
        print("\n Goodbye!")
        break
    
    if len(question.strip()) < 10:
        print("Question too short. Please ask a proper question.")
        continue
    
    has_problem, reason = validate_input(question, llm)
    if has_problem:
        logger.warning(f"Question blocked: {reason}")
        print(f"Question blocked: {reason}")
        continue
    
    q_start = time.perf_counter()
    print(f"\n Question received: '{question}'")

    # ── RETRIEVAL ─────────────────────────────────────────────
    # MultiQueryRetriever internally:
    # Step 1 → LLM generates 3 question variations
    # Step 2 → searches ChromaDB with each variation (k=3 each)
    # Step 3 → deduplicates results
    # Step 4 → returns unique Document objects

    print("\n Running Multi-Query Retrieval...")
    print("\n \n [OpenAI store]")
    try:
        openai_docs      = openai_retriever.invoke(question)
        print(f"\n {len(openai_docs)} unique chunks retrieved after deduplication")
    except Exception as e:
        logger.warning(f"MultiQuery failed for OpenAI: {e}. Falling back to simple search.")
        openai_docs=hf_docs=hf_store.similarity_search(question, k=3)
        print(f"\n MultiQuery failed for OpenAI. Using simple search. Got {len(openai_docs)} chunks")
        
    #openai_docs    = openai_retriever.invoke(question)
    print(f"\n {len(openai_docs)} unique chunks retrieved after deduplication")
    print(f"\n OpenAI chunk positions: {set([doc.metadata.get('start_index') for doc in openai_docs])}")
    
    #print("\n Openai_docs = ",openai_docs)
    print("\n [HuggingFace store]")
    
    try:
        hf_docs = hf_retriever.invoke(question)
        print(f"\n {len(hf_docs)} unique chunks retrieved after deduplication")
    except Exception as e:
        logger.warning(f"MultiQuery failed for Hugging Face: {e}. Falling back to simple search.")
        hf_docs=hf_store.similarity_search(question, k=3)
        print(f"\n MultiQuery failed for Hugging Face. Using simple search. Got {len(hf_docs)} chunks")
        
    print(f"\n {len(hf_docs)} unique chunks retrieved after deduplication")
    print(f"\n HuggingFace chunk positions: {set([doc.metadata.get('start_index') for doc in hf_docs])}")
    
    # Convert Document objects to (doc, score) tuples
    # Score=1.0 is placeholder — reranker will replace this later
    #openai_results = [(doc, 1.0) for doc in openai_docs]
    #hf_results     = [(doc, 1.0) for doc in hf_docs]
    
    # openai_results = [(doc, doc.metadata.get("relevance_score", 1.0)) 
    #               for doc in reranked_openai]
    # hf_results     = [(doc, doc.metadata.get("relevance_score", 1.0)) 
    #               for doc in reranked_hf]

    # ── COMPARISON ────────────────────────────────────────────
    # Compare which chunks each model retrieved
    # Uses start_index position as unique chunk identifier
    # Set operations find common and unique chunks instantly

    
    
    reranked_openai  = reranker.compress_documents(documents=openai_docs,query=question)
    reranked_openai_chunks = [i.metadata for i in reranked_openai]
    print("\n RERANKED OpeanAI DOC METADATA:",reranked_openai_chunks)
    
    reranked_hf = reranker.compress_documents(documents=hf_docs,query=question)
    reranked_hf_chunks = [i.metadata for i in reranked_hf]
    print("\n \n RERANKED HF DOC METADATA:",reranked_hf_chunks)
    
    openai_results = [(doc, doc.metadata.get("relevance_score", 1.0)) for doc in reranked_openai]
    hf_results     = [(doc, doc.metadata.get("relevance_score", 1.0)) for doc in reranked_hf]
    
    
    openai_positions = set([doc.metadata.get('start_index') for doc, score in openai_results])
    hf_positions     = set([doc.metadata.get('start_index') for doc, score in hf_results])
    common      = openai_positions.intersection(hf_positions)
    only_openai = openai_positions - hf_positions
    only_hf     = hf_positions - openai_positions

    print("\n" + "="*60)
    print("\n RETRIEVAL COMPARISON")
    print("="*60)
    print(f"\n Chunks found by BOTH models  : {len(common)}")
    print(f"\n Chunks found ONLY by OpenAI  : {len(only_openai)}")
    print(f"\n Chunks found ONLY by HuggFace: {len(only_hf)}")

    if common:
        print(f"\n Common chunk positions: {common}")
    if only_openai:
        print(f"\n OpenAI-only positions : {only_openai}")
    if only_hf:
        print(f"\n HuggFace-only positions: {only_hf}")

    if len(common) == len(openai_positions) == len(hf_positions):
        print("\n Both models agreed completely — same chunks retrieved")
    elif len(common) == 0:
        print("\n Models disagreed completely — retrieved different chunks")
        print("\n Likely cause: vocabulary mismatch or semantic gap")
    else:
        print(f"\n Partial agreement — {len(common)} common, {len(only_openai)+len(only_hf)} different")

    # ── CHUNK PREVIEW ─────────────────────────────────────────
    print("\n" + "="*60)
    print("\n CHUNKS RETRIEVED — OPENAI EMBEDDINGS")
    print("="*60)
    for i, (doc, score) in enumerate(openai_results):
        print(f"\n Chunk {i+1} | Position in book: {doc.metadata.get('start_index', 'N/A')}")
        print(f"\n Preview: {doc.page_content[:200]}...")

    print("\n" + "="*60)
    print("\n CHUNKS RETRIEVED — HUGGINGFACE EMBEDDINGS")
    print("="*60)
    for i, (doc, score) in enumerate(hf_results):
        print(f"\n Chunk {i+1} | Position in book: {doc.metadata.get('start_index', 'N/A')}")
        print(f"\n Preview: {doc.page_content[:200]}...")

    # ── ANSWER GENERATION ─────────────────────────────────────
    # LLM receives retrieved chunks as context
    # Strict prompt ensures answer comes only from book
    # Called twice — once per embedding model's chunks

    print("\n" + "="*60)
    print("\n GENERATING ANSWERS")
    print("="*60)

    print("\n Answer using OpenAI chunks:")
    print("-"*40)
    openai_answer,openai_llm_used  = get_answer(question, openai_results)
    if check_output_safety(str(openai_answer)):
        openai_answer = "Response blocked by safety filter."
    print(openai_answer)

    print("\n Answer using HuggingFace chunks:")
    print("-"*40)
    hf_answer,hf_llm_used  = get_answer(question, hf_results)
    if check_output_safety(str(hf_answer)):
        hf_answer = "Response blocked by safety filter."
    print(hf_answer)

    elapsed = time.perf_counter() - q_start
    print(f"\n Total time for this question: {elapsed:.2f}s")
    print("="*60)

    # Check if this question has ground truth available
    gt = None
    for item in eval_dataset:
        if item["question"].lower().strip() == question.lower().strip():
            gt = item["ground_truth"]
            break

    # Run RAGAS evaluation
    print("\n Running RAGAS evaluation...")
    eval_result = evaluate_single_question(
        question=question,
        openai_results=openai_results,
        hf_results=hf_results,
        openai_answer=openai_answer,
        hf_answer=hf_answer,
        ground_truth=gt
    )

    print(f"   Faithfulness      : {eval_result.to_pandas()['faithfulness'].mean():.4f}")
    print(f"   Answer Relevancy  : {eval_result.to_pandas()['answer_relevancy'].mean():.4f}")
    print(f"   Context Precision : {eval_result.to_pandas()['context_precision'].mean():.4f}")
    if gt:
        print(f"   Context Recall    : {eval_result.to_pandas()['context_recall'].mean():.4f}")
