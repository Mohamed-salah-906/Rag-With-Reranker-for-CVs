# filename: cv_rag_rerank_app.py
import os
from collections import defaultdict
import streamlit as st

# ---- LangChain & friends ----
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers.multi_query import MultiQueryRetriever

# ---- Reranker (FlashRank) ----
from flashrank import Ranker, RerankRequest

# -------------------------
# Config
# -------------------------
DATA_PATH = "D:\\Downloads\\CVs"                       # put PDFs here
PERSIST_DIR = "chroma_db"               # vector DB on disk
EMBED_MODEL = "nomic-embed-text"        # local embedder via Ollama
LLM_MODEL = "alibayram/smollm3"                   # single required LLM (no fallback)
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"  # multilingual cross-encoder
TOPK_INITIAL = 40                       # retrieve this many before rerank
TOPK_RERANK = 20                        # keep this many chunks after rerank
TOPK_CVS = 10                           # how many CV files to show

# -------------------------
# Helpers
# -------------------------
def load_pdfs(path: str):
    loader = PyPDFDirectoryLoader(path)
    return loader.load()

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.split_documents(docs)

def get_llm():
    # Only SmolLM3. If missing, tell the user and stop.
    try:
        return ChatOllama(model=LLM_MODEL, temperature=0.1)
    except Exception as e:
        st.error(
            "SmolLM3 is not available in your Ollama. "
            "Please pull it first:\n\n```bash\nollama pull smollm3\n```"
        )
        st.stop()

def get_embeddings():
    # Requires: ollama pull nomic-embed-text
    return OllamaEmbeddings(model=EMBED_MODEL, show_progress=True)

def build_vector_db(chunks):
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="cv-rag",
        persist_directory=PERSIST_DIR,
    )

def format_docs(docs):
    return "\n\n---\n".join(
        f"Source:{os.path.basename(d.metadata.get('source',''))}\n{d.page_content}"
        for d in docs
    )

def build_multiquery_retriever(vdb, llm):
    # Expand the query into variants, then search via the *retriever* interface
    base = vdb.as_retriever(search_kwargs={"k": TOPK_INITIAL})

    query_prompt = ChatPromptTemplate.from_template(
        """Rewrite the job/skills query into 3-5 diverse alternatives that capture synonyms,
        abbreviations, and equivalent role names (e.g., 'data scientist' ~ 'ML engineer').
        Return each rewritten query on a NEW line.
        Original: {question}"""
    )

    try:
        # Correct API: pass a retriever, not the vectorstore
        return MultiQueryRetriever.from_llm(
            retriever=base,
            llm=llm,
            prompt=query_prompt,
            include_original=True,
        )
    except TypeError:
        # Some older/newer LC builds shuffle args; keep you unblocked
        try:
            return MultiQueryRetriever.from_llm(llm=llm, retriever=base)
        except Exception:
            st.warning("MultiQueryRetriever not available; using base retriever.")
            return base
# -------------------------
# Reranking & CV-level scoring
# -------------------------
def rerank_chunks(query, docs):
    ranker = Ranker(model_name=RERANK_MODEL)
    passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
    req = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(req)

    id_key = "id" if len(results) and "id" in results[0] else "index"
    score_key = "relevance_score" if len(results) and "relevance_score" in results[0] else "score"
    ranked = []
    for r in results[:TOPK_RERANK]:
        idx = r.get(id_key, 0)
        score = float(r.get(score_key, 0.0))
        d = docs[idx]
        d.metadata["rerank_score"] = score
        ranked.append(d)
    return ranked

def rank_cvs_by_query(vdb, retriever, query):
    candidate_chunks = retriever.invoke(query)
    ranked_chunks = rerank_chunks(query, candidate_chunks)

    per_file = defaultdict(lambda: {"best": 0.0, "sum": 0.0, "count": 0, "snippets": []})
    for d in ranked_chunks:
        src = d.metadata.get("source", "unknown")
        score = float(d.metadata.get("rerank_score", 0.0))
        per_file[src]["best"] = max(per_file[src]["best"], score)
        per_file[src]["sum"] += score
        per_file[src]["count"] += 1
        if len(per_file[src]["snippets"]) < 3:
            per_file[src]["snippets"].append(d.page_content[:500])

    rows = []
    for src, info in per_file.items():
        avg = info["sum"] / max(1, info["count"])
        composite = 0.8 * info["best"] + 0.2 * avg
        rows.append({
            "cv_file": os.path.basename(src),
            "doc_score": round(composite, 4),
            "hits": info["count"],
            "top_snippet": info["snippets"][0] if info["snippets"] else "",
            "all_snippets": info["snippets"],
            "source": src
        })
    rows.sort(key=lambda x: x["doc_score"], reverse=True)
    return rows[:TOPK_CVS]

# -------------------------
# RAG answer chain (uses reranked context)
# -------------------------
def build_answer_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template(
        """You are a CV screener. Using ONLY the context from CVs, answer the user's question
        precisely. Prefer specific skills, years, tools, certifications, and role titles.
        If unsure, say you don't know.

        Context:
        {context}

        Question: {question}
        Helpful, concise answer:"""
    )

    chain = (
        {
            "context": RunnableLambda(lambda q: retriever.invoke(q)) 
                       | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# -------------------------
# App init (cache for speed)
# -------------------------
@st.cache_resource(show_spinner=False)
def init_pipeline():
    raw_docs = load_pdfs(DATA_PATH)
    chunks = split_docs(raw_docs)
    vdb = build_vector_db(chunks)

    llm = get_llm()
    retriever = build_multiquery_retriever(vdb, llm)
    answer_chain = build_answer_chain(retriever, llm)
    return vdb, retriever, answer_chain, llm

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(page_title="CV RAG + Reranker", layout="wide")
    st.title("📄 CV RAG + Re-ranking (Ollama)")

    with st.sidebar:
        st.subheader("Settings")
        st.caption("Models")
        st.text(f"LLM: {LLM_MODEL}")
        st.text(f"Embeddings: {EMBED_MODEL}")
        st.text(f"Reranker: {RERANK_MODEL}")
        if st.button("Rebuild index"):
            st.cache_resource.clear()
            st.experimental_rerun()
        st.caption(f"Data folder: {os.path.abspath(DATA_PATH)}")

    vdb, retriever, answer_chain, llm = init_pipeline()

    mode = st.radio("Mode", ["Rank CVs", "Ask (RAG)"], horizontal=True)
    query = st.text_input(
        "Job title and/or required skills",
        placeholder="e.g., Senior Data Scientist, NLP, LangChain, MLOps, Kubernetes, German",
    )

    if not query:
        st.stop()

    if mode == "Rank CVs":
        with st.spinner("Scoring CVs..."):
            rows = rank_cvs_by_query(vdb, retriever, query)
        st.subheader("Top matching CVs")
        for i, r in enumerate(rows, start=1):
            with st.expander(f"{i}. {r['cv_file']} — score {r['doc_score']} (hits: {r['hits']})", expanded=i==1):
                st.write("Top snippet:")
                st.write(r["top_snippet"])
                if len(r["all_snippets"]) > 1:
                    st.write("More snippets:")
                    for s in r["all_snippets"][1:]:
                        st.write("— " + s)

    else:
        with st.spinner("Answering..."):
            answer = answer_chain.invoke(query)
        st.subheader("Answer")
        st.write(answer)

if __name__ == "__main__":
    main()
