# Task 4: Context-Aware Chatbot Using LangChain / RAG

**Internship:** AI/ML Engineering – Advanced Internship, DevelopersHub Corporation
**Deadline:** 21st July 2026

## Objective
Build a conversational chatbot that (1) remembers context across turns and (2) retrieves answers from a vectorized knowledge base (Retrieval-Augmented Generation), then deploy it with Streamlit.

## Methodology / Approach
1. **Knowledge base:** a custom "course assistant" corpus (curriculum, tools, internship rules, career tips, core RAG/LLM concepts), split into chunks with `RecursiveCharacterTextSplitter`.
2. **Embeddings:** a local, dependency-free `TfidfVectorizer + TruncatedSVD` (LSA) embedding model implementing LangChain's `Embeddings` interface — no external API or model download required.
3. **Vector store:** FAISS, built via `FAISS.from_documents(chunks, embeddings_model)` for nearest-neighbor retrieval.
4. **Conversation memory:** a rolling window of the last 4 turns, plus lightweight pronoun/reference resolution (e.g. "which one of those...") so follow-up questions retrieve the right context.
5. **Answer generation:** a local extractive `LLM` (subclasses LangChain's `LLM` base class) that selects and assembles the most relevant retrieved sentences. Documented as a one-line swap for `ChatOpenAI` / `ChatAnthropic` in production.
6. **Deployment:** `chatbot_app.py` — a Streamlit chat UI with a sidebar showing which knowledge-base chunks were retrieved for transparency.

## Key Results / Observations
- Multi-turn demo conversation in the notebook (Section 9) shows the chatbot correctly resolving **"Which one of those is used for deployment?"** back to the tools discussed two turns earlier — confirming working context memory.
- Retrieval quality was spot-checked (Section 10) by plotting FAISS distances for a sample query, confirming the most relevant chunks are retrieved first.
- The Streamlit app was smoke-tested and boots without errors (`streamlit run chatbot_app.py`).

## Files
- `Task4_Context_Aware_RAG_Chatbot.ipynb` — full notebook (architecture, corpus, embeddings, vector store, memory, generation, multi-turn demo, retrieval-quality visualization, summary)
- `chatbot_app.py` — standalone Streamlit deployment (run with `streamlit run chatbot_app.py`)

## Skills Demonstrated
Conversational AI development · Document embedding and vector search · Retrieval-Augmented Generation (RAG) · LLM integration and deployment
