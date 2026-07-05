"""
Context-Aware RAG Chatbot — Streamlit App
Task 4: AI/ML Engineering Advanced Internship, DevelopersHub Corporation

Run with:
    streamlit run chatbot_app.py

To upgrade to a production LLM, replace `LocalExtractiveLLM()` below with, e.g.:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")
or:
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-sonnet-4-6")
Everything else in this file stays the same, since both expose the same .invoke(prompt) call.
"""

import re
from typing import Any, List, Optional

import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

RANDOM_SEED = 42

# ----------------------------------------------------------------
# 1. Knowledge base
# ----------------------------------------------------------------
RAW_DOCUMENTS = [
    ("The AI and Data Science program covers Python fundamentals, statistics, "
     "SQL, Excel, machine learning, deep learning, and deployment over its full duration.",
     {"topic": "curriculum"}),
    ("Core Python topics include variables, data structures, list comprehensions, "
     "functions, OOP, and file handling, taught before any machine learning content.",
     {"topic": "curriculum"}),
    ("The machine learning module covers supervised learning (regression, classification), "
     "unsupervised learning (clustering, dimensionality reduction), and model evaluation metrics.",
     {"topic": "curriculum"}),
    ("The deep learning module introduces neural networks, CNNs for computer vision, "
     "RNNs/Transformers for sequence data, and transfer learning with pretrained models.",
     {"topic": "curriculum"}),
    ("Tools used across the program include scikit-learn for classical ML, "
     "Hugging Face Transformers for NLP, and pandas/NumPy for data wrangling.",
     {"topic": "tools"}),
    ("For deployment, students use Streamlit and Gradio to turn trained models into "
     "interactive web demos that recruiters and clients can try without any setup.",
     {"topic": "tools"}),
    ("joblib is the standard tool for exporting trained scikit-learn pipelines so they "
     "can be reloaded later for inference without retraining from scratch.",
     {"topic": "tools"}),
    ("LangChain is the framework used for building conversational AI applications that "
     "combine large language models with external tools, memory, and document retrieval.",
     {"topic": "tools"}),
    ("The AI/ML Engineering Internship at DevelopersHub Corporation requires completing "
     "at least 3 out of 5 advanced tasks: BERT news classification, an end-to-end ML pipeline, "
     "multimodal housing price prediction, a context-aware RAG chatbot, and LLM-based ticket tagging.",
     {"topic": "internship"}),
    ("Each internship task must be submitted as a Jupyter notebook on GitHub with a README "
     "covering the objective, methodology, and key results, then linked in Google Classroom.",
     {"topic": "internship"}),
    ("Students are strongly encouraged to complete all five internship tasks, since a fuller "
     "portfolio demonstrates broader competency to future employers and clients.",
     {"topic": "internship"}),
    ("Freelancing platforms commonly used by graduates of the program include Fiverr and Upwork, "
     "where AI/ML skills like chatbot building, data analysis, and automation are in demand.",
     {"topic": "career"}),
    ("A strong GitHub profile with well-documented projects is one of the most persuasive assets "
     "a junior AI/ML freelancer can show a prospective client, more so than certificates alone.",
     {"topic": "career"}),
    ("Retrieval-Augmented Generation (RAG) improves chatbot answers by fetching relevant text "
     "chunks from a vector database before generating a response, reducing hallucination.",
     {"topic": "concepts"}),
    ("A vector store indexes documents as numerical embeddings so that semantically similar "
     "text can be retrieved via nearest-neighbor search, even without exact keyword matches.",
     {"topic": "concepts"}),
    ("Conversation memory lets a chatbot resolve references like 'it' or 'that one' by keeping "
     "track of recent turns in the dialogue rather than treating every message in isolation.",
     {"topic": "concepts"}),
    ("Prompt engineering is the practice of carefully wording instructions and examples given "
     "to a language model to steer its output toward the desired format or behavior.",
     {"topic": "concepts"}),
    ("Zero-shot learning means a model performs a task without having seen labeled examples of "
     "that exact task, relying only on general knowledge from pretraining.",
     {"topic": "concepts"}),
    ("Few-shot learning means the model is shown a handful of labeled examples inside the "
     "prompt itself to improve accuracy on a new task without any weight updates.",
     {"topic": "concepts"}),
    ("Evaluation metrics for regression models include MAE (mean absolute error), which is "
     "robust to outliers, and RMSE (root mean squared error), which penalizes large errors more.",
     {"topic": "concepts"}),
]


# ----------------------------------------------------------------
# 2. Embeddings (local, no internet required)
# ----------------------------------------------------------------
class LocalLSAEmbeddings(Embeddings):
    def __init__(self, texts: List[str], n_components: int = 64):
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        n_components = min(n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        self.svd.fit(tfidf_matrix)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        tfidf = self.vectorizer.transform(texts)
        vectors = self.svd.transform(tfidf)
        vectors = normalize(vectors)
        return vectors.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


# ----------------------------------------------------------------
# 3. Local extractive "LLM" (swap-in point documented at top of file)
# ----------------------------------------------------------------
class LocalExtractiveLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "local-extractive"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        context_match = re.search(r"CONTEXT:\n(.*?)\nQUESTION:", prompt, re.S)
        question_match = re.search(r"QUESTION:\n(.*)", prompt, re.S)
        context = context_match.group(1).strip() if context_match else ""
        question = question_match.group(1).strip() if question_match else prompt

        if not context.strip():
            return "I don't have enough information in the knowledge base to answer that."

        sentences = re.split(r"(?<=[.!?])\s+", context)
        q_tokens = set(re.findall(r"[a-zA-Z']+", question.lower()))

        scored = []
        for s in sentences:
            s_tokens = set(re.findall(r"[a-zA-Z']+", s.lower()))
            overlap = len(q_tokens & s_tokens)
            if overlap > 0:
                scored.append((overlap, s.strip()))

        if not scored:
            return ("Based on the retrieved material, I couldn't find a direct answer, "
                    "but here is the closest related information: " + sentences[0].strip())

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for _, s in scored[:2]]
        return " ".join(top_sentences)


# ----------------------------------------------------------------
# 4. Conversation memory
# ----------------------------------------------------------------
class ConversationMemory:
    def __init__(self, k: int = 4):
        self.k = k
        self.turns: List[dict] = []

    def add_turn(self, question: str, answer: str):
        self.turns.append({"question": question, "answer": answer})
        self.turns = self.turns[-self.k:]

    def as_history_text(self) -> str:
        if not self.turns:
            return "(no previous conversation)"
        return "\n".join(f"User: {t['question']}\nAssistant: {t['answer']}" for t in self.turns)

    def resolve_references(self, question: str) -> str:
        reference_words = {"it", "that", "those", "these", "one", "them", "which"}
        tokens = set(re.findall(r"[a-zA-Z']+", question.lower()))
        if tokens & reference_words and self.turns:
            last_q = self.turns[-1]["question"]
            return f"{last_q} -- follow-up: {question}"
        return question


# ----------------------------------------------------------------
# 5. Build the RAG components once, cache across reruns
# ----------------------------------------------------------------
@st.cache_resource
def build_rag_components():
    documents = [Document(page_content=text, metadata=meta) for text, meta in RAW_DOCUMENTS]
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    chunks = splitter.split_documents(documents)

    embeddings_model = LocalLSAEmbeddings([c.page_content for c in chunks], n_components=64)
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    llm = LocalExtractiveLLM()
    return vector_store, llm


def rag_answer(vector_store, llm, memory: ConversationMemory, question: str, k: int = 3):
    resolved_question = memory.resolve_references(question)
    retrieved = vector_store.similarity_search(resolved_question, k=k)
    context = "\n".join(d.page_content for d in retrieved)

    prompt = (
        f"You are a helpful course assistant. Use only the context to answer.\n\n"
        f"CONVERSATION HISTORY:\n{memory.as_history_text()}\n\n"
        f"CONTEXT:\n{context}\n"
        f"QUESTION:\n{question}"
    )
    answer = llm.invoke(prompt)
    memory.add_turn(question, answer)
    return answer, retrieved


# ----------------------------------------------------------------
# 6. Streamlit UI
# ----------------------------------------------------------------
st.set_page_config(page_title="Course Assistant — RAG Chatbot", page_icon="🎓")
st.title("🎓 AI & Data Science Course Assistant")
st.caption("Context-aware chatbot · LangChain + FAISS RAG pipeline · Task 4 internship submission")

vector_store, llm = build_rag_components()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(k=4)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []

with st.sidebar:
    st.header("🔍 Last Retrieved Chunks")
    if st.session_state.last_retrieved:
        for i, doc in enumerate(st.session_state.last_retrieved, 1):
            st.markdown(f"**{i}.** {doc.page_content}")
            st.caption(f"topic: {doc.metadata.get('topic', 'n/a')}")
    else:
        st.write("Ask a question to see which knowledge-base chunks were used.")
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.memory = ConversationMemory(k=4)
        st.session_state.messages = []
        st.session_state.last_retrieved = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask about the course, tools, or internship...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    answer, retrieved = rag_answer(vector_store, llm, st.session_state.memory, user_input)
    st.session_state.last_retrieved = retrieved
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
