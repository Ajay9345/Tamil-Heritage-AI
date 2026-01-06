import streamlit as st

from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store
from src.retriever import get_retriever
from src.rag import run_rag
from src.memory import ConversationMemory

# Page config
st.set_page_config(page_title="RAG Assistant", layout="centered")

st.title("📚 RAG Assistant")

# Initialize session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=5)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load RAG components (only once)
@st.cache_resource
def load_rag():
    embedding_model = get_embedding_model()
    vectordb = load_vector_store(embedding_model)
    retriever = get_retriever(vectordb)
    return retriever

retriever = load_rag()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask a question...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer
    answer = run_rag(
        user_input,
        retriever,
        st.session_state.memory
    )

    # Show assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
