import streamlit as st

from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store
from src.retriever import get_retriever
from src.rag import run_rag
from src.memory import ConversationMemory


st.set_page_config(
    page_title="Tamil Heritage AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Tamil:wght@300;400;600;700;900&family=Playfair+Display:wght@600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Animated background */
    .stApp {
        background: linear-gradient(-45deg, #1a0000, #330000, #1a1a00, #663300);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main container */
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hero Header */
    .hero-section {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, rgba(139, 0, 0, 0.5) 0%, rgba(218, 165, 32, 0.5) 100%);
        border-radius: 25px;
        margin-bottom: 2rem;
        border: 2px solid rgba(255, 215, 0, 0.4);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 100px rgba(255, 215, 0, 0.2);
        backdrop-filter: blur(10px);
        position: relative;
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 50%, #FF6347 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(255, 215, 0, 0.6)); }
        to { filter: drop-shadow(0 0 40px rgba(255, 215, 0, 0.9)); }
    }
    
    .tamil-subtitle {
        font-family: 'Noto Sans Tamil', sans-serif;
        font-size: 1.8rem;
        color: #FFD700;
        margin-bottom: 1rem;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.6);
    }
    
    .english-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #FFA07A;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    .quote-text {
        font-family: 'Noto Sans Tamil', sans-serif;
        font-size: 0.95rem;
        color: #DAA520;
        font-style: italic;
        margin-top: 1rem;
    }
    
    /* Feature Grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .feature-box {
        background: linear-gradient(135deg, rgba(139, 0, 0, 0.3) 0%, rgba(184, 134, 11, 0.3) 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 215, 0, 0.3);
        backdrop-filter: blur(10px);
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .feature-box:hover {
        transform: translateY(-10px);
        border-color: #FFD700;
        box-shadow: 0 20px 40px rgba(255, 215, 0, 0.3);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFD700;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #FFA07A;
        line-height: 1.5;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #FFD700, transparent);
        margin: 2rem 0;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: linear-gradient(135deg, rgba(40, 20, 20, 0.6) 0%, rgba(60, 30, 10, 0.6) 100%) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        margin-bottom: 1rem !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(15px) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatMessage:hover {
        transform: translateX(8px);
        border-color: rgba(255, 215, 0, 0.5) !important;
        box-shadow: 0 10px 40px rgba(255, 215, 0, 0.2) !important;
    }
    
    /* User messages */
    [data-testid="stChatMessage-user"] {
        background: linear-gradient(135deg, rgba(255, 140, 0, 0.25) 0%, rgba(255, 215, 0, 0.2) 100%) !important;
        border-left: 4px solid #FFD700 !important;
    }
    
    /* Assistant messages */
    [data-testid="stChatMessage-assistant"] {
        background: linear-gradient(135deg, rgba(139, 0, 0, 0.25) 0%, rgba(178, 34, 34, 0.2) 100%) !important;
        border-left: 4px solid #DC143C !important;
    }
    
    /* Chat Input */
    .stChatInputContainer {
        background: linear-gradient(135deg, rgba(139, 0, 0, 0.4) 0%, rgba(184, 134, 11, 0.4) 100%);
        border-radius: 30px;
        border: 2px solid rgba(255, 215, 0, 0.4);
        padding: 0.8rem 1.5rem;
        backdrop-filter: blur(15px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .stChatInputContainer:focus-within {
        border-color: #FFD700;
        box-shadow: 0 10px 50px rgba(255, 215, 0, 0.4);
    }
    
    .stChatInputContainer textarea {
        background: transparent !important;
        color: #FFD700 !important;
        border: none !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
    }
    
    .stChatInputContainer textarea::placeholder {
        color: rgba(255, 215, 0, 0.5) !important;
    }
    
    /* Message text */
    .stChatMessage p {
        color: #FFF8DC !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.8 !important;
        font-size: 1rem !important;
    }
    
    /* Avatars */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #8B0000 0%, #FFD700 100%) !important;
        border: 2px solid #FFD700 !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.5) !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #8B0000 0%, #FFD700 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #DC143C 0%, #FFA500 100%);
    }
    
    /* Hide defaults */
    #MainMenu, footer, header {visibility: hidden;}
    
    .stMarkdown {
        color: #FFF8DC;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero-section">
    <div class="hero-title">Tamil Heritage AI</div>
    <div class="tamil-subtitle">தமிழ் பாரம்பரிய செயற்கை நுண்ணறிவு</div>
    <div class="english-subtitle">Discover 2000+ Years of Cultural Legacy</div>
    <div class="quote-text">"யாமறிந்த மொழிகளிலே தமிழ்மொழி போல் இனிதாவது எங்கும் காணோம்" - பாரதியார்</div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="feature-grid">
    <div class="feature-box">
        <div class="feature-icon">📜</div>
        <div class="feature-title">Sangam Literature</div>
        <div class="feature-desc">Ancient Tamil poetry spanning millennia</div>
    </div>
    <div class="feature-box">
        <div class="feature-icon">🛕</div>
        <div class="feature-title">Temple Architecture</div>
        <div class="feature-desc">Magnificent Dravidian structures</div>
    </div>
    <div class="feature-box">
        <div class="feature-icon">🎭</div>
        <div class="feature-title">Performing Arts</div>
        <div class="feature-desc">Bharatanatyam and Carnatic music</div>
    </div>
    <div class="feature-box">
        <div class="feature-icon">⚔️</div>
        <div class="feature-title">Ancient History</div>
        <div class="feature-desc">Tamil kingdoms and warriors</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=5)

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def load_rag():
    embedding_model = get_embedding_model()
    vectordb = load_vector_store(embedding_model)
    retriever = get_retriever(vectordb)
    return retriever

retriever = load_rag()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👁‍🗨" if msg["role"] == "user" else "🌾"):
        st.markdown(msg["content"])


user_input = st.chat_input("✨ Ask about Tamil culture, history, literature, or temples...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👁‍🗨"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🌾"):
        with st.spinner("🔮 Exploring Tamil heritage..."):
            answer = run_rag(user_input, retriever, st.session_state.memory)
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()