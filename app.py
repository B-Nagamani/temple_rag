"""
Temple Guider AI - Streamlit Interface
Clean UI with proper bullet-point display (NO strats, NO summarize)
"""

import streamlit as st
from pathlib import Path
import os

# Import RAG system
from temple import TempleRAG, Config

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Temple Guider AI 🕉️",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ff9933 0%, #ffffff 50%, #138808 100%);
        padding: 20px 10px;
    }
    
    .temple-card {
        background: white;
        border-radius: 15px;
        padding: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .temple-card:hover {
        transform: translateY(-5px);
    }
    
    .temple-name {
        font-size: 18px;
        font-weight: bold;
        color: #ff6600;
        margin-top: 10px;
        text-align: center;
    }
    
    .temple-location {
        text-align: center;
        color: #666;
        font-size: 14px;
    }
    
    .user-message {
        background: #e3f2fd;
        padding: 15px 20px;
        border-radius: 15px 15px 5px 15px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-size: 16px;
    }
    
    .bot-message {
        background: #fff3e0;
        padding: 20px 25px;
        border-radius: 15px 15px 15px 5px;
        margin: 10px 0 20px 0;
        border-left: 4px solid #ff9800;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-size: 16px;
    }
    
    .bot-message .bullet-item {
        margin: 10px 0;
        line-height: 1.6;
        padding-left: 5px;
    }
    
    .welcome-message {
        background: linear-gradient(135deg, #ff9933 0%, #ff6600 100%);
        color: white;
        padding: 40px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(255,102,0,0.3);
        text-align: center;
    }
    
    .welcome-message h2 {
        font-size: 36px;
        margin-bottom: 15px;
    }
    
    .welcome-message p {
        font-size: 18px;
        margin: 12px 0;
    }
    
    .main-title {
        text-align: center;
        color: #ff6600;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #ff9933, #ff6600);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255,102,0,0.4);
    }
    
    .info-box {
        background: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 20px 0;
    }
    
    .warning-box {
        background: #fff3e0;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 20px 0;
    }
    
    .success-box {
        background: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 20px 0;
        color: #2e7d32;
    }
    
    .stats-box {
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stat-number {
        font-size: 32px;
        font-weight: bold;
        color: #ff6600;
    }
    
    .stat-label {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
    
    .history-container {
        background: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        max-height: 500px;
        overflow-y: auto;
    }
    
    .history-item {
        background: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    
    .feature-badge {
        display: inline-block;
        background: #4caf50;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# TEMPLE DATA
# =============================================================================

TEMPLES = [
    {
        "name": "Somnath Temple",
        "location": "Gujarat",
        "image": "somnath copy.jpg",
        "description": "First Jyotirlinga"
    },
    {
        "name": "Kamakhya Temple",
        "location": "Assam",
        "image": "kamakhya.jpg",
        "description": "Maha Shakti Peetha"
    },
    {
        "name": "Tirupati Balaji",
        "location": "Andhra Pradesh",
        "image": "Tirumala.jpg",
        "description": "Richest Temple"
    },
    {
        "name": "Kedarnath Temple",
        "location": "Uttarakhand",
        "image": "kedarnath.jpg",
        "description": "Himalayan Jyotirlinga"
    }
]

# =============================================================================
# SESSION STATE
# =============================================================================

def initialize_session_state():
    """Initialize session state"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'conversation_count' not in st.session_state:
        st.session_state.conversation_count = 0
    
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False

# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render sidebar"""
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: white;'>🕉️ Sacred Temples</h1>", 
                   unsafe_allow_html=True)
        st.markdown("---")
        
        # Temple cards
        for temple in TEMPLES:
            if os.path.exists(temple['image']):
                try:
                    st.image(temple['image'], use_container_width=True)
                except:
                    st.markdown(f"<div style='background:#ddd;height:200px;display:flex;align-items:center;justify-content:center;border-radius:10px;'>📷 {temple['name']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:#ddd;height:200px;display:flex;align-items:center;justify-content:center;border-radius:10px;'>📷 {temple['name']}</div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="temple-card">
                    <div class="temple-name">{temple['name']}</div>
                    <div class="temple-location">📍 {temple['location']}</div>
                    <p style="text-align: center; color: #888; font-size: 13px;">
                        {temple['description']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class='stats-box'>
                    <div class='stat-number'>{st.session_state.conversation_count}</div>
                    <div class='stat-label'>Questions</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.rag_system:
                chunks = st.session_state.rag_system.collection.count()
                st.markdown(f"""
                    <div class='stats-box'>
                        <div class='stat-number'>{chunks}</div>
                        <div class='stat-label'>Chunks</div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Features
        st.markdown("---")
        st.markdown("""
            <div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;'>
                <p style='color: white; text-align: center; font-size: 14px; font-weight: bold; margin-bottom: 10px;'>
                    ✨ FEATURES
                </p>
                <div style='text-align: center;'>
                    <span class='feature-badge'>5-7 Bullets</span>
                    <span class='feature-badge'>Last 5 History</span>
                    <span class='feature-badge'>Guard Rails</span>
                    <span class='feature-badge'>No Strats/Summary</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # History button
        st.markdown("---")
        if st.button("📜 Show Last 5 Messages", use_container_width=True, key="history_btn"):
            st.session_state.show_history = not st.session_state.show_history
        
        # Clear button
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_all_btn"):
            st.session_state.messages = []
            st.session_state.conversation_count = 0
            if st.session_state.rag_system:
                st.session_state.rag_system.conversation_history.clear()
            st.rerun()

# =============================================================================
# SETUP
# =============================================================================

def setup_rag_system():
    """Initialize RAG system"""
    config = Config()
    rag = TempleRAG(config)
    
    rag.initialize_database()
    
    if rag.is_database_empty():
        st.warning("📚 **First Time Setup Required**")
        
        pdf_count = len(list(Path(config.PDF_DIR).glob("*.pdf")))
        
        if pdf_count == 0:
            st.error(f"❌ No PDF files found in: {config.PDF_DIR}")
            st.info("Please check the PDF_DIR path in temple.py Config class")
            st.stop()
        
        st.markdown(f"""
            <div class='warning-box'>
                <strong>⏱️ Processing {pdf_count} PDFs</strong><br>
                Creating semantic chunks and embeddings...<br>
                This will take 20-40 minutes.<br>
                ☕ Please wait and do NOT close this window!
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Processing PDFs", type="primary"):
            from sentence_transformers import SentenceTransformer
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner(f"📝 Processing {pdf_count} PDFs..."):
                try:
                    status_text.text("Loading embedding model...")
                    rag.embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
                    progress_bar.progress(10)
                    
                    status_text.text("Processing PDFs and creating embeddings...")
                    rag.process_and_store_pdfs()
                    progress_bar.progress(100)
                    
                    st.success(f"✅ Successfully processed {pdf_count} PDFs!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.stop()
            
            st.rerun()
        else:
            st.stop()
    else:
        chunk_count = rag.collection.count()
        st.markdown(f"""
            <div class='success-box'>
                <strong>✅ Knowledge Base Ready!</strong><br>
                📊 {chunk_count:,} semantic chunks loaded<br>
                🔍 Answers in 5-7 bullet points<br>
                📜 Tracks last 5 conversations<br>
                🛡️ Guard rails active (no strats/summaries)
            </div>
        """, unsafe_allow_html=True)
    
    with st.spinner("🧠 Loading AI models..."):
        rag.initialize_models()
    
    return rag

# =============================================================================
# HISTORY DISPLAY
# =============================================================================

def render_history_panel():
    """Render last 5 conversation messages - NO TIMESTAMP"""
    if not st.session_state.rag_system:
        return
    
    history = st.session_state.rag_system.conversation_history.get_recent(5)
    
    if not history:
        st.info("📭 No conversation history yet. Start asking about temples!")
        return
    
    st.markdown("### 📜 Your Last 5 Questions")
    
    st.markdown("<div class='history-container'>", unsafe_allow_html=True)
    
    for i, entry in enumerate(history, 1):
        st.markdown(f"""
            <div class='history-item'>
                <strong>Q{i}:</strong> {entry['question']}
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# CHAT INTERFACE
# =============================================================================

def render_chat_interface():
    """Render chat interface"""
    
    # Title
    st.markdown("<div class='main-title'>🕉️ Temple Guider AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>📖 Clean Facts • No Strats • No Summaries</div>", 
               unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize
    if not st.session_state.initialized:
        try:
            st.session_state.rag_system = setup_rag_system()
            st.session_state.initialized = True
            
        except Exception as e:
            st.error(f"❌ Initialization Error: {str(e)}")
            st.info("💡 Check if the PDF_DIR path is correct in temple.py")
            return
    
    # Show history panel if requested
    if st.session_state.show_history:
        render_history_panel()
        st.markdown("---")
    
    # Welcome
    if not st.session_state.messages:
        st.markdown("""
            <div class='welcome-message'>
                <h2>🙏 Namaste & Welcome!</h2>
                <p>Your AI Guide to Hindu Temples</p>
                <p style='font-size: 16px; margin-top: 20px;'>
                    🕉️ 12 Sacred Jyotirlingas<br>
                    🌺 18 Maha Shakti Peethas<br>
                    🏛️ 175+ Famous Temples<br>
                    📖 Clean factual bullet points
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Display messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class='user-message'>
                        <strong>🙋 You:</strong> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Render bullet points properly
                bullets_html = "<div class='bot-message'>"
                
                # Check if it's a list or string
                if isinstance(message["content"], list):
                    for bullet in message["content"]:
                        bullets_html += f"<div class='bullet-item'>• {bullet}</div>"
                else:
                    # If string, split by newlines
                    lines = message["content"].split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            if not line.startswith('•'):
                                line = f"• {line}"
                            bullets_html += f"<div class='bullet-item'>{line}</div>"
                
                bullets_html += "</div>"
                st.markdown(bullets_html, unsafe_allow_html=True)
    
    # Input
    st.markdown("---")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask about temples:",
            placeholder="e.g., Tell me about Tirupati temple",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("🔍 Ask", use_container_width=True)
    
    # Process question
    if ask_button and user_question:
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })
        
        st.session_state.conversation_count += 1
        
        with st.spinner("🔮 Searching knowledge base..."):
            try:
                # Get answer as list of bullet points
                answer_list = st.session_state.rag_system.answer_question(user_question)
                
                # Store as list for proper rendering
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_list
                })
                
                st.rerun()
                
            except Exception as e:
                error_msg = [
                    "An error occurred while processing your question",
                    "Please try again"
                ]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                st.rerun()
    
    # Quick questions
    st.markdown("---")
    st.markdown("### 💡 Quick Questions:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    suggestions = [
        ("📍 12 Jyotirlingas", "List all 12 Jyotirlingas"),
        ("🌺 18 Shakti Peethas", "List all 18 Maha Shakti Peethas"),
        ("🏛️ Tirupati", "Tell me about Tirupati Balaji temple"),
        ("⛰️ Kedarnath", "Tell me about Kedarnath temple")
    ]
    
    cols = [col1, col2, col3, col4]
    for i, (label, question) in enumerate(suggestions):
        with cols[i]:
            if st.button(label, use_container_width=True, key=f"q{i}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.conversation_count += 1
                st.rerun()
    
    # More suggestions
    st.markdown("---")
    col5, col6, col7, col8 = st.columns(4)
    
    more_suggestions = [
        ("🕉️ Somnath", "Tell me about Somnath temple"),
        ("🌸 Kamakhya", "Tell me about Kamakhya temple"),
        ("📜 Last 5 Messages", "Show my last 5 messages"),
        ("🏛️ Varanasi", "Tell me about Kashi Vishwanath temple")
    ]
    
    cols2 = [col5, col6, col7, col8]
    for i, (label, question) in enumerate(more_suggestions):
        with cols2[i]:
            if st.button(label, use_container_width=True, key=f"q2{i}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.conversation_count += 1
                st.rerun()

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main application"""
    initialize_session_state()
    render_sidebar()
    render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 12px; padding: 20px;'>
            <p>🕉️ Temple Guider AI</p>
            <p>📖 Clean Facts Only • 🛡️ Guard Rails Active • 🚫 No Strats/Summaries</p>
            <p style='margin-top: 10px;'>Built with ❤️ for Hindu Temple Enthusiasts</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
