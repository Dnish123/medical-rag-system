"""
Medical Q&A Chat System - ChatGPT Style Interface
Students can have multi-turn conversations with AI medical assistant
"""

import streamlit as st
from rag.retriever import RAGRetriever
from rag.llm_chain import LLMChain
from utils.config import Config
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat container */
    .main {
        background-color: #0e1117;
    }
    
    /* User message */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideInRight 0.3s ease-out;
    }
    
    /* Assistant message */
    .assistant-message {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: #e2e8f0;
        padding: 1rem 1.5rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideInLeft 0.3s ease-out;
    }
    
    /* Reference chips */
    .reference-chip {
        display: inline-block;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #60a5fa;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    
    /* Thinking animation */
    .thinking {
        color: #94a3b8;
        font-style: italic;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Input area */
    .stTextInput > div > div > input {
        background-color: #1e293b;
        color: white;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 0.75rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #0f172a;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize system
@st.cache_resource
def initialize_system():
    """Initialize RAG system components (cached)"""
    try:
        retriever = RAGRetriever()
        llm_chain = LLMChain()
        return retriever, llm_chain
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return None, None

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

def format_message(role, content, references=None, timestamp=None):
    """Format a chat message with styling"""
    if role == "user":
        return f'<div class="user-message">👤 <strong>You:</strong><br/>{content}</div>'
    else:
        ref_html = ""
        if references:
            ref_html = "<br/><br/><strong>📚 Sources:</strong><br/>"
            for ref in references[:3]:
                ref_html += f'<span class="reference-chip">📖 {ref["book"]} - Page {ref["page"]}</span>'

        return f'<div class="assistant-message">🤖 <strong>Medical AI:</strong><br/>{content}{ref_html}</div>'

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("# 🏥 Medical AI Assistant")
        st.markdown("---")
        st.markdown("### 💬 Chat Features")
        st.markdown("✅ Multi-turn conversations")
        st.markdown("✅ Context-aware responses")
        st.markdown("✅ Citations from textbooks")
        st.markdown("✅ Teacher-style explanations")
        st.markdown("---")

        # Chat stats
        if st.session_state.messages:
            st.markdown(f"**Messages:** {len(st.session_state.messages)}")
            st.markdown(f"**Session:** {st.session_state.session_id}")

        st.markdown("---")

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.rerun()

        # Export chat
        if st.session_state.messages and st.button("📥 Export Chat", use_container_width=True):
            chat_text = "\n\n".join([
                f"{'You' if msg['role'] == 'user' else 'Medical AI'}: {msg['content']}"
                for msg in st.session_state.messages
            ])
            st.download_button(
                label="Download",
                data=chat_text,
                file_name=f"medical_chat_{st.session_state.session_id}.txt",
                mime="text/plain"
            )

        st.markdown("---")
        st.markdown("### 📖 Sample Questions")
        st.markdown("• What are symptoms of MI?")
        st.markdown("• Explain diabetes management")
        st.markdown("• Differential diagnosis of chest pain")
        st.markdown("• Emergency treatment for anaphylaxis")

    # Main chat area
    st.markdown("# 🤖 Medical AI Assistant")
    st.markdown("*Ask me anything about AIIMS medical textbooks*")
    st.markdown("---")

    # Initialize system
    retriever, llm_chain = initialize_system()

    if retriever is None or llm_chain is None:
        st.error("⚠️ System not ready. Please check Pinecone configuration and ensure books are uploaded.")
        st.info("Admin: Run `python admin/ingest_books.py` to upload textbooks.")
        return

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            st.markdown(
                format_message(
                    message["role"],
                    message["content"],
                    message.get("references"),
                    message.get("timestamp")
                ),
                unsafe_allow_html=True
            )

    # Chat input area
    st.markdown("---")

    # Create input area at bottom
    col1, col2 = st.columns([6, 1])

    with col1:
        user_input = st.text_input(
            "Message",
            placeholder="Ask a medical question...",
            key="user_input",
            label_visibility="collapsed"
        )

    with col2:
        send_button = st.button("📤 Send", use_container_width=True)

    # Handle send button or enter key
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        })

        # Show thinking animation
        with st.spinner(""):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown(
                '<div class="thinking">🤔 Searching medical literature and formulating response...</div>',
                unsafe_allow_html=True
            )

            try:
                start_time = time.time()

                # Retrieve relevant documents
                retrieved_docs = retriever.retrieve(user_input)

                if not retrieved_docs:
                    response_text = "I couldn't find relevant information in the medical textbooks for this question. Could you rephrase or ask something else?"
                    references = []
                else:
                    # Generate response
                    response = llm_chain.generate_answer(user_input, retrieved_docs)

                    # Format response with sections
                    response_text = f"""**Answer:**
{response['answer']}

**📚 Detailed Explanation:**
{response['explanation']}"""

                    references = response.get('references', [])

                query_time = time.time() - start_time

                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "references": references,
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "query_time": query_time
                })

                thinking_placeholder.empty()

            except Exception as e:
                thinking_placeholder.empty()
                error_msg = f"❌ Error: {str(e)}\n\nPlease try rephrasing your question or check system logs."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().strftime("%H:%M")
                })

        # Rerun to show new messages
        st.rerun()

    # Welcome message for empty chat
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align: center; padding: 3rem; color: #64748b;'>
            <h2>👋 Welcome to Medical AI Assistant</h2>
            <p style='font-size: 1.1rem; margin-top: 1rem;'>
                I'm here to help you with medical questions from AIIMS textbooks.<br/>
                Ask me anything about diseases, treatments, procedures, or medical concepts.
            </p>
            <p style='margin-top: 2rem; font-size: 0.9rem;'>
                💡 Tip: Be specific in your questions for better answers
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Footer stats
    if st.session_state.messages:
        st.markdown("---")
        last_msg = st.session_state.messages[-1]
        if "query_time" in last_msg:
            st.caption(f"⚡ Last response: {last_msg['query_time']:.2f}s")

if __name__ == "__main__":
    main()