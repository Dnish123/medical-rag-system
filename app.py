"""
Main Streamlit application for students to query medical books.
NO UPLOAD FUNCTIONALITY - Students can only ask questions.
"""

import streamlit as st
from rag.retriever import RAGRetriever
from rag.llm_chain import LLMChain
from utils.config import Config
import time

# Page configuration
st.set_page_config(
    page_title="Medical Q&A System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e40af;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .answer-section {
        background-color: #f0f9ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .teacher-section {
        background-color: #fef3c7;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .reference-section {
        background-color: #f0fdf4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize components
@st.cache_resource
def initialize_system():
    """Initialize RAG system components (cached)"""
    retriever = RAGRetriever()
    llm_chain = LLMChain()
    return retriever, llm_chain


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Q&A System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about AIIMS medical textbooks</p>', unsafe_allow_html=True)

    # Initialize system
    try:
        retriever, llm_chain = initialize_system()
    except Exception as e:
        st.error(f"⚠️ System initialization failed: {str(e)}")
        st.info("Please ensure Pinecone is configured and embeddings are uploaded.")
        return

    # Session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Question input
    question = st.text_input(
        "Enter your medical question:",
        placeholder="e.g., What are the symptoms of myocardial infarction?",
        key="question_input"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        ask_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)

    # Process question
    if ask_button and question.strip():
        with st.spinner("🔎 Searching medical literature..."):
            start_time = time.time()

            try:
                # Retrieve relevant chunks
                retrieved_docs = retriever.retrieve(question)

                if not retrieved_docs:
                    st.warning("No relevant information found. Please try rephrasing your question.")
                    return

                # Generate answer
                response = llm_chain.generate_answer(question, retrieved_docs)

                query_time = time.time() - start_time

                query_time = time.time() - start_time

                # Display answer
                st.markdown("---")

                # Answer section with better visibility
                st.markdown("### 📘 Answer")
                st.info(response["answer"])

                # Teacher explanation
                st.markdown("### 👨‍🏫 Teacher Explanation")
                st.warning(response["explanation"])

                # References
                st.markdown("### 📚 References")
                for i, ref in enumerate(response["references"], 1):
                    with st.expander(f"📖 {i}. {ref['book']} — Page {ref['page']}"):
                        st.markdown(f"**Paragraph:** {ref['paragraph']}")
                        st.markdown(f"**Summary:** {ref['summary']}")

                # Retrieved context (expandable)
                with st.expander("🔍 View Retrieved Context"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        st.markdown(f"**Chunk {i}:** (Score: {doc.get('score', 0):.3f})")
                        st.text_area(
                            f"Content {i}",
                            doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'],
                            height=150,
                            key=f"chunk_{i}"
                        )
                        st.markdown("---")

                # Performance metrics
                st.caption(f"⚡ Query processed in {query_time:.2f} seconds")
                # Retrieved context (expandable)
                with st.expander("🔍 View Retrieved Context"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'])
                        st.markdown("---")

                # Performance metrics
                st.caption(f"⚡ Query processed in {query_time:.2f} seconds")

                # Add to history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': response['answer'],
                    'timestamp': time.time()
                })

            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
                st.info("Please try again or rephrase your question.")

    # Show recent queries
    if st.session_state.chat_history:
        with st.sidebar:
            st.markdown("### 📝 Recent Queries")
            for i, item in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                with st.expander(f"{i}. {item['question'][:50]}..."):
                    st.markdown(item['answer'][:200] + "...")


if __name__ == "__main__":
    main()
