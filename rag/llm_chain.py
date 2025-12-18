"""
LLM chain for generating answers using Groq.
User-focused answers powered by book knowledge.
"""

from groq import Groq
from typing import List, Dict
from utils.config import Config

class LLMChain:
    """
    Handles answer generation using Groq LLM.
    Answers user questions directly using book data as knowledge base.
    """

    def __init__(self):
        """Initialize Groq client"""
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.GROQ_MODEL_PRIMARY
        self.fallback_model = Config.GROQ_MODEL_FALLBACK

    def generate_answer(self, question: str, retrieved_docs: List[Dict]) -> Dict:
        """
        Generate user-focused answer using book content.

        Args:
            question: User question
            retrieved_docs: List of retrieved document chunks

        Returns:
            Dictionary with conversational answer and references
        """
        # Remove duplicates
        unique_docs = self._deduplicate_chunks(retrieved_docs)

        # Build context
        context = self._build_context(unique_docs)

        # Create user-focused prompt
        prompt = self._create_user_focused_prompt(question, context)

        try:
            response = self._call_groq(prompt, self.model)
        except Exception as e:
            print(f"Primary model failed, using fallback: {e}")
            try:
                response = self._call_groq(prompt, self.fallback_model)
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return self._create_error_response(unique_docs)

        # Parse and format response
        parsed = self._parse_response(response, unique_docs)

        return parsed

    def _deduplicate_chunks(self, docs: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks"""
        if not docs:
            return []

        unique = []
        seen_texts = set()

        for doc in docs:
            text_normalized = ' '.join(doc['text'].lower().split()[:50])
            if text_normalized not in seen_texts:
                unique.append(doc)
                seen_texts.add(text_normalized)

        return unique[:10]

    def _build_context(self, docs: List[Dict]) -> str:
        """Build context from retrieved documents"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"[Source {i} - {doc['book']}, Page {doc['page']}]\n{doc['text']}\n"
            )
        return "\n".join(context_parts)

    def _create_user_focused_prompt(self, question: str, context: str) -> str:
        """Create prompt that focuses on answering user's specific question"""
        return f"""You are a helpful medical AI assistant. A student has asked you a question, and you have access to relevant medical textbook content to help answer it.

Your task: Answer the student's question directly and conversationally, using the textbook content as your knowledge base.

Guidelines:
- Focus on answering EXACTLY what the student asked
- Use a natural, conversational tone (like ChatGPT)
- Draw facts and information from the textbook content below
- If the student asks "what is X", explain what X is
- If they ask "how to treat Y", explain treatment
- If they ask "why does Z happen", explain the mechanism
- Be helpful and educational, not rigid
- If the textbook doesn't have the specific info they need, say so politely

AVAILABLE TEXTBOOK CONTENT:
{context}

STUDENT'S QUESTION: {question}

Now answer the student's question naturally and helpfully, using the textbook content as your knowledge source. Make your answer conversational and focused on what they actually asked."""

    def _call_groq(self, prompt: str, model: str) -> str:
        """Call Groq API"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a friendly medical AI assistant helping students learn. Answer their questions naturally and conversationally using textbook knowledge. Focus on what they actually asked."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,  # Balanced between creativity and accuracy
                max_tokens=Config.GROQ_MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {str(e)}")
            raise

    def _create_error_response(self, docs: List[Dict]) -> Dict:
        """Create fallback response on error"""
        return {
            'content': "I encountered an error generating a response. Please try rephrasing your question.",
            'references': self._create_references(docs)
        }

    def _create_references(self, docs: List[Dict]) -> List[Dict]:
        """Create reference list"""
        references = []
        for doc in docs[:5]:
            references.append({
                'book': doc['book'],
                'page': doc['page'],
                'excerpt': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
            })
        return references

    def _parse_response(self, response: str, docs: List[Dict]) -> Dict:
        """Parse response into structured format"""
        return {
            'content': response.strip(),
            'references': self._create_references(docs)
        }