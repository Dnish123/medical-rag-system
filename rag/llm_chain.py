"""
LLM chain for generating answers using Groq.
"""

from groq import Groq
from typing import List, Dict
from utils.config import Config

class LLMChain:
    """
    Handles answer generation using Groq LLM.
    """

    def __init__(self):
        """Initialize Groq client"""
        # Fixed initialization - no extra parameters
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.GROQ_MODEL_PRIMARY
        self.fallback_model = Config.GROQ_MODEL_FALLBACK

    def generate_answer(self, question: str, retrieved_docs: List[Dict]) -> Dict:
        """
        Generate structured answer from retrieved documents.

        Args:
            question: User question
            retrieved_docs: List of retrieved document chunks

        Returns:
            Dictionary with answer, explanation, and references
        """
        # Build context from retrieved docs
        context = self._build_context(retrieved_docs)

        # Create prompt
        prompt = self._create_prompt(question, context)

        try:
            # Generate with primary model
            response = self._call_groq(prompt, self.model)
        except Exception as e:
            print(f"Primary model failed, using fallback: {e}")
            try:
                response = self._call_groq(prompt, self.fallback_model)
            except Exception as e2:
                print(f"Fallback model also failed: {e2}")
                # Return a basic error response
                return {
                    'answer': f"I found relevant information but couldn't generate a complete answer. Error: {str(e2)}",
                    'explanation': "Please try rephrasing your question or check your Groq API key.",
                    'references': self._create_references(retrieved_docs)
                }

        # Parse response
        parsed = self._parse_response(response, retrieved_docs)

        return parsed

    def _build_context(self, docs: List[Dict]) -> str:
        """Build context string from documents"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"[Source {i}] {doc['book']} (Page {doc['page']}):\n{doc['text']}\n"
            )
        return "\n".join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        """Create structured prompt for Groq"""
        return f"""You are a medical professor teaching AIIMS final year students. Answer the question using ONLY the provided context from medical textbooks.

CONTEXT:
{context}

QUESTION: {question}

Provide your response in the following format:

ANSWER:
[Give a direct, accurate answer grounded in the context. Be precise and medically accurate.]

TEACHER EXPLANATION:
[Explain this concept as if teaching a student. Use simple language, analogies, and break down complex terms. Help them understand the "why" and "how".]

SIMPLIFIED VERSION:
[Provide a very simple 2-3 sentence explanation that a layperson could understand.]

Remember: Base everything on the provided context. Cite specific sources when making claims."""

    def _call_groq(self, prompt: str, model: str) -> str:
        """Call Groq API with error handling"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical professor at AIIMS. Provide accurate, well-structured medical information."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=Config.GROQ_TEMPERATURE,
                max_tokens=Config.GROQ_MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {str(e)}")
            raise

    def _create_references(self, docs: List[Dict]) -> List[Dict]:
        """Create reference list from documents"""
        references = []
        for doc in docs[:3]:  # Top 3 references
            summary = doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text']
            references.append({
                'book': doc['book'],
                'page': doc['page'],
                'paragraph': doc['paragraph'],
                'summary': summary
            })
        return references

    def _parse_response(self, response: str, docs: List[Dict]) -> Dict:
        """Parse LLM response into structured format"""
        try:
            # Split response into sections
            sections = response.split("TEACHER EXPLANATION:")
            answer = sections[0].replace("ANSWER:", "").strip()

            rest = sections[1] if len(sections) > 1 else ""
            explanation_parts = rest.split("SIMPLIFIED VERSION:")
            explanation = explanation_parts[0].strip()
            simplified = explanation_parts[1].strip() if len(explanation_parts) > 1 else ""

            # Combine explanations
            full_explanation = explanation
            if simplified:
                full_explanation = f"{explanation}\n\n**In Simple Terms:**\n{simplified}"

            return {
                'answer': answer,
                'explanation': full_explanation,
                'references': self._create_references(docs)
            }
        except Exception as e:
            print(f"Error parsing response: {e}")
            # Return raw response if parsing fails
            return {
                'answer': response,
                'explanation': "Response generated successfully but formatting may vary.",
                'references': self._create_references(docs)
            }