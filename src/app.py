# src/app.py
"""
Main RAG application with Gradio interface.
Connects ingestion, retrieval, and generation.
"""

import os
import sys
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import gradio as gr

from ingest import process_documents
from retriever import create_retriever

# Load environment variables
load_dotenv()


# === RAG PROMPT ===
RAG_PROMPT = """You are a helpful research assistant. Answer the question based ONLY on the following context from research papers.

CONTEXT:
{context}

RULES:
1. Only use information from the provided context
2. If the context doesn't contain the answer, say "I couldn't find this information in the provided papers."
3. Cite the source paper when possible
4. Be concise but thorough

QUESTION: {question}

ANSWER:"""


class RAGApplication:
    """Main RAG application class."""
    
    def __init__(self):
        """Initialize the RAG application."""
        print("=" * 50)
        print("Initializing RAG Research Assistant")
        print("=" * 50)
        
        # Process documents
        print("\n[1/3] Processing documents...")
        self.chunks = process_documents()
        
        # Create retriever
        print("\n[2/3] Building retriever...")
        self.retriever = create_retriever(self.chunks)
        
        # Initialize LLM
        print("\n[3/3] Connecting to Gemini...")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_tokens=1024
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        
        print("\n" + "=" * 50)
        print("RAG Application Ready!")
        print("=" * 50)
    
    def format_docs(self, docs) -> str:
        """Format retrieved documents into a context string."""
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "?")
            formatted.append(
                f"[Source {i+1}: {source}, Page {page}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)
    
    def query(self, question: str) -> dict:
        """
        Process a question through the RAG pipeline.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer, sources, and timing
        """
        start_time = time.time()
        
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(question, k=6)
        
        retrieval_time = time.time() - start_time
        
        # Format context
        context = self.format_docs(docs)
        
        # Generate answer
        chain = self.prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        total_time = time.time() - start_time
        
        # Format sources for display
        sources = []
        for doc in docs:
            sources.append({
                "file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "preview": doc.page_content[:150] + "..."
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieval_time": round(retrieval_time, 2),
            "total_time": round(total_time, 2)
        }


def create_gradio_interface(rag_app: RAGApplication) -> gr.Blocks:
    """Create the Gradio web interface."""
    
    def process_query(question):
        """Handle user query and format response."""
        if not question.strip():
            return "Please enter a question.", ""
        
        result = rag_app.query(question)
        
        # Format sources
        sources_text = "\n\n".join([
            f"**{i+1}. {s['file']}** (Page {s['page']})\n> {s['preview']}"
            for i, s in enumerate(result["sources"])
        ])
        
        # Add timing info
        timing = f"\n\n---\n*Retrieval: {result['retrieval_time']}s | Total: {result['total_time']}s*"
        
        return result["answer"], sources_text + timing
    
    # Build interface
    with gr.Blocks(title="RAG Research Assistant", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 📚 RAG Research Assistant
            
            Ask questions about your research papers. The system will find relevant 
            information and generate an answer with citations.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What is the attention mechanism and how does it work?",
                    lines=2
                )
                submit_btn = gr.Button("Ask", variant="primary")
            
        with gr.Row():
            with gr.Column():
                answer_output = gr.Markdown(label="Answer")
            
        with gr.Row():
            with gr.Column():
                sources_output = gr.Markdown(label="Sources")
        
        # Example questions
        gr.Markdown("### Example Questions")
        gr.Examples(
            examples=[
                "What is the attention mechanism?",
                "How does BERT handle bidirectional context?",
                "What are the main contributions of the transformer architecture?",
                "Explain the concept of self-attention.",
            ],
            inputs=question_input
        )
        
        # Connect components
        submit_btn.click(
            fn=process_query,
            inputs=[question_input],
            outputs=[answer_output, sources_output]
        )
        
        question_input.submit(
            fn=process_query,
            inputs=[question_input],
            outputs=[answer_output, sources_output]
        )
    
    return interface


def main():
    """Main entry point."""
    # Initialize RAG application
    rag_app = RAGApplication()
    
    # Create and launch interface
    interface = create_gradio_interface(rag_app)
    
    print("\nLaunching web interface...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True to get a public link
    )


if __name__ == "__main__":
    main()
