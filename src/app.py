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
from download_arxiv import download_all_papers, clean_downloaded_papers

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

        # Download papers if needed
        print("\n[1/4] Checking for research papers...")
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        arxiv_json = os.path.join(os.path.dirname(os.path.dirname(__file__)), "arxiv_papers.json")

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        pdf_count = len([f for f in os.listdir(data_dir) if f.endswith('.pdf')]) if os.path.exists(data_dir) else 0

        if pdf_count == 0:
            print("No PDFs found. Downloading from arXiv...")
            downloaded_paths = download_all_papers(papers_file=arxiv_json, output_dir=data_dir)

            if not downloaded_paths:
                raise RuntimeError("Failed to download any papers from arXiv. Check network connection and arXiv availability.")

            print(f"Successfully downloaded {len(downloaded_paths)} papers")
        else:
            print(f"Found {pdf_count} PDFs in data directory")

        # Process documents
        print("\n[2/4] Processing documents...")
        self.chunks = process_documents(data_folder=data_dir)

        if not self.chunks:
            raise ValueError("No document chunks were created. Check if PDFs were downloaded and processed correctly.")

        print(f"Created {len(self.chunks)} chunks from documents")

        # Create retriever
        print("\n[3/4] Building retriever...")
        self.retriever = create_retriever(self.chunks)

        # Clean up PDFs after embeddings are created (copyright compliance)
        # ChromaDB has stored both embeddings AND text chunks, so PDFs are no longer needed
        print("\n[3.5/4] Removing PDFs (copyright compliance)...")
        try:
            clean_downloaded_papers(output_dir=data_dir)
            print("✓ PDFs deleted - embeddings and text chunks retained in ChromaDB")
        except Exception as e:
            print(f"Warning: Could not clean PDFs: {e}")

        # Initialize LLM
        print("\n[4/4] Connecting to Gemini...")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_output_tokens=2048
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
                "title": doc.metadata.get("paper_title", "Unknown Paper"),
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
            f"**{i+1}. {s['file']} - {s['title']}** (Page {s['page']})\n> {s['preview']}"
            for i, s in enumerate(result["sources"])
        ])
        
        # Add timing info
        timing = f"\n\n---\n*Retrieval: {result['retrieval_time']}s | Total: {result['total_time']}s*"
        
        return result["answer"], sources_text + timing
    
    # Build interface
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ).set(
        body_background_fill="*neutral_950",
        block_background_fill="*neutral_900",
    )
    with gr.Blocks(title="RAG Research Assistant", theme=custom_theme) as interface:
        gr.Markdown(
            """
            # 📚 RAG Research Assistant

            **Ask questions about research papers in AI/ML**. This system uses hybrid retrieval
            (semantic + keyword search) with fine-tuned embeddings to find relevant information
            and generate answers with source citations.

            ### 📖 Paper Collection (25 Papers)
            - **Foundation Models**: Transformers, BERT, Attention mechanisms
            - **3D Vision & Generation**: DreamFusion, TRELLIS, One-2-3-45, Omni3D, CO3D
            - **Generative AI**: DALL-E, 3D-Aware Diffusion
            - **RAG Systems**: VideoRAG, NodeRAG, UniversalRAG, Agentic RAG, ComoRAG

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
        gr.Markdown("### 💡 Try These Example Questions")
        gr.Examples(
            examples=[
                "What is the attention mechanism and how does it work?",
                "How does BERT handle bidirectional context?",
                "How does DreamFusion generate 3D objects from text?",
                "What are the components of the Transformer architecture?",
                "What is the difference between NodeRAG and traditional RAG?",
                "How does One-2-3-45 reconstruct 3D from a single image?",
            ],
            inputs=question_input
        )
        

        # Architecture details (collapsible)
        with gr.Accordion("🔧 How Does It Work?", open=False):
            gr.Markdown(
                """
                ## Three-Stage Retrieval Pipeline

                ### Stage 1: Hybrid Search (Casting a Wide Net)
                **Goal:** Find ~18 candidate passages from 5,000+ chunks

                - **Semantic Search (60%)**: Fine-tuned embeddings find passages by *meaning*
                  - Model: `all-MiniLM-L6-v2` fine-tuned on 275 domain-specific Q&A pairs
                  - Handles synonyms, paraphrases, conceptual similarity
                  - Example: "attention mechanism" matches "weighted representations"

                - **BM25 Keyword Search (40%)**: Statistical ranking finds *exact matches*
                  - Catches specific terms, acronyms, numbers
                  - Example: "BERT-base" matches exactly, not variations

                **Why Hybrid?** Combines semantic understanding + keyword precision

                ---

                ### Stage 2: CrossEncoder Reranking (Precision)
                **Goal:** Select top 6 passages from 18 candidates

                - **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
                - **How it works:** Sees query + passage together (not separately)
                - **Advantage:** More accurate than bi-encoder similarity alone
                - **Speed:** Only runs on 18 candidates (not all 5,000 chunks)

                **Why Rerank?** Fixes ranking errors from Stage 1

                ---

                ### Stage 3: LLM Generation (Synthesis)
                **Goal:** Generate natural language answer with citations

                - **Model:** Gemini 2.5 Flash
                - **Temperature:** 0.3 (low for factual accuracy)
                - **Context:** Top 6 passages + metadata
                - **Output:** Answer grounded in retrieved context + source citations

                **Why LLM?** Synthesizes information into coherent, readable answers

                ---

                ## Pipeline Flow
                ```
                User Query
                    ↓
                [Stage 1] Hybrid Search → 18 candidates (~50ms)
                    ↓
                [Stage 2] CrossEncoder Rerank → 6 best passages (~100ms)
                    ↓
                [Stage 3] Gemini generates answer (~1-2s)
                    ↓
                Answer + Citations
                ```

                **Total latency:** ~1.5-2 seconds
                """
            )

        # Performance metrics (collapsible)
        with gr.Accordion("📊 Performance Metrics", open=False):
            gr.Markdown(
                """
                ## Retrieval Quality (Evaluated on 275 Q&A Pairs)

                **Evaluation Setup:**
                - 5-fold cross-validation for stability
                - 100 negative samples per query (realistic retrieval)
                - Multiple metrics: MRR, Hits@k, NDCG@10

                ### Results: Base Model vs Fine-Tuned

                | Metric | Base Model | Fine-Tuned | Improvement |
                |--------|------------|------------|-------------|
                | **MRR** (Mean Reciprocal Rank) | 0.7493 | **0.9149** | **+22.1%** ✨ |
                | **Hits@1** (Top answer is correct) | 65.45% | **86.91%** | **+32.8%** ✨ |
                | **Hits@5** (Correct in top 5) | 86.18% | 97.09% | +12.7% |
                | **Hits@10** (Correct in top 10) | 90.18% | 98.55% | +9.3% |
                | **NDCG@10** (Ranking quality) | 0.7836 | **0.9317** | **+18.9%** ✨ |

                ### What This Means:
                - **87% of the time**, the best answer ranks #1
                - **97% of queries** have the correct passage in top 5
                - **MRR of 0.91** means average rank is ~1.1 (almost always first!)

                **Fine-tuning makes a significant difference!** The model is specifically adapted
                to academic paper retrieval, not generic web search.
                """
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
