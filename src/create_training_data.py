# src/create_training_data.py
"""
Create training data for fine-tuning embeddings.
Generates question-passage pairs from your documents.
"""

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from ingest import process_documents

load_dotenv()


def generate_questions_for_chunk(llm, chunk_text: str, num_questions: int = 2) -> list:
    """
    Use LLM to generate questions that this chunk would answer.
    
    Args:
        llm: Language model
        chunk_text: The passage text
        num_questions: Number of questions to generate
        
    Returns:
        List of generated questions
    """
    prompt = f"""Based on the following passage from a research paper, generate {num_questions} questions that this passage would answer. 

The questions should:
- Be specific and answerable from the passage
- Vary in style (what, how, why, explain, etc.)
- Be questions a researcher might actually ask

PASSAGE:
{chunk_text}

Return ONLY the questions, one per line, no numbering or bullets."""

    try:
        response = llm.invoke(prompt)
        questions = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        return questions[:num_questions]
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []


def create_training_pairs(
    output_file: str = "evaluation/training_pairs.json",
    num_chunks: int = 50,
    questions_per_chunk: int = 2
):
    """
    Create training pairs for embedding fine-tuning.
    
    Args:
        output_file: Where to save the training data
        num_chunks: Number of chunks to process
        questions_per_chunk: Questions to generate per chunk
    """
    print("=" * 50)
    print("Creating Training Data for Fine-Tuning")
    print("=" * 50)
    
    # Load documents
    print("\n[1/3] Loading documents...")
    chunks = process_documents()
    
    # Limit chunks to process
    chunks_to_process = chunks[:num_chunks]
    print(f"Processing {len(chunks_to_process)} chunks...")
    
    # Initialize LLM
    print("\n[2/3] Initializing LLM for question generation...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7  # Slightly creative for diverse questions
    )
    
    # Generate training pairs
    print("\n[3/3] Generating question-passage pairs...")
    training_pairs = []
    
    for i, chunk in enumerate(chunks_to_process):
        print(f"  Processing chunk {i+1}/{len(chunks_to_process)}...", end=" ")
        
        questions = generate_questions_for_chunk(
            llm, 
            chunk.page_content, 
            questions_per_chunk
        )
        
        for question in questions:
            training_pairs.append({
                "question": question,
                "passage": chunk.page_content,
                "metadata": {
                    "source": chunk.metadata.get("source_file", "unknown"),
                    "page": chunk.metadata.get("page", "unknown")
                }
            })
        
        print(f"Generated {len(questions)} questions")
    
    # Save training data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(training_pairs, f, indent=2)
    
    print(f"\n{'=' * 50}")
    print(f"Created {len(training_pairs)} training pairs")
    print(f"Saved to: {output_file}")
    print(f"{'=' * 50}")
    
    return training_pairs


if __name__ == "__main__":
    # Generate training data
    # Adjust num_chunks based on how many papers you have
    pairs = create_training_pairs(
        num_chunks=25,  # Process 50 chunks
        questions_per_chunk=4  # 2 questions each = 100 pairs
    )
    
    # Show sample
    print("\n--- Sample Training Pairs ---")
    for pair in pairs[:3]:
        print(f"\nQ: {pair['question']}")
        print(f"A: {pair['passage'][:150]}...")