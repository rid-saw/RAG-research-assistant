#!/usr/bin/env python3
# src/generate_training_data.py
"""
Generate high-quality Q&A training pairs using Gemini API.

Key improvements over previous approach:
1. Uses LLM to generate SPECIFIC questions (not generic ones)
2. Each question must mention unique identifiers from the passage
3. Generates 8-12 pairs per PDF (not just 4)
4. Ensures each question maps to a DIFFERENT passage (no duplicates)
5. Includes validation to filter out generic questions
"""

import json
import os
import sys
import time
import random
from typing import List, Dict, Optional
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ingest import process_documents

load_dotenv()

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found in environment")
    print("Please set it in your .env file")
    sys.exit(1)

import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Words that indicate a question is too generic
GENERIC_INDICATORS = [
    "this paper",
    "this research",
    "this work",
    "this study",
    "the authors",
    "the paper",
    "the research",
    "main contribution",
    "key contribution",
    "what is proposed",
    "what methodology",
    "what approach",
    "what techniques",
]


def is_generic_question(question: str) -> bool:
    """Check if a question is too generic to be useful for training."""
    question_lower = question.lower()

    # Check for generic phrases
    for indicator in GENERIC_INDICATORS:
        if indicator in question_lower:
            # Exception: if the question also contains specific terms, it's okay
            # e.g., "What does this paper say about BERT?" is okay
            specific_terms = ["bert", "transformer", "attention", "gpt", "rag",
                           "diffusion", "3d", "neural", "embedding", "encoder",
                           "decoder", "convolution", "voxel", "mesh", "point cloud"]
            has_specific = any(term in question_lower for term in specific_terms)
            if not has_specific:
                return True

    return False


def generate_questions_for_passage(
    model,
    passage: str,
    source_file: str,
    num_questions: int = 3,
    max_retries: int = 3
) -> List[Dict]:
    """
    Generate specific questions for a passage using Gemini.

    The prompt is designed to produce questions that:
    1. Mention specific names, numbers, or technical terms from the passage
    2. Cannot be answered by other passages in the corpus
    3. Are diverse in type (what/how/why/when/who)
    """

    prompt = f"""You are generating training data for a retrieval system. Given a passage from an academic paper, generate {num_questions} specific questions that can ONLY be answered by this exact passage.

CRITICAL RULES:
1. Each question MUST include at least one specific identifier from the passage (name, number, acronym, method name, dataset name, metric value, etc.)
2. Questions must NOT be generic like "What is the main contribution?" or "What methodology is used?"
3. Questions should be diverse - mix of what/how/why/who/when types
4. Each question should focus on a DIFFERENT fact from the passage

PASSAGE:
{passage}

SOURCE: {source_file}

Generate exactly {num_questions} questions, one per line. Each question must be specific enough that it could ONLY match this passage, not any other academic paper.

Questions:"""

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            questions_text = response.text.strip()

            # Parse questions (one per line)
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                # Remove numbering if present
                if line and line[0].isdigit():
                    line = line.lstrip('0123456789.-) ').strip()
                if line and line.endswith('?'):
                    # Filter out generic questions
                    if not is_generic_question(line):
                        questions.append(line)

            return questions[:num_questions]

        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    return []


def select_diverse_chunks(chunks: List[Dict], num_chunks: int = 8) -> List[Dict]:
    """
    Select diverse chunks from a PDF, avoiding consecutive chunks
    which often contain overlapping content.
    """
    if len(chunks) <= num_chunks:
        return chunks

    # Strategy: select chunks spread across the document
    # Prioritize first chunk (usually abstract), then spread rest
    selected_indices = [0]  # Always include first chunk

    # Spread remaining selections across document
    remaining = num_chunks - 1
    step = max(1, (len(chunks) - 1) // remaining)

    for i in range(1, len(chunks), step):
        if len(selected_indices) >= num_chunks:
            break
        # Skip if too close to already selected
        if all(abs(i - j) >= 2 for j in selected_indices):
            selected_indices.append(i)

    # If we don't have enough, fill with remaining
    while len(selected_indices) < num_chunks and len(selected_indices) < len(chunks):
        for i in range(len(chunks)):
            if i not in selected_indices:
                selected_indices.append(i)
                break

    return [chunks[i] for i in sorted(selected_indices)]


def generate_training_data(
    output_file: str = "evaluation/training_pairs.json",
    questions_per_chunk: int = 2,
    chunks_per_pdf: int = 6,
    start_pdf_idx: int = 0,
    num_pdfs: Optional[int] = None,
    append: bool = False
):
    """
    Generate training data for all PDFs in the data folder.

    Args:
        output_file: Where to save the training pairs
        questions_per_chunk: Questions to generate per chunk (2-3 recommended)
        chunks_per_pdf: Number of chunks to sample per PDF (6-8 recommended)
        start_pdf_idx: Start from this PDF index (for resuming)
        num_pdfs: Number of PDFs to process (None = all)
        append: If True, append to existing file; if False, overwrite
    """
    print("=" * 70)
    print("TRAINING DATA GENERATION")
    print("=" * 70)

    # Initialize Gemini
    print("\n[1/4] Initializing Gemini model...")
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Load documents
    print("\n[2/4] Loading and chunking documents...")
    all_chunks = process_documents()
    print(f"  Total chunks: {len(all_chunks)}")

    # Group by PDF
    chunks_by_pdf = {}
    for chunk in all_chunks:
        pdf_name = chunk.metadata.get("source_file", "unknown")
        if pdf_name not in chunks_by_pdf:
            chunks_by_pdf[pdf_name] = []
        chunks_by_pdf[pdf_name].append({
            "passage": chunk.page_content,
            "source_file": pdf_name,
            "page": chunk.metadata.get("page", 0)
        })

    pdf_list = sorted(chunks_by_pdf.keys())
    print(f"  Total PDFs: {len(pdf_list)}")

    # Determine which PDFs to process
    end_idx = len(pdf_list) if num_pdfs is None else min(start_pdf_idx + num_pdfs, len(pdf_list))
    pdfs_to_process = pdf_list[start_pdf_idx:end_idx]

    print(f"\n[3/4] Processing PDFs {start_pdf_idx + 1} to {end_idx}...")
    print(f"  Questions per chunk: {questions_per_chunk}")
    print(f"  Chunks per PDF: {chunks_per_pdf}")
    print(f"  Expected pairs per PDF: ~{questions_per_chunk * chunks_per_pdf}")

    # Load existing pairs if appending
    existing_pairs = []
    if append and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_pairs = json.load(f)
        print(f"  Existing pairs: {len(existing_pairs)}")

    all_pairs = existing_pairs.copy()

    # Track which passages we've used to avoid duplicates
    used_passage_hashes = set()
    for pair in existing_pairs:
        used_passage_hashes.add(hash(pair["passage"][:200]))

    # Process each PDF
    for pdf_idx, pdf_name in enumerate(pdfs_to_process):
        print(f"\n  [{pdf_idx + 1}/{len(pdfs_to_process)}] {pdf_name}")

        chunks = chunks_by_pdf[pdf_name]
        selected_chunks = select_diverse_chunks(chunks, chunks_per_pdf)

        pdf_pairs = []

        for chunk_idx, chunk in enumerate(selected_chunks):
            passage = chunk["passage"]
            passage_hash = hash(passage[:200])

            # Skip if we've already used this passage
            if passage_hash in used_passage_hashes:
                print(f"    Chunk {chunk_idx + 1}: skipped (duplicate)")
                continue

            # Generate questions
            questions = generate_questions_for_passage(
                model=model,
                passage=passage,
                source_file=pdf_name,
                num_questions=questions_per_chunk
            )

            if questions:
                used_passage_hashes.add(passage_hash)
                for q in questions:
                    pdf_pairs.append({
                        "question": q,
                        "passage": passage,
                        "metadata": {
                            "source": pdf_name,
                            "page": chunk["page"]
                        }
                    })
                print(f"    Chunk {chunk_idx + 1}: {len(questions)} questions")
            else:
                print(f"    Chunk {chunk_idx + 1}: failed to generate")

            # Rate limiting
            time.sleep(2.0)

        all_pairs.extend(pdf_pairs)
        print(f"    Total for {pdf_name}: {len(pdf_pairs)} pairs")

    # Save results
    print(f"\n[4/4] Saving results...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_pairs, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Total pairs generated: {len(all_pairs) - len(existing_pairs)}")
    print(f"  Total pairs in file: {len(all_pairs)}")
    print(f"  Unique passages: {len(used_passage_hashes)}")
    print(f"  Output file: {output_file}")

    # Check for any remaining generic questions
    generic_count = sum(1 for p in all_pairs if is_generic_question(p["question"]))
    if generic_count > 0:
        print(f"\n  WARNING: {generic_count} potentially generic questions detected")
        print("  Consider reviewing and filtering these manually")

    return all_pairs


def validate_training_data(filepath: str = "evaluation/training_pairs.json"):
    """
    Validate and report on training data quality.
    """
    with open(filepath, 'r') as f:
        pairs = json.load(f)

    print("\n" + "=" * 70)
    print("TRAINING DATA VALIDATION")
    print("=" * 70)

    # Basic stats
    print(f"\nTotal pairs: {len(pairs)}")

    # Unique passages
    unique_passages = set(p["passage"][:200] for p in pairs)
    print(f"Unique passages: {len(unique_passages)}")
    print(f"Duplication ratio: {len(pairs) / len(unique_passages):.2f}x")

    # Generic questions
    generic = [p for p in pairs if is_generic_question(p["question"])]
    print(f"\nGeneric questions: {len(generic)} ({100*len(generic)/len(pairs):.1f}%)")

    if generic:
        print("\nSample generic questions (should be replaced):")
        for p in generic[:5]:
            print(f"  - {p['question'][:70]}...")

    # Questions by source
    sources = {}
    for p in pairs:
        src = p["metadata"]["source"]
        sources[src] = sources.get(src, 0) + 1

    print(f"\nQuestions per PDF:")
    for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {src}: {count}")

    # Question length distribution
    lengths = [len(p["question"]) for p in pairs]
    print(f"\nQuestion length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")

    return pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training data for RAG fine-tuning")
    parser.add_argument("--output", default="evaluation/training_pairs_new.json",
                       help="Output file path")
    parser.add_argument("--questions-per-chunk", type=int, default=2,
                       help="Questions to generate per chunk")
    parser.add_argument("--chunks-per-pdf", type=int, default=6,
                       help="Chunks to sample per PDF")
    parser.add_argument("--start-pdf", type=int, default=0,
                       help="Start from this PDF index")
    parser.add_argument("--num-pdfs", type=int, default=None,
                       help="Number of PDFs to process")
    parser.add_argument("--append", action="store_true",
                       help="Append to existing file")
    parser.add_argument("--validate", action="store_true",
                       help="Validate existing training data")

    args = parser.parse_args()

    if args.validate:
        validate_training_data(args.output)
    else:
        generate_training_data(
            output_file=args.output,
            questions_per_chunk=args.questions_per_chunk,
            chunks_per_pdf=args.chunks_per_pdf,
            start_pdf_idx=args.start_pdf,
            num_pdfs=args.num_pdfs,
            append=args.append
        )
