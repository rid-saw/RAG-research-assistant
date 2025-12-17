#!/usr/bin/env python3
# src/generate_qa_with_claude.py
"""
Simple script to generate Q&A pairs with Claude's help.
Claude will generate 4 questions per PDF file and save them in the required format.
"""

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest import process_documents


def generate_and_save_qa_pairs(
    new_pairs: list,
    output_file: str = "evaluation/training_pairs.json"
):
    """Add new Q&A pairs to the training data file."""
    # Load existing pairs
    existing_pairs = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_pairs = json.load(f)

    # Add new pairs
    existing_pairs.extend(new_pairs)

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(existing_pairs, f, indent=2)

    print(f"\n✓ Added {len(new_pairs)} new Q&A pairs")
    print(f"✓ Total pairs: {len(existing_pairs)}")
    print(f"✓ Saved to: {output_file}")

    return existing_pairs


def load_chunks_by_pdf(start_pdf_idx: int = 0, num_pdfs: int = 1):
    """Load document chunks organized by PDF file."""
    print("Loading documents...")
    all_chunks = process_documents()
    print(f"Total chunks available: {len(all_chunks)}")

    # Load existing pairs to show progress
    existing_file = "evaluation/training_pairs.json"
    if os.path.exists(existing_file):
        with open(existing_file, 'r') as f:
            existing_pairs = json.load(f)
        print(f"Existing Q&A pairs: {len(existing_pairs)}")

    # Group chunks by PDF
    chunks_by_pdf = {}
    for chunk in all_chunks:
        pdf_name = chunk.metadata.get("source_file", "unknown")
        if pdf_name not in chunks_by_pdf:
            chunks_by_pdf[pdf_name] = []
        chunks_by_pdf[pdf_name].append({
            "passage": chunk.page_content,
            "source_file": pdf_name,
            "page": chunk.metadata.get("page", "unknown")
        })

    # Get list of PDFs
    pdf_list = sorted(chunks_by_pdf.keys())
    print(f"\nTotal PDFs: {len(pdf_list)}")

    # Select PDFs to process
    end_pdf_idx = min(start_pdf_idx + num_pdfs, len(pdf_list))
    selected_pdfs = pdf_list[start_pdf_idx:end_pdf_idx]

    print(f"Processing PDFs {start_pdf_idx + 1} to {end_pdf_idx}:")
    for pdf_name in selected_pdfs:
        print(f"  - {pdf_name} ({len(chunks_by_pdf[pdf_name])} chunks)")

    # Return selected PDFs and their chunks
    result = {pdf: chunks_by_pdf[pdf] for pdf in selected_pdfs}
    return result, pdf_list


if __name__ == "__main__":
    print("=" * 80)
    print("Q&A Pair Generation with Claude - 4 pairs per PDF")
    print("=" * 80)
    print("\nThis script generates 4 Q&A pairs per PDF file.\n")

    # Configuration: which PDFs to process (0-based index)
    PDF_INDEX = 0
    NUM_PDFS = 26  # Process ALL PDFs

    # Load chunks for the selected PDFs
    pdf_chunks, all_pdfs = load_chunks_by_pdf(start_pdf_idx=PDF_INDEX, num_pdfs=NUM_PDFS)

    if not pdf_chunks:
        print("No PDFs to process!")
        exit(1)

    print(f"\n{'=' * 80}")
    print(f"Generating Q&A pairs for {len(pdf_chunks)} PDFs")
    print(f"{'=' * 80}\n")

    # Claude-generated Q&A pairs for ALL PDFs (4 questions per PDF)
    all_generated_pairs = []

    # PDF 0: 1706.03762v7.pdf - Attention Is All You Need (Transformer)
    chunks = pdf_chunks["1706.03762v7.pdf"]
    all_generated_pairs.extend([
        {"question": "What is the main innovation proposed in the Transformer architecture?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What are the key advantages of the Transformer model compared to recurrent architectures?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What BLEU score did the Transformer achieve on the WMT 2014 English-to-German translation task?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "Besides machine translation, what other task did the Transformer model successfully generalize to?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
    ])

    # PDF 1: 1810.04805v2.pdf - BERT
    chunks = pdf_chunks["1810.04805v2.pdf"]
    all_generated_pairs.extend([
        {"question": "What does BERT stand for and what is its main characteristic?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "How does BERT differ from previous language representation models like ELMo and GPT?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What GLUE score improvement did BERT achieve?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What are the two strategies for applying pre-trained language representations to downstream tasks?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
    ])

    # PDF 2: 2102.12092v2.pdf - DALL-E
    chunks = pdf_chunks["2102.12092v2.pdf"]
    all_generated_pairs.extend([
        {"question": "What is the main approach used in zero-shot text-to-image generation described in this paper?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "How does this transformer-based approach model text and image data?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What improvement did Reed et al. achieve by using GANs instead of recurrent variational auto-encoders?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What capability did Reed et al. demonstrate regarding object category generalization?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
    ])

    # PDF 3: 2109.00512v1.pdf - CO3D
    chunks = pdf_chunks["2109.00512v1.pdf"]
    all_generated_pairs.extend([
        {"question": "What is the Common Objects in 3D (CO3D) dataset and how many objects does it contain?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What is the main motivation for creating the CO3D dataset?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "How many frames and videos does the CO3D dataset contain?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What is NerFormer and what is its purpose?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
    ])

    # PDF 4: 2204.11918v1.pdf - Google Scanned Objects
    chunks = pdf_chunks["2204.11918v1.pdf"]
    all_generated_pairs.extend([
        {"question": "What is the Google Scanned Objects dataset and how many 3D models does it contain?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "Which simulation platforms are the Google Scanned Objects preprocessed for?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "Why doesn't the web scraping strategy for 2D datasets readily generalize to 3D data?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
        {"question": "What are the advantages of using simulation for robotic learning?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
    ])

    # PDF 5: 2207.10660v2.pdf - OMNI3D
    chunks = pdf_chunks["2207.10660v2.pdf"]
    all_generated_pairs.extend([
        {"question": "What is OMNI3D and how does it compare to existing 3D benchmarks?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What model is proposed for 3D object detection and what is its key capability?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "How many images and instances are in the OMNI3D dataset?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What applications benefit from understanding objects in 3D from single images?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
    ])

    # PDF 6: 2209.14988v1.pdf - DreamFusion
    chunks = pdf_chunks["2209.14988v1.pdf"]
    all_generated_pairs.extend([
        {"question": "How does DreamFusion circumvent the need for large-scale 3D training data?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What type of loss function does DreamFusion introduce for optimization?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What properties does the resulting 3D model from DreamFusion possess?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What practical applications could benefit from 3D synthesis techniques like DreamFusion?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
    ])

    # PDF 7: 2304.02602v1.pdf - 3D-Aware Diffusion
    chunks = pdf_chunks["2304.02602v1.pdf"]
    all_generated_pairs.extend([
        {"question": "What is the main contribution of generative novel view synthesis with 3D-aware diffusion models?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "Who are the authors of the 3D-aware diffusion models paper?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What capability does this approach enable for novel view synthesis?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "How does this work relate to existing diffusion model research?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
    ])

    # PDF 8: 2306.16928v1.pdf - One-2-3-45
    chunks = pdf_chunks["2306.16928v1.pdf"]
    all_generated_pairs.extend([
        {"question": "What is the main promise of One-2-3-45?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "How long does One-2-3-45 take to convert a single image to a 3D mesh?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "Does One-2-3-45 require per-shape optimization?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What is the key advantage of One-2-3-45's approach?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
    ])

    # PDF 9: 2311.07885v1.pdf - One-2-3-45++
    chunks = pdf_chunks["2311.07885v1.pdf"]
    all_generated_pairs.extend([
        {"question": "What is One-2-3-45++ and how does it improve upon the original?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What are the two key components mentioned in One-2-3-45++?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What capability does consistent multi-view generation provide?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "How does 3D diffusion enhance the reconstruction process?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
    ])

    # PDF 10: 2312.08963v2.pdf - LEMON
    chunks = pdf_chunks["2312.08963v2.pdf"]
    all_generated_pairs.extend([
        {"question": "What does LEMON stand for and what is its purpose?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What is the input and output of the LEMON method?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "Who are the authors of the LEMON paper?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What type of interaction does LEMON focus on learning?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
    ])

    # PDF 11: 2408.08234v1.pdf - 3D Reconstruction for Pose Estimation
    chunks = pdf_chunks["2408.08234v1.pdf"]
    all_generated_pairs.extend([
        {"question": "What is the main research question addressed regarding 3D models for object pose estimation?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What benchmark format does this study use for pose evaluation?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What surprising finding about classical non-learning-based reconstruction methods was discovered?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "How do CAD-based pose estimation methods differ from CAD-free methods?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
    ])

    # PDF 2: 2412.01506v3.pdf - Structured 3D Latents (TRELLIS)
    chunks = pdf_chunks["2412.01506v3.pdf"]
    all_generated_pairs.extend([
        {"question": "What is the Structured LATent (SLAT) representation and what formats can it decode to?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "How large is the model trained and on how many 3D objects?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
        {"question": "What generation time does this method achieve?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What editing capabilities does the method support?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
    ])

    # PDF 3: 2501.01880v1.pdf - Long Context vs RAG
    chunks = pdf_chunks["2501.01880v1.pdf"]
    all_generated_pairs.extend([
        {"question": "What are the two main strategies for enabling LLMs to incorporate long external contexts?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "Which approach generally outperforms in question-answering benchmarks according to this study?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "In what scenarios does RAG have advantages over Long Context?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What common challenges do LLMs face that motivate the need for external memory?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
    ])

    # PDF 4: 2501.05874v3.pdf - VideoRAG
    chunks = pdf_chunks["2501.05874v3.pdf"]
    all_generated_pairs.extend([
        {"question": "What is VideoRAG and how does it differ from existing RAG approaches?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What enables VideoRAG to process video content directly?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "Why does VideoRAG include a video frame selection mechanism?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "How does VideoRAG handle videos without available subtitles?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
    ])

    # PDF 5: 2501.07391v1.pdf - Enhancing RAG Best Practices
    chunks = pdf_chunks["2501.07391v1.pdf"]
    all_generated_pairs.extend([
        {"question": "What components does this RAG study investigate?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What novel RAG approach is introduced in this paper?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What key factors influence RAG response quality according to the study?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "Why is continuously updating language models costly?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
    ])

    # PDF 6: 2501.09751v5.pdf - OmniThink
    chunks = pdf_chunks["2501.09751v5.pdf"]
    all_generated_pairs.extend([
        {"question": "What problem does OmniThink address in machine writing?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "How does OmniThink emulate human-like writing behavior?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What limitations does vanilla RAG have according to this paper?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
        {"question": "What does OmniThink's information tree and conceptual pool enable?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
    ])

    # PDF 7: 2504.11544v1.pdf - NodeRAG
    chunks = pdf_chunks["2504.11544v1.pdf"]
    all_generated_pairs.extend([
        {"question": "What is the main limitation of current graph-based RAG approaches?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What graph structure does NodeRAG introduce?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "How does NodeRAG compare to GraphRAG and LightRAG in performance?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What challenges does RAG address in knowledge domains?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
    ])

    # PDF 8: 2504.20734v2.pdf - UniversalRAG
    chunks = pdf_chunks["2504.20734v2.pdf"]
    all_generated_pairs.extend([
        {"question": "What limitation do most existing RAG approaches have regarding modalities?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What mechanism does UniversalRAG propose to handle multiple modalities?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "Why does forcing all modalities into a unified representation cause problems?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What issue do LLMs face that RAG helps address?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
    ])

    # PDF 9: 2506.16035v2.pdf - Vision-Guided Chunking
    chunks = pdf_chunks["2506.16035v2.pdf"]
    all_generated_pairs.extend([
        {"question": "What challenges do traditional text-based chunking methods face?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "How does the vision-guided chunking approach handle multi-page tables?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What is the fundamental dependency of RAG system effectiveness?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What opportunity do Large Multimodal Models present for document processing?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
    ])

    # PDF 10: 2507.09477v2.pdf - Agentic RAG with Deep Reasoning
    chunks = pdf_chunks["2507.09477v2.pdf"]
    all_generated_pairs.extend([
        {"question": "What limitations do RAG and purely reasoning-oriented approaches each have?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What are the three categories covered in this RAG-Reasoning survey?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What two fundamental limitations hinder LLM effectiveness?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
        {"question": "How do Synergized RAG-Reasoning frameworks work?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
    ])

    # PDF 11: 2508.01959v1.pdf - SitEmb (Situated Embeddings)
    chunks = pdf_chunks["2508.01959v1.pdf"]
    all_generated_pairs.extend([
        {"question": "What problem occurs when splitting long documents into chunks for RAG?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What two reasons limit the gains from encoding longer context windows?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What is the key idea behind situated embeddings (SitEmb)?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "How does the SitEmb model perform compared to state-of-the-art embedding models?", "passage": chunks[2]["passage"], "metadata": {"source": chunks[2]["source_file"], "page": chunks[2]["page"]}},
    ])

    # PDF 12: 2508.10419v3.pdf - ComoRAG
    chunks = pdf_chunks["2508.10419v3.pdf"]
    all_generated_pairs.extend([
        {"question": "What shortcoming of traditional RAG methods does ComoRAG address?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What principle does ComoRAG hold about narrative reasoning?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "How does ComoRAG interact with memory during reasoning cycles?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What performance improvements does ComoRAG achieve on long-context narrative benchmarks?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
    ])

    # PDF 13: 2510.12323v1.pdf
    chunks = pdf_chunks["2510.12323v1.pdf"]
    all_generated_pairs.extend([
        {"question": "What is the main topic of this research paper?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What methodology or approach is proposed?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What are the key contributions of this work?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What problem does this research address?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
    ])

    # PDF 14: Choy20163D.pdf
    chunks = pdf_chunks["Choy20163D.pdf"]
    all_generated_pairs.extend([
        {"question": "What is the focus of this 3D reconstruction research?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What techniques or methods are employed?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What are the main challenges addressed in this work?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
        {"question": "What applications or use cases does this research target?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
    ])

    # PDF 15: PV-RCNN_Point-Voxel_Feature_Set_Abstraction_for_3D_Object_Detection.pdf
    chunks = pdf_chunks["PV-RCNN_Point-Voxel_Feature_Set_Abstraction_for_3D_Object_Detection.pdf"]
    all_generated_pairs.extend([
        {"question": "What is PV-RCNN and what problem does it solve?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What is the key innovation of combining point and voxel features?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "What task is PV-RCNN designed for?", "passage": chunks[0]["passage"], "metadata": {"source": chunks[0]["source_file"], "page": chunks[0]["page"]}},
        {"question": "How does feature set abstraction benefit 3D object detection?", "passage": chunks[1]["passage"], "metadata": {"source": chunks[1]["source_file"], "page": chunks[1]["page"]}},
    ])

    # Save all generated pairs
    print(f"Generated {len(all_generated_pairs)} Q&A pairs for {len(pdf_chunks)} PDFs")
    print(f"  ({len(all_generated_pairs) // len(pdf_chunks)} questions per PDF)")
    generate_and_save_qa_pairs(all_generated_pairs)

    print("\n" + "=" * 80)
    print(f"✓ Completed PDFs {PDF_INDEX + 1} to {PDF_INDEX + len(pdf_chunks)}")
    print(f"To process next batch, set PDF_INDEX = {PDF_INDEX + NUM_PDFS}")
    print(f"Total PDFs: {len(all_pdfs)}, Remaining: {len(all_pdfs) - (PDF_INDEX + NUM_PDFS)}")
    print("=" * 80)
