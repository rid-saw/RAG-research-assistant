# src/fine_tune_embeddings.py
"""
Fine-tune embedding model on your domain-specific data.
"""

import json
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random
import numpy as np


def load_training_data(filepath: str = "evaluation/training_pairs.json"):
    """Load training pairs from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def prepare_training_examples(training_pairs: list) -> list:
    """Convert training pairs to InputExample format."""
    examples = []
    for pair in training_pairs:
        examples.append(
            InputExample(
                texts=[pair["question"], pair["passage"]]
            )
        )
    return examples


def evaluate_retrieval(model, queries: list, passages: list, k: int = 10):
    """
    Simple retrieval evaluation.
    
    For each query, check if the correct passage is in top-k results.
    
    Args:
        model: SentenceTransformer model
        queries: List of query strings
        passages: List of passage strings (index matches queries)
        k: Top-k to consider
        
    Returns:
        Dictionary with metrics
    """
    # Encode all queries and passages
    query_embeddings = model.encode(queries, convert_to_numpy=True)
    passage_embeddings = model.encode(passages, convert_to_numpy=True)
    
    # Calculate metrics
    hits_at_k = 0
    mrr_sum = 0
    
    for i, query_emb in enumerate(query_embeddings):
        # Calculate similarities to all passages
        similarities = np.dot(passage_embeddings, query_emb)
        
        # Get ranking (indices sorted by similarity, descending)
        ranking = np.argsort(similarities)[::-1]
        
        # Find where the correct passage ranks
        correct_idx = i  # The matching passage has the same index
        rank = np.where(ranking == correct_idx)[0][0] + 1  # 1-indexed
        
        # Hits@k
        if rank <= k:
            hits_at_k += 1
        
        # MRR (Mean Reciprocal Rank)
        mrr_sum += 1.0 / rank
    
    n = len(queries)
    
    return {
        f"hits@{k}": hits_at_k / n,
        f"recall@{k}": hits_at_k / n,
        "mrr": mrr_sum / n,
        "total_queries": n
    }


def fine_tune_embeddings(
    base_model: str = "all-MiniLM-L6-v2",
    output_path: str = "models/fine-tuned-embeddings",
    training_file: str = "evaluation/training_pairs.json",
    epochs: int = 3,
    batch_size: int = 16,
    eval_split: float = 0.2
):
    """
    Fine-tune embedding model on domain-specific data.
    
    Args:
        base_model: Starting model to fine-tune
        output_path: Where to save fine-tuned model
        training_file: Path to training pairs JSON
        epochs: Number of training epochs
        batch_size: Training batch size
        eval_split: Fraction of data for evaluation
    """
    print("=" * 50)
    print("Fine-Tuning Embedding Model")
    print("=" * 50)
    
    # Load training data
    print("\n[1/6] Loading training data...")
    all_pairs = load_training_data(training_file)
    print(f"Loaded {len(all_pairs)} total pairs")
    
    # Split into train and eval
    print("\n[2/6] Splitting data...")
    random.seed(42)  # For reproducibility
    random.shuffle(all_pairs)
    
    split_idx = int(len(all_pairs) * (1 - eval_split))
    train_pairs = all_pairs[:split_idx]
    eval_pairs = all_pairs[split_idx:]
    
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Evaluation pairs: {len(eval_pairs)}")
    
    # Prepare training examples
    print("\n[3/6] Preparing training examples...")
    train_examples = prepare_training_examples(train_pairs)
    
    # Prepare evaluation data
    eval_queries = [p["question"] for p in eval_pairs]
    eval_passages = [p["passage"] for p in eval_pairs]
    
    # Load base model
    print(f"\n[4/6] Loading base model: {base_model}...")
    model = SentenceTransformer(base_model)
    
    # Evaluate BEFORE fine-tuning
    print("\n[5/6] Evaluating BEFORE fine-tuning...")
    results_before = evaluate_retrieval(model, eval_queries, eval_passages)
    print(f"  Hits@10: {results_before['hits@10']:.4f}")
    print(f"  MRR: {results_before['mrr']:.4f}")
    
    # Fine-tune
    print(f"\n[6/6] Fine-tuning for {epochs} epochs...")
    
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=batch_size
    )
    
    # Use MultipleNegativesRankingLoss - great for retrieval
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=10,
        show_progress_bar=True
    )
    
    # Evaluate AFTER fine-tuning
    print("\n--- Evaluating AFTER fine-tuning ---")
    results_after = evaluate_retrieval(model, eval_queries, eval_passages)
    print(f"  Hits@10: {results_after['hits@10']:.4f}")
    print(f"  MRR: {results_after['mrr']:.4f}")
    
    # Save model
    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)
    
    # Print comparison
    print("\n" + "=" * 50)
    print("FINE-TUNING RESULTS")
    print("=" * 50)
    
    print(f"\n{'Metric':<20} {'Before':<15} {'After':<15} {'Change':<15}")
    print("-" * 65)
    
    for metric in ["hits@10", "mrr"]:
        before_val = results_before[metric]
        after_val = results_after[metric]
        
        if before_val > 0:
            change = ((after_val - before_val) / before_val) * 100
            change_str = f"{'+' if change > 0 else ''}{change:.1f}%"
        else:
            if after_val > 0:
                change_str = "+∞%"
            else:
                change_str = "0%"
        
        print(f"{metric.upper():<20} {before_val:<15.4f} {after_val:<15.4f} {change_str:<15}")
    
    print(f"\nEvaluation set size: {len(eval_queries)} queries")
    print(f"Fine-tuned model saved to: {output_path}")
    
    # Save results to file for README
    results_summary = {
        "before": results_before,
        "after": results_after,
        "training_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "epochs": epochs
    }
    
    with open(os.path.join(output_path, "training_results.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"Results saved to: {output_path}/training_results.json")
    
    return results_before, results_after


if __name__ == "__main__":
    before, after = fine_tune_embeddings(
        epochs=3,
        batch_size=16
    )