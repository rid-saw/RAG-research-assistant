# evaluation/evaluate.py
"""
Rigorous evaluation script for embedding models.

Features:
- Proper negative sampling (test against 100+ distractors, not just paired passages)
- K-fold cross-validation for stable metrics
- Multiple metrics: MRR, Hits@1/5/10, NDCG@10
- Comparison between base and fine-tuned models
- Per-query failure analysis
"""

import json
import os
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold


@dataclass
class EvalResult:
    """Container for evaluation metrics."""
    mrr: float
    hits_at_1: float
    hits_at_5: float
    hits_at_10: float
    ndcg_at_10: float
    total_queries: int

    def to_dict(self) -> dict:
        return {
            "mrr": round(self.mrr, 4),
            "hits@1": round(self.hits_at_1, 4),
            "hits@5": round(self.hits_at_5, 4),
            "hits@10": round(self.hits_at_10, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "total_queries": self.total_queries
        }

    def __str__(self) -> str:
        return (
            f"  MRR:      {self.mrr:.4f}\n"
            f"  Hits@1:   {self.hits_at_1:.4f}\n"
            f"  Hits@5:   {self.hits_at_5:.4f}\n"
            f"  Hits@10:  {self.hits_at_10:.4f}\n"
            f"  NDCG@10:  {self.ndcg_at_10:.4f}\n"
            f"  Queries:  {self.total_queries}"
        )


def load_training_pairs(filepath: str = "evaluation/training_pairs.json") -> List[Dict]:
    """Load question-passage pairs from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_unique_passages(pairs: List[Dict]) -> List[str]:
    """Extract unique passages from pairs (for negative sampling pool)."""
    seen = set()
    unique = []
    for pair in pairs:
        passage = pair["passage"]
        # Use hash of first 100 chars to detect duplicates
        key = passage[:100]
        if key not in seen:
            seen.add(key)
            unique.append(passage)
    return unique


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain at k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """Calculate Normalized DCG at k."""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_with_negatives(
    model: SentenceTransformer,
    eval_pairs: List[Dict],
    all_passages: List[str],
    num_negatives: int = 100,
    seed: int = 42
) -> Tuple[EvalResult, List[Dict]]:
    """
    Evaluate retrieval with proper negative sampling.

    For each query, we:
    1. Take the correct passage
    2. Sample num_negatives random passages as distractors
    3. Rank all candidates and find where correct passage lands

    This simulates realistic retrieval where the model must find
    the needle (correct passage) in a haystack (many candidates).

    Args:
        model: SentenceTransformer model to evaluate
        eval_pairs: List of {question, passage} pairs
        all_passages: Pool of all passages for negative sampling
        num_negatives: Number of distractor passages per query
        seed: Random seed for reproducibility

    Returns:
        EvalResult with metrics, and per-query details
    """
    random.seed(seed)
    np.random.seed(seed)

    # Pre-encode all passages for efficiency
    print("    Encoding passage pool...")
    all_passage_embeddings = model.encode(all_passages, convert_to_numpy=True, show_progress_bar=False)
    passage_to_idx = {p[:100]: i for i, p in enumerate(all_passages)}

    mrr_sum = 0.0
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    ndcg_sum = 0.0

    per_query_results = []

    for pair in eval_pairs:
        query = pair["question"]
        correct_passage = pair["passage"]
        correct_key = correct_passage[:100]

        # Get correct passage embedding
        if correct_key in passage_to_idx:
            correct_idx = passage_to_idx[correct_key]
            correct_emb = all_passage_embeddings[correct_idx]
        else:
            # Passage not in pool, encode it
            correct_emb = model.encode(correct_passage, convert_to_numpy=True)

        # Sample negative passages (excluding correct one)
        negative_indices = []
        for idx, p in enumerate(all_passages):
            if p[:100] != correct_key:
                negative_indices.append(idx)

        # Sample up to num_negatives
        if len(negative_indices) > num_negatives:
            sampled_neg_indices = random.sample(negative_indices, num_negatives)
        else:
            sampled_neg_indices = negative_indices

        # Build candidate pool: correct passage + negatives
        candidate_embeddings = [correct_emb]
        candidate_embeddings.extend([all_passage_embeddings[i] for i in sampled_neg_indices])
        candidate_embeddings = np.array(candidate_embeddings)

        # Encode query
        query_emb = model.encode(query, convert_to_numpy=True)

        # Calculate similarities
        similarities = np.dot(candidate_embeddings, query_emb)

        # Rank candidates (descending similarity)
        ranking = np.argsort(similarities)[::-1]

        # Find rank of correct passage (it's at index 0 in candidates)
        rank = np.where(ranking == 0)[0][0] + 1  # 1-indexed

        # Calculate metrics
        mrr_sum += 1.0 / rank

        if rank <= 1:
            hits_at_1 += 1
        if rank <= 5:
            hits_at_5 += 1
        if rank <= 10:
            hits_at_10 += 1

        # NDCG: relevance is 1 for correct, 0 for negatives
        relevances = [1 if i == 0 else 0 for i in ranking]
        ndcg_sum += ndcg_at_k(relevances, 10)

        # Store per-query result
        per_query_results.append({
            "question": query[:80] + "..." if len(query) > 80 else query,
            "rank": rank,
            "mrr_contribution": 1.0 / rank,
            "in_top_10": rank <= 10,
            "source": pair.get("metadata", {}).get("source", "unknown")
        })

    n = len(eval_pairs)

    result = EvalResult(
        mrr=mrr_sum / n,
        hits_at_1=hits_at_1 / n,
        hits_at_5=hits_at_5 / n,
        hits_at_10=hits_at_10 / n,
        ndcg_at_10=ndcg_sum / n,
        total_queries=n
    )

    return result, per_query_results


def kfold_evaluation(
    model: SentenceTransformer,
    all_pairs: List[Dict],
    n_folds: int = 5,
    num_negatives: int = 100,
    seed: int = 42
) -> Tuple[EvalResult, List[EvalResult]]:
    """
    Perform k-fold cross-validation evaluation.

    Each fold uses different pairs as the evaluation set,
    giving more stable metrics with limited data.

    Args:
        model: Model to evaluate
        all_pairs: All question-passage pairs
        n_folds: Number of folds
        num_negatives: Negatives per query
        seed: Random seed

    Returns:
        Average EvalResult across folds, and per-fold results
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Get all unique passages for negative sampling
    all_passages = get_unique_passages(all_pairs)

    fold_results = []
    pairs_array = np.array(all_pairs, dtype=object)

    for fold_idx, (train_idx, eval_idx) in enumerate(kf.split(pairs_array)):
        print(f"    Fold {fold_idx + 1}/{n_folds}...")

        eval_pairs = [all_pairs[i] for i in eval_idx]

        result, _ = evaluate_with_negatives(
            model=model,
            eval_pairs=eval_pairs,
            all_passages=all_passages,
            num_negatives=num_negatives,
            seed=seed + fold_idx
        )
        fold_results.append(result)

    # Average across folds
    avg_result = EvalResult(
        mrr=np.mean([r.mrr for r in fold_results]),
        hits_at_1=np.mean([r.hits_at_1 for r in fold_results]),
        hits_at_5=np.mean([r.hits_at_5 for r in fold_results]),
        hits_at_10=np.mean([r.hits_at_10 for r in fold_results]),
        ndcg_at_10=np.mean([r.ndcg_at_10 for r in fold_results]),
        total_queries=sum(r.total_queries for r in fold_results)
    )

    return avg_result, fold_results


def compare_models(
    base_model_name: str = "all-MiniLM-L6-v2",
    finetuned_model_path: str = "models/fine-tuned-embeddings",
    training_pairs_path: str = "evaluation/training_pairs.json",
    num_negatives: int = 100,
    n_folds: int = 5,
    seed: int = 42,
    output_path: Optional[str] = "evaluation/evaluation_results.json"
) -> Dict:
    """
    Compare base model vs fine-tuned model with rigorous evaluation.

    Args:
        base_model_name: HuggingFace model name for base model
        finetuned_model_path: Path to fine-tuned model
        training_pairs_path: Path to training pairs JSON
        num_negatives: Number of negative samples per query
        n_folds: Number of cross-validation folds
        seed: Random seed
        output_path: Where to save results (None to skip saving)

    Returns:
        Dictionary with all results
    """
    print("=" * 60)
    print("RIGOROUS EMBEDDING MODEL EVALUATION")
    print("=" * 60)
    print(f"\nSettings:")
    print(f"  - Negative samples per query: {num_negatives}")
    print(f"  - Cross-validation folds: {n_folds}")
    print(f"  - Random seed: {seed}")

    # Load data
    print(f"\n[1/4] Loading training pairs from {training_pairs_path}...")
    all_pairs = load_training_pairs(training_pairs_path)
    all_passages = get_unique_passages(all_pairs)
    print(f"  Loaded {len(all_pairs)} pairs")
    print(f"  Unique passages: {len(all_passages)}")

    # Load base model
    print(f"\n[2/4] Evaluating BASE model: {base_model_name}")
    base_model = SentenceTransformer(base_model_name)

    print(f"  Running {n_folds}-fold cross-validation...")
    base_result, base_folds = kfold_evaluation(
        model=base_model,
        all_pairs=all_pairs,
        n_folds=n_folds,
        num_negatives=num_negatives,
        seed=seed
    )

    print(f"\n  BASE MODEL RESULTS (averaged across {n_folds} folds):")
    print(base_result)

    # Load fine-tuned model
    print(f"\n[3/4] Evaluating FINE-TUNED model: {finetuned_model_path}")

    if not os.path.exists(finetuned_model_path):
        print(f"  WARNING: Fine-tuned model not found at {finetuned_model_path}")
        print(f"  Skipping fine-tuned evaluation.")
        finetuned_result = None
        finetuned_folds = None
    else:
        finetuned_model = SentenceTransformer(finetuned_model_path)

        print(f"  Running {n_folds}-fold cross-validation...")
        finetuned_result, finetuned_folds = kfold_evaluation(
            model=finetuned_model,
            all_pairs=all_pairs,
            n_folds=n_folds,
            num_negatives=num_negatives,
            seed=seed
        )

        print(f"\n  FINE-TUNED MODEL RESULTS (averaged across {n_folds} folds):")
        print(finetuned_result)

    # Comparison
    print("\n[4/4] COMPARISON")
    print("=" * 60)

    if finetuned_result:
        print(f"\n{'Metric':<12} {'Base':<12} {'Fine-tuned':<12} {'Change':<12}")
        print("-" * 48)

        metrics = [
            ("MRR", base_result.mrr, finetuned_result.mrr),
            ("Hits@1", base_result.hits_at_1, finetuned_result.hits_at_1),
            ("Hits@5", base_result.hits_at_5, finetuned_result.hits_at_5),
            ("Hits@10", base_result.hits_at_10, finetuned_result.hits_at_10),
            ("NDCG@10", base_result.ndcg_at_10, finetuned_result.ndcg_at_10),
        ]

        for name, base_val, ft_val in metrics:
            if base_val > 0:
                change_pct = ((ft_val - base_val) / base_val) * 100
                change_str = f"{'+' if change_pct > 0 else ''}{change_pct:.1f}%"
            else:
                change_str = "N/A"
            print(f"{name:<12} {base_val:<12.4f} {ft_val:<12.4f} {change_str:<12}")

        # Per-fold variance
        print(f"\n  Fold-level variance (MRR):")
        base_mrrs = [r.mrr for r in base_folds]
        ft_mrrs = [r.mrr for r in finetuned_folds]
        print(f"    Base:       mean={np.mean(base_mrrs):.4f}, std={np.std(base_mrrs):.4f}")
        print(f"    Fine-tuned: mean={np.mean(ft_mrrs):.4f}, std={np.std(ft_mrrs):.4f}")

    # Build results dictionary
    results = {
        "settings": {
            "num_negatives": num_negatives,
            "n_folds": n_folds,
            "seed": seed,
            "total_pairs": len(all_pairs),
            "unique_passages": len(all_passages)
        },
        "base_model": {
            "name": base_model_name,
            "average": base_result.to_dict(),
            "per_fold": [r.to_dict() for r in base_folds]
        }
    }

    if finetuned_result:
        results["finetuned_model"] = {
            "path": finetuned_model_path,
            "average": finetuned_result.to_dict(),
            "per_fold": [r.to_dict() for r in finetuned_folds]
        }

        # Calculate improvement
        results["improvement"] = {
            "mrr": round((finetuned_result.mrr - base_result.mrr) / base_result.mrr * 100, 2) if base_result.mrr > 0 else None,
            "hits@1": round((finetuned_result.hits_at_1 - base_result.hits_at_1) / base_result.hits_at_1 * 100, 2) if base_result.hits_at_1 > 0 else None,
            "hits@10": round((finetuned_result.hits_at_10 - base_result.hits_at_10) / base_result.hits_at_10 * 100, 2) if base_result.hits_at_10 > 0 else None,
        }

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {output_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return results


def analyze_failures(
    model_path: str,
    training_pairs_path: str = "evaluation/training_pairs.json",
    num_negatives: int = 100,
    top_n_failures: int = 10
) -> List[Dict]:
    """
    Analyze which queries the model fails on.

    Returns the worst-performing queries to help diagnose issues.
    """
    print(f"\nAnalyzing failures for: {model_path}")

    model = SentenceTransformer(model_path)
    all_pairs = load_training_pairs(training_pairs_path)
    all_passages = get_unique_passages(all_pairs)

    _, per_query = evaluate_with_negatives(
        model=model,
        eval_pairs=all_pairs,
        all_passages=all_passages,
        num_negatives=num_negatives
    )

    # Sort by rank (worst first)
    failures = sorted(per_query, key=lambda x: x["rank"], reverse=True)

    print(f"\nTop {top_n_failures} worst-performing queries:")
    print("-" * 80)

    for i, fail in enumerate(failures[:top_n_failures]):
        print(f"{i+1}. Rank: {fail['rank']}")
        print(f"   Question: {fail['question']}")
        print(f"   Source: {fail['source']}")
        print()

    return failures[:top_n_failures]


if __name__ == "__main__":
    # Run full comparison
    results = compare_models(
        base_model_name="all-MiniLM-L6-v2",
        finetuned_model_path="models/fine-tuned-embeddings",
        training_pairs_path="evaluation/training_pairs.json",
        num_negatives=100,  # Test against 100 distractors
        n_folds=5,          # 5-fold cross-validation
        seed=42
    )

    # Optionally analyze failures
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS (Fine-tuned model)")
    print("=" * 60)

    if os.path.exists("models/fine-tuned-embeddings"):
        analyze_failures(
            model_path="models/fine-tuned-embeddings",
            top_n_failures=10
        )
