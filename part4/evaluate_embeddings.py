"""
Evaluate fine-tuned E5 model against baseline nomic-embed-text.
Metrics: Recall@k, MRR (Mean Reciprocal Rank), NDCG@k
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from backend.utils import ollama_embed
from backend.config import EMBED_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_recall_at_k(similarities, k=5):
    """
    Compute Recall@k: proportion of queries where correct context is in top-k.
    For our case, diagonal elements are the correct matches.
    """
    n = similarities.shape[0]
    recalls = []
    
    for i in range(n):
        # Get top-k indices for query i
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        # Check if correct context (index i) is in top-k
        recalls.append(1.0 if i in top_k_indices else 0.0)
    
    return np.mean(recalls)


def compute_mrr(similarities):
    """
    Compute Mean Reciprocal Rank: average of reciprocal ranks of correct contexts.
    """
    n = similarities.shape[0]
    reciprocal_ranks = []
    
    for i in range(n):
        # Get ranking of all contexts for query i
        ranked_indices = np.argsort(similarities[i])[::-1]
        # Find rank of correct context (index i)
        rank = np.where(ranked_indices == i)[0][0] + 1  # +1 because rank starts at 1
        reciprocal_ranks.append(1.0 / rank)
    
    return np.mean(reciprocal_ranks)


def compute_ndcg_at_k(similarities, k=5):
    """
    Compute Normalized Discounted Cumulative Gain at k.
    """
    n = similarities.shape[0]
    ndcgs = []
    
    for i in range(n):
        # Get top-k indices for query i
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        
        # Compute DCG
        dcg = 0
        for rank, idx in enumerate(top_k_indices):
            relevance = 1.0 if idx == i else 0.0
            dcg += relevance / np.log2(rank + 2)  # +2 because rank starts at 0
        
        # Compute IDCG (ideal DCG - assuming correct answer at position 0)
        idcg = 1.0 / np.log2(2)  # Only one relevant document
        
        # Compute NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    
    return np.mean(ndcgs)


def evaluate_baseline_nomic(validation_pairs):
    """Evaluate baseline nomic-embed-text model"""
    logger.info("Evaluating baseline: nomic-embed-text")
    
    questions = ["query: " + pair['question'] for pair in validation_pairs]
    contexts = ["passage: " + pair['context'] for pair in validation_pairs]
    
    # Get embeddings using ollama
    logger.info("Computing embeddings with nomic-embed-text...")
    question_embeddings = []
    context_embeddings = []
    
    for q in questions:
        emb = ollama_embed(EMBED_MODEL, q)
        question_embeddings.append(emb)
    
    for c in contexts:
        emb = ollama_embed(EMBED_MODEL, c)
        context_embeddings.append(emb)
    
    question_embeddings = np.array(question_embeddings)
    context_embeddings = np.array(context_embeddings)
    
    # Compute similarity matrix
    similarities = cosine_similarity(question_embeddings, context_embeddings)
    
    # Compute metrics
    recall_at_1 = compute_recall_at_k(similarities, k=1)
    recall_at_3 = compute_recall_at_k(similarities, k=3)
    recall_at_5 = compute_recall_at_k(similarities, k=5)
    mrr = compute_mrr(similarities)
    ndcg_at_5 = compute_ndcg_at_k(similarities, k=5)
    
    results = {
        "model": "nomic-embed-text (baseline)",
        "recall@1": float(recall_at_1),
        "recall@3": float(recall_at_3),
        "recall@5": float(recall_at_5),
        "mrr": float(mrr),
        "ndcg@5": float(ndcg_at_5),
    }
    
    logger.info(f"Baseline Results: {results}")
    return results


def evaluate_finetuned_e5(validation_pairs, model_path='./finetuned_e5_st'):
    """Evaluate fine-tuned E5 model"""
    logger.info(f"Evaluating fine-tuned E5 model from {model_path}")
    
    # Load fine-tuned model
    model = SentenceTransformer(model_path)
    
    questions = ["query: " + pair['question'] for pair in validation_pairs]
    contexts = ["passage: " + pair['context'] for pair in validation_pairs]
    
    # Get embeddings
    logger.info("Computing embeddings with fine-tuned E5...")
    question_embeddings = model.encode(questions, convert_to_numpy=True, show_progress_bar=True)
    context_embeddings = model.encode(contexts, convert_to_numpy=True, show_progress_bar=True)
    
    # Compute similarity matrix
    similarities = cosine_similarity(question_embeddings, context_embeddings)
    
    # Compute metrics
    recall_at_1 = compute_recall_at_k(similarities, k=1)
    recall_at_3 = compute_recall_at_k(similarities, k=3)
    recall_at_5 = compute_recall_at_k(similarities, k=5)
    mrr = compute_mrr(similarities)
    ndcg_at_5 = compute_ndcg_at_k(similarities, k=5)
    
    results = {
        "model": f"fine-tuned E5 ({model_path})",
        "recall@1": float(recall_at_1),
        "recall@3": float(recall_at_3),
        "recall@5": float(recall_at_5),
        "mrr": float(mrr),
        "ndcg@5": float(ndcg_at_5),
    }
    
    logger.info(f"Fine-tuned E5 Results: {results}")
    return results


def evaluate_baseline_e5(validation_pairs):
    """Evaluate baseline E5 model (before fine-tuning)"""
    logger.info("Evaluating baseline E5 model (before fine-tuning)")
    
    # Load baseline E5 model
    model = SentenceTransformer('intfloat/e5-base-v2')
    
    questions = ["query: " + pair['question'] for pair in validation_pairs]
    contexts = ["passage: " + pair['context'] for pair in validation_pairs]
    
    # Get embeddings
    logger.info("Computing embeddings with baseline E5...")
    question_embeddings = model.encode(questions, convert_to_numpy=True, show_progress_bar=True)
    context_embeddings = model.encode(contexts, convert_to_numpy=True, show_progress_bar=True)
    
    # Compute similarity matrix
    similarities = cosine_similarity(question_embeddings, context_embeddings)
    
    # Compute metrics
    recall_at_1 = compute_recall_at_k(similarities, k=1)
    recall_at_3 = compute_recall_at_k(similarities, k=3)
    recall_at_5 = compute_recall_at_k(similarities, k=5)
    mrr = compute_mrr(similarities)
    ndcg_at_5 = compute_ndcg_at_k(similarities, k=5)
    
    results = {
        "model": "E5-base-v2 (baseline)",
        "recall@1": float(recall_at_1),
        "recall@3": float(recall_at_3),
        "recall@5": float(recall_at_5),
        "mrr": float(mrr),
        "ndcg@5": float(ndcg_at_5),
    }
    
    logger.info(f"Baseline E5 Results: {results}")
    return results


def compare_models(validation_pairs):
    """Compare all models and save results"""
    logger.info("\n" + "="*80)
    logger.info("Starting Model Comparison")
    logger.info("="*80 + "\n")
    
    results = []
    
    # Evaluate baseline E5
    try:
        baseline_e5_results = evaluate_baseline_e5(validation_pairs)
        results.append(baseline_e5_results)
    except Exception as e:
        logger.error(f"Error evaluating baseline E5: {e}")
    
    # Evaluate fine-tuned E5
    try:
        finetuned_e5_results = evaluate_finetuned_e5(validation_pairs)
        results.append(finetuned_e5_results)
    except Exception as e:
        logger.error(f"Error evaluating fine-tuned E5: {e}")
    
    # Evaluate baseline nomic
    try:
        nomic_results = evaluate_baseline_nomic(validation_pairs)
        results.append(nomic_results)
    except Exception as e:
        logger.error(f"Error evaluating nomic: {e}")
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison table
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Model':<40} {'Recall@1':<12} {'Recall@3':<12} {'Recall@5':<12} {'MRR':<12} {'NDCG@5':<12}")
    logger.info("-"*80)
    
    for result in results:
        logger.info(
            f"{result['model']:<40} "
            f"{result['recall@1']:<12.4f} "
            f"{result['recall@3']:<12.4f} "
            f"{result['recall@5']:<12.4f} "
            f"{result['mrr']:<12.4f} "
            f"{result['ndcg@5']:<12.4f}"
        )
    
    logger.info("="*80)
    
    # Calculate improvement
    if len(results) >= 2:
        baseline = results[0]
        finetuned = results[1]
        
        logger.info("\nIMPROVEMENT OVER BASELINE E5:")
        for metric in ['recall@1', 'recall@3', 'recall@5', 'mrr', 'ndcg@5']:
            improvement = ((finetuned[metric] - baseline[metric]) / baseline[metric]) * 100
            logger.info(f"  {metric}: {improvement:+.2f}%")
    
    return results


if __name__ == "__main__":
    # Load validation data
    with open('validation_data.json', 'r') as f:
        validation_pairs = json.load(f)
    
    logger.info(f"Loaded {len(validation_pairs)} validation pairs")
    
    # Run comparison
    results = compare_models(validation_pairs)
    
    logger.info("\nResults saved to evaluation_results.json")