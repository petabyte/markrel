#!/usr/bin/env python3
"""
🐟 markrel WikiQA Benchmark with Better Embeddings

Tests multiple embedding models to find the best for markrel:
- all-MiniLM-L6-v2 (384-dim) - baseline
- all-mpnet-base-v2 (768-dim) - better quality
- BAAI/bge-large-en (1024-dim) - SOTA for retrieval

Usage:
    python experiments/run_wikiqa_better_embeddings.py
"""

import json
import sys
import os
import time
import warnings
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from markrel import MarkovRelevanceModel

def load_data_grouped():
    """Load WikiQA data grouped by question."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "wikiqa", "wikiqa_processed.json"
    )
    
    print(f"📂 Loading data from {data_path}...")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Group test data by question
    test_groups = defaultdict(list)
    for i, (q, d, l) in enumerate(zip(data['test']['queries'], 
                                       data['test']['documents'], 
                                       data['test']['labels'])):
        test_groups[q].append((d, l))
    
    # Flatten train data
    train_q = data['train']['queries']
    train_d = data['train']['documents']
    train_l = data['train']['labels']
    
    print(f"✅ Train: {len(train_q)} samples")
    print(f"✅ Test: {len(test_groups)} unique questions, {data['test']['n_samples']} total")
    
    return train_q, train_d, train_l, test_groups


def encode_texts(model_name: str, texts: List[str], batch_size: int = 32):
    """Encode texts using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence-transformers not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
    
    print(f"📥 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"📝 Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    return embeddings


def compute_ap(relevances: List[int]) -> float:
    """Compute Average Precision."""
    if sum(relevances) == 0:
        return 0.0
    
    precisions = []
    num_relevant = 0
    
    for i, rel in enumerate(relevances):
        if rel == 1:
            num_relevant += 1
            precision_at_k = num_relevant / (i + 1)
            precisions.append(precision_at_k)
    
    return sum(precisions) / sum(relevances) if relevances else 0.0


def compute_map(predictions: Dict[str, List[Tuple[str, int, float]]]) -> float:
    """Compute Mean Average Precision."""
    aps = []
    
    for query, candidates in predictions.items():
        ranked = sorted(candidates, key=lambda x: x[2], reverse=True)
        relevances = [label for _, label, _ in ranked]
        ap = compute_ap(relevances)
        aps.append(ap)
    
    return sum(aps) / len(aps) if aps else 0.0


def compute_mrr(predictions: Dict[str, List[Tuple[str, int, float]]]) -> float:
    """Compute Mean Reciprocal Rank."""
    rr_sum = 0.0
    num_queries = 0
    
    for query, candidates in predictions.items():
        ranked = sorted(candidates, key=lambda x: x[2], reverse=True)
        
        for rank, (_, label, _) in enumerate(ranked, 1):
            if label == 1:
                rr_sum += 1.0 / rank
                break
        
        num_queries += 1
    
    return rr_sum / num_queries if num_queries > 0 else 0.0


def compute_recall_at_k(predictions: Dict[str, List[Tuple[str, int, float]]], k: int = 10) -> float:
    """Compute Recall@K."""
    recall_sum = 0.0
    num_queries = 0
    
    for query, candidates in predictions.items():
        ranked = sorted(candidates, key=lambda x: x[2], reverse=True)
        top_k = ranked[:k]
        
        relevant_in_k = sum(1 for _, label, _ in top_k if label == 1)
        total_relevant = sum(1 for _, label, _ in candidates if label == 1)
        
        if total_relevant > 0:
            recall_sum += relevant_in_k / total_relevant
            num_queries += 1
    
    return recall_sum / num_queries if num_queries > 0 else 0.0


def run_embedding_benchmark(train_q, train_d, train_l, test_groups, embedding_model):
    """Run benchmark with specific embedding model."""
    
    print("\n" + "="*70)
    print(f"🧪 Testing Embeddings: {embedding_model}")
    print("="*70)
    
    # Encode training data
    print("\n📊 Encoding training queries...")
    train_q_emb = encode_texts(embedding_model, train_q)
    
    print("\n📊 Encoding training documents...")
    train_d_emb = encode_texts(embedding_model, train_d)
    
    # Encode test data
    test_queries = list(test_groups.keys())
    all_test_docs = []
    for candidates in test_groups.values():
        all_test_docs.extend([doc for doc, _ in candidates])
    unique_test_docs = list(set(all_test_docs))
    
    print("\n📊 Encoding test queries...")
    test_q_emb_dict = {}
    for q, emb in zip(test_queries, encode_texts(embedding_model, test_queries)):
        test_q_emb_dict[q] = emb
    
    print("\n📊 Encoding test documents...")
    test_d_emb_dict = {}
    for d, emb in zip(unique_test_docs, encode_texts(embedding_model, unique_test_docs)):
        test_d_emb_dict[d] = emb
    
    # Train markrel
    print(f"\n🔬 Training markrel with {embedding_model}...")
    start = time.time()
    model = MarkovRelevanceModel(
        metrics=["cosine"],
        n_bins=10,
        use_text_vectorizer=False,
    )
    model.fit(train_q_emb, train_d_emb, train_l)
    train_time = time.time() - start
    
    print(f"✅ Trained in {train_time:.3f}s")
    
    # Predict and rank
    print("📝 Ranking test queries...")
    predictions = {}
    
    for query, candidates in test_groups.items():
        query_emb = test_q_emb_dict[query].reshape(1, -1)
        
        candidate_predictions = []
        for doc, label in candidates:
            doc_emb = test_d_emb_dict[doc].reshape(1, -1)
            score = model.predict_proba(query_emb, doc_emb)[0]
            candidate_predictions.append((doc, label, score))
        
        predictions[query] = candidate_predictions
    
    # Compute metrics
    map_score = compute_map(predictions)
    mrr_score = compute_mrr(predictions)
    recall_at_1 = compute_recall_at_k(predictions, k=1)
    recall_at_5 = compute_recall_at_k(predictions, k=5)
    recall_at_10 = compute_recall_at_k(predictions, k=10)
    
    results = {
        'embedding_model': embedding_model,
        'embedding_dim': train_q_emb.shape[1],
        'MAP': map_score,
        'MRR': mrr_score,
        'Recall@1': recall_at_1,
        'Recall@5': recall_at_5,
        'Recall@10': recall_at_10,
        'train_time': train_time,
    }
    
    print(f"\n📊 Results:")
    print(f"   MAP:       {map_score:.4f}")
    print(f"   MRR:       {mrr_score:.4f}")
    print(f"   Recall@1:  {recall_at_1:.4f}")
    print(f"   Recall@5:  {recall_at_5:.4f}")
    print(f"   Recall@10: {recall_at_10:.4f}")
    
    return results


def main():
    print("\n" + "="*70)
    print("🐟 markrel WikiQA Benchmark - Better Embeddings")
    print("="*70)
    
    # Load data
    train_q, train_d, train_l, test_groups = load_data_grouped()
    
    # Test different embedding models
    embedding_models = [
        'sentence-transformers/all-MiniLM-L6-v2',  # 384-dim, baseline
        'sentence-transformers/all-mpnet-base-v2',  # 768-dim, better
    ]
    
    # Optional: Add BGE if user wants (larger model, slower)
    # 'BAAI/bge-large-en',  # 1024-dim, SOTA
    
    all_results = []
    
    for model_name in embedding_models:
        try:
            results = run_embedding_benchmark(train_q, train_d, train_l, test_groups, model_name)
            all_results.append(results)
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            continue
    
    # Generate report
    print("\n" + "="*70)
    print("📄 FINAL COMPARISON")
    print("="*70)
    
    print("\n| Embedding Model | Dim | MAP | MRR | Recall@1 | Recall@5 | Recall@10 |")
    print("|-----------------|-----|-----|-----|----------|----------|-----------|")
    
    for r in all_results:
        print(f"| {r['embedding_model'].split('/')[-1]} | {r['embedding_dim']} | "
              f"{r['MAP']:.4f} | {r['MRR']:.4f} | {r['Recall@1']:.4f} | "
              f"{r['Recall@5']:.4f} | {r['Recall@10']:.4f} |")
    
    # Save results
    json_path = "experiments/wikiqa_better_embeddings_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Saved results to {json_path}")
    
    # Show improvement
    if len(all_results) >= 2:
        baseline_map = all_results[0]['MAP']
        better_map = all_results[1]['MAP']
        improvement = ((better_map - baseline_map) / baseline_map) * 100
        print(f"\n🚀 Improvement: MAP +{improvement:.1f}% with {all_results[1]['embedding_model'].split('/')[-1]}")


if __name__ == "__main__":
    main()
