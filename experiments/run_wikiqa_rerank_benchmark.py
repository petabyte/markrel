#!/usr/bin/env python3
"""
🐟 markrel WikiQA Re-ranking Benchmark with Balanced Training

This script implements a two-stage approach:
1. Initial retrieval using fast cosine similarity
2. Re-ranking with markrel trained on balanced data

This should significantly improve MAP and MRR.

Usage:
    python experiments/run_wikiqa_rerank_benchmark.py
"""

import json
import sys
import os
import time
import warnings
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Any
from collections import defaultdict

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from markrel import MarkovRelevanceModel
from sklearn.metrics import roc_auc_score

def load_data_grouped():
    """Load WikiQA data grouped by question."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "wikiqa", "wikiqa_processed.json"
    )
    
    print(f"📂 Loading data from {data_path}...")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Group train data by question
    train_groups = defaultdict(list)
    for i, (q, d, l) in enumerate(zip(data['train']['queries'], 
                                       data['train']['documents'], 
                                       data['train']['labels'])):
        train_groups[q].append((d, l))
    
    # Group test data by question
    test_groups = defaultdict(list)
    for i, (q, d, l) in enumerate(zip(data['test']['queries'], 
                                       data['test']['documents'], 
                                       data['test']['labels'])):
        test_groups[q].append((d, l))
    
    print(f"✅ Train: {len(train_groups)} unique questions, {data['train']['n_samples']} total candidates")
    print(f"✅ Test: {len(test_groups)} unique questions, {data['test']['n_samples']} total candidates")
    
    return train_groups, test_groups


def encode_texts(model_name: str, texts: List[str], batch_size: int = 64):
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
    
    return embeddings, model


def create_balanced_training_data(train_groups, query_embeddings, doc_embeddings_dict):
    """Create balanced training pairs - equal relevant and non-relevant."""
    
    balanced_queries = []
    balanced_docs = []
    balanced_labels = []
    
    for query, candidates in train_groups.items():
        # Get query embedding
        query_emb = query_embeddings.get(query)
        if query_emb is None:
            continue
        
        # Separate relevant and non-relevant
        relevant = [(doc, label) for doc, label in candidates if label == 1]
        non_relevant = [(doc, label) for doc, label in candidates if label == 0]
        
        # Balance by sampling equal number
        n_samples = min(len(relevant), len(non_relevant))
        if n_samples == 0:
            continue
        
        # Sample equally
        relevant_sample = relevant[:n_samples]
        non_relevant_sample = np.random.choice(len(non_relevant), n_samples, replace=False)
        non_relevant_sample = [non_relevant[i] for i in non_relevant_sample]
        
        # Add to balanced dataset
        for doc, label in relevant_sample + non_relevant_sample:
            balanced_queries.append(query_emb)
            doc_emb = doc_embeddings_dict.get(doc)
            if doc_emb is not None:
                balanced_docs.append(doc_emb)
                balanced_labels.append(label)
    
    print(f"✅ Balanced dataset: {len(balanced_labels)} samples")
    print(f"   Relevant: {sum(balanced_labels)} ({sum(balanced_labels)/len(balanced_labels)*100:.1f}%)")
    print(f"   Not relevant: {len(balanced_labels) - sum(balanced_labels)} ({(len(balanced_labels)-sum(balanced_labels))/len(balanced_labels)*100:.1f}%)")
    
    return np.array(balanced_queries), np.array(balanced_docs), np.array(balanced_labels)


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


def run_reranking_benchmark(train_groups, test_groups, model_name='all-MiniLM-L6-v2'):
    """Run re-ranking benchmark with balanced training."""
    
    print("\n" + "="*70)
    print(f"🧪 markrel WikiQA Re-ranking Benchmark")
    print(f"   Model: {model_name}")
    print("="*70)
    
    # Get unique queries and docs
    train_queries = list(train_groups.keys())
    all_train_docs = []
    for candidates in train_groups.values():
        all_train_docs.extend([doc for doc, _ in candidates])
    unique_train_docs = list(set(all_train_docs))
    
    test_queries = list(test_groups.keys())
    all_test_docs = []
    for candidates in test_groups.values():
        all_test_docs.extend([doc for doc, _ in candidates])
    unique_test_docs = list(set(all_test_docs))
    
    # Encode all texts
    print("\n📊 Encoding training queries...")
    train_q_embs, encoder = encode_texts(model_name, train_queries)
    train_q_emb_dict = {q: emb for q, emb in zip(train_queries, train_q_embs)}
    
    print("\n📊 Encoding training documents...")
    train_d_embs, _ = encode_texts(model_name, unique_train_docs)
    train_d_emb_dict = {d: emb for d, emb in zip(unique_train_docs, train_d_embs)}
    
    print("\n📊 Encoding test queries...")
    test_q_embs, _ = encode_texts(model_name, test_queries)
    test_q_emb_dict = {q: emb for q, emb in zip(test_queries, test_q_embs)}
    
    print("\n📊 Encoding test documents...")
    test_d_embs, _ = encode_texts(model_name, unique_test_docs)
    test_d_emb_dict = {d: emb for d, emb in zip(unique_test_docs, test_d_embs)}
    
    # Create balanced training data
    print("\n⚖️ Creating balanced training dataset...")
    train_q_balanced, train_d_balanced, train_l_balanced = create_balanced_training_data(
        train_groups, train_q_emb_dict, train_d_emb_dict
    )
    
    results = []
    
    configs = [
        {
            'name': 'markrel Re-rank (Cosine) - Balanced',
            'metrics': ['cosine'],
            'n_bins': 10,
            'initial_k': 20,  # Top-k from initial retrieval
        },
        {
            'name': 'markrel Re-rank (Cosine) - Balanced, k=50',
            'metrics': ['cosine'],
            'n_bins': 10,
            'initial_k': 50,
        },
        {
            'name': 'markrel Re-rank (Cosine + Jaccard) - Balanced',
            'metrics': ['cosine', 'jaccard'],
            'n_bins': 10,
            'initial_k': 20,
        },
    ]
    
    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"🔬 Testing: {cfg['name']}")
        print(f"{'='*70}")
        
        initial_k = cfg.pop('initial_k')
        
        # Train markrel on balanced data
        start = time.time()
        model = MarkovRelevanceModel(
            **{k: v for k, v in cfg.items() if k != 'name'},
            use_text_vectorizer=False,
        )
        model.fit(train_q_balanced, train_d_balanced, train_l_balanced)
        train_time = time.time() - start
        
        print(f"✅ Trained in {train_time:.3f}s")
        
        # Re-ranking evaluation
        print(f"📝 Re-ranking top-{initial_k} candidates...")
        predictions = {}
        
        for query, candidates in test_groups.items():
            query_emb = test_q_emb_dict[query].reshape(1, -1)
            
            # Stage 1: Initial retrieval with cosine similarity
            candidate_scores = []
            for doc, label in candidates:
                doc_emb = test_d_emb_dict[doc]
                # Compute cosine similarity
                cos_sim = np.dot(query_emb.flatten(), doc_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
                )
                candidate_scores.append((doc, label, cos_sim))
            
            # Get top-k from initial retrieval
            top_k_initial = sorted(candidate_scores, key=lambda x: x[2], reverse=True)[:initial_k]
            
            # Stage 2: Re-rank with markrel
            reranked = []
            for doc, label, _ in top_k_initial:
                doc_emb = test_d_emb_dict[doc].reshape(1, -1)
                score = model.predict_proba(query_emb, doc_emb)[0]
                reranked.append((doc, label, score))
            
            # Add remaining candidates (not re-ranked) at bottom
            top_k_docs = {doc for doc, _, _ in top_k_initial}
            remaining = [(doc, label, 0.0) for doc, label in candidates if doc not in top_k_docs]
            
            predictions[query] = reranked + remaining
        
        # Compute metrics
        map_score = compute_map(predictions)
        mrr_score = compute_mrr(predictions)
        recall_at_1 = compute_recall_at_k(predictions, k=1)
        recall_at_5 = compute_recall_at_k(predictions, k=5)
        recall_at_10 = compute_recall_at_k(predictions, k=10)
        
        metrics = {
            'name': cfg['name'],
            'config': str({k: v for k, v in cfg.items() if k != 'name'}),
            'initial_k': initial_k,
            'MAP': map_score,
            'MRR': mrr_score,
            'Recall@1': recall_at_1,
            'Recall@5': recall_at_5,
            'Recall@10': recall_at_10,
            'train_time': train_time,
        }
        
        results.append(metrics)
        
        print(f"\n📊 Results:")
        print(f"   MAP:       {map_score:.4f}")
        print(f"   MRR:       {mrr_score:.4f}")
        print(f"   Recall@1:  {recall_at_1:.4f}")
        print(f"   Recall@5:  {recall_at_5:.4f}")
        print(f"   Recall@10: {recall_at_10:.4f}")
    
    return results


def generate_report(results, output_path, model_name):
    """Generate report."""
    
    report = []
    report.append(f"# 🐟 markrel WikiQA Re-ranking Benchmark Report\n\n")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append(f"**Dataset:** WikiQA Real Dataset (from Hugging Face)\n\n")
    report.append(f"**Embedding Model:** {model_name}\n\n")
    report.append(f"**Approach:** Balanced training + Two-stage re-ranking\n\n")
    
    # Summary table
    report.append("## Results\n\n")
    report.append("| Model | Initial k | MAP | MRR | Recall@1 | Recall@5 | Recall@10 |\n")
    report.append("|-------|-----------|-----|-----|----------|----------|-----------|\n")
    
    for r in results:
        report.append(
            f"| {r['name']} | {r['initial_k']} | {r['MAP']:.4f} | "
            f"{r['MRR']:.4f} | {r['Recall@1']:.4f} | {r['Recall@5']:.4f} | {r['Recall@10']:.4f} |\n"
        )
    
    # Comparison with baseline
    report.append("\n## Comparison with Baseline\n\n")
    report.append("| Method | MAP | MRR | Notes |\n")
    report.append("|--------|-----|-----|-------|\n")
    report.append("| markrel (Cosine) - Baseline | 0.2835 | 0.2887 | Unbalanced, no re-ranking |\n")
    report.append(f"| **{results[0]['name']}** | **{results[0]['MAP']:.4f}** | **{results[0]['MRR']:.4f}** | **Balanced + re-ranking** |\n")
    
    improvement_map = ((results[0]['MAP'] - 0.2835) / 0.2835) * 100
    improvement_mrr = ((results[0]['MRR'] - 0.2887) / 0.2887) * 100
    report.append(f"\n**Improvement:** MAP +{improvement_map:.1f}%, MRR +{improvement_mrr:.1f}%\n\n")
    
    report_text = ''.join(report)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    return report_text


def main():
    print("\n" + "="*70)
    print("🐟 markrel WikiQA Re-ranking Benchmark (Balanced Training)")
    print("="*70)
    
    # Load
    train_groups, test_groups = load_data_grouped()
    
    # Run with embeddings
    model_name = 'all-MiniLM-L6-v2'
    results = run_reranking_benchmark(train_groups, test_groups, model_name)
    
    # Report
    report_path = "experiments/wikiqa_rerank_benchmark_report.md"
    report = generate_report(results, report_path, model_name)
    
    print("\n" + "="*70)
    print("📄 REPORT")
    print("="*70)
    print(report)
    
    # Save JSON
    json_path = "experiments/wikiqa_rerank_benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved to: {report_path} and {json_path}")


if __name__ == "__main__":
    main()
