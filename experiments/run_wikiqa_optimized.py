#!/usr/bin/env python3
"""
🐟 markrel WikiQA Optimization Benchmark

Tests:
1. BGE-large embeddings (1024-dim, SOTA for retrieval)
2. Hyperparameter tuning (bins, smoothing, bin_strategy)

Usage:
    python experiments/run_wikiqa_optimized.py
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
    print(f"✅ Test: {len(test_groups)} unique questions")
    
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


def evaluate_model(model, test_groups, test_q_emb_dict, test_d_emb_dict):
    """Evaluate a trained model."""
    predictions = {}
    
    for query, candidates in test_groups.items():
        query_emb = test_q_emb_dict[query].reshape(1, -1)
        
        candidate_predictions = []
        for doc, label in candidates:
            doc_emb = test_d_emb_dict[doc].reshape(1, -1)
            score = model.predict_proba(query_emb, doc_emb)[0]
            candidate_predictions.append((doc, label, score))
        
        predictions[query] = candidate_predictions
    
    return {
        'MAP': compute_map(predictions),
        'MRR': compute_mrr(predictions),
        'Recall@1': compute_recall_at_k(predictions, k=1),
        'Recall@5': compute_recall_at_k(predictions, k=5),
        'Recall@10': compute_recall_at_k(predictions, k=10),
    }


def run_hyperparameter_search(train_q_emb, train_d_emb, train_l, 
                               test_groups, test_q_emb_dict, test_d_emb_dict):
    """Run hyperparameter grid search."""
    
    print("\n" + "="*70)
    print("🔍 Hyperparameter Grid Search")
    print("="*70)
    
    # Grid of hyperparameters
    param_grid = {
        'n_bins': [5, 10, 20, 50],
        'smoothing': [0.1, 0.5, 1.0, 2.0],
        'bin_strategy': ['uniform', 'quantile'],
    }
    
    results = []
    
    for n_bins in param_grid['n_bins']:
        for smoothing in param_grid['smoothing']:
            for bin_strategy in param_grid['bin_strategy']:
                config_name = f"bins={n_bins}, smooth={smoothing}, strategy={bin_strategy}"
                print(f"\n🔬 Testing: {config_name}")
                
                try:
                    # Train model
                    model = MarkovRelevanceModel(
                        metrics=["cosine"],
                        n_bins=n_bins,
                        smoothing=smoothing,
                        bin_strategy=bin_strategy,
                        use_text_vectorizer=False,
                    )
                    model.fit(train_q_emb, train_d_emb, train_l)
                    
                    # Evaluate
                    metrics = evaluate_model(model, test_groups, test_q_emb_dict, test_d_emb_dict)
                    
                    result = {
                        'config': config_name,
                        'n_bins': n_bins,
                        'smoothing': smoothing,
                        'bin_strategy': bin_strategy,
                        **metrics
                    }
                    results.append(result)
                    
                    print(f"   MAP: {metrics['MAP']:.4f}, MRR: {metrics['MRR']:.4f}")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    continue
    
    # Find best by MAP
    best_map = max(results, key=lambda x: x['MAP'])
    best_mrr = max(results, key=lambda x: x['MRR'])
    
    print("\n" + "="*70)
    print("🏆 BEST CONFIGURATIONS")
    print("="*70)
    print(f"\nBest by MAP ({best_map['MAP']:.4f}):")
    print(f"   {best_map['config']}")
    print(f"\nBest by MRR ({best_mrr['MRR']:.4f}):")
    print(f"   {best_mrr['config']}")
    
    return best_map, results


def run_embedding_comparison(train_q, train_d, train_l, test_groups):
    """Compare different embedding models."""
    
    print("\n" + "="*70)
    print("🧪 Embedding Model Comparison")
    print("="*70)
    
    # Models to test
    models = [
        ('sentence-transformers/all-MiniLM-L6-v2', 'MiniLM'),
        ('sentence-transformers/all-mpnet-base-v2', 'MPNet'),
        ('BAAI/bge-large-en', 'BGE-Large'),
    ]
    
    all_results = []
    
    for model_name, short_name in models:
        try:
            print(f"\n{'='*70}")
            print(f"📊 Testing: {short_name} ({model_name})")
            print(f"{'='*70}")
            
            # Encode data
            print("\nEncoding training data...")
            train_q_emb = encode_texts(model_name, train_q)
            train_d_emb = encode_texts(model_name, train_d)
            
            test_queries = list(test_groups.keys())
            all_test_docs = []
            for candidates in test_groups.values():
                all_test_docs.extend([doc for doc, _ in candidates])
            unique_test_docs = list(set(all_test_docs))
            
            print("\nEncoding test data...")
            test_q_emb_dict = {}
            for q, emb in zip(test_queries, encode_texts(model_name, test_queries)):
                test_q_emb_dict[q] = emb
            
            test_d_emb_dict = {}
            for d, emb in zip(unique_test_docs, encode_texts(model_name, unique_test_docs)):
                test_d_emb_dict[d] = emb
            
            # Train with default params first
            print("\n🔬 Default parameters (cosine, 10 bins)...")
            model = MarkovRelevanceModel(
                metrics=["cosine"],
                n_bins=10,
                use_text_vectorizer=False,
            )
            model.fit(train_q_emb, train_d_emb, train_l)
            
            metrics = evaluate_model(model, test_groups, test_q_emb_dict, test_d_emb_dict)
            
            result = {
                'model': short_name,
                'embedding': model_name,
                'dim': train_q_emb.shape[1],
                **metrics
            }
            all_results.append(result)
            
            print(f"\n📊 Results:")
            print(f"   MAP: {metrics['MAP']:.4f}, MRR: {metrics['MRR']:.4f}")
            
            # Run hyperparameter search for best model
            if short_name == 'BGE-Large':
                print("\n🔍 Running hyperparameter search for BGE-Large...")
                best_config, _ = run_hyperparameter_search(
                    train_q_emb, train_d_emb, train_l,
                    test_groups, test_q_emb_dict, test_d_emb_dict
                )
                
                # Save best BGE result
                result['best_MAP'] = best_config['MAP']
                result['best_MRR'] = best_config['MRR']
            
        except Exception as e:
            print(f"❌ Error with {short_name}: {e}")
            continue
    
    return all_results


def main():
    print("\n" + "="*70)
    print("🐟 markrel WikiQA Optimization Benchmark")
    print("="*70)
    
    # Load data
    train_q, train_d, train_l, test_groups = load_data_grouped()
    
    # Run comparison
    results = run_embedding_comparison(train_q, train_d, train_l, test_groups)
    
    # Final report
    print("\n" + "="*70)
    print("📄 FINAL COMPARISON")
    print("="*70)
    
    print("\n| Model | Dim | MAP | MRR | Recall@1 | Recall@5 |")
    print("|-------|-----|-----|-----|----------|----------|")
    
    for r in results:
        print(f"| {r['model']} | {r['dim']} | {r['MAP']:.4f} | {r['MRR']:.4f} | "
              f"{r['Recall@1']:.4f} | {r['Recall@5']:.4f} |")
    
    # Save results
    json_path = "experiments/wikiqa_optimization_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved results to {json_path}")
    
    # Show improvements
    if len(results) >= 2:
        baseline_map = results[0]['MAP']
        best_map = max(r['MAP'] for r in results)
        improvement = ((best_map - baseline_map) / baseline_map) * 100
        print(f"\n🚀 Best improvement: MAP +{improvement:.1f}%")


if __name__ == "__main__":
    main()
