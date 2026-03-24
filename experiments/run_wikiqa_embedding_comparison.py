#!/usr/bin/env python3
"""
🐟 markrel WikiQA Embedding Model Comparison

Compares multiple embedding models on WikiQA:
- RoBERTa-large (1024-dim)
- BAAI/bge-m3 (1024-dim)
- all-MiniLM-L6-v2 (384-dim) [baseline]

Usage:
    python experiments/run_wikiqa_embedding_comparison.py
"""

import json
import sys
import os
import time
import warnings
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from markrel import MarkovRelevanceModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)


def load_wikiqa_data():
    """Load WikiQA data with proper relevance labels."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "wikiqa", "wikiqa_processed.json"
    )
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data['train'], data['test']


def encode_texts(texts: List[str], model_name: str, batch_size: int = 32):
    """Encode texts using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings, model.get_sentence_embedding_dimension()


def find_optimal_threshold(y_true, y_scores):
    """Find optimal F1 threshold."""
    thresholds = np.linspace(0.001, 0.999, 500)
    best_f1, best_thresh = 0, 0.5
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, thresh
    
    return best_thresh, best_f1


def run_markrel(train_q_emb, train_d_emb, train_l, test_q_emb, test_d_emb, test_l):
    """Run markrel with cosine metric."""
    model = MarkovRelevanceModel(
        metrics=['cosine'],
        n_bins=10,
        use_text_vectorizer=False,
    )
    
    start = time.time()
    model.fit(train_q_emb, train_d_emb, train_l)
    train_time = time.time() - start
    
    start = time.time()
    probs = model.predict_proba(test_q_emb, test_d_emb)
    pred_time = time.time() - start
    
    thresh, _ = find_optimal_threshold(test_l, probs)
    preds = (probs >= thresh).astype(int)
    
    return {
        'threshold': thresh,
        'accuracy': accuracy_score(test_l, preds),
        'precision': precision_score(test_l, preds, zero_division=0),
        'recall': recall_score(test_l, preds, zero_division=0),
        'f1': f1_score(test_l, preds, zero_division=0),
        'auc': roc_auc_score(test_l, probs),
        'avg_precision': average_precision_score(test_l, probs),
        'train_time': train_time,
        'pred_time': pred_time,
        'throughput': len(test_l) / pred_time,
    }


def test_model(model_name: str, train, test):
    """Test a single embedding model."""
    print(f"\n{'='*70}")
    print(f"🧪 Testing: {model_name}")
    print(f"{'='*70}")
    
    print("\n📊 Encoding training data...")
    train_q_emb, train_dim = encode_texts(train['queries'], model_name)
    train_d_emb, _ = encode_texts(train['documents'], model_name)
    
    print("\n📊 Encoding test data...")
    test_q_emb, test_dim = encode_texts(test['queries'], model_name)
    test_d_emb, _ = encode_texts(test['documents'], model_name)
    
    print(f"\n✅ Dimensions: {train_dim}")
    print(f"   Train: {train_q_emb.shape}")
    print(f"   Test: {test_q_emb.shape}")
    
    train_l = np.array(train['labels'])
    test_l = np.array(test['labels'])
    
    results = run_markrel(train_q_emb, train_d_emb, train_l, 
                          test_q_emb, test_d_emb, test_l)
    
    print(f"\n📈 Results:")
    print(f"   F1: {results['f1']:.4f} | AUC: {results['auc']:.4f} | Speed: {results['throughput']:.0f}/s")
    
    return {
        'model': model_name,
        'dim': train_dim,
        **results
    }


def main():
    print("\n" + "="*70)
    print("🐟 markrel WikiQA Embedding Model Comparison")
    print("="*70)
    
    train, test = load_wikiqa_data()
    print(f"\n📊 Dataset: {test['n_samples']} test samples ({test['n_relevant']} relevant)")
    
    # Models to compare
    models = [
        'sentence-transformers/all-MiniLM-L6-v2',  # 384-dim baseline
        'sentence-transformers/all-roberta-large-v1',  # 1024-dim
        'BAAI/bge-m3',  # 1024-dim (BGE-M3)
    ]
    
    results = []
    for model_name in models:
        try:
            result = test_model(model_name, train, test)
            results.append(result)
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            continue
    
    # Generate comparison report
    print("\n" + "="*70)
    print("📊 COMPARISON RESULTS")
    print("="*70)
    
    print("\n| Model | Dim | F1 | AUC | Precision | Recall | Speed |")
    print("|-------|-----|----|-----|-----------|--------|-------|")
    for r in results:
        model_short = r['model'].split('/')[-1][:30]
        print(f"| {model_short} | {r['dim']} | {r['f1']:.3f} | {r['auc']:.3f} | "
              f"{r['precision']:.3f} | {r['recall']:.3f} | {r['throughput']:.0f}/s |")
    
    # Save results
    json_path = os.path.join(
        os.path.dirname(__file__), "wikiqa_embedding_comparison.json"
    )
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {json_path}")


if __name__ == "__main__":
    main()
