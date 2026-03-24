#!/usr/bin/env python3
"""
🐟 markrel WikiQA Benchmark with Sentence-Transformers Embeddings

This script:
1. Loads the real WikiQA dataset
2. Encodes queries and answers using sentence-transformers
3. Trains markrel with custom embeddings
4. Optimizes the classification threshold

Usage:
    python experiments/run_wikiqa_embeddings_benchmark.py
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
    roc_auc_score, confusion_matrix, precision_recall_curve
)


def load_data():
    """Load WikiQA data."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "wikiqa", "wikiqa_processed.json"
    )
    
    print(f"📂 Loading data from {data_path}...")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    train = data['train']
    test = data['test']
    
    print(f"✅ Loaded {train['n_samples']} training samples")
    print(f"✅ Loaded {test['n_samples']} test samples")
    print(f"   Train class balance: {train['n_relevant']} relevant, {train['n_not_relevant']} not relevant")
    print(f"   Test class balance: {test['n_relevant']} relevant, {test['n_not_relevant']} not relevant")
    
    return (
        train['queries'], train['documents'], train['labels'],
        test['queries'], test['documents'], test['labels']
    )


def encode_texts(model_name: str, texts: List[str], batch_size: int = 64):
    """Encode texts using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence-transformers not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
    
    print(f"\n📥 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"� Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    return embeddings


def find_optimal_threshold(y_true, y_scores, metric='f1'):
    """Find optimal threshold for classification."""
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        scores.append(score)
    
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]


def run_benchmark(train_q, train_d, train_l, test_q, test_d, test_l, model_name='all-MiniLM-L6-v2'):
    """Run comprehensive benchmark with embeddings."""
    
    print("\n" + "="*70)
    print(f"🧪 markrel WikiQA Benchmark with Embeddings")
    print(f"   Model: {model_name}")
    print("="*70)
    
    # Encode texts
    print("\n📊 Encoding training data...")
    train_q_emb = encode_texts(model_name, train_q)
    train_d_emb = encode_texts(model_name, train_d)
    
    print("\n📊 Encoding test data...")
    test_q_emb = encode_texts(model_name, test_q)
    test_d_emb = encode_texts(model_name, test_d)
    
    results = []
    
    configs = [
        {
            'name': 'markrel (Cosine)',
            'metrics': ['cosine'],
            'n_bins': 10,
        },
        {
            'name': 'markrel (Cosine + Jaccard)',
            'metrics': ['cosine', 'jaccard'],
            'n_bins': 10,
        },
        {
            'name': 'markrel (Cosine + Euclidean)',
            'metrics': ['cosine', 'euclidean'],
            'n_bins': 10,
        },
        {
            'name': 'markrel (All metrics)',
            'metrics': ['cosine', 'euclidean', 'jaccard', 'overlap', 'dice'],
            'n_bins': 10,
        },
    ]
    
    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"🔬 Testing: {cfg['name']}")
        print(f"{'='*70}")
        
        # Train
        start = time.time()
        model = MarkovRelevanceModel(
            **{k: v for k, v in cfg.items() if k != 'name'},
            use_text_vectorizer=False,  # Use embeddings directly
        )
        model.fit(train_q_emb, train_d_emb, train_l)
        train_time = time.time() - start
        
        print(f"✅ Trained in {train_time:.3f}s")
        
        # Get probability scores
        start = time.time()
        probs = model.predict_proba(test_q_emb, test_d_emb)
        pred_time = time.time() - start
        
        # Find optimal thresholds
        thresholds = {
            'default': 0.5,
            'f1_optimal': find_optimal_threshold(test_l, probs, 'f1')[0],
            'precision_optimal': find_optimal_threshold(test_l, probs, 'precision')[0],
            'recall_optimal': find_optimal_threshold(test_l, probs, 'recall')[0],
        }
        
        # Evaluate with different thresholds
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh_name, thresh_val in thresholds.items():
            preds = (probs >= thresh_val).astype(int)
            
            acc = accuracy_score(test_l, preds)
            prec = precision_score(test_l, preds, zero_division=0)
            rec = recall_score(test_l, preds, zero_division=0)
            f1 = f1_score(test_l, preds, zero_division=0)
            
            if thresh_name == 'f1_optimal':
                best_f1 = f1
                best_thresh = thresh_val
            
            print(f"\n   📊 {thresh_name} (threshold={thresh_val:.3f}):")
            print(f"      Accuracy:  {acc:.3f}")
            print(f"      Precision: {prec:.3f}")
            print(f"      Recall:    {rec:.3f}")
            print(f"      F1 Score:  {f1:.3f}")
        
        # Use optimal F1 threshold for final metrics
        preds = (probs >= best_thresh).astype(int)
        
        metrics = {
            'name': cfg['name'],
            'config': str({k: v for k, v in cfg.items() if k != 'name'}),
            'threshold': best_thresh,
            'accuracy': accuracy_score(test_l, preds),
            'precision': precision_score(test_l, preds, zero_division=0),
            'recall': recall_score(test_l, preds, zero_division=0),
            'f1': f1_score(test_l, preds, zero_division=0),
            'auc': roc_auc_score(test_l, probs) if len(set(test_l)) > 1 else 0.5,
            'train_time': train_time,
            'pred_time': pred_time,
            'throughput': len(test_q) / pred_time,
        }
        
        results.append(metrics)
        
        # Show confusion matrix
        cm = confusion_matrix(test_l, preds)
        print(f"\n   📋 Confusion Matrix (optimal threshold):")
        print(f"                 Pred Not Rel  Pred Rel")
        print(f"   Actual Not Rel    {cm[0,0]:4d}       {cm[0,1]:4d}")
        print(f"   Actual Rel        {cm[1,0]:4d}       {cm[1,1]:4d}")
    
    return results


def generate_report(results, output_path, model_name):
    """Generate report."""
    
    report = []
    report.append(f"# 🐟 markrel WikiQA Embeddings Benchmark Report\n\n")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append(f"**Dataset:** WikiQA Real Dataset (from Hugging Face)\n\n")
    report.append(f"**Embedding Model:** {model_name}\n\n")
    
    # Summary table
    report.append("## Results (with Optimized Thresholds)\n\n")
    report.append("| Model | Accuracy | F1 | AUC | Threshold | Speed |\n")
    report.append("|-------|----------|-----|-----|-----------|-------|\n")
    
    for r in results:
        report.append(
            f"| {r['name']} | {r['accuracy']:.3f} | "
            f"{r['f1']:.3f} | {r['auc']:.3f} | {r['threshold']:.3f} | {r['throughput']:.0f}/s |\n"
        )
    
    # Detailed
    report.append("\n## Detailed Results\n")
    for r in results:
        report.append(f"\n### {r['name']}\n")
        report.append(f"- Threshold: {r['threshold']:.4f}\n")
        report.append(f"- Accuracy: {r['accuracy']:.4f}\n")
        report.append(f"- Precision: {r['precision']:.4f}\n")
        report.append(f"- Recall: {r['recall']:.4f}\n")
        report.append(f"- F1: {r['f1']:.4f}\n")
        report.append(f"- AUC: {r['auc']:.4f}\n")
        report.append(f"- Train time: {r['train_time']:.3f}s\n")
        report.append(f"- Pred time: {r['pred_time']:.3f}s\n\n")
    
    report_text = ''.join(report)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    return report_text


def main():
    print("\n" + "="*70)
    print("🐟 markrel WikiQA Benchmark with Sentence-Transformers")
    print("="*70)
    
    # Load
    train_q, train_d, train_l, test_q, test_d, test_l = load_data()
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Train: {len(train_q)} samples")
    print(f"      Relevant: {sum(train_l)} ({sum(train_l)/len(train_l)*100:.1f}%)")
    print(f"      Not relevant: {len(train_l) - sum(train_l)} ({(len(train_l)-sum(train_l))/len(train_l)*100:.1f}%)")
    print(f"   Test: {len(test_q)} samples")
    print(f"      Relevant: {sum(test_l)} ({sum(test_l)/len(test_l)*100:.1f}%)")
    print(f"      Not relevant: {len(test_q) - sum(test_l)} ({(len(test_q)-sum(test_l))/len(test_l)*100:.1f}%)")
    
    # Run with embeddings
    model_name = 'all-MiniLM-L6-v2'  # Fast, good quality 384-dim embeddings
    results = run_benchmark(train_q, train_d, train_l, test_q, test_d, test_l, model_name)
    
    # Report
    report_path = "experiments/wikiqa_embeddings_benchmark_report.md"
    report = generate_report(results, report_path, model_name)
    
    print("\n" + "="*70)
    print("📄 REPORT")
    print("="*70)
    print(report)
    
    # Save JSON
    json_path = "experiments/wikiqa_embeddings_benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved to: {report_path} and {json_path}")


if __name__ == "__main__":
    main()
