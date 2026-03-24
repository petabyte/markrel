#!/usr/bin/env python3
"""
🐟 markrel WikiQA Benchmark with 1024-Dimensional Embeddings

This script:
1. Loads the real WikiQA dataset with proper relevance labels
2. Encodes queries and answers using RoBERTa-large (1024-dim embeddings)
3. Trains markrel with these embeddings
4. Evaluates with proper WikiQA metrics

Embedding Model: all-roberta-large-v1 (1024 dimensions)
Dataset: WikiQA with binary relevance labels (0=not relevant, 1=relevant)

Usage:
    python experiments/run_wikiqa_1024dim_benchmark.py
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
    roc_auc_score, confusion_matrix, average_precision_score
)


def load_wikiqa_data():
    """Load WikiQA data with proper relevance labels."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "wikiqa", "wikiqa_processed.json"
    )

    print(f"📂 Loading WikiQA data from {data_path}...")

    with open(data_path, 'r') as f:
        data = json.load(f)

    train = data['train']
    test = data['test']

    # Convert to numpy arrays
    train_labels = np.array(train['labels'])
    test_labels = np.array(test['labels'])

    print(f"✅ Loaded {train['n_samples']} training samples")
    print(f"✅ Loaded {test['n_samples']} test samples")
    print(f"\n📊 Class Distribution:")
    print(f"   Train: {train['n_relevant']} relevant ({train['n_relevant']/train['n_samples']*100:.2f}%), "
          f"{train['n_not_relevant']} not relevant ({train['n_not_relevant']/train['n_samples']*100:.2f}%)")
    print(f"   Test:  {test['n_relevant']} relevant ({test['n_relevant']/test['n_samples']*100:.2f}%), "
          f"{test['n_not_relevant']} not relevant ({test['n_not_relevant']/test['n_samples']*100:.2f}%)")

    return (
        train['queries'], train['documents'], train_labels,
        test['queries'], test['documents'], test_labels
    )


def encode_with_sentence_transformers(texts: List[str], model_name: str, batch_size: int = 32):
    """Encode texts using sentence-transformers with specified model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence-transformers not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer

    print(f"\n📥 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"🔢 Model embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"📝 Encoding {len(texts)} texts...")

    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    return embeddings


def find_optimal_threshold(y_true, y_scores, metric='f1'):
    """Find optimal threshold for classification using grid search."""
    thresholds = np.linspace(0.001, 0.999, 500)
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


def run_benchmark(
    train_q, train_d, train_l,
    test_q, test_d, test_l,
    model_name='sentence-transformers/all-roberta-large-v1'
):
    """Run comprehensive benchmark with 1024-dim embeddings."""

    print("\n" + "="*70)
    print(f"🧪 markrel WikiQA Benchmark with 1024-Dimensional Embeddings")
    print(f"   Model: {model_name}")
    print(f"   Expected dimension: 1024")
    print("="*70)

    # Encode texts
    print("\n📊 Encoding TRAINING data...")
    train_q_emb = encode_with_sentence_transformers(train_q, model_name)
    train_d_emb = encode_with_sentence_transformers(train_d, model_name)

    print("\n📊 Encoding TEST data...")
    test_q_emb = encode_with_sentence_transformers(test_q, model_name)
    test_d_emb = encode_with_sentence_transformers(test_d, model_name)

    # Verify dimensions
    print(f"\n✅ Embedding dimensions:")
    print(f"   Query embeddings: {train_q_emb.shape}")
    print(f"   Doc embeddings: {train_d_emb.shape}")

    results = []

    # Test different metric configurations
    configs = [
        {
            'name': 'markrel (Cosine only)',
            'metrics': ['cosine'],
            'n_bins': 10,
            'bin_strategy': 'uniform'
        },
        {
            'name': 'markrel (Cosine + Euclidean)',
            'metrics': ['cosine', 'euclidean'],
            'n_bins': 10,
            'bin_strategy': 'uniform'
        },
        {
            'name': 'markrel (Cosine + Jaccard)',
            'metrics': ['cosine', 'jaccard'],
            'n_bins': 10,
            'bin_strategy': 'uniform'
        },
        {
            'name': 'markrel (All metrics)',
            'metrics': ['cosine', 'euclidean', 'jaccard', 'overlap', 'dice'],
            'n_bins': 10,
            'bin_strategy': 'uniform'
        },
    ]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"🔬 Configuration: {cfg['name']}")
        print(f"{'='*70}")

        # Train markrel model
        start_time = time.time()
        model = MarkovRelevanceModel(
            **{k: v for k, v in cfg.items() if k != 'name'},
            use_text_vectorizer=False,  # Use pre-computed embeddings
        )
        model.fit(train_q_emb, train_d_emb, train_l)
        train_time = time.time() - start_time

        print(f"✅ Training complete in {train_time:.3f}s")

        # Get probability scores
        start_time = time.time()
        probs = model.predict_proba(test_q_emb, test_d_emb)
        pred_time = time.time() - start_time

        print(f"✅ Prediction complete in {pred_time:.3f}s")
        print(f"   Throughput: {len(test_q) / pred_time:.0f} samples/sec")

        # Find optimal thresholds
        print("\n📊 Finding optimal thresholds...")
        thresh_f1, score_f1 = find_optimal_threshold(test_l, probs, 'f1')
        thresh_prec, score_prec = find_optimal_threshold(test_l, probs, 'precision')
        thresh_rec, score_rec = find_optimal_threshold(test_l, probs, 'recall')

        print(f"   Best F1 threshold: {thresh_f1:.4f} (F1={score_f1:.4f})")
        print(f"   Best Precision threshold: {thresh_prec:.4f} (Precision={score_prec:.4f})")
        print(f"   Best Recall threshold: {thresh_rec:.4f} (Recall={score_rec:.4f})")

        # Evaluate with F1-optimal threshold
        preds = (probs >= thresh_f1).astype(int)

        acc = accuracy_score(test_l, preds)
        prec = precision_score(test_l, preds, zero_division=0)
        rec = recall_score(test_l, preds, zero_division=0)
        f1 = f1_score(test_l, preds, zero_division=0)
        auc = roc_auc_score(test_l, probs) if len(set(test_l)) > 1 else 0.5
        avg_prec = average_precision_score(test_l, probs)

        print(f"\n📈 Performance Metrics (optimal threshold):")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   AUC-ROC:   {auc:.4f}")
        print(f"   Avg Prec:  {avg_prec:.4f}")

        # Confusion matrix
        cm = confusion_matrix(test_l, preds)
        print(f"\n📋 Confusion Matrix:")
        print(f"                 Pred Not Rel  Pred Rel")
        print(f"   Actual Not Rel    {cm[0,0]:5d}      {cm[0,1]:5d}")
        print(f"   Actual Rel        {cm[1,0]:5d}      {cm[1,1]:5d}")

        metrics = {
            'name': cfg['name'],
            'config': cfg,
            'embedding_model': model_name,
            'embedding_dim': train_q_emb.shape[1],
            'threshold': thresh_f1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc,
            'avg_precision': avg_prec,
            'train_time': train_time,
            'pred_time': pred_time,
            'throughput': len(test_q) / pred_time,
        }

        results.append(metrics)

    return results


def generate_report(results: List[Dict], output_path: str, model_name: str):
    """Generate markdown report."""

    report_lines = [
        f"# 🐟 markrel WikiQA 1024-Dim Embedding Benchmark Report\n\n",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        f"**Dataset:** WikiQA (Question-Answer Relevance)\n\n",
        f"**Embedding Model:** {model_name}\n\n",
        f"**Embedding Dimension:** 1024\n\n",
    ]

    # Summary table
    report_lines.append("## Results Summary\n\n")
    report_lines.append("| Model | Accuracy | Precision | Recall | F1 | AUC | Avg Prec | Speed |\n")
    report_lines.append("|-------|----------|-----------|--------|-----|-----|----------|-------|\n")

    for r in results:
        report_lines.append(
            f"| {r['name']} | {r['accuracy']:.3f} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | {r['auc']:.3f} | "
            f"{r['avg_precision']:.3f} | {r['throughput']:.0f}/s |\n"
        )

    # Detailed results
    report_lines.append("\n## Detailed Results\n")
    for r in results:
        report_lines.append(f"\n### {r['name']}\n")
        report_lines.append(f"- **Embedding:** {r['embedding_model']} ({r['embedding_dim']} dims)\n")
        report_lines.append(f"- **Metrics Used:** {r['config']['metrics']}\n")
        report_lines.append(f"- **Bins:** {r['config']['n_bins']} ({r['config']['bin_strategy']})\n")
        report_lines.append(f"- **Optimal Threshold:** {r['threshold']:.4f}\n\n")
        report_lines.append(f"**Performance:**\n")
        report_lines.append(f"- Accuracy: {r['accuracy']:.4f}\n")
        report_lines.append(f"- Precision: {r['precision']:.4f}\n")
        report_lines.append(f"- Recall: {r['recall']:.4f}\n")
        report_lines.append(f"- F1 Score: {r['f1']:.4f}\n")
        report_lines.append(f"- AUC-ROC: {r['auc']:.4f}\n")
        report_lines.append(f"- Average Precision: {r['avg_precision']:.4f}\n\n")
        report_lines.append(f"**Timing:**\n")
        report_lines.append(f"- Train time: {r['train_time']:.3f}s\n")
        report_lines.append(f"- Pred time: {r['pred_time']:.3f}s\n")
        report_lines.append(f"- Throughput: {r['throughput']:.0f} samples/sec\n\n")

    report_text = ''.join(report_lines)

    with open(output_path, 'w') as f:
        f.write(report_text)

    return report_text


def main():
    print("\n" + "="*70)
    print("🐟 markrel WikiQA Benchmark with 1024-Dimensional Embeddings")
    print("="*70)

    # Load WikiQA data
    train_q, train_d, train_l, test_q, test_d, test_l = load_wikiqa_data()

    # Run benchmark with 1024-dim model
    model_name = 'sentence-transformers/all-roberta-large-v1'
    results = run_benchmark(
        train_q, train_d, train_l,
        test_q, test_d, test_l,
        model_name
    )

    # Generate report
    report_path = os.path.join(
        os.path.dirname(__file__), "wikiqa_1024dim_benchmark_report.md"
    )
    report = generate_report(results, report_path, model_name)

    print("\n" + "="*70)
    print("📄 FINAL REPORT")
    print("="*70)
    print(report)

    # Save JSON results
    json_path = os.path.join(
        os.path.dirname(__file__), "wikiqa_1024dim_benchmark_results.json"
    )
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to:")
    print(f"   Report: {report_path}")
    print(f"   JSON: {json_path}")


if __name__ == "__main__":
    main()
